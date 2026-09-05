# latte loader containers attached to a podman cluster's rack networks. Each
# container runs the vendored latte_cs_alike.rn workload (from scylla-cluster-tests)
# in a loop against its rack-group's targets, per the lb_policy/workload_mode knobs.

import logging
import os
import shlex
from collections import OrderedDict

from ccmlib.container_client import ContainerClientError

LOGGER = logging.getLogger("ccm")

DEFAULT_LOADER_IMAGE = "scylladb/latte:latest"

RUNE_SCRIPT_HOST_PATH = os.path.join(os.path.dirname(__file__), "resources", "latte_cs_alike.rn")
RUNE_SCRIPT_CONTAINER_PATH = "/workloads/latte_cs_alike.rn"

# Load-balancing policy knob: which nodes a loader's driver connects to.
LB_POLICIES = ("local-rack", "local-dc", "whole-cluster")
DEFAULT_LB_POLICY = "local-rack"

DEFAULT_SCHEMA_CONFIG = {"keyspaces": 1, "tables_per_keyspace": 1, "replication_factor": 3}

# Workload knob: which keyspace/table a loader (or rather, its whole rack-group
# -- see LoaderSet.add) targets, and whether it only writes or also reads.
WORKLOAD_MODES = {"write": ["-f", "write"], "read-write": ["-f", "write:1", "-f", "read:1"]}
DEFAULT_WORKLOAD_MODE = "read-write"


class Loader:
    """A single latte loader container, running latte_cs_alike.rn on one rack's network."""

    def __init__(self, cluster, name, dc, rack, ip, network, image, lb_policy=DEFAULT_LB_POLICY,
                 workload_mode=DEFAULT_WORKLOAD_MODE, keyspace_index=0, table_index=0,
                 schema_config=None):
        if lb_policy not in LB_POLICIES:
            raise ValueError(f"Invalid lb_policy {lb_policy!r}: must be one of {LB_POLICIES}")
        if workload_mode not in WORKLOAD_MODES:
            raise ValueError(f"Invalid workload_mode {workload_mode!r}: must be one of {list(WORKLOAD_MODES)}")
        self.cluster = cluster
        self.name = name
        self.dc = dc
        self.rack = rack
        self.ip = ip
        self.network = network
        self.image = image
        self.lb_policy = lb_policy
        self.workload_mode = workload_mode
        self.keyspace_index = keyspace_index
        self.table_index = table_index
        self.schema_config = dict(schema_config or DEFAULT_SCHEMA_CONFIG)

    def container_name(self):
        return f"ccm-{self.cluster.name}-{self.name}"

    def keyspace(self):
        return f"ks{self.keyspace_index}"

    def table(self):
        return f"tbl{self.table_index}"

    def _target_ips(self):
        """Nodes this loader's driver connects to, per lb_policy."""
        topo = self.cluster.network_topology
        if self.lb_policy == "local-rack":
            ips = [i["ip"] for i in topo.node_assignments.values() if i["dc"] == self.dc and i["rack"] == self.rack]
        elif self.lb_policy == "local-dc":
            ips = [i["ip"] for i in topo.node_assignments.values() if i["dc"] == self.dc]
        else:  # whole-cluster
            ips = [i["ip"] for i in topo.node_assignments.values()]
        if not ips:
            raise RuntimeError(f"No target nodes found for loader {self.name} (lb_policy={self.lb_policy})")
        return ips

    def _ks_table_params(self):
        return ["-P", f"keyspace={self.keyspace()}", "-P", f"table={self.table()}"]

    def _schema_cmd(self):
        return [
            "latte", "schema", RUNE_SCRIPT_CONTAINER_PATH, ",".join(self._target_ips()),
            *self._ks_table_params(),
            "-P", f"replication_factor={self.schema_config['replication_factor']}",
        ]

    def _run_cmd(self):
        return [
            "latte", "run", RUNE_SCRIPT_CONTAINER_PATH, ",".join(self._target_ips()),
            "--warmup", "0", *WORKLOAD_MODES[self.workload_mode],
            *self._ks_table_params(),
        ]

    def start(self):
        from ccmlib.scylla_podman_cluster import (
            _get_podman_client,
            _remove_named_container_if_safe,
            _resource_labels,
        )
        name = self.container_name()
        existing = _remove_named_container_if_safe(name, allow_reuse_current_running=True)
        if existing is not None:
            LOGGER.debug("Reusing existing loader container %s", name)
            return
        client = _get_podman_client()
        # Create schema once, then loop the workload forever -- the script's
        # CREATE ... IF NOT EXISTS makes repeated schema runs across loaders safe.
        schema_and_loop = "{schema} && while true; do {run}; sleep 1; done".format(
            schema=shlex.join(self._schema_cmd()), run=shlex.join(self._run_cmd()),
        )
        try:
            client.run_container(
                image=self.image,
                name=name,
                network=self.network,
                ip=self.ip,
                labels=_resource_labels(),
                cap_add=["NET_ADMIN"],
                volumes={RUNE_SCRIPT_HOST_PATH: RUNE_SCRIPT_CONTAINER_PATH},
                entrypoint="sh",
                command=["-lc", schema_and_loop],
            )
        except ContainerClientError as exc:
            raise RuntimeError(f"Failed to start loader container {name}: {exc}")

        # Give the loader the same cross-rack/cross-DC routes as a real node
        # in its rack, mirroring ScyllaPodmanCluster.start_client_container().
        node_name = self._rack_node_name()
        self.cluster._setup_container_routes(name, node_name)

    def _rack_node_name(self):
        topo = self.cluster.network_topology
        for node_name, info in topo.node_assignments.items():
            if info["dc"] == self.dc and info["rack"] == self.rack:
                return node_name
        raise RuntimeError(
            f"No node found in {self.dc}/{self.rack} to derive loader routing from"
        )

    def stop(self):
        from ccmlib.scylla_podman_cluster import _remove_named_container_if_safe
        _remove_named_container_if_safe(self.container_name(), allow_remove_current_running=True)

    def is_running(self):
        from ccmlib.scylla_podman_cluster import _inspect_container, _RUNNING_CONTAINER_STATES
        info = _inspect_container(self.container_name())
        if info is None:
            return False
        return info.get("State", {}).get("Status") in _RUNNING_CONTAINER_STATES

    def to_dict(self):
        return {
            "dc": self.dc, "rack": self.rack, "image": self.image, "ip": self.ip,
            "lb_policy": self.lb_policy, "workload_mode": self.workload_mode,
            "keyspace_index": self.keyspace_index, "table_index": self.table_index,
        }


class LoaderSet:
    """Manages the collection of loader containers for a podman cluster."""

    def __init__(self, cluster):
        self.cluster = cluster
        self.loaders = OrderedDict()  # name -> Loader
        # Schema knobs are cluster-wide (shared by every loader), not per-loader.
        self.schema_config = dict(DEFAULT_SCHEMA_CONFIG)

    def configure_schema(self, keyspaces=None, tables_per_keyspace=None, replication_factor=None):
        for name, value in (("keyspaces", keyspaces), ("tables_per_keyspace", tables_per_keyspace),
                            ("replication_factor", replication_factor)):
            if value is None:
                continue
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
            self.schema_config[name] = value
        self.cluster._update_config()

    def add(self, dc, rack, count=1, image=DEFAULT_LOADER_IMAGE, lb_policy=DEFAULT_LB_POLICY,
            workload_mode=DEFAULT_WORKLOAD_MODE, keyspace_index=0, table_index=0):
        """Add `count` loaders to one rack. All loaders created by a single call share
        the same lb_policy/workload_mode/keyspace_index/table_index -- i.e. the workload
        is assigned per rack-group (this call), not per individual loader."""
        from ccmlib.scylla_podman_cluster import (
            LOADER_HOST_BASE,
            LOADER_HOST_MAX,
            _sanitize_podman_name,
        )
        if not (0 <= keyspace_index < self.schema_config["keyspaces"]):
            raise ValueError(
                f"keyspace_index {keyspace_index} out of range for "
                f"{self.schema_config['keyspaces']} configured keyspace(s)"
            )
        if not (0 <= table_index < self.schema_config["tables_per_keyspace"]):
            raise ValueError(
                f"table_index {table_index} out of range for "
                f"{self.schema_config['tables_per_keyspace']} configured table(s) per keyspace"
            )
        topo = self.cluster.network_topology
        if topo is None:
            raise RuntimeError("Cluster has no network topology; loaders require a podman cluster")
        key = (dc, rack)
        if key not in topo.rack_networks:
            raise RuntimeError(
                f"Unknown rack {dc}/{rack}: available racks are {list(topo.rack_networks.keys())}"
            )
        rack_info = topo.rack_networks[key]
        network = rack_info["network_name"]
        rack_idx = rack_info["rack_idx"]

        used_offsets = {
            int(loader.ip.rsplit(".", 1)[-1])
            for loader in self.loaders.values()
            if (loader.dc, loader.rack) == key and loader.ip
        }
        created = []
        offset = LOADER_HOST_BASE
        for _ in range(count):
            while offset in used_offsets:
                offset += 1
            if offset > LOADER_HOST_MAX:
                raise RuntimeError(
                    f"Loader band exhausted for {dc}/{rack}: max "
                    f"{LOADER_HOST_MAX - LOADER_HOST_BASE + 1} loaders per rack"
                )
            name = f"loader-{_sanitize_podman_name(dc)}-{_sanitize_podman_name(rack)}-{offset}"
            ip = f"{topo.subnet_prefix}.{rack_idx}.{offset}"
            loader = Loader(self.cluster, name, dc, rack, ip, network, image, lb_policy=lb_policy,
                            workload_mode=workload_mode, keyspace_index=keyspace_index,
                            table_index=table_index, schema_config=self.schema_config)
            loader.start()
            self.loaders[name] = loader
            used_offsets.add(offset)
            created.append(loader)
            offset += 1
        self.cluster._update_config()
        return created

    def remove(self, names=None):
        targets = list(self.loaders.values()) if names is None else [
            self.loaders[n] for n in names if n in self.loaders
        ]
        for loader in targets:
            try:
                loader.stop()
            except Exception:
                LOGGER.warning("Failed to stop loader %s", loader.name, exc_info=True)
            self.loaders.pop(loader.name, None)
        self.cluster._update_config()

    def status(self):
        return {name: loader.is_running() for name, loader in self.loaders.items()}

    def to_dict(self):
        return {
            "loaders": OrderedDict((name, loader.to_dict()) for name, loader in self.loaders.items()),
            "schema_config": self.schema_config,
        }

    @classmethod
    def from_dict(cls, cluster, data):
        """Reconstruct a LoaderSet from persisted config.

        Containers themselves are re-probed by name (see Loader.is_running/stop),
        not trusted from disk -- same reconnect philosophy as the monitoring stack.
        """
        loader_set = cls(cluster)
        loader_set.schema_config.update((data or {}).get("schema_config", {}))
        topo = cluster.network_topology
        for name, info in (data or {}).get("loaders", {}).items():
            dc, rack = info["dc"], info["rack"]
            image = info.get("image", DEFAULT_LOADER_IMAGE)
            lb_policy = info.get("lb_policy", DEFAULT_LB_POLICY)
            workload_mode = info.get("workload_mode", DEFAULT_WORKLOAD_MODE)
            keyspace_index = info.get("keyspace_index", 0)
            table_index = info.get("table_index", 0)
            key = (dc, rack)
            network = topo.rack_networks[key]["network_name"] if topo and key in topo.rack_networks else None
            ip = info.get("ip")
            loader_set.loaders[name] = Loader(cluster, name, dc, rack, ip, network, image, lb_policy=lb_policy,
                                               workload_mode=workload_mode, keyspace_index=keyspace_index,
                                               table_index=table_index, schema_config=loader_set.schema_config)
        return loader_set
