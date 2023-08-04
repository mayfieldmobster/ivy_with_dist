import os
import json


def _get_workers_from_hosts(hosts, tf_ports):
    workers = []
    hosts = hosts.split(",")
    tf_ports = tf_ports.split(",")
    if len(tf_ports) == 1:
        tf_ports = [tf_ports[0]] * len(hosts)
    elif len(tf_ports) != len(hosts):
        raise Exception("Number of Ports must equal Number of Hosts")
    for h, p in zip(hosts, tf_ports):
        workers.append(f"{h.split(':')[0]}:{p}")
    return workers


def _get_workers_from_hostfile(hostfile, tf_ports):
    addresses = []
    workers = []
    with open(hostfile, "r") as file:
        hosts = file.read().split("\n")
        for host in hosts:
            if host[0] == "#":
                hosts.remove(host)
                continue
            host = host.split("#")[0]
            host = host.split(" ")[0]
            if host:
                addresses.append(host)
    tf_ports = tf_ports.split(",")
    if len(tf_ports) == 1:
        tf_ports = [tf_ports[0]] * len(addresses)
    for h, p in zip(hosts, tf_ports):
        workers.append(f"{h}:{p}")
    return workers


def launch(
    *,
    hosts: str,
    hostfile: str,
    num_nodes,
    tf_ports,
    user_script,
    user_args,
    rank,
    **kwargs,
):
    base = f"python3 {user_script} {' '.join([a for a in user_args])}"
    if num_nodes == 1:
        return base
    try:
        if os.environ["TF_CONFIG"]:
            return base
    except KeyError:
        pass

    if hostfile is not None:
        workers = _get_workers_from_hostfile(hostfile, tf_ports)
    elif hosts is not None:
        workers = _get_workers_from_hosts(hosts, tf_ports)

    tf_config = {
        "cluster": {"worker": workers},
        "task": {"index": rank, "type": "worker"},
    }

    cmd = f"TF_CONFIG='{json.dumps(tf_config)}' {base}"
    return cmd


def mpi():
    return False
