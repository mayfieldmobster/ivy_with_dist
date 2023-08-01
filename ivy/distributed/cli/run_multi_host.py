import fabric
import os
import click
from multiprocessing import Process, Pipe

from .host_info import HostInfo


def is_local_host(address):
    if address in ("localhost", "127.0.0.1", "0.0.0.0"):
        return True
    return False


def run_on_node(host, host_port, work_dir, send_pipe, recv_pipe):
    fab_conn = fabric.Connection(host, port=host_port)
    task = True

    env = dict()
    for k, v in os.environ.items():
        if v and "\n" not in v:
            env[k] = v

    env_msg = " ".join([f'{k}="{v}"' for k, v in env.items()])

    while task:
        msg = recv_pipe.recv()

        if msg == "exit":
            task = False
            break
        else:
            try:
                with fab_conn.cd(work_dir):
                    # propagate the runtime environment
                    with fab_conn.prefix(f"export {env_msg}"):
                        if is_local_host(host):
                            # execute on the local machine
                            fab_conn.local(msg, hide=False)
                        else:
                            # execute on the remote machine
                            fab_conn.run(msg, hide=False)
                    send_pipe.send("success")
            except Exception as e:
                click.echo(f"Error: failed to run {msg} on {host}, exception: {e}")


class MultiHostRun:
    """Inspired by Colossal AI."""

    def __init__(self):
        self.process = {}
        self.send_pipes = {}
        self.recv_pipes = {}

    def connect(self, host_info: HostInfo, work_dir):
        for host in host_info:
            master_send_pipe, worker_recv_pipe = Pipe()
            master_recv_pipe, worker_send_pipe = Pipe()
            p = Process(
                target=run_on_node,
                args=(
                    host,
                    host_info.ssh_port,
                    work_dir,
                    worker_send_pipe,
                    worker_recv_pipe,
                ),
            )
            p.start()
            self.process[host] = p
            self.recv_pipes[host] = master_recv_pipe
            self.send_pipes[host] = master_send_pipe

    def send(self, host, msg):
        conn = self.master_send_conns[host]
        conn.send(msg)

    def stop_all(self):
        for _, conn in self.master_send_conns.items():
            conn.send("exit")

    def recv_from_all(self):
        msg_from_node = dict()
        for hostname, conn in self.master_recv_conns.items():
            msg_from_node[hostname] = conn.recv()
        return msg_from_node
