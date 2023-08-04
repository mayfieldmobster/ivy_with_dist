import click
import os
import sys

from ivy.distributed.cli.run_multi_host import MultiHostRun
from ivy.distributed.cli.host_info import HostInfo
import ivy


@click.command()
@click.option(
    "-B",
    "-backend",
    "--backend",
    type=str,
    default=None,
    help="backend used to determine which launcher is appropriate",
)
@click.option(
    "-H",
    "-host",
    "--host",
    type=str,
    default=None,
    help="sequence of host names <host1>,<host2>",
)
@click.option(
    "--hostfile",
    type=str,
    default=None,
    help=(
        "Hostfile path that defines the device pool available to the job, each line in"
        " the file is a hostname"
    ),
)
@click.option(
    "--nproc_per_node", type=int, default=1, help="Number of GPU/TPUs on each device"
)
@click.option("--num_nodes", type=int, default=1, help="Number of workers")
@click.option(
    "--master_port",
    type=int,
    default=29500,
    help=(
        "(optional) Port used by PyTorch distributed for communication during"
        " distributed training."
    ),
)
@click.option(
    "--master_address",
    type=str,
    default="127.0.0.1",
    help=(
        "(optional) IP address of node 0, will be inferred via 'hostname -I' if not"
        " specified."
    ),
)
@click.option(
    "--tf_ports",
    type=str,
    default=12345,
    help=(
        "default ports used by each node to construct in format 123 (for the same port)"
        " or 123,456,111"
    ),
)
@click.argument("user_script", type=str)
@click.argument("user_args", nargs=-1)
def run(
    backend: str,
    host: str,
    hostfile: str,
    nproc_per_node,
    num_nodes,
    master_port,
    master_address,
    tf_ports,
    user_script,
    user_args,
):
    hosts = host  # hosts makes more sence
    os.system(f"export NPROC_PER_NODE={nproc_per_node}")
    if backend is None:
        raise Exception("Backend Must Be Given to ivyrun")
    ivy.set_backend(backend)
    host_info = HostInfo()
    # mpi can be launched from the master node
    mpi = ivy.current_dist_backend().cli.mpi()
    if not mpi:
        if hostfile:
            host_info.load_from_hostfile(hostfile_path=hostfile)
        elif hosts:
            host_info.load_from_host_str(host_str=hosts)
        else:
            host_info.load_from_host_str("localhost")

        runner = MultiHostRun()
        current_path = os.path.abspath(".")

        runner.connect(host_info=host_info, work_dir=current_path)
        for rank, host in enumerate(host_info):
            cmd = ivy.current_dist_backend().cli.launch(
                hosts=hosts,
                hostfile=hostfile,
                nproc_per_node=nproc_per_node,
                num_nodes=num_nodes,
                master_port=master_port,
                master_address=master_address,
                rank=rank,
                tf_ports=tf_ports,
                user_script=user_script,
                user_args=user_args,
            )
            runner.send(host, cmd)

        msg_from_node = runner.recv_from_all()

        msg_from_node = runner.recv_from_all()
        has_error = False

        click.echo("\n====== Training on All Nodes =====")
        for hostname, msg in msg_from_node.items():
            click.echo(f"{hostname}: {msg}")

            # check if a process failed
            if msg == "failure":
                has_error = True

        runner.stop_all()

        # receive the stop status
        msg_from_node = runner.recv_from_all()

        # print node status
        click.echo("\n====== Stopping All Nodes =====")
        for hostname, msg in msg_from_node.items():
            click.echo(f"{hostname}: {msg}")

        # give the process an exit code
        # so that it behaves like a normal process
        if has_error:
            sys.exit(1)
        else:
            sys.exit(0)
    else:
        # allows to get local rank info
        rank = None
        cmd = ivy.current_dist_backend().cli.launch(
            hosts=hosts,
            hostfile=hostfile,
            nproc_per_node=nproc_per_node,
            num_nodes=num_nodes,
            master_port=master_port,
            master_address=master_address,
            rank=rank,
            user_script=user_script,
            user_args=user_args,
        )


if __name__ == "__main__":
    run()
