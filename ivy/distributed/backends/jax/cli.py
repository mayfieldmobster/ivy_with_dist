import fabric


def launch(
    *,
    hosts: str,
    hostfile: str,
    nproc_per_node,
    num_nodes,
    user_script,
    user_args,
    **kwargs,  # collect unsed kwargd
):
    cmd_base = f"MPI4JAX_USE_CUDA_MPI=1 NPROC_PER_NODE={nproc_per_node} mpirun "
    options = []
    num_processes = num_nodes * nproc_per_node
    if hosts is not None:
        workers = ""
        for host in hosts.split(","):
            workers += f"{host}:{nproc_per_node}"
        options.append(f"-H {workers}")
    elif hostfile is not None:
        options.append(f"--hostfile {hostfile}")

    options.append(f"-np {num_processes}")

    if hosts or hostfile:
        options.append("-mca pml ob1 -mca btl ^openib")

    options.append("-bind-to none -map-by slot")

    user_args = " ".join([a for a in user_args])

    cmd = cmd_base + " ".join(options) + f" python3 {user_script} {user_args}"
    fab_conn = fabric.Connection("localhost")
    fab_conn.local(cmd, hide=False)


def mpi():
    return True
