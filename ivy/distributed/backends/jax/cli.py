import os


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
    cmd_base = "MPI4JAX_USE_CUDA_MPI=1 mpirun "
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

    options.append("-bind-to none -map-by slot")
    options.append("-mca pml ob1 -mca btl ^openib")

    user_args = " ".join([a for a in user_args])

    cmd = cmd_base + " ".join(options) + f" python3 {user_script} {user_args}"
    os.system(cmd)


def mpi():
    return True
