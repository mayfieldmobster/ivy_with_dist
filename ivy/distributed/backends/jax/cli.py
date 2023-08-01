import os


def launch(
    host: str,
    hostfile: str,
    nproc_per_node,
    master_port,
    master_address,
    rank,
    user_script,
    user_args,
):
    cmd_base = "MPI4JAX_USE_CUDA_MPI=1 mpirun "
    options = []
    if host is not None:
        num_processes = len(host.split(",")) * nproc_per_node
        options.append(f"-np {num_processes}")
        options.append(f"-H {host}")
    elif hostfile is not None:
        with open(hostfile, "r") as file:
            hosts = file.readlines()
            num_nodes = 0
            for host in hosts:
                if host.replace(" ", "").replace("\n", "") != "":
                    num_nodes += 1
            num_processes = num_nodes * nproc_per_node
        options.append(f"-np {num_processes}")
        options.append(f"--hostfile {hostfile}")
    else:
        options.append(f"-np {nproc_per_node}")

    options.append("-bind-to none -map-by slot")
    options.append("-mca pml ob1 -mca btl ^openib")
    cmd = cmd_base + " ".join(options) + f" python3 {user_script} {user_args}"
    os.system(cmd)


def mpi():
    return True
