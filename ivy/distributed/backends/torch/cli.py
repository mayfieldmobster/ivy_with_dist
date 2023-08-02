def launch(
    nproc_per_node,
    num_nodes,
    master_port,
    master_address,
    user_script,
    user_args,
    rank,
    **kwargs,
):
    base = "torchrun"
    user_args = " ".join([a for a in user_args])
    options = []
    options.append(f"--nproc_per_node {nproc_per_node}")
    options.append(f"--nnodes {num_nodes}")
    options.append(f"--node_rank {rank}")
    options.append(f"--master_addr={master_address} --master_port={master_port}")
    options.append(user_script)
    options.append(user_args)

    cmd = f"{base} {' '.join(options)}"
    return cmd


def mpi():
    return False
