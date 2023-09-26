# Ivy Distributed

Every Framework has its own implementation of distributed computing. This makes it hard to have a framework agnostic distributed framework.

### Pytorch

Pytorch for example brings the idea of each GPU/TPU getting its own process, similar to MPI. Each process communicates to each other through p2p and collective communication operations like send or allgather

### Jax

In pure Jax each Machine has its own process where it controls multiple GPUs/TPUs. Jax has limited built in methods for multi-GPU/TPU computing such as pmap, xmap, and pjit, Jax also  supports tensor sharding. The problem with this is that it is still limited, and completely different to torch making it hard to integrate agnostic property's. A solution to this problem is mpi4jax, mpi4jax gives jax access to mpi operations, thus we can run jax with each GPU/TPU assigned its own process like torch. mpi4jax also supports jax's built in grads functions to work with certain mpi operations.

### Numpy

Numpy doesn't support any distributed computing out of the box, but we can easily integrate some sort of distributed computing strategy with mpi4py, which once gives numpy access to mpi operations.

### Tensorflow

Tensorflow is a weird one, in order to use tensorflow distributed computing you have to use strategies. Strategies replicate your model's parameters and distribute the training across multiple GPUs/TPUs. Although strategies are a very easy way to preform certain high level operations if we want to emulate something more complicated such as a the ZeRO optimizer its near impossible. Theres no easy way to integrate mpi with tensorflow without moving tensors off the GPUs/TPUs, if we exploit the dlpack tensor structure using this we can move tensorflow tensor into tensors that can be used with mpi4py without moving the tensors into CPU memory. The only problem with this method is that tensorflows built in grad function doesn't support the mpi operations so doing something like model parallelism things becomes more complicated, so we recommend using jax or torch backend for training.

## How does it work

```python
#main.py
import ivy
import ivy.distributed as i_dist

ivy.set_backend("torch")
i_dist.init_dist(backend="nccl")

def parallel_gelu(x, pc: i_dist.ParallelContext):
    x = ivy.split(x, num_or_size_splits=pc.world_size)
    out = ivy.gelu(x[pc.rank])
    out = i_dist.all_gather(out)
    return out

pc = i_dist.ParallelContext()

x = ivy.random.random_normal(shape=(pc.world_size,10))

out1 = parallel_gelu(x, pc)

if pc.rank == 0:
    print(out.shape)

# pmap does the same thing
out2 = i_dist.pmap(ivy.gelu)(x)

assert ivy.all(out1 == out2)
```

```bash
#2 GPUS/TPUS/CPU cores
ivyrun -B torch --nproc_per_node 2 main.py

#2 Nodes, 16 GPUS (only needs to be called on 192.168.0.100)
ivyrun -B torch -H 192.168.0.100,192.169.0.101 --nproc_per_node 8 main.py
```

As shown above you write your code with i_dist functions and  and use a launcher to run it. The launcher deals with creating a process for each compute accelerator. You can also use other launchers like torchrun and mpirun when using the appropriate framework for that launcher.

The `ParallelContext` context class is a singleton dict like class, that keeps track of useful information about the current job like world_size, global_rank, etc.

Keep in mind the number --nproc_per_node must be less or equal to than the number of CPU cores. When running with multiple hosts the launcher only needs to be called on a single node, but that node needs passwordless ssh access to each other node in the cluster.

If you want to change the backend framework you can do so as follows:

```python
#main.py
...
ivy.set_backend("numpy")
...
```
```bash
ivyrun -B numpy -H 192.168.0.100,192.169.0.101 --nproc_per_node 8 main.py
```

even though numpy uses mpirun rather than torchrun there is no need to change any args except the given backend.

Its important to know that as of now when using `ivy.distributed` changing backends in the middle of your code is not supported due to different frameworks using different launchers.

ivy.distributed also supports custom process groups:

```python
import ivy
import ivy.distributed as i_dist

ivy.set_backend("torch")
i_dist.init_dist(backend="nccl")

group = i_dist.Group([2,3,5,7])

x = ivy.random.random_normal(shape=(group.size,10))

#global rank 2
if group.rank == 0:
    i_dist.send(x, group=group, dst=2)

#global rank 5
if group.rank == 2:
    # you could pass x directly to the x_buffer and the data would not be over written
    out = ivy.empty_like(x)
    out = i_dist.recv(x_buffer=out, group=group, src=0)

```
```bash
ivyrun -B torch --nproc_per_node 2 main.py
```


## Frontend

frontends have not been fully implemetned yet but the torch frontend is under development with the hope of eventually being able to transpile libraries like colossalai and deepspeed to work with other frameworks

## Ring & Tree Based Collective Communication

More efficient collective communication algorithms can be implemented using p2p operations, the `ivy.distributed.stateful` is where these operation can be located

## Nested Functions

Currently containers cannot be passed in collective communication functions
