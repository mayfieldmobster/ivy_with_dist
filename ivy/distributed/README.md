# Ivy Distributed

Every Framework has its own implementation of distributed computing. This makes it hard to have a framework agnostic distributed framework. 

### Pytorch

Pytorch for example brings the idea of each GPU/TPU getting its own process, similar to how MPI works. Each process communicates to each other through p2p operation and collective communication like allgather

### Jax

In pure Jax each Machine has its own process where it controls multiple GPUs/TPUs, Jax has limited built in methods for multi-GPU/TPU computing such as pmap, xmap, and pjit, Jax also has easy ways to support tensor sharding. The problem is that this is still limited and also hard to integrate with with torch. A solution to this is mpi4jax, mpi4jax gives jax access to mpi operations, thus we can run jax with each GPU/TPU assigned its own process. 

### Numpy

Numpy doesn't support any distributed computing by out of the box but we can easily integrate some sort of distributed computing with mpi4py, which once again gives numpy access to mpi operations.

### Tensorflow

Tensorflow is a weird one, in order to use tensorflow distributed you have to use strategies. strategies replicate your model's parameters and distribute the training across multiple GPUs/TPUs, although strategies are a very easy way to preform certain high level operations if we want to emulate something more complicated such as a the ZeRO optimizer its near impossible. Theres no easy way to integrate mpi with tensorflow but I found away without moving tensors off the GPUs/TPUs, if we exploit the dlpack tensor structure using this we can move tensorflow tensor into tensors that can be used with mpi4py without moving the tensors into CPU memory. The only problem with this is using grads with this becomes alot more complicated so we recommend using jax or torch backend for training

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
    
x = ivy.random.random_norma(shape=(pc.world_size,10))

out1 = parallel_gelu(x, pc)
if pc.rank == 0:
    print(out)

# pmap does the same thing
out2 = i_dist.pmap(ivy.gelu)(x)

assert ivy.all(out1 == out2)
```

```bash
#run on 192.168.0.100
ivyrun -B torch -H 192.168.0.100,192.169.0.101 --nproc_per_node 8 main.py
```

As shown above you write your code with i_dist functions and  and use a launcher to run it. The launcher deals with creating a process for each compute accelerator. You can also use other launchers like torchrun and mpirun when using the appropriate framework for that launcher.

## Frontend

frontends have not been implemetned yet but the torch frontend is under development with the hope of eventually being able to transpile libraries like colossalai and deepspeed to work with other frameworks