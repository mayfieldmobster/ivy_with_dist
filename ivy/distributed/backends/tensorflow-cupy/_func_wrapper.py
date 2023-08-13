import tensorflow as tf
import cupy as cp


def to_dlpack_and_back(fn):
    def _to_dlpack_and_back(*args, **kwargs):
        new_args = []
        new_kwargs = {}
        for arg in args:
            if isinstance(arg, tf.Tensor) or isinstance(arg, tf.Variable):
                new_args.append(cp.from_dlpack(tf.experimental.dlpack.to_dlpack(arg)))
            else:
                new_args.append(arg)

        for k, v in kwargs:
            if isinstance(v, tf.Tensor) or isinstance(v, tf.Variable):
                new_kwargs[k] = cp.from_dlpack(tf.experimental.dlpack.to_dlpack(v))
            else:
                new_kwargs[k] = v

        ret = fn(*new_args, **new_kwargs)

        if not isinstance(ret, tuple):
            ret = (ret,)

        out = []

        for i in ret:
            if isinstance(i, cp.ndarray):
                out.append(tf.experimental.dlpack.from_dlpack(i.toDlpack()))
            else:
                out.append(i)

        out = tuple(out)
        if len(out) == 1:
            return out[0]
        else:
            return out

    return _to_dlpack_and_back
