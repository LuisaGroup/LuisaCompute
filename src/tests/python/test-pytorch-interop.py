import luisa
import torch

luisa.init('cuda')

import numpy as np
import cupy
import ctypes


def cu_device_ptr_to_torch_tensor(ptr, shape, dtype=cupy.float32):
    """
    Convert a CUdeviceptr to a PyTorch tensor.

    Args:
        ptr (ctypes.c_uint64): CUdeviceptr pointing to the GPU memory.
        shape (tuple): Shape of the tensor.
        dtype (cupy.dtype): Data type of the tensor. Default is cupy.float32.

    Returns:
        torch.Tensor: PyTorch tensor.
    """

    size_bytes = cupy.dtype(dtype).itemsize * np.prod(shape)

    # Create an UnownedMemory view of the CUdeviceptr
    umem = cupy.cuda.memory.UnownedMemory(int(ptr), size_bytes, owner=None)
    memptr = cupy.cuda.memory.MemoryPointer(umem, 0)

    # Convert the MemoryPointer to a CuPy ndarray
    array = cupy.ndarray(shape, dtype=dtype, memptr=memptr)

    # Convert the CuPy ndarray to a DLPack tensor and then to a PyTorch tensor
    return torch.utils.dlpack.from_dlpack(array.toDlpack())


def lc_buffer_to_torch(buf):
    assert buf.dtype is float  # TODO
    shape = (buf.size,)
    return cu_device_ptr_to_torch_tensor(buf.native_handle, shape)


def torch_to_lc_buffer(tensor):
    assert tensor.dtype is torch.float32  # TODO
    size = np.prod(tensor.shape)
    buf = luisa.Buffer.import_external_memory(
        tensor.contiguous().data_ptr(),
        size, dtype=float)
    return buf


a = luisa.buffer([0.9, 1.0, 1.1])
luisa.synchronize()
b = lc_buffer_to_torch(a)
b[1] = 2.0
torch.cuda.synchronize()


# del b

@luisa.func
def f():
    a.write(0, 3.0)


f(dispatch_size=1)
print(a.numpy())  # [3.  2.  1.1]
print(b)  # [3.  2.  1.1]
