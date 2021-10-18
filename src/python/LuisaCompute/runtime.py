import ctypes

from ._internal.runtime import *
from ._internal.config import INSTALL_DIR
from .type import Type
import numpy as np

INVALID_HEAP_HANDLE = c_uint64(-1)
INVALID_INDEX_IN_HEAP = c_uint32(-1)


class Command:
    def __init__(self, handle):
        self._as_parameter_ = handle


class Stream:

    _stream_stack = []

    class CommandBuffer:
        def __init__(self):
            self._as_parameter_ = command_list_create()

        def append(self, cmd):
            command_list_append(self, cmd)
            return self

        def __bool__(self):
            return not bool(command_list_empty(self))

    def __init__(self, device):
        self._as_parameter_ = stream_create(device)
        self._device = device
        self._cmd_buffer = None

    def __enter__(self):
        Stream._stream_stack.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type or exc_val or exc_tb:
            return False
        self.commit()

    def _commit(self):
        if self._cmd_buffer:
            stream_dispatch(self.device, self, self._cmd_buffer)
        self._cmd_buffer = None

    def _synchronize(self):
        self.commit()
        stream_synchronize(self.device, self)

    def _dispatch(self, cmd):
        if not self._cmd_buffer:
            self._cmd_buffer = Stream.CommandBuffer()
        self._cmd_buffer.append(cmd)

    @staticmethod
    def synchronize():
        assert Stream._stream_stack
        s: Stream = Stream._stream_stack[-1]
        s._synchronize()

    @staticmethod
    def commit():
        assert Stream._stream_stack
        s: Stream = Stream._stream_stack[-1]
        s._commit()

    @staticmethod
    def dispatch(cmd: Command):
        assert Stream._stream_stack
        s: Stream = Stream._stream_stack[-1]
        s._dispatch(cmd)

    @property
    def device(self):
        return self._device


class Resource:
    def __init__(self, device, heap, heap_index, handle):
        self._device = device
        self._heap = heap
        self._heap_index = INVALID_INDEX_IN_HEAP if heap is None else heap_index
        self._as_parameter_ = handle

    @property
    def device(self):
        return self._device

    @property
    def heap(self):
        return self._heap

    @property
    def index_in_heap(self):
        return self._heap_index


class Buffer(Resource):
    def __init__(self, device, elem_type, count, heap=None, heap_index=-1):
        super().__init__(
            device, heap, heap_index,
            buffer_create(
                device, Type.of(elem_type).size * count,
                INVALID_HEAP_HANDLE if heap is None else heap.handle,
                heap_index))
        self._elem_type = Type.of(elem_type)
        self._size = count

    def __del__(self):
        buffer_destroy(self.device, self)

    @property
    def element_type(self):
        return self._elem_type

    def __len__(self):
        return self._size

    @property
    def size(self):
        return self._size

    @property
    def size_bytes(self):
        return self._size * self._elem_type.size

    def upload(self, data: np.ndarray):
        ptr = np.ascontiguousarray(data).ctypes.data
        Stream.dispatch(Command(command_upload_buffer(self, 0, self.size_bytes, ptr)))

    def download(self, data: np.ndarray):
        ptr = np.ascontiguousarray(data).ctypes.data
        Stream.dispatch(Command(command_download_buffer(self, 0, self.size_bytes, ptr)))


class Device:
    def __init__(self, ctx, name: str, index: int = 0):
        self._context = ctx
        self._as_parameter_ = device_create(self._context, name, index)

    def __del__(self):
        device_destroy(self)

    @property
    def context(self):
        return self._context

    def create_stream(self):
        return Stream(self)

    def create_buffer(self, type, size):
        return Buffer(self, type, size)


class Context:
    def __init__(self):
        self._as_parameter_ = context_create(INSTALL_DIR)

    def __del__(self):
        context_destroy(self)

    def create_device(self, backend, index=0):
        return Device(self, backend, index)

    @property
    def runtime_directory(self):
        return context_runtime_directory(self)

    @property
    def cache_directory(self):
        return context_cache_directory(self)
