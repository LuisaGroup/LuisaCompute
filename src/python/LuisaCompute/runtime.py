from ._internal.runtime import *
from ._internal.config import INSTALL_DIR
from ._internal.logging import *
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

    def __init__(self, _device, _handle):
        self._as_parameter_ = _handle
        self._device = _device
        self._cmd_buffer = None

    def __del__(self):
        stream_destroy(self.device, self)

    def __enter__(self):
        assert self not in Stream._stream_stack
        Stream._stream_stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type or exc_val or exc_tb:
            return False
        self.commit()
        assert Stream._stream_stack[-1] == self
        Stream._stream_stack.remove(self)

    def commit(self):
        if self._cmd_buffer:
            stream_dispatch(self.device, self, self._cmd_buffer)
        self._cmd_buffer = None

    def synchronize(self):
        self.commit()
        stream_synchronize(self.device, self)

    def _dispatch(self, cmd):
        if not self._cmd_buffer:
            self._cmd_buffer = Stream.CommandBuffer()
        self._cmd_buffer.append(cmd)

    @staticmethod
    def dispatch(cmd: Command):
        assert Stream._stream_stack
        s: Stream = Stream._stream_stack[-1]
        s._dispatch(cmd)

    @property
    def device(self):
        return self._device


class Buffer:
    def __init__(self, _device, _handle, _elem_type, _count, _total_count, _offset, _is_slice=False):
        self._device = _device
        self._elem_type = _elem_type
        self._as_parameter_ = _handle
        self._size = _count
        self._total_size = _total_count
        self._offset = _offset
        self._is_slice = _is_slice
        assert self.offset + self.size <= self.total_size

    def __del__(self):
        if not self.is_slice:
            buffer_destroy(self.device, self)

    def __len__(self):
        return self._size

    def __getitem__(self, item):
        assert isinstance(item, slice)
        start = 0 if item.start is None else item.start
        stop = self.size if item.stop is None else item.stop
        assert item.step is None or item.step == 1 and 0 <= start < stop <= self.size
        return Buffer(
            self.device, self._as_parameter_, self.element,
            stop - start, self.total_size, self.offset + item.start, True)

    @property
    def is_slice(self):
        return self._is_slice

    @property
    def device(self):
        return self._device

    @property
    def element(self):
        return self._elem_type

    @property
    def size(self):
        return self._size

    @property
    def size_bytes(self):
        return self._size * self.element.size

    @property
    def total_size(self):
        return self._total_size

    @property
    def total_size_bytes(self):
        return self._total_size * self.element.size

    @property
    def offset(self):
        return self._offset

    @property
    def offset_bytes(self):
        return self._offset * self.element.size

    def upload(self, data: np.ndarray):
        ptr = np.ascontiguousarray(data).ctypes.data
        command = Command(command_upload_buffer(self, self.offset_bytes, self.size_bytes, ptr))
        Stream.dispatch(command)

    def download(self, data: np.ndarray):
        ptr = np.ascontiguousarray(data).ctypes.data
        command = Command(command_download_buffer(self, self.offset_bytes, self.size_bytes, ptr))
        Stream.dispatch(command)


class Context:
    def __init__(self):
        self._as_parameter_ = context_create(INSTALL_DIR)

    def __del__(self):
        context_destroy(self)

    @property
    def runtime_directory(self):
        return context_runtime_directory(self)

    @property
    def cache_directory(self):
        return context_cache_directory(self)


class Device:
    _context = None
    _device_stack = []

    def __init__(self, name: str, index: int = None):
        if Device._context is None:
            Device._context = Context()
        index = 0 if index is None else max(index, 0)
        self._as_parameter_ = device_create(Device._context, name, index)
        self._default_stream = None

    def __del__(self):
        if self._default_stream is not None:
            del self._default_stream
        device_destroy(Device._context, self)

    def __enter__(self):
        assert self not in Device._device_stack
        Device._device_stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type or exc_val or exc_tb:
            return False
        Device._device_stack.remove(self)

    @property
    def default_stream(self):
        if self._default_stream is None:
            self._default_stream = self.create_stream()
        return self._default_stream

    def create_stream(self):
        return Stream(self, stream_create(self))

    def create_buffer(self, t, count):
        t = Type.of(t)
        b = buffer_create(self, t.size * count, INVALID_HEAP_HANDLE, INVALID_INDEX_IN_HEAP)
        return Buffer(self, b, t, count, count, 0)
