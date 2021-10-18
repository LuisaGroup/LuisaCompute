from ._internal.runtime import *
from ._internal.config import INSTALL_DIR
from .type import Type


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


INVALID_HEAP_HANDLE = c_uint64(-1)
INVALID_INDEX_IN_HEAP = c_uint32(-1)


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
        buffer_destroy(
            self.device, self,
            INVALID_HEAP_HANDLE if self.heap is None else self.heap,
            self.index_in_heap)

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


class Device:
    def __init__(self, ctx: Context, name: str, index: int = 0):
        self._context = ctx
        self._as_parameter_ = device_create(self._context, name, index)

    def __del__(self):
        device_destroy(self)

    @property
    def context(self):
        return self._context

    def create_buffer(self, type, size):
        return Buffer(self, type, size)
