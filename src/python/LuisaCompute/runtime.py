from ._internal.runtime import *
from ._internal.config import INSTALL_DIR
from ._internal.logging import *
from .pixel import *
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
    def current():
        assert Stream._stream_stack
        s: Stream = Stream._stream_stack[-1]
        return s

    @staticmethod
    def dispatch(cmd: Command):
        Stream.current()._dispatch(cmd)

    @property
    def device(self):
        return self._device


class Buffer:
    def __init__(self, _device, _handle, _elem_type, _count, _total_count, _offset, _is_slice=False):
        self._device = _device
        self._elem_type: Type = _elem_type
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
        assert item.step is None or item.step == 1
        assert 0 <= start < stop <= self.size
        return Buffer(
            self.device, self._as_parameter_, self.element,
            stop - start, self.total_size, self.offset + item.start, True)

    def reinterpret(self, t):
        t = Type.of(t)
        assert self.offset_bytes % t.alignment == 0 and self.offset_bytes % t.size == 0
        offset = self.offset_bytes // t.size
        count = self.size_bytes // t.size
        total_count = self.total_size_bytes // t.size
        return Buffer(self.device, self._as_parameter_, t, count, total_count, offset, True)

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

    def copy_from(self, src):
        if isinstance(src, np.ndarray):
            self.upload(src)
        elif isinstance(src, Buffer):
            assert src.size_bytes == self.size_bytes
            cmd = command_copy_buffer_to_buffer(src, src.offset_bytes, self, self.offset_bytes, self.size_bytes)
            Stream.dispatch(Command(cmd))
        elif isinstance(src, Texture):
            # TODO: check size
            offset = src.offset if src.dimension == 3 else (src.offset[0], src.offset[1], 0)
            size = src.size if src.dimension == 3 else (src.size[0], src.size[1], 1)
            cmd = command_copy_texture_to_buffer(self, self.offset_bytes, src, src.storage, 0, *offset, *size)
            Stream.dispatch(Command(cmd))
        else:
            raise TypeError(f"Invalid source to copy buffer from: {src}")

    def copy_to(self, dst):
        if isinstance(dst, np.ndarray):
            self.download(dst)
        else:
            dst.copy_from(self)


class Texture:
    def __init__(self, _device, _handle, _format, _dim, _size, _total_size, _offset, _is_slice=False):
        self._device = _device
        self._as_parameter_ = _handle
        self._format = _format
        self._dim = _dim
        self._size = _size
        self._total_size = _total_size
        self._offset = _offset
        self._is_slice = _is_slice

    def __del__(self):
        if not self.is_slice:
            texture_destroy(self.device, self)

    @property
    def device(self):
        return self._device

    @property
    def is_slice(self):
        return self._is_slice

    @property
    def format(self):
        return self._format

    @property
    def storage(self):
        return pixel_format_to_storage(self.format)

    @property
    def dimension(self):
        return self._dim

    @property
    def size(self):
        return self._size[:self.dimension]

    @property
    def total_size(self):
        return self._total_size[:self.dimension]

    @property
    def offset(self):
        return self._offset[:self.dimension]

    def __getitem__(self, item):
        def fix(i):
            if i < len(item):
                print(self._offset, self._size)
                start = 0 if item[i].start is None else item[i].start
                stop = self._size[i] if item[i].stop is None else item[i].stop
                assert item[i].step is None or item[i].step == 1
                assert 0 <= start < stop <= self._size[i]
                return self._offset[i] + start, stop - start
            return self._offset[i], self._size[i]

        assert isinstance(item, tuple) and len(item) <= self.dimension and all(isinstance(x, slice) for x in item)
        ranges = tuple(fix(i) for i in range(3))
        return Texture(
            self.device, self._as_parameter_, self.format, self.dimension,
            tuple(r[1] for r in ranges), self._total_size,
            tuple(r[0] for r in ranges), True)

    def upload(self, data: np.ndarray):
        ptr = np.ascontiguousarray(data).ctypes.data
        command = Command(command_upload_texture(
            self, self.storage, 0,
            *self._offset, *self._size, ptr))
        Stream.dispatch(command)

    def download(self, data: np.ndarray):
        ptr = np.ascontiguousarray(data).ctypes.data
        command = Command(command_download_texture(
            self, self.storage, 0,
            *self._offset, *self._size, ptr))
        Stream.dispatch(command)

    def copy_from(self, src):
        if isinstance(src, np.ndarray):
            self.upload(src)
        elif isinstance(src, Buffer):
            # TODO: check size
            cmd = command_copy_buffer_to_texture(
                src, src.offset_bytes, self, self.storage, 0,
                *self._offset, *self._size)
            Stream.dispatch(Command(cmd))
        elif isinstance(src, Texture):
            # TODO: check size
            cmd = command_copy_texture_to_texture(
                src, 0, *src._offset,
                self, 0, *self._offset,
                *self._size)
            Stream.dispatch(Command(cmd))
        else:
            raise TypeError(f"Invalid source to copy texture from: {src}")

    def copy_to(self, dst):
        if isinstance(dst, np.ndarray):
            self.download(dst)
        else:
            dst.copy_from(self)


class Event:
    def __init__(self, _device, _handle):
        self._device = _device
        self._as_parameter_ = _handle

    def __del__(self):
        event_destroy(self.device, self)

    @property
    def device(self):
        return self._device

    def synchronize(self):
        event_synchronize(self.device, self)

    def wait(self):
        event_wait(self.device, self, Stream.current())

    def signal(self):
        event_signal(self.device, self, Stream.current())


ACCEL_BUILD_HINT_FAST_TRACE = 0
ACCEL_BUILD_HINT_FAST_UPDATE = 1
ACCEL_BUILD_HINT_FAST_REBUILD = 2


class Mesh:
    def __init__(self, _device, _handle):
        self._device = _device
        self._as_parameter_ = _handle
        self._vertex_buffer = None
        self._triangle_buffer = None

    def __del__(self):
        mesh_destroy(self.device, self)

    @property
    def device(self):
        return self._device

    @property
    def handle(self):
        return self._as_parameter_

    @property
    def vertex_buffer(self):
        return self._vertex_buffer

    @property
    def triangle_buffer(self):
        return self._triangle_buffer

    def build(self, hint: int, vertices: Buffer, triangles: Buffer):
        self._vertex_buffer = vertices
        self._triangle_buffer = triangles
        cmd = command_build_mesh(
            self, hint,
            vertices, vertices.offset_bytes,
            vertices.element.size, vertices.size,
            triangles, triangles.offset_bytes, triangles.size)
        Stream.dispatch(cmd)

    def update(self):
        assert self._vertex_buffer is not None and self._triangle_buffer is not None
        cmd = command_update_mesh(self)
        Stream.dispatch(cmd)


class Accel:
    def __init__(self, _device, _handle):
        self._device = _device
        self._handle = _handle
        self._meshes = None
        self._transforms = None

    def __del__(self):
        accel_destroy(self.device, self)

    @property
    def device(self):
        return self._device

    def build(self, hint: int, meshes, transforms):
        count = len(meshes)
        transforms = np.array(transforms, dtype=np.float32)
        assert len(transforms.shape) == 3 and transforms.shape[1] == transforms.shape[2] == 4
        assert len(transforms) == count
        self._meshes = np.array([m.handle for m in meshes], dtype=np.uint64)
        self._transforms = transforms
        cmd = command_build_accel(
            self, hint,
            np.ascontiguousarray(self._meshes),
            np.ascontiguousarray(self._transforms),
            count)
        Stream.dispatch(cmd)

    def update(self, new_transforms=None, offset=None):
        if new_transforms is None:
            cmd = command_update_accel(self, None, 0, 0)
        else:
            offset = 0 if offset is None else offset
            transforms = np.array(new_transforms)
            assert len(transforms.shape) == 3 and transforms.shape[1] == transforms.shape[2] == 4
            assert offset >= 0 and offset + len(transforms) <= len(self._meshes)
            self._transforms[offset:offset + len(transforms)] = transforms
            cmd = command_update_accel(self, np.ascontiguousarray(self._transforms), offset, len(transforms))
        Stream.dispatch(cmd)


HEAP_SLOT_COUNT = 65536


class Heap:

    def __init__(self, _device, _handle, _capacity):
        self._as_parameter_ = _handle
        self._device = _device
        self._capacity = _capacity
        self._max_buffer_index = 0
        self._max_texture_index = 0
        self._buffer_slots = [None for _ in range(HEAP_SLOT_COUNT)]
        self._texture_slots = [None for _ in range(HEAP_SLOT_COUNT)]

    def __del__(self):
        for i in range(self._max_buffer_index):
            buffer = self._buffer_slots[i]
            if buffer is not None:
                buffer_destroy(self.device, buffer)
        for i in range(self._max_texture_index):
            texture = self._texture_slots[i]
            if texture is not None:
                texture_destroy(self.device, texture)
        heap_destroy(self.device, self)

    def _destroy_buffer(self, index):
        assert 0 <= index < HEAP_SLOT_COUNT
        if self._buffer_slots[index] is not None:
            buffer_destroy(self.device, self._buffer_slots[index])
            self._buffer_slots[index] = None

    def _destroy_texture(self, index):
        assert 0 <= index < HEAP_SLOT_COUNT
        if self._texture_slots[index] is not None:
            texture_destroy(self.device, self._texture_slots[index])
            self._texture_slots[index] = None

    @property
    def device(self):
        return self._device

    @property
    def capacity(self):
        return self._capacity

    @property
    def memory_usage(self):
        return heap_query_usage(self.device, self)

    def create_buffer(self, index, t, count):
        t = Type.of(t)
        self._destroy_buffer(index)
        handle = buffer_create(self.device, t.size * count, self, index)
        self._buffer_slots[index] = handle
        self._max_buffer_index = max(self._max_buffer_index, index)
        return Buffer(self.device, handle, t, count, count, 0, True)

    def _create_texture(self, index, dim, fmt, size):
        assert 0 <= fmt <= PIXEL_FORMAT_RGBA32F
        assert dim == 2 or dim == 3
        assert len(size) == dim
        self._destroy_texture(index)
        size = (size[0], size[1], 1) if dim == 2 else (size[0], size[1], size[2])
        handle = texture_create(
            self, fmt, dim,
            *size, 1, 0,
            self, index)
        self._texture_slots[index] = handle
        self._max_texture_index = max(self._max_texture_index, index)
        return Texture(self, handle, fmt, dim, size, size, (0, 0, 0), True)

    def create_image(self, index, fmt, size):
        if isinstance(size, int):
            size = (size, size)
        return self._create_texture(index, 2, fmt, size)

    def create_volume(self, index, fmt, size):
        if isinstance(size, int):
            size = (size, size, size)
        return self._create_texture(index, 3, fmt, size)


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

    def create_event(self):
        return Event(self, event_create(self))

    def create_buffer(self, t, count):
        t = Type.of(t)
        b = buffer_create(self, t.size * count, INVALID_HEAP_HANDLE, INVALID_INDEX_IN_HEAP)
        return Buffer(self, b, t, count, count, 0)

    def _create_texture(self, fmt, dim, size):
        assert 0 <= fmt <= PIXEL_FORMAT_RGBA32F
        assert dim == 2 or dim == 3
        assert len(size) == dim
        size = (size[0], size[1], 1) if dim == 2 else (size[0], size[1], size[2])
        handle = texture_create(
            self, fmt, dim,
            *size, 1, 0,
            INVALID_HEAP_HANDLE,
            INVALID_INDEX_IN_HEAP)
        return Texture(self, handle, fmt, dim, size, size, (0, 0, 0))

    def create_image(self, fmt, size):
        if isinstance(size, int):
            size = (size, size)
        return self._create_texture(fmt, 2, size)

    def create_volume(self, fmt, size):
        if isinstance(size, int):
            size = (size, size, size)
        return self._create_texture(fmt, 3, size)

    def create_heap(self, capacity):
        handle = heap_create(self, capacity)
        return Heap(self, handle, capacity)

    def create_mesh(self):
        handle = mesh_create(self)
        return Mesh(self, handle)

    def create_accel(self):
        handle = accel_create(self)
        return Accel(self, handle)
