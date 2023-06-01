//
// Created by Mike Smith on 2023/5/29.
//

#include <sys/mman.h>
#include <compression.h>

#include <core/clock.h>
#include <core/magic_enum.h>
#include <backends/metal/metal_device.h>
#include <backends/metal/metal_buffer.h>
#include <backends/metal/metal_texture.h>
#include <backends/metal/metal_event.h>
#include <backends/metal/metal_command_encoder.h>
#include <backends/metal/metal_dstorage.h>

namespace luisa::compute::metal {

MetalPinnedMemory::MetalPinnedMemory(MTL::Device *device,
                                     void *host_ptr,
                                     size_t size_bytes) noexcept
    : _host_ptr{host_ptr}, _size_bytes{size_bytes},
      _offset{0u}, _device_buffer{nullptr} {
    auto page_size = pagesize();
    auto host_addr = reinterpret_cast<size_t>(host_ptr);
    auto aligned_addr = luisa::align(host_addr, page_size);
    if (host_addr != aligned_addr) { aligned_addr -= page_size; }
    auto aligned_size = luisa::align(host_addr + size_bytes - aligned_addr, page_size);
    if (mlock(reinterpret_cast<void *>(aligned_addr), aligned_size) != 0) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to lock host memory: {}",
            std::strerror(errno));
    } else {
        _device_buffer = device->newBuffer(
            reinterpret_cast<void *>(aligned_addr),
            aligned_size,
            MTL::ResourceOptionCPUCacheModeWriteCombined |
                MTL::ResourceStorageModeShared |
                MTL::ResourceHazardTrackingModeUntracked,
            ^(void *ptr, NS::UInteger size) noexcept {
                munlock(reinterpret_cast<void *>(aligned_addr), aligned_size);
                LUISA_INFO("Unpinned page-aligned memory "
                           "at 0x{:016x} with size {} bytes.",
                           aligned_addr, aligned_size);
            });
        LUISA_INFO("Pinned host memory at 0x{:016x} with size {} bytes "
                   "(page-aligned address = 0x{:016x}, page-aligned size = {}).",
                   host_addr, size_bytes, aligned_addr, aligned_size);
    }
}

MetalPinnedMemory::~MetalPinnedMemory() noexcept {
    if (_device_buffer) { _device_buffer->release(); }
}

void MetalPinnedMemory::set_name(luisa::string_view name) noexcept {
    if (name.empty()) {
        _device_buffer->setLabel(nullptr);
    } else {
        auto ns_name = NS::String::alloc()->init(
            const_cast<char *>(name.data()), name.size(),
            NS::UTF8StringEncoding, false);
        _device_buffer->setLabel(ns_name);
        ns_name->release();
    }
}

MetalFileHandle::MetalFileHandle(MTL::Device *device,
                                 luisa::string_view path,
                                 size_t size_bytes) noexcept
    : _device{device}, _url{nullptr}, _size_bytes{size_bytes} {
    auto ns_path = NS::String::alloc()->init(
        const_cast<char *>(path.data()), path.size(),
        NS::UTF8StringEncoding, false);
    _url = NS::URL::fileURLWithPath(ns_path)->retain();
    ns_path->release();
}

MTL::IOFileHandle *MetalFileHandle::handle(DStorageCompression compression) noexcept {
    NS::Error *error = nullptr;
    MTL::IOFileHandle *handle = nullptr;
    {
        std::scoped_lock lock{_mutex};
        if (auto iter = _handles.find(to_underlying(compression));
            iter != _handles.end()) { return iter->second; }
        // create the handle if not found
        switch (compression) {
            case DStorageCompression::None: handle = _device->newIOHandle(_url, &error); break;
            case DStorageCompression::LZ4: handle = _device->newIOHandle(_url, MTL::IOCompressionMethodLZ4, &error); break;
            case DStorageCompression::Zlib: handle = _device->newIOHandle(_url, MTL::IOCompressionMethodZlib, &error); break;
            case DStorageCompression::LZFSE: handle = _device->newIOHandle(_url, MTL::IOCompressionMethodLZFSE, &error); break;
            case DStorageCompression::LZMA: handle = _device->newIOHandle(_url, MTL::IOCompressionMethodLZMA, &error); break;
            case DStorageCompression::LZBitmap: handle = _device->newIOHandle(_url, MTL::IOCompressionMethodLZBitmap, &error); break;
            default: break;
        }
        // add to the handle map
        if (handle) { _handles.emplace(to_underlying(compression), handle); }
    }

    if (handle) {
        LUISA_INFO("Opened file handle (URL: {}) with compression method {}.",
                   _url->description()->utf8String(),
                   to_string(compression));
        return handle;
    }

    if (error) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to open file handle (URL: {}) with compression method {}: {}",
            _url->description()->utf8String(),
            to_string(compression),
            error->localizedDescription()->utf8String());
    } else {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to open file handle (URL: {}) with compression method {}.",
            _url->description()->utf8String(),
            to_string(compression));
    }
    return handle;
}

MetalFileHandle::~MetalFileHandle() noexcept {
    for (auto [_, handle] : _handles) { handle->release(); }
    _url->release();
}

void MetalFileHandle::set_name(luisa::string_view name) noexcept {
    std::scoped_lock lock{_mutex};
    if (name.empty()) {
        for (auto [_, handle] : _handles) { handle->setLabel(nullptr); }
    } else {
        for (auto [c, handle] : _handles) {
            auto compression = static_cast<DStorageCompression>(c);
            auto name_with_compression = luisa::format(
                "{} (compression = {})",
                name, luisa::to_string(compression));
            auto mtl_name = NS::String::alloc()->init(
                name_with_compression.data(),
                name_with_compression.size(),
                NS::UTF8StringEncoding, false);
            handle->setLabel(mtl_name);
            mtl_name->release();
        }
    }
}

MetalIOStream::MetalIOStream(MTL::Device *device) noexcept
    : MetalStream{device, 0u},
      _io_queue{nullptr},
      _io_event{nullptr} {
    auto desc = MTL::IOCommandQueueDescriptor::alloc()->init();
    desc->setType(MTL::IOCommandQueueTypeConcurrent);
    desc->setPriority(MTL::IOPriorityNormal);
    NS::Error *error = nullptr;
    _io_queue = device->newIOCommandQueue(desc, &error);
    if (error) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to create IO command queue: {}",
            error->localizedDescription()->utf8String());
    } else {
        _io_event = device->newSharedEvent();
        LUISA_INFO("Created IO command queue.");
    }
}

MetalIOStream::~MetalIOStream() noexcept {
    if (_io_queue) { _io_queue->release(); }
    if (_io_event) { _io_event->release(); }
}

void MetalIOStream::set_name(luisa::string_view name) noexcept {
    if (name.empty()) {
        _io_queue->setLabel(nullptr);
    } else {
        auto mtl_name = NS::String::alloc()->init(
            const_cast<char *>(name.data()), name.size(),
            NS::UTF8StringEncoding, false);
        _io_queue->setLabel(mtl_name);
        mtl_name->release();
    }
    MetalStream::set_name(name);
}

class MetalIOCommandEncoder final : public MetalCommandEncoder {

private:
    MTL::IOCommandBuffer *_io_command_buffer{nullptr};

private:
    [[nodiscard]] auto _io_stream() noexcept { return static_cast<MetalIOStream *>(stream()); }

    void _prepare_io_command_buffer() noexcept {
        if (!_io_command_buffer) {
            auto queue = _io_stream()->io_queue();
            _io_command_buffer = queue->commandBufferWithUnretainedReferences();
        }
    }

    void _copy_from_file(MTL::IOFileHandle *handle, size_t offset,
                         DStorageReadCommand::Request request) noexcept {
        _prepare_io_command_buffer();
        if (luisa::holds_alternative<DStorageReadCommand::BufferRequest>(request)) {
            auto r = luisa::get<DStorageReadCommand::BufferRequest>(request);
            auto buffer = reinterpret_cast<MetalBuffer *>(r.handle);
            _io_command_buffer->loadBuffer(buffer->handle(), r.offset_bytes, r.size_bytes, handle, offset);
        } else if (luisa::holds_alternative<DStorageReadCommand::TextureRequest>(request)) {
            auto r = luisa::get<DStorageReadCommand::TextureRequest>(request);
            auto texture = reinterpret_cast<MetalTexture *>(r.handle);
            auto size = make_uint3(r.size[0], r.size[1], r.size[2]);
            auto pitch_size = pixel_storage_size(texture->storage(), make_uint3(size.x, 1u, 1u));
            auto image_size = pixel_storage_size(texture->storage(), make_uint3(size.xy(), 1u));
            _io_command_buffer->loadTexture(texture->handle(), 0u, r.level,
                                            MTL::Size{r.size[0], r.size[1], r.size[2]},
                                            pitch_size, image_size, MTL::Origin{0, 0, 0},
                                            handle, offset);
        } else if (luisa::holds_alternative<DStorageReadCommand::MemoryRequest>(request)) {
            auto r = luisa::get<DStorageReadCommand::MemoryRequest>(request);
            _io_command_buffer->loadBytes(r.data, r.size_bytes, handle, offset);
        } else {
            LUISA_ERROR_WITH_LOCATION("Unsupported request type.");
        }
    }

    void _copy_from_memory(MTL::Buffer *buffer, size_t offset,
                           DStorageReadCommand::Request request) noexcept {
        auto encoder = command_buffer()->blitCommandEncoder();
        if (luisa::holds_alternative<DStorageReadCommand::BufferRequest>(request)) {
            auto r = luisa::get<DStorageReadCommand::BufferRequest>(request);
            auto dst = reinterpret_cast<MetalBuffer *>(r.handle);
            encoder->copyFromBuffer(buffer, offset, dst->handle(), r.offset_bytes, r.size_bytes);
        } else if (luisa::holds_alternative<DStorageReadCommand::TextureRequest>(request)) {
            auto r = luisa::get<DStorageReadCommand::TextureRequest>(request);
            auto dst = reinterpret_cast<MetalTexture *>(r.handle);
            auto size = make_uint3(r.size[0], r.size[1], r.size[2]);
            auto pitch_size = pixel_storage_size(dst->storage(), make_uint3(size.x, 1u, 1u));
            auto image_size = pixel_storage_size(dst->storage(), make_uint3(size.xy(), 1u));
            encoder->copyFromBuffer(buffer, offset, pitch_size, image_size,
                                    MTL::Size{r.size[0], r.size[1], r.size[2]},
                                    dst->handle(), 0u, r.level, MTL::Origin{0, 0, 0});
        } else if (luisa::holds_alternative<DStorageReadCommand::MemoryRequest>(request)) {
            auto r = luisa::get<DStorageReadCommand::MemoryRequest>(request);
            with_download_buffer(r.size_bytes, [&](auto download_buffer) noexcept {
                encoder->copyFromBuffer(buffer, offset,
                                        download_buffer->buffer(),
                                        download_buffer->offset(),
                                        r.size_bytes);
                add_callback(FunctionCallbackContext::create([download_buffer, data = r.data, size = r.size_bytes] {
                    memcpy(data, download_buffer->data(), size);
                }));
            });
        } else {
            LUISA_ERROR_WITH_LOCATION("Unsupported request type.");
        }
        encoder->endEncoding();
    }

public:
    using MetalCommandEncoder::MetalCommandEncoder;
    MTL::CommandBuffer *submit(CommandList::CallbackContainer &&user_callbacks) noexcept override {
        if (!user_callbacks.empty()) { _prepare_io_command_buffer(); }
        if (_io_command_buffer) {
            auto stream = _io_stream();
            auto io_event_value = stream->signal(_io_command_buffer);
            _io_command_buffer->commit();
            command_buffer()->encodeWait(stream->io_event(), io_event_value);
        }
        return MetalCommandEncoder::submit(std::move(user_callbacks));
    }

    void visit(DStorageReadCommand *command) noexcept {
        if (luisa::holds_alternative<DStorageReadCommand::FileSource>(command->source())) {
            auto src = luisa::get<DStorageReadCommand::FileSource>(command->source());
            auto file = reinterpret_cast<MetalFileHandle *>(src.handle);
            LUISA_ASSERT(src.offset_bytes < file->size() &&
                             src.size_bytes <= file->size() - src.offset_bytes,
                         "Invalid offset or size for DStorageReadCommand.");
            auto file_handle = file->handle(command->compression());
            LUISA_ASSERT(file_handle != nullptr, "Failed to open file handle.");
            _copy_from_file(file_handle, src.offset_bytes, command->request());
        } else if (luisa::holds_alternative<DStorageReadCommand::MemorySource>(command->source())) {
            LUISA_ASSERT(command->compression() == DStorageCompression::None,
                         "Memory source does not support compression.");
            auto src = luisa::get<DStorageReadCommand::MemorySource>(command->source());
            auto memory = reinterpret_cast<MetalPinnedMemory *>(src.handle);
            LUISA_ASSERT(src.offset_bytes < memory->size() &&
                             src.size_bytes <= memory->size() - src.offset_bytes,
                         "Invalid offset or size for DStorageReadCommand.");
            _copy_from_memory(memory->device_buffer(),
                              src.offset_bytes + memory->device_buffer_offset(),
                              command->request());
        } else {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid source type for DStorageReadCommand.");
        }
    }
};

void MetalIOStream::dispatch(CommandList &&list) noexcept {
    MetalIOCommandEncoder encoder{this};
    _do_dispatch(encoder, std::move(list));
}

void MetalIOStream::_encode(MetalCommandEncoder &encoder,
                            Command *command) noexcept {
    LUISA_ASSERT(command->tag() == Command::Tag::ECustomCommand &&
                     static_cast<CustomCommand *>(command)->uuid() ==
                         to_underlying(CustomCommandUUID::DSTORAGE_READ),
                 "Invalid command type for MetalIOStream.");
    auto io_encoder = dynamic_cast<MetalIOCommandEncoder *>(&encoder);
    LUISA_ASSERT(io_encoder != nullptr, "Invalid encoder type for MetalIOStream.");
    io_encoder->visit(static_cast<DStorageReadCommand *>(command));
}

uint64_t MetalIOStream::signal(MTL::IOCommandBuffer *command_buffer) noexcept {
    auto io_event_value = [this] {
        std::scoped_lock lock{_event_mutex};
        return ++_event_value;
    }();
    command_buffer->signalEvent(_io_event, io_event_value);
    return io_event_value;
}

void MetalIOStream::signal(MetalEvent *event) noexcept {
    auto io_command_buffer = _io_queue->commandBufferWithUnretainedReferences();
    auto io_event_value = signal(io_command_buffer);
    auto command_buffer = MetalStream::queue()->commandBufferWithUnretainedReferences();
    command_buffer->encodeWait(_io_event, io_event_value);
    event->signal(command_buffer);
    command_buffer->commit();
}

void MetalIOStream::wait(MetalEvent *event) noexcept {
    auto io_command_buffer = _io_queue->commandBufferWithUnretainedReferences();
    io_command_buffer->wait(event->handle(), event->value_to_wait());
    io_command_buffer->commit();
    MetalStream::wait(event);
}

void MetalIOStream::synchronize() noexcept {
    auto io_command_buffer = _io_queue->commandBufferWithUnretainedReferences();
    auto io_event_value = signal(io_command_buffer);
    io_command_buffer->commit();
    auto command_buffer = MetalStream::queue()->commandBufferWithUnretainedReferences();
    command_buffer->encodeWait(_io_event, io_event_value);
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
}

MetalDStorageExt::MetalDStorageExt(MetalDevice *device) noexcept
    : _device{device} {}

DeviceInterface *MetalDStorageExt::device() const noexcept { return _device; }

[[nodiscard]] ResourceCreationInfo MetalDStorageExt::create_stream_handle(const DStorageStreamOption &option) noexcept {
    return with_autorelease_pool([this] {
        auto p = luisa::new_with_allocator<MetalIOStream>(_device->handle());
        if (!p->valid()) {
            luisa::delete_with_allocator(p);
            return ResourceCreationInfo::make_invalid();
        }
        ResourceCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(p);
        info.native_handle = p;
        return info;
    });
}

[[nodiscard]] DStorageExt::FileCreationInfo MetalDStorageExt::open_file_handle(luisa::string_view path) noexcept {
    return with_autorelease_pool([=, this] {
        std::error_code ec;
        auto size = std::filesystem::file_size(path, ec);
        if (ec) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to open file handle (path: {}): {}",
                path, ec.message());
            return FileCreationInfo::make_invalid();
        }
        auto p = luisa::new_with_allocator<MetalFileHandle>(_device->handle(), path, size);
        FileCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(p);
        info.native_handle = p;
        info.size_bytes = size;
        return info;
    });
}

[[nodiscard]] DStorageExt::PinnedMemoryInfo MetalDStorageExt::pin_host_memory(void *ptr, size_t size_bytes) noexcept {
    return with_autorelease_pool([=, this] {
        auto pinned = luisa::new_with_allocator<MetalPinnedMemory>(_device->handle(), ptr, size_bytes);
        if (!pinned->valid()) {
            luisa::delete_with_allocator(pinned);
            return PinnedMemoryInfo::make_invalid();
        }
        PinnedMemoryInfo info{};
        info.handle = reinterpret_cast<uint64_t>(pinned);
        info.native_handle = pinned;
        info.size_bytes = size_bytes;
        return info;
    });
}

void MetalDStorageExt::close_file_handle(uint64_t handle) noexcept {
    with_autorelease_pool([=] {
        auto file = reinterpret_cast<MetalFileHandle *>(handle);
        luisa::delete_with_allocator(file);
    });
}

void MetalDStorageExt::unpin_host_memory(uint64_t handle) noexcept {
    with_autorelease_pool([=] {
        auto pinned = reinterpret_cast<MetalPinnedMemory *>(handle);
        luisa::delete_with_allocator(pinned);
    });
}

namespace detail {

[[nodiscard]] inline luisa::optional<compression_algorithm>
compression_cpu_select_algorithm(DStorageCompression c) noexcept {
    switch (c) {
        case DStorageCompression::LZ4: return COMPRESSION_LZ4;
        case DStorageCompression::Zlib: return COMPRESSION_ZLIB;
        case DStorageCompression::LZFSE: return COMPRESSION_LZFSE;
        case DStorageCompression::LZMA: return COMPRESSION_LZMA;
        case DStorageCompression::LZBitmap: return COMPRESSION_LZ4_RAW;
        default: break;
    }
    return luisa::nullopt;
}

}// namespace detail

void MetalDStorageExt::compress(const void *data, size_t size_bytes,
                                Compression algorithm, CompressionQuality quality,
                                luisa::vector<std::byte> &result) noexcept {

    Clock clk;

    if (size_bytes == 0u) {
        LUISA_WARNING_WITH_LOCATION("Empty data to compress.");
        return;
    }

    if (algorithm == DStorageCompression::None) {
        LUISA_WARNING_WITH_LOCATION("No compression algorithm specified. "
                                    "The data will be copied as-is.");
        result.resize(size_bytes);
        std::memcpy(result.data(), data, size_bytes);
        return;
    }

    auto algo = detail::compression_cpu_select_algorithm(algorithm);
    LUISA_ASSERT(algo.has_value(),
                 "Unsupported compression algorithm: {}.",
                 to_string(algorithm));

    // initialize the compression stream
    compression_stream stream{};
    auto stream_state = compression_stream_init(
        &stream, COMPRESSION_STREAM_ENCODE, algo.value());

    LUISA_ASSERT(stream_state == COMPRESSION_STATUS_OK,
                 "Failed to initialize compression stream.");

    // set input
    result.resize(std::bit_ceil(std::max<size_t>(
        static_cast<size_t>(.2 * static_cast<double>(size_bytes)), 1u << 16u)));
    stream.dst_ptr = reinterpret_cast<uint8_t *>(result.data());
    stream.dst_size = result.size_bytes();
    stream.src_ptr = reinterpret_cast<const uint8_t *>(data);
    stream.src_size = size_bytes;

    for (; stream_state != COMPRESSION_STATUS_END;
         stream_state = compression_stream_process(
             &stream, COMPRESSION_STREAM_FINALIZE)) {
        LUISA_ASSERT(stream_state == COMPRESSION_STATUS_OK,
                     "Failed to compress data.");
        auto offset = stream.dst_ptr - reinterpret_cast<uint8_t *>(result.data());
        result.resize(result.size_bytes() * 2u);
        stream.dst_ptr = reinterpret_cast<uint8_t *>(result.data()) + offset;
        stream.dst_size = result.size_bytes() - offset;
    }
    result.resize(stream.dst_ptr - reinterpret_cast<uint8_t *>(result.data()));
    compression_stream_destroy(&stream);

    auto ratio = static_cast<double>(result.size_bytes()) /
                 static_cast<double>(size_bytes);
    LUISA_INFO("Compressed {} bytes to {} bytes "
               "(ratio = {}) with {} in {} ms.",
               size_bytes, result.size_bytes(), ratio,
               to_string(algorithm), clk.toc());
}

}// namespace luisa::compute::metal
