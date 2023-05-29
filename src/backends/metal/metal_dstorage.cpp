//
// Created by Mike Smith on 2023/5/29.
//

#include <sys/mman.h>
#include <compression.h>

#include <core/clock.h>
#include <core/magic_enum.h>
#include <backends/metal/metal_device.h>
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
    // TODO: set name
}

MetalFileHandle::MetalFileHandle(MTL::Device *device,
                                 luisa::string_view path,
                                 size_t size_bytes) noexcept
    : _device{device}, _url{nullptr}, _size_bytes{size_bytes} {
    auto ns_path = NS::String::alloc()->init(
        const_cast<char *>(path.data()), path.size(),
        NS::UTF8StringEncoding, false);
    _url = NS::URL::fileURLWithPath(ns_path);
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
        LUISA_INFO("Opened file handle (URL: {}) with {}.",
                   _url->description()->utf8String(),
                   to_string(compression));
        return handle;
    }

    if (error) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to open file handle (URL: {}) with {}: {}",
            _url->description()->utf8String(),
            to_string(compression),
            error->localizedDescription()->utf8String());
    } else {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to open file handle (URL: {}) with {}.",
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
    // TODO
}

MetalIOStream::MetalIOStream(MTL::Device *device) noexcept
    : MetalStream{device, 0u}, _io_queue{nullptr} {
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
        LUISA_INFO("Created IO command queue.");
    }
}

MetalIOStream::~MetalIOStream() noexcept {
    if (_io_queue) { _io_queue->release(); }
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
    LUISA_ASSERT(algo.has_value(), "Unsupported compression algorithm.");

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
