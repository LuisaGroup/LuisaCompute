#include <cuda.h>

#include <luisa/core/clock.h>
#include <luisa/core/magic_enum.h>

#include "cuda_dstorage.h"

#ifdef LUISA_PLATFORM_WINDOWS
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace luisa::compute::cuda {

CUDAPinnedMemory::CUDAPinnedMemory(void *p, size_t size) noexcept
    : _host_pointer{p}, _device_address{}, _size_bytes{size} {
    LUISA_CHECK_CUDA(cuMemHostRegister(
        p, size,
        CU_MEMHOSTREGISTER_DEVICEMAP |
            CU_MEMHOSTREGISTER_READ_ONLY));
    LUISA_CHECK_CUDA(cuMemHostGetDevicePointer(
        &_device_address, p, 0));
    LUISA_VERBOSE("Registered host memory at 0x{:016x} with {} "
                  "byte(s) into device address space at 0x{:016x}.",
                  reinterpret_cast<uint64_t>(p), size, _device_address);
}

CUDAPinnedMemory::~CUDAPinnedMemory() noexcept {
    LUISA_CHECK_CUDA(cuMemHostUnregister(_host_pointer));
}

CUDAMappedFile::CUDAMappedFile(luisa::string_view path) noexcept
    : _file_handle{},
      _file_mapping{},
      _mapped_pointer{},
      _device_address{},
      _size_bytes{} {

    auto file_name = luisa::string{path};

#ifdef LUISA_PLATFORM_WINDOWS
    auto file_handle = CreateFileA(file_name.c_str(), GENERIC_READ, FILE_SHARE_READ,
                                   nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_handle == INVALID_HANDLE_VALUE) {
        LUISA_WARNING_WITH_LOCATION("Failed to open file: {}", file_name);
        return;
    }
    LARGE_INTEGER file_size{};
    if (!GetFileSizeEx(file_handle, &file_size)) {
        LUISA_WARNING_WITH_LOCATION("Failed to get file size: {}", file_name);
        CloseHandle(file_handle);
        return;
    }
    auto file_mapping = CreateFileMapping(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (file_mapping == nullptr) {
        LUISA_WARNING_WITH_LOCATION("Failed to create file mapping: {}", file_name);
        CloseHandle(file_handle);
        return;
    }
    auto mapped_address = MapViewOfFile(file_mapping, FILE_MAP_READ, 0, 0, 0);
    if (mapped_address == nullptr) {
        LUISA_WARNING_WITH_LOCATION("Failed to map file: {}", file_name);
        CloseHandle(file_mapping);
        CloseHandle(file_handle);
        return;
    }
    _file_handle = file_handle;
    _file_mapping = file_mapping;
    _mapped_pointer = mapped_address;
    _size_bytes = file_size.QuadPart;
#else
    auto file_handle = open(file_name.c_str(), O_RDONLY, S_IRUSR);
    if (file_handle == -1) {
        LUISA_WARNING_WITH_LOCATION("Failed to open file: {}", file_name);
        return;
    }
    struct stat file_stat {};
    if (fstat(file_handle, &file_stat) == -1) {
        LUISA_WARNING_WITH_LOCATION("Failed to get file size: {}", file_name);
        close(file_handle);
        return;
    }
    auto mapped_address = mmap(nullptr, file_stat.st_size, PROT_READ, MAP_PRIVATE, file_handle, 0);
    if (mapped_address == MAP_FAILED) {
        LUISA_WARNING_WITH_LOCATION("Failed to map file: {}", file_name);
        close(file_handle);
        return;
    }
    _file_handle = reinterpret_cast<void *>(static_cast<uint64_t>(file_handle));
    _mapped_pointer = mapped_address;
    _size_bytes = file_stat.st_size;
#endif
    LUISA_CHECK_CUDA(cuMemHostRegister(
        _mapped_pointer, _size_bytes,
        CU_MEMHOSTREGISTER_DEVICEMAP |
            CU_MEMHOSTREGISTER_READ_ONLY));
    LUISA_CHECK_CUDA(cuMemHostGetDevicePointer(
        &_device_address, _mapped_pointer, 0));
    LUISA_VERBOSE("Mapped file '{}' to host address "
                  "0x{:016x} and device address 0x{:016x}.",
                  path,
                  reinterpret_cast<uint64_t>(_mapped_pointer),
                  _device_address);
}

CUDAMappedFile::~CUDAMappedFile() noexcept {
    if (_device_address) {
        LUISA_CHECK_CUDA(cuMemHostUnregister(_mapped_pointer));
    }
#ifdef LUISA_PLATFORM_WINDOWS
    if (_mapped_pointer != nullptr) {
        UnmapViewOfFile(_mapped_pointer);
        CloseHandle(_file_mapping);
        CloseHandle(_file_handle);
    }
#else
    if (_mapped_pointer != nullptr) {
        munmap(_mapped_pointer, _size_bytes);
        auto fd = static_cast<int>(reinterpret_cast<uint64_t>(_file_handle));
        close(fd);
    }
#endif
}

namespace detail {

#ifdef LUISA_COMPUTE_ENABLE_NVCOMP
static void cuda_compress_cpu(nvcomp::PimplManager &manager,
                              const std::byte *data, size_t size,
                              DStorageCompressionQuality quality,
                              vector<std::byte> &result) noexcept {
    try {
        auto config = manager.configure_compression(size);
        auto max_output_size = luisa::align(config.max_compressed_buffer_size, 16u);
        auto scratch_size = luisa::align(manager.get_required_scratch_buffer_size(), 16u);
        auto temp_buffer = static_cast<CUdeviceptr>(0u);
        auto temp_buffer_size = max_output_size + scratch_size + size;
        LUISA_CHECK_CUDA(cuMemAllocAsync(&temp_buffer, temp_buffer_size, nullptr));
        auto output_buffer = temp_buffer;
        auto scratch_buffer = output_buffer + max_output_size;
        manager.set_scratch_buffer(reinterpret_cast<uint8_t *>(scratch_buffer));
        auto input_buffer = scratch_buffer + scratch_size;
        LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(input_buffer, data, size, nullptr));
        manager.compress(reinterpret_cast<const uint8_t *>(input_buffer),
                         reinterpret_cast<uint8_t *>(output_buffer), config);
        result.resize(max_output_size);
        LUISA_CHECK_CUDA(cuMemcpyDtoHAsync(result.data(), output_buffer, max_output_size, nullptr));
        LUISA_CHECK_CUDA(cuMemFreeAsync(temp_buffer, nullptr));
        LUISA_CHECK_CUDA(cuStreamSynchronize(nullptr));
        auto compressed_size = manager.get_compressed_output_size(reinterpret_cast<uint8_t *>(result.data()));
        result.resize(compressed_size);
    } catch (const std::exception &e) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to compress data using nvCOMP: {}",
            e.what());
    }
}
#endif

}// namespace detail

void CUDADStorageExt::compress(const void *data, size_t size_bytes,
                               DStorageExt::Compression algorithm,
                               DStorageExt::CompressionQuality quality,
                               vector<std::byte> &result) noexcept {
    Clock clk;
    switch (algorithm) {
        case DStorageCompression::None: {
            LUISA_WARNING_WITH_LOCATION(
                "Compression algorithm is set to None. The data "
                "will be simply copied without compression.");
            result.resize(size_bytes);
            std::memcpy(result.data(), data, size_bytes);
            break;
        }
#ifdef LUISA_COMPUTE_ENABLE_NVCOMP
        case DStorageCompression::GDeflate: {
            _device->with_handle([&] {
                // FIXME: nvCOMP does not support compression quality other than default
                // auto algo = quality == DStorageCompressionQuality::Best ? 1 : 0;
                nvcomp::GdeflateManager manager{nvcompGdeflateCompressionMaxAllowedChunkSize, 0};
                detail::cuda_compress_cpu(
                    manager,
                    static_cast<const std::byte *>(data),
                    size_bytes, quality, result);
            });
            break;
        }
        case DStorageCompression::Cascaded: {
            _device->with_handle([&] {
                nvcomp::CascadedManager manager;
                detail::cuda_compress_cpu(
                    manager,
                    static_cast<const std::byte *>(data),
                    size_bytes, quality, result);
            });
            break;
        }
        case DStorageCompression::LZ4: {
            _device->with_handle([&] {
                nvcomp::LZ4Manager manager{64_k, NVCOMP_TYPE_CHAR};
                detail::cuda_compress_cpu(
                    manager,
                    static_cast<const std::byte *>(data),
                    size_bytes, quality, result);
            });
            break;
        }
        case DStorageCompression::Snappy: {
            _device->with_handle([&] {
                nvcomp::SnappyManager manager{64_k};
                detail::cuda_compress_cpu(
                    manager,
                    static_cast<const std::byte *>(data),
                    size_bytes, quality, result);
            });
            break;
        }
        case DStorageCompression::Bitcomp: {
            _device->with_handle([&] {
                nvcomp::BitcompManager manager{NVCOMP_TYPE_CHAR};
                detail::cuda_compress_cpu(
                    manager,
                    static_cast<const std::byte *>(data),
                    size_bytes, quality, result);
            });
            break;
        }
        case DStorageCompression::ANS: {
            _device->with_handle([&] {
                nvcomp::ANSManager manager{64_k};
                detail::cuda_compress_cpu(
                    manager,
                    static_cast<const std::byte *>(data),
                    size_bytes, quality, result);
            });
            break;
        }
#endif
        default:
            LUISA_ERROR_WITH_LOCATION(
                "Unsupported compression algorithm: {}",
                to_string(algorithm));
    }

    auto ratio = static_cast<double>(result.size()) / static_cast<double>(size_bytes);
    LUISA_VERBOSE("Compressed {}B to {}B (ratio = {}) with {} in {} ms.",
                  size_bytes, result.size(), ratio, to_string(algorithm), clk.toc());
}

ResourceCreationInfo CUDADStorageExt::create_stream_handle(const DStorageStreamOption &option) noexcept {
    auto p = _device->with_handle([this] {
        return luisa::new_with_allocator<CUDACompressionStream>(_device);
    });
    ResourceCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(p);
    info.native_handle = p->handle();
    return info;
}

DStorageExt::FileCreationInfo CUDADStorageExt::open_file_handle(luisa::string_view path) noexcept {
    auto file = _device->with_handle([=] {
        return luisa::new_with_allocator<CUDAMappedFile>(path);
    });
    if (file->mapped_pointer() == nullptr) {
        luisa::delete_with_allocator(file);
        return DStorageExt::FileCreationInfo::make_invalid();
    }
    DStorageExt::FileCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(file);
    info.native_handle = file->mapped_pointer();
    info.size_bytes = file->size_bytes();
    return info;
}

void CUDADStorageExt::close_file_handle(uint64_t handle) noexcept {
    _device->with_handle([handle] {
        luisa::delete_with_allocator<CUDAMappedFile>(
            reinterpret_cast<CUDAMappedFile *>(handle));
    });
}

DStorageExt::PinnedMemoryInfo CUDADStorageExt::pin_host_memory(void *ptr, size_t size_bytes) noexcept {
    auto p = _device->with_handle([=] {
        return luisa::new_with_allocator<CUDAPinnedMemory>(ptr, size_bytes);
    });
    DStorageExt::PinnedMemoryInfo info{};
    info.handle = reinterpret_cast<uint64_t>(p);
    info.native_handle = p->host_pointer();
    info.size_bytes = size_bytes;
    return info;
}

void CUDADStorageExt::unpin_host_memory(uint64_t handle) noexcept {
    _device->with_handle([=] {
        luisa::delete_with_allocator<CUDAPinnedMemory>(
            reinterpret_cast<CUDAPinnedMemory *>(handle));
    });
}

}// namespace luisa::compute::cuda
