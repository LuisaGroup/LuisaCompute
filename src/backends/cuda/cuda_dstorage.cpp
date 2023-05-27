//
// Created by Mike on 5/26/2023.
//

#include <cuda.h>

#include <core/clock.h>
#include <core/magic_enum.h>
#include <backends/cuda/cuda_dstorage.h>

#ifdef LUISA_PLATFORM_WINDOWS
#include <Windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef LUISA_COMPUTE_ENABLE_NVCOMP
#include <gdeflate/gdeflate_cpu.h>
#include <nvcomp/gdeflate.h>

namespace luisa::compute::cuda::detail {

[[nodiscard]] inline auto to_string(nvcompStatus_t status) noexcept {
    using namespace std::string_view_literals;
    switch (status) {
        case nvcompSuccess: return "Success"sv;
        case nvcompErrorInvalidValue: return "ErrorInvalidValue"sv;
        case nvcompErrorNotSupported: return "ErrorNotSupported"sv;
        case nvcompErrorCannotDecompress: return "ErrorCannotDecompress"sv;
        case nvcompErrorBadChecksum: return "ErrorBadChecksum"sv;
        case nvcompErrorCannotVerifyChecksums: return "ErrorCannotVerifyChecksums"sv;
        case nvcompErrorOutputBufferTooSmall: return "ErrorOutputBufferTooSmall"sv;
        case nvcompErrorWrongHeaderLength: return "ErrorWrongHeaderLength"sv;
        case nvcompErrorAlignment: return "ErrorAlignment"sv;
        case nvcompErrorChunkSizeTooLarge: return "ErrorChunkSizeTooLarge"sv;
        case nvcompErrorCudaError: return "CudaError"sv;
        case nvcompErrorInternal: return "ErrorInternal"sv;
        default: break;
    }
    return "Unknown"sv;
}

}// namespace luisa::compute::cuda::detail

#define LUISA_CHECK_NVCOMP(...)                                 \
    do {                                                        \
        if (auto ec = __VA_ARGS__; ec != nvcompSuccess) {       \
            LUISA_ERROR_WITH_LOCATION(                          \
                "nvCOMP error: {}",                             \
                ::luisa::compute::cuda::detail::to_string(ec)); \
        }                                                       \
    } while (false)

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
    LUISA_INFO("Registered host memory at 0x{:016x} with {} "
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
    LUISA_INFO("Mapped file '{}' to host address "
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
static void compress_gdeflate_cpu(const std::byte *data, size_t size,
                                  DStorageCompressionQuality quality,
                                  vector<std::byte> &result) noexcept {

    Clock clk;
    auto chunk_size = nvcompGdeflateCompressionMaxAllowedChunkSize;
    nvcompBatchedGdeflateOpts_t options{};
    switch (quality) {
        case DStorageCompressionQuality::Fastest: options.algo = 0; break;
        case DStorageCompressionQuality::Default: options.algo = 0; break;
        case DStorageCompressionQuality::Best: options.algo = 1; break;
    }
    size_t max_output_chunk_size = 0u;
    LUISA_CHECK_NVCOMP(nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(
        chunk_size, options, &max_output_chunk_size));
    auto max_chunk_count = (size + chunk_size - 1u) / chunk_size;
    result.reserve(std::min(static_cast<size_t>(.1 * static_cast<double>(size)),
                            max_output_chunk_size * max_chunk_count));
    auto accum_output_size = 0u;
    LUISA_INFO("Max output bytes per chunk: {}.", max_output_chunk_size);
    for (size_t input_offset = 0u; input_offset < size; input_offset += chunk_size) {
        auto input_size = std::min(size - input_offset, chunk_size);
        auto input_ptr = static_cast<const void *>(data + input_offset);
        result.resize(accum_output_size + max_output_chunk_size);
        auto output_ptr = static_cast<void *>(result.data() + accum_output_size);
        auto chunk_output_size = static_cast<size_t>(0u);
        gdeflate::compressCPU(&input_ptr, &input_size, chunk_size,
                              1u, &output_ptr, &chunk_output_size);
        accum_output_size += chunk_output_size;
    }
    result.resize(accum_output_size);

    auto ratio = static_cast<double>(accum_output_size) / static_cast<double>(size);
    auto wasted_bytes = result.capacity() - accum_output_size;
    LUISA_INFO("Compressed {} bytes to {} bytes (ratio = {}, "
               "{} bytes wasted in allocation) in {} ms.",
               size, accum_output_size, ratio, wasted_bytes, clk.toc());
}
#endif

}// namespace detail

void CUDADStorageExt::compress(const void *data, size_t size_bytes,
                               DStorageExt::Compression algorithm,
                               DStorageExt::CompressionQuality quality,
                               vector<std::byte> &result) noexcept {
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
            detail::compress_gdeflate_cpu(
                static_cast<const std::byte *>(data),
                size_bytes, quality, result);
            break;
        }
#endif
        default:
            LUISA_ERROR_WITH_LOCATION(
                "Unsupported compression algorithm: {}",
                to_string(algorithm));
    }
}

ResourceCreationInfo CUDADStorageExt::create_stream_handle(const DStorageStreamOption &option) noexcept {
    return _device->create_stream(StreamTag::CUSTOM);
}

DStorageExt::FileCreationInfo CUDADStorageExt::open_file_handle(luisa::string_view path) noexcept {
    auto file = _device->with_handle([=] {
        return luisa::new_with_allocator<CUDAMappedFile>(path);
    });
    if (file->mapped_pointer() == nullptr) {
        luisa::delete_with_allocator<CUDAMappedFile>(file);
        return DStorageExt::FileCreationInfo::make_invalid();
    }
    DStorageExt::FileCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(file);
    info.native_handle = file;
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
    info.native_handle = p;
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
