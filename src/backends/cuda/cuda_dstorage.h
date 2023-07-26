#pragma once

#include <luisa/backends/ext/dstorage_ext_interface.h>
#include "cuda_device.h"

#ifdef LUISA_COMPUTE_ENABLE_NVCOMP

#include <nvcomp/gdeflate.hpp>
#include <nvcomp/lz4.hpp>
#include <nvcomp/snappy.hpp>
#include <nvcomp/cascaded.hpp>
#include <nvcomp/bitcomp.hpp>
#include <nvcomp/ans.hpp>

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

#ifdef LUISA_COMPUTE_ENABLE_NVCOMP

class CUDACompressionStream : public CUDAStream {

private:
    spin_mutex _mutex;
    luisa::unique_ptr<nvcomp::GdeflateManager> _gdeflate;
    luisa::unique_ptr<nvcomp::CascadedManager> _cascaded;
    luisa::unique_ptr<nvcomp::LZ4Manager> _lz4;
    luisa::unique_ptr<nvcomp::SnappyManager> _snappy;
    luisa::unique_ptr<nvcomp::BitcompManager> _bitcomp;
    luisa::unique_ptr<nvcomp::ANSManager> _ans;

public:
    explicit CUDACompressionStream(CUDADevice *device) noexcept
        : CUDAStream{device} {}
    [[nodiscard]] auto gdeflate() noexcept {
        std::scoped_lock lock{_mutex};
        if (!_gdeflate) {
            _gdeflate = luisa::make_unique<nvcomp::GdeflateManager>(
                nvcompGdeflateCompressionMaxAllowedChunkSize, 0,
                handle(), static_cast<int>(device()->handle().index()));
        }
        return _gdeflate.get();
    }
    [[nodiscard]] auto cascaded() noexcept {
        std::scoped_lock lock{_mutex};
        if (!_cascaded) {
            _cascaded = luisa::make_unique<nvcomp::CascadedManager>(
                nvcompBatchedCascadedDefaultOpts, handle(),
                static_cast<int>(device()->handle().index()));
        }
        return _cascaded.get();
    }
    [[nodiscard]] auto lz4() noexcept {
        std::scoped_lock lock{_mutex};
        if (!_lz4) {
            _lz4 = luisa::make_unique<nvcomp::LZ4Manager>(
                64_k, NVCOMP_TYPE_CHAR, handle(),
                static_cast<int>(device()->handle().index()));
        }
        return _lz4.get();
    }
    [[nodiscard]] auto snappy() noexcept {
        std::scoped_lock lock{_mutex};
        if (!_snappy) {
            _snappy = luisa::make_unique<nvcomp::SnappyManager>(
                64_k, handle(), static_cast<int>(device()->handle().index()));
        }
        return _snappy.get();
    }
    [[nodiscard]] auto bitcomp() noexcept {
        std::scoped_lock lock{_mutex};
        if (!_bitcomp) {
            _bitcomp = luisa::make_unique<nvcomp::BitcompManager>(
                NVCOMP_TYPE_CHAR, 0, handle(),
                static_cast<int>(device()->handle().index()));
        }
        return _bitcomp.get();
    }
    [[nodiscard]] auto ans() noexcept {
        std::scoped_lock lock{_mutex};
        if (!_ans) {
            _ans = luisa::make_unique<nvcomp::ANSManager>(
                64_k, handle(), static_cast<int>(device()->handle().index()));
        }
        return _ans.get();
    }
    [[nodiscard]] nvcomp::PimplManager *compressor(DStorageCompression algorithm) noexcept {
        switch (algorithm) {
            case DStorageCompression::GDeflate: return gdeflate();
            case DStorageCompression::Cascaded: return cascaded();
            case DStorageCompression::LZ4: return lz4();
            case DStorageCompression::Snappy: return snappy();
            case DStorageCompression::Bitcomp: return bitcomp();
            case DStorageCompression::ANS: return ans();
            default: break;
        }
        return nullptr;
    }
};
#else
using CUDACompressionStream = CUDAStream;
#endif

class CUDAMappedFile {

private:
    void *_file_handle;
    void *_file_mapping;
    void *_mapped_pointer;
    CUdeviceptr _device_address;
    size_t _size_bytes;

public:
    explicit CUDAMappedFile(luisa::string_view path) noexcept;
    ~CUDAMappedFile() noexcept;
    CUDAMappedFile(CUDAMappedFile &&) noexcept = delete;
    CUDAMappedFile(const CUDAMappedFile &) noexcept = delete;
    CUDAMappedFile &operator=(CUDAMappedFile &&) noexcept = delete;
    CUDAMappedFile &operator=(const CUDAMappedFile &) noexcept = delete;
    [[nodiscard]] auto mapped_pointer() const noexcept { return _mapped_pointer; }
    [[nodiscard]] auto device_address() const noexcept { return _device_address; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size_bytes; }
};

class CUDAPinnedMemory {

private:
    void *_host_pointer;
    CUdeviceptr _device_address;
    size_t _size_bytes;

public:
    CUDAPinnedMemory(void *p, size_t size) noexcept;
    ~CUDAPinnedMemory() noexcept;
    CUDAPinnedMemory(CUDAPinnedMemory &&) noexcept = delete;
    CUDAPinnedMemory(const CUDAPinnedMemory &) noexcept = delete;
    CUDAPinnedMemory &operator=(CUDAPinnedMemory &&) noexcept = delete;
    CUDAPinnedMemory &operator=(const CUDAPinnedMemory &) noexcept = delete;
    [[nodiscard]] auto host_pointer() const noexcept { return _host_pointer; }
    [[nodiscard]] auto device_address() const noexcept { return _device_address; }
    [[nodiscard]] auto size_bytes() const noexcept { return _size_bytes; }
};

class CUDADStorageExt final : public DStorageExt {

private:
    CUDADevice *_device;

public:
    CUDADStorageExt(CUDADevice *device) noexcept : _device{device} {}

public:
    void compress(const void *data,
                  size_t size_bytes,
                  Compression algorithm,
                  CompressionQuality quality,
                  vector<std::byte> &result) noexcept override;

protected:
    [[nodiscard]] DeviceInterface *device() const noexcept override { return _device; }
    [[nodiscard]] ResourceCreationInfo create_stream_handle(const DStorageStreamOption &option) noexcept override;
    [[nodiscard]] FileCreationInfo open_file_handle(luisa::string_view path) noexcept override;
    void close_file_handle(uint64_t handle) noexcept override;
    [[nodiscard]] PinnedMemoryInfo pin_host_memory(void *ptr, size_t size_bytes) noexcept override;
    void unpin_host_memory(uint64_t handle) noexcept override;
};

}// namespace luisa::compute::cuda

