#include <nvtt/nvtt_lowlevel.h>

#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/buffer.h>

#include "../cuda_device.h"
#include "../cuda_stream.h"
#include "../cuda_buffer.h"
#include "../cuda_texture.h"
#include "../cuda_event.h"
#include "cuda_texture_compression.h"

namespace luisa::compute::cuda {

namespace detail {

inline void copy_image_to_temp_memory(CUdeviceptr temp,
                                      CUDATexture *texture,
                                      uint level, uint2 size) noexcept {
    auto array = texture->level(level);
    CUDA_MEMCPY3D copy{};
    auto total_size = pixel_storage_size(texture->storage(), make_uint3(size, 1u));
    auto pitch = pixel_storage_size(texture->storage(), make_uint3(size.x, 1u, 1u));
    auto height = total_size / pitch;
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = temp;
    copy.dstPitch = pitch;
    copy.dstHeight = height;
    copy.WidthInBytes = pitch;
    copy.Height = height;
    copy.Depth = 1u;
    LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, nullptr));
}

inline void compress_image(CUdeviceptr src,
                           CUdeviceptr dst,
                           PixelStorage storage,
                           uint2 size,
                           CUstream stream,
                           float alpha_importance,
                           PixelFormat target) noexcept {
    nvtt::RefImage image{
        .data = reinterpret_cast<void *>(src),
        .width = static_cast<int>(size.x),
        .height = static_cast<int>(size.y),
        .depth = 1,
        .num_channels = static_cast<int>(pixel_storage_channel_count(storage)),
    };
    auto value_type = [storage] {
        switch (storage) {
            case PixelStorage::BYTE1:
            case PixelStorage::BYTE2:
            case PixelStorage::BYTE4:
                return nvtt::UINT8;
            case PixelStorage::HALF1:
            case PixelStorage::HALF2:
            case PixelStorage::HALF4:
                return nvtt::FLOAT16;
            case PixelStorage::FLOAT1:
            case PixelStorage::FLOAT2:
            case PixelStorage::FLOAT4:
                return nvtt::FLOAT32;
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Unsupported pixel storage.");
    }();
    auto format = [target] {
        switch (target) {
            case PixelFormat::BC6HUF16: return nvtt::Format_BC6U;
            case PixelFormat::BC7UNorm: return nvtt::Format_BC7;
            default: break;
        }
        LUISA_ERROR_WITH_LOCATION("Unsupported pixel format.");
    }();
    nvtt::GPUInputBuffer buffer{&image, value_type};
    nvtt::EncodeSettings settings{
        .format = format,
        .quality = nvtt::Quality_Normal,
        .encode_flags = nvtt::EncodeFlags_UseGPU | nvtt::EncodeFlags_OutputToGPUMem,
    };
    nvtt::nvtt_encode(buffer, reinterpret_cast<void *>(dst), settings);
}

}// namespace detail

void CUDATexCompressExt::_compress(Stream &stream,
                                   const ImageView<float> &src,
                                   const BufferView<uint> &result,
                                   float alpha_importance,
                                   PixelFormat target_format) noexcept {
    std::scoped_lock lock{_mutex};
    _device->with_handle([&] {
        nvtt::useCurrentDevice();
        auto cuda_stream = reinterpret_cast<CUDAStream *>(stream.handle());
        auto cuda_texture = reinterpret_cast<CUDATexture *>(src.handle());
        auto cuda_buffer = reinterpret_cast<CUDABuffer *>(result.handle());
        LUISA_CHECK_CUDA(cuEventRecord(_event, cuda_stream->handle()));
        CUdeviceptr temp{};
        LUISA_CHECK_CUDA(cuMemAllocAsync(&temp, src.size_bytes(), nullptr));
        LUISA_CHECK_CUDA(cuStreamWaitEvent(nullptr, _event, 0u));
        detail::copy_image_to_temp_memory(temp, cuda_texture, src.level(), src.size());
        detail::compress_image(temp, cuda_buffer->device_address() + result.offset(),
                               cuda_texture->storage(), src.size(),
                               cuda_stream->handle(),
                               alpha_importance, target_format);
        LUISA_CHECK_CUDA(cuEventRecord(_event, nullptr));
        LUISA_CHECK_CUDA(cuStreamWaitEvent(cuda_stream->handle(), _event, 0u));
        LUISA_CHECK_CUDA(cuMemFreeAsync(temp, cuda_stream->handle()));
    });
}

TexCompressExt::Result CUDATexCompressExt::compress_bc6h(Stream &stream,
                                                         const ImageView<float> &src,
                                                         const BufferView<uint> &result) noexcept {
    _compress(stream, src, result, 0.f, PixelFormat::BC6HUF16);
    return Result::Success;
}

TexCompressExt::Result CUDATexCompressExt::compress_bc7(Stream &stream,
                                                        const ImageView<float> &src,
                                                        const BufferView<uint> &result,
                                                        float alpha_importance) noexcept {
    _compress(stream, src, result, alpha_importance, PixelFormat::BC7UNorm);
    return Result::Success;
}

TexCompressExt::Result CUDATexCompressExt::check_builtin_shader() noexcept {
    // we do not need to check builtin shader for cuda as they are packed in the binary
    return Result::Success;
}

CUDATexCompressExt::CUDATexCompressExt(CUDADevice *device) noexcept
    : _device{device}, _event{nullptr} {
    _device->with_handle([&] {
        LUISA_CHECK_CUDA(cuEventCreate(&_event, CU_EVENT_DISABLE_TIMING));
    });
}

CUDATexCompressExt::~CUDATexCompressExt() noexcept {
    _device->with_handle([&] {
        LUISA_CHECK_CUDA(cuEventDestroy(_event));
    });
}

}// namespace luisa::compute::cuda
