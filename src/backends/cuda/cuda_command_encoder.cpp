#include <luisa/core/magic_enum.h>
#include <luisa/runtime/command_list.h>
#include <luisa/backends/ext/cuda/lcub/cuda_lcub_command.h>

#include "cuda_error.h"
#include "cuda_buffer.h"
#include "cuda_mesh.h"
#include "cuda_curve.h"
#include "cuda_procedural_primitive.h"
#include "cuda_motion_instance.h"
#include "cuda_accel.h"
#include "cuda_stream.h"
#include "cuda_device.h"
#include "cuda_shader.h"
#include "cuda_host_buffer_pool.h"
#include "cuda_texture.h"
#include "cuda_bindless_array.h"
#include "cuda_command_encoder.h"

#include "extensions/cuda_dstorage.h"

namespace luisa::compute::cuda {

class UserCallbackContext : public CUDACallbackContext {

public:
    using CallbackContainer = CommandList::CallbackContainer;

private:
    CallbackContainer _functions;

private:
    [[nodiscard]] static auto &_object_pool() noexcept {
        static Pool<UserCallbackContext, true> pool;
        return pool;
    }

public:
    explicit UserCallbackContext(CallbackContainer &&cbs) noexcept
        : _functions{std::move(cbs)} {}

    [[nodiscard]] static auto create(CallbackContainer &&cbs) noexcept {
        return _object_pool().create(std::move(cbs));
    }

    void recycle() noexcept override {
        for (auto &&f : _functions) { f(); }
        _object_pool().destroy(this);
    }
};

void CUDACommandEncoder::commit(CommandList::CallbackContainer &&user_callbacks) noexcept {
    if (!user_callbacks.empty()) {
        _callbacks.emplace_back(
            UserCallbackContext::create(
                std::move(user_callbacks)));
    }
    if (auto callbacks = std::move(_callbacks); !callbacks.empty()) {
        _stream->callback(std::move(callbacks));
    }
}

void CUDACommandEncoder::visit(BufferUploadCommand *command) noexcept {
    auto buffer = reinterpret_cast<const CUDABuffer *>(command->handle());
    auto address = buffer->device_address() + command->offset();
    auto data = command->data();
    auto size = command->size();
    with_upload_buffer(size, [&](auto upload_buffer) noexcept {
        std::memcpy(upload_buffer->address(), data, size);
        LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(
            address, upload_buffer->address(),
            size, _stream->handle()));
    });
}

void CUDACommandEncoder::visit(BufferDownloadCommand *command) noexcept {
    auto buffer = reinterpret_cast<const CUDABuffer *>(command->handle());
    auto address = buffer->device_address() + command->offset();
    auto data = command->data();
    auto size = command->size();
    with_download_pool_no_fallback(size, [&](auto download_buffer) noexcept {
        if (download_buffer) {
            LUISA_CHECK_CUDA(cuMemcpyDtoHAsync(
                download_buffer->address(), address,
                size, _stream->handle()));
            LUISA_CHECK_CUDA(cuMemcpyAsync(
                reinterpret_cast<CUdeviceptr>(data),
                reinterpret_cast<CUdeviceptr>(download_buffer->address()),
                size, _stream->handle()));
        } else {
            LUISA_CHECK_CUDA(cuMemcpyDtoHAsync(
                data, address, size, _stream->handle()));
        }
    });
}

void CUDACommandEncoder::visit(BufferCopyCommand *command) noexcept {
    auto src_buffer = reinterpret_cast<const CUDABuffer *>(command->src_handle())->device_address() +
                      command->src_offset();
    auto dst_buffer = reinterpret_cast<const CUDABuffer *>(command->dst_handle())->device_address() +
                      command->dst_offset();
    auto size = command->size();
    LUISA_CHECK_CUDA(cuMemcpyDtoDAsync(dst_buffer, src_buffer, size, _stream->handle()));
}

void CUDACommandEncoder::visit(ShaderDispatchCommand *command) noexcept {
    reinterpret_cast<CUDAShader *>(command->handle())->launch(*this, command);
}

namespace detail {

static void memcpy_buffer_to_texture(CUdeviceptr buffer, size_t buffer_offset, size_t buffer_total_size,
                                     CUarray array, PixelStorage array_storage, uint3 array_size,
                                     CUstream stream) noexcept {
    CUDA_MEMCPY3D copy{};
    auto pitch = pixel_storage_size(array_storage, make_uint3(array_size.x, 1u, 1u));
    auto height = pixel_storage_size(array_storage, make_uint3(array_size.xy(), 1u)) / pitch;
    auto full_size = pixel_storage_size(array_storage, array_size);
    LUISA_ASSERT(buffer_offset < buffer_total_size &&
                     buffer_total_size - buffer_offset >= full_size,
                 "Buffer size too small for texture copy.");
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = buffer + buffer_offset;
    copy.srcPitch = pitch;
    copy.srcHeight = height;
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = array;
    copy.WidthInBytes = pitch;
    copy.Height = height;
    copy.Depth = array_size.z;
    LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, stream));
}

}// namespace detail

void CUDACommandEncoder::visit(BufferToTextureCopyCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDATexture *>(command->texture());
    auto array = mipmap_array->level(command->level());
    auto buffer = reinterpret_cast<const CUDABuffer *>(command->buffer());
    detail::memcpy_buffer_to_texture(
        buffer->device_address(), command->buffer_offset(), buffer->size_bytes(),
        array, command->storage(), command->size(), _stream->handle());
}

void CUDACommandEncoder::visit(TextureUploadCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDATexture *>(command->handle());
    auto array = mipmap_array->level(command->level());
    CUDA_MEMCPY3D copy{};
    auto pitch = pixel_storage_size(command->storage(), make_uint3(command->size().x, 1u, 1u));
    auto height = pixel_storage_size(command->storage(), make_uint3(command->size().xy(), 1u)) / pitch;
    auto size_bytes = pixel_storage_size(command->storage(), command->size());
    auto data = command->data();
    with_upload_buffer(size_bytes, [&](auto upload_buffer) noexcept {
        std::memcpy(upload_buffer->address(), data, size_bytes);
        copy.srcMemoryType = CU_MEMORYTYPE_HOST;
        copy.srcHost = upload_buffer->address();
        copy.srcPitch = pitch;
        copy.srcHeight = height;
        copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copy.dstArray = array;
        copy.WidthInBytes = pitch;
        copy.Height = height;
        copy.Depth = command->size().z;
        LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
    });
}

void CUDACommandEncoder::visit(TextureDownloadCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDATexture *>(command->handle());
    auto array = mipmap_array->level(command->level());
    CUDA_MEMCPY3D copy{};
    auto pitch = pixel_storage_size(command->storage(), make_uint3(command->size().x, 1u, 1u));
    auto height = pixel_storage_size(command->storage(), make_uint3(command->size().xy(), 1u)) / pitch;
    auto size_bytes = pixel_storage_size(command->storage(), command->size());
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.WidthInBytes = pitch;
    copy.Height = height;
    copy.Depth = command->size().z;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstPitch = pitch;
    copy.dstHeight = height;
    with_download_pool_no_fallback(size_bytes, [&](auto download_buffer) noexcept {
        if (download_buffer) {
            copy.dstHost = download_buffer->address();
            LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
            LUISA_CHECK_CUDA(cuMemcpyAsync(
                reinterpret_cast<CUdeviceptr>(command->data()),
                reinterpret_cast<CUdeviceptr>(download_buffer->address()),
                size_bytes, _stream->handle()));
        } else {
            copy.dstHost = command->data();
            LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
        }
    });
}

void CUDACommandEncoder::visit(TextureCopyCommand *command) noexcept {
    auto src_mipmap_array = reinterpret_cast<CUDATexture *>(command->src_handle());
    auto dst_mipmap_array = reinterpret_cast<CUDATexture *>(command->dst_handle());
    auto src_array = src_mipmap_array->level(command->src_level());
    auto dst_array = dst_mipmap_array->level(command->dst_level());
    auto pitch = pixel_storage_size(command->storage(), make_uint3(command->size().x, 1u, 1u));
    auto height = pixel_storage_size(command->storage(), make_uint3(command->size().xy(), 1u)) / pitch;
    CUDA_MEMCPY3D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = src_array;
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = dst_array;
    copy.WidthInBytes = pitch;
    copy.Height = height;
    copy.Depth = command->size().z;
    LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
}

void CUDACommandEncoder::visit(TextureToBufferCopyCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDATexture *>(command->texture());
    auto array = mipmap_array->level(command->level());
    CUDA_MEMCPY3D copy{};
    auto pitch = pixel_storage_size(command->storage(), make_uint3(command->size().x, 1u, 1u));
    auto height = pixel_storage_size(command->storage(), make_uint3(command->size().xy(), 1u)) / pitch;
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = reinterpret_cast<const CUDABuffer *>(command->buffer())->device_address() +
                     command->buffer_offset();
    copy.dstPitch = pitch;
    copy.dstHeight = height;
    copy.WidthInBytes = pitch;
    copy.Height = height;
    copy.Depth = command->size().z;
    LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
}

void CUDACommandEncoder::visit(AccelBuildCommand *command) noexcept {
    auto accel = reinterpret_cast<CUDAAccel *>(command->handle());
    accel->build(*this, command);
}

void CUDACommandEncoder::visit(MeshBuildCommand *command) noexcept {
    auto mesh = reinterpret_cast<CUDAMesh *>(command->handle());
    mesh->build(*this, command);
}

void CUDACommandEncoder::visit(CurveBuildCommand *command) noexcept {
    auto curve = reinterpret_cast<CUDACurve *>(command->handle());
    curve->build(*this, command);
}

void CUDACommandEncoder::visit(ProceduralPrimitiveBuildCommand *command) noexcept {
    auto primitive = reinterpret_cast<CUDAProceduralPrimitive *>(command->handle());
    primitive->build(*this, command);
}

void CUDACommandEncoder::visit(BindlessArrayUpdateCommand *command) noexcept {
    auto bindless_array = reinterpret_cast<CUDABindlessArray *>(command->handle());
    bindless_array->update(*this, command);
}

void CUDACommandEncoder::visit(CustomCommand *command) noexcept {
    switch (command->uuid()) {
        case to_underlying(CustomCommandUUID::DSTORAGE_READ): {
            auto ds_command = static_cast<DStorageReadCommand *>(command);
            visit(ds_command);
            break;
        }
        case to_underlying(CustomCommandUUID::CUDA_LCUB_COMMAND): {
            auto lcub_command = static_cast<CudaLCubCommand *>(command);
            LUISA_ASSERT(lcub_command != nullptr, "Invalid CudaLCuBCommand.");
            lcub_command->func(_stream->handle());
            break;
        }
        default:
            LUISA_ERROR_WITH_LOCATION("Custom command (UUID = 0x{:04x}) "
                                      "is not supported on CUDA.",
                                      command->uuid());
    }
}

void CUDACommandEncoder::visit(MotionInstanceBuildCommand *command) noexcept {
    auto motion_instance = reinterpret_cast<CUDAMotionInstance *>(command->handle());
    motion_instance->build(*this, command);
}

namespace detail {

using DSBufferRequest = DStorageReadCommand::BufferRequest;
using DSTextureRequest = DStorageReadCommand::TextureRequest;
using DSMemoryRequest = DStorageReadCommand::MemoryRequest;

static void dstorage_copy(const void *input_host_ptr,
                          CUdeviceptr input_device_ptr,
                          size_t input_size,
                          DStorageReadCommand::Request output_request,
                          CUstream stream) noexcept {

    if (luisa::holds_alternative<DSBufferRequest>(output_request)) {
        auto dst = luisa::get<DSBufferRequest>(output_request);
        auto buffer = reinterpret_cast<const CUDABuffer *>(dst.handle);
        LUISA_ASSERT(dst.offset_bytes < buffer->size_bytes() &&
                         input_size <= buffer->size_bytes() - dst.offset_bytes,
                     "DStorageReadCommand out of range.");
        auto dst_addr = buffer->device_address() + dst.offset_bytes;
        if (dst.size_bytes != input_size) {
            LUISA_WARNING_WITH_LOCATION(
                "DStorageReadCommand size mismatch: "
                "input size = {}, output size = {}.",
                input_size, dst.size_bytes);
        }
        auto valid_size = std::min(dst.size_bytes, input_size);
        LUISA_CHECK_CUDA(cuMemcpyDtoDAsync(dst_addr, input_device_ptr, valid_size, stream));
    } else if (luisa::holds_alternative<DSTextureRequest>(output_request)) {
        auto dst = luisa::get<DSTextureRequest>(output_request);
        auto texture = reinterpret_cast<const CUDATexture *>(dst.handle);
        auto size = make_uint3(dst.size[0], dst.size[1], dst.size[2]);
        LUISA_ASSERT(all(size == max(texture->size() >> dst.level, 1u)),
                     "DStorageReadCommand size mismatch.");
        auto array = texture->level(dst.level);
        detail::memcpy_buffer_to_texture(
            input_device_ptr, 0u, input_size,
            array, texture->storage(), size, stream);
    } else if (luisa::holds_alternative<DSMemoryRequest>(output_request)) {
        auto dst = luisa::get<DSMemoryRequest>(output_request);
        auto p = static_cast<std::byte *>(dst.data);
        if (dst.size_bytes != input_size) {
            LUISA_WARNING_WITH_LOCATION(
                "DStorageReadCommand size mismatch: "
                "input size = {}, output size = {}.",
                input_size, dst.size_bytes);
        }
        auto valid_size = std::min(dst.size_bytes, input_size);
        LUISA_CHECK_CUDA(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(p),
                                       reinterpret_cast<CUdeviceptr>(input_host_ptr),
                                       valid_size, stream));
    } else {
        LUISA_ERROR_WITH_LOCATION("Unreachable.");
    }
}

#ifdef LUISA_COMPUTE_ENABLE_NVCOMP

static void dstorage_decompress(DStorageCompression algorithm,
                                CUdeviceptr input_device_ptr, size_t input_size,
                                DStorageReadCommand::Request output_request,
                                CUDACommandEncoder &encoder) noexcept {

    auto stream = encoder.stream()->handle();
    auto decompress_to_buffer = [&](CUdeviceptr in_ptr, size_t in_size,
                                    CUdeviceptr out_ptr, size_t out_size) noexcept {
        // auto comp_stream = dynamic_cast<CUDACompressionStream *>(encoder.stream());
        // LUISA_ASSERT(comp_stream != nullptr,
        //              "DStorageReadCommand must be used "
        //              "with a compression stream.");
        auto comp_stream = static_cast<CUDACompressionStream *>(encoder.stream());
        auto manager = comp_stream->compressor(algorithm);
        LUISA_ASSERT(manager != nullptr,
                     "Failed to get the compression manager for {}.",
                     luisa::to_string(algorithm));
        auto config = manager->configure_decompression(reinterpret_cast<const uint8_t *>(in_ptr));
        manager->decompress(reinterpret_cast<uint8_t *>(out_ptr),
                            reinterpret_cast<const uint8_t *>(in_ptr), config);
    };

    if (luisa::holds_alternative<DSBufferRequest>(output_request)) {
        auto dst = luisa::get<DSBufferRequest>(output_request);
        auto buffer = reinterpret_cast<const CUDABuffer *>(dst.handle);
        LUISA_ASSERT(dst.offset_bytes < buffer->size_bytes() &&
                         input_size <= buffer->size_bytes() - dst.offset_bytes,
                     "DStorageReadCommand out of range.");
        auto dst_addr = buffer->device_address() + dst.offset_bytes;
        decompress_to_buffer(input_device_ptr, input_size, dst_addr, dst.size_bytes);
    } else if (luisa::holds_alternative<DSTextureRequest>(output_request)) {
        auto dst = luisa::get<DSTextureRequest>(output_request);
        auto texture = reinterpret_cast<const CUDATexture *>(dst.handle);
        auto size = make_uint3(dst.size[0], dst.size[1], dst.size[2]);
        LUISA_ASSERT(all(size == max(texture->size() >> dst.level, 1u)),
                     "DStorageReadCommand size mismatch.");
        auto array = texture->level(dst.level);
        auto storage = texture->storage();
        auto temp_buffer_size = pixel_storage_size(storage, size);
        auto temp_buffer = static_cast<CUdeviceptr>(0ull);
        LUISA_CHECK_CUDA(cuMemAllocAsync(&temp_buffer, temp_buffer_size, stream));
        decompress_to_buffer(input_device_ptr, input_size, temp_buffer, temp_buffer_size);
        detail::memcpy_buffer_to_texture(
            temp_buffer, 0u, temp_buffer_size,
            array, storage, size, stream);
        LUISA_CHECK_CUDA(cuMemFreeAsync(temp_buffer, stream));
    } else if (luisa::holds_alternative<DSMemoryRequest>(output_request)) {
        auto dst = luisa::get<DSMemoryRequest>(output_request);
        auto output_buffer = static_cast<CUdeviceptr>(0u);
        LUISA_CHECK_CUDA(cuMemAllocAsync(&output_buffer, dst.size_bytes, stream));
        decompress_to_buffer(input_device_ptr, input_size, output_buffer, dst.size_bytes);
        LUISA_CHECK_CUDA(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(dst.data),
                                       output_buffer, dst.size_bytes, stream));
        LUISA_CHECK_CUDA(cuMemFreeAsync(output_buffer, stream));
    } else {
        LUISA_ERROR_WITH_LOCATION("Unreachable.");
    }
}

#endif

}// namespace detail

void CUDACommandEncoder::visit(DStorageReadCommand *command) noexcept {
    auto [device_ptr, host_ptr, size] = luisa::visit(
        [](auto src) noexcept {
            using T = std::remove_cvref_t<decltype(src)>;
            auto request_offset_bytes = src.offset_bytes;
            auto request_size_bytes = src.size_bytes;
            if (std::is_same_v<T, DStorageReadCommand::FileSource>) {
                auto source = reinterpret_cast<CUDAMappedFile *>(src.handle);
                LUISA_ASSERT(request_offset_bytes < source->size_bytes() &&
                                 request_size_bytes <= source->size_bytes() - request_offset_bytes,
                             "DStorageReadCommand out of range.");
                return std::make_tuple(source->device_address() + request_offset_bytes,
                                       static_cast<const std::byte *>(source->mapped_pointer()) + request_offset_bytes,
                                       request_size_bytes);
            } else if (std::is_same_v<T, DStorageReadCommand::MemorySource>) {
                auto source = reinterpret_cast<CUDAPinnedMemory *>(src.handle);
                LUISA_ASSERT(request_offset_bytes < source->size_bytes() &&
                                 request_size_bytes <= source->size_bytes() - request_offset_bytes,
                             "DStorageReadCommand out of range.");
                return std::make_tuple(source->device_address() + request_offset_bytes,
                                       static_cast<const std::byte *>(source->host_pointer()) + request_offset_bytes,
                                       request_size_bytes);
            } else {
                LUISA_ERROR_WITH_LOCATION("Unreachable.");
            }
        },
        command->source());

    // copy or decompress
    switch (auto compression = command->compression()) {
        case DStorageCompression::None:
            detail::dstorage_copy(
                host_ptr, device_ptr, size,
                command->request(), _stream->handle());
            break;
#ifdef LUISA_COMPUTE_ENABLE_NVCOMP
        default:
            detail::dstorage_decompress(
                compression, device_ptr, size,
                command->request(), *this);
            break;
#else
        default: LUISA_ERROR_WITH_LOCATION(
            "Unsupported DStorage compression method {}.",
            to_string(compression));
#endif
    }
}

}// namespace luisa::compute::cuda
