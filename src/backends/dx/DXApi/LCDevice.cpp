#include <DXApi/LCDevice.h>
#include <DXRuntime/Device.h>
#include <Resource/DefaultBuffer.h>
#include <Resource/RenderTexture.h>
#include <Resource/DepthBuffer.h>
#include <Resource/BindlessArray.h>
#include <Shader/ComputeShader.h>
#include <Shader/RasterShader.h>
#include <DXApi/LCCmdBuffer.h>
#include <DXApi/LCEvent.h>
#include <luisa/vstl/md5.h>
#include <Shader/ShaderSerializer.h>
#include <Resource/BottomAccel.h>
#include <Resource/TopAccel.h>
#include <DXApi/LCSwapChain.h>
#include <DXApi/dx_hdr_ext.hpp>
#include "ext.h"
#include "../../common/hlsl/hlsl_codegen.h"
#include <luisa/ast/function_builder.h>
#include <Resource/DepthBuffer.h>
#include <luisa/core/clock.h>
#include <luisa/core/stl/filesystem.h>
#include <Resource/ExternalBuffer.h>
#include <luisa/runtime/dispatch_buffer.h>
#include <luisa/runtime/rtx/aabb.h>
#include "../../common/hlsl/binding_to_arg.h"
#include <luisa/runtime/context.h>
#include <DXRuntime/DStorageCommandQueue.h>
#include <DXApi/TypeCheck.h>
#include <Resource/SparseTexture.h>
#include <Resource/SparseBuffer.h>
#include <Resource/SparseHeap.h>
#ifdef LUISA_ENABLE_XIR
#include "../../common/xir_autodiff.h"
#endif

#include <DXApi/dml_ext.h>
#ifdef LUISA_BACKEND_ENABLE_OIDN
#include <DXApi/dx_oidn_denoiser_ext.h>
#endif

namespace lc::dx {
using namespace lc::dx;
static constexpr uint kShaderModel = 65u;
static constexpr uint kHighShaderModel = 66u;
static constexpr uint kTensorShaderModel = 69u;
LCDevice::LCDevice(Context &&ctx, DeviceConfig const *settings)
    : DeviceInterface(std::move(ctx)),
      native_device(Context{_ctx_impl}, settings) {
    // no ext when headless
    bool headless = settings && settings->headless;
    if (!headless) {
#ifdef LUISA_BACKEND_ENABLE_OIDN
        exts.try_emplace(
            DenoiserExt::name,
            [](LCDevice *device) -> DeviceExtension * {
                return new DXOidnDenoiserExt(device);
            },
            [](DeviceExtension *ext) {
                delete static_cast<DXOidnDenoiserExt *>(ext);
            });
#endif
        exts.try_emplace(
#ifdef LUISA_USE_SYSTEM_STL
            luisa::string{TexCompressExt::name},
#else
            TexCompressExt::name,
#endif
            [](LCDevice *device) -> DeviceExtension * {
                return new DxTexCompressExt(&device->native_device);
            },
            [](DeviceExtension *ext) {
                delete static_cast<DxTexCompressExt *>(ext);
            });
        exts.try_emplace(
#ifdef LUISA_USE_SYSTEM_STL
            luisa::string{NativeResourceExt::name},
#else
            NativeResourceExt::name,
#endif
            [](LCDevice *device) -> DeviceExtension * {
                return new DxNativeResourceExt(device, &device->native_device);
            },
            [](DeviceExtension *ext) {
                delete static_cast<DxNativeResourceExt *>(ext);
            });
        exts.try_emplace(
#ifdef LUISA_USE_SYSTEM_STL
            luisa::string{DStorageExt::name},
#else
            DStorageExt::name,
#endif
            [](LCDevice *device) -> DeviceExtension * {
                return new DStorageExtImpl(device->context().runtime_directory(), device);
            },
            [](DeviceExtension *ext) {
                delete static_cast<DStorageExtImpl *>(ext);
            });
        exts.try_emplace(
#ifdef LUISA_USE_SYSTEM_STL
            luisa::string{DirectMLExt::name},
#else
            DirectMLExt::name,
#endif
            [](LCDevice *device) -> DeviceExtension * {
                return new DxDirectMLExt(device);
            },
            [](DeviceExtension *ext) {
                delete static_cast<DxDirectMLExt *>(ext);
            });
        exts.try_emplace(
#ifdef LUISA_USE_SYSTEM_STL
            luisa::string{DXHDRExt::name},
#else
            DXHDRExt::name,
#endif
            [](LCDevice *device) -> DeviceExtension * {
                return new DXHDRExtImpl(device);
            },
            [](DeviceExtension *ext) {
                delete static_cast<DXHDRExtImpl *>(ext);
            });
#ifdef LCDX_ENABLE_CUDA
        exts.try_emplace(
#ifdef LUISA_USE_SYSTEM_STL
            luisa::string{DxCudaInterop::name},
#else
            DxCudaInterop::name,
#endif
            [](LCDevice *device) -> DeviceExtension * {
                return new DxCudaInteropImpl(*device);
            },
            [](DeviceExtension *ext) {
                delete static_cast<DxCudaInteropImpl *>(ext);
            });
#endif
        exts.try_emplace(
#ifdef LUISA_USE_SYSTEM_STL
            luisa::string{PinnedMemoryExt::name},
#else
            PinnedMemoryExt::name,
#endif
            [](LCDevice *device) -> DeviceExtension * {
                return new DxPinnedMemoryExt(device);
            },
            [](DeviceExtension *ext) {
                delete static_cast<DxPinnedMemoryExt *>(ext);
            });
    }

    exts.try_emplace(
#ifdef LUISA_USE_SYSTEM_STL
        luisa::string{RasterExt::name},
#else
        RasterExt::name,
#endif
        [](LCDevice *device) -> DeviceExtension * {
            return new DxRasterExt(device->native_device);
        },
        [](DeviceExtension *ext) {
            delete static_cast<DxRasterExt *>(ext);
        });
}
LCDevice::~LCDevice() = default;
//Hash128 LCDevice::device_hash() const noexcept {
//    vstd::MD5::MD5Data const &md5 = native_device.adapter_id.to_binary();
//    Hash128 r;
//    static_assert(sizeof(Hash128) == sizeof(vstd::MD5::MD5Data));
//    memcpy(&r, &md5, sizeof(Hash128));
//    return r;
//}
void *LCDevice::native_handle() const noexcept {
    return native_device.device.Get();
}
BufferCreationInfo LCDevice::create_buffer(const Type *element,
                                           size_t elem_count,
                                           void *external_memory) noexcept {
    BufferCreationInfo info{};
    Buffer *res{};
    if (element == Type::of<void>()) {
        info.total_size_bytes = elem_count;
        info.element_stride = 1u;
        res = external_memory ?
                  new DefaultBuffer(
                      &native_device,
                      info.total_size_bytes,
                      reinterpret_cast<ID3D12Resource *>(external_memory)) :
                  new DefaultBuffer(
                      &native_device,
                      info.total_size_bytes,
                      native_device.default_allocator.get());
    } else if (element->is_custom()) {
        if (element == Type::of<IndirectKernelDispatch>()) {
            LUISA_ASSERT(external_memory == nullptr,
                         "IndirectKernelDispatch buffer cannot "
                         "be created from external memory.");
            info.element_stride = ComputeShader::kDispatchIndirectStride;
            info.total_size_bytes = 4 + info.element_stride * elem_count;
            res = static_cast<Buffer *>(new DefaultBuffer(&native_device, info.total_size_bytes, native_device.default_allocator.get()));
        } else {
            LUISA_ERROR("Un-known custom type in dx-backend.");
        }
    } else {
        info.total_size_bytes = element->size() * elem_count;
        res = external_memory ?
                  static_cast<Buffer *>(
                      new DefaultBuffer(
                          &native_device,
                          info.total_size_bytes,
                          reinterpret_cast<ID3D12Resource *>(external_memory))) :
                  static_cast<Buffer *>(
                      new DefaultBuffer(
                          &native_device,
                          info.total_size_bytes,
                          native_device.default_allocator.get()));
        info.element_stride = element->size();
    }
    info.handle = resource_to_handle(res);
    info.native_handle = res->GetResource();
    return info;
}
void LCDevice::destroy_buffer(uint64 handle) noexcept {
    delete reinterpret_cast<Buffer *>(handle);
}
ResourceCreationInfo LCDevice::create_texture(
    PixelFormat format,
    uint dimension,
    uint width,
    uint height,
    uint depth,
    uint mipmap_levels,
    void *external_native_handle,
    bool simultaneous_access,
    bool allow_raster_target) noexcept {
    LUISA_ASSERT(external_native_handle == nullptr, "Importing external textures is not supported on DirectX.");
    if (allow_raster_target) {
        if (simultaneous_access) {
            LUISA_INFO("DX do not allow simultaneous access texture as render target, set simultaneous_access = false");
        }
        simultaneous_access = false;
    }
    bool allow_uav = !is_block_compressed(format);
    ResourceCreationInfo info{};
    auto res = new RenderTexture(
        &native_device,
        width,
        height,
        TextureBase::ToGFXFormat(format),
        (TextureDimension)dimension,
        depth,
        mipmap_levels,
        allow_uav,
        simultaneous_access,
        allow_raster_target,
        native_device.default_allocator.get());
    info.handle = resource_to_handle(res);
    info.native_handle = res->GetResource();
    return info;
}
//string LCDevice::cache_name(string_view file_name) const noexcept {
//    return Shader::PSOName(&native_device, file_name);
//}
void LCDevice::destroy_texture(uint64 handle) noexcept {
    delete reinterpret_cast<TextureBase *>(handle);
}
ResourceCreationInfo LCDevice::create_bindless_array(size_t size, BindlessSlotType type) noexcept {
    ResourceCreationInfo info{};
    auto res = new BindlessArray(&native_device, size, type);
    info.handle = resource_to_handle(res);
    info.native_handle = res->GetResource();
    return info;
}
void LCDevice::destroy_bindless_array(uint64 handle) noexcept {
    delete reinterpret_cast<BindlessArray *>(handle);
}
ResourceCreationInfo LCDevice::create_stream(StreamTag stream_tag) noexcept {
    ResourceCreationInfo info{};
    auto res = new LCCmdBuffer(
        &native_device,
        native_device.default_allocator.get(),
        [&] {
            switch (stream_tag) {
                case compute::StreamTag::COMPUTE:
                    return D3D12_COMMAND_LIST_TYPE_COMPUTE;
                case compute::StreamTag::GRAPHICS:
                    return D3D12_COMMAND_LIST_TYPE_DIRECT;
                case compute::StreamTag::COPY:
                    return D3D12_COMMAND_LIST_TYPE_COPY;
                default:
                    break;
            }
            LUISA_ERROR_WITH_LOCATION("Unreachable.");
        }());
    info.handle = resource_to_handle(res);
    info.native_handle = res->queue.queue();
    return info;
}

void LCDevice::destroy_stream(uint64 handle) noexcept {
    auto queue = reinterpret_cast<CmdQueueBase *>(handle);
    switch (queue->tag()) {
        case CmdQueueTag::MainCmd:
            delete static_cast<LCCmdBuffer *>(queue);
            break;
        case CmdQueueTag::DStorage:
            delete static_cast<DStorageCommandQueue *>(queue);
            break;
    }
}
void LCDevice::synchronize_stream(uint64 stream_handle) noexcept {
    auto queue = reinterpret_cast<CmdQueueBase *>(stream_handle);
    switch (queue->tag()) {
        case CmdQueueTag::MainCmd:
            static_cast<LCCmdBuffer *>(queue)->Sync();
            break;
        case CmdQueueTag::DStorage:
            static_cast<DStorageCommandQueue *>(queue)->Complete();
            break;
    }
}
void LCDevice::dispatch(uint64 stream_handle, CommandList &&list) noexcept {
    auto queue = reinterpret_cast<CmdQueueBase *>(stream_handle);
    switch (queue->tag()) {
        case CmdQueueTag::MainCmd:
            reinterpret_cast<LCCmdBuffer *>(stream_handle)
                ->Execute(
                    list.commands(), list.steal_callbacks(), list.presents(),
                    native_device.max_allocator_count);
            break;
        case CmdQueueTag::DStorage:
            static_cast<DStorageCommandQueue *>(queue)->Execute(list.commands(), list.steal_callbacks());
            break;
    }
}
void LCDevice::set_stream_log_callback(uint64_t stream_handle,
                                       const StreamLogCallback &callback) noexcept {
    auto queue = reinterpret_cast<CmdQueueBase *>(stream_handle);
    queue->log_callback = callback;
}

ShaderCreationInfo LCDevice::create_shader(const ShaderOption &option, Function kernel) noexcept {
    LUISA_ASSERT(Device::compiler(), "Shader compiler not loaded.");
    if (kernel.requires_autodiff()) {
#ifdef LUISA_ENABLE_XIR
        auto lowered = luisa::compute::backend_detail::lower_autodiff_to_ast(kernel);
        return create_shader(option, lowered->function());
#else
        LUISA_ERROR_WITH_LOCATION("DirectX AutoDiff requires XIR support.");
#endif
    }

    ShaderCreationInfo info;
    uint mask = 0;
    if (option.enable_fast_math) {
        mask |= (1 << 0);
    }
    if (option.enable_debug_info) {
        mask |= (1 << 1);
    }
    // use default control flow
    constexpr uint compiler_version = 202403u;// dxc version at march 2024
    mask |= (1 << 2);
    mask |= compiler_version << 3u;
    auto code = hlsl::CodegenUtility{}.Codegen(kernel, option.native_include, mask, false, Device::compiler() == nullptr, option.enable_debug_info, option.enable_fast_math);
    // TODO get result from codegen
    auto choose_shader_model = [&]() -> uint {
        if (kernel.use_cooperative_operations() || code.use_8bit) {
            // Cooperative-vector kernels are always compiled to SM 6.9 because
            // the long-vector types they use require it.  When the runtime does
            // not actually support the feature, the device-level feature check
            // lets callers skip device execution instead of failing here.
            return kTensorShaderModel;
        }
        return kernel.allowed_warp_size().has_value() ? kHighShaderModel : kShaderModel;
    };
    if (option.compile_only) {
        LUISA_ASSUME(!option.name.empty());
        ComputeShader::save_compute(
            native_device.file_io,
            native_device.profiler,
            kernel,
            code,
            kernel.block_size(),
            choose_shader_model(),
            option.name,
            option.enable_fast_math,
            option.enable_debug_info);
        info.invalidate();
        info.block_size = kernel.block_size();

    } else {
        vstd::string_view file_name;
        vstd::string str_cache;
        vstd::MD5 check_md5({reinterpret_cast<uint8_t const *>(code.result.data() + code.immutableHeaderSize), code.result.size() - code.immutableHeaderSize});
        CacheType cache_type{};
        if (option.enable_cache) {
            if (option.name.empty()) {
                str_cache << check_md5.to_string(false) << ".dxil"sv;
                file_name = str_cache;
                cache_type = CacheType::Cache;
            } else {
                file_name = option.name;
                cache_type = CacheType::ByteCode;
            }
        }
        auto res = ComputeShader::compile_compute(
            native_device.file_io,
            native_device.profiler,
            &native_device,
            kernel,
            [&]() { return std::move(code); },
            check_md5,
            hlsl::binding_to_arg(kernel.bound_arguments()),
            kernel.block_size(),
            choose_shader_model(),
            file_name,
            cache_type,
            option.enable_fast_math,
            option.enable_debug_info,
            code.validation_count);
        info.block_size = kernel.block_size();
        info.handle = reinterpret_cast<uint64>(res);
        info.native_handle = res->pso();
        return info;
    }
    return info;
}
ShaderCreationInfo LCDevice::load_shader(
    vstd::string_view file_name,
    vstd::span<Type const *const> types) noexcept {
    auto res = ComputeShader::load_preset_compute(
        native_device.file_io,
        native_device.profiler,
        &native_device,
        types,
        file_name);
    ShaderCreationInfo info;
    if (res) {
        info.handle = reinterpret_cast<uint64>(res);
        info.native_handle = res->pso();
        info.block_size = res->block_size();
    } else {
        info.invalidate();
        info.block_size = uint3(0);
    }
    return info;
}
Usage LCDevice::shader_argument_usage(uint64_t handle, size_t index) noexcept {
    auto shader = reinterpret_cast<Shader *>(handle);
    return shader->args()[index].var_usage;
}
void LCDevice::destroy_shader(uint64 handle) noexcept {
    auto shader = reinterpret_cast<Shader *>(handle);
    delete shader;
}
ResourceCreationInfo LCDevice::create_event() noexcept {
    ResourceCreationInfo info{};
    auto res = new LCEvent(&native_device);
    info.handle = resource_to_handle(res);
    info.native_handle = res->fence();
    return info;
}
void LCDevice::destroy_event(uint64 handle) noexcept {
    delete reinterpret_cast<LCEvent *>(handle);
}
void LCDevice::signal_event(uint64 handle, uint64 stream_handle, uint64_t fence) noexcept {
    auto queue = reinterpret_cast<CmdQueueBase *>(stream_handle);
    switch (queue->tag()) {
        case CmdQueueTag::MainCmd:
            reinterpret_cast<LCEvent *>(handle)->signal(
                &reinterpret_cast<LCCmdBuffer *>(stream_handle)->queue, fence);
            break;
        case CmdQueueTag::DStorage:
            reinterpret_cast<LCEvent *>(handle)->signal(
                reinterpret_cast<DStorageCommandQueue *>(stream_handle), fence);
            break;
    }
}
bool LCDevice::is_event_completed(uint64_t handle, uint64_t fence) const noexcept {
    return reinterpret_cast<LCEvent *>(handle)->is_complete(fence);
}
void LCDevice::wait_event(uint64 handle, uint64 stream_handle, uint64_t fence) noexcept {
    auto queue = reinterpret_cast<CmdQueueBase *>(stream_handle);
    if (queue->tag() != CmdQueueTag::MainCmd) [[unlikely]] {
        LUISA_ERROR("Wait command not allowed in Direct-Storage.");
    }
    reinterpret_cast<LCEvent *>(handle)->wait(
        &reinterpret_cast<LCCmdBuffer *>(stream_handle)->queue, fence);
}
void LCDevice::synchronize_event(uint64 handle, uint64_t fence) noexcept {
    reinterpret_cast<LCEvent *>(handle)->sync(fence);
}
ResourceCreationInfo LCDevice::create_procedural_primitive(const AccelOption &option) noexcept {
    return create_mesh(option);
}
void LCDevice::destroy_procedural_primitive(uint64 handle) noexcept {
    destroy_mesh(handle);
}
ResourceCreationInfo LCDevice::create_mesh(const AccelOption &option) noexcept {
    ResourceCreationInfo info{};
    auto res = new BottomAccel(&native_device, option);
    info.handle = resource_to_handle(res);
    info.native_handle = nullptr;
    return info;
}
void LCDevice::destroy_mesh(uint64 handle) noexcept {
    delete reinterpret_cast<BottomAccel *>(handle);
}
ResourceCreationInfo LCDevice::create_accel(const AccelOption &option) noexcept {
    ResourceCreationInfo info{};
    auto res = new TopAccel(
        &native_device,
        option);

    info.handle = resource_to_handle(res);
    info.native_handle = nullptr;
    return info;
}
void LCDevice::destroy_accel(uint64 handle) noexcept {
    delete reinterpret_cast<TopAccel *>(handle);
}
SwapchainCreationInfo LCDevice::create_swapchain(const SwapchainOption &option, uint64_t stream_handle) noexcept {
    auto queue = reinterpret_cast<CmdQueueBase *>(stream_handle);
    if (queue->tag() != CmdQueueTag::MainCmd) [[unlikely]] {
        LUISA_ERROR("swapchain not allowed in Direct-Storage.");
    }
    SwapchainCreationInfo info{};
    auto res = new LCSwapChain(
        &native_device,
        &reinterpret_cast<LCCmdBuffer *>(stream_handle)->queue,
        native_device.default_allocator.get(),
        reinterpret_cast<HWND>(option.window),
        option.size.x,
        option.size.y,
        option.wants_hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM,
        option.wants_vsync,
        option.back_buffer_count, option.wants_transparent);
    info.handle = resource_to_handle(res);
    info.native_handle = res->swap_chain.Get();
    info.storage = option.wants_hdr ? PixelStorage::HALF4 : PixelStorage::BYTE4;
    return info;
}
void LCDevice::destroy_swapchain(uint64 handle) noexcept {
    delete reinterpret_cast<LCSwapChain *>(handle);
}
void LCDevice::present_display_in_stream(uint64 stream_handle, uint64 swapchain_handle, uint64 image_handle) noexcept {
    auto queue = reinterpret_cast<CmdQueueBase *>(stream_handle);
    if (queue->tag() != CmdQueueTag::MainCmd) [[unlikely]] {
        LUISA_ERROR("present not allowed in Direct-Storage.");
    }
    reinterpret_cast<LCCmdBuffer *>(stream_handle)
        ->Present(
            reinterpret_cast<LCSwapChain *>(swapchain_handle),
            reinterpret_cast<TextureBase *>(image_handle), 0, native_device.max_allocator_count);
}
ResourceCreationInfo DxRasterExt::create_raster_shader(
    [[maybe_unused]] const MeshFormat &mesh_format,
    Function vert,
    Function pixel,
    const ShaderOption &option) noexcept {
    uint mask = 0;
    if (option.enable_fast_math) {
        mask |= 1;
    }
    if (option.enable_debug_info) {
        mask |= 2;
    }
    auto code = hlsl::CodegenUtility{}.RasterCodegen(vert, pixel, option.native_include, mask, false, Device::compiler() == nullptr, option.enable_debug_info, option.enable_fast_math);
    vstd::MD5 check_md5({reinterpret_cast<uint8_t const *>(code.result.data() + code.immutableHeaderSize), code.result.size() - code.immutableHeaderSize});
    if (option.compile_only) {
        LUISA_ASSUME(!option.name.empty());
        RasterShader::save_raster(
            _native_device.file_io,
            &_native_device,
            code,
            check_md5,
            option.name,
            vert,
            pixel,
            kShaderModel,
            option.enable_fast_math,
            option.enable_debug_info);
        return ResourceCreationInfo::make_invalid();
    } else {
        vstd::string_view file_name;
        vstd::string str_cache;
        CacheType cache_type{};
        if (option.enable_cache) {
            if (option.name.empty()) {
                str_cache << check_md5.to_string(false) << ".dxil"sv;
                file_name = str_cache;
                cache_type = CacheType::Cache;
            } else {
                file_name = option.name;
                cache_type = CacheType::ByteCode;
            }
        }
        auto res = RasterShader::compile_raster(
            _native_device.file_io,
            &_native_device,
            vert,
            pixel,
            [&]() { return std::move(code); },
            check_md5,
            kShaderModel,
            file_name,
            cache_type,
            option.enable_fast_math,
            option.enable_debug_info);
        ResourceCreationInfo info{};
        if (res) {
            info.handle = reinterpret_cast<uint64>(res);
            info.native_handle = nullptr;
        } else {
            info.invalidate();
        }
        return info;
    }
}

ResourceCreationInfo DxRasterExt::load_raster_shader(
    span<Type const *const> types,
    string_view ser_path) noexcept {
    ResourceCreationInfo info{};
    auto res = RasterShader::load_raster(
        _native_device.file_io,
        &_native_device,
        types,
        ser_path);

    if (res) {
        info.handle = reinterpret_cast<uint64>(res);
        info.native_handle = nullptr;
        return info;
    } else {
        return ResourceCreationInfo::make_invalid();
    }
}
void DxRasterExt::destroy_raster_shader(uint64_t handle) noexcept {
    delete reinterpret_cast<RasterShader *>(handle);
}
ResourceCreationInfo DxRasterExt::create_depth_buffer(DepthFormat format, uint width, uint height) noexcept {
    ResourceCreationInfo info{};
    auto res = new DepthBuffer(
        &_native_device,
        width, height,
        format, _native_device.default_allocator.get());
    info.handle = resource_to_handle(res);
    info.native_handle = res->GetResource();
    return info;
}
void DxRasterExt::destroy_depth_buffer(uint64_t handle) noexcept {
    delete reinterpret_cast<TextureBase *>(handle);
}
DeviceExtension *LCDevice::extension(vstd::string_view name) noexcept {
    auto ite = exts.find(name);
    if (ite == exts.end()) return nullptr;
    auto &v = ite->second;
    {
        std::lock_guard lck{ext_mtx};
        if (v.ext == nullptr) {
            v.ext = v.ctor(this);
        }
    }
    return v.ext;
}
void LCDevice::set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept {
    vstd::vector<wchar_t> vec;
    luisa::enlarge_by(vec, name.size() + 1);
    vec[name.size()] = 0;
    for (auto i : vstd::range(static_cast<int64>(name.size()))) {
        vec[i] = name[i];
    }
    using Tag = luisa::compute::Resource::Tag;
    switch (resource_tag) {
        case Tag::ACCEL: {
            auto accel_buffer = reinterpret_cast<TopAccel *>(resource_handle)->GetAccelBuffer();
            if (accel_buffer) {
                accel_buffer->GetResource()->SetName(vec.data());
            }
            auto inst_buffer = reinterpret_cast<TopAccel *>(resource_handle)->GetInstBuffer();
            constexpr auto inst = L"_Instance"sv;
            luisa::vector_resize(vec, name.size() + inst.size() + 1);
            vec[vec.size() - 1] = 0;
            for (auto i : vstd::range(inst.size())) {
                vec[name.size() + i] = inst[i];
            }
            inst_buffer->GetResource()->SetName(vec.data());
        } break;
        case Tag::BINDLESS_ARRAY: {
            reinterpret_cast<BindlessArray *>(resource_handle)->BindlessBuffer()->GetResource()->SetName(vec.data());
        } break;
        case Tag::DEPTH_BUFFER:
        case Tag::TEXTURE: {
            reinterpret_cast<TextureBase *>(resource_handle)->GetResource()->SetName(vec.data());
        } break;
        case Tag::PROCEDURAL_PRIMITIVE:
        case Tag::MESH: {
            auto accel_buffer = reinterpret_cast<BottomAccel *>(resource_handle)->GetAccelBuffer();
            if (accel_buffer) {
                accel_buffer->GetResource()->SetName(vec.data());
            }
        } break;
        case Tag::STREAM: {
            reinterpret_cast<LCCmdBuffer *>(resource_handle)->queue.queue()->SetName(vec.data());
        } break;
        case Tag::EVENT: {
            reinterpret_cast<LCEvent *>(resource_handle)->fence()->SetName(vec.data());
        } break;
        case Tag::SHADER: {
            reinterpret_cast<ComputeShader *>(resource_handle)->pso()->SetName(vec.data());
        } break;
        case Tag::RASTER_SHADER: {
            // reinterpret_cast<RasterShader *>(resource_handle)->pso()->SetName(vec.data());
        } break;
        case Tag::SWAP_CHAIN: {
            size_t back_buffer = 0;
            for (auto &&i : reinterpret_cast<LCSwapChain *>(resource_handle)->render_targets) {
                luisa::vector_resize(vec, name.size());
                vec.push_back(L'_');
                auto num = vstd::to_string(back_buffer);
                for (auto &&i : num) {
                    vec.push_back(i);
                }
                vec.push_back(0);
                i.GetResource()->SetName(vec.data());
                back_buffer += 1;
            }
        } break;
        default: {
            LUISA_WARNING("Unknown resource tag.");
        } break;
    }
}

[[nodiscard]] SparseTextureCreationInfo LCDevice::create_sparse_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth,
    uint mipmap_levels, bool simultaneous_access) noexcept {
    bool allow_uav = !is_block_compressed(format);
    SparseTextureCreationInfo info;
    auto res = new SparseTexture(
        &native_device,
        width,
        height,
        TextureBase::ToGFXFormat(format),
        (TextureDimension)dimension,
        depth,
        mipmap_levels,
        allow_uav,
        simultaneous_access);
    info.handle = resource_to_handle(res);
    info.native_handle = res->GetResource();
    auto v = res->TilingSize();
    info.tile_size = v;
    info.tile_size_bytes = D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
    return info;
}
void LCDevice::destroy_sparse_texture(uint64_t handle) noexcept {
    delete reinterpret_cast<SparseTexture *>(handle);
}

SparseBufferCreationInfo LCDevice::create_sparse_buffer(const Type *element, size_t elem_count) noexcept {
    SparseBufferCreationInfo info{};
    SparseBuffer *res;
    if (element->is_custom()) {
        if (element == Type::of<IndirectKernelDispatch>()) {
            info.element_stride = ComputeShader::kDispatchIndirectStride;
            info.total_size_bytes = 4 + info.element_stride * elem_count;
            res = new SparseBuffer(&native_device, info.total_size_bytes);
        } else {
            LUISA_ERROR("Un-known custom type in dx-backend.");
        }
    } else {
        info.total_size_bytes = element->size() * elem_count;
        res = new SparseBuffer(
            &native_device,
            info.total_size_bytes);
        info.element_stride = element->size();
    }
    info.handle = resource_to_handle(res);
    info.native_handle = res->GetResource();
    info.tile_size_bytes = D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
    return info;
}
void LCDevice::destroy_sparse_buffer(uint64_t handle) noexcept {
    delete reinterpret_cast<SparseBuffer *>(handle);
}
void LCDevice::update_sparse_resources(
    uint64_t stream_handle,
    luisa::vector<SparseUpdateTile> &&update_cmds) noexcept {
    auto queue = reinterpret_cast<CmdQueueBase *>(stream_handle);

    if (queue->tag() != CmdQueueTag::MainCmd) [[unlikely]] {
        LUISA_ERROR("sparse-texture update not allowed in Direct-Storage.");
    }
    auto &queue_ptr = static_cast<LCCmdBuffer *>(queue)->queue;
    UpdateTileTracker tile_tracker;
    for (auto &&i : update_cmds) {
        luisa::visit(
            [&]<typename T>(T const &t) {
                if constexpr (std::is_same_v<T, SparseTextureMapOperation>) {
                    auto tex = reinterpret_cast<SparseTexture *>(i.handle);
                    tex->AllocateTile(t.start_tile, t.tile_count, t.mip_level, t.allocated_heap, &tile_tracker);
                } else if constexpr (std::is_same_v<T, SparseBufferMapOperation>) {
                    auto buffer = reinterpret_cast<SparseBuffer *>(i.handle);
                    buffer->AllocateTile(t.start_tile, t.tile_count, t.allocated_heap, &tile_tracker);
                } else if constexpr (std::is_same_v<T, SparseTextureUnMapOperation>) {
                    auto tex = reinterpret_cast<SparseTexture *>(i.handle);
                    tex->DeAllocateTile(t.start_tile, t.tile_count, t.mip_level, &tile_tracker);
                } else {
                    auto buffer = reinterpret_cast<SparseBuffer *>(i.handle);
                    buffer->DeAllocateTile(t.start_tile, t.tile_count, &tile_tracker);
                }
            },
            i.operations);
    }
    tile_tracker.update(queue_ptr.queue(), D3D12_TILE_MAPPING_FLAG_NONE);
    queue_ptr.signal();
}

ResourceCreationInfo LCDevice::allocate_sparse_buffer_heap(size_t byte_size) noexcept {
    auto heap = reinterpret_cast<SparseHeap *>(vengine_malloc(sizeof(SparseHeap)));
    heap->allocation = native_device.default_allocator->AllocateBufferHeap(&native_device, "sparse buffer heap", byte_size, D3D12_HEAP_TYPE_DEFAULT, &heap->heap, &heap->offset, D3D12_HEAP_FLAG_NONE, true);
    heap->size_bytes = byte_size;
    ResourceCreationInfo r{};
    r.handle = reinterpret_cast<uint64>(heap);
    r.native_handle = heap->heap;
    return r;
}
void LCDevice::deallocate_sparse_buffer_heap(uint64_t handle) noexcept {
    auto heap = reinterpret_cast<SparseHeap *>(handle);
    native_device.default_allocator->Release(heap->allocation);
    vengine_free(heap);
}
ResourceCreationInfo LCDevice::allocate_sparse_texture_heap(size_t byte_size) noexcept {
    auto heap = reinterpret_cast<SparseHeap *>(vengine_malloc(sizeof(SparseHeap)));
    heap->allocation = native_device.default_allocator->AllocateTextureHeap(&native_device, "sparse texture heap", byte_size, &heap->heap, &heap->offset, false, D3D12_HEAP_FLAG_NONE, true);
    heap->size_bytes = byte_size;
    ResourceCreationInfo r{};
    r.handle = reinterpret_cast<uint64>(heap);
    r.native_handle = heap->heap;
    return r;
}

void LCDevice::deallocate_sparse_texture_heap(uint64_t handle) noexcept {
    deallocate_sparse_buffer_heap(handle);
}

uint LCDevice::compute_warp_size() const noexcept {
    return native_device.wave_size();
}

uint64_t LCDevice::memory_granularity() const noexcept {
    // should be 64kb
    static_assert(D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES == D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
    return D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
}

LUISA_EXPORT_API DeviceInterface *create(Context &&c, DeviceConfig const *settings) {
    return new LCDevice(std::move(c), settings);
}

LUISA_EXPORT_API void destroy(DeviceInterface *device) {
    delete static_cast<LCDevice *>(device);
}

luisa::string LCDevice::query(luisa::string_view property) noexcept {
    if (property == "device_name") {
        return "dx";
    }
    LUISA_WARNING_WITH_LOCATION("Unknown device property '{}'.", property);
    return {};
}

}// namespace lc::dx

#include "../common/export_version.inl.h"
