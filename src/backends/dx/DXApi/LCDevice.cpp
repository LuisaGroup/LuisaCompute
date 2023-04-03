#include <filesystem>
#include <DXApi/LCDevice.h>
#include <DXRuntime/Device.h>
#include <Resource/DefaultBuffer.h>
#include <Shader/ShaderCompiler.h>
#include <Resource/RenderTexture.h>
#include <Resource/DepthBuffer.h>
#include <Resource/BindlessArray.h>
#include <Shader/ComputeShader.h>
#include <Shader/RasterShader.h>
#include <DXApi/LCCmdBuffer.h>
#include <DXApi/LCEvent.h>
#include <vstl/md5.h>
#include <Shader/ShaderSerializer.h>
#include <Resource/BottomAccel.h>
#include <Resource/TopAccel.h>
#include <DXApi/LCSwapChain.h>
#include "../ext.h"
#include "HLSL/dx_codegen.h"
#include <ast/function_builder.h>
#include <Resource/DepthBuffer.h>
#include <core/clock.h>
#include <core/stl/filesystem.h>
#include <Resource/ExternalBuffer.h>
#include <runtime/context_paths.h>
#include <runtime/dispatch_buffer.h>
#include <runtime/rtx/aabb.h>
#include <ext.h>

#ifdef LUISA_ENABLE_IR
#include <ir/ir2ast.h>
#endif

namespace lc::dx {
using namespace lc::dx;
static constexpr uint kShaderModel = 65u;
LCDevice::LCDevice(Context &&ctx, DeviceConfig const *settings)
    : DeviceInterface(std::move(ctx)),
      nativeDevice(_ctx, settings) {
    exts.try_emplace(
        TexCompressExt::name,
        [](LCDevice *device) -> DeviceExtension * {
            return new DxTexCompressExt(&device->nativeDevice);
        },
        [](DeviceExtension *ext) {
            delete static_cast<DxTexCompressExt *>(ext);
        });
    exts.try_emplace(
        NativeResourceExt::name,
        [](LCDevice *device) -> DeviceExtension * {
            return new DxNativeResourceExt(device, &device->nativeDevice);
        },
        [](DeviceExtension *ext) {
            delete static_cast<DxNativeResourceExt *>(ext);
        });
    exts.try_emplace(
        RasterExt::name,
        [](LCDevice *device) -> DeviceExtension * {
            return new DxRasterExt(device->nativeDevice);
        },
        [](DeviceExtension *ext) {
            delete static_cast<DxRasterExt *>(ext);
        });
}
LCDevice::~LCDevice() {
}
//Hash128 LCDevice::device_hash() const noexcept {
//    vstd::MD5::MD5Data const &md5 = nativeDevice.adapterID.to_binary();
//    Hash128 r;
//    static_assert(sizeof(Hash128) == sizeof(vstd::MD5::MD5Data));
//    memcpy(&r, &md5, sizeof(Hash128));
//    return r;
//}
void *LCDevice::native_handle() const noexcept {
    return nativeDevice.device;
}

BufferCreationInfo LCDevice::create_buffer(const Type *element, size_t elem_count) noexcept {
    BufferCreationInfo info;
    Buffer *res;
    if (element->is_custom()) {
        if (element == Type::of<IndirectKernelDispatch>()) {
            info.element_stride = 28;
            info.total_size_bytes = 4 + info.element_stride * elem_count;
            res = static_cast<Buffer *>(new DefaultBuffer(&nativeDevice, info.total_size_bytes, nativeDevice.defaultAllocator.get()));
        } else {
            LUISA_ERROR("Un-known custom type in dx-backend.");
        }
    } else {
        info.total_size_bytes = element->size() * elem_count;
        res = static_cast<Buffer *>(
            new DefaultBuffer(
                &nativeDevice,
                info.total_size_bytes,
                nativeDevice.defaultAllocator.get()));
        info.element_stride = element->size();
    }
    info.handle = reinterpret_cast<uint64>(res);
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
    uint mipmap_levels) noexcept {
    bool allowUAV = true;
    switch (format) {
        case PixelFormat::BC4UNorm:
        case PixelFormat::BC5UNorm:
        case PixelFormat::BC6HUF16:
        case PixelFormat::BC7UNorm:
            allowUAV = false;
            break;
        default: break;
    }
    ResourceCreationInfo info;
    auto res = static_cast<TextureBase *>(
        new RenderTexture(
            &nativeDevice,
            width,
            height,
            TextureBase::ToGFXFormat(format),
            (TextureDimension)dimension,
            depth,
            mipmap_levels,
            allowUAV,
            nativeDevice.defaultAllocator.get()));
    info.handle = reinterpret_cast<uint64>(res);
    info.native_handle = res->GetResource();
    return info;
}
//string LCDevice::cache_name(string_view file_name) const noexcept {
//    return Shader::PSOName(&nativeDevice, file_name);
//}
void LCDevice::destroy_texture(uint64 handle) noexcept {
    delete reinterpret_cast<TextureBase *>(handle);
}
ResourceCreationInfo LCDevice::create_bindless_array(size_t size) noexcept {
    ResourceCreationInfo info;
    auto res = new BindlessArray(&nativeDevice, size);
    info.handle = reinterpret_cast<uint64>(res);
    info.native_handle = res->GetResource();
    return info;
}
void LCDevice::destroy_bindless_array(uint64 handle) noexcept {
    delete reinterpret_cast<BindlessArray *>(handle);
}
ResourceCreationInfo LCDevice::create_stream(StreamTag type) noexcept {
    ResourceCreationInfo info;
    auto res = new LCCmdBuffer(
        &nativeDevice,
        nativeDevice.defaultAllocator.get(),
        [&] {
            switch (type) {
                case compute::StreamTag::COMPUTE:
                    return D3D12_COMMAND_LIST_TYPE_COMPUTE;
                case compute::StreamTag::GRAPHICS:
                    return D3D12_COMMAND_LIST_TYPE_DIRECT;
                case compute::StreamTag::COPY:
                    return D3D12_COMMAND_LIST_TYPE_COPY;
            }
        }());
    info.handle = reinterpret_cast<uint64>(res);
    info.native_handle = res->queue.Queue();
    return info;
}

void LCDevice::destroy_stream(uint64 handle) noexcept {
    delete reinterpret_cast<LCCmdBuffer *>(handle);
}
void LCDevice::synchronize_stream(uint64 stream_handle) noexcept {
    reinterpret_cast<LCCmdBuffer *>(stream_handle)->Sync();
}
void LCDevice::dispatch(uint64 stream_handle, CommandList &&list) noexcept {
    reinterpret_cast<LCCmdBuffer *>(stream_handle)
        ->Execute(std::move(list), nativeDevice.maxAllocatorCount);
}

ShaderCreationInfo LCDevice::create_shader(const ShaderOption &option, Function kernel) noexcept {
    ShaderCreationInfo info;
    Clock clk;
    auto code = CodegenUtility::Codegen(kernel, nativeDevice.fileIo);
    LUISA_INFO("HLSL Codegen: {} ms", clk.toc());
    if (option.compile_only) {
        assert(!option.name.empty());
        ComputeShader::SaveCompute(
            nativeDevice.fileIo,
            kernel,
            code,
            kernel.block_size(),
            kShaderModel,
            option.name,
            option.enable_fast_math);
        info.invalidate();
        info.block_size = kernel.block_size();

    } else {
        vstd::string_view file_name;
        vstd::string str_cache;
        vstd::MD5 checkMD5({reinterpret_cast<uint8_t const *>(code.result.data() + code.immutableHeaderSize), code.result.size() - code.immutableHeaderSize});
        CacheType cacheType;
        if (option.name.empty()) {
            str_cache << checkMD5.to_string(false) << ".dxil"sv;
            file_name = str_cache;
            cacheType = CacheType::Cache;
        } else {
            file_name = option.name;
            cacheType = CacheType::ByteCode;
        }
        auto res = ComputeShader::CompileCompute(
            nativeDevice.fileIo,
            &nativeDevice,
            kernel,
            [&]() { return std::move(code); },
            checkMD5,
            Shader::BindingToArg(kernel.bound_arguments()),
            kernel.block_size(),
            kShaderModel,
            file_name,
            cacheType,
            option.enable_fast_math);
        info.block_size = kernel.block_size();
        info.handle = reinterpret_cast<uint64>(res);
        info.native_handle = res->Pso();
        return info;
    }
    return info;
}
ShaderCreationInfo LCDevice::load_shader(
    vstd::string_view file_name,
    vstd::span<Type const *const> types) noexcept {
    auto res = ComputeShader::LoadPresetCompute(
        nativeDevice.fileIo,
        &nativeDevice,
        types,
        file_name);
    ShaderCreationInfo info;
    if (res) {
        info.handle = reinterpret_cast<uint64>(res);
        info.native_handle = res->Pso();
        info.block_size = res->BlockSize();
    } else {
        info.invalidate();
        info.block_size = uint3(0);
    }
    return info;
}
Usage LCDevice::shader_argument_usage(uint64_t handle, size_t index) noexcept {
    auto shader = reinterpret_cast<Shader *>(handle);
    return shader->Args()[index].varUsage;
}
void LCDevice::destroy_shader(uint64 handle) noexcept {
    auto shader = reinterpret_cast<Shader *>(handle);
    delete shader;
}
ResourceCreationInfo LCDevice::create_event() noexcept {
    ResourceCreationInfo info;
    auto res = new LCEvent(&nativeDevice);
    info.handle = reinterpret_cast<uint64>(res);
    info.native_handle = res->Fence();
    return info;
}
void LCDevice::destroy_event(uint64 handle) noexcept {
    delete reinterpret_cast<LCEvent *>(handle);
}
void LCDevice::signal_event(uint64 handle, uint64 stream_handle) noexcept {
    reinterpret_cast<LCEvent *>(handle)->Signal(
        &reinterpret_cast<LCCmdBuffer *>(stream_handle)->queue);
}

void LCDevice::wait_event(uint64 handle, uint64 stream_handle) noexcept {
    reinterpret_cast<LCEvent *>(handle)->Wait(
        &reinterpret_cast<LCCmdBuffer *>(stream_handle)->queue);
}
void LCDevice::synchronize_event(uint64 handle) noexcept {
    reinterpret_cast<LCEvent *>(handle)->Sync();
}
ResourceCreationInfo LCDevice::create_procedural_primitive(const AccelOption &option) noexcept {
    return create_mesh(option);
}
void LCDevice::destroy_procedural_primitive(uint64 handle) noexcept {
    destroy_mesh(handle);
}
ResourceCreationInfo LCDevice::create_mesh(const AccelOption &option) noexcept {
    ResourceCreationInfo info;
    auto res = new BottomAccel(&nativeDevice, option);
    info.handle = reinterpret_cast<uint64>(res);
    info.native_handle = nullptr;
    return info;
}
void LCDevice::destroy_mesh(uint64 handle) noexcept {
    delete reinterpret_cast<BottomAccel *>(handle);
}
ResourceCreationInfo LCDevice::create_accel(const AccelOption &option) noexcept {
    ResourceCreationInfo info;
    auto res = new TopAccel(
        &nativeDevice,
        option);

    info.handle = reinterpret_cast<uint64>(res);
    info.native_handle = nullptr;
    return info;
}
void LCDevice::destroy_accel(uint64 handle) noexcept {
    delete reinterpret_cast<TopAccel *>(handle);
}
SwapChainCreationInfo LCDevice::create_swap_chain(
    uint64 window_handle,
    uint64 stream_handle,
    uint width,
    uint height,
    bool allow_hdr,
    bool vsync,
    uint back_buffer_size) noexcept {
    SwapChainCreationInfo info;
    auto res = new LCSwapChain(
        &nativeDevice,
        &reinterpret_cast<LCCmdBuffer *>(stream_handle)->queue,
        nativeDevice.defaultAllocator.get(),
        reinterpret_cast<HWND>(window_handle),
        width,
        height,
        allow_hdr,
        vsync,
        back_buffer_size);
    info.handle = reinterpret_cast<uint64>(res);
    info.native_handle = res->swapChain.Get();
    info.storage = PixelStorage::BYTE4;
    return info;
}
void LCDevice::destroy_swap_chain(uint64 handle) noexcept {
    delete reinterpret_cast<LCSwapChain *>(handle);
}
void LCDevice::present_display_in_stream(uint64 stream_handle, uint64 swapchain_handle, uint64 image_handle) noexcept {
    reinterpret_cast<LCCmdBuffer *>(stream_handle)
        ->Present(
            reinterpret_cast<LCSwapChain *>(swapchain_handle),
            reinterpret_cast<TextureBase *>(image_handle), nativeDevice.maxAllocatorCount);
}
ResourceCreationInfo DxRasterExt::create_raster_shader(
    const MeshFormat &mesh_format,
    Function vert,
    Function pixel,
    const ShaderOption &option) noexcept {
    auto code = CodegenUtility::RasterCodegen(mesh_format, vert, pixel, nativeDevice.fileIo);
    vstd::MD5 checkMD5({reinterpret_cast<uint8_t const *>(code.result.data() + code.immutableHeaderSize), code.result.size() - code.immutableHeaderSize});
    if (option.compile_only) {
        assert(!option.name.empty());
        RasterShader::SaveRaster(
            nativeDevice.fileIo,
            &nativeDevice,
            code,
            checkMD5,
            option.name,
            vert,
            pixel,
            kShaderModel,
            option.enable_fast_math);
        return ResourceCreationInfo::make_invalid();
    } else {
        vstd::string_view file_name;
        vstd::string str_cache;
        CacheType cacheType;
        if (option.name.empty()) {
            str_cache << checkMD5.to_string(false) << ".dxil"sv;
            file_name = str_cache;
            cacheType = CacheType::Cache;
        } else {
            file_name = option.name;
            cacheType = CacheType::ByteCode;
        }
        ResourceCreationInfo info;
        auto res = RasterShader::CompileRaster(
            nativeDevice.fileIo,
            &nativeDevice,
            vert,
            pixel,
            [&] { return std::move(code); },
            checkMD5,
            kShaderModel,
            mesh_format,
            file_name,
            cacheType,
            option.enable_fast_math);
        info.handle = reinterpret_cast<uint64>(res);
        info.native_handle = nullptr;
        return info;
    }
}

ResourceCreationInfo DxRasterExt::load_raster_shader(
    const MeshFormat &mesh_format,
    span<Type const *const> types,
    string_view ser_path) noexcept {
    ResourceCreationInfo info;
    auto res = RasterShader::LoadRaster(
        nativeDevice.fileIo,
        &nativeDevice,
        mesh_format,
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
    ResourceCreationInfo info;
    auto res =
        static_cast<TextureBase *>(
            new DepthBuffer(
                &nativeDevice,
                width, height,
                format, nativeDevice.defaultAllocator.get()));
    info.handle = reinterpret_cast<uint64>(res);
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
        std::lock_guard lck{extMtx};
        if (v.ext == nullptr) {
            v.ext = v.ctor(this);
        }
    }
    return v.ext;
}
void LCDevice::set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept {
    vstd::vector<wchar_t> vec;
    vec.push_back_uninitialized(name.size() + 1);
    vec[name.size()] = 0;
    for (auto i : vstd::range(name.size())) {
        vec[i] = name[i];
    }
    using Tag = luisa::compute::Resource::Tag;
    switch (resource_tag) {
        case Tag::ACCEL: {
            auto accelBuffer = reinterpret_cast<TopAccel *>(resource_handle)->GetAccelBuffer();
            if (accelBuffer) {
                accelBuffer->GetResource()->SetName(vec.data());
            }
            auto instBuffer = reinterpret_cast<TopAccel *>(resource_handle)->GetInstBuffer();
            constexpr auto inst = L"_Instance"sv;
            vec.resize_uninitialized(name.size() + inst.size() + 1);
            vec[vec.size() - 1] = 0;
            for (auto i : vstd::range(inst.size())) {
                vec[name.size() + i] = inst[i];
            }
            instBuffer->GetResource()->SetName(vec.data());
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
            auto accelBuffer = reinterpret_cast<BottomAccel *>(resource_handle)->GetAccelBuffer();
            if (accelBuffer) {
                accelBuffer->GetResource()->SetName(vec.data());
            }
        } break;
        case Tag::STREAM: {
            reinterpret_cast<LCCmdBuffer *>(resource_handle)->queue.Queue()->SetName(vec.data());
        } break;
        case Tag::EVENT: {
            reinterpret_cast<LCEvent *>(resource_handle)->Fence()->SetName(vec.data());
        } break;
        case Tag::SHADER: {
            reinterpret_cast<ComputeShader *>(resource_handle)->Pso()->SetName(vec.data());
        } break;
        case Tag::RASTER_SHADER: {
            // reinterpret_cast<RasterShader *>(resource_handle)->Pso()->SetName(vec.data());
        } break;
        case Tag::SWAP_CHAIN: {
            size_t backBuffer = 0;
            for (auto &&i : reinterpret_cast<LCSwapChain *>(resource_handle)->m_renderTargets) {
                vec.resize_uninitialized(name.size());
                vec.push_back(L'_');
                auto num = vstd::to_string(backBuffer);
                for (auto &&i : num) {
                    vec.push_back(i);
                }
                vec.push_back(0);
                i.GetResource()->SetName(vec.data());
                backBuffer += 1;
            }
        } break;
        default: {
            LUISA_ERROR("Unknown resource tag.");
        } break;
    }
}

BufferCreationInfo LCDevice::create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count) noexcept {
#ifdef LUISA_ENABLE_IR
    auto type = IR2AST::get_type(element->get());
    return create_buffer(type, elem_count);
#else
    LUISA_ERROR_WITH_LOCATION("DirectX device does not support creating shader from IR types.");
#endif
}

ShaderCreationInfo LCDevice::create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept {
#ifdef LUISA_ENABLE_IR
    Clock clk;
    auto function = IR2AST::build(kernel);
    LUISA_INFO("IR2AST done in {} ms.", clk.toc());
    return create_shader(option, function->function());
#else
    LUISA_ERROR_WITH_LOCATION("DirectX device does not support creating shader from IR types.");
#endif
}

VSTL_EXPORT_C DeviceInterface *create(Context &&c, DeviceConfig const *settings) {
    return new LCDevice(std::move(c), settings);
}
VSTL_EXPORT_C void destroy(DeviceInterface *device) {
    delete static_cast<LCDevice *>(device);
}

}// namespace lc::dx