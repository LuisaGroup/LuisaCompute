//
// Created by mike on 12/25/25.
//

#include <luisa/core/dll_export.h>
#include <luisa/core/clock.h>

#include "hip_check.h"
#include "hip_buffer.h"
#include "hip_texture.h"
#include "hip_bindless_array.h"
#include "hip_stream.h"
#include "hip_event.h"
#include "hip_swapchain.h"
#include "hip_shader.h"
#include "hip_shader_native.h"
#include "hip_mesh.h"
#include "hip_accel.h"
#include "hip_device.h"

#ifdef LUISA_COMPUTE_ENABLE_LLVM
#include <luisa/xir/translators/ast2xir.h>
#include "llvm_codegen/hip_codegen_llvm.h"
#endif

namespace luisa::compute::hip {

void luisa_initialize_hip() noexcept {
    static std::once_flag flag;
    std::call_once(flag, [] {
        LUISA_CHECK_HIP(hipInit(0));
    });
}

struct HIPDeviceGuard {
    int current_device_id = 0;
    int previous_device_id = 0;
    explicit HIPDeviceGuard(int device_id) noexcept : current_device_id{device_id} {
        LUISA_CHECK_HIP(hipGetDevice(&previous_device_id));
        if (previous_device_id != current_device_id) {
            LUISA_CHECK_HIP(hipSetDevice(current_device_id));
        }
    }
    ~HIPDeviceGuard() noexcept {
        if (current_device_id != previous_device_id) {
            LUISA_CHECK_HIP(hipSetDevice(previous_device_id));
        }
    }
    HIPDeviceGuard(HIPDeviceGuard &&) noexcept = delete;
    HIPDeviceGuard(const HIPDeviceGuard &) noexcept = delete;
    HIPDeviceGuard &operator=(HIPDeviceGuard &&) noexcept = delete;
    HIPDeviceGuard &operator=(const HIPDeviceGuard &) noexcept = delete;
};

template<typename F>
decltype(auto) HIPDevice::with_device(F &&f) const noexcept {
    HIPDeviceGuard guard{_device_id};
    return std::invoke(std::forward<F>(f));
}

HIPDevice::HIPDevice(Context &&ctx, const DeviceConfig *config) noexcept
    : DeviceInterface{std::move(ctx)},
      _default_io{config == nullptr || config->binary_io == nullptr ?
                      luisa::make_unique<DefaultBinaryIO>(context()) :
                      nullptr},
      _io{_default_io == nullptr ? config->binary_io : _default_io.get()},
      _device_id{config == nullptr || config->device_index == ~0ull ? 0 : static_cast<int>(config->device_index)} {
    auto device_count = 0;
    LUISA_CHECK_HIP(hipGetDeviceCount(&device_count));
    LUISA_ASSERT(_device_id < device_count,
                 "HIP device index out of range (required = {}, count = {}).",
                 _device_id, device_count);
    // log device name and version
    hipDeviceProp_t prop;
    LUISA_CHECK_HIP(hipGetDeviceProperties(&prop, _device_id));
    _gcn_arch = std::stoi(prop.gcnArchName + 3);
    auto driver_version = 0;
    auto runtime_version = 0;
    LUISA_CHECK_HIP(hipDriverGetVersion(&driver_version));
    LUISA_CHECK_HIP(hipRuntimeGetVersion(&runtime_version));
    auto version_major = [](auto x) noexcept { return x / 10000000; };
    auto version_minor = [](auto x) noexcept { return x % 10000000 / 100000; };
    auto version_patch = [](auto x) noexcept { return x % 100000; };
    LUISA_INFO("Created HIP device {}: {} (cc = {}.{}, driver = {}.{}.{}, runtime = {}.{}.{}, build = {}.{}.{}).",
               _device_id, prop.name, prop.major, prop.minor,
               version_major(driver_version), version_minor(driver_version), version_patch(driver_version),
               version_major(runtime_version), version_minor(runtime_version), version_patch(runtime_version),
               HIP_VERSION_MAJOR, HIP_VERSION_MINOR, HIP_VERSION_PATCH);

    // create hiprt context
    hiprtContextCreationInput hiprt_ctx_input{};
    hiprt_ctx_input.deviceType = std::string_view{prop.name}.find("NVIDIA") != std::string::npos ?
                                     hiprtDeviceNVIDIA :
                                     hiprtDeviceAMD;
    hiprt_ctx_input.device = device_id();
    LUISA_CHECK_HIP(hipCtxGetCurrent(reinterpret_cast<hipCtx_t *>(&hiprt_ctx_input.ctxt)));
    LUISA_CHECK_HIPRT(hiprtCreateContext(HIPRT_API_VERSION, hiprt_ctx_input, _hiprt_context));
    LUISA_CHECK_HIPRT(hiprtSetLogLevel(_hiprt_context,
                                       static_cast<hiprtLogLevel>(hiprtLogLevelInfo | hiprtLogLevelWarn | hiprtLogLevelError)));

    // create global stack buffer for HIPRT traversal (LDS-backed, avoids scratch memory)
    hiprtGlobalStackBufferInput stack_input{};
    stack_input.type = hiprtStackTypeGlobal;
    stack_input.entryType = hiprtStackEntryTypeInteger;
    stack_input.stackSize = 64u;
    // threadCount determines global stack buffer allocation:
    // size = stackSize * threadCount * sizeof(uint32_t)
    // Use max expected dispatch size. 2048*2048 covers typical RT renders.
    stack_input.threadCount = 2048u * 2048u;
    LUISA_CHECK_HIPRT(hiprtCreateGlobalStackBuffer(_hiprt_context, stack_input, _hiprt_global_stack_buffer));
    LUISA_INFO("Created HIPRT global stack buffer (stackSize={}, threadCount={}).",
               stack_input.stackSize, stack_input.threadCount);

    // hipLimitStackSize controls per-thread scratch allocation for uses_dynamic_stack=true kernels.
    // HIPRT traversal uses its own GlobalStack (LDS + global memory), not this.
    // With full LTO inlining of HIPRT functions, no dynamic call stack is needed.
    {
        auto ret = hipDeviceSetLimit(hipLimitStackSize, 0u);
        if (ret != hipSuccess) {
            LUISA_WARNING("hipDeviceSetLimit(hipLimitStackSize) failed: {}",
                          hipGetErrorString(ret));
        }
    }
}

HIPDevice::~HIPDevice() noexcept {
    LUISA_CHECK_HIPRT(hiprtDestroyGlobalStackBuffer(_hiprt_context, _hiprt_global_stack_buffer));
    LUISA_CHECK_HIPRT(hiprtDestroyContext(_hiprt_context));
}

void HIPDevice::set_stream_log_callback(uint64_t stream_handle, const StreamLogCallback &callback) noexcept {
    DeviceInterface::set_stream_log_callback(stream_handle, callback);
}

ShaderCreationInfo HIPDevice::create_shader(const ShaderOption &option, const ir_v2::KernelModule &kernel) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

ResourceCreationInfo HIPDevice::create_curve(const AccelOption &option) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_curve(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

ResourceCreationInfo HIPDevice::create_motion_instance(const AccelMotionOption &option) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_motion_instance(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

luisa::string HIPDevice::query(luisa::string_view property) noexcept {
    // TODO: support more properties
    return DeviceInterface::query(property);
}

DeviceExtension *HIPDevice::extension(luisa::string_view name) noexcept {
    // TODO: support extensions
    return DeviceInterface::extension(name);
}

luisa::string_view HIPDevice::get_name(uint64_t resource_handle) const noexcept {
    return DeviceInterface::get_name(resource_handle);
}

SparseBufferCreationInfo HIPDevice::create_sparse_buffer(const Type *element, size_t elem_count) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

ResourceCreationInfo HIPDevice::allocate_sparse_buffer_heap(size_t byte_size) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::deallocate_sparse_buffer_heap(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::update_sparse_resources(uint64_t stream_handle, luisa::vector<SparseUpdateTile> &&textures_update) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_sparse_buffer(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

ResourceCreationInfo HIPDevice::allocate_sparse_texture_heap(size_t byte_size) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::deallocate_sparse_texture_heap(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

SparseTextureCreationInfo HIPDevice::create_sparse_texture(PixelFormat format, uint dimension,
                                                           uint width, uint height, uint depth,
                                                           uint mipmap_levels, bool simultaneous_access) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_sparse_texture(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

hipUUID_t HIPDevice::device_uuid() const noexcept {
    hipUUID_t uuid{};
    LUISA_CHECK_HIP(hipDeviceGetUuid(&uuid, _device_id));
    return uuid;
}

hipUUID_t HIPDevice::device_uuid_for_vulkan() const noexcept {
    // The value that hipDeviceGetUuid returns does not correspond with those returned
    // by mesa (see https://gitlab.freedesktop.org/mesa/mesa/-/blob/5cd3e395037250946ba2519600836341df02c8ca/src/amd/common/ac_gpu_info.c#L1366-1382)
    // and by xgl (see https://github.com/GPUOpen-Drivers/xgl/blob/4118707939c2f4783d28ce2a383184a3794ca477/icd/api/vk_physical_device.cpp#L4363-L4421)
    // Those drivers _do_ align with each other, so we can create our own UUID here.
    // \see https://github.com/ROCm-Developer-Tools/hipamd/issues/50.
    hipDeviceProp_t props;
    LUISA_CHECK_HIP(hipGetDeviceProperties(&props, device_id()));
    hipUUID_t result = {};
    auto uuid_ints = reinterpret_cast<uint32_t *>(result.bytes);
    uuid_ints[0] = props.pciDomainID;
    uuid_ints[1] = props.pciBusID;
    uuid_ints[2] = props.pciDeviceID;
    return result;
}

void *HIPDevice::native_handle() const noexcept {
    return reinterpret_cast<void *>(static_cast<uintptr_t>(_device_id));
}

uint HIPDevice::compute_warp_size() const noexcept {
    int warp_size = 0;
    LUISA_CHECK_HIP(hipDeviceGetAttribute(
        &warp_size,
        hipDeviceAttributeWarpSize,
        _device_id));
    return static_cast<uint>(warp_size);
}

uint64_t HIPDevice::memory_granularity() const noexcept {
    LUISA_NOT_IMPLEMENTED();
}

BufferCreationInfo HIPDevice::create_buffer(const Type *element, size_t elem_count, void *external_memory) noexcept {
    LUISA_ASSERT(element == nullptr || element->is_basic() || element->is_structure() || element->is_array(),
                 "Invalid buffer element type {}.", element->description());
    auto elem_stride = element == nullptr ? 1u : element->size();
    auto size_bytes = elem_stride * elem_count;
    auto buffer = with_device([&] {
        return external_memory == nullptr ?
                   HIPBuffer::create_device_buffer(size_bytes) :
                   HIPBuffer::import_external_device_buffer(external_memory, size_bytes);
    });
    BufferCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(buffer);
    info.native_handle = buffer->handle();
    info.element_stride = elem_stride;
    info.total_size_bytes = size_bytes;
    return info;
}

BufferCreationInfo HIPDevice::create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count, void *external_memory) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_buffer(uint64_t handle) noexcept {
    auto buffer = reinterpret_cast<HIPBuffer *>(handle);
    with_device([&] { HIPBuffer::destroy(buffer); });
}

ResourceCreationInfo HIPDevice::create_texture(PixelFormat format, uint dimension,
                                               uint width, uint height, uint depth,
                                               uint mipmap_levels, void *external_native_handle,
                                               bool simultaneous_access, bool allow_raster_target) noexcept {
    auto p = with_device([&] {
        return external_native_handle == nullptr ?
                   HIPTexture::create_device_texture(format, dimension,
                                                     make_uint3(width, height, depth),
                                                     mipmap_levels) :
                   HIPTexture::import_external_texture(
                       reinterpret_cast<uint64_t>(external_native_handle),
                       format, dimension,
                       make_uint3(width, height, depth),
                       mipmap_levels);
    });
    return {.handle = reinterpret_cast<uint64_t>(p),
            .native_handle = p->handle()};
}

void HIPDevice::destroy_texture(uint64_t handle) noexcept {
    auto texture = reinterpret_cast<HIPTexture *>(handle);
    with_device([&] {
        luisa::delete_with_allocator(texture);
    });
}

ResourceCreationInfo HIPDevice::create_bindless_array(size_t size, BindlessSlotType type) noexcept {
    auto p = with_device([&] {
        return luisa::new_with_allocator<HIPBindlessArray>(size);
    });
    return {.handle = reinterpret_cast<uint64_t>(p),
            .native_handle = reinterpret_cast<void *>(p->handle())};
}

void HIPDevice::destroy_bindless_array(uint64_t handle) noexcept {
    with_device([&] {
        auto array = reinterpret_cast<HIPBindlessArray *>(handle);
        luisa::delete_with_allocator(array);
    });
}

ResourceCreationInfo HIPDevice::create_stream(StreamTag stream_tag) noexcept {
    auto p = with_device([&] {
        return luisa::new_with_allocator<HIPStream>(this);
    });
    return {.handle = reinterpret_cast<uint64_t>(p),
            .native_handle = p->handle()};
}

void HIPDevice::destroy_stream(uint64_t handle) noexcept {
    with_device([&] {
        auto stream = reinterpret_cast<HIPStream *>(handle);
        delete_with_allocator(stream);
    });
}

void HIPDevice::synchronize_stream(uint64_t stream_handle) noexcept {
    with_device([&] {
        auto stream = reinterpret_cast<HIPStream *>(stream_handle);
        stream->synchronize();
    });
}

void HIPDevice::dispatch(uint64_t stream_handle, CommandList &&list) noexcept {
    if (!list.empty()) {
        with_device([&] {
            auto stream = reinterpret_cast<HIPStream *>(stream_handle);
            stream->dispatch(std::move(list));
        });
    }
}

namespace {

#ifndef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
void report_swapchain_not_enabled() noexcept {
    LUISA_ERROR_WITH_LOCATION("Swapchains are not enabled on the HIP backend. "
                              "You need to enable the GUI module and install "
                              "the Vulkan SDK (>= 1.1) to enable it.");
}
#endif

}// namespace

SwapchainCreationInfo HIPDevice::create_swapchain(const SwapchainOption &option, uint64_t stream_handle) noexcept {
#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
    auto chain = with_device([&] {
        return new_with_allocator<HIPSwapchain>(this, option);
    });
    SwapchainCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(chain);
    info.native_handle = chain->native_handle();
    info.storage = chain->pixel_storage();
    return info;
#else
    report_swapchain_not_enabled();
#endif
}

void HIPDevice::destroy_swapchain(uint64_t handle) noexcept {
#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
    with_device([chain = reinterpret_cast<HIPSwapchain *>(handle)] {
        delete_with_allocator(chain);
    });
#else
    report_swapchain_not_enabled();
#endif
}

void HIPDevice::present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {
#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
    with_device([stream = reinterpret_cast<HIPStream *>(stream_handle),
                 chain = reinterpret_cast<HIPSwapchain *>(swapchain_handle),
                 image = reinterpret_cast<HIPTexture *>(image_handle)] {
        chain->present(stream, image);
    });
#else
    report_swapchain_not_enabled();
#endif
}

ShaderCreationInfo HIPDevice::create_shader(const ShaderOption &option, Function kernel) noexcept {
#ifdef LUISA_COMPUTE_ENABLE_LLVM
    Clock translate_clk;
    auto xir_module = xir::ast_to_xir_translate(kernel, {});
    xir_module->set_name(luisa::format("kernel_{:016x}", kernel.hash()));
    LUISA_VERBOSE("AST to XIR translation done in {} ms.", translate_clk.toc());

    HIPCodegenLLVMConfig config{
        .source_file = option.name,
        .bindings = kernel.bound_arguments(),
        .block_size = {kernel.block_size().x, kernel.block_size().y, kernel.block_size().z},
        .amdgpu_arch = 1201,
        .opt_level = HIPCodegenLLVMConfig::OptLevel::LEVEL_AGGRESSIVE,
        .enable_fast_math = option.enable_fast_math,
        .enable_debug_info = option.enable_debug_info,
        .requires_ray_tracing = kernel.requires_raytracing(),
    };

    auto code = hip_codegen_llvm(*xir_module, config);
    LUISA_INFO("Generated AMDGPU code ({} bytes)", code.size());

    luisa::vector<Usage> argument_usages;
    argument_usages.reserve(kernel.arguments().size());
    for (auto &&arg : kernel.arguments()) {
        argument_usages.push_back(kernel.variable_usage(arg.uid()));
    }

    HIPShaderMetadata metadata{
        .checksum = 0u,
        .kind = kernel.requires_raytracing() ?
                    HIPShaderMetadata::Kind::RAY_TRACING :
                    HIPShaderMetadata::Kind::COMPUTE,
        .enable_debug = option.enable_debug_info,
        .requires_trace_closest = kernel.propagated_builtin_callables().test(CallOp::RAY_TRACING_TRACE_CLOSEST) ||
                                  kernel.propagated_builtin_callables().test(CallOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR),
        .requires_trace_any = kernel.propagated_builtin_callables().test(CallOp::RAY_TRACING_TRACE_ANY) ||
                              kernel.propagated_builtin_callables().test(CallOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR),
        .requires_ray_query = kernel.propagated_builtin_callables().uses_ray_query(),
        .requires_printing = kernel.requires_printing(),
        .requires_motion_blur = kernel.requires_motion_blur(),
        .max_register_count = 0u,
        .block_size = kernel.block_size(),
        .argument_types = {},
        .argument_usages = std::move(argument_usages),
        .format_types = {},
    };

    // process bound arguments
    luisa::vector<ShaderDispatchCommand::Argument> bound_arguments;
    bound_arguments.reserve(kernel.bound_arguments().size());
    for (auto &&arg : kernel.bound_arguments()) {
        luisa::visit(
            [&bound_arguments]<typename T>(T binding) noexcept {
                ShaderDispatchCommand::Argument argument{};
                if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                    argument.tag = ShaderDispatchCommand::Argument::Tag::BUFFER;
                    argument.buffer.handle = binding.handle;
                    argument.buffer.offset = binding.offset;
                    argument.buffer.size = binding.size;
                } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                    argument.tag = ShaderDispatchCommand::Argument::Tag::TEXTURE;
                    argument.texture.handle = binding.handle;
                    argument.texture.level = binding.level;
                } else if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                    argument.tag = ShaderDispatchCommand::Argument::Tag::BINDLESS_ARRAY;
                    argument.bindless_array.handle = binding.handle;
                } else if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                    argument.tag = ShaderDispatchCommand::Argument::Tag::ACCEL;
                    argument.accel.handle = binding.handle;
                } else {
                    LUISA_ERROR_WITH_LOCATION("Unsupported binding type.");
                }
                bound_arguments.emplace_back(argument);
            },
            arg);
    }

    HIPShaderNative *shader = nullptr;
    if (metadata.kind == HIPShaderMetadata::Kind::RAY_TRACING) {
        shader = luisa::new_with_allocator<HIPShaderNative>(
            this, std::move(code),
            "kernel_main", metadata,
            _hiprt_context,
            std::move(bound_arguments));
    } else {
        shader = luisa::new_with_allocator<HIPShaderNative>(
            this, std::move(code),
            "kernel_main", metadata, std::move(bound_arguments));
    }

    ShaderCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(shader);
    info.block_size = kernel.block_size();
    return info;
#else
    LUISA_ERROR_WITH_LOCATION("HIP backend requires LLVM to be enabled.");
    return ShaderCreationInfo::make_invalid();
#endif
}

ShaderCreationInfo HIPDevice::create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

ShaderCreationInfo HIPDevice::load_shader(luisa::string_view name, luisa::span<Type const *const> arg_types) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

Usage HIPDevice::shader_argument_usage(uint64_t handle, size_t index) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_shader(uint64_t handle) noexcept {
    auto shader = reinterpret_cast<HIPShader *>(handle);
    luisa::delete_with_allocator(shader);
}

ResourceCreationInfo HIPDevice::create_event() noexcept {
    auto event = with_device([&] {
        return luisa::new_with_allocator<HIPEvent>(this);
    });
    return {.handle = reinterpret_cast<uint64_t>(event),
            .native_handle = event->handle()};
}

void HIPDevice::destroy_event(uint64_t handle) noexcept {
    auto event = reinterpret_cast<HIPEvent *>(handle);
    with_device([&] {
        luisa::delete_with_allocator(event);
    });
}

void HIPDevice::signal_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept {
    auto event = reinterpret_cast<HIPEvent *>(handle);
    auto stream = reinterpret_cast<HIPStream *>(stream_handle);
    with_device([&] {
        event->signal(stream->handle(), fence_value);
    });
}

void HIPDevice::wait_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept {
    auto event = reinterpret_cast<HIPEvent *>(handle);
    auto stream = reinterpret_cast<HIPStream *>(stream_handle);
    with_device([&] {
        event->wait(stream->handle(), fence_value);
    });
}

bool HIPDevice::is_event_completed(uint64_t handle, uint64_t fence_value) const noexcept {
    auto event = reinterpret_cast<HIPEvent *>(handle);
    return event->has_signaled(fence_value);
}

void HIPDevice::synchronize_event(uint64_t handle, uint64_t fence_value) noexcept {
    auto event = reinterpret_cast<HIPEvent *>(handle);
    event->synchronize(fence_value);
}

ResourceCreationInfo HIPDevice::create_mesh(const AccelOption &option) noexcept {
    auto mesh = with_device([&] {
        return luisa::new_with_allocator<HIPMesh>(_hiprt_context, option);
    });
    return {.handle = reinterpret_cast<uint64_t>(mesh),
            .native_handle = mesh};
}

void HIPDevice::destroy_mesh(uint64_t handle) noexcept {
    with_device([=] {
        auto mesh = reinterpret_cast<HIPMesh *>(handle);
        luisa::delete_with_allocator(mesh);
    });
}

ResourceCreationInfo HIPDevice::create_procedural_primitive(const AccelOption &option) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

void HIPDevice::destroy_procedural_primitive(uint64_t handle) noexcept {
    LUISA_NOT_IMPLEMENTED();
}

ResourceCreationInfo HIPDevice::create_accel(const AccelOption &option) noexcept {
    auto accel = with_device([&] {
        return luisa::new_with_allocator<HIPAccel>(_hiprt_context, option);
    });
    return {.handle = reinterpret_cast<uint64_t>(accel),
            .native_handle = accel};
}

void HIPDevice::destroy_accel(uint64_t handle) noexcept {
    with_device([=] {
        auto accel = reinterpret_cast<HIPAccel *>(handle);
        luisa::delete_with_allocator(accel);
    });
}

void HIPDevice::set_name(Resource::Tag resource_tag,
                         uint64_t resource_handle,
                         luisa::string_view name) noexcept {
    // ignored
}

}// namespace luisa::compute::hip

LUISA_EXPORT_API luisa::compute::DeviceInterface *create(luisa::compute::Context &&ctx,
                                                         const luisa::compute::DeviceConfig *config) noexcept {
    luisa::compute::hip::luisa_initialize_hip();
    return luisa::new_with_allocator<luisa::compute::hip::HIPDevice>(std::move(ctx), config);
}

LUISA_EXPORT_API void destroy(luisa::compute::DeviceInterface *device) noexcept {
    luisa::delete_with_allocator(device);
}

LUISA_EXPORT_API void backend_device_names(luisa::vector<luisa::string> &names) noexcept {
    names.clear();
    auto count = 0;
    luisa::compute::hip::luisa_initialize_hip();
    LUISA_CHECK_HIP(hipGetDeviceCount(&count));
    names.reserve(count);
    for (int i = 0; i < count; i++) {
        hipDeviceProp_t prop;
        LUISA_CHECK_HIP(hipGetDeviceProperties(&prop, i));
        names.emplace_back(prop.name);
    }
}

#include "../common/export_version.inl.h"
