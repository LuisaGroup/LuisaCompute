#include "simd_device.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <string_view>
#include <thread>

#if LUISA_COMPUTE_SIMD_HAS_TBB_SCHEDULER_HANDLE
#include <oneapi/tbb/global_control.h>
#endif

#include <luisa/ast/type_registry.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/platform.h>
#include <luisa/core/stl/memory.h>
#include <luisa/backends/ext/simd_config_ext.h>

#include "simd_bindless_array.h"
#include "simd_accel.h"
#include "simd_buffer.h"
#include "simd_curve.h"
#include "simd_event.h"
#include "simd_mesh.h"
#include "simd_motion_instance.h"
#include "simd_procedural_primitive.h"
#include "simd_shader.h"
#include "simd_stream.h"
#include "simd_thread_pool.h"
#include "simd_texture.h"

namespace luisa::compute::simd {

namespace {

class SharedEmbreeDevice {

private:
    RTCDevice _handle;

public:
    SharedEmbreeDevice() noexcept
        : _handle{rtcNewDevice(nullptr)} {
        LUISA_ASSERT(
            _handle != nullptr,
            "Failed to create the shared SIMD Embree device.");
        rtcSetDeviceErrorFunction(
            _handle,
            [](void *, RTCError code, const char *message) noexcept {
                if (code != RTC_ERROR_NONE) {
                    LUISA_ERROR_WITH_LOCATION(
                        "SIMD Embree error (code = {}): {}",
                        luisa::to_underlying(code), message);
                }
            },
            nullptr);
        LUISA_VERBOSE(
            "SIMD Embree {}.{}.{} native ray packets: W4={}, W8={}, W16={}",
            rtcGetDeviceProperty(
                _handle, RTC_DEVICE_PROPERTY_VERSION_MAJOR),
            rtcGetDeviceProperty(
                _handle, RTC_DEVICE_PROPERTY_VERSION_MINOR),
            rtcGetDeviceProperty(
                _handle, RTC_DEVICE_PROPERTY_VERSION_PATCH),
            rtcGetDeviceProperty(
                _handle, RTC_DEVICE_PROPERTY_NATIVE_RAY4_SUPPORTED),
            rtcGetDeviceProperty(
                _handle, RTC_DEVICE_PROPERTY_NATIVE_RAY8_SUPPORTED),
            rtcGetDeviceProperty(
                _handle, RTC_DEVICE_PROPERTY_NATIVE_RAY16_SUPPORTED));
    }

    ~SharedEmbreeDevice() noexcept {
#if LUISA_COMPUTE_SIMD_HAS_TBB_SCHEDULER_HANDLE
        if (rtcGetDeviceProperty(
                _handle,
                RTC_DEVICE_PROPERTY_TASKING_SYSTEM) == 1) {
            oneapi::tbb::task_scheduler_handle scheduler{
                oneapi::tbb::attach{}};
            rtcReleaseDevice(_handle);
            _handle = nullptr;
            if (!oneapi::tbb::finalize(scheduler, std::nothrow)) {
                LUISA_WARNING_WITH_LOCATION(
                    "Failed to quiesce the Embree TBB scheduler before "
                    "unloading the SIMD backend.");
            }
            return;
        }
#endif
        rtcReleaseDevice(_handle);
        _handle = nullptr;
    }

    [[nodiscard]] auto handle() const noexcept { return _handle; }
};

[[nodiscard]] RTCDevice shared_embree_device() noexcept {
    static SharedEmbreeDevice device;
    return device.handle();
}

}// namespace

SIMDDevice::SIMDDevice(
    Context &&context, const DeviceConfig *config) noexcept
    : DeviceInterface{std::move(context)} {
    auto requested_worker_count = uint32_t{0u};
    auto requested_width = uint32_t{0u};
    if (config != nullptr && config->extension != nullptr) {
        auto *simd_config = static_cast<const SIMDDeviceConfigExt *>(
            config->extension.get());
        requested_width = simd_config->warp_width();
        requested_worker_count = simd_config->worker_count();
    }
    // Diagnostic/benchmark override for examples that do not construct a
    // SIMDDeviceConfigExt. An explicit nonzero API setting always wins.
    if (requested_width == 0u) {
        if (auto *environment =
                std::getenv("LUISA_SIMD_WARP_WIDTH");
            environment != nullptr) {
            auto text = std::string_view{environment};
            auto result = std::from_chars(
                text.data(), text.data() + text.size(),
                requested_width);
            LUISA_ASSERT(
                result.ec == std::errc{} &&
                    result.ptr == text.data() + text.size(),
                "Invalid LUISA_SIMD_WARP_WIDTH value '{}'.",
                text);
        }
    }
    if (requested_width != 0u) {
        LUISA_ASSERT(
            requested_width == 1u || requested_width == 2u ||
                requested_width == 4u || requested_width == 8u ||
                requested_width == 16u,
            "Invalid SIMD warp width {}. Expected 1, 2, 4, 8, or 16.",
            requested_width);
        _warp_width = requested_width;
    }
    // Diagnostic/benchmark override matching LUISA_SIMD_WARP_WIDTH. The
    // explicit backend extension remains authoritative so applications can
    // make worker-count selection deterministic without inheriting process
    // environment state.
    if (requested_worker_count == 0u) {
        if (auto *environment =
                std::getenv("LUISA_SIMD_WORKER_COUNT");
            environment != nullptr) {
            auto text = std::string_view{environment};
            auto result = std::from_chars(
                text.data(), text.data() + text.size(),
                requested_worker_count);
            LUISA_ASSERT(
                result.ec == std::errc{} &&
                    result.ptr == text.data() + text.size() &&
                    requested_worker_count != 0u,
                "Invalid LUISA_SIMD_WORKER_COUNT value '{}'.",
                text);
        }
    }
    _rtc_device = shared_embree_device();
    auto hardware_worker_count = static_cast<uint32_t>(
        std::max(std::thread::hardware_concurrency(), 1u));
    _thread_pool = luisa::make_unique<SIMDThreadPool>(
        requested_worker_count == 0u ?
            hardware_worker_count :
            requested_worker_count);
}

SIMDDevice::~SIMDDevice() noexcept = default;

void *SIMDDevice::native_handle() const noexcept {
    return const_cast<SIMDDevice *>(this);
}

uint SIMDDevice::compute_warp_size() const noexcept { return _warp_width; }

uint64_t SIMDDevice::memory_granularity() const noexcept { return 1u; }

BufferCreationInfo SIMDDevice::create_buffer(
    const Type *element, size_t elem_count,
    void *external_memory) noexcept {
    BufferCreationInfo info{};
    info.element_stride = element == Type::of<void>() ?
                              1u :
                              element->size();
    info.total_size_bytes = info.element_stride * elem_count;
    auto *buffer = external_memory == nullptr ?
                       luisa::new_with_allocator<SIMDBuffer>(info.total_size_bytes) :
                       luisa::new_with_allocator<SIMDBuffer>(
                           static_cast<std::byte *>(external_memory),
                           info.total_size_bytes);
    info.handle = reinterpret_cast<uint64_t>(buffer);
    info.native_handle = buffer->data();
    return info;
}

void SIMDDevice::destroy_buffer(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<SIMDBuffer *>(handle));
}

ResourceCreationInfo SIMDDevice::create_texture(
    PixelFormat format, uint dimension,
    uint width, uint height, uint depth, uint mipmap_levels,
    void *external_native_handle, bool, bool) noexcept {
    auto storage = pixel_format_to_storage(format);
    auto size = make_uint3(width, height, depth);
    auto *texture = external_native_handle == nullptr ?
                        luisa::new_with_allocator<SIMDTexture>(
                            storage, dimension, size, mipmap_levels) :
                        luisa::new_with_allocator<SIMDTexture>(
                            storage, dimension, size, mipmap_levels,
                            static_cast<std::byte *>(external_native_handle));
    return {
        .handle = reinterpret_cast<uint64_t>(texture),
        .native_handle = texture->native_handle(),
    };
}

void SIMDDevice::destroy_texture(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<SIMDTexture *>(handle));
}

ResourceCreationInfo SIMDDevice::create_bindless_array(
    size_t size, BindlessSlotType type) noexcept {
    auto *array = luisa::new_with_allocator<SIMDBindlessArray>(
        size, type);
    return {
        .handle = reinterpret_cast<uint64_t>(array),
        .native_handle = array->native_handle(),
    };
}

void SIMDDevice::destroy_bindless_array(uint64_t handle) noexcept {
    luisa::delete_with_allocator(
        reinterpret_cast<SIMDBindlessArray *>(handle));
}

ResourceCreationInfo SIMDDevice::create_stream(StreamTag) noexcept {
    auto *stream = luisa::new_with_allocator<SIMDStream>(
        _thread_pool.get());
    return {
        .handle = reinterpret_cast<uint64_t>(stream),
        .native_handle = stream->native_handle(),
    };
}

void SIMDDevice::destroy_stream(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<SIMDStream *>(handle));
}

void SIMDDevice::synchronize_stream(uint64_t stream_handle) noexcept {
    reinterpret_cast<SIMDStream *>(stream_handle)->synchronize();
}

void SIMDDevice::dispatch(
    uint64_t stream_handle, CommandList &&list) noexcept {
    reinterpret_cast<SIMDStream *>(stream_handle)->dispatch(std::move(list));
}

void SIMDDevice::set_stream_log_callback(
    uint64_t stream_handle,
    const StreamLogCallback &callback) noexcept {
    reinterpret_cast<SIMDStream *>(stream_handle)->set_log_callback(callback);
}

SwapchainCreationInfo SIMDDevice::create_swapchain(
    const SwapchainOption &, uint64_t) noexcept {
    SwapchainCreationInfo info{};
    info.invalidate();
    return info;
}

void SIMDDevice::destroy_swapchain(uint64_t) noexcept {}

void SIMDDevice::present_display_in_stream(
    uint64_t, uint64_t, uint64_t) noexcept {
    LUISA_ERROR_WITH_LOCATION(
        "The SIMD backend does not provide a display swapchain yet.");
}

ShaderCreationInfo SIMDDevice::create_shader(
    const ShaderOption &option, Function kernel) noexcept {
    Clock clock;
    auto block_size = kernel.block_size();
    auto *shader = luisa::new_with_allocator<SIMDShader>(
        option, kernel, _warp_width,
        _thread_pool->worker_count());
    LUISA_VERBOSE(
        "SIMD shader compilation took {} ms.", clock.toc());
    ShaderCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(shader);
    info.native_handle = reinterpret_cast<void *>(shader->native_handle());
    info.block_size = block_size;
    return info;
}

ShaderCreationInfo SIMDDevice::load_shader(
    luisa::string_view, luisa::span<const Type *const>) noexcept {
    return ShaderCreationInfo::make_invalid();
}

Usage SIMDDevice::shader_argument_usage(
    uint64_t handle, size_t index) noexcept {
    return reinterpret_cast<SIMDShader *>(handle)->argument_usage(index);
}

void SIMDDevice::destroy_shader(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<SIMDShader *>(handle));
}

ResourceCreationInfo SIMDDevice::create_event() noexcept {
    auto *event = luisa::new_with_allocator<SIMDEvent>();
    return {
        .handle = reinterpret_cast<uint64_t>(event),
        .native_handle = event->native_handle(),
    };
}

void SIMDDevice::destroy_event(uint64_t handle) noexcept {
    luisa::delete_with_allocator(reinterpret_cast<SIMDEvent *>(handle));
}

void SIMDDevice::signal_event(
    uint64_t handle, uint64_t, uint64_t fence_value) noexcept {
    reinterpret_cast<SIMDEvent *>(handle)->signal(fence_value);
}

void SIMDDevice::wait_event(
    uint64_t handle, uint64_t, uint64_t fence_value) noexcept {
    reinterpret_cast<SIMDEvent *>(handle)->wait(fence_value);
}

bool SIMDDevice::is_event_completed(
    uint64_t handle, uint64_t fence_value) const noexcept {
    return reinterpret_cast<SIMDEvent *>(handle)->is_completed(fence_value);
}

void SIMDDevice::synchronize_event(
    uint64_t handle, uint64_t fence_value) noexcept {
    reinterpret_cast<SIMDEvent *>(handle)->wait(fence_value);
}

ResourceCreationInfo SIMDDevice::create_mesh(
    const AccelOption &option) noexcept {
    auto *mesh = luisa::new_with_allocator<SIMDMesh>(
        _rtc_device, option);
    auto *primitive = static_cast<SIMDPrimitive *>(mesh);
    return {
        .handle = reinterpret_cast<uint64_t>(primitive),
        .native_handle = mesh->handle(),
    };
}

void SIMDDevice::destroy_mesh(uint64_t handle) noexcept {
    auto *primitive = reinterpret_cast<SIMDPrimitive *>(handle);
    LUISA_ASSERT(
        primitive != nullptr && primitive->kind() == SIMDPrimitive::Kind::mesh,
        "Invalid SIMD mesh handle.");
    luisa::delete_with_allocator(static_cast<SIMDMesh *>(primitive));
}

ResourceCreationInfo SIMDDevice::create_curve(
    const AccelOption &option) noexcept {
    auto *curve = luisa::new_with_allocator<SIMDCurve>(
        _rtc_device, option);
    auto *primitive = static_cast<SIMDPrimitive *>(curve);
    return {
        .handle = reinterpret_cast<uint64_t>(primitive),
        .native_handle = curve->handle(),
    };
}

void SIMDDevice::destroy_curve(uint64_t handle) noexcept {
    auto *primitive = reinterpret_cast<SIMDPrimitive *>(handle);
    LUISA_ASSERT(
        primitive != nullptr &&
            primitive->kind() == SIMDPrimitive::Kind::curve,
        "Invalid SIMD curve handle.");
    luisa::delete_with_allocator(static_cast<SIMDCurve *>(primitive));
}

ResourceCreationInfo SIMDDevice::create_procedural_primitive(
    const AccelOption &option) noexcept {
    auto *procedural = luisa::new_with_allocator<SIMDProceduralPrimitive>(
        _rtc_device, option);
    auto *primitive = static_cast<SIMDPrimitive *>(procedural);
    return {
        .handle = reinterpret_cast<uint64_t>(primitive),
        .native_handle = procedural->handle(),
    };
}

void SIMDDevice::destroy_procedural_primitive(uint64_t handle) noexcept {
    auto *primitive = reinterpret_cast<SIMDPrimitive *>(handle);
    LUISA_ASSERT(
        primitive != nullptr &&
            primitive->kind() == SIMDPrimitive::Kind::procedural,
        "Invalid SIMD procedural-primitive handle.");
    luisa::delete_with_allocator(
        static_cast<SIMDProceduralPrimitive *>(primitive));
}

ResourceCreationInfo SIMDDevice::create_motion_instance(
    const AccelMotionOption &option) noexcept {
    auto *instance = luisa::new_with_allocator<SIMDMotionInstance>(option);
    auto *primitive = static_cast<SIMDPrimitive *>(instance);
    return {
        .handle = reinterpret_cast<uint64_t>(primitive),
        .native_handle = instance,
    };
}

void SIMDDevice::destroy_motion_instance(uint64_t handle) noexcept {
    auto *primitive = reinterpret_cast<SIMDPrimitive *>(handle);
    LUISA_ASSERT(
        primitive != nullptr &&
            primitive->kind() == SIMDPrimitive::Kind::motion_instance,
        "Invalid SIMD motion-instance handle.");
    luisa::delete_with_allocator(
        static_cast<SIMDMotionInstance *>(primitive));
}

ResourceCreationInfo SIMDDevice::create_accel(
    const AccelOption &option) noexcept {
    auto *accel = luisa::new_with_allocator<SIMDAccel>(
        _rtc_device, option, _warp_width);
    return {
        .handle = reinterpret_cast<uint64_t>(accel),
        .native_handle = accel->native_handle(),
    };
}

void SIMDDevice::destroy_accel(uint64_t handle) noexcept {
    luisa::delete_with_allocator(
        reinterpret_cast<SIMDAccel *>(handle));
}

void SIMDDevice::set_name(
    Resource::Tag, uint64_t, luisa::string_view) noexcept {}

}// namespace luisa::compute::simd

LUISA_EXPORT_API luisa::compute::DeviceInterface *create(
    luisa::compute::Context &&context,
    const luisa::compute::DeviceConfig *config) noexcept {
    return luisa::new_with_allocator<luisa::compute::simd::SIMDDevice>(
        std::move(context), config);
}

LUISA_EXPORT_API void destroy(
    luisa::compute::DeviceInterface *device) noexcept {
    luisa::delete_with_allocator(device);
}

LUISA_EXPORT_API void backend_device_names(
    luisa::vector<luisa::string> &names) noexcept {
    names.clear();
    names.emplace_back(luisa::cpu_name());
}

#include "../../common/export_version.inl.h"
