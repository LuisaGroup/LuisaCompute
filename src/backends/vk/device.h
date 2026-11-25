#pragma once
#include <volk.h>
#include "vk_shader_untyped_pointers.h"
#include <luisa/runtime/device.h>
#include "VulkanDevice.h"
#include <luisa/vstl/common.h>
#include <luisa/core/first_fit.h>
#include "../common/default_binary_io.h"
#include "../common/command_reorder_switch.h"
#include <luisa/backends/ext/command_reorder_ext.h>
#include "vk_allocator.h"
#include "sparse_residency_registry.h"
#include <luisa/backends/ext/vk_config_ext.h>
#include <luisa/backends/ext/vk_extension_ext.h>
#include "vk_extension_registry.h"
#include "layout_intern.h"
#include "pipeline_cache.h"
#include "in_flight_depth.h"
#include "descriptor_update_template.h"
#include "push_descriptor.h"
#include "device_capability.h"
#include "replay_graph.h"
#include "vulkan_replay_ext.h"
#include "shader_prepare.h"
#include "shader_compile_pool.h"
#include "vulkan_shader_compile_ext.h"
#include "../common/rtx/fallback_rtx.h"
#include <atomic>
#include <limits>
namespace lc::hlsl {
class ShaderCompiler;
}// namespace lc::hlsl
namespace lc::vk {
static constexpr uint kShaderModel = 65u;
static constexpr uint kHighShaderModel = 66u;
static constexpr uint kTensorShaderModel = 69u;
class ComputeShader;
class Stream;
class UploadBuffer;
class Texture;
struct NativeImageState;
using namespace luisa;
using namespace luisa::compute;
static constexpr size_t kSparseBufferSize = 65536ull;
class Event;

/// A stream-like object owned by a device *extension* rather than by the
/// backend's built-in `Stream`.
///
/// `Device::create_stream` rejects anything that is not GRAPHICS/COMPUTE/COPY
/// (see `stream.cpp`), so an extension that needs its own command stream (e.g.
/// `DStorageExt`) allocates a `CustomStream`, registers it through
/// `Device::add_custom_stream`, and lets the device route
/// `dispatch` / `synchronize_stream` / `destroy_stream` / `signal_event` /
/// `wait_event` / `set_stream_log_callback` to it.  Handles that are not in the
/// registry keep the pre-existing `reinterpret_cast<Stream *>` behavior, so
/// normal streams are completely unaffected.
class CustomStream {
public:
    virtual ~CustomStream() = default;
    virtual void dispatch(CommandList &&list) noexcept = 0;
    virtual void synchronize() noexcept = 0;
    virtual void signal(Event *event, uint64_t fence) noexcept = 0;
    virtual void wait(Event *event, uint64_t fence) noexcept = 0;
    virtual void set_log_callback(const DeviceInterface::StreamLogCallback &callback) noexcept {
        static_cast<void>(callback);
    }
};

class Device : public DeviceInterface, public vstd::IOperatorNewBase {
    friend class Texture;
    // Volk uses process-global instance/device dispatch tables. Keep one live
    // backend Device at a time so constructing another device cannot replace
    // entry points while the first device is still executing.
    struct GlobalDispatchLease {
        GlobalDispatchLease() noexcept;
        ~GlobalDispatchLease() noexcept;
        GlobalDispatchLease(GlobalDispatchLease const &) = delete;
        GlobalDispatchLease(GlobalDispatchLease &&) = delete;
        GlobalDispatchLease &operator=(GlobalDispatchLease const &) = delete;
        GlobalDispatchLease &operator=(GlobalDispatchLease &&) = delete;
    } _global_dispatch_lease;
    struct Ext {
        using Ctor = vstd::func_ptr_t<DeviceExtension *(Device *)>;
        using Dtor = vstd::func_ptr_t<void(DeviceExtension *)>;
        DeviceExtension *ext;
        Ctor ctor;
        Dtor dtor;
        Ext(Ctor ctor, Dtor dtor) : ext{nullptr}, ctor{ctor}, dtor{dtor} {}
        Ext(Ext const &) = delete;
        Ext(Ext &&rhs) : ext{rhs.ext}, ctor{rhs.ctor}, dtor{rhs.dtor} {
            rhs.ext = nullptr;
        }
        ~Ext() {
            if (ext) {
                dtor(ext);
            }
        }
    };
    luisa::spin_mutex _graphics_queue_mtx;
    luisa::spin_mutex _compute_queue_mtx;
    luisa::spin_mutex _copy_queue_mtx;
    luisa::spin_mutex _sparse_queue_mtx;
    luisa::spin_mutex *_graphics_queue_lock{&_graphics_queue_mtx};
    luisa::spin_mutex *_compute_queue_lock{&_compute_queue_mtx};
    luisa::spin_mutex *_copy_queue_lock{&_copy_queue_mtx};
    luisa::spin_mutex *_sparse_queue_lock{&_sparse_queue_mtx};
    luisa::spin_mutex _stream_mtx;
    std::mutex _native_image_state_mtx;
    vstd::unordered_map<uint64_t, luisa::weak_ptr<NativeImageState>>
        _native_image_states;
    luisa::shared_ptr<std::atomic_size_t>
        _native_image_state_expiration_counter{
            luisa::make_shared<std::atomic_size_t>(0u)};
    size_t _native_image_state_acquisitions_since_sweep{};
    detail::SparseResidencyRegistry _sparse_residency_registry;
    // Replayable recording (plan P4). Declared next to the sparse registry, i.e.
    // late among the device-scoped registries, so it is destroyed after the
    // logical device member `_vk_device` - and `~Device` drains it explicitly while
    // the logical device, the descriptor pools, the interned layouts and the
    // pipeline cache are all still alive.
    detail::ReplayRegistry _replay_registry;
    // The engine behind `Device::extension<VulkanReplayExt>()`. It holds no state
    // of its own: everything lives in `_replay_registry`, so a caller that never
    // asks for the extension still gets correct invalidation and teardown. Held by
    // value (like `_command_reorder_ext`) because it has no per-device allocation:
    // `extension()` can hand out its address without a factory.
    VkReplayExtImpl _replay_ext{this};
    // Plan P5.1: the device-owned worker pool behind
    // `VulkanShaderCompileExt::compile_shaders()`. Constructed by the constructor -
    // the only place that can capture the executor, because the executor is "which
    // `marl::Scheduler` is bound to *this* thread" - but it owns nothing (no thread,
    // no lane) until a batch needs it. Stopped for good by
    // `shutdown_shader_compile_pool()` and by `~Device`, which does it while the
    // pipeline cache, the interned layouts, the BinaryIO and the logical device are
    // all still alive.
    //
    // A captured scheduler is *external*: it must outlive this device and every
    // batch, since the pool enqueues each batch's lane tasks on it directly.
    //
    // Declared *after* `_vk_device`, `_layout_interner` and `_pipeline_cache`: the
    // pool only holds lanes and host scratch, and every Vulkan call it can cause is
    // made by the publishing thread through `_publish_compute_shader`, which the batch
    // finishes before it returns. Stopping it early in `~Device` is therefore enough,
    // and is done explicitly there.
    luisa::unique_ptr<detail::ShaderCompilePool> _shader_compile_pool;
    // Serializes every shader-cache read/write of this device. A `BinaryIO` is one
    // resource that a user implementation need not make thread-safe, and one writer
    // keeps the artifact sequence deterministic. It is never held over code
    // generation, SPIR-V optimization or pipeline creation.
    //
    // Recursive only for the documented nested-compilation fallback: a BinaryIO
    // read runs on a preparation lane under this lock, and that callback may start
    // a nested batch. The nested batch must compile serially on the same flow (the
    // outer batch gate is already held), which includes reading/writing this cache.
    // A plain mutex turns that supported fallback into a self-deadlock (and an MSVC
    // resource-deadlock fail-fast). Recursion does not relax cross-thread exclusion:
    // distinct callers are still serialized exactly as before.
    std::recursive_mutex _shader_cache_mtx;
    VkShaderCompileExtImpl _shader_compile_ext{this};
    // Runtime command-reorder control. It lives as long as the device does, so
    // `extension()` hands out its address without registering a factory.
    class CommandReorderExtImpl final : public CommandReorderExt {
        Device *_device;

    public:
        explicit CommandReorderExtImpl(Device *device) noexcept : _device{device} {}
        [[nodiscard]] bool command_reorder_enabled() const noexcept override {
            return _device->command_reorder_enabled();
        }
        void set_command_reorder_enabled(bool enabled) noexcept override {
            _device->_command_reorder.set_enabled(enabled);
        }
    };
    CommandReorderSwitch _command_reorder;
    CommandReorderExtImpl _command_reorder_ext{this};
    std::mutex _ext_mtx;
    vstd::unordered_set<Stream *> _streams;
    // Extension-owned command streams (see `CustomStream`).  The registry never
    // owns the objects: `destroy_stream` removes and deletes them, and ~Device
    // deletes any leftover so a misbehaving extension cannot leak them.
    std::mutex _custom_stream_mtx;
    vstd::unordered_map<uint64_t, CustomStream *> _custom_streams;
    luisa::unique_ptr<VulkanDeviceConfigExt> _config_ext;
    VkInstance _instance{};
    vstd::optional<vks::VulkanDevice> _vk_device;
    // Structural interning of the per-shader descriptor-set and pipeline layouts
    // (plan P1.1). Declared *after* `_vk_device` so its own destruction happens
    // first; the handles it owns are destroyed by `~Device` explicitly, while the
    // logical device is still alive.
    detail::LayoutInterner _layout_interner;
    // The one VkPipelineCache of this Device (plan P1.3). Declared next to the
    // interner, i.e. after `_vk_device`, and torn down explicitly by `~Device`
    // while the logical device and the BinaryIO are still alive. Shaders only
    // borrow the handle; nothing else destroys it.
    detail::DevicePipelineCache _pipeline_cache;
    // Declared *after* `_vk_device` and therefore destroyed *before* it:
    // extensions may own GPU objects (shader instances, pipelines, descriptor
    // pools, ...) whose destructors need a live logical device. Destroying the
    // extension map first keeps that ordering safe.
    vstd::unordered_map<vstd::string, Ext> _exts;
    vstd::vector<vstd::string> _enable_device_exts;
    // Negotiated state of every optional device extension. The registry only
    // holds plain values and function pointers, so destruction order relative
    // to the logical device does not matter.
    detail::OptionalExtensionRegistry _optional_extensions;
    // Bitmask of optional extensions the backend actually routes work through
    // (see `mark_optional_extension_active`). Atomic: streams record command
    // lists concurrently and each may mark a fast path on first use.
    std::atomic<uint64_t> _active_optional_extensions{0u};
    VkQueue _graphics_queue{};
    VkQueue _compute_queue{};
    VkQueue _copy_queue{};
    VkQueue _sparse_queue{};
    VkDescriptorPool _sampler_pool{};
    VkDescriptorSet _sampler_set{};
    VkDescriptorSetLayout _sampler_set_layout{};
    VkDescriptorPool _bdls_buffer_desc_pool{};
    VkDescriptorSet _bdls_buffer_set{};
    VkDescriptorSetLayout _bdls_buffer_set_layout{};
    VkDescriptorPool _bdls_tex2d_desc_pool{};
    VkDescriptorSet _bdls_tex2d_set{};
    VkDescriptorSetLayout _bdls_tex2d_set_layout{};
    VkDescriptorPool _bdls_tex3d_desc_pool{};
    VkDescriptorSet _bdls_tex3d_set{};
    VkDescriptorSetLayout _bdls_tex3d_set_layout{};
    uint32_t _bindless_heap_capacity{};
    VkPhysicalDeviceDescriptorIndexingProperties
        _descriptor_indexing_properties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES};
    VkPhysicalDeviceTimelineSemaphoreProperties
        _timeline_semaphore_properties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_PROPERTIES};
    VkPhysicalDeviceAccelerationStructurePropertiesKHR
        _acceleration_structure_properties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    // VK_NV_cluster_acceleration_structure limits and byte alignments. Only
    // meaningful when `cluster_acceleration_structure_enabled` is set; the
    // struct keeps its sType otherwise so it can be chained unconditionally.
    VkPhysicalDeviceClusterAccelerationStructurePropertiesNV
        _cluster_acceleration_structure_properties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV};
    VkPipelineCacheHeaderVersionOne _pso_header{};
    VkPhysicalDeviceSubgroupSizeControlProperties _subgroup_size_control_properties{};
    // Plan P6.1/P6.2/P6.3: the one numeric capability record of this device. It
    // is filled once in `_init_device` from this physical device's own
    // VkPhysicalDeviceLimits, VkPhysicalDeviceSubgroupProperties and
    // VkPhysicalDeviceSubgroupSizeControlProperties (plus the calibrated SM-count
    // input latched in the constructor), and it never changes afterwards. Both
    // `Device::query` and `VulkanExtensionStatusExt::capability_status()` answer
    // from this copy, so a query answer and a status report cannot drift.
    //
    // It is deliberately separate from `_subgroup_size_control_properties` above,
    // which is the *policy* input (only filled when subgroup-size control is
    // actually enabled) that decides the shader warp size. Reporting a physical
    // bound must not touch the warp-size policy.
    VulkanCapabilityStatus _capability_status{};
    vstd::vector<VkSampler> _samplers;
    float _max_sampler_anisotropy{1.0f};
    vstd::optional<VkAllocator> _allocator;
    // Native SPIR-V statically declares the indirect metadata descriptor even
    // though direct execution never enters its load arm. Bind one persistent,
    // zero-filled storage buffer instead of allocating/copying a dummy slice
    // for every direct dispatch.
    luisa::unique_ptr<UploadBuffer> _indirect_dispatch_dummy;
    BinaryIO const *_binary_io{};
    vstd::unique_ptr<DefaultBinaryIO> _default_file_io;
    bool _inqueue_limit = true;// TODO
    // Number of command batches a stream may keep in flight before it throttles
    // the host (plan P1.5). Latched once at creation from
    // `VulkanDeviceConfigExt::max_frames_in_flight()` (default
    // `detail::kDefaultFramesInFlight`), validated there, and read back by every
    // stream per submission through `max_frames_in_flight()`. Nothing mutates it
    // afterwards, so two batches of one stream can never disagree about the
    // bound. `DeviceConfig::inqueue_buffer_limit` remains the enable/disable
    // switch and takes precedence over this value.
    uint32_t _max_frames_in_flight{detail::kDefaultFramesInFlight};
    // Phase 3B: monotonic counter of destroyed interned descriptor-set layouts.
    // An interned layout is reference counted by `detail::LayoutInterner` and
    // destroyed when its last user releases it, so a `VkDescriptorSet` allocated
    // from it stops being usable at that moment even though the pool it came from
    // is still alive. `CommandBufferState` therefore caches sets together with the
    // generation it observed; a mismatch after a recycle invalidates the cache
    // (pools are reset, sets are re-allocated) instead of handing out a set whose
    // layout no longer exists. Bumped on the thread that destroys the layout,
    // which is always the thread that released the last reference (a shader
    // destruction), so a relaxed increment is sufficient: the reader pairs it with
    // acquire semantics before reusing pooled sets, and a stale value can only
    // cause an extra pool reset, never a stale set.
    std::atomic_uint64_t _shader_layout_destroy_generation{0u};
    // Phase 3B (opt A): the ids of the interned descriptor-set layouts that were
    // destroyed, in destruction order. Index `i` of this log corresponds to
    // generation `i + 1`, so a `CommandBufferState` that remembers the generation it
    // observed can ask for exactly the ids that were destroyed since then and drop
    // only the descriptor-set buckets that really became invalid, instead of
    // flushing every bucket of the device on any shader destruction. The log is
    // append-only and bounded by the number of destroyed layouts (one per destroyed
    // shader layout), which is a compile-rate quantity, not a per-frame one.
    mutable luisa::spin_mutex _destroyed_layout_ids_mtx;
    mutable vstd::vector<uint64_t> _destroyed_layout_ids;
    struct NumericFeatures {
        bool shader_float8{false};
        bool shader_bfloat16{false};
        bool shader_float16{false};
        bool shader_float64{false};
        bool shader_int8{false};
        bool shader_int16{false};
        bool shader_int64{false};
        bool storage_buffer_8bit_access{false};
        bool uniform_storage_buffer_8bit_access{false};
        bool storage_buffer_16bit_access{false};
        bool uniform_storage_buffer_16bit_access{false};
    } _numeric_features;
    // Exact extension feature bits requested on an internally created logical
    // device. Imported devices remain conservative because Vulkan exposes no
    // query for their enabled feature chain.
    struct FloatAtomicFeatures {
        bool shader_buffer_float32_atomics{false};
        bool shader_buffer_float32_atomic_add{false};
        bool shader_buffer_float32_atomic_min_max{false};
        bool shader_shared_float32_atomics{false};
        bool shader_shared_float32_atomic_add{false};
        bool shader_shared_float32_atomic_min_max{false};
    } _float_atomic_features;
    struct Int64AtomicFeatures {
        bool shader_buffer_int64_atomics{false};
        bool shader_shared_int64_atomics{false};
    } _int64_atomic_features;
    bool _shader_device_clock_enabled{false};

    // VkPhysicalDeviceVulkan12Properties::shaderDenormPreserveFloat32 and
    // shaderDenormFlushToZeroFloat32. Both are VK_FALSE on a driver without
    // 32-bit denormal control, and declaring the matching SPIR-V execution mode
    // without the property violates VUID-VkShaderModuleCreateInfo-pCode-08740.
    bool _denorm_preserve_float32{false};
    bool _denorm_flush_float32{false};
    void _init_device(VkPhysicalDevice external_physical_device, VkDevice external_device, uint32_t selected_device);
    [[nodiscard]] luisa::shared_ptr<NativeImageState>
    acquire_native_image_state(
        VkImage image, VkFormat format, uint dimension, uint3 size,
        uint mip_levels, bool simultaneous_access);
    [[nodiscard]] ShaderCreationInfo _create_shader_hlsl(
        const ShaderOption &option, Function kernel,
        bool requires_sampler_anisotropy,
        bool fallback_rtx) noexcept;
    // Plan P5.1: the two phases of compute-shader compilation. `_prepare` creates
    // no Vulkan object and may run on any thread (including a pool worker);
    // `_publish` is the serial, ordered phase that creates the pipeline. A serial
    // `create_shader` is exactly `_publish(_prepare(...))`.
    [[nodiscard]] detail::PreparedComputeShader _prepare_compute_shader(
        const ShaderOption &option, Function kernel,
        detail::WorkerContext *worker) noexcept;
    [[nodiscard]] ShaderCreationInfo _publish_compute_shader(
        detail::PreparedComputeShader &&prepared,
        const ShaderOption &option, Function kernel) noexcept;
    // The unchanged serial routes (compatibility HLSL-to-SPIR-V, LLVM-to-SPIR-V,
    // and any build without the native XIR-to-SPIR-V path).
    [[nodiscard]] ShaderCreationInfo _create_shader_serial(
        const ShaderOption &option, Function kernel,
        bool requires_sampler_anisotropy,
        bool requires_fallback_rtx_traversal) noexcept;
    friend class detail::ShaderCompilePool;
    // Read-only view over `_optional_extensions`, handed out by `extension()`.
    class VulkanExtensionStatusExtImpl final : public VulkanExtensionStatusExt {
    private:
        Device *_device;
        // Built lazily on first query: the negotiation is immutable after
        // device creation, so one build is enough and the returned span stays
        // valid for the lifetime of the device.
        luisa::vector<VulkanExtensionStatus> _cache;

        void build() noexcept;

    public:
        explicit VulkanExtensionStatusExtImpl(Device *device) noexcept : _device{device} {}
        [[nodiscard]] luisa::span<const VulkanExtensionStatus> extensions() const noexcept override;
        [[nodiscard]] bool is_supported(luisa::string_view extension) const noexcept override;
        [[nodiscard]] bool is_enabled(luisa::string_view extension) const noexcept override;
        [[nodiscard]] bool is_active(luisa::string_view extension) const noexcept override;
        // Cooperative matrix/vector state: negotiated outside the registry, so
        // it reads the Device members recorded during negotiation instead of the
        // optional-extension table.
        [[nodiscard]] VulkanCooperativeStatus cooperative_status() const noexcept override;
        // Plan P6.3: one value struct carrying the P6.1 core limits, the
        // descriptor-heap negotiation state of the registry entry and the P6.2
        // SM-count answer. The limits and the SM-count fields are a copy of the
        // device's single capability record (the same one `Device::query`
        // answers from, so the two cannot drift); the descriptor-heap state is
        // read from `_optional_extensions` / `_active_optional_extensions` here,
        // exactly like `extensions()` does.
        [[nodiscard]] VulkanCapabilityStatus capability_status() const noexcept override;
    };
    VulkanExtensionStatusExtImpl _optional_extension_status{this};

public:
    // Optional (non-required) Vulkan device extensions, negotiated once at
    // device creation from the physical device's extension list and the
    // VulkanDeviceConfigExt switches. Every one of them has a fallback path,
    // so `false` never breaks the backend.
    [[nodiscard]] bool optional_extension_supported(detail::OptionalExtension ext) const noexcept {
        return _optional_extensions.supported(ext);
    }
    [[nodiscard]] bool optional_extension_enabled(detail::OptionalExtension ext) const noexcept {
        return _optional_extensions.enabled(ext);
    }
    [[nodiscard]] bool optional_extension_active(detail::OptionalExtension ext) const noexcept {
        return (_active_optional_extensions.load(std::memory_order_relaxed) &
                (uint64_t{1u} << static_cast<uint32_t>(ext))) != 0u;
    }
    // Records that the backend really takes this extension's fast path, so the
    // status extension can distinguish "may use" from "does use". Safe to call
    // from any recording thread; never clears a bit.
    void mark_optional_extension_active(detail::OptionalExtension ext) noexcept {
        _active_optional_extensions.fetch_or(
            uint64_t{1u} << static_cast<uint32_t>(ext),
            std::memory_order_relaxed);
    }
    // Instance extensions this backend actually enabled on the instance it
    // created (surface/platform, VK_KHR_get_surface_capabilities2,
    // VK_KHR_surface_maintenance1). Vulkan cannot report them back, so the
    // swapchain attests its optional present extensions against this list.
    // Always empty for a borrowed instance.
    [[nodiscard]] luisa::span<const vstd::string> enabled_instance_extensions() const noexcept;
    // Returned by value on purpose: `states()` hands out a temporary span, and
    // binding that temporary to a reference return type dangles (MSVC C4172,
    // clang -Wreturn-stack-address). Callers either copy the span or iterate it
    // immediately.
    [[nodiscard]] luisa::span<const detail::OptionalExtensionState>
    optional_extensions() const noexcept {
        return _optional_extensions.states();
    }

public:
    struct HeapAlloc {
        uint count = 0;
        vstd::vector<uint> release_pool;
        vstd::vector<luisa::FirstFit::Node *> sub_allocations;
        luisa::spin_mutex mtx;
        luisa::FirstFit sub_allocator;
        uint full_size{};
        uint alloc();
        void dealloc(uint idx);
        luisa::FirstFit::Node *sub_alloc(uint32_t size);
        void free(luisa::FirstFit::Node *ptr);
        uint get_index(luisa::FirstFit::Node const *ptr) const;

        HeapAlloc();
        ~HeapAlloc();
    };

    struct LazyLoadShader {
    public:
        using LoadFunc = vstd::func_ptr_t<ComputeShader *(Device *)>;

    private:
        vstd::unique_ptr<ComputeShader> _shader;
        LoadFunc _load_func;

    public:
    explicit LazyLoadShader(LoadFunc load_func);
    ComputeShader *get(Device *self);
    bool check(Device *self);
    void reset() noexcept;
    ~LazyLoadShader();
    };
    vstd::vector<VkImageView> tex2d_bindless_imgview;
    vstd::vector<VkImageView> tex3d_bindless_imgview;
    HeapAlloc tex2d_heap_pool;
    HeapAlloc tex3d_heap_pool;
    HeapAlloc buffer_heap_pool;
    LazyLoadShader set_bindless_kernel;
    LazyLoadShader set_accel_kernel;
    LazyLoadShader prepare_indirect_kernel;
    bool external_instance : 1 {false};
    bool external_device : 1 {false};
    // Capability-adaptive Vulkan API floor. The backend prefers core
    // synchronization2/copy_commands2 (Vulkan 1.3) but falls back to the
    // classic vkCmdPipelineBarrier/vkCmdCopyBuffer entry points on Vulkan 1.2
    // devices (and on 1.3 devices when the minimum API version is forced to
    // 1.2 for testing).
    bool sync2_capable_bit : 1 {false};
    bool copy2_capable_bit : 1 {false};
    // Plan P2.2: per-dispatch descriptor-set-0 updates through a cached
    // VkDescriptorUpdateTemplate (VK_KHR_descriptor_update_template, core
    // Vulkan 1.2). Decided once in `_init_device`: the extension must have been
    // negotiated *and* enabled (or be usable through the device's API version),
    // the config switch must allow it, and all three entry points must have been
    // resolved. When any of that is missing, every update keeps the pre-P2.2
    // vkUpdateDescriptorSets call and the three template counters stay zero.
    bool descriptor_update_template_capable_bit : 1 {false};
    // The entry points a template update needs, resolved at device creation
    // (core spelling preferred, KHR alias otherwise - the same rule
    // `cmd_pipeline_barrier` and `detail::queue_submit` follow).
    PFN_vkCreateDescriptorUpdateTemplate _create_descriptor_update_template{nullptr};
    PFN_vkDestroyDescriptorUpdateTemplate _destroy_descriptor_update_template{nullptr};
    PFN_vkUpdateDescriptorSetWithTemplate _update_descriptor_set_with_template{nullptr};
    // Plan P2.3: set-0 binding through vkCmdPushDescriptorSetKHR. Decided once in
    // `_init_device`: the registry must have negotiated *and enabled*
    // VK_KHR_push_descriptor (by name - see OptionalExtension::PUSH_DESCRIPTOR),
    // the config switch must allow it, the loader must have resolved the entry
    // point, and VkPhysicalDevicePushDescriptorPropertiesKHR::maxPushDescriptors
    // must be usable. When any of that is missing, every shader keeps the classic
    // set-0 layout and the classic allocate/update/bind sequence.
    bool push_descriptor_capable_bit : 1 {false};
    PFN_vkCmdPushDescriptorSetKHR _cmd_push_descriptor_set{nullptr};
    // Resolved limit of one push (VUID-VkDescriptorSetLayoutCreateInfo-flags-00281);
    // 0 whenever the push path is unavailable.
    uint32_t _max_push_descriptors{0u};
    // Minimum API version to honor during capability negotiation. 0 means
    // "auto" (use whatever the physical device exposes). A non-zero value
    // (e.g. VK_API_VERSION_1_2) forces the legacy barrier/copy paths even on
    // 1.3 devices, which is useful for CI validation of the 1.2 floor.
    uint32_t _min_api_version{0u};
    uint32_t external_graphics_queue_family_index{VK_QUEUE_FAMILY_IGNORED};
    uint32_t external_compute_queue_family_index{VK_QUEUE_FAMILY_IGNORED};
    uint32_t external_copy_queue_family_index{VK_QUEUE_FAMILY_IGNORED};
    bool bindless_enabled : 1 {true};
    bool raytracing_enabled : 1 {true};
    bool surface_enabled : 1 {true};
    bool device_address_enabled : 1 {true};
    bool interop_enabled : 1 {true};
    // VK_NV_cuda_kernel_launch (compiled only with LUISA_VULKAN_ENABLE_CUDA_INTEROP;
    // owned logical devices only — imported devices cannot be queried).
    bool cuda_kernel_launch_enabled : 1 {false};
    bool motion_blur_enabled : 1 {false};
    // VK_NV_cluster_acceleration_structure (RTX Mega Geometry). Decided once in
    // _init_device from the physical-device extension + feature bit, the
    // ray-query decision it depends on and the software-fallback choice; forced
    // off for an imported logical device whose enabled feature chain cannot be
    // attested.
    bool cluster_acceleration_structure_enabled : 1 {false};
    // Software ray tracing (`lc::fallback_rtx::FallbackRtxDevice`). Decided
    // once, in `_init_device`: the user may force it through
    // `VulkanDeviceConfigExt::use_fallback_rtx()`, and a physical device without
    // the extensions/feature bits the hardware ray-query path needs has no
    // other one. When the bit is clear no fallback object is ever created, no
    // buffer is allocated, no descriptor changes and no generated source
    // changes, so the hardware path stays byte-for-byte what it was.
    bool use_fallback_rtx_bit : 1 {false};
    mutable std::mutex _fallback_rtx_mutex;
    mutable luisa::unique_ptr<lc::fallback_rtx::FallbackRtxDevice> _fallback_rtx;
    bool subgroup_size_control_enabled : 1 {false};
    bool subgroup_extended_types_enabled : 1 {false};
    bool cooperative_vector_enabled : 1 {false};
    bool cooperative_vector_fp32_enabled : 1 {false};
    // VK_NV_cooperative_vector negotiation details. `cooperative_vector_enabled`
    // alone cannot answer "will a cooperative-vector instruction actually run":
    // SPV_NV_cooperative_vector has two prerequisites (the Vulkan memory model and
    // SPV_EXT_replicated_composites), and the single configuration the codegen can
    // emit is FLOAT32/FLOAT32/FLOAT32/FLOAT32 with transpose = false. These bits
    // record each of those, so VulkanExtensionStatusExt::cooperative_status() can
    // distinguish "enabled" from "usable" instead of collapsing both into one bool.
    bool cooperative_vector_physical_support : 1 {false};
    bool cooperative_vector_memory_model : 1 {false};
    bool cooperative_vector_replicated_composites : 1 {false};
    // VK_EXT_shader_replicated_composites, negotiated independently of
    // VK_NV_cooperative_vector. "Enabled" means the extension name and the
    // shaderReplicatedComposites feature bit were actually requested on the
    // logical device, so the native codegen may lower exact splat composites to
    // replicated instructions. Imported logical devices cannot attest their
    // enabled feature chain and therefore keep this false.
    bool shader_replicated_composites_enabled : 1 {false};
    // Number of VkCooperativeVectorPropertiesNV entries the driver enumerated,
    // including the packed/quantized ones the Info log's sanity guard hides.
    uint32_t cooperative_vector_config_count{0u};
    // Cooperative matrix: the extension name the backend passed to vkCreateDevice
    // (nullptr when none was enabled) and whether VK_NV_cooperative_matrix2's
    // workgroup scope feature was enabled on top of VK_KHR_cooperative_matrix.
    // "Enabled" never implies "emitted": nothing in this tree lowers to a
    // cooperative-matrix instruction yet.
    const char *cooperative_matrix_extension{nullptr};
    bool cooperative_matrix_workgroup_scope : 1 {false};
    bool shader_untyped_pointers_enabled : 1 {false};
    bool async_copy_enabled : 1 {false};
    bool sampler_anisotropy_enabled : 1 {false};
    auto &graphics_queue_mtx() { return *_graphics_queue_lock; }
    auto &compute_queue_mtx() { return *_compute_queue_lock; }
    auto &copy_queue_mtx() { return *_copy_queue_lock; }
    auto &sparse_queue_mtx() { return *_sparse_queue_lock; }
    VulkanDeviceConfigExt *config_ext() const { return _config_ext.get(); }
    // Command-reorder switch sampled when a batch starts (see
    // CommandReorderSwitch for the precedence of config extension, runtime
    // override and LUISA_DISABLE_COMMAND_REORDER). Streams forward it to their
    // CommandReorderVisitor once per batch.
    [[nodiscard]] bool command_reorder_enabled() const noexcept {
        return _command_reorder.enabled();
    }
    [[nodiscard]] CommandReorderSwitch &command_reorder_switch() noexcept {
        return _command_reorder;
    }
    auto binary_io() const { return _binary_io; }
    auto sampler_set() const { return _sampler_set; }
    auto bdls_buffer_set() const { return _bdls_buffer_set; }
    auto bdls_tex2d_set() const { return _bdls_tex2d_set; }
    auto bdls_tex3d_set() const { return _bdls_tex3d_set; }
    auto samplers() const { return luisa::span{_samplers}; }
    bool enable_surface_feature() const { return surface_enabled; }
    bool enable_bindless() const { return bindless_enabled; }
    [[nodiscard]] auto bindless_heap_capacity() const noexcept {
        return _bindless_heap_capacity;
    }
    [[nodiscard]] const auto &descriptor_indexing_properties() const noexcept {
        return _descriptor_indexing_properties;
    }
    // In-flight depth latched at creation (see `_max_frames_in_flight` and
    // VulkanDeviceConfigExt::max_frames_in_flight()). Always >= 1; the configure
    // path fails closed on zero and on a value that does not fit the physical
    // device's timeline value window.
    [[nodiscard]] uint32_t max_frames_in_flight() const noexcept {
        return _max_frames_in_flight;
    }
    // Phase 3B: a `VkDescriptorSetLayout` created by the interner was destroyed.
    // Callers that cache `VkDescriptorSet`s keyed by `Shader::desc_set_layout_id`
    // must drop them (see the member's comment).
    void note_shader_layout_destroyed(uint64_t layout_id) noexcept {
        {
            std::lock_guard lock{_destroyed_layout_ids_mtx};
            _destroyed_layout_ids.emplace_back(layout_id);
        }
        _shader_layout_destroy_generation.fetch_add(
            1u, std::memory_order_release);
    }
    // The ids destroyed since `generation` (which is the value a caller read from
    // `shader_layout_destroy_generation()`), appended to `out`.
    void collect_destroyed_layout_ids(uint64_t generation,
                                      vstd::vector<uint64_t> &out) const {
        std::lock_guard lock{_destroyed_layout_ids_mtx};
        for (auto i = static_cast<size_t>(generation);
             i < _destroyed_layout_ids.size(); ++i) {
            out.emplace_back(_destroyed_layout_ids[i]);
        }
    }
    [[nodiscard]] uint64_t shader_layout_destroy_generation() const noexcept {
        return _shader_layout_destroy_generation.load(
            std::memory_order_acquire);
    }
    [[nodiscard]] uint64_t max_timeline_semaphore_value_difference() const noexcept {
        // Per the Vulkan spec, maxTimelineSemaphoreValueDifference == 0 means
        // "no limit". Normalize once at the Device boundary so every caller
        // sees an unbounded window.
        auto raw = _timeline_semaphore_properties
                       .maxTimelineSemaphoreValueDifference;
        return raw == 0u ? std::numeric_limits<uint64_t>::max() : raw;
    }
    [[nodiscard]] const auto &acceleration_structure_properties() const noexcept {
        return _acceleration_structure_properties;
    }
    bool enable_interop() const { return interop_enabled; }
    // Whether this device compiles the native BF16 type.  Without
    // VK_KHR_shader_bfloat16 (`shaderBFloat16Type`) the native XIR-to-SPIR-V
    // route cannot express BF16 at all, so BF16 kernels take the compatibility
    // HLSL route with the software emulation of
    // src/backends/common/hlsl/builtin/bfloat16_fallback.bytes.
    [[nodiscard]] bool supports_native_bfloat16() const noexcept {
        return _numeric_features.shader_bfloat16;
    }
    bool enable_motion_blur() const { return motion_blur_enabled; }
    bool enable_raytracing() const { return raytracing_enabled; }
    // Whether this device can build and trace cluster (mega-geometry)
    // acceleration structures through VK_NV_cluster_acceleration_structure.
    [[nodiscard]] bool enable_cluster_acceleration_structure() const noexcept {
        return cluster_acceleration_structure_enabled;
    }
    [[nodiscard]] const auto &cluster_acceleration_structure_properties() const noexcept {
        return _cluster_acceleration_structure_properties;
    }
    // Whether this device answers ray tracing with the software fallback. It is
    // `false` on the hardware path, where every fallback branch is inert.
    [[nodiscard]] bool use_fallback_rtx() const noexcept {
        return use_fallback_rtx_bit;
    }
    // The fallback device, or `nullptr` when the hardware path is in use. It is
    // created on the first call and lives as long as this device does; the
    // object is bound to this `DeviceInterface`, so it must not outlive it.
    [[nodiscard]] lc::fallback_rtx::FallbackRtxDevice *fallback_rtx() noexcept;
    // Whether an acceleration-structure handle has to be routed into the
    // fallback instead of the backend's native BLAS/TLAS path. Both are
    // `false` whenever the fallback is not in use.
    [[nodiscard]] bool owns_fallback_blas(uint64_t handle) noexcept {
        auto *fallback = use_fallback_rtx() ? fallback_rtx() : nullptr;
        return fallback != nullptr && fallback->owns_blas(handle);
    }
    [[nodiscard]] bool owns_fallback_accel(uint64_t handle) noexcept {
        auto *fallback = use_fallback_rtx() ? fallback_rtx() : nullptr;
        return fallback != nullptr && fallback->owns_accel(handle);
    }
    bool enable_device_address() const { return device_address_enabled; }
    // True when the shaders this device compiles declare the
    // `OpacityMicromapIdKHR` execution mode, i.e. only for the KHR spelling of
    // VK_KHR_opacity_micromap: the EXT alias makes ray-query traversal use micromap
    // data without any execution mode, so the attach plan accepts there anyway.
    [[nodiscard]] bool opacity_micromap_execution_mode_available() const noexcept {
        if (!optional_extension_enabled(detail::OptionalExtension::OPACITY_MICROMAP)) {
            return false;
        }
        for (auto &&state : optional_extensions()) {
            if (state.id != detail::OptionalExtension::OPACITY_MICROMAP) { continue; }
            return state.name != nullptr &&
                   luisa::string_view{state.name} !=
                       luisa::string_view{VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME};
        }
        return false;
    }
    [[nodiscard]] bool enable_cuda_kernel_launch() const noexcept {
        return cuda_kernel_launch_enabled;
    }
    bool enable_async_copy() const { return async_copy_enabled; }
    [[nodiscard]] bool enable_sampler_anisotropy() const noexcept {
        return sampler_anisotropy_enabled;
    }
    // Whether a module may declare OpExecutionMode DenormPreserve /
    // DenormFlushToZero for 32-bit floats.
    [[nodiscard]] bool denorm_preserve_float32() const noexcept {
        return _denorm_preserve_float32;
    }
    [[nodiscard]] bool denorm_flush_float32() const noexcept {
        return _denorm_flush_float32;
    }
    [[nodiscard]] bool enable_shader_untyped_pointers() const noexcept {
        return shader_untyped_pointers_enabled;
    }
    // True when the backend may use vkCmdPipelineBarrier2 and vkQueueSubmit2 (core
    // 1.3 or VK_KHR_synchronization2), false when it must use
    // vkCmdPipelineBarrier and the legacy vkQueueSubmit.
    [[nodiscard]] bool sync2_capable() const noexcept {
        return sync2_capable_bit;
    }
    // True when the backend may use vkCmdCopyBuffer2/vkCmdCopyImage2 (core 1.3
    // or VK_KHR_copy_commands2), false when it must use the classic copies.
    [[nodiscard]] bool copy2_capable() const noexcept {
        return copy2_capable_bit;
    }
    // True when per-dispatch descriptor-set updates may go through a cached
    // VkDescriptorUpdateTemplate (plan P2.2). Latched at device creation; a
    // recording path that gets `false` here must issue vkUpdateDescriptorSets.
    [[nodiscard]] bool descriptor_update_template_capable() const noexcept {
        return descriptor_update_template_capable_bit;
    }
    // The resolved entry points of the template path. `available()` is false
    // whenever `descriptor_update_template_capable()` is false, so a caller may
    // pass them on unconditionally. The allocation callbacks are the device's own,
    // so every VkDescriptorUpdateTemplate is created and destroyed exactly like
    // the rest of the backend's Vulkan objects.
    [[nodiscard]] detail::DescriptorUpdateTemplateApi
    descriptor_update_template_api() const noexcept {
        return detail::DescriptorUpdateTemplateApi{
            .create = _create_descriptor_update_template,
            .destroy = _destroy_descriptor_update_template,
            .update = _update_descriptor_set_with_template,
            .callbacks = alloc_callbacks()};
    }
    // True when a shader may be created with a push descriptor set 0 (plan
    // P2.3). Latched at device creation; a shader that gets `true` here still
    // has to pass its own shape/limit check, and a shader that got the classic
    // layout never switches (see push_descriptor.h, shader.cpp).
    [[nodiscard]] bool push_descriptor_capable() const noexcept {
        return push_descriptor_capable_bit;
    }
    // The resolved entry point of the push path; `available()` is false whenever
    // `push_descriptor_capable()` is false, so a caller may pass it on
    // unconditionally.
    [[nodiscard]] detail::PushDescriptorApi push_descriptor_api() const noexcept {
        return detail::PushDescriptorApi{.cmd = _cmd_push_descriptor_set};
    }
    // VkPhysicalDevicePushDescriptorPropertiesKHR::maxPushDescriptors, or 0 when
    // the push path is unavailable.
    [[nodiscard]] uint32_t max_push_descriptors() const noexcept {
        return _max_push_descriptors;
    }
    // Exact optional Vulkan features enabled on this logical device that may
    // be consumed by persisted SPIR-V artifacts. Imported logical devices are
    // deliberately fail-closed for enable-chain features that Vulkan cannot
    // query after creation.
    [[nodiscard]] uint64_t enabled_spirv_artifact_features() const noexcept;
    [[nodiscard]] float max_sampler_anisotropy() const noexcept {
        return _max_sampler_anisotropy;
    }
    [[nodiscard]] luisa::string query(
        luisa::string_view property) noexcept override;
    static hlsl::ShaderCompiler *compiler();
    static VkAllocationCallbacks *alloc_callbacks();
    [[nodiscard]] VkInstance instance() const noexcept;
    uint compute_warp_size() const noexcept override;
    size_t compute_max_shared_memory_size() const noexcept override;
    uint64_t memory_granularity() const noexcept override;
    auto &allocator() { return *_allocator; }
    [[nodiscard]] const UploadBuffer *indirect_dispatch_dummy() const noexcept {
        return _indirect_dispatch_dummy.get();
    }
    auto physical_device() const { return _vk_device->physical_device; }
    auto logic_device() const { return _vk_device->logical_device; }
    // The device-owned interning facility for shader descriptor-set and pipeline
    // layouts (plan P1.1). `Shader` acquires references from it and gives them
    // back in its destructor; only this facility calls vkCreate*/vkDestroy* for
    // those objects.
    [[nodiscard]] detail::LayoutInterner &layout_interner() noexcept {
        return _layout_interner;
    }
    // Replayable recording (plan P4). `replay_registry()` is the device-owned
    // graph registry: the extension bookkeeps through it, and every lifetime path
    // (`destroy_buffer`, `destroy_stream`, `~Device`) invalidates or drains through
    // it, so a caller that never asks for `VulkanReplayExt` cannot leave a graph
    // pointing at a destroyed resource.
    [[nodiscard]] detail::ReplayRegistry &replay_registry() noexcept {
        return _replay_registry;
    }
    // Ordered batch shader compilation (plan P5.1). The pool object exists from
    // construction (it captured the executor there), while the lanes it drives are
    // created by the first batch, so a device that never compiles a batch never
    // creates a thread.
    [[nodiscard]] detail::ShaderCompilePool &shader_compile_pool() noexcept;
    // Stops and joins the pool for good; every later batch reports
    // `POOL_SHUTDOWN`. Idempotent, and called by `~Device` before the pipeline
    // cache, the interned layouts and the BinaryIO are torn down.
    void shutdown_shader_compile_pool() noexcept;
    // Serializes every shader-cache read and write of this device (plan P5.1). A
    // `BinaryIO` is one resource that a user implementation need not make
    // thread-safe, and a single writer keeps the artifact sequence deterministic.
    // The lock is never held over code generation, SPIR-V optimization or pipeline
    // creation; the raster path (a different translation unit) takes it for its own
    // artifact read/write.
    [[nodiscard]] std::recursive_mutex &shader_cache_mutex() noexcept {
        return _shader_cache_mtx;
    }
    // In-flight throttle switch of this device (`_inqueue_limit`): a replay
    // submission uses the same bound as an ordinary batch.
    [[nodiscard]] bool inqueue_limit() const noexcept { return _inqueue_limit; }
    // Validated stream lookup by handle (plan P4 keeps the extension away from
    // `reinterpret_cast<Stream *>`); `nullptr` for an unknown handle or for an
    // extension-owned `CustomStream`.
    [[nodiscard]] Stream *find_stream(uint64_t handle) noexcept;
    // Device-scoped Vulkan pipeline cache (plan P1.3). Every DSL shader
    // (compute, raster, ray tracing) creates its pipeline into this cache; the
    // native shader paths keep their own per-object caches because their
    // lifetime and binding ABI are separate contracts.
    [[nodiscard]] detail::DevicePipelineCache &pipeline_cache() noexcept {
        return _pipeline_cache;
    }
    [[nodiscard]] detail::DevicePipelineCache const &pipeline_cache() const noexcept {
        return _pipeline_cache;
    }
    auto const &pso_header() const { return _pso_header; }
    auto const &subgroup_size_control_properties() const noexcept { return _subgroup_size_control_properties; }
    // The numeric capability record captured at creation (plan P6.1/P6.2/P6.3).
    // `Device::query` answers the 14 core-limit keys and the SM-count keys from
    // it, and VulkanExtensionStatusExt::capability_status() copies it, so both
    // reports are the same values.
    auto const &capability_status() const noexcept { return _capability_status; }
    // On variable-subgroup GPUs that support native Wave32 via compute
    // subgroup-size control (e.g., AMD RDNA reporting min=32, max=64), prefer
    // Wave32 as the default compute subgroup size so warp-specialized kernels
    // and small threadgroups execute in full 32-lane waves.
    [[nodiscard]] bool prefers_default_wave32() const noexcept {
        return subgroup_size_control_enabled &&
               (_subgroup_size_control_properties.requiredSubgroupSizeStages &
                VK_SHADER_STAGE_COMPUTE_BIT) != 0u &&
               _subgroup_size_control_properties.minSubgroupSize == 32u &&
               _subgroup_size_control_properties.maxSubgroupSize > 32u;
    }
    [[nodiscard]] luisa::optional<uint8_t> effective_required_subgroup_size(
        Function kernel, bool compile_only) const noexcept {
        if (auto requested = kernel.allowed_warp_size()) {
            return requested;
        }
        if (!compile_only && prefers_default_wave32()) {
            const auto block_size = kernel.block_size();
            const uint64_t invocations =
                static_cast<uint64_t>(block_size.x) *
                static_cast<uint64_t>(block_size.y) *
                static_cast<uint64_t>(block_size.z);
            const uint64_t needed_subgroups = (invocations + 31u) / 32u;
            if (needed_subgroups <=
                _subgroup_size_control_properties.maxComputeWorkgroupSubgroups) {
                return static_cast<uint8_t>(32u);
            }
        }
        return luisa::nullopt;
    }
    bool is_pso_same(VkPipelineCacheHeaderVersionOne const &pso);
    auto const &properties() const { return _vk_device->properties; }
    auto const &features() const { return _vk_device->features; }
    auto const &enabled_features() const {
        return _vk_device->enabled_features;
    }
    auto graphics_queue_index() const { return _vk_device->queue_family_indices.graphics; }
    auto compute_queue_index() const { return _vk_device->queue_family_indices.compute; }
    auto copy_queue_index() const { return _vk_device->queue_family_indices.transfer; }
    auto sparse_queue_index() const { return _vk_device->queue_family_indices.sparse; }
    [[nodiscard]] auto queue_family_properties() const noexcept {
        return luisa::span<const VkQueueFamilyProperties>{
            _vk_device->queue_family_properties.data(),
            _vk_device->queue_family_properties.size()};
    }
    [[nodiscard]] auto &sparse_residency_registry() noexcept {
        return _sparse_residency_registry;
    }
    Device(Context &&ctx, DeviceConfig const *configs);
    ~Device();
    void *native_handle() const noexcept override;
    BufferCreationInfo create_buffer(const luisa::compute::Type *element, size_t elem_count, void *external_ptr) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;
    auto graphics_queue() const { return _graphics_queue; }
    auto compute_queue() const { return _compute_queue; }
    auto copy_queue() const { return _copy_queue; }
    auto sparse_queue() const { return _sparse_queue; }
    static bool print_code();
    // texture
    ResourceCreationInfo create_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, void *external_native_handle,
        bool simultaneous_access, bool allow_raster_target) noexcept override;
    void destroy_texture(uint64_t handle) noexcept override;

    // bindless array
    ResourceCreationInfo create_bindless_array(size_t size, BindlessSlotType type) noexcept override;
    void destroy_bindless_array(uint64_t handle) noexcept override;

    // stream
    ResourceCreationInfo create_stream(StreamTag stream_tag) noexcept override;
    void destroy_stream(uint64_t handle) noexcept override;
    void synchronize_stream(uint64_t stream_handle) noexcept override;
    void dispatch(
        uint64_t stream_handle, CommandList &&list) noexcept override;

    /// Register an extension-owned stream handle (see `CustomStream`).
    void add_custom_stream(uint64_t handle, CustomStream *stream) noexcept;
    /// Look up a registered custom stream; `nullptr` when `handle` is a normal
    /// `Stream`.
    [[nodiscard]] CustomStream *find_custom_stream(uint64_t handle) noexcept;
    /// Remove a registered custom stream from the registry without deleting it.
    [[nodiscard]] CustomStream *remove_custom_stream(uint64_t handle) noexcept;

    // swap chain
    SwapchainCreationInfo create_swapchain(const SwapchainOption &option, uint64_t stream_handle) noexcept override;
    void destroy_swapchain(uint64_t handle) noexcept override;
    void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept override;

    // kernel
    ShaderCreationInfo create_shader(const ShaderOption &option, Function kernel) noexcept override;
    ShaderCreationInfo create_tile_kernel(const ShaderOption &option, const tile::Function &kernel,
                                          const tile::CompileOptions &tile_options,
                                          tile::KernelMetadata &metadata) noexcept override;
    ShaderCreationInfo load_shader(luisa::string_view name, luisa::span<const luisa::compute::Type *const> arg_types) noexcept override;
    Usage shader_argument_usage(uint64_t handle, size_t index) noexcept override;
    void destroy_shader(uint64_t handle) noexcept override;

    // event
    ResourceCreationInfo create_event() noexcept override;
    void destroy_event(uint64_t handle) noexcept override;
    void signal_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept override;
    void wait_event(uint64_t handle, uint64_t stream_handle, uint64_t fence_value) noexcept override;
    bool is_event_completed(uint64_t handle, uint64_t fence_value) const noexcept override;
    void synchronize_event(uint64_t handle, uint64_t fence_value) noexcept override;

    // accel
    ResourceCreationInfo create_mesh(
        const AccelOption &option) noexcept override;
    void destroy_mesh(uint64_t handle) noexcept override;

    // cluster (mega-geometry) acceleration structures. Refuses loudly when
    // VK_NV_cluster_acceleration_structure is not enabled on this device.
    // TODO(dx): D3D12 counterpart (NVAPI cluster build or DXR clusters) — see
    // the RTXMG research report §7 for the divergence matrix.
    [[nodiscard]] ResourceCreationInfo create_mega_geometry(
        const AccelOption &option) noexcept override;
    void destroy_mega_geometry(uint64_t handle) noexcept override;
    [[nodiscard]] uint64_t mega_geometry_device_address(uint64_t handle) noexcept override;
    // 0 when this device cannot expose buffer device addresses (never asserts).
    [[nodiscard]] uint64_t buffer_device_address(uint64_t buffer_handle) noexcept override;

    // Opacity micromaps (VK_KHR/VK_EXT_opacity_micromap). `create` raises with an
    // actionable message - naming the extension, the feature and the
    // implementation limit - whenever an attach would not be legal on this device;
    // it never hands out a handle a build would then silently ignore.
    [[nodiscard]] ResourceCreationInfo create_opacity_micromap(
        const OpacityMicromapOption &option) noexcept override;
    void destroy_opacity_micromap(uint64_t handle) noexcept override;
    [[nodiscard]] uint32_t opacity_micromap_max_subdivision_level(bool four_state) noexcept override;

    ResourceCreationInfo create_procedural_primitive(
        const AccelOption &option) noexcept override;
    void destroy_procedural_primitive(uint64_t handle) noexcept override;

    ResourceCreationInfo create_accel(const AccelOption &option) noexcept override;
    void destroy_accel(uint64_t handle) noexcept override;

    // motion instance
    ResourceCreationInfo create_motion_instance(const AccelMotionOption &option) noexcept override;
    void destroy_motion_instance(uint64_t handle) noexcept override;

    // query
    void set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept override;
    ResourceCreationInfo allocate_sparse_texture_heap(size_t byte_size) noexcept override;
    void deallocate_sparse_texture_heap(uint64_t handle) noexcept override;
    ResourceCreationInfo allocate_sparse_buffer_heap(size_t byte_size) noexcept override;
    void deallocate_sparse_buffer_heap(uint64_t handle) noexcept override;
    void update_sparse_resources(
        uint64_t stream_handle,
        luisa::vector<SparseUpdateTile> &&textures_update) noexcept override;
    SparseBufferCreationInfo create_sparse_buffer(const luisa::compute::Type *element, size_t elem_count) noexcept override;
    SparseTextureCreationInfo create_sparse_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, bool simultaneous_access) noexcept override;
    void destroy_sparse_texture(uint64_t handle) noexcept override;
    void destroy_sparse_buffer(uint64_t handle) noexcept override;
    void set_stream_log_callback(uint64_t stream_handle,
                                 const StreamLogCallback &callback) noexcept override;
    DeviceExtension *extension(vstd::string_view name) noexcept override;
};
}// namespace lc::vk
