#pragma once
#include <volk.h>
#include "vk_shader_untyped_pointers.h"
#include <luisa/runtime/device.h>
#include "VulkanDevice.h"
#include <luisa/vstl/common.h>
#include <luisa/core/first_fit.h>
#include "../common/default_binary_io.h"
#include "vk_allocator.h"
#include "sparse_residency_registry.h"
#include <luisa/backends/ext/vk_config_ext.h>
#include <atomic>
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
    std::mutex _ext_mtx;
    vstd::unordered_map<vstd::string, Ext> _exts;
    vstd::unordered_set<Stream *> _streams;
    luisa::unique_ptr<VulkanDeviceConfigExt> _config_ext;
    VkInstance _instance{};
    vstd::optional<vks::VulkanDevice> _vk_device;
    vstd::vector<vstd::string> _enable_device_exts;
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
    VkPipelineCacheHeaderVersionOne _pso_header{};
    VkPhysicalDeviceSubgroupSizeControlProperties _subgroup_size_control_properties{};
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
    struct NumericFeatures {
        bool shader_float8{false};
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
    void _init_device(VkPhysicalDevice external_physical_device, VkDevice external_device, uint32_t selected_device);
    [[nodiscard]] luisa::shared_ptr<NativeImageState>
    acquire_native_image_state(
        VkImage image, VkFormat format, uint dimension, uint3 size,
        uint mip_levels, bool simultaneous_access);
    [[nodiscard]] ShaderCreationInfo _create_shader_hlsl(
        const ShaderOption &option, Function kernel,
        bool requires_sampler_anisotropy) noexcept;
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
    uint32_t external_graphics_queue_family_index{VK_QUEUE_FAMILY_IGNORED};
    uint32_t external_compute_queue_family_index{VK_QUEUE_FAMILY_IGNORED};
    uint32_t external_copy_queue_family_index{VK_QUEUE_FAMILY_IGNORED};
    bool bindless_enabled : 1 {true};
    bool raytracing_enabled : 1 {true};
    bool surface_enabled : 1 {true};
    bool device_address_enabled : 1 {true};
    bool interop_enabled : 1 {true};
    bool motion_blur_enabled : 1 {false};
    bool subgroup_size_control_enabled : 1 {false};
    bool subgroup_extended_types_enabled : 1 {false};
    bool cooperative_vector_enabled : 1 {false};
    bool cooperative_vector_fp32_enabled : 1 {false};
    bool shader_untyped_pointers_enabled : 1 {false};
    bool async_copy_enabled : 1 {false};
    bool sampler_anisotropy_enabled : 1 {false};
    auto &graphics_queue_mtx() { return *_graphics_queue_lock; }
    auto &compute_queue_mtx() { return *_compute_queue_lock; }
    auto &copy_queue_mtx() { return *_copy_queue_lock; }
    auto &sparse_queue_mtx() { return *_sparse_queue_lock; }
    VulkanDeviceConfigExt *config_ext() const { return _config_ext.get(); }
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
    [[nodiscard]] uint64_t max_timeline_semaphore_value_difference() const noexcept {
        return _timeline_semaphore_properties
            .maxTimelineSemaphoreValueDifference;
    }
    [[nodiscard]] const auto &acceleration_structure_properties() const noexcept {
        return _acceleration_structure_properties;
    }
    bool enable_interop() const { return interop_enabled; }
    bool enable_motion_blur() const { return motion_blur_enabled; }
    bool enable_raytracing() const { return raytracing_enabled; }
    bool enable_device_address() const { return device_address_enabled; }
    bool enable_async_copy() const { return async_copy_enabled; }
    [[nodiscard]] bool enable_sampler_anisotropy() const noexcept {
        return sampler_anisotropy_enabled;
    }
    [[nodiscard]] bool enable_shader_untyped_pointers() const noexcept {
        return shader_untyped_pointers_enabled;
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
    uint64_t memory_granularity() const noexcept override;
    auto &allocator() { return *_allocator; }
    [[nodiscard]] const UploadBuffer *indirect_dispatch_dummy() const noexcept {
        return _indirect_dispatch_dummy.get();
    }
    auto physical_device() const { return _vk_device->physical_device; }
    auto logic_device() const { return _vk_device->logical_device; }
    auto const &pso_header() const { return _pso_header; }
    auto const &subgroup_size_control_properties() const noexcept { return _subgroup_size_control_properties; }
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

    // swap chain
    SwapchainCreationInfo create_swapchain(const SwapchainOption &option, uint64_t stream_handle) noexcept override;
    void destroy_swapchain(uint64_t handle) noexcept override;
    void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept override;

    // kernel
    ShaderCreationInfo create_shader(const ShaderOption &option, Function kernel) noexcept override;
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
