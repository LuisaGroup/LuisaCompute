#pragma once
#include <volk.h>
#include <luisa/runtime/device.h>
#include "VulkanDevice.h"
#include <luisa/vstl/common.h>
#include <luisa/core/first_fit.h>
#include "../common/default_binary_io.h"
#include "vk_allocator.h"
#include <luisa/backends/ext/vk_config_ext.h>
namespace lc::hlsl {
class ShaderCompiler;
}// namespace lc::hlsl
namespace lc::vk {
class ComputeShader;
using namespace luisa;
using namespace luisa::compute;
static constexpr size_t sparse_buffer_size = 65536ull;
class Device : public DeviceInterface, public vstd::IOperatorNewBase {
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
    std::mutex ext_mtx;
    vstd::unordered_map<vstd::string, Ext> exts;
    luisa::unique_ptr<VulkanDeviceConfigExt> _config_ext;
    vstd::optional<vks::VulkanDevice> _vk_device;
    VkPhysicalDeviceProperties _device_properties{};
    VkPhysicalDeviceFeatures _device_features{};
    VkPhysicalDeviceMemoryProperties _device_memory_properties{};
    vstd::vector<vstd::string> _enable_device_exts;
    VkQueue _graphics_queue{};
    VkQueue _compute_queue{};
    VkQueue _copy_queue{};
    VkDescriptorPool _sampler_pool;
    VkDescriptorSet _sampler_set;
    VkDescriptorSetLayout _sampler_set_layout;
    VkDescriptorPool _bdls_buffer_desc_pool;
    VkDescriptorSet _bdls_buffer_set;
    VkDescriptorSetLayout _bdls_buffer_set_layout;
    VkDescriptorPool _bdls_tex2d_desc_pool;
    VkDescriptorSet _bdls_tex2d_set;
    VkDescriptorSetLayout _bdls_tex2d_set_layout;
    VkDescriptorPool _bdls_tex3d_desc_pool;
    VkDescriptorSet _bdls_tex3d_set;
    VkDescriptorSetLayout _bdls_tex3d_set_layout;
    VkPipelineCacheHeaderVersionOne _pso_header{};
    vstd::vector<VkSampler> _samplers;
    vstd::optional<VkAllocator> _allocator;
    BinaryIO const *_binary_io{};
    vstd::unique_ptr<DefaultBinaryIO> _default_file_io;
    bool inqueue_limit = true;// TODO
    void _init_device(VkPhysicalDevice external_physical_device, VkDevice external_device, uint32_t selectedDevice, bool fallback);
public:
    struct HeapAlloc {
        uint count = 0;
        vstd::vector<uint> release_pool;
        luisa::spin_mutex mtx;
        luisa::FirstFit sub_allocator;
        uint full_size;
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
        vstd::unique_ptr<ComputeShader> shader;
        LoadFunc loadFunc;

    public:
        LazyLoadShader(LoadFunc loadFunc);
        ComputeShader *Get(Device *self);
        bool Check(Device *self);
        ~LazyLoadShader();
    };
    vstd::vector<VkImageView> tex2d_bindless_imgview;
    vstd::vector<VkImageView> tex3d_bindless_imgview;
    HeapAlloc tex2d_heap_pool;
    HeapAlloc tex3d_heap_pool;
    HeapAlloc buffer_heap_pool;
    LazyLoadShader set_bindless_kernel;
    LazyLoadShader set_accel_kernel;
    bool _external_instance : 1 {false};
    bool _external_device : 1 {false};
    bool _external_graphics_queue : 1 {false};
    bool _external_compute_queue : 1 {false};
    bool _external_copy_queue : 1 {false};
    auto &graphics_queue_mtx() { return _graphics_queue_mtx; }
    auto &compute_queue_mtx() { return _compute_queue_mtx; }
    auto &copy_queue_mtx() { return _copy_queue_mtx; }
    VulkanDeviceConfigExt *config_ext() const { return _config_ext.get(); }
    auto binary_io() const { return _binary_io; }
    auto sampler_set() const { return _sampler_set; }
    auto bdls_buffer_set() const { return _bdls_buffer_set; }
    auto bdls_tex2d_set() const { return _bdls_tex2d_set; }
    auto bdls_tex3d_set() const { return _bdls_tex3d_set; }
    auto samplers() const { return luisa::span{_samplers}; }
    static hlsl::ShaderCompiler *Compiler();
    static VkAllocationCallbacks *alloc_callbacks();
    VkInstance instance() const;
    uint compute_warp_size() const noexcept override;
    uint64_t memory_granularity() const noexcept override;
    auto &allocator() { return *_allocator; }
    auto physical_device() const { return _vk_device->physicalDevice; }
    auto logic_device() const { return _vk_device->logicalDevice; }
    auto const &pso_header() const { return _pso_header; }
    bool is_pso_same(VkPipelineCacheHeaderVersionOne const &pso);
    auto const &properties() const { return _vk_device->properties; }
    auto const &features() const { return _vk_device->features; }
    auto graphics_queue_index() const { return _vk_device->queueFamilyIndices.graphics; }
    auto compute_queue_index() const { return _vk_device->queueFamilyIndices.compute; }
    auto copy_queue_index() const { return _vk_device->queueFamilyIndices.transfer; }
    Device(Context &&ctx, DeviceConfig const *configs);
    ~Device();
    void *native_handle() const noexcept override;
    BufferCreationInfo create_buffer(const Type *element, size_t elem_count, void *external_ptr) noexcept override;
    BufferCreationInfo create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count, void *external_ptr) noexcept override;
    void destroy_buffer(uint64_t handle) noexcept override;
    auto graphics_queue() const { return _graphics_queue; }
    auto compute_queue() const { return _compute_queue; }
    auto copy_queue() const { return _copy_queue; }
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
    void destroy_swap_chain(uint64_t handle) noexcept override;
    void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept override;

    // kernel
    ShaderCreationInfo create_shader(const ShaderOption &option, Function kernel) noexcept override;
    ShaderCreationInfo create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept override;
    ShaderCreationInfo load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept override;
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

    // query
    void set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept override;
    ResourceCreationInfo allocate_sparse_texture_heap(size_t byte_size) noexcept override;
    void deallocate_sparse_texture_heap(uint64_t handle) noexcept override;
    ResourceCreationInfo allocate_sparse_buffer_heap(size_t byte_size) noexcept override;
    void deallocate_sparse_buffer_heap(uint64_t handle) noexcept override;
    void update_sparse_resources(
        uint64_t stream_handle,
        luisa::vector<SparseUpdateTile> &&textures_update) noexcept override;
    SparseBufferCreationInfo create_sparse_buffer(const Type *element, size_t elem_count) noexcept override;
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
