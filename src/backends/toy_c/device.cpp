#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/logging.h>
#include "stream.h"
#include "shader.h"
#include "../common/c_codegen/codegen_utils.h"
#include "../common/c_codegen/codegen_visitor.h"
#include <luisa/core/stl/filesystem.h>
namespace lc::toy_c {
using namespace luisa;
using namespace luisa::compute;
class LCDevice : public DeviceInterface, public vstd::IOperatorNewBase {
public:
    DynamicModule dyn_module;
    LCDevice(Context &&ctx, DeviceConfig const *settings)
        : DeviceInterface(std::move(ctx)) {
        dyn_module = DynamicModule::load("lc-script");
    }
    void *native_handle() const noexcept override { return nullptr; }
    uint compute_warp_size() const noexcept override {
        return {};
    };
    BufferCreationInfo create_buffer(
        const Type *element,
        size_t elem_count,
        void *external_memory /* nullptr if now imported from external memory */) override {
        LUISA_ERROR("Not supported.");
        return {};
    }
    BufferCreationInfo create_buffer(const ir::CArc<ir::Type> *element,
                                     size_t elem_count,
                                     void *external_memory /* nullptr if now imported from external memory */) noexcept override {
        LUISA_ERROR("Not supported.");
    }
    void destroy_buffer(uint64_t handle) noexcept override {
        LUISA_ERROR("Not supported.");
    }
    ResourceCreationInfo create_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, bool simultaneous_access, bool allow_raster_target) noexcept override {
        LUISA_ERROR("Not supported.");
        return {};
    }
    void destroy_texture(uint64_t handle) noexcept override {
        LUISA_ERROR("Not supported.");
    }
    ResourceCreationInfo create_bindless_array(size_t size) noexcept override {
        LUISA_ERROR("Not supported.");
        return {};
    }
    void destroy_bindless_array(uint64_t handle) noexcept override {
        LUISA_ERROR("Not supported.");
    }
    Usage shader_argument_usage(uint64_t handle, size_t index) noexcept override {
        LUISA_ERROR("Not supported.");
        return {};
    }
    SwapchainCreationInfo create_swapchain(
        const SwapchainOption &option, uint64_t stream_handle) noexcept override {
        LUISA_ERROR("Not supported.");
        return {};
    }
    void destroy_swap_chain(uint64_t handle) noexcept override {
        LUISA_ERROR("Not supported.");
    }
    void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept override {
        LUISA_ERROR("Not supported.");
    }

    // kernel
    ShaderCreationInfo create_shader(const ShaderOption &option, Function kernel) noexcept override {
        Clanguage_CodegenUtils codegen;
        auto kernel_name = luisa::to_string(luisa::filesystem::path{option.name}.filename().replace_extension());
        codegen.codegen(option.name, kernel_name, kernel);
        return ShaderCreationInfo::make_invalid();
    }
    ShaderCreationInfo create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept override {
        LUISA_ERROR("Not supported.");
        return {};
    }
    ResourceCreationInfo create_mesh(
        const AccelOption &option) noexcept override {
        LUISA_ERROR("Not supported.");
        return {};
    }
    void destroy_mesh(uint64_t handle) noexcept override {
        LUISA_ERROR("Not supported.");
    }

    ResourceCreationInfo create_procedural_primitive(
        const AccelOption &option) noexcept override {
        LUISA_ERROR("Not supported.");
        return {};
    }
    void destroy_procedural_primitive(uint64_t handle) noexcept override {
        LUISA_ERROR("Not supported.");
    }

    ResourceCreationInfo create_accel(const AccelOption &option) noexcept override {
        LUISA_ERROR("Not supported.");
        return {};
    }
    void destroy_accel(uint64_t handle) noexcept override {
        LUISA_ERROR("Not supported.");
    }

    // query
    luisa::string query(luisa::string_view property) noexcept override {
        LUISA_ERROR("Not supported.");
        return {};
    }
    DeviceExtension *extension(luisa::string_view name) noexcept override {
        LUISA_ERROR("Not supported.");
        return {};
    }
    void set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle, luisa::string_view name) noexcept override {
        LUISA_ERROR("Not supported.");
    }

    // sparse buffer
    [[nodiscard]] SparseBufferCreationInfo create_sparse_buffer(const Type *element, size_t elem_count) noexcept override {
        LUISA_ERROR("Not supported.");
        return {};
    }

    void destroy_sparse_buffer(uint64_t handle) noexcept override {
        LUISA_ERROR("Not supported.");
    }

    // sparse texture
    [[nodiscard]] SparseTextureCreationInfo create_sparse_texture(
        PixelFormat format, uint dimension,
        uint width, uint height, uint depth,
        uint mipmap_levels, bool simultaneous_access) noexcept override {
        LUISA_ERROR("Not supported.");
        return {};
    }
    void destroy_sparse_texture(uint64_t handle) noexcept override {
        LUISA_ERROR("Not supported.");
    }
    void update_sparse_resources(
        uint64_t stream_handle,
        luisa::vector<SparseUpdateTile> &&update_cmds) noexcept override {
        LUISA_ERROR("Not supported.");
    }
    ResourceCreationInfo allocate_sparse_buffer_heap(size_t byte_size) noexcept override {
        LUISA_ERROR("Not supported.");
        return {};
    }
    void deallocate_sparse_buffer_heap(uint64_t handle) noexcept override {
        LUISA_ERROR("Not supported.");
    }
    ResourceCreationInfo allocate_sparse_texture_heap(size_t byte_size, bool is_compressed_type) noexcept override {
        LUISA_ERROR("Not supported.");
        return {};
    }
    void deallocate_sparse_texture_heap(uint64_t handle) noexcept override {
        LUISA_ERROR("Not supported.");
    }
    // event
    ResourceCreationInfo create_event() noexcept override {
        LUISA_ERROR("Not supported.");
        return {};
    }
    void destroy_event(uint64_t handle) noexcept override {
        LUISA_ERROR("Not supported.");
    }
    void signal_event(uint64_t handle, uint64_t stream_handle, uint64_t fence) noexcept override {
        LUISA_ERROR("Not supported.");
    }
    void wait_event(uint64_t handle, uint64_t stream_handle, uint64_t fence) noexcept override {
        LUISA_ERROR("Not supported.");
    }
    bool is_event_completed(uint64_t handle, uint64_t fence) const noexcept override {
        LUISA_ERROR("Not supported.");
        return {};
    }
    void synchronize_event(uint64_t handle, uint64_t fence) noexcept override {
        LUISA_ERROR("Not supported.");
    }

    ResourceCreationInfo create_stream(StreamTag stream_tag) noexcept override {
        auto ptr = new LCStream();
        return {
            .handle = reinterpret_cast<uint64_t>(ptr),
            .native_handle = ptr};
    }
    void destroy_stream(uint64_t handle) noexcept override {
        delete reinterpret_cast<LCStream *>(handle);
    }
    void synchronize_stream(uint64_t stream_handle) noexcept override {
        reinterpret_cast<LCStream *>(stream_handle)->sync();
    }
    void dispatch(
        uint64_t stream_handle, CommandList &&list) noexcept override {
        reinterpret_cast<LCStream *>(stream_handle)->dispatch(std::move(list));
    }
    void set_stream_log_callback(
        uint64_t stream_handle,
        const StreamLogCallback &callback) noexcept override {
        LUISA_ERROR("TODO");
    }
    ShaderCreationInfo load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept override {
        auto ptr = new LCShader(dyn_module, arg_types, name);
        ShaderCreationInfo r;
        r.handle = reinterpret_cast<uint64_t>(ptr);
        r.native_handle = ptr;
        r.block_size = ptr->block_size;
        return r;
    }
    void destroy_shader(uint64_t handle) noexcept override {
        delete reinterpret_cast<LCShader *>(handle);
    }
};
VSTL_EXPORT_C DeviceInterface *create(Context &&c, DeviceConfig const *settings) {
    return new LCDevice(std::move(c), settings);
}
VSTL_EXPORT_C void destroy(DeviceInterface *device) {
    delete static_cast<LCDevice *>(device);
}
}// namespace lc::toy_c