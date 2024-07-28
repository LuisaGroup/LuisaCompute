#include <luisa/runtime/rhi/device_interface.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/logging.h>
#include "stream.h"
#include "shader.h"
#include "../common/c_codegen/codegen_utils.h"
#include <luisa/backends/ext/toy_c_ext.h>
#include <luisa/core/stl/filesystem.h>
#include "memory_manager.h"
#include "../common/shader_print_formatter.h"
namespace lc::toy_c {
using namespace luisa;
using namespace luisa::compute;
struct Dtor {
    vstd::vector<std::pair<uint64_t, Dtor *>> sub_elements;
    vstd::func_ptr_t<void(Dtor *self, void *)> self_dtor{};
};
class LCDevice : public DeviceInterface, public vstd::IOperatorNewBase {
public:
    DynamicModule dyn_module;
    vstd::optional<MemoryManager> manager;
    ToyCDeviceConfig::FuncTable *func_table_ptr;
    LCDevice(Context &&ctx, DeviceConfig const *settings)
        : DeviceInterface(std::move(ctx)) {
        if (!settings->headless) {
            if (!settings->extension) [[unlikely]] {
                LUISA_ERROR("DeviceConfig::extension must be an instance of luisa::compute::ToyCDeviceConfig.");
            }
            manager.create();
            auto ext = static_cast<ToyCDeviceConfig *>(settings->extension.get());
            auto module_name = ext->dynamic_module_name();
            dyn_module = DynamicModule::load(module_name);
            if (!dyn_module) [[unlikely]] {
                LUISA_ERROR("Dynamic module {} not found.", module_name);
            }
            auto func_name = ext->set_func_table_name();
            vstd::func_ptr_t<ToyCDeviceConfig::FuncTable *(void *)> set_functable = dyn_module.function<ToyCDeviceConfig::FuncTable *(void *)>(func_name);
            if (!set_functable) [[unlikely]] {
                LUISA_ERROR("{} not found.", func_name);
            }
            auto table_opt = ext->get_functable();
            if (table_opt) {
                func_table_ptr = set_functable(&table_opt.value());
            } else {
                ToyCDeviceConfig::FuncTable table{};
                table.persist_malloc = vengine_malloc,
                table.temp_malloc = +[](size_t size) -> void * {
                    auto handle = MemoryManager::get_tlocal_ctx()->temp_alloc.allocate(size, 16);
                    return (void *)(handle.handle + handle.offset);
                },
                table.persist_free = vengine_free,
                table.push_print_str =
                    +[](char const *ptr, uint64_t len) {
                        MemoryManager::get_tlocal_ctx()->print_format = luisa::string_view(ptr, len);
                    },
                table.push_print_value =
                    +[](void *value, uint32_t type) {
                        auto ctx = MemoryManager::get_tlocal_ctx();
                        switch (type) {
                            case 0: ctx->print_values.emplace_back(*((bool *)value)); break;
                            case 1: ctx->print_values.emplace_back(*((bool2 *)value)); break;
                            case 2: ctx->print_values.emplace_back(*((bool3 *)value)); break;
                            case 3: ctx->print_values.emplace_back(*((bool4 *)value)); break;
                            case 4: ctx->print_values.emplace_back(*((float *)value)); break;
                            case 5: ctx->print_values.emplace_back(*((double *)value)); break;
                            case 6: ctx->print_values.emplace_back(*((int32_t *)value)); break;
                            case 7: ctx->print_values.emplace_back(*((int8_t *)value)); break;
                            case 8: ctx->print_values.emplace_back(*((int16_t *)value)); break;
                            case 9: ctx->print_values.emplace_back(*((int64_t *)value)); break;
                            case 10: ctx->print_values.emplace_back(*((uint32_t *)value)); break;
                            case 11: ctx->print_values.emplace_back(*((uint8_t *)value)); break;
                            case 12: ctx->print_values.emplace_back(*((uint16_t *)value)); break;
                            case 13: ctx->print_values.emplace_back(*((uint64_t *)value)); break;
                            case 14: ctx->print_values.emplace_back(*((float2 *)value)); break;
                            case 15: ctx->print_values.emplace_back(*((double2 *)value)); break;
                            case 16: ctx->print_values.emplace_back(*((int2 *)value)); break;
                            case 17: ctx->print_values.emplace_back(*((byte2 *)value)); break;
                            case 18: ctx->print_values.emplace_back(*((short2 *)value)); break;
                            case 19: ctx->print_values.emplace_back(*((slong2 *)value)); break;
                            case 20: ctx->print_values.emplace_back(*((uint2 *)value)); break;
                            case 21: ctx->print_values.emplace_back(*((ubyte2 *)value)); break;
                            case 22: ctx->print_values.emplace_back(*((ushort2 *)value)); break;
                            case 23: ctx->print_values.emplace_back(*((ulong2 *)value)); break;
                            case 24: ctx->print_values.emplace_back(*((float3 *)value)); break;
                            case 25: ctx->print_values.emplace_back(*((double3 *)value)); break;
                            case 26: ctx->print_values.emplace_back(*((int3 *)value)); break;
                            case 27: ctx->print_values.emplace_back(*((byte3 *)value)); break;
                            case 28: ctx->print_values.emplace_back(*((short3 *)value)); break;
                            case 29: ctx->print_values.emplace_back(*((slong3 *)value)); break;
                            case 30: ctx->print_values.emplace_back(*((uint3 *)value)); break;
                            case 31: ctx->print_values.emplace_back(*((ubyte3 *)value)); break;
                            case 32: ctx->print_values.emplace_back(*((ushort3 *)value)); break;
                            case 33: ctx->print_values.emplace_back(*((ulong3 *)value)); break;
                            case 34: ctx->print_values.emplace_back(*((float4 *)value)); break;
                            case 35: ctx->print_values.emplace_back(*((double4 *)value)); break;
                            case 36: ctx->print_values.emplace_back(*((int4 *)value)); break;
                            case 37: ctx->print_values.emplace_back(*((byte4 *)value)); break;
                            case 38: ctx->print_values.emplace_back(*((short4 *)value)); break;
                            case 39: ctx->print_values.emplace_back(*((slong4 *)value)); break;
                            case 40: ctx->print_values.emplace_back(*((uint4 *)value)); break;
                            case 41: ctx->print_values.emplace_back(*((ubyte4 *)value)); break;
                            case 42: ctx->print_values.emplace_back(*((ushort4 *)value)); break;
                            case 43: ctx->print_values.emplace_back(*((ulong4 *)value)); break;
                            case 44: ctx->print_values.emplace_back(*((float2x2 *)value)); break;
                            case 45: ctx->print_values.emplace_back(*((float3x3 *)value)); break;
                            case 46: ctx->print_values.emplace_back(*((float4x4 *)value)); break;
                            default: break;
                        }
                    },
                table.print = +[]() {
                    auto ctx = MemoryManager::get_tlocal_ctx();
                    if(!ctx->stream->print_callback) return;
                    luisa::vector<Type const*> types;
                    vstd::push_back_func(types, ctx->print_values.size(), [&](size_t i){
                        return luisa::visit([&]<typename T>(T const& t){
                            return Type::of<T>();
                        }, ctx->print_values[i]);
                    });
                    auto str_type = Type::structure(types);
                    luisa::vector<std::byte> bytes;
                    bytes.reserve(str_type->size());
                    for(auto i : vstd::range(types.size())){
                        auto type = types[i];
                        auto start_idx = (bytes.size() + type->alignment() - 1) & (~(type->alignment() - 1));
                        bytes.resize_uninitialized(start_idx + type->size());
                        luisa::visit([&]<typename T>(T const& t){
                            std::memcpy(bytes.data() + start_idx, &t, type->size());
                        }, ctx->print_values[i]);
                    }
                    
                    ShaderPrintFormatter fmt{ctx->print_format, str_type, false};
                    luisa::string str;
                    fmt(str, bytes);
                    ctx->stream->print_callback(str);
                    ctx->print_format = {};
                    ctx->print_values.clear(); };
                func_table_ptr = set_functable(&table);
            }
        }
    }
    void *native_handle() const noexcept override { return nullptr; }
    uint compute_warp_size() const noexcept override {
        return {};
    };
    BufferCreationInfo create_buffer(
        const Type *element,
        size_t elem_count,
        void *external_memory /* nullptr if now imported from external memory */) noexcept override {
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
        LUISA_ERROR("Synchronize is fobidden in this backend, all commands are executed synchronizly.");
    }
    void dispatch(
        uint64_t stream_handle, CommandList &&list) noexcept override {
        reinterpret_cast<LCStream *>(stream_handle)->dispatch(*manager, this, std::move(list));
    }
    void set_stream_log_callback(
        uint64_t stream_handle,
        const StreamLogCallback &callback) noexcept override {
        reinterpret_cast<LCStream *>(stream_handle)->print_callback = callback;
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
VSTL_EXPORT_C void backend_device_names(luisa::vector<luisa::string> &r) {
    r.clear();
}
}// namespace lc::toy_c