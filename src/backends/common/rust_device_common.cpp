#include <core/logging.h>
#include <luisa_compute_ir/bindings.hpp>
#include <luisa_compute_api_types/bindings.hpp>

namespace luisa::compute::backend {
    using namespace luisa::compute::api;
    using luisa::compute::ir::CArc;
    using luisa::compute::ir::KernelModule;
    using luisa::compute::ir::Type;
}// namespace luisa::compute::backend

#include "rust_device_common.h"
#include <runtime/context.h>
#include <core/dynamic_module.h>
#include <core/logging.h>
#include <luisa_compute_api_types/bindings.h>
// must go last to avoid name conflicts
#include <runtime/rhi/resource.h>

namespace luisa::compute::rust {
    template<class T>
    T unwrap(const api::Result<T> &r) noexcept {
        if (r.tag == api::Result<T>::Tag::OK) {
            return r.OK._0;
        }
        LUISA_ERROR_WITH_LOCATION("unwrap an Err value");
    }

// @Mike-Leo-Smith: fill-in the blanks pls
    class RustDevice final : public DeviceInterface {
        api::DeviceInterface device{};
        api::LibInterface lib{};
        luisa::filesystem::path runtime_path;
        DynamicModule dll;

        api::LibInterface (*luisa_compute_lib_interface)();

        api::Context api_ctx{};

    public:
        ~RustDevice() noexcept override {
            device.destroy_device(device);
            lib.destroy_context(api_ctx);
        }

        RustDevice(Context &&ctx, luisa::filesystem::path runtime_path, string_view name) noexcept: DeviceInterface(
                std::move(ctx)), runtime_path(std::move(runtime_path)) {
            dll = DynamicModule::load(this->runtime_path, "luisa_compute_backend_impl");
            luisa_compute_lib_interface = dll.function<api::LibInterface()>("luisa_compute_lib_interface");
            lib = luisa_compute_lib_interface();
            api_ctx = lib.create_context(this->runtime_path.c_str());
            device = lib.create_device(api_ctx, name.data(), nullptr);
            lib.set_logger_callback([](api::LoggerMessage message) {
                luisa::string_view target(message.target);
                luisa::string_view level(message.level);
                luisa::string_view body(message.message);
                if (level == "I") {
                    LUISA_INFO("[{}] {}", target, body);
                } else if (level == "W") {
                    LUISA_WARNING("[{}] {}", target, body);
                } else if (level == "E") {
                    LUISA_ERROR("[{}] {}", target, body);
                } else {
                    LUISA_VERBOSE("[{}] {}", target, body);
                }
            });
        }

        void *native_handle() const noexcept override {
            return (void *) device.device._0;
        }

        BufferCreationInfo create_buffer(const Type *element, size_t elem_count) noexcept override {
            LUISA_ERROR_WITH_LOCATION("@Mike-Leo-Smith: fix fix pls");
        }

        BufferCreationInfo create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count) noexcept override {
            api::CreatedBufferInfo buffer = unwrap(device.create_buffer(device.device, element, elem_count));
            BufferCreationInfo info{};
            info.element_stride = buffer.element_stride;
            info.total_size_bytes = buffer.total_size_bytes;
            info.handle = buffer.resource.handle;
            info.native_handle = buffer.resource.native_handle;
            return info;
        }

        void destroy_buffer(uint64_t handle) noexcept override {
            device.destroy_buffer(device.device, api::Buffer{handle});
        }

        ResourceCreationInfo create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth,
                                            uint mipmap_levels) noexcept override {
            api::CreatedResourceInfo texture = unwrap(
                    device.create_texture(device.device, (api::PixelFormat) format, dimension, width, height, depth,
                                          mipmap_levels));
            ResourceCreationInfo info{};
            info.handle = texture.handle;
            info.native_handle = texture.native_handle;
            return info;
        }

        void destroy_texture(uint64_t handle) noexcept override {
            device.destroy_texture(device.device, api::Texture{handle});
        }

        ResourceCreationInfo create_bindless_array(size_t size) noexcept override {
            api::CreatedResourceInfo array = unwrap(device.create_bindless_array(device.device, size));
            ResourceCreationInfo info{};
            info.handle = array.handle;
            info.native_handle = array.native_handle;
            return info;
        }

        void destroy_bindless_array(uint64_t handle) noexcept override {
            device.destroy_bindless_array(device.device, api::BindlessArray{handle});
        }

        ResourceCreationInfo create_stream(StreamTag stream_tag) noexcept override {
            api::CreatedResourceInfo stream = unwrap(device.create_stream(device.device, (api::StreamTag) stream_tag));
            ResourceCreationInfo info{};
            info.handle = stream.handle;
            info.native_handle = stream.native_handle;
            return info;
        }

        void destroy_stream(uint64_t handle) noexcept override {
            device.destroy_stream(device.device, api::Stream{handle});
        }

        void synchronize_stream(uint64_t stream_handle) noexcept override {
            device.synchronize_stream(device.device, api::Stream{stream_handle});
        }

        void dispatch(uint64_t stream_handle, CommandList &&list) noexcept override {

        }

        SwapChainCreationInfo
        create_swap_chain(uint64_t window_handle, uint64_t stream_handle, uint width, uint height, bool allow_hdr,
                          bool vsync, uint back_buffer_size) noexcept override {
            auto swapchain = unwrap(
                    device.create_swapchain(device.device, window_handle, api::Stream{stream_handle}, width, height,
                                            allow_hdr, vsync, back_buffer_size));
            SwapChainCreationInfo info{};
            info.handle = swapchain.resource.handle;
            info.native_handle = swapchain.resource.native_handle;
            info.storage = (PixelStorage) swapchain.storage;
            return info;
        }

        void destroy_swap_chain(uint64_t handle) noexcept override {
            device.destroy_swapchain(device.device, api::Swapchain{handle});
        }

        void present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle,
                                       uint64_t image_handle) noexcept override {
            device.present_display_in_stream(device.device, api::Stream{stream_handle},
                                             api::Swapchain{swapchain_handle}, api::Texture{image_handle});
        }

        ShaderCreationInfo create_shader(const ShaderOption &option, Function kernel) noexcept override {
            LUISA_ERROR_WITH_LOCATION("@Mike-Leo-Smith: fix fix pls");
        }

        ShaderCreationInfo
        create_shader(const ShaderOption &option_, const ir::KernelModule *kernel) noexcept override {
            api::ShaderOption option{};
            option.compile_only = option_.compile_only;
            option.enable_cache = option_.enable_cache;
            option.enable_debug_info = option_.enable_debug_info;
            option.enable_fast_math = option_.enable_fast_math;
            auto shader = unwrap(device.create_shader(device.device, api::KernelModule{(uint64_t) kernel}, &option));
            ShaderCreationInfo info{};
            info.block_size[0] = shader.block_size[0];
            info.block_size[1] = shader.block_size[1];
            info.block_size[2] = shader.block_size[2];
            info.handle = shader.resource.handle;
            info.native_handle = shader.resource.native_handle;
            return info;

        }

        ShaderCreationInfo
        load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept override {
            LUISA_ERROR_WITH_LOCATION("unimplemented");
        }

        Usage shader_argument_usage(uint64_t handle, size_t index) noexcept override {
            return Usage::NONE;
        }

        void destroy_shader(uint64_t handle) noexcept override {
            device.destroy_shader(device.device, api::Shader{handle});
        }

        ResourceCreationInfo create_event() noexcept override {
            api::CreatedResourceInfo event = unwrap(device.create_event(device.device));
            ResourceCreationInfo info{};
            info.handle = event.handle;
            info.native_handle = event.native_handle;
            return info;
        }

        void destroy_event(uint64_t handle) noexcept override {
            device.destroy_event(device.device, api::Event{handle});
        }

        void signal_event(uint64_t handle, uint64_t stream_handle) noexcept override {
            device.signal_event(device.device, api::Event{handle}, api::Stream{stream_handle});
        }

        void wait_event(uint64_t handle, uint64_t stream_handle) noexcept override {
            device.wait_event(device.device, api::Event{handle}, api::Stream{stream_handle});
        }

        void synchronize_event(uint64_t handle) noexcept override {
            device.synchronize_event(device.device, api::Event{handle});
        }

        ResourceCreationInfo create_mesh(const AccelOption &option_) noexcept override {
            api::AccelOption option{};
            option.allow_compaction = option_.allow_compaction;
            option.allow_update = option_.allow_update;
            option.hint = (api::AccelUsageHint) option_.hint;
            auto mesh = unwrap(device.create_mesh(device.device, &option));
            ResourceCreationInfo info{};
            info.handle = mesh.handle;
            info.native_handle = mesh.native_handle;
            return info;
        }

        void destroy_mesh(uint64_t handle) noexcept override {
            device.destroy_mesh(device.device, api::Mesh{handle});
        }

        ResourceCreationInfo create_procedural_primitive(const AccelOption &option) noexcept override {
            LUISA_ERROR_WITH_LOCATION("unimplemented");
        }

        void destroy_procedural_primitive(uint64_t handle) noexcept override {
            LUISA_ERROR_WITH_LOCATION("unimplemented");
        }

        ResourceCreationInfo create_accel(const AccelOption &option_) noexcept override {
            api::AccelOption option{};
            option.allow_compaction = option_.allow_compaction;
            option.allow_update = option_.allow_update;
            option.hint = (api::AccelUsageHint) option_.hint;
            auto accel = unwrap(device.create_accel(device.device, &option));
            ResourceCreationInfo info{};
            info.handle = accel.handle;
            info.native_handle = accel.native_handle;
            return info;
        }

        void destroy_accel(uint64_t handle) noexcept override {
            device.destroy_accel(device.device, api::Accel{handle});
        }

        void set_name(luisa::compute::Resource::Tag resource_tag, uint64_t resource_handle,
                      luisa::string_view name) noexcept override {

        }
    };

    luisa::compute::DeviceInterface *create(luisa::compute::Context &&ctx, const luisa::compute::DeviceConfig *config,
                                            luisa::string_view name) noexcept {
        auto path = ctx.runtime_directory();
        return luisa::new_with_allocator<luisa::compute::rust::RustDevice>(std::move(ctx), std::move(path), "cpu");
    }

    void destroy(luisa::compute::DeviceInterface *device) noexcept {
        luisa::delete_with_allocator(device);
    }
}// namespace luisa::compute::rust
