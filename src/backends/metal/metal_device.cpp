//
// Created by Mike Smith on 2023/4/8.
//

#include <core/logging.h>

#ifdef LUISA_ENABLE_IR
#include <backends/metal/metal_codegen_ir.h>
#endif

#include <backends/metal/metal_builtin_embedded.h>
#include <backends/metal/metal_codegen_ast.h>
#include <backends/metal/metal_compiler.h>
#include <backends/metal/metal_buffer.h>
#include <backends/metal/metal_texture.h>
#include <backends/metal/metal_stream.h>
#include <backends/metal/metal_event.h>
#include <backends/metal/metal_swapchain.h>
#include <backends/metal/metal_bindless_array.h>
#include <backends/metal/metal_accel.h>
#include <backends/metal/metal_mesh.h>
#include <backends/metal/metal_procedrual_primitive.h>
#include <backends/metal/metal_device.h>

namespace luisa::compute::metal {

MetalDevice::MetalDevice(Context &&ctx, const DeviceConfig *config) noexcept
    : DeviceInterface{std::move(ctx)},
      _io{nullptr},
      _inqueue_buffer_limit{config == nullptr || config->inqueue_buffer_limit} {

    auto device_index = config == nullptr ? 0u : config->device_index;
    auto all_devices = MTL::CopyAllDevices();
    auto device_count = all_devices->count();
    LUISA_ASSERT(device_index < device_count,
                 "Metal device index out of range.");
    _handle = all_devices->object<MTL::Device>(device_index)->retain();
    all_devices->release();

    LUISA_ASSERT(_handle->supportsFamily(MTL::GPUFamilyMetal3),
                 "Metal device '{}' at index {} does not support Metal 3.",
                 _handle->name()->utf8String(), device_index);

    LUISA_INFO("Metal device '{}' at index {}",
               _handle->name()->utf8String(), device_index);

    // create a default binary IO if none is provided
    if (config == nullptr || config->binary_io == nullptr) {
        _default_io = luisa::make_unique<DefaultBinaryIO>(context());
        _io = _default_io.get();
    } else {
        _io = config->binary_io;
    }

    // create a compiler
    _compiler = luisa::make_unique<MetalCompiler>(this);

    // TODO: load built-in kernels
    auto builtin_kernel_source = NS::String::alloc()->init(
        const_cast<char *>(luisa_metal_builtin_metal_builtin_kernels),
        sizeof(luisa_metal_builtin_metal_builtin_kernels),
        NS::UTF8StringEncoding, false);
    auto compile_options = MTL::CompileOptions::alloc()->init();
    compile_options->setFastMathEnabled(true);
    compile_options->setLanguageVersion(MTL::LanguageVersion3_0);
    compile_options->setLibraryType(MTL::LibraryTypeExecutable);
    NS::Error *error{nullptr};
    auto builtin_library = _handle->newLibrary(builtin_kernel_source, compile_options, &error);

    builtin_kernel_source->release();
    compile_options->release();

    if (error != nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to compile built-in Metal kernels: {}",
            error->localizedDescription()->utf8String());
    }
    error = nullptr;
    LUISA_ASSERT(builtin_library != nullptr,
                 "Failed to compile built-in Metal kernels.");

    // compute pipelines
    auto compute_pipeline_desc = MTL::ComputePipelineDescriptor::alloc()->init();
    compute_pipeline_desc->setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true);
    compute_pipeline_desc->setMaxTotalThreadsPerThreadgroup(256u);
    auto create_builtin_compute_shader = [&](auto name) noexcept {
        auto function = builtin_library->newFunction(name);
        compute_pipeline_desc->setComputeFunction(function);
        auto pipeline = _handle->newComputePipelineState(
            compute_pipeline_desc, MTL::PipelineOptionNone, nullptr, &error);
        if (error != nullptr) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to compile built-in Metal kernel '{}': {}",
                name->utf8String(), error->localizedDescription()->utf8String());
        }
        error = nullptr;
        LUISA_ASSERT(pipeline != nullptr,
                     "Failed to compile built-in Metal kernel '{}'.",
                     name->utf8String());
        function->release();
        return pipeline;
    };
    _builtin_update_bindless_slots = create_builtin_compute_shader(MTLSTR("update_bindless_array"));
    _builtin_update_accel_instances = create_builtin_compute_shader(MTLSTR("update_accel_instances"));
    compute_pipeline_desc->release();

    // render pipeline
    auto builtin_swapchain_vertex_shader = builtin_library->newFunction(MTLSTR("swapchain_vertex_shader"));
    auto builtin_swapchain_fragment_shader = builtin_library->newFunction(MTLSTR("swapchain_fragment_shader"));
    auto render_pipeline_desc = MTL::RenderPipelineDescriptor::alloc()->init();
    render_pipeline_desc->setVertexFunction(builtin_swapchain_vertex_shader);
    render_pipeline_desc->setFragmentFunction(builtin_swapchain_fragment_shader);
    auto color_attachment = render_pipeline_desc->colorAttachments()->object(0u);
    color_attachment->setBlendingEnabled(false);
    auto create_builtin_present_shader = [&](auto format) noexcept {
        color_attachment->setPixelFormat(format);
        auto shader = _handle->newRenderPipelineState(
            render_pipeline_desc, MTL::PipelineOptionNone, nullptr, &error);
        if (error != nullptr) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to compile built-in Metal kernel 'swapchain_fragment_shader': {}",
                error->localizedDescription()->utf8String());
        }
        error = nullptr;
        LUISA_ASSERT(shader != nullptr,
                     "Failed to compile built-in Metal kernel 'swapchain_fragment_shader'.");
        return shader;
    };
    _builtin_swapchain_present_ldr = create_builtin_present_shader(MTL::PixelFormatRGBA8Unorm);
    _builtin_swapchain_present_hdr = create_builtin_present_shader(MTL::PixelFormatRGBA16Float);
    render_pipeline_desc->release();
    builtin_swapchain_vertex_shader->release();
    builtin_swapchain_fragment_shader->release();

    builtin_library->release();
}

MetalDevice::~MetalDevice() noexcept {
    _builtin_update_bindless_slots->release();
    _builtin_update_accel_instances->release();
    _builtin_swapchain_present_ldr->release();
    _builtin_swapchain_present_hdr->release();
    _handle->release();
}

void *MetalDevice::native_handle() const noexcept {
    return _handle;
}

[[nodiscard]] inline auto create_device_buffer(MTL::Device *device, size_t element_stride, size_t element_count) noexcept {
    auto buffer_size = element_stride * element_count;
    auto buffer = new_with_allocator<MetalBuffer>(device, buffer_size);
    BufferCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(buffer);
    info.native_handle = buffer;
    info.element_stride = element_stride;
    info.total_size_bytes = buffer_size;
    return info;
}

BufferCreationInfo MetalDevice::create_buffer(const Type *element, size_t elem_count) noexcept {
    return with_autorelease_pool([=, this] {
        auto elem_size = MetalCodegenAST::type_size_bytes(element);
        return create_device_buffer(_handle, elem_size, elem_count);
    });
}

BufferCreationInfo MetalDevice::create_buffer(const ir::CArc<ir::Type> *element, size_t elem_count) noexcept {
#ifdef LUISA_ENABLE_IR
    return with_autorelease_pool([=, this] {
        auto elem_size = MetalCodegenIR::type_size_bytes(element->get());
        return create_device_buffer(_handle, elem_size, elem_count);
    });
#else
    LUISA_WARNING_WITH_LOCATION("IR is not enabled. Returning an invalid buffer.");
    return BufferCreationInfo::make_invalid();
#endif
}

void MetalDevice::destroy_buffer(uint64_t handle) noexcept {
    with_autorelease_pool([=] {
        auto buffer = reinterpret_cast<MetalBuffer *>(handle);
        delete_with_allocator(buffer);
    });
}

ResourceCreationInfo MetalDevice::create_texture(PixelFormat format, uint dimension,
                                                 uint width, uint height, uint depth,
                                                 uint mipmap_levels) noexcept {
    return with_autorelease_pool([=, this] {
        auto texture = new_with_allocator<MetalTexture>(
            _handle, format, dimension, width, height, depth, mipmap_levels);
        ResourceCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(texture);
        info.native_handle = texture;
        return info;
    });
}

void MetalDevice::destroy_texture(uint64_t handle) noexcept {
    with_autorelease_pool([=] {
        auto texture = reinterpret_cast<MetalTexture *>(handle);
        delete_with_allocator(texture);
    });
}

ResourceCreationInfo MetalDevice::create_bindless_array(size_t size) noexcept {
    return with_autorelease_pool([=, this] {
        auto array = new_with_allocator<MetalBindlessArray>(this, size);
        ResourceCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(array);
        info.native_handle = array;
        return info;
    });
}

void MetalDevice::destroy_bindless_array(uint64_t handle) noexcept {
    with_autorelease_pool([=] {
        auto array = reinterpret_cast<MetalBindlessArray *>(handle);
        delete_with_allocator(array);
    });
}

ResourceCreationInfo MetalDevice::create_stream(StreamTag stream_tag) noexcept {
    return with_autorelease_pool([=, this] {
        auto stream = new_with_allocator<MetalStream>(
            _handle, stream_tag, _inqueue_buffer_limit ? 4u : 0u);
        ResourceCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(stream);
        info.native_handle = stream;
        return info;
    });
}

void MetalDevice::destroy_stream(uint64_t handle) noexcept {
    with_autorelease_pool([=] {
        auto stream = reinterpret_cast<MetalStream *>(handle);
        delete_with_allocator(stream);
    });
}

void MetalDevice::synchronize_stream(uint64_t stream_handle) noexcept {
    with_autorelease_pool([=] {
        auto stream = reinterpret_cast<MetalStream *>(stream_handle);
        stream->synchronize();
    });
}

void MetalDevice::dispatch(uint64_t stream_handle, CommandList &&list) noexcept {
    with_autorelease_pool([stream_handle, &list] {
        auto stream = reinterpret_cast<MetalStream *>(stream_handle);
        stream->dispatch(std::move(list));
    });
}

SwapChainCreationInfo MetalDevice::create_swap_chain(uint64_t window_handle, uint64_t stream_handle,
                                                     uint width, uint height, bool allow_hdr,
                                                     bool vsync, uint back_buffer_size) noexcept {
    return with_autorelease_pool([=, this] {
        auto swapchain = new_with_allocator<MetalSwapchain>(
            this, window_handle, width, height,
            allow_hdr, vsync, back_buffer_size);
        SwapChainCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(swapchain);
        info.native_handle = swapchain;
        info.storage = swapchain->pixel_storage();
        return info;
    });
}

void MetalDevice::destroy_swap_chain(uint64_t handle) noexcept {
    with_autorelease_pool([=] {
        auto swpachain = reinterpret_cast<MetalSwapchain *>(handle);
        delete_with_allocator(swpachain);
    });
}

void MetalDevice::present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {
    with_autorelease_pool([=] {
        auto stream = reinterpret_cast<MetalStream *>(stream_handle);
        auto swapchain = reinterpret_cast<MetalSwapchain *>(swapchain_handle);
        auto image = reinterpret_cast<MetalTexture *>(image_handle);
        stream->present(swapchain, image);
    });
}

ShaderCreationInfo MetalDevice::create_shader(const ShaderOption &option, Function kernel) noexcept {
    return with_autorelease_pool([=, this] {
        return ShaderCreationInfo();
    });
}

ShaderCreationInfo MetalDevice::create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept {
    return with_autorelease_pool([=, this] {
        return ShaderCreationInfo();
    });
}

ShaderCreationInfo MetalDevice::load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept {
    return with_autorelease_pool([=, this] {
        return ShaderCreationInfo();
    });
}

Usage MetalDevice::shader_argument_usage(uint64_t handle, size_t index) noexcept {
    return Usage::NONE;
}

void MetalDevice::destroy_shader(uint64_t handle) noexcept {
    with_autorelease_pool([=] {
        // TODO
    });
}

ResourceCreationInfo MetalDevice::create_event() noexcept {
    return with_autorelease_pool([=, this] {
        auto event = new_with_allocator<MetalEvent>(_handle);
        ResourceCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(event);
        info.native_handle = event;
        return info;
    });
}

void MetalDevice::destroy_event(uint64_t handle) noexcept {
    with_autorelease_pool([=] {
        auto event = reinterpret_cast<MetalEvent *>(handle);
        delete_with_allocator(event);
    });
}

void MetalDevice::signal_event(uint64_t handle, uint64_t stream_handle) noexcept {
    with_autorelease_pool([=] {
        auto event = reinterpret_cast<MetalEvent *>(handle);
        auto stream = reinterpret_cast<MetalStream *>(stream_handle);
        stream->signal(event);
    });
}

void MetalDevice::wait_event(uint64_t handle, uint64_t stream_handle) noexcept {
    with_autorelease_pool([=] {
        auto event = reinterpret_cast<MetalEvent *>(handle);
        auto stream = reinterpret_cast<MetalStream *>(stream_handle);
        stream->wait(event);
    });
}

void MetalDevice::synchronize_event(uint64_t handle) noexcept {
    with_autorelease_pool([=] {
        auto event = reinterpret_cast<MetalEvent *>(handle);
        event->synchronize();
    });
}

ResourceCreationInfo MetalDevice::create_mesh(const AccelOption &option) noexcept {
    return with_autorelease_pool([=, this] {
        return ResourceCreationInfo();
    });
}

void MetalDevice::destroy_mesh(uint64_t handle) noexcept {
    with_autorelease_pool([=] {
        // TODO
    });
}

ResourceCreationInfo MetalDevice::create_procedural_primitive(const AccelOption &option) noexcept {
    return with_autorelease_pool([=, this] {
        return ResourceCreationInfo();
    });
}

void MetalDevice::destroy_procedural_primitive(uint64_t handle) noexcept {
    with_autorelease_pool([=] {
        // TODO
    });
}

ResourceCreationInfo MetalDevice::create_accel(const AccelOption &option) noexcept {
    return with_autorelease_pool([=, this] {
        auto accel = new_with_allocator<MetalAccel>(this, option);
        ResourceCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(accel);
        info.native_handle = accel;
        return info;
    });
}

void MetalDevice::destroy_accel(uint64_t handle) noexcept {
    with_autorelease_pool([=] {
        auto accel = reinterpret_cast<MetalAccel *>(handle);
        delete_with_allocator(accel);
    });
}

string MetalDevice::query(luisa::string_view property) noexcept {
    LUISA_WARNING_WITH_LOCATION("Device property \"{}\" is not supported on Metal.", property);
    return {};
}

DeviceExtension *MetalDevice::extension(luisa::string_view name) noexcept {
    LUISA_WARNING_WITH_LOCATION("Device extension \"{}\" is not supported on Metal.", name);
    return nullptr;
}

void MetalDevice::set_name(luisa::compute::Resource::Tag resource_tag,
                           uint64_t resource_handle, luisa::string_view name) noexcept {

    with_autorelease_pool([=] {
        switch (resource_tag) {
            case Resource::Tag::BUFFER: {
                auto buffer = reinterpret_cast<MetalBuffer *>(resource_handle);
                buffer->set_name(name);
                break;
            }
            case Resource::Tag::TEXTURE: {
                auto texture = reinterpret_cast<MetalTexture *>(resource_handle);
                texture->set_name(name);
                break;
            }
            case Resource::Tag::BINDLESS_ARRAY: break;
            case Resource::Tag::MESH: break;
            case Resource::Tag::PROCEDURAL_PRIMITIVE: break;
            case Resource::Tag::ACCEL: break;
            case Resource::Tag::STREAM: {
                auto stream = reinterpret_cast<MetalStream *>(resource_handle);
                stream->set_name(name);
                break;
            }
            case Resource::Tag::EVENT: {
                auto event = reinterpret_cast<MetalEvent *>(resource_handle);
                event->set_name(name);
                break;
            }
            case Resource::Tag::SHADER: break;
            case Resource::Tag::RASTER_SHADER: break;
            case Resource::Tag::SWAP_CHAIN: {
                auto swapchain = reinterpret_cast<MetalSwapchain *>(resource_handle);
                swapchain->set_name(name);
                break;
            }
            case Resource::Tag::DEPTH_BUFFER: break;
        }
    });
}

}// namespace luisa::compute::metal

LUISA_EXPORT_API luisa::compute::DeviceInterface *create(luisa::compute::Context &&ctx,
                                                         const luisa::compute::DeviceConfig *config) noexcept {
    return luisa::compute::metal::with_autorelease_pool([&] {
        return ::luisa::new_with_allocator<::luisa::compute::metal::MetalDevice>(std::move(ctx), config);
    });
}

LUISA_EXPORT_API void destroy(luisa::compute::DeviceInterface *device) noexcept {
    luisa::compute::metal::with_autorelease_pool([device] {
        auto p_device = dynamic_cast<::luisa::compute::metal::MetalDevice *>(device);
        LUISA_ASSERT(p_device != nullptr, "Invalid device.");
        ::luisa::delete_with_allocator(p_device);
    });
}

LUISA_EXPORT_API void backend_device_names(luisa::vector<luisa::string> &names) noexcept {
    ::luisa::compute::metal::with_autorelease_pool([&names] {
        names.clear();
        auto all_devices = MTL::CopyAllDevices();
        if (auto n = all_devices->count()) {
            names.reserve(n);
            for (auto i = 0u; i < n; i++) {
                auto device = all_devices->object<MTL::Device>(i);
                names.emplace_back(device->name()->utf8String());
            }
        }
        all_devices->release();
    });
}
