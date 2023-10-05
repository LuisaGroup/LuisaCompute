#include <luisa/core/clock.h>
#include <luisa/core/logging.h>

#ifdef LUISA_ENABLE_IR
#include "metal_codegen_ir.h"
#endif

#include "metal_builtin_embedded.h"
#include "metal_codegen_ast.h"
#include "metal_compiler.h"
#include "metal_buffer.h"
#include "metal_texture.h"
#include "metal_stream.h"
#include "metal_event.h"
#include "metal_swapchain.h"
#include "metal_bindless_array.h"
#include "metal_accel.h"
#include "metal_mesh.h"
#include "metal_procedural_primitive.h"
#include "metal_shader.h"
#include "metal_device.h"

// extensions
#include "metal_dstorage.h"
#include "metal_pinned_memory.h"
#include "metal_debug_capture.h"

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
    builtin_library->setLabel(MTLSTR("luisa_builtin"));

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
    auto create_builtin_compute_shader = [&](auto name, auto block_size) noexcept {
        compute_pipeline_desc->setMaxTotalThreadsPerThreadgroup(block_size);
        auto function_desc = MTL::FunctionDescriptor::alloc()->init();
        function_desc->setName(name);
        function_desc->setOptions(MTL::FunctionOptionCompileToBinary);
        auto function = builtin_library->newFunction(function_desc, &error);
        function_desc->release();
        if (error != nullptr) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to compile built-in Metal kernel '{}': {}",
                name->utf8String(), error->localizedDescription()->utf8String());
        }
        error = nullptr;
        LUISA_ASSERT(function != nullptr,
                     "Failed to compile built-in Metal kernel '{}'.",
                     name->utf8String());
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
    _builtin_update_bindless_slots = create_builtin_compute_shader(
        MTLSTR("update_bindless_array"), update_bindless_slots_block_size);
    _builtin_update_accel_instances = create_builtin_compute_shader(
        MTLSTR("update_accel_instances"), update_accel_instances_block_size);
    _builtin_prepare_indirect_dispatches = create_builtin_compute_shader(
        MTLSTR("prepare_indirect_dispatches"), prepare_indirect_dispatches_block_size);
    compute_pipeline_desc->release();

    // render pipeline
    auto create_builtin_raster_shader = [&](auto name) noexcept {
        auto shader_desc = MTL::FunctionDescriptor::alloc()->init();
        shader_desc->setName(name);
        shader_desc->setOptions(MTL::FunctionOptionCompileToBinary);
        auto shader = builtin_library->newFunction(shader_desc, &error);
        shader_desc->release();
        if (error != nullptr) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to compile built-in Metal vertex shader '{}': {}",
                name->utf8String(), error->localizedDescription()->utf8String());
        }
        error = nullptr;
        LUISA_ASSERT(shader != nullptr,
                     "Failed to compile built-in Metal rasterization shader '{}'.",
                     name->utf8String());
        return shader;
    };
    auto builtin_swapchain_vertex_shader = create_builtin_raster_shader(MTLSTR("swapchain_vertex_shader"));
    auto builtin_swapchain_fragment_shader = create_builtin_raster_shader(MTLSTR("swapchain_fragment_shader"));

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
    _builtin_swapchain_present_ldr = create_builtin_present_shader(MTL::PixelFormatBGRA8Unorm);
    _builtin_swapchain_present_hdr = create_builtin_present_shader(MTL::PixelFormatRGBA16Float);
    render_pipeline_desc->release();
    builtin_swapchain_vertex_shader->release();
    builtin_swapchain_fragment_shader->release();

    builtin_library->release();

    LUISA_INFO("Created Metal device '{}' at index {}.",
               _handle->name()->utf8String(), device_index);
}

MetalDevice::~MetalDevice() noexcept {
    _builtin_update_bindless_slots->release();
    _builtin_update_accel_instances->release();
    _builtin_prepare_indirect_dispatches->release();
    _builtin_swapchain_present_ldr->release();
    _builtin_swapchain_present_hdr->release();
    _handle->release();
}

void *MetalDevice::native_handle() const noexcept {
    return _handle;
}

uint MetalDevice::compute_warp_size() const noexcept {
    return _builtin_update_bindless_slots->threadExecutionWidth();
}

[[nodiscard]] inline auto create_device_buffer(MTL::Device *device, size_t element_stride, size_t element_count) noexcept {
    auto buffer_size = element_stride * element_count;
    auto buffer = new_with_allocator<MetalBuffer>(device, buffer_size);
    BufferCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(buffer);
    info.native_handle = buffer->handle();
    info.element_stride = element_stride;
    info.total_size_bytes = buffer_size;
    return info;
}

BufferCreationInfo MetalDevice::create_buffer(const Type *element, size_t elem_count) noexcept {
    return with_autorelease_pool([=, this] {
        // special handling of the indirect dispatch buffer
        if (element == Type::of<IndirectKernelDispatch>()) {
            auto p = new_with_allocator<MetalIndirectDispatchBuffer>(_handle, elem_count);
            BufferCreationInfo info{};
            info.handle = reinterpret_cast<uint64_t>(p);
            info.native_handle = p->dispatch_buffer();
            info.element_stride = sizeof(MetalIndirectDispatchBuffer::Dispatch);
            info.total_size_bytes = p->dispatch_buffer()->length();
            return info;
        }
        if (element == Type::of<void>()) {
            return create_device_buffer(_handle, 1u, elem_count);
        }
        // normal buffer
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
        auto buffer = reinterpret_cast<MetalBufferBase *>(handle);
        delete_with_allocator(buffer);
    });
}

ResourceCreationInfo MetalDevice::create_texture(PixelFormat format, uint dimension,
                                                 uint width, uint height, uint depth, uint mipmap_levels,
                                                 bool allow_simultaneous_access) noexcept {
    return with_autorelease_pool([=, this] {
        auto texture = new_with_allocator<MetalTexture>(
            _handle, format, dimension, width, height, depth,
            mipmap_levels, allow_simultaneous_access);
        ResourceCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(texture);
        info.native_handle = texture->handle();
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
        info.native_handle = array->handle();
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
            _handle, _inqueue_buffer_limit ? 4u : 0u);
        ResourceCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(stream);
        info.native_handle = stream->queue();
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

SwapchainCreationInfo MetalDevice::create_swapchain(uint64_t window_handle, uint64_t stream_handle,
                                                    uint width, uint height, bool allow_hdr,
                                                    bool vsync, uint back_buffer_size) noexcept {
    return with_autorelease_pool([=, this] {
        auto swapchain = new_with_allocator<MetalSwapchain>(
            this, window_handle, width, height,
            allow_hdr, vsync, back_buffer_size);
        SwapchainCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(swapchain);
        info.native_handle = swapchain->layer();
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
        MetalShaderMetadata metadata{};
        metadata.block_size = kernel.block_size();
        metadata.argument_types.reserve(kernel.arguments().size());
        metadata.argument_usages.reserve(kernel.arguments().size());
        for (auto &&arg : kernel.arguments()) {
            metadata.argument_types.emplace_back(arg.type()->description());
            metadata.argument_usages.emplace_back(kernel.variable_usage(arg.uid()));
        }
        luisa::vector<MetalShader::Argument> bound_arguments;
        bound_arguments.reserve(kernel.bound_arguments().size());
        for (auto &&binding : kernel.bound_arguments()) {
            luisa::visit([&bound_arguments](auto b) noexcept {
                using T = std::remove_cvref_t<decltype(b)>;
                MetalShader::Argument argument{};
                if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                    argument.tag = MetalShader::Argument::Tag::BUFFER;
                    argument.buffer.handle = b.handle;
                    argument.buffer.offset = b.offset;
                    argument.buffer.size = b.size;
                } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                    argument.tag = MetalShader::Argument::Tag::TEXTURE;
                    argument.texture.handle = b.handle;
                    argument.texture.level = b.level;
                } else if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                    argument.tag = MetalShader::Argument::Tag::BINDLESS_ARRAY;
                    argument.bindless_array.handle = b.handle;
                } else if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                    argument.tag = MetalShader::Argument::Tag::ACCEL;
                    argument.accel.handle = b.handle;
                } else {
                    LUISA_ERROR_WITH_LOCATION("Invalid binding type.");
                }
                bound_arguments.emplace_back(argument);
            },
                         binding);
        }

        // codegen
        StringScratch scratch;
        MetalCodegenAST codegen{scratch};
        codegen.emit(kernel, option.native_include);

        // create shader
        auto pipeline = _compiler->compile(scratch.string_view(), option, metadata);
        auto shader = luisa::new_with_allocator<MetalShader>(
            this, std::move(pipeline),
            std::move(metadata.argument_usages),
            std::move(bound_arguments),
            kernel.block_size());
        ShaderCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(shader);
        info.native_handle = shader->pso();
        info.block_size = kernel.block_size();
        return info;
    });
}

ShaderCreationInfo MetalDevice::create_shader(const ShaderOption &option, const ir::KernelModule *kernel) noexcept {
    // TODO: codegen from IR directly
    return with_autorelease_pool([=, this] {
#ifdef LUISA_ENABLE_IR
        Clock clk;
        auto function = IR2AST::build(kernel);
        LUISA_VERBOSE("IR2AST done in {} ms.", clk.toc());
        return create_shader(option, function->function());
#else
        LUISA_ERROR_WITH_LOCATION("Metal device does not support creating shader from IR types.");
        return ShaderCreationInfo{};
#endif
    });
}

ShaderCreationInfo MetalDevice::load_shader(luisa::string_view name, luisa::span<const Type *const> arg_types) noexcept {
    return with_autorelease_pool([=, this] {
        MetalShaderMetadata metadata{};
        auto pipeline = _compiler->load(name, metadata);
        LUISA_ASSERT(pipeline.entry && pipeline.indirect_entry,
                     "Failed to load Metal AOT shader '{}'.", name);
        LUISA_ASSERT(metadata.argument_types.size() == arg_types.size(),
                     "Argument count mismatch in Metal AOT "
                     "shader '{}': expected {}, but got {}.",
                     name, metadata.argument_types.size(), arg_types.size());
        for (auto i = 0u; i < arg_types.size(); i++) {
            LUISA_ASSERT(metadata.argument_types[i] == arg_types[i]->description(),
                         "Argument type mismatch in Metal AOT "
                         "shader '{}': expected {}, but got {}.",
                         name, metadata.argument_types[i],
                         arg_types[i]->description());
        }
        auto shader = new_with_allocator<MetalShader>(
            this, std::move(pipeline),
            std::move(metadata.argument_usages),
            luisa::vector<MetalShader::Argument>{},
            metadata.block_size);
        ShaderCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(shader);
        info.native_handle = shader->pso();
        info.block_size = metadata.block_size;
        return info;
    });
}

Usage MetalDevice::shader_argument_usage(uint64_t handle, size_t index) noexcept {
    auto shader = reinterpret_cast<MetalShader *>(handle);
    return shader->argument_usage(index);
}

void MetalDevice::destroy_shader(uint64_t handle) noexcept {
    with_autorelease_pool([=] {
        auto shader = reinterpret_cast<MetalShader *>(handle);
        luisa::delete_with_allocator(shader);
    });
}

ResourceCreationInfo MetalDevice::create_event() noexcept {
    return with_autorelease_pool([=, this] {
        auto event = new_with_allocator<MetalEvent>(_handle);
        ResourceCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(event);
        info.native_handle = event->handle();
        return info;
    });
}

void MetalDevice::destroy_event(uint64_t handle) noexcept {
    with_autorelease_pool([=] {
        auto event = reinterpret_cast<MetalEvent *>(handle);
        delete_with_allocator(event);
    });
}

void MetalDevice::signal_event(uint64_t handle, uint64_t stream_handle, uint64_t value) noexcept {
    // TODO: fence not implemented
    with_autorelease_pool([=] {
        auto event = reinterpret_cast<MetalEvent *>(handle);
        auto stream = reinterpret_cast<MetalStream *>(stream_handle);
        stream->signal(event, value);
    });
}

void MetalDevice::wait_event(uint64_t handle, uint64_t stream_handle, uint64_t value) noexcept {
    // TODO: fence not implemented
    with_autorelease_pool([=] {
        auto event = reinterpret_cast<MetalEvent *>(handle);
        auto stream = reinterpret_cast<MetalStream *>(stream_handle);
        stream->wait(event, value);
    });
}

void MetalDevice::synchronize_event(uint64_t handle, uint64_t value) noexcept {
    // TODO: fence not implemented
    with_autorelease_pool([=] {
        auto event = reinterpret_cast<MetalEvent *>(handle);
        event->synchronize(value);
    });
}

bool MetalDevice::is_event_completed(uint64_t handle, uint64_t value) const noexcept {
    // TODO: fence not implemented
    return with_autorelease_pool([=] {
        auto event = reinterpret_cast<MetalEvent *>(handle);
        return event->is_completed(value);
    });
}

ResourceCreationInfo MetalDevice::create_mesh(const AccelOption &option) noexcept {
    return with_autorelease_pool([=, this] {
        auto mesh = new_with_allocator<MetalMesh>(_handle, option);
        ResourceCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(mesh);
        info.native_handle = mesh->pointer_to_handle();
        return info;
    });
}

void MetalDevice::destroy_mesh(uint64_t handle) noexcept {
    with_autorelease_pool([=] {
        auto mesh = reinterpret_cast<MetalMesh *>(handle);
        delete_with_allocator(mesh);
    });
}

ResourceCreationInfo MetalDevice::create_procedural_primitive(const AccelOption &option) noexcept {
    return with_autorelease_pool([=, this] {
        auto primitive = new_with_allocator<MetalProceduralPrimitive>(_handle, option);
        ResourceCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(primitive);
        info.native_handle = primitive->pointer_to_handle();
        return info;
    });
}

void MetalDevice::destroy_procedural_primitive(uint64_t handle) noexcept {
    with_autorelease_pool([=] {
        auto primitive = reinterpret_cast<MetalProceduralPrimitive *>(handle);
        delete_with_allocator(primitive);
    });
}

ResourceCreationInfo MetalDevice::create_accel(const AccelOption &option) noexcept {
    return with_autorelease_pool([=, this] {
        auto accel = new_with_allocator<MetalAccel>(this, option);
        ResourceCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(accel);
        info.native_handle = accel->pointer_to_handle();
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
    return with_autorelease_pool([=, this]() noexcept -> DeviceExtension * {
        if (name == DStorageExt::name) {
            std::scoped_lock lock{_ext_mutex};
            if (!_dstorage_ext) { _dstorage_ext = luisa::make_unique<MetalDStorageExt>(this); }
            return _dstorage_ext.get();
        }
        if (name == PinnedMemoryExt::name) {
            std::scoped_lock lock{_ext_mutex};
            if (!_pinned_memory_ext) { _pinned_memory_ext = luisa::make_unique<MetalPinnedMemoryExt>(this); }
            return _pinned_memory_ext.get();
        }
        if (name == DebugCaptureExt::name) {
            std::scoped_lock lock{_ext_mutex};
            if (!_debug_capture_ext) { _debug_capture_ext = luisa::make_unique<MetalDebugCaptureExt>(this); }
            return _debug_capture_ext.get();
        }
        LUISA_WARNING_WITH_LOCATION("Device extension \"{}\" is not supported on Metal.", name);
        return nullptr;
    });
}

void MetalDevice::set_name(luisa::compute::Resource::Tag resource_tag,
                           uint64_t resource_handle, luisa::string_view name) noexcept {

    with_autorelease_pool([=] {
        switch (resource_tag) {
            case Resource::Tag::BUFFER: {
                auto buffer = reinterpret_cast<MetalBufferBase *>(resource_handle);
                buffer->set_name(name);
                break;
            }
            case Resource::Tag::TEXTURE: {
                auto texture = reinterpret_cast<MetalTexture *>(resource_handle);
                texture->set_name(name);
                break;
            }
            case Resource::Tag::BINDLESS_ARRAY: {
                auto bindless_array = reinterpret_cast<MetalBindlessArray *>(resource_handle);
                bindless_array->set_name(name);
                break;
            }
            case Resource::Tag::MESH: {
                auto mesh = reinterpret_cast<MetalMesh *>(resource_handle);
                mesh->set_name(name);
                break;
            }
            case Resource::Tag::PROCEDURAL_PRIMITIVE: {
                auto prim = reinterpret_cast<MetalProceduralPrimitive *>(resource_handle);
                prim->set_name(name);
                break;
            }
            case Resource::Tag::ACCEL: {
                auto accel = reinterpret_cast<MetalAccel *>(resource_handle);
                accel->set_name(name);
                break;
            }
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
            case Resource::Tag::SHADER: {
                auto shader = reinterpret_cast<MetalShader *>(resource_handle);
                shader->set_name(name);
                break;
            }
            case Resource::Tag::RASTER_SHADER: {
                // TODO
                break;
            }
            case Resource::Tag::SWAP_CHAIN: {
                auto swapchain = reinterpret_cast<MetalSwapchain *>(resource_handle);
                swapchain->set_name(name);
                break;
            }
            case Resource::Tag::DEPTH_BUFFER: {
                // TODO
                break;
            }
            case Resource::Tag::DSTORAGE_FILE: {
                auto file = reinterpret_cast<MetalFileHandle *>(resource_handle);
                file->set_name(name);
                break;
            }
            case Resource::Tag::DSTORAGE_PINNED_MEMORY: {
                auto mem = reinterpret_cast<MetalPinnedMemory *>(resource_handle);
                mem->set_name(name);
                break;
            }
            case Resource::Tag::SPARSE_BUFFER: break;
            case Resource::Tag::SPARSE_TEXTURE: break;
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
