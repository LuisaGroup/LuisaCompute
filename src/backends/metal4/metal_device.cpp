#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/ast/statement.h>

#include "metal_builtin_air.h"
#include "metal_compiler.h"
#include "metal_buffer.h"
#include "metal_texture.h"
#include "metal_stream.h"
#include "metal_event.h"
#include "metal_swapchain.h"
#include "metal_bindless_array.h"
#include "metal_accel.h"
#include "metal_mesh.h"
#include "metal_curve.h"
#include "metal_procedural_primitive.h"
#include "metal_motion_instance.h"
#include "metal_shader.h"
#include "metal_depth_buffer.h"
#include "metal_raster_ext.h"
#include "metal_raster_shader.h"
#include "metal_device.h"
#include "metal_static_backend.h"

#include "llvm_codegen/metal_codegen_llvm.h"
#include "metal_air_pipeline.h"
#include "metal_xir_pipeline.h"

// extensions
#include "metal_denoiser.h"
#include "metal_dstorage.h"
#include "metal_pinned_memory.h"
#include "metal_debug_capture.h"
#include "metal_tex_compress.h"
#ifdef LUISA_ENABLE_XIR
#include "../common/xir_autodiff.h"
#endif

#include <cstdlib>
#include <algorithm>

namespace luisa::compute::metal {

namespace {

class SampledTextureArgumentAnalysis {

private:
    luisa::unordered_map<uint64_t, luisa::unordered_set<uint32_t>> _sampled;

private:
    [[nodiscard]] bool _is_sampled(Function function, Variable variable) const noexcept {
        if (auto iter = _sampled.find(function.hash()); iter != _sampled.end()) {
            return iter->second.contains(variable.uid());
        }
        return false;
    }

    void _analyze(Function function) noexcept {
        auto [iter, inserted] = _sampled.try_emplace(function.hash());
        if (!inserted) { return; }
        for (auto callable : function.custom_callables()) {
            _analyze(callable->function());
        }
        auto &&sampled = iter->second;
        auto mark_argument = [&sampled](const Expression *expression) noexcept {
            if (expression->tag() != Expression::Tag::REF) { return; }
            auto variable = static_cast<const RefExpr *>(expression)->variable();
            if (variable.type()->is_texture()) { sampled.emplace(variable.uid()); }
        };
        traverse_expressions<true>(
            function.body(),
            [&](const Expression *expression) noexcept {
                if (expression->tag() != Expression::Tag::CALL) { return; }
                auto call = static_cast<const CallExpr *>(expression);
                switch (call->op()) {
                    case CallOp::TEXTURE2D_SAMPLE: [[fallthrough]];
                    case CallOp::TEXTURE2D_SAMPLE_LEVEL: [[fallthrough]];
                    case CallOp::TEXTURE2D_SAMPLE_GRAD: [[fallthrough]];
                    case CallOp::TEXTURE2D_SAMPLE_GRAD_LEVEL: [[fallthrough]];
                    case CallOp::TEXTURE3D_SAMPLE: [[fallthrough]];
                    case CallOp::TEXTURE3D_SAMPLE_LEVEL: [[fallthrough]];
                    case CallOp::TEXTURE3D_SAMPLE_GRAD: [[fallthrough]];
                    case CallOp::TEXTURE3D_SAMPLE_GRAD_LEVEL:
                        mark_argument(call->arguments().front());
                        break;
                    case CallOp::CUSTOM: {
                        auto callable = call->custom();
                        for (auto i = 0u; i < callable.arguments().size(); i++) {
                            if (_is_sampled(callable, callable.arguments()[i])) {
                                mark_argument(call->arguments()[i]);
                            }
                        }
                        break;
                    }
                    default: break;
                }
            },
            [](auto) noexcept {},
            [](auto) noexcept {});
    }

public:
    [[nodiscard]] luisa::vector<uint8_t> analyze(Function kernel) noexcept {
        _analyze(kernel);
        luisa::vector<uint8_t> result;
        result.reserve(kernel.arguments().size());
        for (auto argument : kernel.arguments()) {
            result.emplace_back(
                argument.type()->is_texture() && _is_sampled(kernel, argument));
        }
        return result;
    }
};

}// namespace

MetalDevice::MetalDevice(Context &&ctx, const DeviceConfig *config) noexcept
    : DeviceInterface{std::move(ctx)}, _io{nullptr},
      _inqueue_buffer_limit{config == nullptr || config->inqueue_buffer_limit} {

    auto device_index = config == nullptr ||
                                config->device_index == std::numeric_limits<size_t>::max() ?
                            0u :
                            config->device_index;
#if defined(LUISA_PLATFORM_IOS)
    if (!__builtin_available(iOS 26.0, *)) {
        LUISA_ERROR_WITH_LOCATION(
            "The Metal 4 backend requires iOS 26.0 or newer.");
    }
    auto compatible_device_count = static_cast<size_t>(0u);
    if (auto device = MTL::CreateSystemDefaultDevice()) {
        if (device->supportsFamily(MTL::GPUFamilyMetal4)) {
            compatible_device_count = 1u;
            if (device_index == 0u) { _handle = device; }
        }
        if (_handle == nullptr) { device->release(); }
    }
#else
    if (!__builtin_available(macOS 26.0, *)) {
        LUISA_ERROR_WITH_LOCATION(
            "The Metal 4 backend requires macOS 26.0 or newer.");
    }
    auto all_devices = MTL::CopyAllDevices();
    auto compatible_device_count = static_cast<size_t>(0u);
    for (auto i = static_cast<NS::UInteger>(0u); i < all_devices->count(); i++) {
        auto device = all_devices->object<MTL::Device>(i);
        if (!device->supportsFamily(MTL::GPUFamilyMetal4)) { continue; }
        if (compatible_device_count++ == device_index) {
            _handle = device->retain();
        }
    }
    all_devices->release();
#endif
    LUISA_ASSERT(
        _handle != nullptr,
        "Metal 4 device index out of range (required = {}, count = {}).",
        device_index, compatible_device_count);

    NS::Error *error{nullptr};
    auto metal4_compiler_desc = MTL4::CompilerDescriptor::alloc()->init();
    metal4_compiler_desc->setLabel(MTLSTR("LuisaCompute Metal 4 compiler"));
    _metal4_compiler = _handle->newCompiler(metal4_compiler_desc, &error);
    metal4_compiler_desc->release();
    if (error != nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to create Metal 4 compiler: {}.",
            error->localizedDescription()->utf8String());
    }
    LUISA_ASSERT(_metal4_compiler != nullptr,
                 "Failed to create Metal 4 compiler for device '{}'.",
                 _handle->name()->utf8String());

    // create a default binary IO if none is provided
    if (config == nullptr || config->binary_io == nullptr) {
        auto headless = config != nullptr && config->headless;
        auto use_lmdb = config != nullptr && config->use_lmdb;
        _default_io = luisa::make_unique<DefaultBinaryIO>(
            context(), headless, use_lmdb);
        _io = _default_io.get();
    } else {
        _io = config->binary_io;
    }

    // create a compiler
    _compiler = luisa::make_unique<MetalCompiler>(this);

    // Generate the fixed runtime-support library through the same LLVM/AIR
    // path as user shaders. The Metal 4 backend never invokes an MSL compiler.
    auto builtin_metallib = metal_codegen_builtin_air(
        metal_air_target_for_current_device());
    auto builtin_library_data = dispatch_data_create(
        builtin_metallib.data(), builtin_metallib.size(), nullptr,
        DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    error = nullptr;
    auto builtin_library = _handle->newLibrary(
        builtin_library_data, &error);
    dispatch_release(builtin_library_data);

    if (error != nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to load built-in Metal AIR library: {}",
            error->localizedDescription()->utf8String());
    }
    error = nullptr;
    LUISA_ASSERT(builtin_library != nullptr,
                 "Failed to load built-in Metal AIR library.");
    builtin_library->setLabel(MTLSTR("luisa_builtin"));

    // compute pipelines
    auto compute_pipeline_desc = MTL4::ComputePipelineDescriptor::alloc()->init();
    auto compute_function_desc = MTL4::LibraryFunctionDescriptor::alloc()->init();
    compute_function_desc->setLibrary(builtin_library);
    compute_pipeline_desc->setThreadGroupSizeIsMultipleOfThreadExecutionWidth(true);
    auto create_builtin_compute_shader = [&](auto name, auto block_size) noexcept {
        compute_pipeline_desc->setMaxTotalThreadsPerThreadgroup(block_size);
        compute_pipeline_desc->setLabel(name);
        compute_function_desc->setName(name);
        // MTL4ComputePipelineDescriptor copies its function descriptor. Set the
        // function name before assigning it so that the copied descriptor is
        // complete (and repeat the assignment when reusing the descriptor).
        compute_pipeline_desc->setComputeFunctionDescriptor(compute_function_desc);
        error = nullptr;
        auto pipeline = _metal4_compiler->newComputePipelineState(
            compute_pipeline_desc, nullptr, &error);
        if (error != nullptr) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to compile built-in Metal 4 kernel '{}': {}",
                name->utf8String(), error->localizedDescription()->utf8String());
        }
        LUISA_ASSERT(pipeline != nullptr,
                     "Failed to compile built-in Metal 4 kernel '{}'.",
                     name->utf8String());
        return pipeline;
    };
    _builtin_update_bindless_slots = create_builtin_compute_shader(
        MTLSTR("update_bindless_array"), update_bindless_slots_block_size);
    _builtin_update_accel_instances = create_builtin_compute_shader(
        MTLSTR("update_accel_instances"), update_accel_instances_block_size);
    _builtin_prepare_indirect_dispatches = create_builtin_compute_shader(
        MTLSTR("prepare_indirect_dispatches"), prepare_indirect_dispatches_block_size);
    compute_function_desc->release();
    compute_pipeline_desc->release();

    // render pipeline
    auto builtin_swapchain_vertex_shader = MTL4::LibraryFunctionDescriptor::alloc()->init();
    builtin_swapchain_vertex_shader->setLibrary(builtin_library);
    builtin_swapchain_vertex_shader->setName(MTLSTR("swapchain_vertex_shader"));
    auto builtin_swapchain_fragment_shader = MTL4::LibraryFunctionDescriptor::alloc()->init();
    builtin_swapchain_fragment_shader->setLibrary(builtin_library);
    builtin_swapchain_fragment_shader->setName(MTLSTR("swapchain_fragment_shader"));

    auto render_pipeline_desc = MTL4::RenderPipelineDescriptor::alloc()->init();
    render_pipeline_desc->setVertexFunctionDescriptor(builtin_swapchain_vertex_shader);
    render_pipeline_desc->setFragmentFunctionDescriptor(builtin_swapchain_fragment_shader);
    auto color_attachment = render_pipeline_desc->colorAttachments()->object(0u);
    color_attachment->setBlendingState(MTL4::BlendStateDisabled);
    auto create_builtin_present_shader = [&](auto format) noexcept {
        color_attachment->setPixelFormat(format);
        error = nullptr;
        auto shader = _metal4_compiler->newRenderPipelineState(
            render_pipeline_desc, nullptr, &error);
        if (error != nullptr) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to compile built-in Metal 4 present pipeline: {}",
                error->localizedDescription()->utf8String());
        }
        LUISA_ASSERT(shader != nullptr,
                     "Failed to compile built-in Metal 4 present pipeline.");
        return shader;
    };
    _builtin_swapchain_present_ldr = create_builtin_present_shader(MTL::PixelFormatBGRA8Unorm);
    _builtin_swapchain_present_hdr = create_builtin_present_shader(MTL::PixelFormatRGBA16Float);
    render_pipeline_desc->release();
    builtin_swapchain_vertex_shader->release();
    builtin_swapchain_fragment_shader->release();

    builtin_library->release();

    LUISA_INFO("Created Metal 4 device '{}' at index {}.",
               _handle->name()->utf8String(), device_index);
}

MetalDevice::~MetalDevice() noexcept {
    _compiler.reset();
    _builtin_update_bindless_slots->release();
    _builtin_update_accel_instances->release();
    _builtin_prepare_indirect_dispatches->release();
    _builtin_swapchain_present_ldr->release();
    _builtin_swapchain_present_hdr->release();
    _metal4_compiler->release();
    _handle->release();
}

void *MetalDevice::native_handle() const noexcept {
    return _handle;
}

uint MetalDevice::compute_warp_size() const noexcept {
    return _builtin_update_bindless_slots->threadExecutionWidth();
}

uint64_t MetalDevice::memory_granularity() const noexcept {
    return 65536ull;// TODO
}

[[nodiscard]] inline auto create_device_buffer(MTL::Device *device,
                                               size_t element_stride,
                                               size_t element_count,
                                               void *external_memory) noexcept {
    auto buffer_size = element_stride * element_count;
    auto buffer = [&] {
        if (external_memory) {
            auto mtl_buffer = static_cast<MTL::Buffer *>(external_memory);
            LUISA_ASSERT(mtl_buffer->length() >= buffer_size, "External memory is not large enough.");
            return new_with_allocator<MetalBuffer>(mtl_buffer);
        }
        return new_with_allocator<MetalBuffer>(device, buffer_size);
    }();
    BufferCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(buffer);
    info.native_handle = buffer->handle();
    info.element_stride = element_stride;
    info.total_size_bytes = buffer_size;
    return info;
}

[[nodiscard]] inline auto create_device_buffer_from_external_memory(MTL::Buffer *external_buffer,
                                                                    size_t total_size,
                                                                    size_t element_stride) noexcept {
    auto buffer = new_with_allocator<MetalBuffer>(external_buffer);
    BufferCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(buffer);
    info.native_handle = buffer->handle();
    info.element_stride = element_stride;
    info.total_size_bytes = total_size;
    return info;
}

BufferCreationInfo MetalDevice::create_buffer(const Type *element,
                                              size_t elem_count,
                                              void *external_memory) noexcept {
    return with_autorelease_pool([=, this] {
        if (element == Type::of<void>()) {
            return create_device_buffer(_handle, 1u, elem_count, external_memory);
        }
        if (element->is_custom()) {
            // special handling of the indirect dispatch buffer
            if (element == Type::of<IndirectKernelDispatch>()) {
                LUISA_ASSERT(external_memory == nullptr,
                             "External memory is not supported "
                             "for indirect dispatch buffer.");
                auto p = new_with_allocator<MetalIndirectDispatchBuffer>(_handle, elem_count);
                BufferCreationInfo info{};
                info.handle = reinterpret_cast<uint64_t>(p);
                info.native_handle = p->dispatch_buffer();
                info.element_stride = sizeof(MetalIndirectDispatchBuffer::Dispatch);
                info.total_size_bytes = p->dispatch_buffer()->length();
                return info;
            }
            LUISA_ERROR_WITH_LOCATION("Invalid custom buffer type: {}",
                                      element->description());
        }
        // normal buffer
        auto elem_size = element->size();
        return create_device_buffer(_handle, elem_size, elem_count, external_memory);
    });
}

void MetalDevice::destroy_buffer(uint64_t handle) noexcept {
    with_autorelease_pool([=] {
        auto buffer = reinterpret_cast<MetalBufferBase *>(handle);
        delete_with_allocator(buffer);
    });
}

ResourceCreationInfo MetalDevice::create_texture(PixelFormat format, uint dimension,
                                                 uint width, uint height, uint depth,
                                                 uint mipmap_levels, void *external_native_handle,
                                                 bool allow_simultaneous_access, bool allow_raster_target) noexcept {
    LUISA_ASSERT(external_native_handle == nullptr, "Not implemented.");
    return with_autorelease_pool([=, this] {
        auto texture = new_with_allocator<MetalTexture>(
            _handle, format, dimension, width, height, depth,
            mipmap_levels, allow_simultaneous_access, allow_raster_target);
        ResourceCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(
            static_cast<MetalTextureBase *>(texture));
        info.native_handle = texture->handle();
        return info;
    });
}

void MetalDevice::destroy_texture(uint64_t handle) noexcept {
    with_autorelease_pool([=] {
        auto texture_base = reinterpret_cast<MetalTextureBase *>(handle);
        LUISA_ASSERT(texture_base->kind() == MetalTextureBase::Kind::TEXTURE,
                     "Attempting to destroy a non-color Metal texture as an image.");
        auto texture = static_cast<MetalTexture *>(texture_base);
        delete_with_allocator(texture);
    });
}

ResourceCreationInfo MetalDevice::create_bindless_array(size_t size, BindlessSlotType type) noexcept {
    LUISA_ASSERT(type == BindlessSlotType::MULTIPLE);
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

void MetalDevice::set_stream_log_callback(
    uint64_t stream_handle,
    const StreamLogCallback &callback) noexcept {
    auto stream = reinterpret_cast<MetalStream *>(stream_handle);
    stream->set_log_callback(callback);
}

void MetalDevice::dispatch(uint64_t stream_handle, CommandList &&list) noexcept {
    with_autorelease_pool([stream_handle, &list] {
        auto stream = reinterpret_cast<MetalStream *>(stream_handle);
        stream->dispatch(std::move(list));
    });
}

SwapchainCreationInfo MetalDevice::create_swapchain(const SwapchainOption &option, uint64_t stream_handle) noexcept {
    return with_autorelease_pool([=, this] {
        auto swapchain = new_with_allocator<MetalSwapchain>(
            this, option.window, option.size.x, option.size.y,
            option.wants_hdr, option.wants_vsync, option.back_buffer_count);
        SwapchainCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(swapchain);
        info.native_handle = swapchain->layer();
        info.storage = swapchain->pixel_storage();
        return info;
    });
}

void MetalDevice::destroy_swapchain(uint64_t handle) noexcept {
    with_autorelease_pool([=] {
        auto swpachain = reinterpret_cast<MetalSwapchain *>(handle);
        delete_with_allocator(swpachain);
    });
}

void MetalDevice::present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {
    with_autorelease_pool([=] {
        auto stream = reinterpret_cast<MetalStream *>(stream_handle);
        auto swapchain = reinterpret_cast<MetalSwapchain *>(swapchain_handle);
        auto image_base = reinterpret_cast<MetalTextureBase *>(image_handle);
        LUISA_ASSERT(image_base->kind() == MetalTextureBase::Kind::TEXTURE,
                     "A Metal swap chain can only present a color texture.");
        auto image = static_cast<MetalTexture *>(image_base);
        stream->present(swapchain, image);
    });
}

ShaderCreationInfo MetalDevice::create_shader(const ShaderOption &option, Function kernel) noexcept {
    if (kernel.allowed_warp_size().value_or(32) != 32) [[unlikely]] {
        LUISA_ERROR("Metal4 backend only supports warp size 32.");
    }

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
            luisa::visit(
                [&bound_arguments](auto b) noexcept {
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

        MetalShaderHandle pipeline;
        size_t generated_size_bytes = 0u;
        size_t generated_line_count = 0u;
        double codegen_ms = 0.0;
        double compile_ms = 0.0;
        Clock codegen_clock;
        metadata.argument_sampled =
            SampledTextureArgumentAnalysis{}.analyze(kernel);
        auto xir_module = metal_translate_ast_to_xir(kernel, option);
        luisa::string unsupported_reason;
        if (!luisa_compute_metal_codegen_llvm_supported(
                *xir_module, &unsupported_reason)) {
            LUISA_ERROR_WITH_LOCATION(
                "Metal4 LLVM/AIR code generation does not support this shader yet: {}.",
                unsupported_reason);
        }
        auto air = metal_codegen_air(*xir_module, option);
        codegen_ms = codegen_clock.toc();
        generated_size_bytes = air.library.size();
        metadata.format_types = std::move(air.format_types);
        Clock compile_clock;
        pipeline = _compiler->compile(air.library, option, metadata);
        compile_ms = compile_clock.toc();
        LUISA_ASSERT(pipeline.entry && pipeline.indirect_entry,
                     "Metal4 LLVM/AIR compilation failed to create both compute entry points.");
        auto shader = luisa::new_with_allocator<MetalShader>(
            this, std::move(pipeline),
            std::move(metadata.argument_usages),
            std::move(metadata.argument_sampled),
            std::move(bound_arguments),
            std::move(metadata.format_types),
            kernel.block_size(), metadata.checksum,
            generated_size_bytes, generated_line_count,
            codegen_ms, compile_ms);
        ShaderCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(shader);
        info.native_handle = shader->pso();
        info.block_size = kernel.block_size();
        return info;
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
            std::move(metadata.argument_sampled),
            luisa::vector<MetalShader::Argument>{},
            std::move(metadata.format_types),
            metadata.block_size, metadata.checksum,
            0u, 0u, 0.0, 0.0);
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
    with_autorelease_pool([=] {
        auto event = reinterpret_cast<MetalEvent *>(handle);
        auto stream = reinterpret_cast<MetalStream *>(stream_handle);
        stream->signal(event, value);
    });
}

void MetalDevice::wait_event(uint64_t handle, uint64_t stream_handle, uint64_t value) noexcept {
    with_autorelease_pool([=] {
        auto event = reinterpret_cast<MetalEvent *>(handle);
        auto stream = reinterpret_cast<MetalStream *>(stream_handle);
        stream->wait(event, value);
    });
}

void MetalDevice::synchronize_event(uint64_t handle, uint64_t value) noexcept {
    with_autorelease_pool([=] {
        auto event = reinterpret_cast<MetalEvent *>(handle);
        event->synchronize(value);
    });
}

bool MetalDevice::is_event_completed(uint64_t handle, uint64_t value) const noexcept {
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

ResourceCreationInfo MetalDevice::create_curve(const AccelOption &option) noexcept {
    return with_autorelease_pool([=, this] {
        auto curve = new_with_allocator<MetalCurve>(_handle, option);
        ResourceCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(curve);
        info.native_handle = curve->pointer_to_handle();
        return info;
    });
}

void MetalDevice::destroy_curve(uint64_t handle) noexcept {
    with_autorelease_pool([=] {
        auto curve = reinterpret_cast<MetalCurve *>(handle);
        delete_with_allocator(curve);
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

ResourceCreationInfo MetalDevice::create_motion_instance(
    const AccelMotionOption &option) noexcept {
    return with_autorelease_pool([=, this] {
        LUISA_ASSERT(
            _handle->supportsPrimitiveMotionBlur(),
            "Metal4 motion instances require a device with ray-tracing "
            "motion-blur support (device '{}').",
            _handle->name()->utf8String());
        if (option.mode == AccelMotionMode::SRT) {
            LUISA_ASSERT(
                _handle->supportsFamily(MTL::GPUFamilyApple9),
                "Metal4 SRT motion instances require Apple9 or newer for "
                "per-component motion interpolation; device '{}' supports "
                "matrix motion only.",
                _handle->name()->utf8String());
        }
        auto instance =
            new_with_allocator<MetalMotionInstance>(option);
        ResourceCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(instance);
        info.native_handle = nullptr;
        return info;
    });
}

void MetalDevice::destroy_motion_instance(uint64_t handle) noexcept {
    with_autorelease_pool([=] {
        auto instance = reinterpret_cast<MetalMotionInstance *>(handle);
        delete_with_allocator(instance);
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

luisa::string MetalDevice::query(luisa::string_view property) noexcept {
    auto bool_string = [](bool value) noexcept -> luisa::string {
        return value ? "true" : "false";
    };
    if (property == "device_name") {
        return _handle->name()->utf8String();
    }
    if (property == "total_memory") {
        return luisa::format("{}", _handle->recommendedMaxWorkingSetSize());
    }
    if (property == "free_memory") {
        return luisa::format("{}", _handle->recommendedMaxWorkingSetSize());
    }
    if (property == "metal4_address_driven_acceleration_structures") {
        return bool_string(_handle->supportsFamily(MTL::GPUFamilyApple9));
    }
    if (property == "metal4_component_motion") {
        return bool_string(_handle->supportsFamily(MTL::GPUFamilyApple9));
    }
    if (property == "metal_motion_blur") {
        return bool_string(_handle->supportsPrimitiveMotionBlur());
    }
    if (property == "metal4_gpu_family") {
        if (_handle->supportsFamily(MTL::GPUFamilyApple10)) { return "Apple10"; }
        if (_handle->supportsFamily(MTL::GPUFamilyApple9)) { return "Apple9"; }
        if (_handle->supportsFamily(MTL::GPUFamilyApple8)) { return "Apple8"; }
        if (_handle->supportsFamily(MTL::GPUFamilyApple7)) { return "Apple7"; }
        if (_handle->supportsFamily(MTL::GPUFamilyApple6)) { return "Apple6"; }
        if (_handle->supportsFamily(MTL::GPUFamilyApple5)) { return "Apple5"; }
        if (_handle->supportsFamily(MTL::GPUFamilyApple4)) { return "Apple4"; }
        if (_handle->supportsFamily(MTL::GPUFamilyApple3)) { return "Apple3"; }
        if (_handle->supportsFamily(MTL::GPUFamilyApple2)) { return "Apple2"; }
        if (_handle->supportsFamily(MTL::GPUFamilyApple1)) { return "Apple1"; }
        if (_handle->supportsFamily(MTL::GPUFamilyMac2)) { return "Mac2"; }
        if (_handle->supportsFamily(MTL::GPUFamilyMac1)) { return "Mac1"; }
        return "unknown";
    }
    if (property == "metal4_runtime") {
        return bool_string(_handle->supportsFamily(MTL::GPUFamilyMetal4) &&
                           _metal4_compiler != nullptr);
    }
    if (property == "metal_ray_tracing") {
        return bool_string(_handle->supportsRaytracing());
    }
    if (property == "metal_ray_tracing_from_render") {
        return bool_string(_handle->supportsRaytracingFromRender());
    }
    if (property == "metal_function_pointers") {
        return bool_string(_handle->supportsFunctionPointers());
    }
    if (property == "metal_function_pointers_from_render") {
        return bool_string(_handle->supportsFunctionPointersFromRender());
    }
    if (property == "metal_dynamic_libraries") {
        return bool_string(_handle->supportsDynamicLibraries());
    }
    if (property == "metal_render_dynamic_libraries") {
        return bool_string(_handle->supportsRenderDynamicLibraries());
    }
    if (property == "metal_argument_buffer_tier") {
        return luisa::format("{}", static_cast<uint32_t>(
                                      _handle->argumentBuffersSupport()));
    }
    if (property == "metal_read_write_texture_tier") {
        return luisa::format("{}", static_cast<uint32_t>(
                                      _handle->readWriteTextureSupport()));
    }
    if (property == "metal_raster_order_groups") {
        return bool_string(_handle->areRasterOrderGroupsSupported());
    }
    if (property == "metal_bc_texture_compression") {
        return bool_string(_handle->supportsBCTextureCompression());
    }
    if (property == "metal_shader_barycentric_coordinates") {
        return bool_string(_handle->supportsShaderBarycentricCoordinates());
    }
    if (property == "metal_pull_model_interpolation") {
        return bool_string(_handle->supportsPullModelInterpolation());
    }
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
        if (name == TexCompressExt::name) {
            std::scoped_lock lock{_ext_mutex};
            if (!_tex_compress_ext) { _tex_compress_ext = luisa::make_unique<MetalTexCompressExt>(this); }
            return _tex_compress_ext.get();
        }
        if (name == RasterExt::name) {
            std::scoped_lock lock{_ext_mutex};
            if (!_raster_ext) { _raster_ext = luisa::make_unique<MetalRasterExt>(this); }
            return _raster_ext.get();
        }
#if LUISA_BACKEND_ENABLE_OIDN
        if (name == DenoiserExt::name) {
            std::scoped_lock lock{_ext_mutex};
            if (!_denoiser_ext) { _denoiser_ext = luisa::make_unique<MetalDenoiserExt>(this); }
            return _denoiser_ext.get();
        }
#endif
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
                auto texture = reinterpret_cast<MetalTextureBase *>(resource_handle);
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
            case Resource::Tag::CURVE: {
                auto curve = reinterpret_cast<MetalCurve *>(resource_handle);
                curve->set_name(name);
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
                auto shader = reinterpret_cast<MetalRasterShader *>(resource_handle);
                shader->set_name(name);
                break;
            }
            case Resource::Tag::SWAP_CHAIN: {
                auto swapchain = reinterpret_cast<MetalSwapchain *>(resource_handle);
                swapchain->set_name(name);
                break;
            }
            case Resource::Tag::DEPTH_BUFFER: {
                auto texture = reinterpret_cast<MetalTextureBase *>(resource_handle);
                LUISA_ASSERT(texture->kind() == MetalTextureBase::Kind::DEPTH,
                             "Invalid Metal depth-buffer resource handle.");
                texture->set_name(name);
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
            case Resource::Tag::SPARSE_BUFFER_HEAP: break;
            case Resource::Tag::SPARSE_TEXTURE_HEAP: break;
            case Resource::Tag::MOTION_INSTANCE: {
                auto instance =
                    reinterpret_cast<MetalMotionInstance *>(resource_handle);
                instance->set_name(name);
                break;
            }
            case Resource::Tag::TENSOR_GRAPH: break;
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
        auto p_device = static_cast<::luisa::compute::metal::MetalDevice *>(device);
        // auto p_device = dynamic_cast<::luisa::compute::metal::MetalDevice *>(device);
        // LUISA_ASSERT(p_device != nullptr, "Invalid device.");
        ::luisa::delete_with_allocator(p_device);
    });
}

LUISA_EXPORT_API void backend_device_names(luisa::vector<luisa::string> &names) noexcept {
    ::luisa::compute::metal::with_autorelease_pool([&names] {
        names.clear();
#if defined(LUISA_PLATFORM_IOS)
        if (!__builtin_available(iOS 26.0, *)) { return; }
        if (auto device = MTL::CreateSystemDefaultDevice()) {
            if (device->supportsFamily(MTL::GPUFamilyMetal4)) {
                names.emplace_back(device->name()->utf8String());
            }
            device->release();
        }
#else
        if (!__builtin_available(macOS 26.0, *)) { return; }
        auto all_devices = MTL::CopyAllDevices();
        if (auto n = all_devices->count()) {
            names.reserve(n);
            for (auto i = 0u; i < n; i++) {
                auto device = all_devices->object<MTL::Device>(i);
                if (!device->supportsFamily(MTL::GPUFamilyMetal4)) { continue; }
                names.emplace_back(device->name()->utf8String());
            }
        }
        all_devices->release();
#endif
    });
}

#if defined(LUISA_PLATFORM_IOS)
LUISA_EXPORT_API void
luisa_compute_metal4_register_static_backend() noexcept {
    luisa::compute::Context::register_static_backend(
        "metal4", create, destroy, backend_device_names);
}
#endif

#include "../common/export_version.inl.h"
