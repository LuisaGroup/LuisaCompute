#include <algorithm>
#include <array>

#include <luisa/core/logging.h>
#include <luisa/core/stl/hash.h>

#include "metal_air_pipeline.h"
#include "metal_depth_buffer.h"
#include "metal_device.h"
#include "metal_metallib.h"
#include "metal_raster_archive.h"
#include "metal_raster_ext.h"
#include "metal_raster_shader.h"
#include "metal_xir_pipeline.h"

namespace luisa::compute::metal {

namespace {

constexpr auto raster_root_argument_alignment = static_cast<size_t>(16u);
constexpr auto raster_root_argument_capacity = static_cast<size_t>(65536u);
constexpr auto indirect_dispatch_buffer_type_name =
    luisa::string_view{"LC_IndirectDispatchBuffer"};

[[nodiscard]] bool is_indirect_dispatch_buffer_type(
    const Type *type) noexcept {
    return type != nullptr && type->is_custom() &&
           type->description() == indirect_dispatch_buffer_type_name;
}

[[nodiscard]] luisa::optional<size_t> raster_root_argument_size(
    luisa::span<Type const *const> types) noexcept {
    auto size = static_cast<size_t>(0u);
    for (auto type : types) {
        if (type == nullptr) { return luisa::nullopt; }
        auto field_size = static_cast<size_t>(0u);
        if (type->is_basic() || type->is_array() || type->is_structure()) {
            field_size = type->size();
        } else if (type->is_buffer() || type->is_accel() ||
                   is_indirect_dispatch_buffer_type(type)) {
            field_size = 16u;
        } else if (type->is_texture() || type->is_bindless_array()) {
            field_size = 8u;
        } else {
            return luisa::nullopt;
        }
        size = luisa::align(size, raster_root_argument_alignment);
        if (size > raster_root_argument_capacity ||
            field_size > raster_root_argument_capacity - size) {
            return luisa::nullopt;
        }
        size += field_size;
    }
    return std::max(
        raster_root_argument_alignment,
        luisa::align(size, raster_root_argument_alignment));
}

[[nodiscard]] Usage conservative_aot_usage(
    const Type *type, Usage archived_usage) noexcept {
    if (type->is_texture()) {
        // A depth texture exposed through DepthBuffer::to_img() is read-only.
        // Preserve its reflected access instead of incorrectly adding WRITE.
        return archived_usage;
    }
    if (type->is_resource() || is_indirect_dispatch_buffer_type(type)) {
        return Usage::READ_WRITE;
    }
    return Usage::NONE;
}

[[nodiscard]] MetalRasterShader::Argument encode_binding(
    const Function::Binding &binding) noexcept {
    return luisa::visit(
        []<typename T>(const T &value) noexcept -> MetalRasterShader::Argument {
            using Binding = std::remove_cvref_t<T>;
            MetalRasterShader::Argument argument{};
            if constexpr (std::is_same_v<Binding, Function::BufferBinding>) {
                argument.tag = MetalRasterShader::Argument::Tag::BUFFER;
                argument.buffer = {value.handle, value.offset, value.size};
            } else if constexpr (std::is_same_v<Binding, Function::TextureBinding>) {
                argument.tag = MetalRasterShader::Argument::Tag::TEXTURE;
                argument.texture = {value.handle, value.level};
            } else if constexpr (std::is_same_v<Binding, Function::BindlessArrayBinding>) {
                argument.tag = MetalRasterShader::Argument::Tag::BINDLESS_ARRAY;
                argument.bindless_array = {value.handle};
            } else if constexpr (std::is_same_v<Binding, Function::AccelBinding>) {
                argument.tag = MetalRasterShader::Argument::Tag::ACCEL;
                argument.accel = {value.handle};
            } else {
                LUISA_ERROR_WITH_LOCATION("Cannot encode an unbound raster argument as a binding.");
            }
            return argument;
        },
        binding);
}

void append_root_arguments(
    Function stage, MTL::RenderStages stages,
    luisa::vector<MetalRasterShader::RootArgument> &result) noexcept {
    auto arguments = stage.arguments();
    auto bindings = stage.bound_arguments();
    LUISA_ASSERT(!arguments.empty(), "Raster stage has no payload argument.");
    LUISA_ASSERT(bindings.size() == arguments.size(),
                 "Raster-stage arguments and bindings are inconsistent.");
    result.reserve(result.size() + arguments.size() - 1u);
    for (auto i = 1u; i < arguments.size(); i++) {
        auto is_bound = !luisa::holds_alternative<luisa::monostate>(bindings[i]);
        MetalRasterShader::RootArgument root{};
        root.usage = stage.variable_usage(arguments[i].uid());
        root.stages = stages;
        root.is_bound = is_bound;
        if (is_bound) { root.binding = encode_binding(bindings[i]); }
        result.emplace_back(root);
    }
}

[[nodiscard]] luisa::vector<MetalRasterShader::RootArgument>
make_root_arguments(Function vertex, Function fragment) noexcept {
    luisa::vector<MetalRasterShader::RootArgument> result;
    append_root_arguments(vertex, MTL::RenderStageVertex, result);
    append_root_arguments(fragment, MTL::RenderStageFragment, result);
    return result;
}

void append_archive_arguments(
    Function stage, MetalRasterArchiveStage archive_stage,
    luisa::vector<MetalRasterArchiveArgument> &result) noexcept {
    auto arguments = stage.arguments();
    auto bindings = stage.bound_arguments();
    LUISA_ASSERT(bindings.size() == arguments.size(),
                 "Raster-stage arguments and bindings are inconsistent.");
    for (auto i = 1u; i < arguments.size(); i++) {
        LUISA_ASSERT(luisa::holds_alternative<luisa::monostate>(bindings[i]),
                     "Serialized Metal raster shaders cannot contain implicit bindings.");
        result.emplace_back(MetalRasterArchiveArgument{
            .type = luisa::string{arguments[i].type()->description()},
            .usage = stage.variable_usage(arguments[i].uid()),
            .stage = archive_stage});
    }
}

[[nodiscard]] MetalRasterArchive make_archive(
    const MeshFormat &mesh_format,
    Function vertex, Function fragment,
    MetalAIRRasterCodegenResult air) noexcept {
    MetalRasterArchive archive{};
    archive.mesh_format = mesh_format;
    archive.root_argument_size = air.root_argument_size;
    archive.fragment_output_count = air.fragment_output_count;
    append_archive_arguments(
        vertex, MetalRasterArchiveStage::VERTEX, archive.arguments);
    append_archive_arguments(
        fragment, MetalRasterArchiveStage::FRAGMENT, archive.arguments);
    archive.library = std::move(air.library);
    return archive;
}

}// namespace

ResourceCreationInfo MetalRasterExt::create_raster_shader(
    const MeshFormat &mesh_format,
    Function vert,
    Function pixel,
    const ShaderOption &shader_option) noexcept {
    luisa::string mesh_format_reason;
    LUISA_ASSERT(validate_metal_raster_mesh_format(
                     mesh_format, &mesh_format_reason),
                 "Invalid Metal raster mesh format: {}", mesh_format_reason);
    LUISA_ASSERT(!shader_option.enable_extended_accel_limits,
                 "Metal raster AIR does not support extended acceleration-structure limits.");
    return with_autorelease_pool([&] {
        auto vertex_module = metal_translate_raster_ast_to_xir(
            vert, xir::RasterStage::VERTEX, shader_option);
        auto fragment_module = metal_translate_raster_ast_to_xir(
            pixel, xir::RasterStage::FRAGMENT, shader_option);
        auto air = metal_codegen_air(
            *vertex_module, *fragment_module, mesh_format, shader_option);
        LUISA_ASSERT(air.vertex_entry == "vertex_main" &&
                         air.fragment_entry == "fragment_main",
                     "Metal raster AIR emitted unexpected entry-point names.");
        if (shader_option.compile_only) {
            LUISA_ASSERT(!shader_option.name.empty(),
                         "Serialized Metal raster shaders require a non-empty name.");
            auto archive = make_archive(
                mesh_format, vert, pixel, std::move(air));
            auto data = serialize_metal_raster_archive(archive);
            static_cast<void>(_device->io()->write_shader_bytecode(
                shader_option.name, data));
            return ResourceCreationInfo::make_invalid();
        }
        auto root_arguments = make_root_arguments(vert, pixel);
        auto name = shader_option.name;
        if (name.empty()) {
            name = luisa::format(
                "metal_air_raster_{:016x}",
                luisa::hash_combine({vert.hash(), pixel.hash()}));
        }
        auto shader = new_with_allocator<MetalRasterShader>(
            _device, air.library, mesh_format,
            std::move(root_arguments), air.root_argument_size,
            air.fragment_output_count, name);
        if (!shader->valid()) {
            delete_with_allocator(shader);
            LUISA_ERROR_WITH_LOCATION(
                "Failed to create Metal raster AIR shader '{}'.", name);
        }
        ResourceCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(shader);
        return info;
    });
}

ResourceCreationInfo MetalRasterExt::load_raster_shader(
    luisa::span<Type const *const> types,
    luisa::string_view name) noexcept {
    return with_autorelease_pool([&] {
        auto stream = _device->io()->read_shader_bytecode(name);
        if (stream == nullptr || stream->length() == 0u) {
            LUISA_WARNING_WITH_LOCATION(
                "Failed to load Metal raster shader archive '{}'.", name);
            return ResourceCreationInfo::make_invalid();
        }
        auto data = stream->read(~0ull);
        auto archive = deserialize_metal_raster_archive(
            luisa::span<const std::byte>{data.data(), data.size()});
        if (!archive) {
            LUISA_WARNING_WITH_LOCATION(
                "Invalid Metal raster shader archive '{}'.", name);
            return ResourceCreationInfo::make_invalid();
        }
        constexpr std::array<luisa::string_view, 2u> entry_points{
            "vertex_main", "fragment_main"};
        constexpr std::array program_types{
            MetalLibProgramType::VERTEX,
            MetalLibProgramType::FRAGMENT};
        if (!validate_metallib(
                archive->library, entry_points, program_types)) {
            LUISA_WARNING_WITH_LOCATION(
                "Metal raster shader archive '{}' contains an invalid library.", name);
            return ResourceCreationInfo::make_invalid();
        }
        if (archive->arguments.size() != types.size()) {
            LUISA_WARNING_WITH_LOCATION(
                "Metal raster shader '{}' expects {} argument(s), got {}.",
                name, archive->arguments.size(), types.size());
            return ResourceCreationInfo::make_invalid();
        }
        luisa::vector<MetalRasterShader::RootArgument> root_arguments;
        root_arguments.reserve(types.size());
        constexpr auto all_raster_stages = static_cast<MTL::RenderStages>(
            MTL::RenderStageVertex | MTL::RenderStageFragment);
        for (auto i = 0u; i < types.size(); i++) {
            if (types[i] == nullptr ||
                archive->arguments[i].type != types[i]->description()) {
                LUISA_WARNING_WITH_LOCATION(
                    "Metal raster shader '{}' argument {} has type '{}', expected '{}'.",
                    name, i,
                    types[i] == nullptr ? "<null>" : types[i]->description(),
                    archive->arguments[i].type);
                return ResourceCreationInfo::make_invalid();
            }
            root_arguments.emplace_back(MetalRasterShader::RootArgument{
                .usage = conservative_aot_usage(
                    types[i], archive->arguments[i].usage),
                .stages = all_raster_stages,
                .is_bound = false});
        }
        auto expected_root_size = raster_root_argument_size(types);
        if (!expected_root_size ||
            *expected_root_size != archive->root_argument_size) {
            LUISA_WARNING_WITH_LOCATION(
                "Metal raster shader '{}' root ABI has size {}, expected {}.",
                name, archive->root_argument_size,
                expected_root_size.value_or(0u));
            return ResourceCreationInfo::make_invalid();
        }
        auto shader = new_with_allocator<MetalRasterShader>(
            _device, archive->library,
            std::move(archive->mesh_format),
            std::move(root_arguments), archive->root_argument_size,
            archive->fragment_output_count, name);
        if (!shader->valid()) {
            LUISA_WARNING_WITH_LOCATION(
                "Metal raster shader archive '{}' is incompatible with this device.",
                name);
            delete_with_allocator(shader);
            return ResourceCreationInfo::make_invalid();
        }
        ResourceCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(shader);
        return info;
    });
}

void MetalRasterExt::destroy_raster_shader(uint64_t handle) noexcept {
    delete_with_allocator(reinterpret_cast<MetalRasterShader *>(handle));
}

ResourceCreationInfo MetalRasterExt::create_depth_buffer(
    DepthFormat format, uint width, uint height) noexcept {
    return with_autorelease_pool([&] {
        auto depth = new_with_allocator<MetalDepthBuffer>(
            _device->handle(), format, width, height);
        ResourceCreationInfo info{};
        info.handle = reinterpret_cast<uint64_t>(
            static_cast<MetalTextureBase *>(depth));
        info.native_handle = depth->handle();
        return info;
    });
}

void MetalRasterExt::destroy_depth_buffer(uint64_t handle) noexcept {
    with_autorelease_pool([&] {
        auto texture = reinterpret_cast<MetalTextureBase *>(handle);
        LUISA_ASSERT(texture->kind() == MetalTextureBase::Kind::DEPTH,
                     "Attempting to destroy a non-depth Metal texture as a depth buffer.");
        delete_with_allocator(static_cast<MetalDepthBuffer *>(texture));
    });
}

}// namespace luisa::compute::metal
