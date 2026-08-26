#include "metal_codegen_llvm_impl.h"

namespace luisa::compute::metal::detail {

luisa::string MetalCodegenLLVMImpl::_air_type_name(const Type *type) const noexcept {
    switch (type->tag()) {
        case Type::Tag::BOOL: return "bool";
        case Type::Tag::INT8: return "char";
        case Type::Tag::UINT8: return "uchar";
        case Type::Tag::INT16: return "short";
        case Type::Tag::UINT16: return "ushort";
        case Type::Tag::INT32: return "int";
        case Type::Tag::UINT32: return "uint";
        case Type::Tag::INT64: return "long";
        case Type::Tag::UINT64: return "ulong";
        case Type::Tag::FLOAT16: return "half";
        case Type::Tag::FLOAT32: return "float";
        case Type::Tag::VECTOR: {
            auto name = _air_type_name(type->element());
            auto dimension = std::to_string(type->dimension());
            name.append(dimension.data(), dimension.size());
            return name;
        }
        case Type::Tag::MATRIX: {
            auto name = _air_type_name(type->element());
            auto dimension = std::to_string(type->dimension());
            name.append(dimension.data(), dimension.size());
            name.push_back('x');
            name.append(dimension.data(), dimension.size());
            return name;
        }
        case Type::Tag::BUFFER: {
            auto element = type->element();
            return element == nullptr ? luisa::string{"uchar"} : _air_type_name(element);
        }
        case Type::Tag::ACCEL:
            return luisa::string{accel_air_type_name};
        case Type::Tag::ARRAY: [[fallthrough]];
        case Type::Tag::STRUCTURE: {
            auto description = type->description();
            return luisa::string{description.data(), description.size()};
        }
        case Type::Tag::CUSTOM: {
            if (is_indirect_dispatch_buffer_type(type)) {
                return luisa::string{
                    indirect_dispatch_buffer_air_type_name};
            }
            if (is_ray_query_type(type)) {
                auto description = type->description();
                return luisa::string{
                    description.data(), description.size()};
            }
            _unsupported_type(type);
        }
        default: _unsupported_type(type);
    }
}

luisa::string MetalCodegenLLVMImpl::_air_texture_type_name(const Type *type, uint32_t access) const noexcept {
    LUISA_ASSERT(type != nullptr && type->is_texture(),
                 "AIR texture type name requested for a non-texture type.");
    auto access_name = access == air_texture_access_read   ? luisa::string_view{"read"} :
                       access == air_texture_access_write  ? luisa::string_view{"write"} :
                       access == air_texture_access_sample ? luisa::string_view{"sample"} :
                                                             luisa::string_view{"read_write"};
    return luisa::format("texture{}d<{}, {}>",
                         type->dimension(), _air_type_name(type->element()), access_name);
}

llvm::MDNode *MetalCodegenLLVMImpl::_air_struct_type_info(const Type *type) noexcept {
    LUISA_ASSERT(type->is_structure(), "AIR struct type metadata requested for a non-structure type.");
    llvm::SmallVector<llvm::Metadata *> fields;
    auto type_info = _type(type);
    auto array_base = [](const Type *member) noexcept {
        auto count = 0u;
        while (member->is_array()) {
            count = count == 0u ? member->dimension() : count * member->dimension();
            member = member->element();
        }
        return std::pair{member, count};
    };
    for (auto i = 0u; i < type->members().size(); i++) {
        auto member = type->members()[i];
        auto [base, array_count] = array_base(member);
        auto member_name = "member." + std::to_string(i);
        if (base->is_structure()) {
            fields.append({md_string(_context, "air.struct_type_info"), _air_struct_type_info(base)});
        }
        fields.append({md_i32(_context, static_cast<uint32_t>(type_info->member_offsets[i])),
                       md_i32(_context, static_cast<uint32_t>(base->size())), md_i32(_context, array_count),
                       md_string(_context, _air_type_name(base)), md_string(_context, member_name)});
    }
    return llvm::MDNode::get(_context, fields);
}

size_t MetalCodegenLLVMImpl::_air_indirect_location_count(const Type *type) const noexcept {
    LUISA_ASSERT(type != nullptr && !type->is_resource(),
                 "AIR indirect constant location count requested for a resource type.");
    if (type->is_array()) {
        return type->dimension() * _air_indirect_location_count(type->element());
    }
    if (type->is_structure()) {
        auto count = static_cast<size_t>(0u);
        for (auto member : type->members()) {
            count += _air_indirect_location_count(member);
        }
        return count;
    }
    return 1u;
}

llvm::MDNode *MetalCodegenLLVMImpl::_air_indirect_struct_type_info(const Type *type) noexcept {
    LUISA_ASSERT(type->is_structure(),
                 "AIR indirect struct metadata requested for a non-structure type.");
    auto node = [this](llvm::ArrayRef<llvm::Metadata *> operands) noexcept {
        return llvm::MDNode::get(_context, operands);
    };
    auto array_base = [](const Type *member) noexcept {
        auto count = 0u;
        while (member->is_array()) {
            count = count == 0u ? member->dimension() : count * member->dimension();
            member = member->element();
        }
        return std::pair{member, count};
    };
    llvm::SmallVector<llvm::Metadata *> fields;
    auto type_info = _type(type);
    auto location_index = static_cast<size_t>(0u);
    for (auto logical_index = 0u; logical_index < type->members().size(); logical_index++) {
        auto member = type->members()[logical_index];
        auto [base, array_count] = array_base(member);
        auto member_name = "member." + std::to_string(logical_index);
        if (base->is_structure()) {
            fields.append({md_string(_context, "air.struct_type_info"),
                           _air_indirect_struct_type_info(base),
                           md_i32(_context, static_cast<uint32_t>(type_info->member_offsets[logical_index])),
                           md_i32(_context, static_cast<uint32_t>(base->size())),
                           md_i32(_context, array_count),
                           md_string(_context, _air_type_name(base)), md_string(_context, member_name),
                           md_string(_context, "air.indirect_argument"),
                           md_i32(_context, static_cast<uint32_t>(location_index))});
        } else {
            auto detail = node({md_i32(_context, logical_index),
                                md_string(_context, "air.indirect_constant"),
                                md_string(_context, "air.location_index"),
                                md_i32(_context, static_cast<uint32_t>(location_index)), md_i32(_context, 1u),
                                md_string(_context, "air.arg_type_name"), md_string(_context, _air_type_name(base)),
                                md_string(_context, "air.arg_name"), md_string(_context, member_name)});
            fields.append({md_i32(_context, static_cast<uint32_t>(type_info->member_offsets[logical_index])),
                           md_i32(_context, static_cast<uint32_t>(base->size())),
                           md_i32(_context, array_count),
                           md_string(_context, _air_type_name(base)), md_string(_context, member_name),
                           md_string(_context, "air.indirect_argument"), detail});
        }
        location_index += _air_indirect_location_count(member);
    }
    return node(fields);
}

llvm::MDNode *MetalCodegenLLVMImpl::_root_argument_metadata(
    size_t argument_struct_size, uint32_t argument_index) noexcept {
    auto node = [this](llvm::ArrayRef<llvm::Metadata *> operands) noexcept { return llvm::MDNode::get(_context, operands); };
    llvm::SmallVector<llvm::Metadata *> struct_fields;
    auto layout = _root_argument_layout();
    auto physical_index = 0u;
    auto logical_index = 0u;
    auto struct_field_index = 0u;
    auto array_base = [](const Type *type) noexcept {
        auto count = 0u;
        while (type->is_array()) {
            count = count == 0u ? type->dimension() : count * type->dimension();
            type = type->element();
        }
        return std::pair{type, count};
    };
    for (auto argument : _root_arguments) {
        auto name = argument->name().value_or("arg");
        auto offset = layout.offsets[logical_index];
        if (argument->type()->is_buffer()) {
            auto element = argument->type()->element();
            auto element_name = element == nullptr ? luisa::string{"uchar"} : _air_type_name(element);
            auto element_size = element == nullptr ? 1u : static_cast<uint32_t>(element->size());
            auto element_alignment = element == nullptr ? 1u : static_cast<uint32_t>(element->alignment());
            auto buffer_type_name = luisa::string{"LCBuffer."};
            buffer_type_name.append(element_name);
            llvm::SmallVector<llvm::Metadata *> buffer_detail_fields{
                md_i32(_context, 0u), md_string(_context, "air.buffer"),
                md_string(_context, "air.location_index"), md_i32(_context, 0u), md_i32(_context, 1u),
                md_string(_context, "air.read_write"),
                md_string(_context, "air.address_space"), md_i32(_context, air_address_space_device)};
            if (element != nullptr && element->is_structure()) {
                buffer_detail_fields.append({md_string(_context, "air.struct_type_info"), _air_struct_type_info(element)});
            }
            buffer_detail_fields.append({md_string(_context, "air.arg_type_size"), md_i32(_context, element_size),
                                         md_string(_context, "air.arg_type_align_size"), md_i32(_context, element_alignment),
                                         md_string(_context, "air.arg_type_name"), md_string(_context, element_name),
                                         md_string(_context, "air.arg_name"), md_string(_context, "data")});
            auto buffer_detail = node(buffer_detail_fields);
            llvm::SmallVector<llvm::Metadata *> buffer_fields{
                md_i32(_context, 0u), md_i32(_context, 8u), md_i32(_context, 0u),
                md_string(_context, element_name), md_string(_context, "data"),
                md_string(_context, "air.indirect_argument"), buffer_detail};
            auto size_detail = node({md_i32(_context, 1u), md_string(_context, "air.indirect_constant"),
                                     md_string(_context, "air.location_index"), md_i32(_context, 1u), md_i32(_context, 1u),
                                     md_string(_context, "air.arg_type_name"), md_string(_context, "ulong"),
                                     md_string(_context, "air.arg_name"), md_string(_context, "size")});
            buffer_fields.append({md_i32(_context, 8u), md_i32(_context, 8u), md_i32(_context, 0u),
                                  md_string(_context, "ulong"), md_string(_context, "size"),
                                  md_string(_context, "air.indirect_argument"), size_detail});
            struct_fields.append({md_string(_context, "air.struct_type_info"), node(buffer_fields),
                                  md_i32(_context, static_cast<uint32_t>(offset)), md_i32(_context, 16u), md_i32(_context, 0u),
                                  md_string(_context, buffer_type_name), md_string(_context, name),
                                  md_string(_context, "air.indirect_argument"), md_i32(_context, physical_index)});
            physical_index += 2u;
        } else if (is_indirect_dispatch_buffer_type(argument->type())) {
            auto buffer_detail = node({md_i32(_context, 0u), md_string(_context, "air.buffer"),
                                       md_string(_context, "air.location_index"), md_i32(_context, 0u), md_i32(_context, 1u),
                                       md_string(_context, "air.read_write"),
                                       md_string(_context, "air.address_space"), md_i32(_context, air_address_space_device),
                                       md_string(_context, "air.arg_type_name"), md_string(_context, "void"),
                                       md_string(_context, "air.arg_name"), md_string(_context, "buffer")});
            auto offset_detail = node({md_i32(_context, 1u), md_string(_context, "air.indirect_constant"),
                                       md_string(_context, "air.location_index"), md_i32(_context, 1u), md_i32(_context, 1u),
                                       md_string(_context, "air.arg_type_name"), md_string(_context, "uint"),
                                       md_string(_context, "air.arg_name"), md_string(_context, "offset")});
            auto capacity_detail = node({md_i32(_context, 2u), md_string(_context, "air.indirect_constant"),
                                         md_string(_context, "air.location_index"), md_i32(_context, 2u), md_i32(_context, 1u),
                                         md_string(_context, "air.arg_type_name"), md_string(_context, "uint"),
                                         md_string(_context, "air.arg_name"), md_string(_context, "capacity")});
            auto fields = node({md_i32(_context, 0u), md_i32(_context, 8u), md_i32(_context, 0u),
                                md_string(_context, "void"), md_string(_context, "buffer"),
                                md_string(_context, "air.indirect_argument"), buffer_detail,
                                md_i32(_context, 8u), md_i32(_context, 4u), md_i32(_context, 0u),
                                md_string(_context, "uint"), md_string(_context, "offset"),
                                md_string(_context, "air.indirect_argument"), offset_detail,
                                md_i32(_context, 12u), md_i32(_context, 4u), md_i32(_context, 0u),
                                md_string(_context, "uint"), md_string(_context, "capacity"),
                                md_string(_context, "air.indirect_argument"), capacity_detail});
            struct_fields.append({md_string(_context, "air.struct_type_info"), fields,
                                  md_i32(_context, static_cast<uint32_t>(offset)), md_i32(_context, 16u), md_i32(_context, 0u),
                                  md_string(_context, indirect_dispatch_buffer_air_type_name), md_string(_context, name),
                                  md_string(_context, "air.indirect_argument"), md_i32(_context, physical_index)});
            physical_index += 3u;
        } else if (argument->type()->is_bindless_array()) {
            auto buffer_detail = node({md_i32(_context, 0u), md_string(_context, "air.buffer"),
                                       md_string(_context, "air.location_index"), md_i32(_context, 0u), md_i32(_context, 1u),
                                       md_string(_context, "air.read_write"),
                                       md_string(_context, "air.address_space"), md_i32(_context, air_address_space_device),
                                       md_string(_context, "air.arg_type_name"), md_string(_context, "void"),
                                       md_string(_context, "air.arg_name"), md_string(_context, "buffer")});
            auto buffer_size_detail = node({md_i32(_context, 1u), md_string(_context, "air.indirect_constant"),
                                            md_string(_context, "air.location_index"), md_i32(_context, 1u), md_i32(_context, 1u),
                                            md_string(_context, "air.arg_type_name"), md_string(_context, "ulong"),
                                            md_string(_context, "air.arg_name"), md_string(_context, "buffer_size")});
            auto sampler2d_detail = node({md_i32(_context, 2u), md_string(_context, "air.indirect_constant"),
                                          md_string(_context, "air.location_index"), md_i32(_context, 2u), md_i32(_context, 1u),
                                          md_string(_context, "air.arg_type_name"), md_string(_context, "uint"),
                                          md_string(_context, "air.arg_name"), md_string(_context, "sampler2d")});
            auto sampler3d_detail = node({md_i32(_context, 3u), md_string(_context, "air.indirect_constant"),
                                          md_string(_context, "air.location_index"), md_i32(_context, 3u), md_i32(_context, 1u),
                                          md_string(_context, "air.arg_type_name"), md_string(_context, "uint"),
                                          md_string(_context, "air.arg_name"), md_string(_context, "sampler3d")});
            auto texture2d_detail = node({md_i32(_context, 4u), md_string(_context, "air.texture"),
                                          md_string(_context, "air.location_index"), md_i32(_context, 4u), md_i32(_context, 1u),
                                          md_string(_context, "air.sample"),
                                          md_string(_context, "air.arg_type_name"), md_string(_context, "texture2d<float, sample>"),
                                          md_string(_context, "air.arg_name"), md_string(_context, "tex2d")});
            auto texture3d_detail = node({md_i32(_context, 5u), md_string(_context, "air.texture"),
                                          md_string(_context, "air.location_index"), md_i32(_context, 5u), md_i32(_context, 1u),
                                          md_string(_context, "air.sample"),
                                          md_string(_context, "air.arg_type_name"), md_string(_context, "texture3d<float, sample>"),
                                          md_string(_context, "air.arg_name"), md_string(_context, "tex3d")});
            auto item_info = node({md_i32(_context, 0u), md_i32(_context, 8u), md_i32(_context, 0u),
                                   md_string(_context, "void"), md_string(_context, "buffer"),
                                   md_string(_context, "air.indirect_argument"), buffer_detail,
                                   md_i32(_context, 8u), md_i32(_context, 8u), md_i32(_context, 0u),
                                   md_string(_context, "ulong"), md_string(_context, "buffer_size"),
                                   md_string(_context, "air.indirect_argument"), buffer_size_detail,
                                   md_i32(_context, 14u), md_i32(_context, 4u), md_i32(_context, 0u),
                                   md_string(_context, "uint"), md_string(_context, "sampler2d"),
                                   md_string(_context, "air.indirect_argument"), sampler2d_detail,
                                   md_i32(_context, 15u), md_i32(_context, 4u), md_i32(_context, 0u),
                                   md_string(_context, "uint"), md_string(_context, "sampler3d"),
                                   md_string(_context, "air.indirect_argument"), sampler3d_detail,
                                   md_i32(_context, 16u), md_i32(_context, 8u), md_i32(_context, 0u),
                                   md_string(_context, "texture2d<float, sample>"), md_string(_context, "tex2d"),
                                   md_string(_context, "air.indirect_argument"), texture2d_detail,
                                   md_i32(_context, 24u), md_i32(_context, 8u), md_i32(_context, 0u),
                                   md_string(_context, "texture3d<float, sample>"), md_string(_context, "tex3d"),
                                   md_string(_context, "air.indirect_argument"), texture3d_detail});
            auto items_detail = node({md_i32(_context, 0u), md_string(_context, "air.indirect_buffer"),
                                      md_string(_context, "air.location_index"), md_i32(_context, 0u), md_i32(_context, 1u),
                                      md_string(_context, "air.read"),
                                      md_string(_context, "air.address_space"), md_i32(_context, air_address_space_device),
                                      md_string(_context, "air.struct_type_info"), item_info,
                                      md_string(_context, "air.arg_type_size"), md_i32(_context, 32u),
                                      md_string(_context, "air.arg_type_align_size"), md_i32(_context, 16u),
                                      md_string(_context, "air.arg_type_name"), md_string(_context, "LCBindlessItem"),
                                      md_string(_context, "air.arg_name"), md_string(_context, "items")});
            auto wrapper_info = node({md_i32(_context, 0u), md_i32(_context, 8u), md_i32(_context, 0u),
                                      md_string(_context, "LCBindlessItem"), md_string(_context, "items"),
                                      md_string(_context, "air.indirect_argument"), items_detail});
            struct_fields.append({md_string(_context, "air.struct_type_info"), wrapper_info,
                                  md_i32(_context, static_cast<uint32_t>(offset)), md_i32(_context, 8u), md_i32(_context, 0u),
                                  md_string(_context, "LCBindlessArray"), md_string(_context, name),
                                  md_string(_context, "air.indirect_argument"), md_i32(_context, physical_index)});
            physical_index++;
        } else if (argument->type()->is_accel()) {
            auto handle_detail = node({md_i32(_context, 0u),
                                       md_string(_context, "air.instance_acceleration_structure"),
                                       md_string(_context, "air.location_index"),
                                       md_i32(_context, 0u), md_i32(_context, 1u),
                                       md_string(_context, "air.read"),
                                       md_string(_context, "air.arg_type_name"),
                                       md_string(_context, accel_handle_air_type_name),
                                       md_string(_context, "air.arg_name"),
                                       md_string(_context, "handle")});
            auto transform_array_info = node({md_i32(_context, 0u), md_i32(_context, 4u), md_i32(_context, 12u),
                                              md_string(_context, "float"), md_string(_context, "__elems")});
            auto instance_info = node({md_string(_context, "air.struct_type_info"), transform_array_info,
                                       md_i32(_context, 0u), md_i32(_context, 48u), md_i32(_context, 0u),
                                       md_string(_context, "metal::array"), md_string(_context, "transform"),
                                       md_i32(_context, 48u), md_i32(_context, 4u), md_i32(_context, 0u),
                                       md_string(_context, "uint"), md_string(_context, "options"),
                                       md_i32(_context, 52u), md_i32(_context, 4u), md_i32(_context, 0u),
                                       md_string(_context, "uint"), md_string(_context, "mask"),
                                       md_i32(_context, 56u), md_i32(_context, 4u), md_i32(_context, 0u),
                                       md_string(_context, "uint"), md_string(_context, "intersection_function_offset"),
                                       md_i32(_context, 60u), md_i32(_context, 4u), md_i32(_context, 0u),
                                       md_string(_context, "uint"), md_string(_context, "mesh_index"),
                                       md_i32(_context, 64u), md_i32(_context, 8u), md_i32(_context, 0u),
                                       md_string(_context, "ulong"), md_string(_context, "acceleration_structure_id")});
            auto instances_detail = node({md_i32(_context, 1u), md_string(_context, "air.buffer"),
                                          md_string(_context, "air.location_index"),
                                          md_i32(_context, 1u), md_i32(_context, 1u),
                                          md_string(_context, "air.read_write"),
                                          md_string(_context, "air.address_space"),
                                          md_i32(_context, air_address_space_device),
                                          md_string(_context, "air.struct_type_info"), instance_info,
                                          md_string(_context, "air.arg_type_size"), md_i32(_context, 72u),
                                          md_string(_context, "air.arg_type_align_size"), md_i32(_context, 8u),
                                          md_string(_context, "air.arg_type_name"),
                                          md_string(_context, accel_instance_air_type_name),
                                          md_string(_context, "air.arg_name"),
                                          md_string(_context, "instances")});
            auto accel_info = node({md_i32(_context, 0u), md_i32(_context, 8u), md_i32(_context, 0u),
                                    md_string(_context, accel_handle_air_type_name),
                                    md_string(_context, "handle"),
                                    md_string(_context, "air.indirect_argument"), handle_detail,
                                    md_i32(_context, 8u), md_i32(_context, 8u), md_i32(_context, 0u),
                                    md_string(_context, accel_instance_air_type_name),
                                    md_string(_context, "instances"),
                                    md_string(_context, "air.indirect_argument"), instances_detail});
            struct_fields.append({md_string(_context, "air.struct_type_info"), accel_info,
                                  md_i32(_context, static_cast<uint32_t>(offset)),
                                  md_i32(_context, 16u), md_i32(_context, 0u),
                                  md_string(_context, accel_air_type_name), md_string(_context, name),
                                  md_string(_context, "air.indirect_argument"),
                                  md_i32(_context, physical_index)});
            physical_index += 2u;
        } else if (argument->type()->is_texture()) {
            auto access = _texture_access(argument);
            auto type_name = _air_texture_type_name(argument->type(), access);
            auto access_name = access == air_texture_access_read   ? luisa::string_view{"air.read"} :
                               access == air_texture_access_write  ? luisa::string_view{"air.write"} :
                               access == air_texture_access_sample ? luisa::string_view{"air.sample"} :
                                                                     luisa::string_view{"air.read_write"};
            auto texture_detail = node({md_i32(_context, struct_field_index), md_string(_context, "air.texture"),
                                        md_string(_context, "air.location_index"), md_i32(_context, physical_index), md_i32(_context, 1u),
                                        md_string(_context, access_name),
                                        md_string(_context, "air.arg_type_name"), md_string(_context, type_name),
                                        md_string(_context, "air.arg_name"), md_string(_context, name)});
            struct_fields.append({md_i32(_context, static_cast<uint32_t>(offset)), md_i32(_context, 8u), md_i32(_context, 0u),
                                  md_string(_context, type_name), md_string(_context, name),
                                  md_string(_context, "air.indirect_argument"), texture_detail});
            physical_index++;
            if (_texture_needs_sampled_split(argument)) {
                auto sampled_name = luisa::string{name};
                sampled_name.append("_sampled");
                auto sampled_type_name = _air_texture_type_name(
                    argument->type(), air_texture_access_sample);
                auto sampled_detail = node({
                    md_i32(_context, struct_field_index + 1u),
                    md_string(_context, "air.texture"),
                    md_string(_context, "air.location_index"),
                    md_i32(_context, physical_index), md_i32(_context, 1u),
                    md_string(_context, "air.sample"),
                    md_string(_context, "air.arg_type_name"),
                    md_string(_context, sampled_type_name),
                    md_string(_context, "air.arg_name"),
                    md_string(_context, sampled_name)});
                struct_fields.append({
                    md_i32(_context, static_cast<uint32_t>(
                                         layout.sampled_texture_offsets[
                                             logical_index])),
                    md_i32(_context, 8u), md_i32(_context, 0u),
                    md_string(_context, sampled_type_name),
                    md_string(_context, sampled_name),
                    md_string(_context, "air.indirect_argument"),
                    sampled_detail});
                physical_index++;
                struct_field_index++;
            }
        } else if (argument->type()->is_array() || argument->type()->is_structure()) {
            auto [base, array_count] = array_base(argument->type());
            if (base->is_structure()) {
                struct_fields.append({md_string(_context, "air.struct_type_info"),
                                      _air_indirect_struct_type_info(base),
                                      md_i32(_context, static_cast<uint32_t>(offset)),
                                      md_i32(_context, static_cast<uint32_t>(base->size())),
                                      md_i32(_context, array_count),
                                      md_string(_context, _air_type_name(base)), md_string(_context, name),
                                      md_string(_context, "air.indirect_argument"),
                                      md_i32(_context, physical_index)});
            } else {
                auto detail = node({md_i32(_context, struct_field_index),
                                    md_string(_context, "air.indirect_constant"),
                                    md_string(_context, "air.location_index"), md_i32(_context, physical_index), md_i32(_context, 1u),
                                    md_string(_context, "air.arg_type_name"), md_string(_context, _air_type_name(base)),
                                    md_string(_context, "air.arg_name"), md_string(_context, name)});
                struct_fields.append({md_i32(_context, static_cast<uint32_t>(offset)),
                                      md_i32(_context, static_cast<uint32_t>(base->size())),
                                      md_i32(_context, array_count),
                                      md_string(_context, _air_type_name(base)), md_string(_context, name),
                                      md_string(_context, "air.indirect_argument"), detail});
            }
            physical_index += static_cast<uint32_t>(
                _air_indirect_location_count(argument->type()));
        } else {
            auto type_name = _air_type_name(argument->type());
            auto field_size = static_cast<uint32_t>(_data_layout.getTypeAllocSize(_type(argument->type())->mem_type).getFixedValue());
            auto field_detail = node({md_i32(_context, struct_field_index), md_string(_context, "air.indirect_constant"),
                                      md_string(_context, "air.location_index"), md_i32(_context, physical_index), md_i32(_context, 1u),
                                      md_string(_context, "air.arg_type_name"), md_string(_context, type_name),
                                      md_string(_context, "air.arg_name"), md_string(_context, name)});
            struct_fields.append({md_i32(_context, static_cast<uint32_t>(offset)), md_i32(_context, field_size), md_i32(_context, 0u),
                                  md_string(_context, type_name), md_string(_context, name),
                                  md_string(_context, "air.indirect_argument"), field_detail});
            physical_index++;
        }
        struct_field_index++;
        logical_index++;
    }
    auto struct_info = node(struct_fields);
    return node({md_i32(_context, argument_index), md_string(_context, "air.indirect_buffer"),
                 md_string(_context, "air.buffer_size"), md_i32(_context, static_cast<uint32_t>(argument_struct_size)),
                 md_string(_context, "air.location_index"), md_i32(_context, 0u), md_i32(_context, 1u),
                 md_string(_context, "air.read"),
                 md_string(_context, "air.address_space"), md_i32(_context, air_address_space_constant),
                 md_string(_context, "air.struct_type_info"), struct_info,
                 md_string(_context, "air.arg_type_size"), md_i32(_context, static_cast<uint32_t>(argument_struct_size)),
                 md_string(_context, "air.arg_type_align_size"), md_i32(_context, kernel_argument_alignment),
                 md_string(_context, "air.arg_type_name"), md_string(_context, "Arguments"),
                 md_string(_context, "air.arg_name"), md_string(_context, "args")});
}

void MetalCodegenLLVMImpl::_add_kernel_metadata(llvm::Function *function, size_t argument_struct_size, bool indirect) noexcept {
    LUISA_ASSERT(_kernel != nullptr, "Kernel metadata requested without an XIR kernel.");
    auto node = [this](llvm::ArrayRef<llvm::Metadata *> operands) noexcept { return llvm::MDNode::get(_context, operands); };
    auto args_info = _root_argument_metadata(argument_struct_size, 0u);
    auto dispatch_info = node({md_i32(_context, 1u), md_string(_context, "air.buffer"),
                               md_string(_context, "air.buffer_size"), md_i32(_context, 16u),
                               md_string(_context, "air.location_index"), md_i32(_context, 1u), md_i32(_context, 1u),
                               md_string(_context, indirect ? "air.read_write" : "air.read"),
                               md_string(_context, "air.address_space"), md_i32(_context, indirect ? air_address_space_device : air_address_space_constant),
                               md_string(_context, "air.arg_type_size"), md_i32(_context, 16u),
                               md_string(_context, "air.arg_type_align_size"), md_i32(_context, 16u),
                               md_string(_context, "air.arg_type_name"), md_string(_context, indirect ? "uint4" : "uint3"),
                               md_string(_context, "air.arg_name"), md_string(_context, indirect ? "dispatch_size_and_kernel_id" : "dispatch_size")});
    auto builtin = [this, &node](uint32_t index, luisa::string_view semantic, luisa::string_view type, luisa::string_view name) noexcept {
        return node({md_i32(_context, index), md_string(_context, semantic),
                     md_string(_context, "air.arg_type_name"), md_string(_context, type),
                     md_string(_context, "air.arg_name"), md_string(_context, name)});
    };
    auto argument_info = node({args_info,
                               dispatch_info,
                               builtin(2u, "air.thread_position_in_threadgroup", "uint3", "thread_id"),
                               builtin(3u, "air.threadgroup_position_in_grid", "uint3", "block_id"),
                               builtin(4u, "air.thread_position_in_grid", "uint3", "dispatch_id"),
                               builtin(5u, "air.threads_per_threadgroup", "uint3", "block_size"),
                               builtin(6u, "air.threads_per_simdgroup", "uint", "warp_size"),
                               builtin(7u, "air.thread_index_in_simdgroup", "uint", "warp_lane_id")});
    auto stage_info = node({});
    auto kernel_info = node({llvm::ValueAsMetadata::get(function), stage_info, argument_info});
    _module.getOrInsertNamedMetadata("air.kernel")->addOperand(kernel_info);
}

void MetalCodegenLLVMImpl::_add_raster_vertex_metadata(
    llvm::Function *function,
    llvm::ArrayRef<llvm::Type *> outputs) noexcept {
    LUISA_ASSERT(_raster_stage != nullptr &&
                     _raster_stage->stage() == xir::RasterStage::VERTEX,
                 "Vertex metadata requested without a vertex XIR stage.");
    auto node = [this](llvm::ArrayRef<llvm::Metadata *> operands) noexcept {
        return llvm::MDNode::get(_context, operands);
    };
    auto output_type = _raster_stage->type();
    auto output_member = [output_type](size_t index) noexcept {
        return output_type->is_structure() ?
                   output_type->members()[index] :
                   output_type;
    };
    llvm::SmallVector<llvm::Metadata *> output_metadata;
    output_metadata.reserve(outputs.size());
    for (auto i = 0u; i < outputs.size(); i++) {
        auto type = output_member(i);
        auto name = i == 0u ? luisa::string{"position"} :
                              luisa::format("varying{}", i - 1u);
        if (i == 0u) {
            output_metadata.emplace_back(node({md_string(_context, "air.position"),
                                               md_string(_context, "air.arg_type_name"),
                                               md_string(_context, _air_type_name(type)),
                                               md_string(_context, "air.arg_name"),
                                               md_string(_context, name)}));
        } else {
            auto semantic = luisa::format("user(locn{})", i - 1u);
            output_metadata.emplace_back(node({md_string(_context, "air.vertex_output"),
                                               md_string(_context, semantic),
                                               md_string(_context, "air.arg_type_name"),
                                               md_string(_context, _air_type_name(type)),
                                               md_string(_context, "air.arg_name"),
                                               md_string(_context, name)}));
        }
    }
    auto attribute_name = [](VertexAttributeType semantic) noexcept {
        switch (semantic) {
            case VertexAttributeType::Position: return luisa::string_view{"position"};
            case VertexAttributeType::Normal: return luisa::string_view{"normal"};
            case VertexAttributeType::Tangent: return luisa::string_view{"tangent"};
            case VertexAttributeType::Color: return luisa::string_view{"color"};
            case VertexAttributeType::UV0: return luisa::string_view{"uv0"};
            case VertexAttributeType::UV1: return luisa::string_view{"uv1"};
            case VertexAttributeType::UV2: return luisa::string_view{"uv2"};
            case VertexAttributeType::UV3: return luisa::string_view{"uv3"};
        }
        return luisa::string_view{"attribute"};
    };
    llvm::SmallVector<llvm::Metadata *> argument_metadata;
    auto attribute_count = _config.raster.vertex_attributes.size();
    argument_metadata.reserve(attribute_count + 4u);
    for (auto i = 0u; i < attribute_count; i++) {
        auto descriptor = _config.raster.vertex_attributes[i];
        auto input = _raster_vertex_input(descriptor.format);
        argument_metadata.emplace_back(node({md_i32(_context, i),
                                             md_string(_context, "air.vertex_input"),
                                             md_string(_context, "air.location_index"),
                                             md_i32(_context, i), md_i32(_context, 1u),
                                             md_string(_context, "air.arg_type_name"),
                                             md_string(_context, input.air_type_name),
                                             md_string(_context, "air.arg_name"),
                                             md_string(_context, attribute_name(descriptor.semantic))}));
    }
    auto builtin = [this, &node](uint32_t index,
                                 luisa::string_view semantic,
                                 luisa::string_view type,
                                 luisa::string_view name) noexcept {
        return node({md_i32(_context, index), md_string(_context, semantic),
                     md_string(_context, "air.arg_type_name"), md_string(_context, type),
                     md_string(_context, "air.arg_name"), md_string(_context, name)});
    };
    argument_metadata.emplace_back(builtin(
        static_cast<uint32_t>(attribute_count),
        "air.vertex_id", "uint", "vertex_id"));
    argument_metadata.emplace_back(builtin(
        static_cast<uint32_t>(attribute_count + 1u),
        "air.instance_id", "uint", "instance_id"));
    auto layout = _root_argument_layout();
    argument_metadata.emplace_back(_root_argument_metadata(
        layout.size, static_cast<uint32_t>(attribute_count + 2u)));
    argument_metadata.emplace_back(node({md_i32(_context, static_cast<uint32_t>(attribute_count + 3u)),
                                         md_string(_context, "air.buffer"),
                                         md_string(_context, "air.buffer_size"), md_i32(_context, 4u),
                                         md_string(_context, "air.location_index"), md_i32(_context, 1u), md_i32(_context, 1u),
                                         md_string(_context, "air.read"), md_string(_context, "air.address_space"),
                                         md_i32(_context, air_address_space_constant),
                                         md_string(_context, "air.arg_type_size"), md_i32(_context, 4u),
                                         md_string(_context, "air.arg_type_align_size"), md_i32(_context, 4u),
                                         md_string(_context, "air.arg_type_name"), md_string(_context, "uint"),
                                         md_string(_context, "air.arg_name"), md_string(_context, "object_id")}));
    auto stage_info = node(output_metadata);
    auto argument_info = node(argument_metadata);
    auto vertex_info = node({llvm::ValueAsMetadata::get(function),
                             stage_info, argument_info});
    _module.getOrInsertNamedMetadata("air.vertex")->addOperand(vertex_info);
}

void MetalCodegenLLVMImpl::_add_raster_fragment_metadata(
    llvm::Function *function,
    llvm::ArrayRef<llvm::Type *> outputs) noexcept {
    LUISA_ASSERT(_raster_stage != nullptr &&
                     _raster_stage->stage() == xir::RasterStage::FRAGMENT,
                 "Fragment metadata requested without a fragment XIR stage.");
    auto node = [this](llvm::ArrayRef<llvm::Metadata *> operands) noexcept {
        return llvm::MDNode::get(_context, operands);
    };
    auto return_type = _raster_stage->type();
    auto return_member = [return_type](size_t index) noexcept {
        return return_type->is_structure() ?
                   return_type->members()[index] :
                   return_type;
    };
    llvm::SmallVector<llvm::Metadata *> output_metadata;
    output_metadata.reserve(
        outputs.size() +
        (_raster_depth_mode == AIRRasterDepthMode::NONE ? 0u : 1u));
    for (auto i = 0u; i < outputs.size(); i++) {
        output_metadata.emplace_back(node({md_string(_context, "air.render_target"),
                                           md_i32(_context, i), md_i32(_context, 0u),
                                           md_string(_context, "air.arg_type_name"),
                                           md_string(_context, _air_type_name(return_member(i)))}));
    }
    if (_raster_depth_mode != AIRRasterDepthMode::NONE) {
        auto qualifier = [&]() noexcept -> luisa::string_view {
            switch (_raster_depth_mode) {
                case AIRRasterDepthMode::ANY: return "air.any";
                case AIRRasterDepthMode::GREATER_EQUAL: return "air.greater";
                case AIRRasterDepthMode::LESS_EQUAL: return "air.less";
                case AIRRasterDepthMode::NONE: break;
            }
            LUISA_ERROR_WITH_LOCATION("Invalid Metal AIR shader-depth mode.");
        }();
        output_metadata.emplace_back(node({
            md_string(_context, "air.depth"),
            md_string(_context, "air.depth_qualifier"),
            md_string(_context, qualifier),
            md_string(_context, "air.arg_type_name"),
            md_string(_context, "float"),
            md_string(_context, "air.arg_name"),
            md_string(_context, "depth")}));
    }
    const xir::Argument *payload_argument = nullptr;
    for (auto argument : _raster_stage->arguments()) {
        payload_argument = argument;
        break;
    }
    LUISA_ASSERT(payload_argument != nullptr,
                 "Fragment metadata requires a payload argument.");
    auto payload_type = payload_argument->type();
    auto payload_count = payload_type->is_structure() ?
                             payload_type->members().size() :
                             1u;
    auto payload_member = [payload_type](size_t index) noexcept {
        return payload_type->is_structure() ?
                   payload_type->members()[index] :
                   payload_type;
    };
    llvm::SmallVector<llvm::Metadata *> argument_metadata;
    argument_metadata.reserve(payload_count + 5u);
    for (auto i = 0u; i < payload_count; i++) {
        auto type = payload_member(i);
        auto type_name = _air_type_name(type);
        auto name = i == 0u ? luisa::string{"position"} :
                              luisa::format("varying{}", i - 1u);
        if (i == 0u) {
            argument_metadata.emplace_back(node({md_i32(_context, i), md_string(_context, "air.position"),
                                                 md_string(_context, "air.center"),
                                                 md_string(_context, "air.no_perspective"),
                                                 md_string(_context, "air.arg_type_name"), md_string(_context, type_name),
                                                 md_string(_context, "air.arg_name"), md_string(_context, name)}));
        } else {
            auto semantic = luisa::format("user(locn{})", i - 1u);
            auto element = type->is_vector() ? type->element() : type;
            if (element->is_float16() || element->is_float32()) {
                argument_metadata.emplace_back(node({md_i32(_context, i), md_string(_context, "air.fragment_input"),
                                                     md_string(_context, semantic), md_string(_context, "air.center"),
                                                     md_string(_context, "air.perspective"),
                                                     md_string(_context, "air.arg_type_name"), md_string(_context, type_name),
                                                     md_string(_context, "air.arg_name"), md_string(_context, name)}));
            } else {
                argument_metadata.emplace_back(node({md_i32(_context, i), md_string(_context, "air.fragment_input"),
                                                     md_string(_context, semantic), md_string(_context, "air.flat"),
                                                     md_string(_context, "air.arg_type_name"), md_string(_context, type_name),
                                                     md_string(_context, "air.arg_name"), md_string(_context, name)}));
            }
        }
    }
    auto primitive_index = static_cast<uint32_t>(payload_count);
    auto barycentrics_index = primitive_index + 1u;
    auto front_facing_index = primitive_index + 2u;
    auto root_index = primitive_index + 3u;
    auto object_id_index = primitive_index + 4u;
    argument_metadata.emplace_back(node({md_i32(_context, primitive_index), md_string(_context, "air.primitive_id"),
                                         md_string(_context, "air.arg_type_name"), md_string(_context, "uint"),
                                         md_string(_context, "air.arg_name"), md_string(_context, "primitive_id")}));
    argument_metadata.emplace_back(node({md_i32(_context, barycentrics_index), md_string(_context, "air.barycentric_coord"),
                                         md_string(_context, "air.center"), md_string(_context, "air.perspective"),
                                         md_string(_context, "air.arg_type_name"), md_string(_context, "float3"),
                                         md_string(_context, "air.arg_name"), md_string(_context, "barycentrics")}));
    argument_metadata.emplace_back(node({md_i32(_context, front_facing_index), md_string(_context, "air.front_facing"),
                                         md_string(_context, "air.arg_type_name"), md_string(_context, "bool"),
                                         md_string(_context, "air.arg_name"), md_string(_context, "front_facing")}));
    auto layout = _root_argument_layout();
    argument_metadata.emplace_back(
        _root_argument_metadata(layout.size, root_index));
    argument_metadata.emplace_back(node({md_i32(_context, object_id_index), md_string(_context, "air.buffer"),
                                         md_string(_context, "air.buffer_size"), md_i32(_context, 4u),
                                         md_string(_context, "air.location_index"), md_i32(_context, 1u), md_i32(_context, 1u),
                                         md_string(_context, "air.read"), md_string(_context, "air.address_space"),
                                         md_i32(_context, air_address_space_constant),
                                         md_string(_context, "air.arg_type_size"), md_i32(_context, 4u),
                                         md_string(_context, "air.arg_type_align_size"), md_i32(_context, 4u),
                                         md_string(_context, "air.arg_type_name"), md_string(_context, "uint"),
                                         md_string(_context, "air.arg_name"), md_string(_context, "object_id")}));
    auto stage_info = node(output_metadata);
    auto argument_info = node(argument_metadata);
    auto fragment_info = node({llvm::ValueAsMetadata::get(function),
                               stage_info, argument_info});
    _module.getOrInsertNamedMetadata("air.fragment")->addOperand(fragment_info);
}

void MetalCodegenLLVMImpl::_add_module_metadata() noexcept {
    auto sdk_values = std::array<uint32_t, 2u>{_config.sdk_version.major, _config.sdk_version.minor};
    auto sdk = llvm::ConstantDataArray::get(_context, sdk_values);
    _module.addModuleFlag(llvm::Module::ModFlagBehavior::Warning, "SDK Version", llvm::ConstantAsMetadata::get(sdk));
    _module.addModuleFlag(llvm::Module::ModFlagBehavior::Error, "wchar_size", 4u);
    _module.addModuleFlag(llvm::Module::ModFlagBehavior::Max, "frame-pointer", 2u);
    _module.addModuleFlag(llvm::Module::ModFlagBehavior::Max, "air.max_device_buffers", 31u);
    _module.addModuleFlag(llvm::Module::ModFlagBehavior::Max, "air.max_constant_buffers", 31u);
    _module.addModuleFlag(llvm::Module::ModFlagBehavior::Max, "air.max_threadgroup_buffers", 31u);
    _module.addModuleFlag(llvm::Module::ModFlagBehavior::Max, "air.max_textures", 128u);
    _module.addModuleFlag(llvm::Module::ModFlagBehavior::Max, "air.max_read_write_textures", 8u);
    _module.addModuleFlag(llvm::Module::ModFlagBehavior::Max, "air.max_samplers", 16u);
    auto node = [this](llvm::ArrayRef<llvm::Metadata *> operands) noexcept { return llvm::MDNode::get(_context, operands); };
    _module.getOrInsertNamedMetadata("llvm.ident")->addOperand(node({md_string(_context, "LuisaCompute XIR Metal AIR LLVM codegen")}));
    _module.getOrInsertNamedMetadata("air.version")->addOperand(node({md_i32(_context, _config.air_version.major), md_i32(_context, _config.air_version.minor), md_i32(_context, _config.air_version.patch)}));
    _module.getOrInsertNamedMetadata("air.language_version")->addOperand(node({md_string(_context, "Metal"), md_i32(_context, _config.metal_version.major), md_i32(_context, _config.metal_version.minor), md_i32(_context, _config.metal_version.patch)}));
    auto compile_options = _module.getOrInsertNamedMetadata("air.compile_options");
    compile_options->addOperand(node({md_string(_context, "air.compile.denorms_disable")}));
    compile_options->addOperand(node({md_string(_context, _config.enable_fast_math ?
                                                              "air.compile.fast_math_enable" :
                                                              "air.compile.fast_math_disable")}));
    compile_options->addOperand(node({md_string(_context, "air.compile.framebuffer_fetch_enable")}));
    if (!_config.source_file.empty()) {
        _module.getOrInsertNamedMetadata("air.source_file_name")->addOperand(node({md_string(_context, _config.source_file)}));
    }
}

void MetalCodegenLLVMImpl::_collect_print_formats(
    const xir::Module &module) noexcept {
    LUISA_ASSERT(_print_formats.empty() && _print_tokens.empty(),
                 "Metal printer formats were collected twice.");
    for (auto function : module.function_list()) {
        if (auto definition = function->definition()) {
            definition->traverse_instructions(
                [this](const xir::Instruction *instruction) noexcept {
                    if (!instruction->isa<xir::PrintInst>()) { return; }
                    auto print = static_cast<const xir::PrintInst *>(instruction);
                    luisa::vector<const Type *> members;
                    luisa::vector<const Type *> argument_types;
                    members.reserve(print->operand_count() + 2u);
                    argument_types.reserve(print->operand_count());
                    members.emplace_back(Type::of<uint32_t>());
                    members.emplace_back(Type::of<uint32_t>());
                    for (auto operand_use : print->operand_uses()) {
                        auto operand = operand_use->value();
                        LUISA_ASSERT(operand != nullptr && operand->type() != nullptr,
                                     "Metal printer has an untyped operand.");
                        members.emplace_back(operand->type());
                        argument_types.emplace_back(operand->type());
                    }
                    auto record_type = Type::structure(members);
                    auto native_format = _shader_log_format(
                        print->format(), argument_types);
                    auto token = static_cast<uint32_t>(_print_formats.size());
                    for (auto i = 0u; i < _print_formats.size(); i++) {
                        auto &&format = _print_formats[i];
                        if (format.format == print->format() &&
                            format.record_type == record_type) {
                            token = i;
                            break;
                        }
                    }
                    if (token == _print_formats.size()) {
                        _print_formats.emplace_back(
                            PrintFormat{print->format(),
                                        std::move(native_format),
                                        record_type});
                    }
                    auto [iter, inserted] = _print_tokens.try_emplace(print, token);
                    LUISA_ASSERT(inserted && iter->second == token,
                                 "Metal printer token was assigned twice.");
                });
        }
    }
    _result.format_types.reserve(_print_formats.size());
    for (auto &&format : _print_formats) {
        _result.format_types.emplace_back(
            format.format, luisa::string{format.record_type->description()});
    }
}

}// namespace luisa::compute::metal::detail
