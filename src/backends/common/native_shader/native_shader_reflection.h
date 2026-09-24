#pragma once

// Shared native-shader reflection helpers (Phase 0.5 of the native-shader
// injection plan).
//
// The DirectX and Vulkan native-shader extensions both have to turn a compiled
// module into the same `NativeShaderResourceBinding` table, and both need the
// shader's workgroup size and a canonical binding order.  The Vulkan route has
// no DXIL reflection available (it consumes SPIR-V, also on non-Windows hosts),
// so the SPIR-V module itself is the authoritative reflection source:
//
//  * `OpVariable` (with its `OpTypePointer` pointee) gives the resource class:
//    images/samplers/typed buffers/UBOs/SSBOs.
//  * `OpDecorate Binding/DescriptorSet` gives `(register_index, space_index)`.
//  * `OpDecorate NonWritable/NonReadable` (on the block or its members)
//    distinguishes a read-only storage buffer (`readonly buffer` /
//    `StructuredBuffer`) from a writable one.
//  * `OpExecutionMode ... LocalSize` gives the workgroup size, and a
//    `PushConstant` storage-class variable declares the push-constant block.
//
// This header is intentionally dependency-free (it parses the binary format
// directly instead of using SPIRV-Headers / SPIRV-Tools), so the DirectX
// backend, the Vulkan backend and the host-side unit tests can all include it.

#include <algorithm>
#include <cstdint>
#include <cstring>

#include <luisa/core/basic_traits.h>
#include <luisa/core/basic_types.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/backends/ext/native_shader_ext.h>

#include "../spirv/spirv_codegen/target_feature_mask.h"

namespace luisa::compute::native_shader_reflection {

namespace spirv {

inline constexpr uint32_t magic = 0x07230203u;
inline constexpr uint32_t magic_reversed = 0x03022307u;
inline constexpr size_t header_word_count = 5u;

enum Op : uint32_t {
    OpName = 5u,
    OpMemberName = 6u,
    OpCapability = 17u,
    OpEntryPoint = 15u,
    OpExecutionMode = 16u,
    OpTypeVoid = 19u,
    OpTypeBool = 20u,
    OpTypeInt = 21u,
    OpTypeFloat = 22u,
    OpTypeVector = 23u,
    OpTypeMatrix = 24u,
    OpTypeImage = 25u,
    OpTypeSampler = 26u,
    OpTypeSampledImage = 27u,
    OpTypeArray = 28u,
    OpTypeRuntimeArray = 29u,
    OpTypeStruct = 30u,
    OpTypePointer = 32u,
    OpTypeAccelerationStructureKHR = 5341u,
    OpConstantTrue = 41u,
    OpConstantFalse = 42u,
    OpConstant = 43u,
    OpConstantComposite = 44u,
    OpVariable = 59u,
    OpDecorate = 71u,
    OpMemberDecorate = 72u,
    OpDecorateId = 332u,
};

enum Decoration : uint32_t {
    DecorationBlock = 2u,
    DecorationBufferBlock = 3u,
    DecorationRowMajor = 4u,
    DecorationColMajor = 5u,
    DecorationArrayStride = 6u,
    DecorationBuiltIn = 11u,
    DecorationNonWritable = 24u,
    DecorationNonReadable = 25u,
    DecorationOffset = 35u,
    DecorationBinding = 33u,
    DecorationDescriptorSet = 34u,
};

enum StorageClass : uint32_t {
    StorageClassUniformConstant = 0u,
    StorageClassInput = 1u,
    StorageClassUniform = 2u,
    StorageClassOutput = 3u,
    StorageClassWorkgroup = 4u,
    StorageClassCrossWorkgroup = 5u,
    StorageClassPrivate = 6u,
    StorageClassFunction = 7u,
    StorageClassGeneric = 8u,
    StorageClassPushConstant = 9u,
    StorageClassAtomicCounter = 10u,
    StorageClassImage = 11u,
    StorageClassStorageBuffer = 12u,
};

enum ExecutionModel : uint32_t {
    ExecutionModelGLCompute = 5u,
    ExecutionModelKernel = 6u,
};

enum ExecutionMode : uint32_t {
    ExecutionModeLocalSize = 17u,
    ExecutionModeLocalSizeId = 38u,
};

enum ImageDim : uint32_t {
    Dim1D = 0u,
    Dim2D = 1u,
    Dim3D = 2u,
    DimCube = 3u,
    DimBuffer = 5u,
};

// `OpTypeImage`'s `Sampled` operand.
enum ImageSampled : uint32_t {
    SampledUnknown = 0u,
    SampledImage = 1u,
    SampledStorage = 2u,
};

}// namespace spirv

struct SpirvReflection {
    luisa::vector<NativeShaderResourceBinding> bindings;
    uint3 block_size{0u, 0u, 0u};
    bool has_push_constant{false};
    uint32_t push_constant_size{0u};
    // Entry point name of the (first) compute entry point, i.e. the name the
    // Vulkan pipeline must be created with.
    luisa::string entry_point;
    // Raw `OpCapability` operands, and the corresponding requirement bits in
    // the runtime's persistent SPIR-V artifact feature mask (`lc::spirv::
    // target_feature`), used to fail closed before pipeline creation (R12).
    luisa::vector<uint32_t> capabilities;
    uint64_t required_features{0u};
    luisa::string error;
    [[nodiscard]] bool ok() const noexcept { return error.empty(); }
};

namespace detail {

// Maps a SPIR-V capability to the runtime's persistent artifact requirement
// mask. Capabilities that are core for compute shaders in every supported
// Vulkan version (Shader, Matrix, ImageQuery, DerivativeControl, ...) map to 0.
[[nodiscard]] inline uint64_t capability_requirement(uint32_t capability) noexcept {
    using namespace lc::spirv;
    switch (capability) {
        case 8u: return target_feature::storage_buffer_16bit_access;// Float16Buffer
        case 9u: return target_feature::shader_float16;            // Float16
        case 10u: return target_feature::shader_float64;           // Float64
        case 11u: return target_feature::shader_int64;             // Int64
        case 12u: return target_feature::shader_buffer_int64_atomics |
                         target_feature::shader_shared_int64_atomics;// Int64Atomics
        case 22u: return target_feature::shader_int16;             // Int16
        case 39u: return target_feature::shader_int8;              // Int8
        case 55u: return target_feature::storage_image_read_without_format;
        case 56u: return target_feature::storage_image_write_without_format;
        default: return 0u;
    }
}

// Human-readable names of the requirement bits set in `mask`.
[[nodiscard]] inline luisa::string describe_required_features(uint64_t mask) noexcept {
    luisa::string result;
    for (auto feature : lc::spirv::spirv_target_feature_descriptions) {
        if ((mask & feature.bit) == 0u || feature.bit == 0u) { continue; }
        if (!result.empty()) { result.append(", "); }
        result.append(feature.name.data(), feature.name.size());
    }
    return result;
}

struct TypeInfo {
    enum class Kind : uint8_t {
        Unknown,
        Void,
        Bool,
        Int,
        Float,
        Vector,
        Matrix,
        Image,
        Sampler,
        SampledImage,
        Array,
        RuntimeArray,
        Struct,
        Pointer,
        AccelerationStructure,
    };
    Kind kind{Kind::Unknown};
    uint32_t width{0u};       // Int/Float bit width
    uint32_t count{0u};       // Vector components / Matrix columns
    uint32_t element{0u};     // Vector/Matrix/Array/Pointer/Image sampled type
    uint32_t length_id{0u};   // Array length (constant result id)
    uint32_t storage_class{0u};// Pointer
    uint32_t dim{0u};          // Image
    uint32_t sampled{0u};      // Image
    luisa::vector<uint32_t> members;// Struct
};

struct DecorationInfo {
    uint32_t binding{~0u};
    uint32_t descriptor_set{~0u};
    bool non_writable{false};
    bool non_readable{false};
    bool block{false};
    bool buffer_block{false};
    bool built_in{false};
};

struct VariableInfo {
    uint32_t id{0u};
    uint32_t type_id{0u};
    uint32_t storage_class{~0u};
};

[[nodiscard]] inline uint32_t word_count_of(uint32_t instruction) noexcept {
    return instruction >> 16u;
}
[[nodiscard]] inline uint32_t opcode_of(uint32_t instruction) noexcept {
    return instruction & 0xffffu;
}

}// namespace detail

// Parses a SPIR-V module (as raw words, i.e. `NativeShaderCompileResult::binary`
// of the Vulkan route) into the shared reflection form.
//
// `set`/`binding`: taken from the module's decorations; a variable without a
// `Binding` decoration is assigned the next free binding of its descriptor set
// (glslang's automatic assignment), which keeps synthetic/test modules usable.
//
// `kind`: derived from the resource's SPIR-V type as described at the top of
// this file.  `usage` receives the default usage of that kind (SRV/CBV/sampler
// -> READ, UAV class -> READ_WRITE); the caller may override it.
[[nodiscard]] inline SpirvReflection parse_spirv(
    luisa::span<const std::byte> code) noexcept {
    using namespace detail;
    SpirvReflection result;
    if (code.size() < spirv::header_word_count * sizeof(uint32_t) ||
        code.size() % sizeof(uint32_t) != 0u) {
        result.error = "SPIR-V module is too small or not word-aligned.";
        return result;
    }
    auto words = reinterpret_cast<const uint32_t *>(code.data());
    auto word_count = code.size() / sizeof(uint32_t);
    if (words[0] != spirv::magic) {
        result.error = words[0] == spirv::magic_reversed ?
                           "SPIR-V module is byte-swapped; the native shader "
                           "route requires little-endian words." :
                           "SPIR-V module has an invalid magic number.";
        return result;
    }
    auto bound = words[3];

    luisa::unordered_map<uint32_t, TypeInfo> types;
    luisa::unordered_map<uint32_t, uint64_t> constants;
    luisa::unordered_map<uint32_t, DecorationInfo> decorations;
    luisa::unordered_map<uint32_t, luisa::unordered_map<uint32_t, DecorationInfo>> member_decorations;
    luisa::vector<VariableInfo> variables;
    luisa::unordered_map<uint32_t, luisa::string> names;

    for (auto offset = spirv::header_word_count; offset < word_count;) {
        auto instruction = words[offset];
        auto length = word_count_of(instruction);
        if (length == 0u || offset + length > word_count) {
            result.error = "SPIR-V module contains a malformed instruction.";
            return result;
        }
        auto op = opcode_of(instruction);
        auto *operands = words + offset + 1u;
        auto operand_count = static_cast<size_t>(length) - 1u;
        switch (op) {
            case spirv::OpName: {
                if (operand_count >= 2u) {
                    auto id = operands[0];
                    auto *chars = reinterpret_cast<const char *>(operands + 1u);
                    auto byte_count = operand_count - 1u;
                    auto size = strnlen(chars, byte_count * sizeof(uint32_t));
                    names.emplace(id, luisa::string{chars, size});
                }
                break;
            }
            case spirv::OpTypeVoid:
            case spirv::OpTypeBool:
            case spirv::OpTypeSampler:
            case spirv::OpTypeAccelerationStructureKHR: {
                auto kind = op == spirv::OpTypeVoid ? TypeInfo::Kind::Void :
                            op == spirv::OpTypeBool ? TypeInfo::Kind::Bool :
                            op == spirv::OpTypeSampler ? TypeInfo::Kind::Sampler :
                                                         TypeInfo::Kind::AccelerationStructure;
                if (operand_count >= 1u) {
                    TypeInfo info;
                    info.kind = kind;
                    types.emplace(operands[0], info);
                }
                break;
            }
            case spirv::OpTypeInt:
            case spirv::OpTypeFloat: {
                if (operand_count >= 2u) {
                    TypeInfo info;
                    info.kind = op == spirv::OpTypeInt ? TypeInfo::Kind::Int : TypeInfo::Kind::Float;
                    info.width = operands[1];
                    types.emplace(operands[0], info);
                }
                break;
            }
            case spirv::OpTypeVector: {
                if (operand_count >= 3u) {
                    TypeInfo info;
                    info.kind = TypeInfo::Kind::Vector;
                    info.element = operands[1];
                    info.count = operands[2];
                    types.emplace(operands[0], info);
                }
                break;
            }
            case spirv::OpTypeMatrix: {
                if (operand_count >= 3u) {
                    TypeInfo info;
                    info.kind = TypeInfo::Kind::Matrix;
                    info.element = operands[1];
                    info.count = operands[2];
                    types.emplace(operands[0], info);
                }
                break;
            }
            case spirv::OpTypeImage: {
                if (operand_count >= 8u) {
                    TypeInfo info;
                    info.kind = TypeInfo::Kind::Image;
                    info.element = operands[1];// sampled type
                    info.dim = operands[2];
                    info.sampled = operands[6];
                    types.emplace(operands[0], info);
                }
                break;
            }
            case spirv::OpTypeSampledImage: {
                if (operand_count >= 2u) {
                    TypeInfo info;
                    info.kind = TypeInfo::Kind::SampledImage;
                    info.element = operands[1];
                    types.emplace(operands[0], info);
                }
                break;
            }
            case spirv::OpTypeArray: {
                if (operand_count >= 3u) {
                    TypeInfo info;
                    info.kind = TypeInfo::Kind::Array;
                    info.element = operands[1];
                    info.length_id = operands[2];
                    types.emplace(operands[0], info);
                }
                break;
            }
            case spirv::OpTypeRuntimeArray: {
                if (operand_count >= 2u) {
                    TypeInfo info;
                    info.kind = TypeInfo::Kind::RuntimeArray;
                    info.element = operands[1];
                    types.emplace(operands[0], info);
                }
                break;
            }
            case spirv::OpTypeStruct: {
                if (operand_count >= 1u) {
                    TypeInfo info;
                    info.kind = TypeInfo::Kind::Struct;
                    info.members.assign(operands + 1u, operands + operand_count);
                    types.emplace(operands[0], info);
                }
                break;
            }
            case spirv::OpTypePointer: {
                if (operand_count >= 3u) {
                    TypeInfo info;
                    info.kind = TypeInfo::Kind::Pointer;
                    info.storage_class = operands[1];
                    info.element = operands[2];
                    types.emplace(operands[0], info);
                }
                break;
            }
            case spirv::OpConstant:
            case spirv::OpConstantTrue:
            case spirv::OpConstantFalse: {
                if (operand_count >= 2u) {
                    uint64_t value = 0u;
                    if (op == spirv::OpConstantTrue) {
                        value = 1u;
                    } else if (op == spirv::OpConstantFalse) {
                        value = 0u;
                    } else {
                        auto type_id = operands[0];
                        auto width = 32u;
                        if (auto it = types.find(type_id); it != types.end()) {
                            width = it->second.width;
                        }
                        if (width > 32u && operand_count >= 4u) {
                            value = (static_cast<uint64_t>(operands[2]) << 32u) |
                                    static_cast<uint64_t>(operands[1]);
                        } else {
                            value = operands[1];
                        }
                    }
                    constants.emplace(operands[0], value);
                }
                break;
            }
            case spirv::OpVariable: {
                // OpVariable %type %result_id [%storage_class] [initializer]
                if (operand_count >= 2u) {
                    VariableInfo info;
                    info.type_id = operands[0];
                    info.id = operands[1];
                    auto pointer_storage_class = [&]() noexcept {
                        if (auto it = types.find(info.type_id); it != types.end()) {
                            return it->second.storage_class;
                        }
                        return ~0u;
                    }();
                    info.storage_class = operand_count >= 3u ?
                                             operands[2] :
                                             pointer_storage_class;
                    variables.emplace_back(info);
                }
                break;
            }
            case spirv::OpDecorate:
            case spirv::OpDecorateId: {
                if (operand_count >= 2u) {
                    auto id = operands[0];
                    auto decoration = operands[1];
                    auto &info = decorations[id];
                    switch (decoration) {
                        case spirv::DecorationBinding:
                            if (operand_count >= 3u) { info.binding = operands[2]; }
                            break;
                        case spirv::DecorationDescriptorSet:
                            if (operand_count >= 3u) { info.descriptor_set = operands[2]; }
                            break;
                        case spirv::DecorationNonWritable: info.non_writable = true; break;
                        case spirv::DecorationNonReadable: info.non_readable = true; break;
                        case spirv::DecorationBlock: info.block = true; break;
                        case spirv::DecorationBufferBlock: info.buffer_block = true; break;
                        case spirv::DecorationBuiltIn: info.built_in = true; break;
                        default: break;
                    }
                }
                break;
            }
            case spirv::OpMemberDecorate: {
                if (operand_count >= 3u) {
                    auto id = operands[0];
                    auto member = operands[1];
                    auto decoration = operands[2];
                    auto &info = member_decorations[id][member];
                    switch (decoration) {
                        case spirv::DecorationNonWritable: info.non_writable = true; break;
                        case spirv::DecorationNonReadable: info.non_readable = true; break;
                        case spirv::DecorationOffset:
                            if (operand_count >= 4u) { info.binding = operands[3]; }
                            break;
                        default: break;
                    }
                }
                break;
            }
            case spirv::OpExecutionMode: {
                // OpExecutionMode %entry_point <mode> <operands...>
                if (operand_count >= 2u && result.block_size.x == 0u) {
                    auto mode = operands[1];
                    if (mode == spirv::ExecutionModeLocalSize && operand_count >= 5u) {
                        result.block_size = uint3{operands[2], operands[3], operands[4]};
                    } else if (mode == spirv::ExecutionModeLocalSizeId && operand_count >= 5u) {
                        auto get = [&](uint32_t at) noexcept {
                            if (auto it = constants.find(operands[at]); it != constants.end()) {
                                return static_cast<uint32_t>(it->second);
                            }
                            return 0u;
                        };
                        result.block_size = uint3{get(2), get(3), get(4)};
                    }
                }
                break;
            }
            case spirv::OpCapability: {
                if (operand_count >= 1u) {
                    auto capability = operands[0];
                    result.capabilities.emplace_back(capability);
                    result.required_features |= detail::capability_requirement(capability);
                }
                break;
            }
            case spirv::OpEntryPoint: {
                // OpEntryPoint <execution model> <entry point id> <name...>
                if (operand_count >= 3u && result.entry_point.empty()) {
                    auto model = operands[0];
                    if (model == spirv::ExecutionModelGLCompute ||
                        model == spirv::ExecutionModelKernel) {
                        auto *chars = reinterpret_cast<const char *>(operands + 2u);
                        auto byte_count = (operand_count - 2u) * sizeof(uint32_t);
                        auto size = strnlen(chars, byte_count);
                        result.entry_point = luisa::string{chars, size};
                    }
                }
                break;
            }
            default: break;
        }
        offset += length;
    }

    if (bound == 0u) {
        result.error = "SPIR-V module declares a zero id bound.";
        return result;
    }

    // Best-effort byte size of a type (used for UBO/SSBO/push-constant sizes).
    luisa::vector<int32_t> size_cache(bound, -1);
    auto type_size = [&](auto &&self, uint32_t type_id) noexcept -> int64_t {
        if (type_id == 0u || type_id >= bound) { return -1; }
        if (size_cache[type_id] >= 0) { return size_cache[type_id]; }
        size_cache[type_id] = -2;// visiting
        if (auto it = types.find(type_id); it != types.end()) {
            auto &info = it->second;
            switch (info.kind) {
                case TypeInfo::Kind::Bool:
                    size_cache[type_id] = 1;
                    return 1;
                case TypeInfo::Kind::Int:
                case TypeInfo::Kind::Float:
                    size_cache[type_id] = static_cast<int32_t>(info.width / 8u);
                    return info.width / 8u;
                case TypeInfo::Kind::Vector: {
                    auto element = self(self, info.element);
                    if (element < 0) { return -1; }
                    auto size = element * info.count;
                    size_cache[type_id] = static_cast<int32_t>(size);
                    return size;
                }
                case TypeInfo::Kind::Matrix: {
                    auto column = self(self, info.element);
                    if (column < 0) { return -1; }
                    auto size = column * info.count;
                    size_cache[type_id] = static_cast<int32_t>(size);
                    return size;
                }
                case TypeInfo::Kind::Array: {
                    auto element = self(self, info.element);
                    auto length = 0u;
                    if (auto c = constants.find(info.length_id); c != constants.end()) {
                        length = static_cast<uint32_t>(c->second);
                    }
                    if (element < 0 || length == 0u) { return -1; }
                    auto size = element * length;
                    size_cache[type_id] = static_cast<int32_t>(size);
                    return size;
                }
                case TypeInfo::Kind::Struct: {
                    int64_t size = 0;
                    auto any = false;
                    for (auto i = 0u; i < info.members.size(); i++) {
                        auto member_size = self(self, info.members[i]);
                        if (member_size < 0) { return -1; }
                        auto offset = static_cast<int64_t>(i) * member_size;// fallback
                        if (auto m = member_decorations.find(type_id); m != member_decorations.end()) {
                            if (auto d = m->second.find(i); d != m->second.end() && d->second.binding != ~0u) {
                                offset = d->second.binding;
                            }
                        }
                        size = std::max(size, offset + member_size);
                        any = true;
                    }
                    if (!any) { return -1; }
                    size_cache[type_id] = static_cast<int32_t>(size);
                    return size;
                }
                default: break;
            }
        }
        return -1;
    };

    // Push-constant block.
    for (auto &&variable : variables) {
        if (variable.storage_class != spirv::StorageClassPushConstant) { continue; }
        result.has_push_constant = true;
        auto pointee = variable.type_id;
        if (auto it = types.find(pointee); it != types.end() &&
            it->second.kind == TypeInfo::Kind::Pointer) {
            pointee = it->second.element;
        }
        auto size = type_size(type_size, pointee);
        if (size > 0) {
            result.push_constant_size = static_cast<uint32_t>(size);
        }
        break;
    }

    auto is_writable = [&](uint32_t type_id, DecorationInfo const &decoration) noexcept {
        if (decoration.non_writable) { return false; }
        if (auto it = member_decorations.find(type_id); it != member_decorations.end()) {
            for (auto &&member : it->second) {
                if (!member.second.non_writable) { return true; }
            }
            return false;// every decorated member is read-only
        }
        return true;
    };
    static const DecorationInfo empty_decoration{};
    auto decoration_of = [&](uint32_t id) noexcept -> DecorationInfo const & {
        if (auto it = decorations.find(id); it != decorations.end()) {
            return it->second;
        }
        return empty_decoration;
    };

    auto texture_kind = [](uint32_t dim, bool writable) noexcept {
        switch (dim) {
            case spirv::Dim3D:
                return writable ? NativeShaderResourceKind::RWTexture3D :
                                  NativeShaderResourceKind::Texture3D;
            case spirv::DimBuffer:
                return writable ? NativeShaderResourceKind::RWTypedBuffer :
                                  NativeShaderResourceKind::TypedBuffer;
            default:
                return writable ? NativeShaderResourceKind::RWTexture2D :
                                  NativeShaderResourceKind::Texture2D;
        }
    };

    // Descriptor bindings.
    auto next_free_binding = [&](uint32_t set) noexcept {
        auto candidate = 0u;
        for (auto &&binding : result.bindings) {
            if (binding.space_index == set && binding.register_index >= candidate) {
                candidate = binding.register_index + 1u;
            }
        }
        return candidate;
    };
    for (auto &&variable : variables) {
        auto storage_class = variable.storage_class;
        if (storage_class != spirv::StorageClassUniformConstant &&
            storage_class != spirv::StorageClassUniform &&
            storage_class != spirv::StorageClassStorageBuffer) {
            continue;
        }
        auto pointee = variable.type_id;
        if (auto it = types.find(pointee); it != types.end() &&
            it->second.kind == TypeInfo::Kind::Pointer) {
            pointee = it->second.element;
        }
        auto type_it = types.find(pointee);
        if (type_it == types.end()) { continue; }
        auto &type = type_it->second;
        // `Binding` / `DescriptorSet` / `NonWritable` / `NonReadable` are
        // emitted on the *variable*, while `Block` / `BufferBlock` sit on the
        // pointee struct; merge both into one effective decoration set.
        auto decoration = decoration_of(pointee);
        if (auto const &variable_decoration = decoration_of(variable.id);
            variable.id != 0u) {
            if (variable_decoration.binding != ~0u) {
                decoration.binding = variable_decoration.binding;
            }
            if (variable_decoration.descriptor_set != ~0u) {
                decoration.descriptor_set = variable_decoration.descriptor_set;
            }
            decoration.non_writable |= variable_decoration.non_writable;
            decoration.non_readable |= variable_decoration.non_readable;
            decoration.block |= variable_decoration.block;
            decoration.buffer_block |= variable_decoration.buffer_block;
        }
        NativeShaderResourceBinding binding;
        binding.array_size = 1u;
        auto array_size_from = [&](uint32_t id) noexcept {
            if (auto it = types.find(id); it != types.end() &&
                it->second.kind == TypeInfo::Kind::Array) {
                if (auto c = constants.find(it->second.length_id); c != constants.end()) {
                    return static_cast<uint32_t>(c->second);
                }
            }
            return 1u;
        };
        switch (type.kind) {
            case TypeInfo::Kind::Image: {
                auto writable = type.sampled == spirv::SampledStorage && !decoration.non_writable;
                binding.kind = texture_kind(type.dim, writable);
                break;
            }
            case TypeInfo::Kind::SampledImage: {
                uint32_t dim = spirv::Dim2D;
                if (auto inner = types.find(type.element); inner != types.end()) {
                    if (inner->second.kind == TypeInfo::Kind::Image) {
                        dim = inner->second.dim;
                    } else if (inner->second.kind == TypeInfo::Kind::Array) {
                        binding.array_size = array_size_from(type.element);
                        if (auto element = types.find(inner->second.element);
                            element != types.end() &&
                            element->second.kind == TypeInfo::Kind::Image) {
                            dim = element->second.dim;
                        }
                    }
                }
                binding.kind = texture_kind(dim, false);
                break;
            }
            case TypeInfo::Kind::Sampler:
                binding.kind = NativeShaderResourceKind::Sampler;
                break;
            case TypeInfo::Kind::Array:
                // Array of samplers / sampled images / buffers.
                binding.array_size = array_size_from(pointee);
                if (auto element = types.find(type.element); element != types.end()) {
                    switch (element->second.kind) {
                        case TypeInfo::Kind::Sampler:
                            binding.kind = NativeShaderResourceKind::Sampler;
                            break;
                        case TypeInfo::Kind::SampledImage: {
                            uint32_t dim = spirv::Dim2D;
                            if (auto image = types.find(element->second.element);
                                image != types.end()) {
                                dim = image->second.dim;
                            }
                            binding.kind = texture_kind(dim, false);
                            break;
                        }
                        case TypeInfo::Kind::Struct:
                            binding.kind = is_writable(type.element, decoration_of(type.element)) ?
                                               NativeShaderResourceKind::RWStructuredBuffer :
                                               NativeShaderResourceKind::StructuredBuffer;
                            break;
                        default: continue;
                    }
                } else {
                    continue;
                }
                break;
            case TypeInfo::Kind::Struct: {
                if (storage_class == spirv::StorageClassUniform &&
                    decoration.block) {
                    binding.kind = NativeShaderResourceKind::ConstantBuffer;
                    auto size = type_size(type_size, pointee);
                    if (size > 0) {
                        binding.size_bytes = static_cast<uint32_t>(size);
                    }
                } else {
                    binding.kind = is_writable(pointee, decoration) ?
                                       NativeShaderResourceKind::RWStructuredBuffer :
                                       NativeShaderResourceKind::StructuredBuffer;
                    auto size = type_size(type_size, pointee);
                    if (size > 0) {
                        binding.size_bytes = static_cast<uint32_t>(size);
                    }
                }
                break;
            }
            case TypeInfo::Kind::AccelerationStructure:
                binding.kind = NativeShaderResourceKind::AccelerationStructure;
                break;
            default: continue;
        }
        binding.space_index = decoration.descriptor_set == ~0u ? 0u : decoration.descriptor_set;
        binding.register_index = decoration.binding == ~0u ?
                                     next_free_binding(binding.space_index) :
                                     decoration.binding;
        binding.usage = native_shader_default_usage(binding.kind);
        result.bindings.emplace_back(binding);
    }

    std::stable_sort(result.bindings.begin(), result.bindings.end(),
                     [](NativeShaderResourceBinding const &a,
                        NativeShaderResourceBinding const &b) noexcept {
                         if (a.space_index != b.space_index) { return a.space_index < b.space_index; }
                         if (a.register_index != b.register_index) { return a.register_index < b.register_index; }
                         return luisa::to_underlying(a.kind) < luisa::to_underlying(b.kind);
                     });
    return result;
}

}// namespace luisa::compute::native_shader_reflection
