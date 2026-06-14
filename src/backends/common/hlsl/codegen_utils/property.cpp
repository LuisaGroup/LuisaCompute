// Bindless Resources & Register Management

#include "../hlsl_codegen.h"
#include "../shader_property.h"
#include "../register_indexer.h"
#include "../codegen_stack_data.h"
#include <limits>

// Forward declarations for types used in this file
namespace luisa::compute {
class Type;
}

namespace lc::hlsl {
struct CodegenResult;
struct Property;
struct RegisterIndexer;
class CodegenStackData;
enum class ShaderVariableType : uint8_t;
}// namespace lc::hlsl

namespace lc::hlsl {

// Generate bindless resource bindings
void CodegenUtility::GenerateBindless(
    CodegenResult::Properties &properties,
    vstd::StringBuilder &str,
    bool isSpirV,
    uint &bind_count) {
    uint table_idx = isSpirV ? 2 : 1;
    auto add_prop = [&](ShaderVariableType svt) {
        properties.emplace_back(
            Property{
                svt,
                table_idx,
                0u, std::numeric_limits<uint>::max()});
    };

    if (opt->useBufferBindless) {
        if (opt->noRegister) {
            str << "ByteAddressBuffer bdls[];\n";
        } else {
            str << "ByteAddressBuffer bdls[]:register(t0,space"sv << vstd::to_string(table_idx) << ");\n"sv;
        }
        add_prop(ShaderVariableType::SRVBufferHeap);
        table_idx++;
        bind_count += 1;
    }
    if (opt->useTex2DBindless) {
        if (opt->noRegister) {
            str << "Texture2D<float4> _BindlessTex[];\n";
        } else {
            str << "Texture2D<float4> _BindlessTex[]:register(t0,space"sv << vstd::to_string(table_idx) << ");"sv;
        }
        add_prop(ShaderVariableType::SRVTextureHeap);
        table_idx++;
        str << CodegenUtility::ReadInternalHLSLFile("tex2d_bindless");
        bind_count += 1;
    }
    if (opt->useTex3DBindless) {
        if (opt->noRegister) {
            str << "Texture3D<float4> _BindlessTex3D[];\n"sv;
        } else {
            str << "Texture3D<float4> _BindlessTex3D[]:register(t0,space"sv << vstd::to_string(table_idx) << ");"sv;
        }
        add_prop(ShaderVariableType::SRVTextureHeap);
        table_idx++;
        str << CodegenUtility::ReadInternalHLSLFile("tex3d_bindless");
        bind_count += 1;
    }
}

// Preprocess properties before codegen
void CodegenUtility::PreprocessCodegenProperties(
    CodegenResult::Properties &properties,
    vstd::StringBuilder &varData,
    RegisterIndexer &registerCount,
    bool cbufferNonEmpty, bool isRaster, bool isSpirv, uint &bind_count) {
    // 1,0,0
    registerCount.init();
    if (isSpirv) {
        properties.emplace_back(
            Property{
                ShaderVariableType::ConstantValue,
                0,
                0,
                1});
    } else {
        if (!isRaster) {
            properties.emplace_back(
                Property{
                    ShaderVariableType::ConstantValue,
                    4,
                    0,
                    1});
        }
    }
    properties.emplace_back(
        Property{
            ShaderVariableType::SamplerHeap,
            1u,
            0u,
            16u});
    if (cbufferNonEmpty) {
        registerCount.get(2)++;
        properties.emplace_back(
            Property{
                ShaderVariableType::StructuredBuffer,
                0,
                0u,
                1});
        bind_count += 2;
    }
    GenerateBindless(properties, varData, isSpirv, bind_count);
}

namespace detail {
[[nodiscard]] static auto can_accum_grad(const Type *t) noexcept {
    auto tt = t->tag();
    if (tt == Type::Tag::FLOAT16 ||
        tt == Type::Tag::FLOAT32 ||
        tt == Type::Tag::FLOAT64 ||
        tt == Type::Tag::STRUCTURE) {
        return true;
    }
    if (tt == Type::Tag::ARRAY ||
        tt == Type::Tag::VECTOR ||
        tt == Type::Tag::MATRIX) {
        return can_accum_grad(t->element());
    }
    return false;
}
}// namespace detail

// Postprocess final result
void CodegenUtility::PostprocessCodegenProperties(vstd::StringBuilder &finalResult, bool use_autodiff) {
    if (!opt->customStruct.empty()) {
        auto declStruct = [&](auto &&v) {
            finalResult << "struct " << v->GetStructName() << "{\n"
                        << v->GetStructDesc() << "};\n";
            // accum grad while using autodiff
            if (use_autodiff) {
                auto accum_grad = [&s = finalResult](luisa::string_view access, const Type *t) noexcept {
                    if (t->is_structure() || t->is_array()) {
                        s << luisa::format("_accum_grad_{:016X}(x_grad{}, dx{});\n", t->hash(), access, access);
                    } else {
                        s << luisa::format("_accum_grad(x_grad{}, dx{});\n", access, access);
                    }
                };
                if (auto t = v->GetType(); t->is_structure() || t->is_array()) {
                    finalResult << luisa::format("void _accum_grad_{:016X}(inout {} x_grad, {} dx){{\n",
                                                 t->hash(), v->GetStructName(), v->GetStructName());
                    if (t->is_structure()) {
                        for (auto i = 0u; i < t->members().size(); i++) {
                            if (auto m = t->members()[i]; detail::can_accum_grad(m)) {
                                accum_grad(luisa::format(".v{}", i), m);
                            }
                        }
                    } else if (detail::can_accum_grad(t->element())) {
                        finalResult << luisa::format("for(uint i=0;i<{};++i){{", t->dimension());
                        accum_grad(luisa::format(".v[i]"), t->element());
                        finalResult << "}\n";
                    }
                    finalResult << "}\n";
                }
            }
        };
        for (auto v : opt->customStructVector) {
            declStruct(v);
        }
        for (auto v : opt->customStructVectorAliased) {
            declStruct(v);
        }
    }
    for (auto &&kv : opt->sharedVariable) {
        finalResult << "groupshared "sv;
        GetTypeName(*kv.var.type()->element(), finalResult, Usage::READ, false);
        finalResult << ' ';
        GetVariableName(kv.func, kv.var, finalResult);
        finalResult << '[';
        vstd::to_string(kv.var.type()->dimension(), finalResult);
        finalResult << "];\n"sv;
    }
}

// Add debug printer for struct types
uint CodegenUtility::AddPrinter(vstd::string_view name, luisa::compute::Type const *structType) {
    auto z = opt->printer.size();
    opt->printer.emplace_back(name, structType);
    return z;
}

// Generate property bindings
void CodegenUtility::CodegenProperties(
    CodegenResult::Properties &properties,
    vstd::StringBuilder &varData,
    Function kernel,
    uint offset,
    RegisterIndexer &registerCount,
    uint &bind_count) {
    enum class RegisterType : uint8_t {
        CBV,
        UAV,
        SRV
    };
    auto Writable = [&](Variable const &v) {
        return (static_cast<uint>(kernel.variable_usage(v.uid())) & static_cast<uint>(Usage::WRITE)) != 0;
    };
    auto args = kernel.arguments();
    auto globally_coherent_iter = opt->globallyCoherentBuffers.find(kernel.builder());
    for (auto &&i : vstd::ptr_range(args.data() + offset, args.size() - offset)) {
        auto print = [&] {
            auto usage = kernel.variable_usage(i.uid());
            if (i.type()->is_buffer() || i.type()->is_texture()) {
                bool use_globallycoherent = false;
                auto attris = i.type()->member_attributes();
                if (!attris.empty()) {
                    for (auto &a : attris) {
                        if ((to_underlying(usage) & to_underlying(Usage::WRITE)) != 0) {
                            if (a.key == "cache"sv) {
                                if (a.value == "coherent"sv) {
                                    use_globallycoherent = true;
                                }
                            }
                        }
                    }
                }
                use_globallycoherent |= (globally_coherent_iter && globally_coherent_iter.value().contains(i.uid()));

                if (use_globallycoherent && (luisa::to_underlying(kernel.variable_usage(i.uid())) & luisa::to_underlying(Usage::WRITE)) != 0) {
                    varData << "globallycoherent "sv;
                }
            }
            GetTypeName(*i.type(), varData, usage);
            varData << ' ';
            GetVariableName(kernel, i, varData);
        };
        auto printInstBuffer = [&]<bool writable>() {
            if constexpr (writable)
                varData << "RWStructuredBuffer<_MeshInst> "sv;
            else
                varData << "StructuredBuffer<_MeshInst> "sv;
            GetVariableName(kernel, i, varData);
            varData << "Inst"sv;
        };
        auto genArg = [&]<RegisterType regisT, bool rtBuffer = false, bool writable = false>(ShaderVariableType sT, char v) {
            auto &&r = registerCount.get((uint8_t)regisT);
            Property prop = {
                .type = sT,
                .space_index = 0,
                .register_index = r,
                .array_size = 1};
            if constexpr (rtBuffer) {
                printInstBuffer.operator()<writable>();
                properties.emplace_back(prop);
            } else {
                print();
                properties.emplace_back(prop);
            }
            if (!opt->noRegister) {
                varData << ":register("sv << v;
                vstd::to_string(r, varData);
                varData << ");\n"sv;
            } else {
                varData << ";\n"sv;
            }
            r++;
            switch (sT) {
                case ShaderVariableType::ConstantBuffer:
                case ShaderVariableType::StructuredBuffer:
                case ShaderVariableType::RWStructuredBuffer:
                case ShaderVariableType::ConstantValue:
                case ShaderVariableType::SamplerHeap:
                    bind_count += 2;
                    break;
                case ShaderVariableType::SRVTextureHeap:
                case ShaderVariableType::UAVTextureHeap:
                case ShaderVariableType::SRVBufferHeap:
                case ShaderVariableType::UAVBufferHeap:
                case ShaderVariableType::CBVBufferHeap:
                case ShaderVariableType::SPIRVAccel:
                    bind_count += 1;
                    break;
                default: break;
            }
        };
        switch (i.type()->tag()) {
            case Type::Tag::TEXTURE:
                if (Writable(i)) {
                    genArg.operator()<RegisterType::UAV>(ShaderVariableType::UAVTextureHeap, 'u');
                } else {
                    genArg.operator()<RegisterType::SRV>(ShaderVariableType::SRVTextureHeap, 't');
                }
                break;
            case Type::Tag::BUFFER: {
                if (Writable(i)) {
                    genArg.operator()<RegisterType::UAV>(ShaderVariableType::RWStructuredBuffer, 'u');
                } else {
                    genArg.operator()<RegisterType::SRV>(ShaderVariableType::StructuredBuffer, 't');
                }
            } break;
            case Type::Tag::BINDLESS_ARRAY:
                genArg.operator()<RegisterType::SRV>(ShaderVariableType::StructuredBuffer, 't');
                break;
            case Type::Tag::ACCEL:
                if (Writable(i)) {
                    genArg.operator()<RegisterType::UAV, true, true>(ShaderVariableType::RWStructuredBuffer, 'u');
                } else {
                    genArg.operator()<RegisterType::SRV>(opt->isSpirv ? ShaderVariableType::SPIRVAccel : ShaderVariableType::StructuredBuffer, 't');
                    genArg.operator()<RegisterType::SRV, true>(ShaderVariableType::StructuredBuffer, 't');
                }
                break;
            case Type::Tag::CUSTOM: {
                if (i.type()->description() == "LC_IndirectDispatchBuffer"sv) {
                    genArg.operator()<RegisterType::UAV>(ShaderVariableType::RWStructuredBuffer, 'u');
                }
            } break;
            default: break;
        }
    }
    if (kernel.requires_printing()) {
        auto &&r = registerCount.get((uint8_t)RegisterType::UAV);
        {
            Property prop = {
                .type = ShaderVariableType::RWStructuredBuffer,
                .space_index = 0,
                .register_index = r,
                .array_size = 1};
            properties.emplace_back(prop);
            if (opt->noRegister) {
                varData << "RWStructuredBuffer<uint> _printCounter;\n";
            } else {
                varData << "RWStructuredBuffer<uint> _printCounter:register(u"sv;
            }
            vstd::to_string(r, varData);
            varData << ");\n"sv;
            r += 1;
        }
        {
            Property prop = {
                .type = ShaderVariableType::RWStructuredBuffer,
                .space_index = 0,
                .register_index = r,
                .array_size = 1};
            properties.emplace_back(prop);
            if (opt->noRegister) {
                varData << "RWByteAddressBuffer _printBuffer;\n";
            } else {
                varData << "RWByteAddressBuffer _printBuffer:register(u"sv;
            }
            vstd::to_string(r, varData);
            varData << ");\n"sv;
            r += 1;
        }
        bind_count += 4;
    }
}

}// namespace lc::hlsl
