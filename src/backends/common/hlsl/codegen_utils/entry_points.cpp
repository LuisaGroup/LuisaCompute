// Main Codegen Entry Points

#include "../hlsl_codegen.h"
#include "../codegen_stack_data.h"
#include "../register_indexer.h"
#include "../constant_printer.h"
#include "../register_indexer.h"

#ifndef LC_NO_HLSL_BUILTIN
#include "../builtin/hlsl_builtin.hpp"
#endif
bool shown_buffer_warning = false;
#ifdef LC_NO_HLSL_BUILTIN
namespace lc_hlsl {
struct HLSLCompressedHeader {
    void const *ptr{};
    size_t size{};
};
static HLSLCompressedHeader get_hlsl_builtin(luisa::string_view ss) { return {}; }
}// namespace lc_hlsl
#endif
namespace lc::hlsl {
#ifdef LUISA_ENABLE_IR
void glob_variables_with_grad(Function f, vstd::unordered_set<Variable> &gradient_variables) noexcept {
    if (f.requires_autodiff())
        traverse_expressions<true>(
            f.body(),
            [&](auto expr) noexcept {
                if (expr->tag() == Expression::Tag::CALL) {
                    if (auto call = static_cast<const CallExpr *>(expr);
                        call->op() == CallOp::GRADIENT ||
                        call->op() == CallOp::GRADIENT_MARKER ||
                        call->op() == CallOp::REQUIRES_GRADIENT) {
                        LUISA_ASSERT(!call->arguments().empty() &&
                                         call->arguments().front()->tag() == Expression::Tag::REF,
                                     "Invalid gradient function call.");
                        auto v = static_cast<const RefExpr *>(call->arguments().front())->variable();
                        gradient_variables.emplace(v);
                    }
                }
            },
            [](auto) noexcept {},
            [](auto) noexcept {});
}
#endif
vstd::string_view CodegenUtility::ReadInternalHLSLFile(vstd::string_view name) {
    auto data = lc_hlsl::get_hlsl_builtin(name);
    return {static_cast<char const *>(data.ptr), data.size};
}
vstd::MD5 CodegenUtility::GetTypeMD5(vstd::span<Type const *const> types) {
    vstd::vector<uint64_t> typeDescs;
    typeDescs.reserve(types.size());
    for (auto &&i : types) {
        if ((i->is_buffer() || i->is_texture()) && !i->member_attributes().empty())
            if (i->is_buffer())
                typeDescs.emplace_back(Type::buffer(i->element())->hash());
            else
                typeDescs.emplace_back(Type::texture(i->element(), i->dimension())->hash());
        else
            typeDescs.emplace_back(i->hash());
    }
    return {vstd::span<uint8_t const>(reinterpret_cast<uint8_t const *>(typeDescs.data()), luisa::size_bytes(typeDescs))};
}
vstd::MD5 CodegenUtility::GetTypeMD5(std::initializer_list<vstd::IRange<Variable> *> f) {
    vstd::vector<uint64_t> typeDescs;
    for (auto &&rg : f) {
        for (auto &&i : *rg) {
            auto type = i.type();
            if ((type->is_buffer() || type->is_texture()) && !type->member_attributes().empty())
                if (type->is_buffer())
                    typeDescs.emplace_back(Type::buffer(type->element())->hash());
                else
                    typeDescs.emplace_back(Type::texture(type->element(), type->dimension())->hash());
            else
                typeDescs.emplace_back(type->hash());
        }
    }
    return {vstd::span<uint8_t const>(reinterpret_cast<uint8_t const *>(typeDescs.data()), luisa::size_bytes(typeDescs))};
}
vstd::MD5 CodegenUtility::GetTypeMD5(Function func) {
    vstd::vector<uint64_t> typeDescs;
    auto args = func.arguments();
    typeDescs.reserve(args.size());
    for (auto &&i : args) {
        auto type = i.type();
        if ((type->is_buffer() || type->is_texture()) && !type->member_attributes().empty())
            if (type->is_buffer())
                typeDescs.emplace_back(Type::buffer(type->element())->hash());
            else
                typeDescs.emplace_back(Type::texture(type->element(), type->dimension())->hash());
        else
            typeDescs.emplace_back(type->hash());
    }
    return {vstd::span<uint8_t const>(reinterpret_cast<uint8_t const *>(typeDescs.data()), luisa::size_bytes(typeDescs))};
}

namespace detail {
size_t AddHeader(CallOpSet const &ops, vstd::StringBuilder &builder, bool isRaster, bool isWorkGraph, bool is_spirv, bool fallback, bool linalg) {
    builder << CodegenUtility::ReadInternalHLSLFile(fallback ? "hlsl_header_fallback" : "hlsl_header");
    size_t immutable_size = builder.size();
    if (ops.uses_raytracing()) {
        builder << CodegenUtility::ReadInternalHLSLFile("raytracing_header");
    }
    if (is_spirv) {
        builder << CodegenUtility::ReadInternalHLSLFile("spv_alias");
    }
    if (ops.test(CallOp::DETERMINANT)) {
        builder << CodegenUtility::ReadInternalHLSLFile("determinant");
    }
    if (ops.test(CallOp::INVERSE)) {
        builder << CodegenUtility::ReadInternalHLSLFile("inverse");
    }
    if (ops.test(CallOp::INDIRECT_SET_DISPATCH_KERNEL) || ops.test(CallOp::INDIRECT_SET_DISPATCH_COUNT)) {
        builder << CodegenUtility::ReadInternalHLSLFile("indirect");
    }
    if (ops.test(CallOp::BUFFER_SIZE) || ops.test(CallOp::TEXTURE_SIZE) || ops.test(CallOp::BYTE_BUFFER_SIZE)) {
        builder << CodegenUtility::ReadInternalHLSLFile("resource_size");
    }
    if (linalg || ops.uses_cooperative()) {
        if (!is_spirv) {
            builder << CodegenUtility::ReadInternalHLSLFile("dx_linalg");
        } else {
            LUISA_ERROR("Vulkan tensor not supported yet.");
        }
    }
    bool useBindless = false;
    for (auto i : vstd::range(
             luisa::to_underlying(CallOp::BINDLESS_TEXTURE2D_SAMPLE),
             luisa::to_underlying(CallOp::TYPED_BINDLESS_BUFFER_ADDRESS) + 1)) {
        if (ops.test(static_cast<CallOp>(i))) {
            useBindless = true;
            break;
        }
    }
    if (
        ops.test(CallOp::BINDLESS_COOPERATIVE_MUL_ADD) ||
        ops.test(CallOp::TYPED_BINDLESS_COOPERATIVE_MUL_ADD) ||
        ops.test(CallOp::BINDLESS_COOPERATIVE_MUL) ||
        ops.test(CallOp::TYPED_BINDLESS_COOPERATIVE_MUL)) {
        useBindless = true;
    }
    if (useBindless) {
        builder << CodegenUtility::ReadInternalHLSLFile("bindless_common");
    }
    if (ops.test(CallOp::RAY_TRACING_INSTANCE_TRANSFORM) ||
        ops.test(CallOp::RAY_TRACING_INSTANCE_USER_ID) ||
        ops.test(CallOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK) ||
        ops.test(CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM) ||
        ops.test(CallOp::RAY_TRACING_SET_INSTANCE_OPACITY) ||
        ops.test(CallOp::RAY_TRACING_SET_INSTANCE_USER_ID) ||
        ops.test(CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY)) {
        builder << CodegenUtility::ReadInternalHLSLFile("accel_header");
    }
    if (ops.test(CallOp::COPYSIGN)) {
        builder << CodegenUtility::ReadInternalHLSLFile("copy_sign");
    }
    if (!isRaster && (ops.test(CallOp::DDX) || ops.test(CallOp::DDY))) {
        builder << CodegenUtility::ReadInternalHLSLFile("compute_quad");
    }
    if (ops.uses_autodiff()) {
        builder << CodegenUtility::ReadInternalHLSLFile("auto_diff");
    }
    if (ops.test(CallOp::REDUCE_MAX) ||
        ops.test(CallOp::REDUCE_MIN) ||
        ops.test(CallOp::REDUCE_PRODUCT) ||
        ops.test(CallOp::REDUCE_SUM) ||
        ops.test(CallOp::OUTER_PRODUCT) ||
        ops.test(CallOp::MATRIX_COMPONENT_WISE_MULTIPLICATION)) {
        builder << CodegenUtility::ReadInternalHLSLFile("reduce");
    }
    if (isWorkGraph) {
        builder << CodegenUtility::ReadInternalHLSLFile("work_graph");
    }
    return immutable_size;
}
bool IsCBuffer(Variable::Tag t);
}// namespace detail

// Main compute kernel codegen
CodegenResult CodegenUtility::Codegen(Function kernel, luisa::string_view native_code, uint custom_mask, bool isSpirV, bool noRegister) {
    opt = CodegenStackData::Allocate(this);
    opt->isSpirv = isSpirV;
    opt->noRegister = noRegister;
    opt->atomicFloatToInt = isSpirV && kernel.propagated_builtin_callables().uses_atomic();
    auto disposeOpt = vstd::scope_exit([&] {
        CodegenStackData::DeAllocate(std::move(opt));
    });
    // CodegenStackData::ThreadLocalSpirv() = false;
    opt->kernel = kernel;
    bool nonEmptyCbuffer = IsCBufferNonEmpty(kernel);

    vstd::StringBuilder codegenData;
    vstd::StringBuilder varData;
    vstd::StringBuilder incrementalFunc;
    vstd::StringBuilder finalResult;
    opt->incrementalFunc = &incrementalFunc;
    finalResult.reserve(65500);
    uint64 immutableHeaderSize = detail::AddHeader(kernel.propagated_builtin_callables(), finalResult, false, false, isSpirV, noRegister, kernel.use_cooperative_operations());
    finalResult << native_code << "\n//"sv;
    finalResult << luisa::format("{}", custom_mask);
    finalResult << '\n';
    CodegenFunction(kernel, codegenData, nonEmptyCbuffer, true);

    opt->funcType = CodegenStackData::FuncType::Callable;
    auto argRange = vstd::make_ite_range(kernel.arguments()).i_range();
    uint bind_count = 2;
    if (nonEmptyCbuffer) {
        GenerateCBuffer({&argRange}, varData);
    }
    if (isSpirV) {
        if (opt->noRegister) {
            varData << R"(
struct _CBType{
uint4 v;
};
[[vk::push_constant]] ConstantBuffer<_CBType> dsp_c;
)"sv;
        } else {
            varData << R"(
struct _CBType{
uint4 v;
};
[[vk::push_constant]] ConstantBuffer<_CBType> dsp_c:register(b0);
)"sv;
        }
        bind_count += 2;
    } else {
        if (opt->noRegister) {
            varData << "uint4 dsp_c;\n"sv;
        } else {
            varData << "uint4 dsp_c:register(b0);\n"sv;
        }
        bind_count += 2;
    }
    CodegenResult::Properties properties;
    DXILRegisterIndexer dxilRegisters;
    SpirVRegisterIndexer spvRegisters;
    RegisterIndexer &indexer = isSpirV ? static_cast<RegisterIndexer &>(spvRegisters) : static_cast<RegisterIndexer &>(dxilRegisters);
    PreprocessCodegenProperties(properties, varData, indexer, nonEmptyCbuffer, false, isSpirV, bind_count);
    CodegenProperties(properties, varData, kernel, 0, indexer, bind_count);
    PostprocessCodegenProperties(finalResult, kernel.requires_autodiff());
    finalResult << varData << incrementalFunc << codegenData;
    if (!isSpirV) {
        // https://learn.microsoft.com/en-us/windows/win32/direct3d12/root-signature-limits
        if (bind_count >= 64) [[unlikely]] {
            LUISA_ERROR("Arguments binding size: {} exceeds 64 32-bit units not supported by hardware device. Try to use bindless instead.", bind_count);
        }
    }
    return {
        std::move(finalResult),
        std::move(opt->printer),
        std::move(properties),
        opt->useTex2DBindless,
        opt->useTex3DBindless,
        opt->useBufferBindless,
        immutableHeaderSize,
        GetTypeMD5(kernel)};
}

// Ray tracing pipeline codegen for motion blur
// Generates a lib_6_5 HLSL with raygen/miss/closesthit entry points
CodegenResult CodegenUtility::RayTracingCodegen(Function kernel, luisa::string_view native_code, uint custom_mask, bool isSpirV, bool noRegister) {
    opt = CodegenStackData::Allocate(this);
    opt->isSpirv = isSpirV;
    opt->noRegister = noRegister;
    opt->isRayTracing = true;
    opt->atomicFloatToInt = isSpirV && kernel.propagated_builtin_callables().uses_atomic();
    auto disposeOpt = vstd::scope_exit([&] {
        CodegenStackData::DeAllocate(std::move(opt));
    });
    opt->kernel = kernel;
    bool nonEmptyCbuffer = IsCBufferNonEmpty(kernel);

    vstd::StringBuilder codegenData;
    vstd::StringBuilder varData;
    vstd::StringBuilder incrementalFunc;
    vstd::StringBuilder finalResult;
    opt->incrementalFunc = &incrementalFunc;
    finalResult.reserve(65500);
    uint64 immutableHeaderSize = detail::AddHeader(kernel.propagated_builtin_callables(), finalResult, false, false, isSpirV, noRegister, kernel.use_cooperative_operations());
    // Add motion blur ray tracing header (miss/closesthit entry points + _TraceClosestMotion)
    finalResult << ReadInternalHLSLFile("raytracing_motion_header");
    finalResult << native_code << "\n//"sv;
    finalResult << luisa::format("{}", custom_mask);
    finalResult << '\n';
    CodegenFunction(kernel, codegenData, nonEmptyCbuffer, true);

    opt->funcType = CodegenStackData::FuncType::Callable;
    auto argRange = vstd::make_ite_range(kernel.arguments()).i_range();
    uint bind_count = 2;
    if (nonEmptyCbuffer) {
        GenerateCBuffer({&argRange}, varData);
    }
    // For ray tracing pipeline, we use push constants for dispatch dimensions
    if (isSpirV) {
        if (opt->noRegister) {
            varData << R"(
struct _CBType{
uint4 v;
};
[[vk::push_constant]] ConstantBuffer<_CBType> dsp_c;
)"sv;
        } else {
            varData << R"(
struct _CBType{
uint4 v;
};
[[vk::push_constant]] ConstantBuffer<_CBType> dsp_c:register(b0);
)"sv;
        }
        bind_count += 2;
    } else {
        if (opt->noRegister) {
            varData << "uint4 dsp_c;\n"sv;
        } else {
            varData << "uint4 dsp_c:register(b0);\n"sv;
        }
        bind_count += 2;
    }
    CodegenResult::Properties properties;
    DXILRegisterIndexer dxilRegisters;
    SpirVRegisterIndexer spvRegisters;
    RegisterIndexer &indexer = isSpirV ? static_cast<RegisterIndexer &>(spvRegisters) : static_cast<RegisterIndexer &>(dxilRegisters);
    PreprocessCodegenProperties(properties, varData, indexer, nonEmptyCbuffer, false, isSpirV, bind_count);
    CodegenProperties(properties, varData, kernel, 0, indexer, bind_count);
    PostprocessCodegenProperties(finalResult, kernel.requires_autodiff());
    finalResult << varData << incrementalFunc << codegenData;

    return {
        std::move(finalResult),
        std::move(opt->printer),
        std::move(properties),
        opt->useTex2DBindless,
        opt->useTex3DBindless,
        opt->useBufferBindless,
        immutableHeaderSize,
        GetTypeMD5(kernel)};
}

// Main rasterization pipeline codegen
CodegenResult CodegenUtility::RasterCodegen(
    Function vertFunc,
    Function pixelFunc,
    luisa::string_view native_code,
    uint custom_mask,
    bool isSpirV,
    bool noRegister) {
    opt = CodegenStackData::Allocate(this);
    opt->isSpirv = isSpirV;
    // CodegenStackData::ThreadLocalSpirv() = false;
    opt->kernel = vertFunc;
    opt->noRegister = noRegister;
    opt->isRaster = true;
    opt->atomicFloatToInt = isSpirV && (vertFunc.propagated_builtin_callables().uses_atomic() || pixelFunc.propagated_builtin_callables().uses_atomic());
    auto disposeOpt = vstd::scope_exit([&] {
        opt->isRaster = false;
        CodegenStackData::DeAllocate(std::move(opt));
    });
    vstd::StringBuilder codegenData;
    vstd::StringBuilder varData;
    vstd::StringBuilder finalResult;
    vstd::StringBuilder incrementalFunc;
    opt->incrementalFunc = &incrementalFunc;
    finalResult.reserve(65500);
    auto opSet = vertFunc.propagated_builtin_callables();
    opSet.propagate(pixelFunc.propagated_builtin_callables());
    uint64 immutableHeaderSize = detail::AddHeader(opSet, finalResult, true, false, isSpirV, noRegister, vertFunc.use_cooperative_operations() || pixelFunc.use_cooperative_operations());
    finalResult << native_code << "\n//"sv;
    finalResult << luisa::format("{}", custom_mask);
    finalResult << '\n';
    // Vertex
    codegenData << "struct v2p{\n"sv;
    auto v2pType = vertFunc.return_type();
    if (v2pType->is_structure()) {
        opt->internalStruct.emplace(v2pType, "v2p");
        if (v2pType->members().size() != v2pType->member_attributes().size()) [[unlikely]] {
            LUISA_ERROR("Vertex-to-pixel structure's attribute size is illegal.");
        }
        size_t memberIdx = 0;
        bool pos = false;
        for (auto &&i : v2pType->members()) {
            bool is_sv_pos = v2pType->member_attributes()[memberIdx].key == "position"sv;
            if (!is_sv_pos && opt->isSpirv) {
                codegenData << luisa::format("[[vk::location({})]] ", memberIdx - 1);
            }
            GetTypeName(*i, codegenData, Usage::READ);
            codegenData << " v"sv << vstd::to_string(memberIdx);
            if (is_sv_pos) {
                if (pos) [[unlikely]] {
                    LUISA_ERROR("Vertex-to-pixel structure can only have one position.");
                }
                codegenData << ":SV_POSITION;\n"sv;
                pos = true;
                if (!i->is_vector() || i->dimension() != 4) [[unlikely]] {
                    LUISA_ERROR("Position must be float4.");
                }
            } else {
                codegenData << ":TEXCOORD"sv << vstd::to_string(memberIdx - 1) << ";\n"sv;
            }
            ++memberIdx;
        }
        if (!pos) [[unlikely]] {
            LUISA_ERROR("Vertex-to-pixel structure should contained position.");
        }
    } else {
        LUISA_ERROR("Illegal vertex return type!");
    }
    uint bind_count = 2;
    if (isSpirV) {
        codegenData << R"(};
struct _CBType{
uint v;
};
[[vk::push_constant]] ConstantBuffer<_CBType> obj_id:register(b0);
)"sv;
        bind_count += 2;
    } else {
        if (opt->noRegister) {
            codegenData << R"(};
uint obj_id;
)"sv;
        } else {
            codegenData << R"(};
uint obj_id:register(b0);
)"sv;
        }
        bind_count += 2;
    }
    codegenData << "#ifdef VS\n";
    auto vert_args = vertFunc.arguments();
    if (vert_args.empty()) [[unlikely]] {
        LUISA_ERROR("Vertex arguments illegal.");
    }
    auto appdataType = vert_args[0].type();
    if (appdataType->is_structure()) {
        auto appdataAttris = appdataType->member_attributes();
        auto appdataMems = appdataType->members();
        if (appdataAttris.size() != appdataMems.size()) [[unlikely]] {
            LUISA_ERROR("Mesh-to-vertex structure must have attributes.");
        }
        opt->internalStruct.try_emplace(appdataType, "_mesh");
        codegenData << "struct _mesh{\n"sv;
        for (int64_t i : vstd::range(static_cast<int64_t>(appdataAttris.size()))) {
            auto member = appdataMems[static_cast<size_t>(i)];
            auto &attr = appdataAttris[static_cast<size_t>(i)];
            if (attr.key.empty()) [[unlikely]] {
                LUISA_ERROR("Mesh-to-vertex structure member {} miss attributes.", i);
            }
            if (!(member->is_scalar() || member->is_vector())) [[unlikely]] {
                LUISA_ERROR("Mesh-to-vertex structure do not support type {}", member->description());
            }

            auto iter = attributes.find(attr.key);
            if (iter == attributes.end()) [[unlikely]] {
                LUISA_ERROR("Invalid attribute: {}", attr.key);
            }

            if (iter->second.second && iter->second.second != member) [[unlikely]] {
                LUISA_ERROR("Attribute {} type {} mismatch with {}", attr.key, iter->second.second->description(), member->description());
            }
            if (opt->isSpirv) {
                codegenData << luisa::format("[[vk::location({})]] ", i);
            }
            GetTypeName(*member, codegenData, Usage::READ);
            codegenData
                << " v"sv << vstd::to_string(i) << ':'
                << iter->second.first
                << ";\n"sv;
        }
        codegenData << "};\n";
    } else {
        LUISA_ERROR("Mesh-to-vertex must be a structure");
    }

    auto vertRange = vstd::make_ite_range(vert_args.subspan(1)).i_range();
    auto pixelRange = vstd::make_ite_range(pixelFunc.arguments().subspan(1)).i_range();
    std::initializer_list<vstd::IRange<Variable> *> funcs = {&vertRange, &pixelRange};

    bool nonEmptyCbuffer = IsCBufferNonEmpty(funcs);
    opt->appdataId = vert_args[0].uid();
    CodegenVertex(vertFunc, codegenData, nonEmptyCbuffer);
    opt->appdataId = -1;
    // TODO: gen vertex data
    codegenData << "#elif defined(PS)\n"sv;
    size_t vert_arg_offset = 0;
    for (auto &i : vert_args.subspan(1)) {
        if (detail::IsCBuffer(i.tag())) {
            vert_arg_offset += 1;
        }
    }
    opt->argOffset = vert_arg_offset;
    // TODO: gen pixel data
    CodegenPixel(pixelFunc, codegenData, nonEmptyCbuffer);
    codegenData << "#endif\n"sv;

    opt->funcType = CodegenStackData::FuncType::Callable;
    if (nonEmptyCbuffer) {
        GenerateCBuffer(funcs, varData);
    }
    CodegenResult::Properties properties;
    DXILRegisterIndexer dxilRegisters;
    SpirVRegisterIndexer spvRegisters;
    RegisterIndexer &indexer = isSpirV ? static_cast<RegisterIndexer &>(spvRegisters) : static_cast<RegisterIndexer &>(dxilRegisters);
    PreprocessCodegenProperties(properties, varData, indexer, nonEmptyCbuffer, true, isSpirV, bind_count);
    CodegenProperties(properties, varData, vertFunc, 1, indexer, bind_count);
    CodegenProperties(properties, varData, pixelFunc, 1, indexer, bind_count);
    PostprocessCodegenProperties(finalResult, false);
    finalResult << varData << incrementalFunc << codegenData;
    // https://learn.microsoft.com/en-us/windows/win32/direct3d12/root-signature-limits
    if (bind_count >= 64) [[unlikely]] {
        LUISA_ERROR("Arguments binding size: {} exceeds 64 32-bit units not supported by hardware device. Try to use bindless instead.", bind_count);
    }
    return {
        std::move(finalResult),
        std::move(opt->printer),
        std::move(properties),
        opt->useTex2DBindless,
        opt->useTex3DBindless,
        opt->useBufferBindless,
        immutableHeaderSize,
        GetTypeMD5(funcs)};
}
// Convenience overload for callers (e.g. tests) that don't need the captured binding data.
CodegenResult CodegenUtility::WorkGraphCodegen(
    const WorkGraph &work_graph,
    luisa::string_view native_code,
    uint custom_mask,
    bool noRegister) {
    CodegenResult::Properties properties;
    vstd::unordered_map<uint64_t, uint32_t> uid_map;
    uint bind_count = 0;
    auto captured = CollectWorkGraphBindings(work_graph, properties, uid_map, bind_count);
    return WorkGraphCodegen(work_graph, native_code, custom_mask, captured, std::move(properties), std::move(uid_map), bind_count, noRegister);
}

CodegenResult CodegenUtility::WorkGraphCodegen(
    const WorkGraph &work_graph,
    luisa::string_view native_code,
    uint custom_mask,
    vstd::span<const WorkGraphCapturedBinding> captured,
    CodegenResult::Properties properties,
    vstd::unordered_map<uint64_t, uint32_t> handle_to_canonical_uid,
    uint bind_count,
    bool noRegister) {
    opt = CodegenStackData::Allocate(this);
    opt->noRegister = noRegister;
    opt->isWorkGraph = true;
    auto disposeOpt = vstd::scope_exit([&] {
        opt->isWorkGraph = false;
        CodegenStackData::DeAllocate(std::move(opt));
    });
    vstd::StringBuilder codegenData;
    vstd::StringBuilder varData;
    vstd::StringBuilder finalResult;
    vstd::StringBuilder incrementalFunc;
    opt->incrementalFunc = &incrementalFunc;
    finalResult.reserve(65500);
    CallOpSet opSet{};
    auto linalg = false;
    for (const auto &node : work_graph.nodes()) {
        linalg |= node.fn_builder->use_cooperative_operations();
        opSet.propagate(node.fn_builder->propagated_builtin_callables());
    }
    uint64 immutableHeaderSize = detail::AddHeader(opSet, finalResult, false, true, false, noRegister, linalg);

    finalResult << native_code << "\n//"sv;
    static_cast<void>(vstd::to_string(custom_mask));
    finalResult << '\n';

    vstd::unordered_set<uint64_t> globalCallableMap;
    const auto &nodes = work_graph.nodes();

    for (const auto& node : nodes) {
        if (node.input_record_has_dispatch_grid) {
            opt->dispatch_grid_records.emplace(node.input_record_type);
        }
    }

    if (bind_count >= 64) [[unlikely]] {
        LUISA_ERROR(
            "Arguments binding size: {} exceeds 64 32-bit units not supported by hardware device."
            "Try to use bindless instead.",
            bind_count);
    }

    // Emit one global variable declaration per captured binding (in UID/first-encounter order)
    for (size_t i = 0; i < captured.size(); i++) {
        auto &c = captured[i];
        auto &prop = properties[i];
        GetTypeName(*c.type, varData, c.usage);
        varData << ' ';
        varData << (c.argument.tag == Argument::Tag::BUFFER ? "_b"sv : "_t"sv);
        vstd::to_string(i, varData);
        if (!opt->noRegister) {
            bool is_uav = (prop.type == ShaderVariableType::RWStructuredBuffer ||
                           prop.type == ShaderVariableType::UAVTextureHeap ||
                           prop.type == ShaderVariableType::UAVBufferHeap);
            varData << " : register("sv << (is_uav ? 'u' : 't');
            vstd::to_string(prop.register_index, varData);
            varData << ");\n"sv;
        } else {
            varData << ";\n"sv;
        }
    }

    for (size_t i = 0; i < nodes.size(); ++i) {
        // Map each node's variable UIDs to the canonical UIDs from CollectWorkGraphBindings
        opt->uid_remap.clear();
        auto func = nodes[i].fn_builder->function();
        auto args = func.arguments();
        auto bindings = func.bound_arguments();
        for (size_t j = 0; j < bindings.size(); j++) {
            luisa::visit([&]<typename T>(T const &binding) noexcept {
                if constexpr (std::is_same_v<T, Function::BufferBinding> ||
                              std::is_same_v<T, Function::TextureBinding>) {
                    auto it = handle_to_canonical_uid.find(binding.handle);
                    LUISA_ASSERT(it != handle_to_canonical_uid.end(), "all bindings should be canonicalized");
                    opt->uid_remap[args[j].uid()] = it->second;
                }
            }, bindings[j]);
        }

        CodegenWorkGraphNode(work_graph, i, codegenData, globalCallableMap, handle_to_canonical_uid);
    }
    opt->uid_remap.clear();

    // Post-process properties (generates struct definitions)
    PostprocessCodegenProperties(finalResult, false);
    finalResult << varData << codegenData;

    vstd::vector<Type const *> recordTypes;
    recordTypes.reserve(nodes.size());
    for (auto &&node : nodes) {
        if (node.input_record_type != nullptr) {
            recordTypes.push_back(node.input_record_type);
        }
    }

    return CodegenResult(
        std::move(finalResult),
        std::move(opt->printer),
        std::move(properties),
        opt->useTex2DBindless,
        opt->useTex3DBindless,
        opt->useBufferBindless,
        immutableHeaderSize,
        GetTypeMD5(recordTypes));
}

void CodegenUtility::CodegenFunction(Function func, vstd::StringBuilder &result, bool cbufferNonEmpty, bool codegen_self) {
    auto codegenOneFunc = [&](Function func) {
        auto constants = func.constants();
        for (auto &&i : constants) {
            vstd::StringBuilder constValueName;
            if (!GetConstName(i.hash(), i, constValueName)) continue;
            result << "static const "sv;
            GetTypeName(*i.type(), result, Usage::READ);
            result << ' ' << constValueName << " = "sv;
            CodegenConstantPrinter printer{*this, result};
            i.decode(printer);
            result << ";\n"sv;
        }
#ifdef LUISA_ENABLE_IR
        vstd::unordered_set<Variable> grad_vars;
        glob_variables_with_grad(func, grad_vars);
#endif
        if (func.tag() == Function::Tag::KERNEL) {
            opt->funcType = CodegenStackData::FuncType::Kernel;
            if (opt->isRayTracing) {
                // Ray tracing pipeline: generate raygen entry point
                result << R"([shader("raygeneration")]
void main_raygen(){
uint3 dspId = DispatchRaysIndex();
uint3 thdId = uint3(0,0,0);
uint3 grpId = uint3(0,0,0);
)"sv;
                auto blockSize = func.block_size();
                auto dsp_c = opt->isSpirv ? "dsp_c.v"sv : "dsp_c"sv;
                // Bounds check using DispatchRaysDimensions
                result << "if(any(dspId >= "sv << dsp_c << ")) return;\n"sv;
            } else {
                // Compute shader: generate standard entry point
                auto warp_size = func.allowed_warp_size();
                if (warp_size.has_value()) {
                    result << luisa::format("[WaveSize({})]\n", int(warp_size.value()));
                }
                result << "[numthreads("
                       << vstd::to_string(func.block_size().x)
                       << ','
                       << vstd::to_string(func.block_size().y)
                       << ','
                       << vstd::to_string(func.block_size().z)
                       << R"()]
void main(uint3 thdId:SV_GroupThreadId,uint3 dspId:SV_DispatchThreadID,uint3 grpId:SV_GroupId){
)"sv;
                auto blockSize = func.block_size();
                vstd::fixed_vector<char, 3> swizzle;
                if (blockSize.x > 1) {
                    swizzle.emplace_back('x');
                }
                if (blockSize.y > 1) {
                    swizzle.emplace_back('y');
                }
                if (blockSize.z > 1) {
                    swizzle.emplace_back('z');
                }
                if (!swizzle.empty()) {
                    auto dsp_c = opt->isSpirv ? "dsp_c.v"sv : "dsp_c"sv;
                    if (swizzle.size() == 1) {
                        result << "if(dspId."sv << swizzle[0] << ">="sv << dsp_c << "."sv << swizzle[0] << ") return;\n"sv;
                    } else {
                        vstd::string_view strv(swizzle.data(), swizzle.size());
                        result << "if(any(dspId."sv << strv << ">="sv << dsp_c << "."sv << strv << ")) return;\n"sv;
                    }
                }
            }
            if (cbufferNonEmpty) {
                result << "_Args a = _Global[0];\n"sv;
            }
            opt->arguments.clear();
            opt->arguments.reserve(func.arguments().size());
            size_t idx = 0;
            for (auto &&i : func.arguments()) {
                opt->arguments.try_emplace(i.uid(), idx);
                ++idx;
            }
        } else {
            opt->funcType = CodegenStackData::FuncType::Callable;
            GetFunctionDecl(func, result);
            result << "{\n"sv;
        }
        {

            StringStateVisitor vis(func, result, this);
            vis.sharedVariables = &opt->sharedVariable;
            vis.VisitFunction(
#ifdef LUISA_ENABLE_IR
                grad_vars,
#endif
                func);
        }
        result << "}\n"sv;
    };
    vstd::unordered_set<uint64_t> callableMap;
    auto callable = [&](auto &&callable, Function func) -> void {
        for (auto &&i : func.custom_callables()) {
            if (callableMap.emplace(i->hash()).second) {
                callable(callable, i->function());
            }
        }
        codegenOneFunc(func);
    };
    if (codegen_self)
        callable(callable, func);
    else {
        for (auto &&i : func.custom_callables()) {
            if (callableMap.emplace(i->hash()).second) {
                callable(callable, i->function());
            }
        }
    }
}
}// namespace lc::hlsl
