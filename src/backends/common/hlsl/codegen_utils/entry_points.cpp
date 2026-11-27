// Main Codegen Entry Points

#include "../hlsl_codegen.h"
#include "../codegen_stack_data.h"
#include "../register_indexer.h"
#include "../constant_printer.h"
#include "../register_indexer.h"

#include <algorithm>
#include <luisa/dsl/raster/raster_interpolation.h>

#ifndef LC_NO_HLSL_BUILTIN
#include "../builtin/hlsl_builtin.hpp"
#endif
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
[[nodiscard]] bool uses_accel_intrinsics(CallOpSet const &ops) noexcept {
    return ops.test(CallOp::RAY_TRACING_INSTANCE_TRANSFORM) ||
           ops.test(CallOp::RAY_TRACING_INSTANCE_USER_ID) ||
           ops.test(CallOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK) ||
           ops.test(CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM) ||
           ops.test(CallOp::RAY_TRACING_SET_INSTANCE_OPACITY) ||
           ops.test(CallOp::RAY_TRACING_SET_INSTANCE_USER_ID) ||
           ops.test(CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY) ||
           ops.test(CallOp::RAY_TRACING_INSTANCE_MOTION_MATRIX) ||
           ops.test(CallOp::RAY_TRACING_INSTANCE_MOTION_SRT) ||
           ops.test(CallOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX) ||
           ops.test(CallOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT);
}
size_t AddHeader(CallOpSet const &ops, vstd::StringBuilder &builder, bool isRaster, bool is_spirv, bool fallback, bool linalg, bool fallback_rtx, bool bf16_fallback, SpirvShaderCapabilities spirv_capabilities) {
    // The capability switches have to be defined before the header that tests them
    // with #ifdef (builtin/hlsl_header*.bytes).  Each one selects an instruction that
    // declares an OpCapability, and a module that carries that capability is rejected by
    // a device whose logical device did not enable the extension - so a macro is only
    // defined for a device that negotiated it, and a device that negotiated neither gets
    // the pre-feature header byte for byte.
    if (is_spirv) {
        if (spirv_capabilities.fma) {
            // VK_KHR_shader_fma: `mad()` on 32-bit float becomes OpFmaKHR.
            builder << "#define LUISA_SPV_FMA 1\n";
        }
        if (spirv_capabilities.abort) {
            // VK_KHR_shader_abort: `$debug_break` becomes OpAbortKHR.
            builder << "#define LUISA_SPV_ABORT 1\n";
        }
        if (spirv_capabilities.split_barrier) {
            // VK_EXT_shader_split_barrier: the split-barrier arrive/wait
            // become OpControlBarrierArriveEXT/WaitEXT (compute only).
            builder << "#define LUISA_SPV_SPLIT_BARRIER 1\n";
        }
        if (spirv_capabilities.cluster_acceleration_structure) {
            // VK_NV_cluster_acceleration_structure: the ray-query cluster-ID
            // accessor becomes OpRayQueryGetIntersectionClusterIdNV.  The accessor
            // is only declared when the device negotiated the extension, so a
            // shader that reads a cluster ID on a device without it fails to
            // compile instead of emitting a module that declares an extension the
            // logical device never enabled.
            builder << "#define LUISA_SPV_CLUSTER_AS 1\n";
        }
    }
    builder << CodegenUtility::ReadInternalHLSLFile(fallback ? "hlsl_header_fallback" : "hlsl_header");
    if (bf16_fallback) {
        // Software BF16: the host selected this header because the target
        // cannot compile the native HLSL `bfloat16_t` type (see
        // builtin/bfloat16_fallback.bytes).  A device with the native type
        // never includes it.
        builder << CodegenUtility::ReadInternalHLSLFile("bfloat16_fallback");
    }
    if (is_spirv) {
        // Vulkan versions typed bindless descriptors per slot. DXIL keeps its
        // established contiguous base+slot ABI.
        builder << "#define LUISA_SPIRV_TYPED_BINDLESS_INDIRECT 1\n";
        // DX root UAVs do not carry a view size for capacity queries.
        builder << "#define LUISA_INDIRECT_DISPATCH_BOUNDS_CHECK 1\n";
        builder << CodegenUtility::ReadInternalHLSLFile("spv_alias");
    }
    if (is_spirv && ops.test(CallOp::ASYNC_COPY)) {
        builder << R"(
// --- Vulkan async-copy builtins ---
// Workgroup scratch buffer for async copies (byte-addressed).
// The async_copy src/dst offsets index into this buffer (dst)
// and the first StructuredBuffer argument (src).
groupshared uint _vk_wg_copy_buf[4096];
)";
    }
    size_t immutable_size = builder.size();
    // Software ray tracing: the traversal of builtin/fallback_rtx_header.bytes
    // replaces both the inline-query header (there is no `RayQuery` and no
    // `RaytracingAccelerationStructure` on a device without hardware ray tracing)
    // and the instance-accessor header.  It is *not* appended here: the traversal
    // resolves the regions of its acceleration structure through the device's
    // global bindless heap (`bdls`, declared by GenerateBindless), which is part
    // of the per-shader binding block, so `Codegen` appends it there - right after
    // the declarations and before the generated body.  `bindless_common` is then
    // not added below either.
    auto fallback_traversal =
        fallback_rtx && (ops.uses_raytracing() || uses_accel_intrinsics(ops));
    if (!fallback_rtx && ops.uses_raytracing()) {
        builder << CodegenUtility::ReadInternalHLSLFile("raytracing_header");
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
    if (ops.test(CallOp::BUFFER_SIZE) || ops.test(CallOp::TEXTURE_SIZE) ||
        ops.test(CallOp::BYTE_BUFFER_SIZE) ||
        ops.test(CallOp::ACCEL_SIZE)) {
        builder << CodegenUtility::ReadInternalHLSLFile("resource_size");
    }
    if (linalg || ops.uses_cooperative()) {
        builder << CodegenUtility::ReadInternalHLSLFile(is_spirv ? "vk_linalg" : "dx_linalg");
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
        ops.test(CallOp::TYPED_BINDLESS_COOPERATIVE_MUL) ||
        ops.test(CallOp::BINDLESS_COOPERATIVE_VECTOR_LOAD) ||
        ops.test(CallOp::TYPED_BINDLESS_COOPERATIVE_VECTOR_LOAD) ||
        ops.test(CallOp::BINDLESS_COOPERATIVE_VECTOR_STORE) ||
        ops.test(CallOp::TYPED_BINDLESS_COOPERATIVE_VECTOR_STORE)) {
        useBindless = true;
    }
    if (useBindless && !fallback_traversal) {
        builder << CodegenUtility::ReadInternalHLSLFile("bindless_common");
    }
    if (!fallback_rtx && uses_accel_intrinsics(ops)) {
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
    return immutable_size;
}
bool IsCBuffer(Variable::Tag t);
}// namespace detail

// Main compute kernel codegen
CodegenResult CodegenUtility::Codegen(Function kernel, luisa::string_view native_code, uint custom_mask, bool isSpirV, bool noRegister, bool enable_debug_info, bool enable_fast_math, bool fallback_rtx, bool bf16_fallback, SpirvShaderCapabilities spirv_capabilities, bool emulate_8bit, luisa::optional<uint8_t> default_warp_size) {
    opt = CodegenStackData::Allocate(this);
    opt->isSpirv = isSpirV;
    opt->fallback_rtx = fallback_rtx;
    opt->bf16_fallback = bf16_fallback;
    opt->emulate_8bit = emulate_8bit && !isSpirV;
    opt->default_warp_size = isSpirV ? luisa::nullopt : default_warp_size;
    opt->spirv_capabilities = spirv_capabilities;
    opt->noRegister = noRegister;
    opt->enable_debug_info = enable_debug_info;
    opt->enable_fast_math = enable_fast_math;
    auto fallback_traversal = false;
    if (fallback_rtx) {
        // A software (fallback) acceleration structure reaches the shader as a
        // *bindless heap* - the traversal resolves the region of a referenced
        // BLAS by its slot (fallback_rtx_layout.h) - so a shader that traces or
        // touches instances declares the device's global bindless heap
        // (`bdls`, `GenerateBindless`).  A hardware-path shader never sets this,
        // so its generated source, its hash and its cached artifact are
        // unchanged.
        auto builtin_ops = kernel.propagated_builtin_callables();
        fallback_traversal =
            builtin_ops.uses_raytracing() || detail::uses_accel_intrinsics(builtin_ops);
        if (fallback_traversal) {
            opt->useBufferBindless = true;
        }
    }
    auto disposeOpt = vstd::scope_exit([&] {
        CodegenStackData::DeAllocate(std::move(opt));
    });
    // CodegenStackData::ThreadLocalSpirv() = false;
    opt->kernel = kernel;
    bool nonEmptyCbuffer = IsCBufferNonEmpty(kernel);
    // The generated HLSL is assembled in the following order:
    //   1. finalResult: prelude (debug macro, builtin headers, native code,
    //                  custom mask comment) and post-processed custom structs
    //   2. varData: cbuffer, push constants, and resource bindings
    //   3. incrementalFunc: helper functions generated on demand for types /
    //                       access chains
    //   4. codegenData: entry point functions (kernel / vertex / pixel) and
    //                   callables
    // Final concatenation: finalResult << varData << incrementalFunc << codegenData;
    vstd::StringBuilder codegenData;
    vstd::StringBuilder varData;
    vstd::StringBuilder incrementalFunc;
    vstd::StringBuilder finalResult;
    opt->incrementalFunc = &incrementalFunc;
    finalResult.reserve(65500);
    if (enable_debug_info) {
        finalResult << "#define LUISA_DEBUG_INFO 1\n";
    }
    uint64 immutableHeaderSize = detail::AddHeader(kernel.propagated_builtin_callables(), finalResult, false, isSpirV, noRegister, kernel.use_cooperative_operations(), fallback_rtx, bf16_fallback, spirv_capabilities);
    finalResult << native_code << "\n//"sv;
    finalResult << luisa::format("{}", custom_mask);
    finalResult << '\n';

    // Generate cbuffer FIRST to populate validate_index_map before codegen
    opt->funcType = CodegenStackData::FuncType::Callable;
    auto argRange = vstd::make_ite_range(kernel.arguments()).i_range();
    uint bind_count = 2;
    uint validation_count = 0;
    if (nonEmptyCbuffer || enable_debug_info) {
        uint64_t func_hash = kernel.hash();
        uint64_t hashes[] = {func_hash};
        GenerateCBuffer({&argRange}, varData, enable_debug_info, validation_count, hashes);
    }
    // Promote unique kernel buffer/bindless-array formal parameters directly to
    // the kernel's global declarations before emitting callables and main.
    AnalyzePromotedResourceArgs(kernel);
    // Generate the compute-shader entry point and every callable it depends on.
    // CodegenFunction emits literal constants, the HLSL entry function signature
    // ([numthreads], dispatch-bounds check against dsp_c, thread-id loading),
    // the _Global args load when a cbuffer is present, and the kernel/callable
    // bodies through StringStateVisitor. Custom callables are emitted recursively
    // so the resulting HLSL is self-contained.
    CodegenFunction(kernel, codegenData, nonEmptyCbuffer || enable_debug_info, true);
    if (isSpirV) {
        // Software (fallback) ray tracing: the acceleration view of a fallback
        // `accel` argument addresses the shared acceleration buffer with absolute
        // uint4 handles, so the traversal also needs the absolute uint4 offset of
        // that tree's region ("accelBase", fallback_rtx_header.bytes).  It travels
        // in the push-constant block, right after the fixed dispatch block the
        // command encoder writes first (`IndirectDispatchPushConstants`, 32 bytes):
        // one 32-bit word per `accel` argument, in kernel-argument order - the same
        // offset and order `Device::_create_shader_hlsl` sizes the block for and the
        // command encoder writes them to (stream.cpp).  A device whose fallback is
        // off adds nothing, so its push-constant block stays exactly as it was.
        auto fallback_accel_base_count = 0u;
        if (opt->fallback_rtx) {
            for (auto &&arg : kernel.arguments()) {
                fallback_accel_base_count += arg.type()->is_accel() ? 1u : 0u;
            }
        }
        auto emit_fallback_accel_bases = [&](luisa::string_view prefix) {
            for (auto &&arg : kernel.arguments()) {
                if (!arg.type()->is_accel()) { continue; }
                varData << prefix;
                GetVariableName(kernel, arg, varData);
                varData << "Base;\n"sv;
            }
        };
        if (opt->noRegister) {
            varData << R"(
	struct _CBType{
	uint4 v;
)"sv;
            if (fallback_accel_base_count != 0u) {
                varData << "	uint4 _lc_fallback_dispatch_tail;\n"sv;
                emit_fallback_accel_bases("	uint "sv);
            }
            varData << R"(	};
	[[vk::push_constant]] ConstantBuffer<_CBType> dsp_c;
	)"sv;
        } else {
            varData << R"(
struct _CBType{
uint4 v;
)"sv;
            if (fallback_accel_base_count != 0u) {
                varData << "uint4 _lc_fallback_dispatch_tail;\n"sv;
                emit_fallback_accel_bases("uint "sv);
            }
            varData << R"(};
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
    // Build the runtime property table and emit all resource/register bindings.
    // PreprocessCodegenProperties reserves fixed slots for the push-constant
    // block, the sampler heap, the argument cbuffer (if any), and bindless
    // arrays. CodegenProperties then walks the kernel arguments to declare
    // buffers, textures, bindless arrays, acceleration structures, etc., and
    // records their ShaderVariableType/register indices. Postprocess flushes
    // generated custom struct definitions and groupshared variables. The four
    // pieces are then concatenated in the final shader order: prelude,
    // bindings, incremental helpers, entry functions/callables.
    CodegenResult::Properties properties;
    DXILRegisterIndexer dxilRegisters;
    SpirVRegisterIndexer spvRegisters;
    RegisterIndexer &indexer = isSpirV ? static_cast<RegisterIndexer &>(spvRegisters) : static_cast<RegisterIndexer &>(dxilRegisters);
    PreprocessCodegenProperties(properties, varData, indexer, nonEmptyCbuffer || enable_debug_info, false, isSpirV, bind_count);
    CodegenProperties(properties, varData, kernel, 0, indexer, bind_count);
    PostprocessCodegenProperties(finalResult, kernel.requires_autodiff());
    if (fallback_traversal) {
        // The software (fallback) traversal is appended to the binding block, not
        // to the immutable header: it resolves the regions of its acceleration
        // structure through the device's global bindless heap (`bdls`), which is
        // declared by `GenerateBindless` above, and it has to precede the
        // generated body that calls into it.  `bindless_common` is added here for
        // the same reason (and `AddHeader` then leaves it out).
        varData << CodegenUtility::ReadInternalHLSLFile("bindless_common");
        varData << CodegenUtility::ReadInternalHLSLFile("fallback_rtx_header");
    }
    finalResult << varData << incrementalFunc << codegenData;
    CodegenResult result{
            std::move(finalResult),
            std::move(opt->printer),
            std::move(properties),
            opt->useTex2DBindless,
            opt->useTex3DBindless,
            opt->useBufferBindless,
            opt->use_8bit,
            validation_count,
            immutableHeaderSize,
            GetTypeMD5(kernel)};
    result.use_atomic64 = opt->use_atomic64;
    return result;
}

// Ray tracing pipeline codegen for motion blur
// Generates a lib_6_5 HLSL with raygen/miss/closesthit entry points
CodegenResult CodegenUtility::RayTracingCodegen(Function kernel, luisa::string_view native_code, uint custom_mask, bool isSpirV, bool noRegister, bool enable_debug_info, bool enable_fast_math, bool fallback_rtx, bool bf16_fallback, SpirvShaderCapabilities spirv_capabilities, bool emulate_8bit) {
    opt = CodegenStackData::Allocate(this);
    opt->isSpirv = isSpirV;
    opt->fallback_rtx = fallback_rtx;
    opt->bf16_fallback = bf16_fallback;
    opt->emulate_8bit = emulate_8bit && !isSpirV;
    opt->spirv_capabilities = spirv_capabilities;
    opt->noRegister = noRegister;
    opt->isRayTracing = true;
    opt->enable_debug_info = enable_debug_info;
    opt->enable_fast_math = enable_fast_math;
    auto disposeOpt = vstd::scope_exit([&] {
        CodegenStackData::DeAllocate(std::move(opt));
    });
    opt->kernel = kernel;
    bool nonEmptyCbuffer = IsCBufferNonEmpty(kernel);

    vstd::StringBuilder codegenData;
    vstd::StringBuilder varData;
    vstd::StringBuilder incrementalFunc;
    opt->incrementalFunc = &incrementalFunc;
    vstd::StringBuilder finalResult;
    finalResult.reserve(65500);
    if (enable_debug_info) {
        finalResult << "#define LUISA_DEBUG_INFO 1\n";
    }
    uint64 immutableHeaderSize = detail::AddHeader(kernel.propagated_builtin_callables(), finalResult, false, isSpirV, noRegister, kernel.use_cooperative_operations(), fallback_rtx, bf16_fallback, spirv_capabilities);
    // Add motion blur ray tracing header (miss/closesthit entry points + _TraceClosestMotion)
    finalResult << ReadInternalHLSLFile("raytracing_motion_header");
    finalResult << native_code << "\n//"sv;
    finalResult << luisa::format("{}", custom_mask);
    finalResult << '\n';
    // Generate cbuffer FIRST to populate validate_index_map before codegen
    opt->funcType = CodegenStackData::FuncType::Callable;
    auto argRange = vstd::make_ite_range(kernel.arguments()).i_range();
    uint bind_count = 2;
    uint validation_count = 0;
    if (nonEmptyCbuffer || enable_debug_info) {
        uint64_t func_hash = kernel.hash();
        uint64_t hashes[] = {func_hash};
        GenerateCBuffer({&argRange}, varData, enable_debug_info, validation_count, hashes);
    }

    CodegenFunction(kernel, codegenData, nonEmptyCbuffer || enable_debug_info, true);

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
    PreprocessCodegenProperties(properties, varData, indexer, nonEmptyCbuffer || enable_debug_info, false, isSpirV, bind_count);
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
        opt->use_8bit,
        validation_count,
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
    bool noRegister,
    bool enable_debug_info,
    bool enable_fast_math,
    bool fallback_rtx,
    bool bf16_fallback,
    SpirvShaderCapabilities spirv_capabilities,
    bool emulate_8bit) {
    opt = CodegenStackData::Allocate(this);
    opt->isSpirv = isSpirV;
    opt->fallback_rtx = fallback_rtx;
    opt->bf16_fallback = bf16_fallback;
    opt->emulate_8bit = emulate_8bit && !isSpirV;
    opt->spirv_capabilities = spirv_capabilities;
    opt->enable_debug_info = enable_debug_info;
    opt->enable_fast_math = enable_fast_math;
    // CodegenStackData::ThreadLocalSpirv() = false;
    opt->kernel = vertFunc;
    opt->noRegister = noRegister;
    opt->isRaster = true;
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
    if (enable_debug_info) {
        finalResult << "#define LUISA_DEBUG_INFO 1\n";
    }
    auto opSet = vertFunc.propagated_builtin_callables();
    opSet.propagate(pixelFunc.propagated_builtin_callables());
    uint64 immutableHeaderSize = detail::AddHeader(opSet, finalResult, true, isSpirV, noRegister, vertFunc.use_cooperative_operations() || pixelFunc.use_cooperative_operations(), fallback_rtx, bf16_fallback, spirv_capabilities);
    finalResult << native_code << "\n//"sv;
    finalResult << luisa::format("{}", custom_mask);
    finalResult << '\n';
    // Vertex
    codegenData << "struct v2p{\n"sv;
    auto v2pType = vertFunc.return_type();
    if (v2pType->is_structure()) {
        opt->internalStruct.emplace(v2pType, "v2p");
        auto const &v2pMembers = v2pType->members();
        auto const &v2pAttrs = v2pType->member_attributes();
        // If member_attributes are not set (e.g. from DSL LUISA_STRUCT without
        // Clang annotations), default: member 0 = position, rest = TEXCOORDs.
        bool hasAttrs = (v2pMembers.size() == v2pAttrs.size());
        auto emit_interpolation_modifier = [&](size_t index,
                                               const Type *type) noexcept {
            auto interpolation = RasterInterpolation::DEFAULT;
            if (hasAttrs &&
                v2pAttrs[index].key ==
                    raster_interpolation_attribute_key) {
                if (!parse_raster_interpolation(
                        v2pAttrs[index], interpolation)) {
                    LUISA_ERROR(
                        "Invalid raster interpolation mode '{}'.",
                        v2pAttrs[index].value);
                }
            }
            auto element = type->is_vector() ? type->element() : type;
            auto floating = element->is_float16() ||
                            element->is_float32();
            if (interpolation == RasterInterpolation::DEFAULT) {
                interpolation = floating ?
                                    RasterInterpolation::CENTER_PERSPECTIVE :
                                    RasterInterpolation::FLAT;
            }
            if (interpolation != RasterInterpolation::FLAT &&
                !floating) {
                LUISA_ERROR(
                    "Perspective, centroid, and sample interpolation "
                    "require a floating-point varying.");
            }
            switch (interpolation) {
                case RasterInterpolation::CENTER_PERSPECTIVE: break;
                case RasterInterpolation::CENTER_NO_PERSPECTIVE:
                    codegenData << "noperspective "sv;
                    break;
                case RasterInterpolation::CENTROID_PERSPECTIVE:
                    codegenData << "centroid "sv;
                    break;
                case RasterInterpolation::CENTROID_NO_PERSPECTIVE:
                    codegenData << "centroid noperspective "sv;
                    break;
                case RasterInterpolation::SAMPLE_PERSPECTIVE:
                    codegenData << "sample "sv;
                    break;
                case RasterInterpolation::SAMPLE_NO_PERSPECTIVE:
                    codegenData << "sample noperspective "sv;
                    break;
                case RasterInterpolation::FLAT:
                    codegenData << "nointerpolation "sv;
                    break;
                case RasterInterpolation::DEFAULT:
                    LUISA_ERROR(
                        "Unresolved default raster interpolation.");
            }
        };
        size_t memberIdx = 0;
        bool pos = false;
        for (auto &&i : v2pMembers) {
            bool is_sv_pos = hasAttrs ? (v2pAttrs[memberIdx].key == "position"sv) : (memberIdx == 0);// default: first member is SV_POSITION
            if (!is_sv_pos && opt->isSpirv) {
                codegenData << luisa::format("[[vk::location({})]] ", memberIdx - 1);
            }
            if (!is_sv_pos) {
                emit_interpolation_modifier(memberIdx, i);
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
        bool hasAppAttrs = (appdataAttris.size() == appdataMems.size());
        // Default attribute keys when DSL LUISA_STRUCT doesn't provide member_attributes.
        // Matches the standard AppData layout used in raster_kernel.h.
        static constexpr vstd::string_view defaultAppKeys[] = {
            "position"sv, "normal"sv, "tangent"sv, "color"sv,
            "uv0"sv, "vertex_id"sv, "instance_id"sv, "is_front_face"sv};
        static constexpr vstd::string_view uintSVKeys[] = {
            "instance_id"sv, "vertex_id"sv, "is_front_face"sv};
        size_t uintSVIdx = 0;// tracks which SV_* key to assign for uint members
        opt->internalStruct.try_emplace(appdataType, "_mesh");
        codegenData << "struct _mesh{\n"sv;
        for (auto i : vstd::range(appdataMems.size())) {
            auto member = appdataMems[static_cast<size_t>(i)];
            vstd::string_view attrKey;
            if (hasAppAttrs) {
                attrKey = appdataAttris[static_cast<size_t>(i)].key;
                if (attrKey.empty()) [[unlikely]] {
                    LUISA_ERROR("Mesh-to-vertex structure member {} miss attributes.", static_cast<int64_t>(i));
                }
            } else {
                // Default: use index into standard AppData layout
                attrKey = (i < std::size(defaultAppKeys)) ? defaultAppKeys[i] : "uv0"sv;
                // Type-based correction for custom minimal AppData structs:
                // If a uint member would be mapped to a non-SV semantic (POSITION, NORMAL, etc.)
                // redirect it to the next available SV_* semantic (instance_id, vertex_id, etc.)
                // since uint members in mesh input are typically system-generated values.
                if (member == Type::of<uint>()) {
                    auto testIter = attributes.find(attrKey);
                    if (testIter != attributes.end()) {
                        auto sem = testIter->second.first;// e.g. "NORMAL", "SV_InstanceID"
                        if (sem.size() < 3 || !(sem[0] == 'S' && sem[1] == 'V' && sem[2] == '_')) {
                            // uint mapped to non-SV semantic — redirect to SV_* key
                            if (uintSVIdx < std::size(uintSVKeys)) {
                                attrKey = uintSVKeys[uintSVIdx++];
                            }
                        }
                    }
                }
            }
            if (!member->is_scalar() && !member->is_vector() && !member->is_array()) [[unlikely]] {
                LUISA_ERROR("Mesh-to-vertex structure do not support type {}", member->description());
            }

            auto iter = attributes.find(attrKey);
            if (iter == attributes.end()) [[unlikely]] {
                LUISA_ERROR("Invalid attribute: {}", attrKey);
            }

            // For array members, check element type instead of the array itself
            auto typeForCheck = member->is_array() ? member->element() : member;
            if (iter->second.second && iter->second.second != typeForCheck) [[unlikely]] {
                LUISA_ERROR("Attribute {} type {} mismatch with {}", attrKey, iter->second.second->description(), member->description());
            }
            if (opt->isSpirv) {
                codegenData << luisa::format("[[vk::location({})]] ", i);
            }
            if (member->is_array()) {
                // Emit as: elemType vN[arraySize] : SEMANTIC;
                auto elemType = member->element();
                GetTypeName(*elemType, codegenData, Usage::READ);
                codegenData
                    << " v"sv << vstd::to_string(i)
                    << '[' << vstd::to_string(member->dimension()) << "]:"
                    << iter->second.first
                    << ";\n"sv;
            } else {
                GetTypeName(*member, codegenData, Usage::READ);
                codegenData
                    << " v"sv << vstd::to_string(i) << ':'
                    << iter->second.first
                    << ";\n"sv;
            }
        }
        codegenData << "};\n";
    } else {
        LUISA_ERROR("Mesh-to-vertex must be a structure");
    }
    auto vertRange = vstd::make_ite_range(vert_args.subspan(1)).i_range();
    auto pixelRange = vstd::make_ite_range(pixelFunc.arguments().subspan(1)).i_range();
    std::initializer_list<vstd::IRange<Variable> *> funcs = {&vertRange, &pixelRange};

    bool nonEmptyCbuffer = IsCBufferNonEmpty(funcs);

    // Generate cbuffer FIRST to populate validate_index_map before codegen
    uint validation_count = 0;
    if (nonEmptyCbuffer || enable_debug_info) {
        uint64_t vert_hash = vertFunc.hash();
        uint64_t pixel_hash = pixelFunc.hash();
        uint64_t hashes[] = {vert_hash, pixel_hash};
        auto cbufferVertRange = vstd::make_ite_range(vert_args.subspan(1)).i_range();
        auto cbufferPixelRange = vstd::make_ite_range(pixelFunc.arguments().subspan(1)).i_range();
        std::initializer_list<vstd::IRange<Variable> *> cbufferFuncs = {&cbufferVertRange, &cbufferPixelRange};
        GenerateCBuffer(cbufferFuncs, varData, enable_debug_info, validation_count, hashes);
    }

    opt->appdataId = vert_args[0].uid();
    CodegenVertex(vertFunc, codegenData, nonEmptyCbuffer || enable_debug_info);
    opt->appdataId = -1;
    codegenData << "#elif defined(PS)\n"sv;
    size_t vert_arg_offset = 0;
    for (auto &i : vert_args.subspan(1)) {
        vert_arg_offset = std::max(vert_arg_offset, static_cast<size_t>(i.uid() + 1));
    }
    opt->argOffset = vert_arg_offset;
    CodegenPixel(pixelFunc, codegenData, nonEmptyCbuffer || enable_debug_info);
    codegenData << "#endif\n"sv;

    opt->funcType = CodegenStackData::FuncType::Callable;
    CodegenResult::Properties properties;
    DXILRegisterIndexer dxilRegisters;
    SpirVRegisterIndexer spvRegisters;
    RegisterIndexer &indexer = isSpirV ? static_cast<RegisterIndexer &>(spvRegisters) : static_cast<RegisterIndexer &>(dxilRegisters);
    PreprocessCodegenProperties(properties, varData, indexer, nonEmptyCbuffer || enable_debug_info, true, isSpirV, bind_count);
    // Reset argOffset for vertex function properties so buffer/texture names
    // use raw UIDs. Pixel function properties get offset UIDs to avoid
    // name collisions when both stages have resource variables with the same UID.
    auto savedArgOffset = opt->argOffset;
    opt->argOffset = 0;
    CodegenProperties(properties, varData, vertFunc, 1, indexer, bind_count);
    opt->argOffset = savedArgOffset;
    CodegenProperties(properties, varData, pixelFunc, 1, indexer, bind_count);
    PostprocessCodegenProperties(finalResult, false);
    finalResult << varData << incrementalFunc << codegenData;
    // https://learn.microsoft.com/en-us/windows/win32/direct3d12/root-signature-limits
    if (bind_count >= 64) [[unlikely]] {
        LUISA_ERROR("Arguments binding size: {} exceeds 64 32-bit units not supported by hardware device. Try to use bindless instead.", bind_count);
    }
    // Fresh ranges for GetTypeMD5 (also consumes the iterator)
    auto vr3 = vstd::make_ite_range(vert_args.subspan(1)).i_range();
    auto pr3 = vstd::make_ite_range(pixelFunc.arguments().subspan(1)).i_range();
    std::initializer_list<vstd::IRange<Variable> *> funcs3 = {&vr3, &pr3};
    return {
        std::move(finalResult),
        std::move(opt->printer),
        std::move(properties),
        opt->useTex2DBindless,
        opt->useTex3DBindless,
        opt->useBufferBindless,
        opt->use_8bit,
        validation_count,
        immutableHeaderSize,
        GetTypeMD5(funcs3)};
}

void CodegenUtility::AnalyzePromotedResourceArgs(Function kernel) {
    opt->promoted_resource_args.clear();
    if (!kernel || opt->enable_debug_info || opt->isRaster || opt->isRayTracing) {
        return;
    }
    auto is_promotable_resource = [](Variable const &v) noexcept {
        return v.tag() == Variable::Tag::BUFFER ||
               v.tag() == Variable::Tag::BINDLESS_ARRAY;
    };
    auto usage_writes = [](Usage u) noexcept {
        return (luisa::to_underlying(u) & luisa::to_underlying(Usage::WRITE)) != 0u;
    };

    luisa::unordered_map<uint32_t, Variable> kernel_res_args;
    for (auto &&arg : kernel.arguments()) {
        if (is_promotable_resource(arg)) {
            kernel_res_args.emplace(arg.uid(), arg);
        }
    }
    if (kernel_res_args.empty()) {
        return;
    }

    // Collect every function that CodegenFunction(kernel, ..., true) will emit.
    luisa::vector<Function> all_funcs;
    luisa::unordered_set<uint64_t> seen_funcs;
    auto collect_funcs = [&](auto &&self, Function f) -> void {
        if (!seen_funcs.emplace(f.hash()).second) { return; }
        all_funcs.emplace_back(f);
        for (auto &&c : f.custom_callables()) {
            self(self, c->function());
        }
    };
    collect_funcs(collect_funcs, kernel);
    if (all_funcs.size() <= 1u) {
        return;
    }

    struct CallSite {
        Function caller;
        CallExpr const *call;
    };
    luisa::vector<CallSite> call_sites;
    luisa::unordered_map<uint64_t, luisa::vector<uint64_t>> call_adj;

    struct CustomCallCollector final : public StmtVisitor, public ExprVisitor {
        Function caller;
        luisa::vector<CallSite> &sites;
        luisa::unordered_map<uint64_t, luisa::vector<uint64_t>> &adj;
        CustomCallCollector(Function c,
                            luisa::vector<CallSite> &s,
                            luisa::unordered_map<uint64_t, luisa::vector<uint64_t>> &a) noexcept
            : caller{c}, sites{s}, adj{a} {}

        void visit(const UnaryExpr *e) override {
            if (e->operand()) { e->operand()->accept(*this); }
        }
        void visit(const BinaryExpr *e) override {
            if (e->lhs()) { e->lhs()->accept(*this); }
            if (e->rhs()) { e->rhs()->accept(*this); }
        }
        void visit(const MemberExpr *e) override {
            if (e->self()) { e->self()->accept(*this); }
        }
        void visit(const AccessExpr *e) override {
            if (e->range()) { e->range()->accept(*this); }
            if (e->index()) { e->index()->accept(*this); }
        }
        void visit(const LiteralExpr *) override {}
        void visit(const RefExpr *) override {}
        void visit(const ConstantExpr *) override {}
        void visit(const TypeIDExpr *) override {}
        void visit(const StringIDExpr *) override {}
        void visit(const FuncRefExpr *) override {}
        void visit(const CpuCustomOpExpr *) override {}
        void visit(const GpuCustomOpExpr *) override {}
        void visit(const CastExpr *e) override {
            if (e->expression()) { e->expression()->accept(*this); }
        }
        void visit(const CallExpr *e) override {
            if (e->op() == CallOp::CUSTOM) {
                sites.emplace_back(CallSite{caller, e});
                adj[caller.hash()].emplace_back(e->custom().hash());
            }
            for (auto *a : e->arguments()) {
                if (a) { a->accept(*this); }
            }
        }
        void visit(const BreakStmt *) override {}
        void visit(const ContinueStmt *) override {}
        void visit(const CommentStmt *) override {}
        void visit(const SuspendStmt *) override {}
        void visit(const DebugBreakStmt *) override {}
        void visit(const ReturnStmt *s) override {
            if (s->expression()) { s->expression()->accept(*this); }
        }
        void visit(const ScopeStmt *s) override {
            for (auto *sub : s->statements()) {
                if (sub) { sub->accept(*this); }
            }
        }
        void visit(const IfStmt *s) override {
            if (s->condition()) { s->condition()->accept(*this); }
            if (s->true_branch()) { s->true_branch()->accept(*this); }
            if (s->false_branch()) { s->false_branch()->accept(*this); }
        }
        void visit(const LoopStmt *s) override {
            if (s->body()) { s->body()->accept(*this); }
        }
        void visit(const ForStmt *s) override {
            if (s->variable()) { s->variable()->accept(*this); }
            if (s->condition()) { s->condition()->accept(*this); }
            if (s->step()) { s->step()->accept(*this); }
            if (s->body()) { s->body()->accept(*this); }
        }
        void visit(const ExprStmt *s) override {
            if (s->expression()) { s->expression()->accept(*this); }
        }
        void visit(const SwitchStmt *s) override {
            if (s->expression()) { s->expression()->accept(*this); }
            if (s->body()) { s->body()->accept(*this); }
        }
        void visit(const SwitchCaseStmt *s) override {
            for (auto *e : s->expressions()) {
                if (e) { e->accept(*this); }
            }
            if (s->body()) { s->body()->accept(*this); }
        }
        void visit(const SwitchDefaultStmt *s) override {
            if (s->body()) { s->body()->accept(*this); }
        }
        void visit(const AssignStmt *s) override {
            if (s->lhs()) { s->lhs()->accept(*this); }
            if (s->rhs()) { s->rhs()->accept(*this); }
        }
        void visit(const RayQueryStmt *s) override {
            if (s->query()) { s->query()->accept(*this); }
            if (s->on_triangle_candidate()) { s->on_triangle_candidate()->accept(*this); }
            if (s->on_procedural_candidate()) { s->on_procedural_candidate()->accept(*this); }
        }
        void visit(const AutoDiffStmt *s) override {
            if (s->body()) { s->body()->accept(*this); }
        }
        void visit(const PrintStmt *s) override {
            for (auto *a : s->arguments()) {
                if (a) { a->accept(*this); }
            }
        }
    };

    for (auto &&f : all_funcs) {
        if (f.body() != nullptr) {
            CustomCallCollector collector{f, call_sites, call_adj};
            f.body()->accept(collector);
        }
    }
    if (call_sites.empty()) {
        return;
    }

    // Mark functions actually reachable from kernel via direct custom calls.
    luisa::unordered_set<uint64_t> reachable;
    luisa::vector<uint64_t> worklist;
    reachable.emplace(kernel.hash());
    worklist.emplace_back(kernel.hash());
    while (!worklist.empty()) {
        auto h = worklist.back();
        worklist.pop_back();
        if (auto it = call_adj.find(h); it != call_adj.end()) {
            for (auto callee_h : it->second) {
                if (reachable.emplace(callee_h).second) {
                    worklist.emplace_back(callee_h);
                }
            }
        }
    }

    struct LatticeVal {
        enum class Kind : uint8_t { UNSEEN, UNIQUE, CONFLICT };
        Kind kind{Kind::UNSEEN};
        Variable var{};
    };
    luisa::unordered_map<CodegenStackData::ValidateKey, LatticeVal, CodegenStackData::ValidateKeyHash> state;
    for (auto &&f : all_funcs) {
        if (f.tag() == Function::Tag::KERNEL) { continue; }
        auto init_kind = reachable.contains(f.hash()) ? LatticeVal::Kind::UNSEEN : LatticeVal::Kind::CONFLICT;
        for (auto &&arg : f.arguments()) {
            if (is_promotable_resource(arg)) {
                state.emplace(CodegenStackData::ValidateKey{f.hash(), arg.uid()}, LatticeVal{init_kind, {}});
            }
        }
    }
    if (state.empty()) {
        return;
    }

    auto compatible = [&](Variable const &kernel_var, Function callee, Variable const &formal) noexcept {
        if (kernel_var.tag() != formal.tag()) { return false; }
        if (formal.tag() == Variable::Tag::BUFFER &&
            kernel_var.type()->element() != formal.type()->element()) {
            return false;
        }
        if (usage_writes(callee.variable_usage(formal.uid())) &&
            !usage_writes(kernel.variable_usage(kernel_var.uid()))) {
            return false;
        }
        return true;
    };

    bool changed = true;
    while (changed) {
        changed = false;
        for (auto &&site : call_sites) {
            auto callee = site.call->custom();
            auto formal_args = callee.arguments();
            auto actual_args = site.call->arguments();
            if (formal_args.size() != actual_args.size()) { continue; }
            for (size_t i = 0u; i < formal_args.size(); ++i) {
                auto const &formal = formal_args[i];
                if (!is_promotable_resource(formal)) { continue; }
                auto &cur = state[CodegenStackData::ValidateKey{callee.hash(), formal.uid()}];
                if (cur.kind == LatticeVal::Kind::CONFLICT) { continue; }

                LatticeVal incoming{LatticeVal::Kind::CONFLICT, {}};
                if (actual_args[i] != nullptr &&
                    actual_args[i]->tag() == Expression::Tag::REF) {
                    auto actual_var = static_cast<RefExpr const *>(actual_args[i])->variable();
                    if (site.caller.tag() == Function::Tag::KERNEL) {
                        if (auto kit = kernel_res_args.find(actual_var.uid());
                            kit != kernel_res_args.end() &&
                            compatible(kit->second, callee, formal)) {
                            incoming = LatticeVal{LatticeVal::Kind::UNIQUE, kit->second};
                        }
                    } else {
                        if (auto sit = state.find(CodegenStackData::ValidateKey{site.caller.hash(), actual_var.uid()});
                            sit != state.end()) {
                            if (sit->second.kind == LatticeVal::Kind::UNSEEN) {
                                incoming = LatticeVal{LatticeVal::Kind::UNSEEN, {}};
                            } else if (sit->second.kind == LatticeVal::Kind::UNIQUE &&
                                       compatible(sit->second.var, callee, formal)) {
                                incoming = sit->second;
                            }
                        }
                    }
                }

                if (incoming.kind == LatticeVal::Kind::UNSEEN) {
                    continue;
                }
                if (cur.kind == LatticeVal::Kind::UNSEEN) {
                    cur = incoming;
                    changed = true;
                } else if (incoming.kind == LatticeVal::Kind::CONFLICT ||
                           cur.var.uid() != incoming.var.uid() ||
                           cur.var.tag() != incoming.var.tag()) {
                    cur.kind = LatticeVal::Kind::CONFLICT;
                    changed = true;
                }
            }
        }
    }

    // Compute saturating dynamic call-site multiplicity (0, 1, or >=2) from
    // kernel along the custom call graph. Only multi-call callables (>=2) are
    // promoted; single-use callables are already inlined and keep their
    // explicit parameter ABI unchanged.
    luisa::unordered_map<uint64_t, luisa::unordered_map<uint64_t, uint32_t>> edge_mult;
    for (auto &&site : call_sites) {
        auto &cnt = edge_mult[site.caller.hash()][site.call->custom().hash()];
        if (cnt < 2u) { ++cnt; }
    }
    luisa::unordered_map<uint64_t, uint32_t> dyn_calls;
    dyn_calls[kernel.hash()] = 1u;
    bool count_changed = true;
    while (count_changed) {
        count_changed = false;
        luisa::unordered_map<uint64_t, uint32_t> next_calls;
        next_calls[kernel.hash()] = 1u;
        for (auto &&[caller_h, callees] : edge_mult) {
            auto cit = dyn_calls.find(caller_h);
            if (cit == dyn_calls.end() || cit->second == 0u) { continue; }
            for (auto &&[callee_h, mult] : callees) {
                uint32_t contrib = std::min<uint32_t>(2u, cit->second * mult);
                auto &dst = next_calls[callee_h];
                dst = std::min<uint32_t>(2u, dst + contrib);
            }
        }
        for (auto &&[h, c] : next_calls) {
            if (dyn_calls[h] != c) {
                dyn_calls[h] = c;
                count_changed = true;
            }
        }
    }

    for (auto &&[key, val] : state) {
        if (val.kind == LatticeVal::Kind::UNIQUE) {
            auto dit = dyn_calls.find(key.func_hash);
            if (dit != dyn_calls.end() && dit->second >= 2u) {
                opt->promoted_resource_args.emplace(key, val.var);
            }
        }
    }
    if (opt->promoted_resource_args.empty()) {
        return;
    }

    // Prevent any HLSL lexical shadowing where a non-promoted parameter or local
    // of a callable happens to have the exact same emitted identifier as the
    // promoted global kernel resource.
    auto saved_func_type = opt->funcType;
    opt->funcType = CodegenStackData::FuncType::Callable;
    auto restore_func_type = vstd::scope_exit([&] { opt->funcType = saved_func_type; });
    for (auto &&f : all_funcs) {
        if (f.tag() == Function::Tag::KERNEL) { continue; }
        bool removed = true;
        while (removed) {
            removed = false;
            luisa::unordered_set<luisa::string> local_names;
            for (auto &&arg : f.arguments()) {
                if (opt->IsPromotedResourceArg(f, arg)) { continue; }
                vstd::StringBuilder sb;
                GetVariableName(f, arg, sb);
                local_names.emplace(sb.view());
            }
            for (auto &&loc : f.local_variables()) {
                vstd::StringBuilder sb;
                GetVariableName(f, loc, sb);
                local_names.emplace(sb.view());
            }
            for (auto &&arg : f.arguments()) {
                auto const *promoted = opt->FindPromotedResourceArg(f, arg);
                if (promoted == nullptr) { continue; }
                vstd::StringBuilder global_sb;
                GetVariableName(kernel, *promoted, global_sb);
                if (local_names.contains(luisa::string{global_sb.view()})) {
                    opt->promoted_resource_args.erase(
                        CodegenStackData::ValidateKey{f.hash(), arg.uid()});
                    removed = true;
                }
            }
        }
    }
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
                if (!warp_size.has_value()) {
                    warp_size = opt->default_warp_size;
                }
                if (warp_size.has_value() && !opt->isSpirv) {
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
            vis.VisitFunction(func);
        }
        result << "}\n"sv;
    };
    vstd::unordered_set<uint64_t> callableMap;
    auto sorted_callables = [](Function func) {
        vstd::vector<luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder>> ordered;
        ordered.reserve(func.custom_callables().size());
        for (auto &&c : func.custom_callables()) {
            ordered.emplace_back(c);
        }
        std::sort(ordered.begin(), ordered.end(), [](auto const &lhs, auto const &rhs) noexcept {
            return lhs->hash() < rhs->hash();
        });
        return ordered;
    };
    auto callable = [&](auto &&callable, Function func) -> void {
        for (auto &&i : sorted_callables(func)) {
            if (callableMap.emplace(i->hash()).second) {
                callable(callable, i->function());
            }
        }
        codegenOneFunc(func);
    };
    if (codegen_self)
        callable(callable, func);
    else {
        for (auto &&i : sorted_callables(func)) {
            if (callableMap.emplace(i->hash()).second) {
                callable(callable, i->function());
            }
        }
    }
}
}// namespace lc::hlsl
