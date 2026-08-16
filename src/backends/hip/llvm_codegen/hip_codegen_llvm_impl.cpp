//
// Created by mike on 3/18/26.
//

#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/IR/PassInstrumentation.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/ModRef.h>
#include <llvm/TargetParser/TargetParser.h>
#include <llvm/MC/MCSubtargetInfo.h>

#include <algorithm>

#include <luisa/core/clock.h>
#include <luisa/core/stl/hash.h>
#include <luisa/ast/type_registry.h>
#include "hip_codegen_llvm_impl.h"
#include "hip_callable_inline_graph.h"
#include "hip_private_memory.h"
#include "hip_llvm_pipeline.h"
#include "../../common/env_flag.h"
#include "hiprt_device_wrapper.hip"
#include "hip_codegen_llvm_device_bitcode.h"

// Per-arch HIPRT wrapper bitcode (compiled per-arch so arch-specific features
// like the hardware BVH stack on gfx1200/1201 are correctly compiled).
#include "hiprt_wrapper_gfx1030_embedded.h"
#include "hiprt_wrapper_gfx1100_embedded.h"
#include "hiprt_wrapper_gfx1200_embedded.h"
#include "hiprt_wrapper_gfx1201_embedded.h"

#undef None

namespace luisa::compute::hip {

namespace {

[[nodiscard]] luisa::span<const std::byte>
hip_codegen_llvm_embedded_rt_wrapper(
    luisa::string_view amdgpu_arch) noexcept {
    const unsigned char *data = nullptr;
    size_t size = 0u;
    if (amdgpu_arch == "gfx1030") {
        data = luisa_compute_hip_hiprt_wrapper_gfx1030;
        size = luisa_compute_hip_hiprt_wrapper_gfx1030_size;
    } else if (amdgpu_arch == "gfx1100") {
        data = luisa_compute_hip_hiprt_wrapper_gfx1100;
        size = luisa_compute_hip_hiprt_wrapper_gfx1100_size;
    } else if (amdgpu_arch == "gfx1200") {
        data = luisa_compute_hip_hiprt_wrapper_gfx1200;
        size = luisa_compute_hip_hiprt_wrapper_gfx1200_size;
    } else if (amdgpu_arch == "gfx1201") {
        data = luisa_compute_hip_hiprt_wrapper_gfx1201;
        size = luisa_compute_hip_hiprt_wrapper_gfx1201_size;
    } else {
        LUISA_ERROR_WITH_LOCATION(
            "HIP ray tracing does not have an embedded wrapper for "
            "AMDGPU architecture '{}'.",
            amdgpu_arch);
    }
    LUISA_ASSERT(
        data != nullptr && size != 0u,
        "HIPRT wrapper bitcode is empty for architecture '{}'.",
        amdgpu_arch);
    return {
        reinterpret_cast<const std::byte *>(data),
        size};
}

[[nodiscard]] bool function_uses_ray_tracing_recursively(
    const xir::Function *function,
    llvm::DenseSet<const xir::Function *> &visited) noexcept {
    if (function == nullptr || !visited.insert(function).second) {
        return false;
    }
    auto definition = function->definition();
    if (definition == nullptr) {
        // A native/external callee is opaque to XIR. Treat it as potentially
        // reentrant rather than allowing it to corrupt a suspended LDS stack.
        return true;
    }
    auto uses_ray_tracing = false;
    definition->traverse_instructions(
        [&](const xir::Instruction *instruction) noexcept {
            if (uses_ray_tracing) { return; }
            if (instruction->isa<xir::ResourceQueryInst>()) {
                auto query = static_cast<const xir::ResourceQueryInst *>(
                    instruction);
                switch (query->op()) {
                    case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST:
                    case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY:
                    case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL:
                    case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY:
                    case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR:
                    case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR:
                    case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR:
                    case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR:
                        uses_ray_tracing = true;
                        return;
                    default: break;
                }
            }
            for (auto operand_use : instruction->operand_uses()) {
                if (auto operand = operand_use->value();
                    operand != nullptr && operand->isa<xir::Function>() &&
                    function_uses_ray_tracing_recursively(
                        static_cast<const xir::Function *>(operand),
                        visited)) {
                    uses_ray_tracing = true;
                    return;
                }
            }
        });
    return uses_ray_tracing;
}

}// namespace

uint64_t hip_codegen_llvm_embedded_rt_wrapper_hash(
    luisa::string_view amdgpu_arch) noexcept {
    constexpr auto seed = 0x4849505254575241ull;
    const auto wrapper =
        hip_codegen_llvm_embedded_rt_wrapper(amdgpu_arch);
    return luisa::hash64(
        wrapper.data(), wrapper.size_bytes(), seed);
}

HIPCodegenLLVMImpl::FunctionContext::FunctionContext(llvm::Function *f) noexcept
    : llvm_func{f},
      llvm_alloca_block{llvm::BasicBlock::Create(f->getContext(), "alloca", f)},
      llvm_entry_block{llvm::BasicBlock::Create(f->getContext(), "entry", f)} {
    IB b{llvm_alloca_block};
    b.CreateBr(llvm_entry_block);
}

HIPCodegenLLVMImpl::HIPCodegenLLVMImpl(HIPCodegenLLVMConfig config) noexcept
    : _config{std::move(config)} {
    LUISA_ASSERT(!_config.entry_point.empty(),
                 "HIP kernel entry point must not be empty.");
    LUISA_ASSERT(_config.block_size[0] > 0u && _config.block_size[1] > 0u && _config.block_size[2] > 0u,
                 "Block size must be constant and greater than zero for now.");
    Clock clk;
    _initialize();
    LUISA_VERBOSE_WITH_LOCATION("HIP LLVM codegen initialized in {} ms.", clk.toc());
}

void HIPCodegenLLVMImpl::_collect_print_info(
    const xir::Module &xir_module) noexcept {
    _print_info.clear();
    _print_formats.clear();
    for (auto function : xir_module.function_list()) {
        if (auto definition = function->definition()) {
            definition->traverse_instructions(
                [this](const xir::Instruction *instruction) noexcept {
                    if (instruction->derived_instruction_tag() !=
                        xir::DerivedInstructionTag::PRINT) {
                        return;
                    }
                    auto print = static_cast<const xir::PrintInst *>(instruction);
                    luisa::vector<const Type *> member_types;
                    member_types.reserve(print->operand_count() + 2u);
                    member_types.emplace_back(Type::of<uint>());
                    member_types.emplace_back(Type::of<uint>());
                    for (auto operand_use : print->operand_uses()) {
                        LUISA_ASSERT(operand_use->value() != nullptr,
                                     "Print operand is null.");
                        member_types.emplace_back(operand_use->value()->type());
                    }
                    auto argument_pack_type = Type::structure(member_types);
                    auto index = static_cast<uint32_t>(_print_formats.size());
                    auto [_, inserted] = _print_info.emplace(
                        print, PrintInfo{argument_pack_type, index});
                    LUISA_ASSERT(inserted, "Duplicate XIR PrintInst encountered.");
                    _print_formats.emplace_back(
                        print->format(), argument_pack_type);
                });
        }
    }
    LUISA_ASSERT(_print_formats.empty() != _config.requires_printing,
                 "HIP printing metadata mismatch: config requires_printing={}, "
                 "but {} print format(s) were found.",
                 _config.requires_printing, _print_formats.size());
}

void HIPCodegenLLVMImpl::_analyze_ray_tracing_usage(
    const xir::Module &module) noexcept {
    llvm::DenseSet<const xir::Function *> visited;
    for (auto function : module.function_list()) {
        // Only code reachable from kernels affects this module's kernel ABI and
        // traversal-stack selection. Unused callables must not pessimize it.
        if (function->isa<xir::KernelFunction>()) {
            _analyze_ray_tracing_in_function(function, visited);
        }
    }
}

void HIPCodegenLLVMImpl::_analyze_ray_tracing_in_function(
    const xir::Function *function,
    llvm::DenseSet<const xir::Function *> &visited) noexcept {
    if (function == nullptr || !visited.insert(function).second) { return; }
    auto definition = function->definition();
    if (definition == nullptr) { return; }
    definition->traverse_instructions([&](const xir::Instruction *instruction) noexcept {
        if (instruction->isa<xir::RayQueryPipelineInst>()) {
            _rt_analysis.uses_ray_query_pipeline = true;
            auto pipeline =
                static_cast<const xir::RayQueryPipelineInst *>(instruction);
            if (_ray_query_pipeline_admits_native_closest_reduction(
                    pipeline)) {
                _native_closest_reduction_pipelines.insert(pipeline);
            }
            llvm::DenseSet<const xir::Function *> handler_visited;
            _rt_analysis.ray_query_pipeline_handler_uses_ray_tracing |=
                function_uses_ray_tracing_recursively(
                    pipeline->on_surface_function(), handler_visited) ||
                function_uses_ray_tracing_recursively(
                    pipeline->on_procedural_function(), handler_visited);
        } else if (instruction->isa<xir::RayQueryLoopInst>() ||
                   instruction->isa<xir::RayQueryDispatchInst>()) {
            _rt_analysis.uses_resumable_ray_query_control = true;
        } else if (instruction->isa<xir::RayQueryObjectWriteInst>()) {
            auto write = static_cast<const xir::RayQueryObjectWriteInst *>(instruction);
            if (write->op() ==
                xir::RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED) {
                _rt_analysis.uses_resumable_ray_query_control = true;
            }
        } else if (instruction->isa<xir::ResourceWriteInst>()) {
            auto write = static_cast<const xir::ResourceWriteInst *>(instruction);
            if (write->op() ==
                xir::ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY) {
                // The same dispatch may make the update observable from a
                // candidate handler or another lane. Only a complete absence
                // proof over the kernel-reachable graph licenses the packed
                // instance-node snapshot.
                _rt_analysis.writes_instance_opacity = true;
            }
        }
        if (instruction->isa<xir::ResourceQueryInst>()) {
            switch (static_cast<const xir::ResourceQueryInst *>(instruction)->op()) {
                case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST: [[fallthrough]];
                case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY: {
                    _rt_analysis.uses_ray_tracing = true;
                    _rt_analysis.uses_static_trace = true;
                    break;
                }
                case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL: [[fallthrough]];
                case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY: {
                    _rt_analysis.uses_ray_tracing = true;
                    _rt_analysis.uses_ray_query = true;
                    break;
                }
                case xir::ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR: [[fallthrough]];
                case xir::ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR: {
                    _rt_analysis.uses_ray_tracing = true;
                    _rt_analysis.uses_motion_blur = true;
                    break;
                }
                case xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR: [[fallthrough]];
                case xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: {
                    _rt_analysis.uses_ray_tracing = true;
                    _rt_analysis.uses_ray_query = true;
                    _rt_analysis.uses_motion_blur = true;
                    _rt_analysis.uses_motion_ray_query = true;
                    break;
                }
                default: break;
            }
        }
        for (auto operand_use : instruction->operand_uses()) {
            if (auto operand = operand_use->value();
                operand != nullptr && operand->isa<xir::Function>()) {
                _analyze_ray_tracing_in_function(
                    static_cast<const xir::Function *>(operand), visited);
            }
        }
    });
}

bool HIPCodegenLLVMImpl::_function_uses_resumable_ray_query_state(
    const xir::Function *function) const noexcept {
    LUISA_ASSERT(function != nullptr,
                 "Cannot classify a null HIP RayQuery state domain.");
    // Every function owns exactly one private rq.state allocation. Thus the
    // selected representation is constant over all pipelines in a function:
    // individual pipeline selection would make the same allocation serve two
    // incompatible state machines. Mixed codegen varies only across disjoint
    // function-owned storage domains.
    return _rt_analysis.uses_ray_query &&
           !_uses_synchronous_ray_query_pipeline &&
           (!_uses_mixed_ray_query_pipeline ||
            std::find(
                _config.resumable_ray_query_state_functions.begin(),
                _config.resumable_ray_query_state_functions.end(),
                function) !=
                _config.resumable_ray_query_state_functions.end());
}

void HIPCodegenLLVMImpl::_initialize() noexcept {
    static std::once_flag once_flag;
    std::call_once(once_flag, [] {
        LLVMInitializeAMDGPUTargetInfo();
        LLVMInitializeAMDGPUTarget();
        LLVMInitializeAMDGPUTargetMC();
        LLVMInitializeAMDGPUAsmPrinter();
        LLVMInitializeAMDGPUAsmParser();
    });

    static auto target = [] {
        std::string error;
        if (auto t = llvm::TargetRegistry::lookupTarget(llvm::Triple{amdgpu_target_triple}, error)) {
            return t;
        }
        LUISA_ERROR_WITH_LOCATION("Failed to lookup target '{}': {}", amdgpu_target_triple, error);
    }();

    llvm::TargetOptions options;
    options.NoTrappingFPMath = true;
    if (_config.enable_fast_math) {
        options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
    }

    auto opt_level = llvm::CodeGenOptLevel::Default;
    switch (_config.opt_level) {
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_NONE: opt_level = llvm::CodeGenOptLevel::None; break;
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_LESS: opt_level = llvm::CodeGenOptLevel::Less; break;
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_DEFAULT: opt_level = llvm::CodeGenOptLevel::Default; break;
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_AGGRESSIVE: opt_level = llvm::CodeGenOptLevel::Aggressive; break;
    }

    LUISA_ASSERT(!_config.amdgpu_arch.empty(), "AMDGPU architecture must not be empty.");
    auto cpu_name = llvm::StringRef{_config.amdgpu_arch};
    auto gpu_kind = llvm::AMDGPU::parseArchAMDGCN(cpu_name);
    LUISA_ASSERT(gpu_kind != llvm::AMDGPU::GK_NONE,
                 "Unsupported AMDGPU architecture '{}'.", _config.amdgpu_arch);
    auto isa = llvm::AMDGPU::getIsaVersion(cpu_name);
    _supports_hardware_rt_stack = isa.Major >= 12u;
    // Direct motion closest/any traversal has its own private stack. A motion
    // ray query needs the generic dynamic stack. Candidate handlers that trace
    // are also non-reentrant with the lane-local LDS stack and therefore keep
    // the module on the software path.
    _uses_hardware_rt_stack =
        _supports_hardware_rt_stack &&
        !_rt_analysis.uses_motion_ray_query &&
        !_rt_analysis.ray_query_pipeline_handler_uses_ray_tracing;
    // The compact path is selected from XIR semantics. RayQueryPipelineInst
    // guarantees that traversal and handlers form one synchronous operation;
    // any explicit proceed/dispatch operation or nested traversal in a handler
    // requires the reentrant software state instead.
    // HIPRT's dynamic stack currently linearizes x/y lanes only, so retain the
    // resumable path for a true 3-D workgroup until that upstream ABI is extended.
    const auto ray_query_pipeline_is_synchronously_eligible =
        !_config.force_resumable_ray_query_pipeline &&
        _rt_analysis.uses_ray_query_pipeline &&
        !_rt_analysis.uses_resumable_ray_query_control &&
        !_rt_analysis.uses_motion_ray_query &&
        !_rt_analysis.ray_query_pipeline_handler_uses_ray_tracing &&
        _config.block_size[2] == 1u;
    const auto resumable_state_functions_are_valid = [&]() noexcept {
        llvm::DenseSet<const xir::Function *> unique_functions;
        for (auto function :
             _config.resumable_ray_query_state_functions) {
            if (function == nullptr ||
                !unique_functions.insert(function).second) {
                return false;
            }
        }
        return true;
    }();
    LUISA_ASSERT(
        resumable_state_functions_are_valid,
        "HIP resumable RayQuery state functions must be non-null and unique.");
    _uses_mixed_ray_query_pipeline =
        ray_query_pipeline_is_synchronously_eligible &&
        !_config.resumable_ray_query_state_functions.empty();
    _uses_synchronous_ray_query_pipeline =
        ray_query_pipeline_is_synchronously_eligible &&
        !_uses_mixed_ray_query_pipeline;
    _uses_resumable_hardware_ray_query_pipeline =
        _uses_hardware_rt_stack &&
        (!_uses_synchronous_ray_query_pipeline ||
         _uses_mixed_ray_query_pipeline);
    // Dynamic global-stack storage is part of the HIP ray-query kernel ABI on
    // every architecture. The resumable gfx12 path does not consume it, but
    // keeping the host metadata conservative lets XIR select the compact
    // synchronous implementation without changing the already-cached AST ABI.
    _requires_global_rt_stack =
        _rt_analysis.uses_ray_query ||
        (!_uses_hardware_rt_stack && _rt_analysis.uses_static_trace);
    auto requires_hiprt = _rt_analysis.uses_ray_tracing ||
                          _config.requires_ray_tracing ||
                          _config.requires_ray_query ||
                          _config.requires_motion_blur;
    LUISA_ASSERT(!_supports_hardware_rt_stack || !requires_hiprt || _config.wave_size == 32u,
                 "The gfx12 HIPRT hardware stack requires wave32 code generation, "
                 "but this ray-tracing kernel requests wave{}.",
                 _config.wave_size);
    // The AMDGPU target's default wave size is architecture-dependent. The XIR
    // lowering below specializes ballots, masks and shuffle trees for one exact
    // size, so leaving wave32 implicit can silently generate wave64 code with
    // 32-lane reduction logic on targets whose default differs.
    auto features = _config.wave_size == 64 ?
                        llvm::StringRef{"+wavefrontsize64"} :
                        llvm::StringRef{"+wavefrontsize32"};
    _target_machine = target->createTargetMachine(
        llvm::Triple{amdgpu_target_triple}, cpu_name, features,
        options, llvm::Reloc::Static, llvm::CodeModel::Small, opt_level);

    _data_layout = std::make_unique<llvm::DataLayout>(_target_machine->createDataLayout());

    // parse OCML bitcode as the starting module (like CUDA's libdevice)
    _llvm_module = [&] {
        llvm::SMDiagnostic error;
        llvm::StringRef bc{reinterpret_cast<const char *>(luisa_compute_hip_ocml),
                           luisa_compute_hip_ocml_size};
        if (auto m = llvm::parseIR({bc, "ocml.bc"}, error, _llvm_context)) {
            llvm::StripDebugInfo(*m);
            return m;
        }
        LUISA_ERROR_WITH_LOCATION("Failed to parse OCML bitcode: {}", error.getMessage());
    }();

    // set target triple and data layout
    _llvm_module->setTargetTriple(llvm::Triple{amdgpu_target_triple});
    _llvm_module->setDataLayout(*_data_layout);

    // internalize all OCML functions
    for (auto &&f : *_llvm_module) {
        if (f.getName().starts_with("__ocml_")) {
            f.setLinkage(llvm::Function::PrivateLinkage);
            f.removeFnAttr(llvm::Attribute::StackProtect);
        }
    }

    _specialize_oclc_options();

    _llvm_buffer_type = _get_llvm_buffer_type();
    _llvm_texture_type = _get_llvm_texture_type();
    _llvm_bindless_array_type = _get_llvm_bindless_array_type();
    _llvm_bindless_array_slot_type = _get_llvm_bindless_array_slot_type();
    _llvm_accel_type = _get_llvm_accel_type();
    _llvm_accel_instance_type = _get_llvm_accel_instance_type();
    _llvm_ray_type = _get_llvm_ray_type();
    _llvm_surface_hit_type = _get_llvm_surface_hit_type();
    _llvm_procedural_hit_type = _get_llvm_procedural_hit_type();
    _llvm_committed_hit_type = _get_llvm_committed_hit_type();
    _llvm_ray_query_type = _get_llvm_ray_query_type();
}

void HIPCodegenLLVMImpl::_specialize_oclc_options() noexcept {
    // Provide the target configuration globals required by OCML/OCKL.
    auto set_oclc_option = [&](llvm::StringRef name, llvm::Value *value) {
        if (auto gv = _llvm_module->getGlobalVariable(name)) {
            llvm::SmallVector<llvm::LoadInst *, 8> loads;
            for (auto user : gv->users()) {
                if (auto load = llvm::dyn_cast<llvm::LoadInst>(user)) {
                    loads.emplace_back(load);
                }
            }
            for (auto load : loads) {
                load->replaceAllUsesWith(value);
                load->eraseFromParent();
            }
            if (gv->use_empty()) {
                gv->eraseFromParent();
            }
        }
    };
    auto llvm_i8_type = llvm::Type::getInt8Ty(_llvm_context);
    auto llvm_i32_type = llvm::Type::getInt32Ty(_llvm_context);
    set_oclc_option("__oclc_finite_only_opt", llvm::ConstantInt::get(llvm_i8_type, _config.enable_fast_math ? 1 : 0));
    set_oclc_option("__oclc_unsafe_math_opt", llvm::ConstantInt::get(llvm_i8_type, _config.enable_fast_math ? 1 : 0));
    set_oclc_option("__oclc_ABI_version", llvm::ConstantInt::get(llvm_i32_type, 600u));
    set_oclc_option("__oclc_wavefrontsize64", llvm::ConstantInt::get(llvm_i8_type, _config.wave_size == 64u));
    set_oclc_option("__oclc_wavefrontsize_log2", llvm::ConstantInt::get(llvm_i32_type, _config.wave_size == 64u ? 6u : 5u));
    auto isa = llvm::AMDGPU::getIsaVersion(llvm::StringRef{_config.amdgpu_arch});
    auto isa_version = isa.Major * 1000u + isa.Minor * 100u + isa.Stepping;
    set_oclc_option("__oclc_ISA_version", llvm::ConstantInt::get(llvm_i32_type, isa_version));
}

void HIPCodegenLLVMImpl::_link_native_include() noexcept {
    if (_config.native_include.empty()) { return; }
    auto source_name = _config.source_file.empty() ?
                           llvm::StringRef{"hip_native_include.ll"} :
                           llvm::StringRef{_config.source_file};
    auto buffer = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef{_config.native_include.data(),
                        _config.native_include.size()},
        source_name, false);
    llvm::SMDiagnostic error;
    auto native_module = llvm::parseIR(buffer->getMemBufferRef(), error,
                                       _llvm_context);
    if (native_module == nullptr) {
        std::string diagnostic;
        llvm::raw_string_ostream stream{diagnostic};
        error.print("LuisaCompute HIP native include", stream);
        stream.flush();
        LUISA_ERROR_WITH_LOCATION(
            "Failed to parse HIP native include as LLVM IR/bitcode:\n{}",
            diagnostic);
    }
    auto expected_triple = llvm::Triple{amdgpu_target_triple};
    auto native_triple = native_module->getTargetTriple();
    LUISA_ASSERT(native_triple.str().empty() || native_triple == expected_triple,
                 "HIP native include targets '{}', expected '{}'.",
                 native_triple.str(), expected_triple.str());
    auto native_layout = native_module->getDataLayoutStr();
    auto expected_layout = _llvm_module->getDataLayoutStr();
    LUISA_ASSERT(native_layout.empty() || native_layout == expected_layout,
                 "HIP native include has an incompatible LLVM data layout.");
    native_module->setTargetTriple(expected_triple);
    native_module->setDataLayout(*_data_layout);

    // Native IR may have been produced for a different subtarget. Letting such
    // attributes survive until optimization can silently specialize helper
    // functions for the wrong ISA even though the module triple/layout match.
    // Accept attributes that describe this target, then canonicalize every
    // definition to the TargetMachine configuration used for the whole shader.
    auto target_cpu = _target_machine->getTargetCPU();
    auto target_features = _target_machine->getTargetFeatureString();
    auto subtarget = _target_machine->getMCSubtargetInfo();
    LUISA_ASSERT(subtarget != nullptr,
                 "HIP target machine does not expose subtarget information.");
    for (auto &function : *native_module) {
        if (auto cpu_attr = function.getFnAttribute("target-cpu");
            cpu_attr.isValid()) {
            auto native_cpu = cpu_attr.getValueAsString();
            LUISA_ASSERT(native_cpu.empty() || native_cpu == target_cpu,
                         "HIP native function '{}' targets CPU '{}', expected '{}'.",
                         function.getName().str(), native_cpu.str(), target_cpu.str());
        }
        if (auto features_attr = function.getFnAttribute("target-features");
            features_attr.isValid()) {
            auto native_features = features_attr.getValueAsString();
            LUISA_ASSERT(native_features.empty() ||
                             subtarget->checkFeatures(native_features),
                         "HIP native function '{}' requests target features '{}' "
                         "that are incompatible with CPU '{}' and features '{}'.",
                         function.getName().str(), native_features.str(),
                         target_cpu.str(), target_features.str());
        }
        function.removeFnAttr("target-cpu");
        function.removeFnAttr("target-features");
        if (!function.isDeclaration()) {
            function.addFnAttr("target-cpu", target_cpu);
            if (!target_features.empty()) {
                function.addFnAttr("target-features", target_features);
            }
        }
    }

    static constexpr llvm::Attribute::AttrKind abi_attributes[]{
        llvm::Attribute::InReg,
        llvm::Attribute::SExt,
        llvm::Attribute::ZExt,
#if LLVM_VERSION_MAJOR >= 22
        llvm::Attribute::NoExt,
#endif
        llvm::Attribute::ByRef,
        llvm::Attribute::ByVal,
        llvm::Attribute::ElementType,
        llvm::Attribute::InAlloca,
        llvm::Attribute::Preallocated,
        llvm::Attribute::StructRet,
        llvm::Attribute::Nest,
        llvm::Attribute::Returned,
        llvm::Attribute::SwiftAsync,
        llvm::Attribute::SwiftError,
        llvm::Attribute::SwiftSelf,
        llvm::Attribute::Naked,
        llvm::Attribute::NoRedZone,
        llvm::Attribute::ReturnsTwice,
        llvm::Attribute::StackAlignment,
    };

    for (auto &[xir_value, llvm_value] : _xir_to_llvm_global) {
        auto external = llvm::dyn_cast<llvm::Function>(llvm_value);
        if (external == nullptr ||
            !xir_value->isa<xir::ExternalFunction>()) {
            continue;
        }
        auto definition = native_module->getFunction(external->getName());
        LUISA_ASSERT(definition != nullptr && !definition->isDeclaration(),
                     "HIP native include does not define external function '{}'.",
                     external->getName().str());
        LUISA_ASSERT(!definition->hasLocalLinkage() &&
                         !definition->hasAvailableExternallyLinkage(),
                     "HIP native function '{}' must provide an externally linkable definition.",
                     external->getName().str());
        LUISA_ASSERT(definition->getFunctionType() ==
                         external->getFunctionType(),
                     "HIP native function '{}' does not match its ExternalCallable ABI.",
                     external->getName().str());
        LUISA_ASSERT(definition->getAddressSpace() == external->getAddressSpace(),
                     "HIP native function '{}' has an incompatible function address space.",
                     external->getName().str());
        LUISA_ASSERT(definition->getCallingConv() == external->getCallingConv(),
                     "HIP native function '{}' has an incompatible LLVM calling convention.",
                     external->getName().str());
        auto definition_attributes = definition->getAttributes();
        auto external_attributes = external->getAttributes();
        auto check_abi_attributes = [&](llvm::AttributeSet native_attributes,
                                        llvm::AttributeSet expected_attributes,
                                        llvm::StringRef position) noexcept {
            for (auto kind : abi_attributes) {
                auto native_attribute = native_attributes.getAttribute(kind);
                auto expected_attribute = expected_attributes.getAttribute(kind);
                LUISA_ASSERT(native_attribute == expected_attribute,
                             "HIP native function '{}' has incompatible '{}' ABI "
                             "attribute on {}.",
                             external->getName().str(),
                             llvm::Attribute::getNameFromAttrKind(kind).str(),
                             position.str());
            }
        };
        check_abi_attributes(definition_attributes.getFnAttrs(),
                             external_attributes.getFnAttrs(), "the function");
        check_abi_attributes(definition_attributes.getRetAttrs(),
                             external_attributes.getRetAttrs(), "the return value");
        auto xir_external = static_cast<const xir::ExternalFunction *>(xir_value);
        auto xir_argument = xir_external->arguments().begin();
        for (auto i = 0u; i < definition->arg_size(); ++i) {
            LUISA_ASSERT(xir_argument != xir_external->arguments().end(),
                         "HIP ExternalCallable argument count changed during ABI validation.");
            auto argument = *xir_argument;
            ++xir_argument;
            auto native_attributes = definition_attributes.getParamAttrs(i);
            if (auto alignment_attribute =
                    native_attributes.getAttribute(llvm::Attribute::Alignment);
                alignment_attribute.isValid()) {
                auto native_alignment = alignment_attribute.getAlignment();
                auto available_alignment = _get_type_alignment(argument->type());
                LUISA_ASSERT(argument->is_reference() && native_alignment.has_value() &&
                                 native_alignment->value() <= available_alignment,
                             "HIP native function '{}' requires alignment {} on parameter {}, "
                             "but its ExternalCallable argument only guarantees alignment {}.",
                             external->getName().str(),
                             native_alignment ? native_alignment->value() : 0u,
                             i, available_alignment);
            }
            auto position = fmt::format("parameter {}", i);
            check_abi_attributes(native_attributes,
                                 external_attributes.getParamAttrs(i), position);
        }
        LUISA_ASSERT(xir_argument == xir_external->arguments().end(),
                     "HIP ExternalCallable argument count changed during ABI validation.");
    }
    if (llvm::Linker::linkModules(
            *_llvm_module, std::move(native_module),
            llvm::Linker::Flags::LinkOnlyNeeded)) {
        LUISA_ERROR_WITH_LOCATION(
            "Failed to link HIP native include into the generated LLVM module.");
    }
}

void HIPCodegenLLVMImpl::_link_ockl_if_needed() noexcept {
    auto llvm_printf_begin = _llvm_module->getFunction("__ockl_printf_begin");
    if (llvm_printf_begin == nullptr || llvm_printf_begin->use_empty() || !llvm_printf_begin->isDeclaration()) {
        return;
    }

    llvm::StringRef bitcode{
        reinterpret_cast<const char *>(luisa_compute_hip_ockl),
        static_cast<size_t>(luisa_compute_hip_ockl_size)};
    auto buffer = llvm::MemoryBuffer::getMemBuffer(bitcode, "ockl.bc", false);
    auto module = llvm::parseBitcodeFile(*buffer, _llvm_context);
    if (!module) {
        LUISA_ERROR_WITH_LOCATION("Failed to parse embedded AMD OCKL bitcode.");
    }
    if (llvm::Linker::linkModules(
            *_llvm_module, std::move(*module),
            llvm::Linker::Flags::LinkOnlyNeeded)) {
        LUISA_ERROR_WITH_LOCATION("Failed to link AMD OCKL printf support.");
    }
    _specialize_oclc_options();
}

void HIPCodegenLLVMImpl::_postprocess_rt_kernel() noexcept {

    if (!_rt_analysis.uses_ray_tracing) { return; }

    // Step 1: Link the per-arch RT wrapper bitcode (hiprt traversal wrappers)
    const auto wrapper =
        hip_codegen_llvm_embedded_rt_wrapper(
            _config.amdgpu_arch);
    llvm::StringRef wrapper_bc{
        reinterpret_cast<const char *>(wrapper.data()),
        wrapper.size_bytes()};
    auto wrapper_buf = llvm::MemoryBuffer::getMemBuffer(wrapper_bc, "hiprt_wrapper", false);
    auto wrapper_module = llvm::parseBitcodeFile(*wrapper_buf, _llvm_context);
    if (!wrapper_module) {
        LUISA_ERROR_WITH_LOCATION("Failed to parse HIPRT wrapper bitcode.");
    }

    if (auto *wrapper_flags = (*wrapper_module)->getNamedMetadata("llvm.module.flags")) {
        wrapper_flags->eraseFromParent();
    }
    // Strip llvm.used and llvm.compiler.used from the wrapper module so that
    // __attribute__((used)) on ray query functions doesn't force them to survive
    // DCE when linked with LinkOnlyNeeded.
    if (auto *used = (*wrapper_module)->getNamedGlobal("llvm.used")) {
        used->eraseFromParent();
    }
    if (auto *compiler_used = (*wrapper_module)->getNamedGlobal("llvm.compiler.used")) {
        compiler_used->eraseFromParent();
    }

    if (llvm::Linker::linkModules(*_llvm_module, std::move(*wrapper_module),
                                  llvm::Linker::Flags::LinkOnlyNeeded)) {
        LUISA_ERROR_WITH_LOCATION("Failed to link kernel module with HIPRT wrapper bitcode.");
    }

    // HIPRT has one dynamically-dispatched intersectFunc/filterFunc pair for
    // all ray types. Consequently LinkOnlyNeeded may retain the native-closest
    // callback arm while linking an otherwise resumable query module, even
    // though no traversal in that module can construct the reserved native
    // ray type. Such a module intentionally has no generated pipeline
    // dispatcher. Close this optional cross-bitcode interface with a trapping
    // definition: the call is unreachable under the ray-type construction
    // invariant and regular IPO can remove it, while an ABI regression fails
    // deterministically instead of surfacing as an unresolved device symbol.
    // A module with a native pipeline already owns the real strong definition
    // and never enters this branch.
    auto close_optional_pipeline_dispatch =
        [&](llvm::StringRef name) noexcept {
            if (auto dispatcher = _llvm_module->getFunction(name);
                dispatcher != nullptr && dispatcher->isDeclaration()) {
                auto entry = llvm::BasicBlock::Create(
                    _llvm_context, "unavailable", dispatcher);
                IB b{entry};
                auto trap = llvm::Intrinsic::getOrInsertDeclaration(
                    _llvm_module.get(), llvm::Intrinsic::trap);
                b.CreateCall(trap);
                b.CreateUnreachable();
            }
        };
    close_optional_pipeline_dispatch(
        "luisa_ray_query_pipeline_dispatch");
    close_optional_pipeline_dispatch(
        "luisa_pipeline_ray_query_dispatch_compact");
    close_optional_pipeline_dispatch(
        "luisa_pipeline_ray_query_dispatch_compact_object_ray");

    // The embedded wrapper must retain support for every curve basis, but each
    // generated kernel only needs the bases declared by its trace operations.
    // Turning the externally mutable wrapper mask into a constant here lets the
    // regular optimization pipeline eliminate unreachable curve branches. In
    // particular, triangle-only static trace kernels no longer inline the full
    // software curve intersector into both trace_closest and trace_any.
    {
        constexpr auto supported_curve_basis_mask = (1u << curve_basis_count) - 1u;
        auto curve_basis_mask = _config.curve_bases.to_u64();
        LUISA_ASSERT((curve_basis_mask & ~supported_curve_basis_mask) == 0u,
                     "Unsupported HIP curve basis mask 0x{:x}.", curve_basis_mask);
        if (auto *gv = _llvm_module->getGlobalVariable("luisa_hiprt_curve_basis_mask")) {
            LUISA_ASSERT(gv->getValueType()->isIntegerTy(32),
                         "Invalid HIPRT curve-basis mask type.");
            gv->setInitializer(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(_llvm_context), curve_basis_mask));
            gv->setConstant(true);
            gv->setExternallyInitialized(false);
            gv->setLinkage(llvm::GlobalValue::InternalLinkage);
        } else {
            LUISA_ERROR_WITH_LOCATION(
                "HIPRT wrapper is missing the curve-basis specialization mask.");
        }
    }

    // Step 2: Replace the extern __shared__ traversal-stack declaration with a
    // sized definition based on the actual kernel block size. Size from linked
    // traversal calls rather than conservative module capability flags: a
    // native-only or private-stack kernel must not reserve an unused hardware
    // frontier.
    {
        auto block_size = _config.block_size[0] * _config.block_size[1] * _config.block_size[2];
        LUISA_ASSERT(block_size > 0u, "Block size must be greater than zero.");
        uint32_t shared_array_size = 0u;
        auto *generic_hw_stack_dummy = _llvm_module->getFunction(
            "luisa_amdgcn_ds_bvh_stack_push8_pop1_rtn");
        auto *pipeline_hw_stack_dummy = _llvm_module->getFunction(
            "luisa_pipeline_amdgcn_ds_bvh_stack_push8_pop1_rtn");
        auto has_calls = [](llvm::Function *function) noexcept {
            return function != nullptr &&
                   std::any_of(
                       function->user_begin(), function->user_end(),
                       [](auto user) noexcept {
                           return llvm::isa<llvm::CallInst>(user);
                       });
        };
        const auto has_generic_hardware_stack_calls =
            has_calls(generic_hw_stack_dummy);
        const auto has_pipeline_hardware_stack_calls =
            has_calls(pipeline_hw_stack_dummy);
        const auto has_hardware_stack_calls =
            has_generic_hardware_stack_calls ||
            has_pipeline_hardware_stack_calls;
        if (_uses_hardware_rt_stack && has_hardware_stack_calls) {
            // A generic HIPRT stack owns disjoint TLAS/BLAS regions. A
            // resumable query needs nine entries per region; an ordinary
            // one-shot trace needs eight. The synchronous instance protocol
            // owns one 16-entry region and uses a separate intrinsic symbol.
            constexpr auto hardware_stack_lane_count = 32u;
            constexpr auto generic_hardware_stack_region_count = 2u;
            constexpr auto pipeline_hardware_stack_region_count = 1u;
            const auto generic_hardware_stack_max_entries =
                _uses_resumable_hardware_ray_query_pipeline ? 9u : 8u;
            constexpr auto pipeline_hardware_stack_max_entries = 16u;
            const auto generic_dwords_per_wave32 =
                has_generic_hardware_stack_calls ?
                    generic_hardware_stack_max_entries *
                        hardware_stack_lane_count *
                        generic_hardware_stack_region_count :
                    0u;
            const auto pipeline_dwords_per_wave32 =
                has_pipeline_hardware_stack_calls ?
                    pipeline_hardware_stack_max_entries *
                        hardware_stack_lane_count *
                        pipeline_hardware_stack_region_count :
                    0u;

            // Let F_r be the per-wave LDS footprint of route r and
            // S=max_r(F_r). Wave w owns [w*S,(w+1)*S), while every route uses
            // only its prefix [w*S,w*S+F_r). Hence routes may overlay within a
            // non-reentrant lane, but distinct waves remain disjoint even when
            // they execute different routes concurrently. Nested traversal is
            // already excluded from the hardware plan by RT analysis.
            const auto common_dwords_per_wave32 = std::max(
                generic_dwords_per_wave32,
                pipeline_dwords_per_wave32);
            LUISA_ASSERT(common_dwords_per_wave32 != 0u,
                         "HIP hardware stack has no reachable layout.");
            const auto num_waves = (block_size + 31u) / 32u;
            shared_array_size = num_waves * common_dwords_per_wave32;

            auto specialize_u32_global = [&](llvm::StringRef name,
                                             uint32_t value,
                                             bool required) noexcept {
                auto *global = _llvm_module->getGlobalVariable(name);
                if (global == nullptr) {
                    LUISA_ASSERT(!required,
                                 "HIPRT wrapper is missing required hardware "
                                 "stack layout constant '{}'.",
                                 name.str());
                    return;
                }
                auto *i32_type = llvm::Type::getInt32Ty(_llvm_context);
                LUISA_ASSERT(global->getValueType() == i32_type,
                             "Invalid HIPRT hardware stack layout constant "
                             "type for '{}'.",
                             name.str());
                auto *replacement = new llvm::GlobalVariable(
                    *_llvm_module, i32_type, true,
                    llvm::GlobalValue::InternalLinkage,
                    llvm::ConstantInt::get(i32_type, value),
                    llvm::Twine{name} + ".specialized",
                    nullptr, llvm::GlobalValue::NotThreadLocal, 0u);
                global->replaceAllUsesWith(replacement);
                global->eraseFromParent();
                replacement->setName(name);
            };
            specialize_u32_global(
                "luisa_hiprt_hw_stack_dwords_per_wave32",
                common_dwords_per_wave32, true);
            specialize_u32_global(
                "luisa_hiprt_hw_stack_max_entries",
                generic_hardware_stack_max_entries,
                has_generic_hardware_stack_calls);

            // dummy: <2 x i32> (i32, i32, <8 x i32>) ->
            // real: {i32, i32} (i32, i32, <8 x i32>, i32 immarg)
            auto replace_dummy_calls = [&](llvm::Function *dummy,
                                           uint32_t max_entries,
                                           llvm::StringRef route) noexcept {
                if (dummy == nullptr) { return 0u; }
                auto *i32_type = llvm::Type::getInt32Ty(_llvm_context);
                auto *intrinsic = llvm::Intrinsic::getOrInsertDeclaration(
                    _llvm_module.get(),
                    llvm::Intrinsic::amdgcn_ds_bvh_stack_push8_pop1_rtn);
                auto *immarg = llvm::ConstantInt::get(
                    i32_type, max_entries);
                llvm::SmallVector<llvm::CallInst *, 16> calls;
                for (auto *user : dummy->users()) {
                    if (auto *call = llvm::dyn_cast<llvm::CallInst>(user)) {
                        calls.emplace_back(call);
                    }
                }
                for (auto *call : calls) {
                    IB builder{call};
                    auto *result = builder.CreateCall(
                        intrinsic,
                        {call->getArgOperand(0), call->getArgOperand(1),
                         call->getArgOperand(2), immarg});
                    auto *first = builder.CreateExtractValue(result, {0});
                    auto *second = builder.CreateExtractValue(result, {1});
                    llvm::Value *vector = llvm::UndefValue::get(
                        llvm::FixedVectorType::get(i32_type, 2));
                    vector = builder.CreateInsertElement(
                        vector, first, builder.getInt32(0));
                    vector = builder.CreateInsertElement(
                        vector, second, builder.getInt32(1));
                    call->replaceAllUsesWith(vector);
                    call->eraseFromParent();
                }
                if (dummy->use_empty()) { dummy->eraseFromParent(); }
                LUISA_INFO(
                    "Replaced {} {} ds_bvh_stack dummy call(s) with "
                    "intrinsic (MaxStackEntries={}).",
                    calls.size(), route.str(), max_entries);
                return static_cast<uint32_t>(calls.size());
            };
            static_cast<void>(replace_dummy_calls(
                generic_hw_stack_dummy,
                generic_hardware_stack_max_entries, "generic"));
            static_cast<void>(replace_dummy_calls(
                pipeline_hw_stack_dummy,
                pipeline_hardware_stack_max_entries, "pipeline"));
        }
        if (!_uses_hardware_rt_stack) {
            // Pre-gfx12 and reentrant traversal use HIPRT's generic dynamic
            // stack through the historical shared-cache symbol.
            shared_array_size =
                LUISA_HIPRT_DYNAMIC_SHARED_STACK_SIZE * block_size;
        }
        if (_uses_static_global_rt_stack) {
            // The native closest path and an exact gfx12 query may coexist in
            // one kernel but execute sequentially. Reserve the larger LDS
            // requirement rather than adding two mutually exclusive stacks.
            shared_array_size = std::max(
                shared_array_size,
                LUISA_HIPRT_GLOBAL_SHARED_STACK_SIZE * block_size);
        }
        if (auto old_gv = _llvm_module->getGlobalVariable(
                "luisa_hiprt_shared_stack_cache")) {
            LUISA_ASSERT(shared_array_size > 0u,
                         "Shared traversal stack has zero size.");
            auto i32_ty = llvm::Type::getInt32Ty(_llvm_context);
            auto array_ty = llvm::ArrayType::get(
                i32_ty, shared_array_size);
            auto new_gv = new llvm::GlobalVariable(
                *_llvm_module, array_ty, false,
                llvm::GlobalValue::InternalLinkage,
                llvm::UndefValue::get(array_ty),
                "luisa_hiprt_shared_stack_cache.tmp",
                nullptr,
                llvm::GlobalValue::NotThreadLocal,
                3u);// addrspace(3) = shared/LDS
            new_gv->setAlignment(llvm::Align(4));
            LUISA_INFO(
                "Replacing shared traversal stack: {} uses, "
                "old type = [0 x i32], new type = [{} x i32]",
                old_gv->getNumUses(), shared_array_size);
            old_gv->replaceAllUsesWith(new_gv);
            old_gv->eraseFromParent();
            new_gv->setName("luisa_hiprt_shared_stack_cache");
        } else {
            // LinkOnlyNeeded plus global DCE removes the extern LDS symbol when
            // every reachable traversal wrapper uses a private stack (notably
            // direct motion closest/any tracing). That is a valid zero-LDS RT
            // kernel, not a broken wrapper link.
            LUISA_VERBOSE("HIPRT kernel uses no shared traversal stacks.");
        }
    }

    // Step 3: Provide trivial intersectFunc/filterFunc definitions.
    // HIPRT library calls these for custom geometry/filter callbacks.
    // We use numGeomTypes=0, numRayTypes=1, funcNameSets=nullptr, so both always return false.
    {
        for (auto &func : *_llvm_module) {
            if (!func.isDeclaration()) { continue; }
            auto name = func.getName();
            if (!name.contains("intersectFunc") && !name.contains("filterFunc")) { continue; }
            if (func.getReturnType() != llvm::Type::getInt1Ty(_llvm_context)) { continue; }
            auto *entry_bb = llvm::BasicBlock::Create(_llvm_context, "entry", &func);
            IB builder{entry_bb};
            builder.CreateRet(builder.getFalse());
            LUISA_INFO("Provided trivial definition for HIPRT function: {}", name.str());
        }
    }
}

void HIPCodegenLLVMImpl::_dump_module(const std::filesystem::path &path) const noexcept {
    std::error_code ec;
    llvm::raw_fd_ostream out{path.string(), ec};
    if (ec) {
        LUISA_WARNING_WITH_LOCATION("Failed to open file for dumping LLVM module: {}.", ec.message());
    } else {
        _llvm_module->print(out, nullptr);
    }
}

void HIPCodegenLLVMImpl::_run_optimization_passes() noexcept {

    if (_config.enable_fast_math) {
        for (auto &f : *_llvm_module) {
            for (auto &bb : f) {
                for (auto &inst : bb) {
                    if (llvm::isa<llvm::FPMathOperator>(inst)) {
                        inst.setFast(true);
                    }
                }
            }
        }
    }

    // Legacy HIPRT ray queries pass private storage through a generic pointer
    // and access full hiprtHit objects through multiple typed views. Strip TBAA
    // and widen wrapper effects there to prevent incorrect interprocedural
    // alias conclusions. gfx12 keeps its compact state in addrspace(5) across
    // the whole wrapper ABI, so its LuisaRayQueryStateHw TBAA is consistent and
    // useful; retaining it lets DSE/CSE remove redundant scratch traffic.
    if (_rt_analysis.uses_ray_query && !_uses_hardware_rt_stack) {
        uint32_t tbaa_count = 0, tbaa_struct_count = 0;
        for (auto &func : *_llvm_module) {
            for (auto &bb : func) {
                for (auto &inst : bb) {
                    if (inst.getMetadata(llvm::LLVMContext::MD_tbaa)) {
                        inst.setMetadata(llvm::LLVMContext::MD_tbaa, nullptr);
                        tbaa_count++;
                    }
                    if (inst.getMetadata(llvm::LLVMContext::MD_tbaa_struct)) {
                        inst.setMetadata(llvm::LLVMContext::MD_tbaa_struct, nullptr);
                        tbaa_struct_count++;
                    }
                }
            }
        }
        LUISA_INFO("Stripped {} TBAA and {} TBAA_STRUCT metadata entries from module.",
                   tbaa_count, tbaa_struct_count);

        // Widen memory effects of ray query wrapper functions to prevent the
        // optimizer from performing interprocedural dead-store/load elimination.
        // The HIP compiler infers precise memory attributes (e.g. memory(argmem: read))
        // on the compiled wrapper functions, which - combined with nosync/willreturn/
        // norecurse - allows LLVM to prove certain reads are dead and eliminate calls.
        // Setting memory(readwrite) and removing purity-related attributes forces the
        // optimizer to assume these functions may have arbitrary side effects.
        for (auto &func : *_llvm_module) {
            auto name = func.getName();
            if (name.starts_with("luisa_ray_query_") ||
                name.starts_with("luisa_motion_ray_query_") ||
                name.starts_with("luisa_pipeline_ray_query_")) {
                func.setMemoryEffects(llvm::MemoryEffects::unknown());
                func.removeFnAttr(llvm::Attribute::NoSync);
                func.removeFnAttr(llvm::Attribute::WillReturn);
                func.removeFnAttr(llvm::Attribute::NoRecurse);
                func.removeFnAttr(llvm::Attribute::MustProgress);
                func.removeFnAttr(llvm::Attribute::NoFree);
                func.removeFnAttr(llvm::Attribute::ReadOnly);
                func.removeFnAttr(llvm::Attribute::ReadNone);
                // Also strip readonly/readnone from parameters to prevent
                // the optimizer from inferring restricted aliasing on the pointer arg.
                for (unsigned i = 0; i < func.arg_size(); ++i) {
                    func.removeParamAttr(i, llvm::Attribute::ReadOnly);
                    func.removeParamAttr(i, llvm::Attribute::ReadNone);
                    func.removeParamAttr(i, llvm::Attribute::NoAlias);
                }
            }
        }
    }

    // Model generated callable expansion before assigning LLVM attributes.
    // Mutually exclusive call frontiers are discovered from the actual LLVM
    // CFG rather than inferred from source names or scene-specific counts.
    auto inline_graph = build_generated_callable_inline_graph(
        *_llvm_module, llvm_generated_callable_attribute);
    auto &generated_callables = inline_graph.functions;
    auto generated_callable_indices =
        llvm::DenseMap<const llvm::Function *, size_t>{};
    for (auto node_index = size_t{0u};
         node_index < generated_callables.size(); node_index++) {
        generated_callable_indices.try_emplace(
            generated_callables[node_index], node_index);
    }
    auto generated_callable_boundaries =
        select_generated_callable_boundaries(inline_graph.nodes);
    const auto dump_callable_boundaries =
        luisa::compute::detail::env_flag(
            "LUISA_HIP_DUMP_CALLABLE_BOUNDARIES");
    if (dump_callable_boundaries) {
        for (auto node_index = size_t{0u};
             node_index < generated_callables.size(); node_index++) {
            const auto &node = inline_graph.nodes[node_index];
            LUISA_INFO(
                "HIP generated callable '{}': instructions={}, "
                "calls={}, alternative_groups={}, preserve={}.",
                generated_callables[node_index]->getName().str(),
                node.instruction_count,
                node.callees.size(),
                node.alternative_call_groups.size(),
                generated_callable_boundaries[node_index] != 0u);
        }
    }

    for (auto &&func : *_llvm_module) {
        if (!func.isDeclaration() && func.getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL) {
            func.setLinkage(llvm::Function::PrivateLinkage);
            // Legacy mutating ray-query wrappers must remain call barriers. They
            // write through a generic/flat pointer while the kernel owns the state
            // in private address space, and exposing both sides to
            // InferAddressSpaces has caused invalid non-alias conclusions. gfx12
            // keeps the ABI in addrspace(5), so its scalar accessors and the small
            // surface-commit helper are safe to inline. Keep initialize/proceed and
            // the less common mutations out of line to contain register pressure.
            auto name = func.getName();
            auto is_ray_query_wrapper =
                name.starts_with("luisa_ray_query_") ||
                name.starts_with("luisa_motion_ray_query_") ||
                name.starts_with("luisa_pipeline_ray_query_");
            auto is_stack_overflow_fallback =
                name.starts_with(
                    "luisa_hiprt_stack_overflow_fallback_");
            auto is_generated_callable =
                func.hasFnAttribute(
                    llvm_generated_callable_attribute);
            if (is_generated_callable) {
                // Luisa Callable is an intentional DSL/JIT-stage function
                // boundary. LLVM gives a large bonus to single-call-site local
                // functions, which can otherwise inline mutually exclusive
                // generated alternatives into one enormous caller. Bound both
                // individual bodies and formally modeled alternative expansion;
                // small linear call graphs still use LLVM's ordinary cost model.
                func.removeFnAttr(llvm::Attribute::AlwaysInline);
                const auto index =
                    generated_callable_indices.find(&func);
                if (index != generated_callable_indices.end() &&
                    generated_callable_boundaries[index->second]) {
                    func.addFnAttr(llvm::Attribute::NoInline);
                } else {
                    func.removeFnAttr(llvm::Attribute::NoInline);
                }
            } else if (is_stack_overflow_fallback) {
                func.removeFnAttr(llvm::Attribute::AlwaysInline);
                func.addFnAttr(llvm::Attribute::NoInline);
            } else if (is_ray_query_wrapper) {
                auto is_pipeline_wrapper =
                    name.starts_with("luisa_pipeline_ray_query_");
                auto is_pipeline_initialize =
                    name == "luisa_pipeline_ray_query_initialize";
                auto is_pipeline_trace =
                    name.starts_with("luisa_pipeline_ray_query_trace_");
                auto is_inline_wrapper =
                    is_pipeline_wrapper ?
                        !is_pipeline_initialize &&
                            !is_pipeline_trace :
                        _uses_hardware_rt_stack &&
                            (name == "luisa_ray_query_state" ||
                             name == "luisa_ray_query_advance" ||
                             name == "luisa_ray_query_commit_surface_hit" ||
                             name.starts_with("luisa_ray_query_is_") ||
                             name.starts_with("luisa_ray_query_candidate_") ||
                             (name.starts_with("luisa_ray_query_committed_") &&
                              name != "luisa_ray_query_committed_hit") ||
                             name.starts_with("luisa_ray_query_ray_"));
                if (is_inline_wrapper) {
                    func.removeFnAttr(llvm::Attribute::NoInline);
                    func.addFnAttr(llvm::Attribute::AlwaysInline);
                } else if (is_pipeline_initialize || is_pipeline_trace) {
                    // Synchronous query construction and traversal each have
                    // one generated call site per query operation. Their size
                    // and scalarized state vary with the handler, so a blanket
                    // ABI barrier is not a valid profitability model. Preserve
                    // neither directive and let the target-aware inliner decide
                    // after handler projection and constant specialization.
                    func.removeFnAttr(llvm::Attribute::AlwaysInline);
                    func.removeFnAttr(llvm::Attribute::NoInline);
                } else {
                    func.removeFnAttr(llvm::Attribute::AlwaysInline);
                    func.addFnAttr(llvm::Attribute::NoInline);
                }
            } else if (func.hasFnAttribute(llvm::Attribute::Cold)) {
                // A cold helper represents a source-level path-frequency
                // proof, not a mandatory ABI boundary. Do not override that
                // proof with the blanket wrapper AlwaysInline policy; LLVM's
                // cost model remains free to inline it when profitable.
                func.removeFnAttr(llvm::Attribute::AlwaysInline);
                func.removeFnAttr(llvm::Attribute::NoInline);
            } else {
                func.addFnAttr(llvm::Attribute::AlwaysInline);
            }
        }
    }

    // Resolve aliases to actual functions so they get PrivateLinkage + AlwaysInline.
    // HIPRT bitcode has C++ ctor/dtor delegation aliases (C1→C2, D1→D2) which the
    // function iterator above doesn't visit — leaving 18 functions un-inlined.
    {
        llvm::SmallVector<llvm::GlobalAlias *, 32> aliases_to_resolve;
        for (auto &alias : _llvm_module->aliases()) {
            aliases_to_resolve.push_back(&alias);
        }
        for (auto *alias : aliases_to_resolve) {
            auto *aliasee = alias->getAliasee();
            auto *fn = llvm::dyn_cast<llvm::Function>(aliasee);
            if (!fn) { continue; }
            auto *new_fn = llvm::Function::Create(
                fn->getFunctionType(), llvm::Function::PrivateLinkage,
                alias->getName() + ".resolved", _llvm_module.get());
            new_fn->copyAttributesFrom(fn);
            new_fn->setLinkage(llvm::Function::PrivateLinkage);
            new_fn->addFnAttr(llvm::Attribute::AlwaysInline);
            auto *entry = llvm::BasicBlock::Create(_llvm_context, "entry", new_fn);
            IB builder{entry};
            llvm::SmallVector<llvm::Value *, 8> args;
            for (auto &arg : new_fn->args()) {
                args.push_back(&arg);
            }
            auto *call = builder.CreateCall(fn, args);
            call->setCallingConv(fn->getCallingConv());
            call->setTailCall(true);
            if (fn->getReturnType()->isVoidTy()) {
                builder.CreateRetVoid();
            } else {
                builder.CreateRet(call);
            }
            alias->replaceAllUsesWith(new_fn);
            alias->eraseFromParent();
        }
    }

    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;

    llvm::PipelineTuningOptions PTO;
    PTO.LoopInterleaving = true;
#if LLVM_VERSION_MAJOR >= 21
    PTO.LoopInterchange = true;
#endif
    PTO.LoopVectorization = true;
    PTO.SLPVectorization = true;
    PTO.LoopUnrolling = true;
    PTO.MergeFunctions = true;
    llvm::PassInstrumentationCallbacks instrumentation;
    llvm::PassBuilder PB{
        _target_machine, PTO, std::nullopt,
        &instrumentation};
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

#if LLVM_VERSION_MAJOR >= 19
    _target_machine->registerPassBuilderCallbacks(PB);
#else
    _target_machine->registerPassBuilderCallbacks(PB, true);
#endif

    auto opt_level = llvm::OptimizationLevel::O2;
    switch (_config.opt_level) {
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_NONE: opt_level = llvm::OptimizationLevel::O0; break;
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_LESS: opt_level = llvm::OptimizationLevel::O1; break;
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_DEFAULT: opt_level = llvm::OptimizationLevel::O2; break;
        case HIPCodegenLLVMConfig::OptLevel::LEVEL_AGGRESSIVE: opt_level = llvm::OptimizationLevel::O3; break;
    }
    auto MPM = PB.buildPerModuleDefaultPipeline(opt_level);
    // gfx12 ray queries lower to resumable traversal loops whose state is
    // carried through nested callback loops. LLVM's pre-SLP cleanup normally
    // opts out of preserving canonical loops; on AMDGPU this can collapse the
    // unique latches into a multi-latch CFG that the downstream structurizer
    // miscompiles. Keep the exact default pipeline, but retain canonical loop
    // form at that one stage for hardware ray-query modules.
    auto preserve_hardware_ray_query_loops =
        _uses_hardware_rt_stack && _rt_analysis.uses_ray_query &&
        (_config.opt_level == HIPCodegenLLVMConfig::OptLevel::LEVEL_DEFAULT ||
         _config.opt_level == HIPCodegenLLVMConfig::OptLevel::LEVEL_AGGRESSIVE);
    if (preserve_hardware_ray_query_loops) {
        auto pipeline = std::string{};
        auto stream = llvm::raw_string_ostream{pipeline};
        MPM.printPipeline(
            stream,
            [&instrumentation](llvm::StringRef class_name) noexcept {
                return instrumentation.getPassNameForClassName(class_name);
            });
        stream.flush();
        auto replacement_count =
            preserve_hardware_ray_query_loop_form(pipeline);
        LUISA_ASSERT(replacement_count == 1u,
                     "Expected exactly one non-canonical loop optimization "
                     "stage in the HIP LLVM pipeline, found {}.",
                     replacement_count);
        auto canonical_mpm = llvm::ModulePassManager{};
        if (auto error = PB.parsePassPipeline(canonical_mpm, pipeline)) {
            LUISA_ERROR_WITH_LOCATION(
                "Failed to rebuild the canonical-loop HIP LLVM pipeline: {}.",
                llvm::toString(std::move(error)));
        }
        MPM = std::move(canonical_mpm);
    }
    MPM.run(*_llvm_module, MAM);

    // make hiprt/hiprtc happy
    // Resolve by the stable IR spelling instead of referring to the generated
    // C++ enum name. Downstream LLVM branches can recognize an attribute in
    // bitcode while omitting its named enumerator from their public headers.
    // The resolved AttrKind is still required: LLVM's StringRef removal
    // overload only removes target-dependent string attributes, not enum
    // attributes with the same spelling.
    constexpr auto no_create_undef_or_poison =
        "nocreateundeforpoison";
    const auto no_create_undef_or_poison_kind =
        llvm::Attribute::getAttrKindFromName(
            no_create_undef_or_poison);
    auto remove_no_create_undef_or_poison =
        [no_create_undef_or_poison_kind](
            llvm::AttributeList attrs,
            llvm::LLVMContext &context) noexcept {
            return no_create_undef_or_poison_kind !=
                           llvm::Attribute::None ?
                       attrs.removeAttributeAtIndex(
                           context,
                           llvm::AttributeList::FunctionIndex,
                           no_create_undef_or_poison_kind) :
                       attrs.removeAttributeAtIndex(
                           context,
                           llvm::AttributeList::FunctionIndex,
                           "nocreateundeforpoison");
        };
    for (auto &func : *_llvm_module) {
        auto attrs = remove_no_create_undef_or_poison(
            func.getAttributes(), func.getContext());
        func.setAttributes(attrs);
        for (auto &bb : func) {
            for (auto &inst : bb) {
                if (auto *cb = llvm::dyn_cast<llvm::CallBase>(&inst)) {
                    auto cb_attrs = remove_no_create_undef_or_poison(
                        cb->getAttributes(), cb->getContext());
                    cb->setAttributes(cb_attrs);
                }
            }
        }
    }
}

luisa::string HIPCodegenLLVMImpl::_generate_code() const noexcept {
    std::string code;
    llvm::raw_string_ostream os{code};
    llvm::WriteBitcodeToFile(*_llvm_module, os);
    os.flush();
    return luisa::string{code};
}

luisa::string HIPCodegenLLVMImpl::generate(const xir::Module &xir_module) noexcept {
    Clock clk;
    _rt_analysis = {};
    _uses_iterative_synchronous_ray_query_pipeline = false;
    _uses_resumable_hardware_ray_query_pipeline = false;
    _uses_native_closest_ray_query_pipeline = false;
    _uses_native_effect_only_ray_query_pipeline = false;
    _uses_static_global_rt_stack = false;
    _native_closest_reduction_pipelines.clear();
    _analyze_ray_tracing_usage(xir_module);
    // AST-derived flags are conservative: optimization may have removed the
    // last reachable operation, but the serialized shader metadata still uses
    // these flags and therefore must observe the same kernel argument ABI.
    _rt_analysis.uses_ray_tracing |=
        _config.requires_ray_tracing || _config.requires_ray_query;
    _rt_analysis.uses_ray_query |= _config.requires_ray_query;
    _rt_analysis.uses_motion_blur |= _config.requires_motion_blur;
    _rt_analysis.uses_static_trace |= _config.requires_static_trace;
    _rt_analysis.uses_motion_ray_query |= _config.requires_motion_ray_query;
    _rt_analysis.uses_ray_tracing |= _rt_analysis.uses_motion_blur;
    if (!_config.resumable_ray_query_state_functions.empty()) {
        llvm::DenseSet<const xir::Function *> module_functions;
        for (auto function : xir_module.function_list()) {
            module_functions.insert(function);
        }
        LUISA_ASSERT(
            std::all_of(
                _config.resumable_ray_query_state_functions.begin(),
                _config.resumable_ray_query_state_functions.end(),
                [&](auto function) noexcept {
                    return module_functions.contains(function);
                }),
            "HIP mixed RayQuery retry referenced a function outside the "
            "immutable XIR module used by the first translation.");
    }
    _initialize();

    _collect_print_info(xir_module);

    for (auto f : xir_module.function_list()) {
        if (auto def = f->definition()) {
            static_cast<void>(_translate_function(def));
        }
    }
    auto ray_query_projection =
        _finalize_ray_query_pipeline_contexts();
    if ((_uses_native_closest_ray_query_pipeline ||
         _uses_native_effect_only_ray_query_pipeline) &&
        _config.max_register_count == 0u) {
        // The ordinary synchronous frontier is latency-bound and benefits from
        // a high occupancy target. A native closest callback instead contains
        // the complete user intersection/filter reduction; forcing that same
        // target spills its larger live set. Match ordinary HIPRT closest
        // traversal and let the AMDGPU allocator choose the resource balance.
        auto llvm_kernel = _llvm_module->getFunction(
            llvm::StringRef{_config.entry_point.data(),
                            _config.entry_point.size()});
        LUISA_ASSERT(llvm_kernel != nullptr,
                     "Missing HIP kernel while selecting native callback "
                     "RayQuery resources.");
        llvm_kernel->removeFnAttr("amdgpu-waves-per-eu");
    }
    if (_uses_synchronous_ray_query_pipeline &&
        (!ray_query_projection.exact_state_required_functions.empty() ||
         !ray_query_projection
              .oversized_budget_constrained_state_functions.empty())) {
        _retry_with_resumable_ray_query_state_functions =
            ray_query_projection.exact_state_required_functions;
        for (auto function : ray_query_projection
                                 .oversized_budget_constrained_state_functions) {
            if (std::find(
                    _retry_with_resumable_ray_query_state_functions.begin(),
                    _retry_with_resumable_ray_query_state_functions.end(),
                    function) ==
                _retry_with_resumable_ray_query_state_functions.end()) {
                _retry_with_resumable_ray_query_state_functions.emplace_back(
                    function);
            }
        }
        LUISA_VERBOSE(
            "HIP synchronous RayQuery plan rejected: maximum projected "
            "callback environment requiring an exact candidate or observable "
            "post-state transaction is {} bytes (budget={} bytes; overall "
            "maximum={} bytes); {} function domain(s) require simultaneous "
            "world/object ray states.",
            ray_query_projection
                .maximum_budget_constrained_context_bytes,
            hip_synchronous_ray_query_environment_budget,
            ray_query_projection.maximum_context_bytes,
            ray_query_projection.exact_state_required_functions.size());
        return {};
    }
    if (_uses_synchronous_ray_query_pipeline &&
        ray_query_projection.maximum_context_bytes >
            hip_synchronous_ray_query_environment_budget) {
        LUISA_VERBOSE(
            "HIP synchronous RayQuery retained {} handler-only pipeline(s) "
            "above the ordinary {}-byte callback budget (overall maximum={} "
            "bytes): their query post-state is unobservable and their "
            "handlers admit the compact candidate-action transaction.",
            ray_query_projection
                .oversized_compact_handler_only_pipeline_count,
            hip_synchronous_ray_query_environment_budget,
            ray_query_projection.maximum_context_bytes);
    }

    _link_native_include();
    for (auto f : xir_module.function_list()) {
        if (f->isa<xir::ExternalFunction>()) {
            auto llvm_f = _get_or_declare_llvm_function(f);
            LUISA_ASSERT(!llvm_f->isDeclaration(),
                         "HIP external function '{}' has no definition. "
                         "ShaderOption::native_include must contain matching LLVM IR/bitcode.",
                         f->name().value_or("<unnamed>"));
        }
    }
    _link_ockl_if_needed();
    _postprocess_rt_kernel();

    if (llvm::verifyModule(*_llvm_module, &llvm::errs())) {
        _dump_module("debug_bad_module.ll");
        LUISA_ERROR_WITH_LOCATION("Module verification failed.");
    }

    static auto dump_ir = [] {
        using namespace std::string_view_literals;
        auto env = getenv("LUISA_DUMP_LLVM_IR");
        return env != nullptr && env == "1"sv;
    }();
    static std::atomic<uint32_t> dump_counter{0u};
    uint32_t dump_idx = 0u;
    if (dump_ir) {
        dump_idx = dump_counter.fetch_add(1u);
        auto filename = fmt::format("hip_kernel_before_opt_{}.ll", dump_idx);
        _dump_module(filename);
        LUISA_INFO("Dumped LLVM IR to: {}", filename);
    }

    _run_optimization_passes();

    // IPO can prove every use of a private aggregate while still retaining a
    // dead self-address store that blocks generic capture tracking. Remove
    // only self-references whose complete constant-offset access relation has
    // no overlapping read, then expose the non-escaping aggregate to SROA.
    const auto private_memory_stats =
        optimize_hip_private_memory(*_llvm_module, _target_machine);
    if (private_memory_stats.eliminated_self_reference_stores != 0u) {
        LUISA_VERBOSE(
            "Eliminated {} dead HIP private self-reference store(s) across "
            "{} analyzed alloca(s), then reran scalar cleanup.",
            private_memory_stats.eliminated_self_reference_stores,
            private_memory_stats.analyzed_allocas);
    }

    // The synchronous native traversal remains one shared function, so IPO
    // sees its callback dispatcher through dynamic parameters even when every
    // surviving call site supplies constants. Specialize only the explicitly
    // marked tuple after IPO: at most one outlined body per distinct tuple
    // removes the selected switches without cloning that body into every
    // triangle/procedural candidate site; semantically equal bodies are merged
    // again afterwards. Parameters are opted in individually because removing
    // a branch can still worsen target register allocation and code layout.
    auto constant_dispatch_stats =
        specialize_marked_constant_integer_arguments(*_llvm_module);
    if (constant_dispatch_stats.rewritten_function_count != 0u) {
        LUISA_VERBOSE(
            "Specialized {} HIP constant dispatcher(s) into {} outlined "
            "body/bodies ({} identical clone(s) merged) at {} direct call "
            "site(s).",
            constant_dispatch_stats.rewritten_function_count,
            constant_dispatch_stats.cloned_function_count,
            constant_dispatch_stats.merged_clone_count,
            constant_dispatch_stats.rewritten_call_count);
    }

    // IPO has now selected the final generated-Callable boundaries, while
    // their internal marker and noinline attributes are still present. Narrow
    // aggregate ABIs here: no later IPO pass may widen the signatures again,
    // and the attribute cleanup below can still recognize the new functions.
    auto callable_abi_stats =
        specialize_generated_callable_aggregate_arguments(*_llvm_module);
    if (callable_abi_stats.rewritten_function_count != 0u) {
        LUISA_VERBOSE(
            "Specialized aggregate arguments for {} generated HIP "
            "callable(s), removing {} bytes from their direct call ABIs.",
            callable_abi_stats.rewritten_function_count,
            callable_abi_stats.removed_aggregate_bytes);
    }

    // GlobalISel demotes a return wider than RetCC_AMDGPU_Func's 32 VGPRs to
    // one hidden private frame object per static call site. Make that ABI
    // explicit after IPO and share the result storage within each caller.
    // Doing this here preserves all SSA optimization inside the callable while
    // preventing mutually exclusive polymorphic calls from accumulating a
    // linear amount of private memory.
    auto large_return_stats =
        demote_generated_callable_large_returns(*_llvm_module);
    if (large_return_stats.rewritten_function_count != 0u) {
        LUISA_VERBOSE(
            "Demoted {} large generated HIP callable return(s) at {} "
            "call site(s) into {} shared private result slot(s), moving {} "
            "return ABI bytes behind explicit storage.",
            large_return_stats.rewritten_function_count,
            large_return_stats.rewritten_call_count,
            large_return_stats.shared_result_slot_count,
            large_return_stats.demoted_return_bytes);
    }

    if (dump_ir) {
        auto after_opt_filename = fmt::format("hip_kernel_after_opt_{}.ll", dump_idx);
        _dump_module(after_opt_filename);
        LUISA_INFO("Dumped post-optimization LLVM IR to: {}", after_opt_filename);
    }

    auto target_cpu = _target_machine->getTargetCPU();
    auto target_features = _target_machine->getTargetFeatureString();
    auto max_vgpr_count = std::min(_config.max_register_count, 256u);
    auto max_vgpr_count_string = std::to_string(max_vgpr_count);
    for (auto &func : *_llvm_module) {
        // Preserve every function boundary deliberately retained by the
        // module optimizer across the ABI-attribute cleanup below. This
        // includes large shared DSL callables and mutating ray-query wrappers.
        // In particular, the gfx12 proceed helper is a large resumable
        // traversal state machine; letting the downstream compiler inline it
        // causes severe register pressure and scratch spills.
        auto name = func.getName();
        auto preserve_generated_callable =
            func.hasFnAttribute(
                llvm_generated_callable_attribute);
        auto preserve_noinline =
            preserve_generated_callable ||
            ((name.starts_with("luisa_ray_query_") ||
              name.starts_with("luisa_motion_ray_query_") ||
              name.starts_with(
                  "luisa_hiprt_stack_overflow_fallback_")) &&
             func.hasFnAttribute(llvm::Attribute::NoInline));
        auto preserve_convergent = preserve_noinline &&
                                   func.hasFnAttribute(llvm::Attribute::Convergent);

        const auto is_generated_kernel =
            func.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL &&
            func.getName() == _config.entry_point;

        // Collect amdgpu-no-* string attributes from the generated kernel
        // before stripping. These are added by AMDGPUAttributor and are
        // critical for correct kernarg segment layout: without them, the
        // AMDGPU backend assumes 256 bytes of implicit arguments, which can
        // cause memory faults.
        llvm::SmallVector<llvm::StringRef, 24> amdgpu_no_attrs;
        llvm::SmallVector<std::pair<llvm::StringRef, llvm::StringRef>, 8> amdgpu_codegen_attrs;
        if (is_generated_kernel) {
            for (auto &attr : func.getAttributes().getFnAttrs()) {
                if (attr.isStringAttribute()) {
                    auto key = attr.getKindAsString();
                    if (key.starts_with("amdgpu-no-")) {
                        amdgpu_no_attrs.push_back(key);
                    } else if (key == "amdgpu-waves-per-eu" ||
                               key == "amdgpu-flat-work-group-size" ||
                               key == "amdgpu-unsafe-fp-atomics" ||
                               key == "amdgpu-num-vgpr" ||
                               key == "amdgpu-num-sgpr") {
                        amdgpu_codegen_attrs.emplace_back(key, attr.getValueAsString());
                    }
                }
            }
        }

        func.setAttributes(llvm::AttributeList{});

        if (is_generated_kernel) {
            func.addFnAttr(llvm::Attribute::NoInline);
            for (auto &attr_name : amdgpu_no_attrs) {
                func.addFnAttr(attr_name);
            }
            for (auto &[key, val] : amdgpu_codegen_attrs) {
                func.addFnAttr(key, val);
            }
        } else if (preserve_noinline) {
            func.addFnAttr(llvm::Attribute::NoInline);
            if (preserve_convergent) {
                func.addFnAttr(llvm::Attribute::Convergent);
            }
        }
        // Re-add target CPU and features so that downstream consumers
        // (e.g., HIPRT bitcode compiler) know the GPU architecture.
        if (!func.isDeclaration()) {
            func.addFnAttr("target-cpu", target_cpu);
            if (!target_features.empty()) {
                func.addFnAttr("target-features", target_features);
            }
            // ShaderOption::max_registers is a whole-shader constraint. Apply
            // it to the complete device call graph, including linked HIPRT
            // helpers, because an unconstrained callee determines the kernel's
            // actual VGPR allocation just as much as the root kernel does.
            if (max_vgpr_count != 0u) {
                func.addFnAttr("amdgpu-num-vgpr", max_vgpr_count_string);
            }
        }
    }

    // This is the second and final verifier boundary for HIP LLVM codegen. It
    // covers both the ordinary optimization pipeline and the callable ABI
    // rewrite above without repeatedly verifying between individual passes.
    if (llvm::verifyModule(*_llvm_module, &llvm::errs())) {
        _dump_module("debug_bad_module_after_opt.ll");
        LUISA_ERROR_WITH_LOCATION(
            "Post-optimization module verification failed.");
    }

    if (dump_ir) {
        auto final_filename = fmt::format(
            "hip_kernel_final_{}.ll", dump_idx);
        _dump_module(final_filename);
        LUISA_INFO("Dumped final LLVM IR to: {}", final_filename);
    }

    static auto print_ir = [] {
        using namespace std::string_view_literals;
        auto env = getenv("LUISA_PRINT_LLVM_IR");
        return env != nullptr && env == "1"sv;
    }();
    if (print_ir) {
        _llvm_module->print(llvm::outs(), nullptr);
    }

    LUISA_INFO_WITH_LOCATION("HIP LLVM codegen completed in {} ms.", clk.toc());
    return _generate_code();
}

luisa::vector<std::pair<luisa::string, luisa::string>>
HIPCodegenLLVMImpl::take_print_formats() && noexcept {
    luisa::vector<std::pair<luisa::string, luisa::string>> result;
    result.reserve(_print_formats.size());
    for (auto &&[format, type] : _print_formats) {
        result.emplace_back(std::move(format), type->description());
    }
    return result;
}

}// namespace luisa::compute::hip
