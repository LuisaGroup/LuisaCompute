#include "llvm_jit.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <pthread.h>
#endif

#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ObjectTransformLayer.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Verifier.h>
#include "../../common/llvm_pass_builder.h"
#include <llvm/Support/Error.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/Scalarizer.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Vectorize/SLPVectorizer.h>

#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>

#include "../../common/env_flag.h"

namespace luisa::compute::simd {

namespace {

// Module keeps a non-owning reference to its LLVMContext. Function parameters
// are otherwise destroyed in the opposite order of this public (module,
// context) API, so every early return must explicitly release the module
// first. Moved-from pointers make the successful ORC path a no-op here.
struct ModuleContextLifetime {
    std::unique_ptr<::llvm::Module> &module;
    std::unique_ptr<::llvm::LLVMContext> &context;

    ~ModuleContextLifetime() noexcept {
        module.reset();
        context.reset();
    }
};

// Large vectorized kernels can recurse deeply inside LLVM -O2 / codegen passes
// and exceed the 1 MB default main-thread stack on Windows (or 512 KB worker
// stacks on macOS). Run heavy compiler steps on a 16 MB stack when needed.
template<typename Fn>
void run_with_compiler_stack(Fn &&fn) noexcept {
    constexpr size_t compiler_stack_bytes = 16u * 1024u * 1024u;
    struct Context {
        std::remove_reference_t<Fn> *fn;
    } ctx{&fn};
#ifdef _WIN32
    auto proc = [](LPVOID param) noexcept -> DWORD {
        (*static_cast<Context *>(param)->fn)();
        return 0u;
    };
    HANDLE thread = ::CreateThread(
        nullptr, compiler_stack_bytes, proc, &ctx,
        STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
    if (thread != nullptr) {
        ::WaitForSingleObject(thread, INFINITE);
        ::CloseHandle(thread);
        return;
    }
    fn();
#else
    auto proc = [](void *param) noexcept -> void * {
        (*static_cast<Context *>(param)->fn)();
        return nullptr;
    };
    pthread_attr_t attr{};
    if (::pthread_attr_init(&attr) == 0) {
        ::pthread_attr_setstacksize(&attr, compiler_stack_bytes);
        pthread_t thread{};
        if (::pthread_create(&thread, &attr, proc, &ctx) == 0) {
            ::pthread_join(thread, nullptr);
            ::pthread_attr_destroy(&attr);
            return;
        }
        ::pthread_attr_destroy(&attr);
    }
    fn();
#endif
}

// Strip <1 x T> pseudo-vectors in W=1 functions after SROA/scalar opts and
// immediately before LoopVectorize/SLPVectorizer so LLVM SLP can pack
// independent scalar lanes (e.g. 8-corner 3D Perlin noise, Float3/Float4 math)
// into native SIMD registers on both x86-64 and AArch64.
struct ScalarizeW1FunctionPass
    : ::llvm::PassInfoMixin<ScalarizeW1FunctionPass> {
    ::llvm::PreservedAnalyses run(
        ::llvm::Function &function,
        ::llvm::FunctionAnalysisManager &am) noexcept {
        if (!function.getName().ends_with(".simd_w1") &&
            !function.getName().ends_with(".live_w1")) {
            return ::llvm::PreservedAnalyses::all();
        }
        bool has_v1 = false;
        bool has_wide = false;
        auto inspect_type = [&](::llvm::Type *ty) noexcept {
            if (auto *vt = ::llvm::dyn_cast<::llvm::FixedVectorType>(ty)) {
                auto n = vt->getNumElements();
                if (n == 1u) {
                    has_v1 = true;
                } else if (n > 1u) {
                    has_wide = true;
                }
            }
        };
        for (auto &&bb : function) {
            for (auto &&inst : bb) {
                inspect_type(inst.getType());
                for (auto &op : inst.operands()) {
                    inspect_type(op->getType());
                }
                if (has_wide) { break; }
            }
            if (has_wide) { break; }
        }
        if (!has_v1 || has_wide) {
            return ::llvm::PreservedAnalyses::all();
        }
        ::llvm::ScalarizerPassOptions opts{};
        opts.ScalarizeLoadStore = true;
        opts.ScalarizeVariableInsertExtract = true;
        ::llvm::ScalarizerPass scalarizer{opts};
        auto pa = scalarizer.run(function, am);
        if (pa.areAllPreserved()) { return pa; }
        am.invalidate(function, pa);
        ::llvm::InstCombinePass ic{};
        auto ic_pa = ic.run(function, am);
        am.invalidate(function, ic_pa);
        pa.intersect(ic_pa);
        ::llvm::SLPVectorizerPass slp{};
        auto slp_pa = slp.run(function, am);
        pa.intersect(slp_pa);
        return pa;
    }
};

}// namespace

void LLVMJIT::_fail(std::string message) noexcept {
    if (_error.empty()) { _error = std::move(message); }
}

LLVMJIT::LLVMJIT(bool capture_object) noexcept {
    if (::llvm::InitializeNativeTarget() ||
        ::llvm::InitializeNativeTargetAsmPrinter()) {
        _fail("failed to initialize the LLVM native target");
        return;
    }
    auto host = ::llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!host) {
        _fail("failed to detect the LLVM host target: " +
              ::llvm::toString(host.takeError()));
        return;
    }
    host->setCodeGenOptLevel(::llvm::CodeGenOptLevel::Aggressive);
    // LLJIT selects PIC for an unspecified relocation model. Make that
    // choice explicit before creating the audit TargetMachine as well, so
    // captured assembly and ORC's final relocatable object have identical
    // instruction layout.
    host->setRelocationModel(::llvm::Reloc::PIC_);
    host->setCodeModel(::llvm::CodeModel::Small);
    _target_triple = host->getTargetTriple().str();
    auto target_machine = host->createTargetMachine();
    if (!target_machine) {
        _fail("failed to create the LLVM host target machine: " +
              ::llvm::toString(target_machine.takeError()));
        return;
    }
    _target_machine = std::move(*target_machine);
    _target_machine->Options.MCOptions.AsmVerbose = true;
    ::llvm::orc::LLJITBuilder builder;
    const auto &triple = host->getTargetTriple();
    if (triple.isOSBinFormatCOFF() && triple.isX86()) {
        // RuntimeDyldCOFFX86_64 requires all sections referenced by 32-bit
        // image-base-relative relocations (IMAGE_REL_AMD64_ADDR32NB) to be
        // allocated in a single monotonically ordered address range.
        builder.setObjectLinkingLayerCreator(
            [](::llvm::orc::ExecutionSession &es, auto &&...)
                -> ::llvm::Expected<std::unique_ptr<::llvm::orc::ObjectLayer>> {
                auto layer = std::make_unique<::llvm::orc::RTDyldObjectLinkingLayer>(
                    es, [](const ::llvm::MemoryBuffer &) {
                        return std::make_unique<::llvm::SectionMemoryManager>(
                            nullptr, /*ReserveAlloc=*/true);
                    });
                layer->setOverrideObjectFlagsWithResponsibilityFlags(true);
                layer->setAutoClaimResponsibilityForObjectSymbols(true);
                return layer;
            });
    }
    builder.setJITTargetMachineBuilder(std::move(*host));
    auto jit = builder.create();
    if (!jit) {
        _fail("failed to create LLVM ORC JIT: " +
              ::llvm::toString(jit.takeError()));
        _target_machine.reset();
        return;
    }
    _jit = std::move(*jit);
    // Uniform inverse hyperbolic operations call scalar libm functions.
    // Static CRTs do not export these symbols for ORC's process lookup.
    ::llvm::orc::SymbolMap math_symbols;
    auto bind_math_symbol = [this, &math_symbols](
                                const char *name, auto *function) noexcept {
        math_symbols.try_emplace(
            _jit->mangleAndIntern(name),
            ::llvm::orc::ExecutorSymbolDef{
                ::llvm::orc::ExecutorAddr::fromPtr(function),
                ::llvm::JITSymbolFlags::Callable});
    };
    using FloatMath = float (*)(float);
    using DoubleMath = double (*)(double);
    using FloatBinaryMath = float (*)(float, float);
    using DoubleBinaryMath = double (*)(double, double);
    bind_math_symbol("asinhf", static_cast<FloatMath>(&std::asinh));
    bind_math_symbol("acoshf", static_cast<FloatMath>(&std::acosh));
    bind_math_symbol("atanhf", static_cast<FloatMath>(&std::atanh));
    bind_math_symbol("asinh", static_cast<DoubleMath>(&std::asinh));
    bind_math_symbol("acosh", static_cast<DoubleMath>(&std::acosh));
    bind_math_symbol("atanh", static_cast<DoubleMath>(&std::atanh));
    bind_math_symbol("log2f", static_cast<FloatMath>(&std::log2));
    bind_math_symbol("log2", static_cast<DoubleMath>(&std::log2));
    bind_math_symbol("exp2f", static_cast<FloatMath>(&std::exp2));
    bind_math_symbol("exp2", static_cast<DoubleMath>(&std::exp2));
    bind_math_symbol("log10f", static_cast<FloatMath>(&std::log10));
    bind_math_symbol("log10", static_cast<DoubleMath>(&std::log10));
    bind_math_symbol("fmodf", static_cast<FloatBinaryMath>(&std::fmod));
    bind_math_symbol("fmod", static_cast<DoubleBinaryMath>(&std::fmod));
    bind_math_symbol("powf", static_cast<FloatBinaryMath>(&std::pow));
    bind_math_symbol("pow", static_cast<DoubleBinaryMath>(&std::pow));
    if (auto error = _jit->getMainJITDylib().define(
            ::llvm::orc::absoluteSymbols(std::move(math_symbols)))) {
        _fail("failed to define LLVM JIT scalar math symbols: " +
              ::llvm::toString(std::move(error)));
        return;
    }
    if (capture_object) {
        _object = std::make_shared<std::string>();
        _jit->getObjTransformLayer().setTransform(
            [capture = _object](
                std::unique_ptr<::llvm::MemoryBuffer> object)
                -> ::llvm::Expected<
                    std::unique_ptr<::llvm::MemoryBuffer>> {
                auto buffer = object->getBuffer();
                capture->assign(buffer.data(), buffer.size());
                return std::move(object);
            });
    }
}

LLVMJIT::~LLVMJIT() noexcept = default;
LLVMJIT::LLVMJIT(LLVMJIT &&) noexcept = default;
LLVMJIT &LLVMJIT::operator=(LLVMJIT &&) noexcept = default;

bool LLVMJIT::selects_size_bounded_pipeline(
    size_t block_count, size_t instruction_count) noexcept {
    static constexpr auto max_o2_blocks = size_t{65536u};
    static constexpr auto max_o2_instructions = size_t{1048576u};
    return block_count >= max_o2_blocks ||
           instruction_count >= max_o2_instructions;
}

bool LLVMJIT::add_module(
    std::unique_ptr<::llvm::Module> module,
    std::unique_ptr<::llvm::LLVMContext> context) noexcept {
    ModuleContextLifetime lifetime{module, context};
    if (!succeeded()) { return false; }
    if (module == nullptr || context == nullptr ||
        &module->getContext() != context.get()) {
        _fail("LLVM JIT module and context ownership do not match");
        return false;
    }
    auto size_bounded = false;
    if (!_prepare_module(*module, &size_bounded)) { return false; }

    if (size_bounded) {
        ::llvm::SmallVector<char, 0u> object_storage;
        ::llvm::raw_svector_ostream object_stream{object_storage};
        ::llvm::legacy::PassManager object_codegen;
        _target_machine->setOptLevel(::llvm::CodeGenOptLevel::Less);
        if (_target_machine->addPassesToEmitFile(
                object_codegen, object_stream, nullptr,
                ::llvm::CodeGenFileType::ObjectFile)) {
            _target_machine->setOptLevel(::llvm::CodeGenOptLevel::Aggressive);
            _fail("LLVM host target cannot emit relocatable object");
            return false;
        }
        object_codegen.run(*module);
        _target_machine->setOptLevel(::llvm::CodeGenOptLevel::Aggressive);
        if (_object != nullptr) {
            _object->assign(object_storage.data(), object_storage.size());
        }
        auto object_buffer = ::llvm::MemoryBuffer::getMemBufferCopy(
            ::llvm::StringRef{object_storage.data(), object_storage.size()},
            module->getModuleIdentifier());
        if (auto error = _jit->addObjectFile(std::move(object_buffer))) {
            _fail("failed to add size-bounded object to ORC JIT: " +
                  ::llvm::toString(std::move(error)));
            return false;
        }
        return true;
    }

    if (auto error = _jit->addIRModule(::llvm::orc::ThreadSafeModule(
            std::move(module), std::move(context)))) {
        _fail("failed to add LLVM IR module to ORC JIT: " +
              ::llvm::toString(std::move(error)));
        return false;
    }
    return true;
}

bool LLVMJIT::add_object(
    luisa::string_view object,
    luisa::string_view buffer_name) noexcept {
    if (!succeeded()) { return false; }
    if (object.empty()) {
        _fail("cannot add an empty object buffer to LLVM ORC JIT");
        return false;
    }
    auto object_buffer = ::llvm::MemoryBuffer::getMemBufferCopy(
        ::llvm::StringRef{object.data(), object.size()},
        ::llvm::StringRef{buffer_name.data(), buffer_name.size()});
    if (auto error = _jit->addObjectFile(std::move(object_buffer))) {
        _fail("failed to add cached object to ORC JIT: " +
              ::llvm::toString(std::move(error)));
        return false;
    }
    return true;
}

bool LLVMJIT::_prepare_module(
    ::llvm::Module &module, bool *out_size_bounded) noexcept {
    module.setDataLayout(_target_machine->createDataLayout());
#if LLVM_VERSION_MAJOR >= 21
    module.setTargetTriple(_target_machine->getTargetTriple());
#else
    module.setTargetTriple(_target_machine->getTargetTriple().str());
#endif
    if (::llvm::verifyModule(module, &::llvm::errs())) {
        _fail("refusing to process an invalid LLVM module");
        return false;
    }

    ::llvm::LoopAnalysisManager loop_analyses;
    ::llvm::FunctionAnalysisManager function_analyses;
    ::llvm::CGSCCAnalysisManager cgscc_analyses;
    ::llvm::ModuleAnalysisManager module_analyses;
    ::llvm::PipelineTuningOptions tuning_options;
#if LLVM_VERSION_MAJOR >= 23
    tuning_options.MergeFunctions = false;
#endif
    auto pass_builder = luisa::compute::detail::make_llvm_pass_builder(
        module.getContext(), _target_machine.get(), tuning_options);
    if (!luisa::compute::detail::env_flag("LUISA_SIMD_DISABLE_W1_SCALARIZE")) {
        pass_builder.registerVectorizerStartEPCallback(
            [](::llvm::FunctionPassManager &fpm, ::llvm::OptimizationLevel) {
                fpm.addPass(ScalarizeW1FunctionPass{});
            });
    }
    pass_builder.registerModuleAnalyses(module_analyses);
    pass_builder.registerCGSCCAnalyses(cgscc_analyses);
    pass_builder.registerFunctionAnalyses(function_analyses);
    pass_builder.registerLoopAnalyses(loop_analyses);
    pass_builder.crossRegisterProxies(
        loop_analyses, function_analyses,
        cgscc_analyses, module_analyses);
    auto block_count = size_t{0u};
    auto instruction_count = size_t{0u};
    auto max_function_blocks = size_t{0u};
    auto max_function_instructions = size_t{0u};
    for (auto &&function : module) {
        if (function.isDeclaration()) { continue; }
        auto fn_blocks = size_t{0u};
        auto fn_instructions = size_t{0u};
        for (auto &&block : function) {
            fn_blocks++;
            fn_instructions += block.size();
        }
        block_count += fn_blocks;
        instruction_count += fn_instructions;
        max_function_blocks = std::max(max_function_blocks, fn_blocks);
        max_function_instructions =
            std::max(max_function_instructions, fn_instructions);
    }
    auto size_bounded = selects_size_bounded_pipeline(
        max_function_blocks, max_function_instructions);
    if (out_size_bounded != nullptr) {
        *out_size_bounded = size_bounded;
    }
    auto report = luisa::compute::detail::env_flag("LUISA_SIMD_REPORT_OPTIMIZATIONS");
    if (report) {
        LUISA_INFO(
            "SIMD LLVM pipeline: {}, blocks={}, instructions={}, "
            "max_fn_blocks={}, max_fn_instructions={}",
            size_bounded ? "linear-SROA/O1-CG size-bounded" : "O2",
            block_count, instruction_count,
            max_function_blocks, max_function_instructions);
    }
    // In large kernels (e.g. multi-hundred-material path tracers), force-inlining
    // heavy transcendental math helpers (pow/exp/log/atan2/sin/cos) at hundreds of
    // sites causes massive code-size explosion, I-cache thrashing, and quadratic
    // register-allocation compile time. Outline heavily reused pure-IR native math
    // helpers with FastCC (never PreserveAll on AVX-512, since ymm saves zero the
    // upper 256 bits of live caller zmm registers).
    if (instruction_count >= size_t{32768u} &&
        !luisa::compute::detail::env_flag("LUISA_SIMD_KEEP_INLINE_NATIVE_MATH")) {
        for (auto &&function : module) {
            if (function.isDeclaration() ||
                !function.hasFnAttribute("luisa.cpu.native_math")) {
                continue;
            }
            size_t call_count = 0u;
            bool all_direct_calls = true;
            for (auto *user : function.users()) {
                auto *cb = ::llvm::dyn_cast<::llvm::CallBase>(user);
                if (cb == nullptr || cb->getCalledFunction() != &function) {
                    all_direct_calls = false;
                    break;
                }
                ++call_count;
            }
            if (all_direct_calls && call_count >= size_t{8u}) {
                function.removeFnAttr(::llvm::Attribute::AlwaysInline);
                function.addFnAttr(::llvm::Attribute::NoInline);
                function.setCallingConv(::llvm::CallingConv::Fast);
                for (auto *user : function.users()) {
                    ::llvm::cast<::llvm::CallBase>(user)->setCallingConv(
                        ::llvm::CallingConv::Fast);
                }
            }
        }
    }
    auto started = report ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto run_opt = [&]() noexcept {
        if (size_bounded) {
            ::llvm::ModulePassManager pipeline;
            pipeline.addPass(::llvm::AlwaysInlinerPass{});
            ::llvm::FunctionPassManager fpm;
            fpm.addPass(::llvm::SROAPass{::llvm::SROAOptions::PreserveCFG});
            fpm.addPass(::llvm::EarlyCSEPass{/*UseMemorySSA=*/false});
            fpm.addPass(::llvm::SimplifyCFGPass{});
            pipeline.addPass(
                ::llvm::createModuleToFunctionPassAdaptor(std::move(fpm)));
            pipeline.run(module, module_analyses);
        } else {
            auto pipeline = pass_builder.buildPerModuleDefaultPipeline(
                ::llvm::OptimizationLevel::O2);
            pipeline.run(module, module_analyses);
        }
    };
    if (instruction_count >= size_t{32768u}) {
        run_with_compiler_stack(run_opt);
    } else {
        run_opt();
    }
    if (report) {
        auto elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        auto post_block_count = size_t{0u};
        auto post_instruction_count = size_t{0u};
        auto phi_count = size_t{0u};
        auto max_block_size = size_t{0u};
        auto max_block_phis = size_t{0u};
        for (auto &&function : module) {
            if (function.isDeclaration()) { continue; }
            for (auto &&block : function) {
                post_block_count++;
                post_instruction_count += block.size();
                max_block_size = std::max(max_block_size, block.size());
                auto block_phis = size_t{0u};
                for (auto &&instruction : block) {
                    block_phis += ::llvm::isa<::llvm::PHINode>(instruction);
                }
                phi_count += block_phis;
                max_block_phis = std::max(max_block_phis, block_phis);
            }
        }
        LUISA_INFO(
            "SIMD LLVM post-opt '{}': pipeline={}, optimize_ms={:.3f}, "
            "blocks={}, instructions={}, phis={}, max_block_size={}, max_block_phis={}",
            module.getModuleIdentifier(), size_bounded ? "O0-IR/O1-CG size-bounded" : "O2", elapsed_ms,
            post_block_count, post_instruction_count, phi_count, max_block_size, max_block_phis);
    }
    return true;
}

std::string LLVMJIT::_emit_assembly(::llvm::Module &module) noexcept {
    auto size_bounded = false;
    if (!_prepare_module(module, &size_bounded)) { return {}; }
    ::llvm::SmallVector<char, 0u> storage;
    ::llvm::raw_svector_ostream output{storage};
    ::llvm::legacy::PassManager codegen;
    if (size_bounded) {
        _target_machine->setOptLevel(::llvm::CodeGenOptLevel::Less);
    }
    if (_target_machine->addPassesToEmitFile(
            codegen, output, nullptr,
            ::llvm::CodeGenFileType::AssemblyFile)) {
        if (size_bounded) {
            _target_machine->setOptLevel(::llvm::CodeGenOptLevel::Aggressive);
        }
        _fail("LLVM host target cannot emit assembly");
        return {};
    }
    codegen.run(module);
    if (size_bounded) {
        _target_machine->setOptLevel(::llvm::CodeGenOptLevel::Aggressive);
    }
    return std::string{storage.begin(), storage.end()};
}

std::string LLVMJIT::emit_assembly(
    std::unique_ptr<::llvm::Module> module,
    std::unique_ptr<::llvm::LLVMContext> context) noexcept {
    ModuleContextLifetime lifetime{module, context};
    if (!succeeded()) { return {}; }
    if (module == nullptr || context == nullptr ||
        &module->getContext() != context.get()) {
        _fail("LLVM assembly module and context ownership do not match");
        return {};
    }
    return _emit_assembly(*module);
}

std::string LLVMJIT::emit_assembly_copy(
    const ::llvm::Module &module) noexcept {
    if (!succeeded()) { return {}; }
    auto copy = ::llvm::CloneModule(module);
    return _emit_assembly(*copy);
}

void *LLVMJIT::lookup(luisa::string_view name) noexcept {
    if (!succeeded()) { return nullptr; }
    auto report = luisa::compute::detail::env_flag("LUISA_SIMD_REPORT_OPTIMIZATIONS");
    auto started = report ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    std::optional<::llvm::Expected<::llvm::orc::ExecutorAddr>> symbol_opt;
    run_with_compiler_stack([&]() noexcept {
        symbol_opt.emplace(_jit->lookup(::llvm::StringRef{name.data(), name.size()}));
    });
    auto &symbol = *symbol_opt;
    if (report) {
        auto elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        // ORC lookup includes materialization, native codegen and linking on
        // the first lookup. Later symbol lookups may reuse that materialization.
        LUISA_INFO("SIMD ORC lookup '{}': lookup_ms={:.3f}, succeeded={}", name, elapsed_ms, static_cast<bool>(symbol));
    }
    if (!symbol) {
        _fail("failed to look up LLVM JIT symbol '" +
              std::string{name} + "': " +
              ::llvm::toString(symbol.takeError()));
        return nullptr;
    }
    return symbol->toPtr<void *>();
}

bool LLVMJIT::supports_native_paired_leaf_gather(
    uint32_t width) const noexcept {
    // The measured rewrite is specific to one 512-bit vector of eight packed
    // 64-bit lanes. W4 was neutral and W16 requires two such instructions.
    // The generic legality/scalarization query does not establish relative
    // profitability against two 32-bit gathers. Keep that distinction in the
    // measured exact-W8 policy instead of treating legality as a cost proof.
    if (!succeeded() || width != 8u) { return false; }
    ::llvm::LLVMContext context;
    ::llvm::Module module{"simd-paired-gather-probe", context};
    module.setDataLayout(_target_machine->createDataLayout());
#if LLVM_VERSION_MAJOR >= 21
    module.setTargetTriple(_target_machine->getTargetTriple());
#else
    module.setTargetTriple(_target_machine->getTargetTriple().str());
#endif
    auto *type = ::llvm::FixedVectorType::get(
        ::llvm::Type::getInt64Ty(context), width);
    auto *function_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(context), false);
    auto *function = ::llvm::Function::Create(
        function_type, ::llvm::GlobalValue::PrivateLinkage,
        "simd_paired_gather_probe", module);
    auto target = _target_machine->getTargetTransformInfo(*function);
    auto fixed_register_bits = target.getRegisterBitWidth(
        ::llvm::TargetTransformInfo::RGK_FixedWidthVector);
    return fixed_register_bits.getKnownMinValue() >= 512u &&
           target.isLegalMaskedGather(type, ::llvm::Align{1u}) &&
           !target.forceScalarizeMaskedGather(type, ::llvm::Align{1u});
}

bool LLVMJIT::supports_native_biased_narrow_buffer_gather(
    uint32_t width) const noexcept {
    if (!succeeded() || (width != 8u && width != 16u)) {
        return false;
    }
    ::llvm::LLVMContext context;
    ::llvm::Module module{"simd-biased-narrow-gather-probe", context};
    auto data_layout = _target_machine->createDataLayout();
    // The address identity used by the rewrite is deliberately audited in
    // the backend's 64-bit host ABI. Do not silently generalize it to a
    // target with a different pointer-index width.
    if (data_layout.getPointerSizeInBits() != 64u ||
        data_layout.getIndexSizeInBits(0u) != 64u) {
        return false;
    }
    module.setDataLayout(data_layout);
#if LLVM_VERSION_MAJOR >= 21
    module.setTargetTriple(_target_machine->getTargetTriple());
#else
    module.setTargetTriple(_target_machine->getTargetTriple().str());
#endif
    auto *function_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(context), false);
    auto *function = ::llvm::Function::Create(
        function_type, ::llvm::GlobalValue::PrivateLinkage,
        "simd_biased_narrow_gather_probe", module);
    auto target = _target_machine->getTargetTransformInfo(*function);
    auto fixed_register_bits = target.getRegisterBitWidth(
        ::llvm::TargetTransformInfo::RGK_FixedWidthVector);
    // W16 is deliberately formed as two W8 gathers. A single W16 narrow
    // gather was slower than the existing wide-index lowering on the
    // reference host, while two independently schedulable halves preserve
    // the logical W16 mask and expose the same legal W8 machine shape.
    auto gather_width = std::min(width, 8u);
    auto *gather_type = ::llvm::FixedVectorType::get(
        ::llvm::Type::getInt32Ty(context), gather_width);
    auto alignment = ::llvm::Align{1u};
    return fixed_register_bits.getKnownMinValue() >= 512u &&
           target.isLegalMaskedGather(gather_type, alignment) &&
           !target.forceScalarizeMaskedGather(gather_type, alignment);
}

bool LLVMJIT::supports_native_predicated_loop(
    uint32_t width) const noexcept {
    if (!succeeded() || (width != 8u && width != 16u)) {
        return false;
    }
    ::llvm::LLVMContext context;
    ::llvm::Module module{"simd-predicated-loop-probe", context};
    module.setDataLayout(_target_machine->createDataLayout());
#if LLVM_VERSION_MAJOR >= 21
    module.setTargetTriple(_target_machine->getTargetTriple());
#else
    module.setTargetTriple(_target_machine->getTargetTriple().str());
#endif
    auto *function_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(context), false);
    auto *function = ::llvm::Function::Create(
        function_type, ::llvm::GlobalValue::PrivateLinkage,
        "simd_predicated_loop_probe", module);
    auto target = _target_machine->getTargetTransformInfo(*function);
    auto fixed_register_bits = target.getRegisterBitWidth(
        ::llvm::TargetTransformInfo::RGK_FixedWidthVector);
    auto *gather_type = ::llvm::FixedVectorType::get(
        ::llvm::Type::getInt32Ty(context), width);
    auto alignment = ::llvm::Align{alignof(uint32_t)};
    return fixed_register_bits.getKnownMinValue() >= 512u &&
           target.isLegalMaskedGather(gather_type, alignment) &&
           !target.forceScalarizeMaskedGather(gather_type, alignment);
}

bool LLVMJIT::supports_inlined_packet_batch(
    uint32_t width) const noexcept {
    if (!succeeded() || (width != 8u && width != 16u)) {
        return false;
    }
    ::llvm::LLVMContext context;
    ::llvm::Module module{"simd-packet-batch-probe", context};
    module.setDataLayout(_target_machine->createDataLayout());
#if LLVM_VERSION_MAJOR >= 21
    module.setTargetTriple(_target_machine->getTargetTriple());
#else
    module.setTargetTriple(_target_machine->getTargetTriple().str());
#endif
    auto *function_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(context), false);
    auto *function = ::llvm::Function::Create(
        function_type, ::llvm::GlobalValue::PrivateLinkage,
        "simd_packet_batch_probe", module);
    auto target = _target_machine->getTargetTransformInfo(*function);
    auto fixed_register_bits = target.getRegisterBitWidth(
        ::llvm::TargetTransformInfo::RGK_FixedWidthVector);
    auto *vector_type = ::llvm::FixedVectorType::get(
        ::llvm::Type::getInt32Ty(context), width);
    auto register_class = target.getRegisterClassForType(
        true, vector_type);
    return fixed_register_bits.getKnownMinValue() >= 512u &&
           target.getNumberOfRegisters(register_class) >= 32u;
}

bool LLVMJIT::supports_native_vector_compress(
    uint32_t width) const noexcept {
#if LLVM_VERSION_MAJOR < 18
    static_cast<void>(width);
    return false;
#else
    if (!succeeded() || width != 16u) { return false; }
    ::llvm::LLVMContext context;
    ::llvm::Module module{"simd-vector-compress-probe", context};
    module.setDataLayout(_target_machine->createDataLayout());
#if LLVM_VERSION_MAJOR >= 21
    module.setTargetTriple(_target_machine->getTargetTriple());
#else
    module.setTargetTriple(_target_machine->getTargetTriple().str());
#endif
    auto *function_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(context), false);
    auto *function = ::llvm::Function::Create(
        function_type, ::llvm::GlobalValue::PrivateLinkage,
        "simd_vector_compress_probe", module);
    auto target = _target_machine->getTargetTransformInfo(*function);
    auto fixed_register_bits = target.getRegisterBitWidth(
        ::llvm::TargetTransformInfo::RGK_FixedWidthVector);
    auto *vector_type = ::llvm::FixedVectorType::get(
        ::llvm::Type::getInt32Ty(context), width);
    return fixed_register_bits.getKnownMinValue() >= 512u &&
           target.isLegalMaskedCompressStore(
               vector_type, ::llvm::Align{16u});
#endif
}

bool LLVMJIT::supports_native_half_conversion(
    uint32_t width) const noexcept {
    if (!succeeded() ||
        (width != 4u && width != 8u && width != 16u)) {
        return false;
    }
    ::llvm::LLVMContext context;
    ::llvm::Module module{"simd-half-conversion-probe", context};
    module.setDataLayout(_target_machine->createDataLayout());
#if LLVM_VERSION_MAJOR >= 21
    module.setTargetTriple(_target_machine->getTargetTriple());
#else
    module.setTargetTriple(_target_machine->getTargetTriple().str());
#endif
    auto *function_type = ::llvm::FunctionType::get(
        ::llvm::Type::getVoidTy(context), false);
    auto *function = ::llvm::Function::Create(
        function_type, ::llvm::GlobalValue::PrivateLinkage,
        "simd_half_conversion_probe", module);
    auto target = _target_machine->getTargetTransformInfo(*function);
    auto *float_lanes = ::llvm::FixedVectorType::get(
        ::llvm::Type::getFloatTy(context), width);
    auto *half_lanes = ::llvm::FixedVectorType::get(
        ::llvm::Type::getHalfTy(context), width);
    auto extension_cost = target.getCastInstrCost(
        ::llvm::Instruction::FPExt, float_lanes, half_lanes,
        ::llvm::TTI::CastContextHint::None,
        ::llvm::TTI::TCK_RecipThroughput);
    auto truncation_cost = target.getCastInstrCost(
        ::llvm::Instruction::FPTrunc, half_lanes, float_lanes,
        ::llvm::TTI::CastContextHint::None,
        ::llvm::TTI::TCK_RecipThroughput);
    if (!extension_cost.isValid() || !truncation_cost.isValid()) {
        return false;
    }
    // One W8 cast or at most a bounded split of W16 is the measured useful
    // shape. Generic scalar/helper lowering is priced in the hundreds by TTI.
    auto maximum_packed_cost = static_cast<int64_t>(width / 4u);
    return extension_cost.getValue() <= maximum_packed_cost &&
           truncation_cost.getValue() <= maximum_packed_cost;
}

}// namespace luisa::compute::simd
