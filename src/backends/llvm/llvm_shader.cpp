//
// Created by Mike Smith on 2022/2/11.
//

#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/IR/DiagnosticPrinter.h>
#if LLVM_VERSION_MAJOR >= 15
#include <llvm/Transforms/Coroutines/CoroEarly.h>
#include <llvm/Transforms/Coroutines/CoroSplit.h>
#include <llvm/Transforms/Coroutines/CoroElide.h>
#include <llvm/Transforms/Coroutines/CoroCleanup.h>
#include <llvm/Transforms/Coroutines/CoroConditionalWrapper.h>
#endif
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <core/mathematics.h>
#include <backends/llvm/llvm_shader.h>
#include <backends/llvm/llvm_device.h>
#include <backends/llvm/llvm_codegen.h>
#include <backends/llvm/llvm_accel.h>

#define LC_LLVM_CODEGEN_MAGIC ".codegen.0004"

namespace luisa::compute::llvm {

LLVMShader::LLVMShader(LLVMDevice *device, Function func) noexcept
    : _name{luisa::format("kernel.{:016x}", func.hash())} {
    // compute argument offsets
    _argument_offsets.reserve(func.arguments().size());
    for (auto &&arg : func.arguments()) {
        auto aligned_offset = luisa::align(_argument_buffer_size, 16u);
        _argument_offsets.emplace(arg.uid(), aligned_offset);
        if (arg.type()->is_buffer()) {
            _argument_buffer_size = aligned_offset + LLVMCodegen::buffer_handle_size;
        } else if (arg.type()->is_texture()) {
            _argument_buffer_size = aligned_offset + LLVMCodegen::texture_handle_size;
        } else if (arg.type()->is_accel()) {
            _argument_buffer_size = aligned_offset + LLVMCodegen::accel_handle_size;
        } else if (arg.type()->is_bindless_array()) {
            _argument_buffer_size = aligned_offset + LLVMCodegen::bindless_array_handle_size;
        } else {
            _argument_buffer_size = aligned_offset + arg.type()->size();
        }
    }
    _argument_buffer_size = luisa::align(_argument_buffer_size, 16u);

    for (auto s : func.shared_variables()) {
        _shared_memory_size = luisa::align(_shared_memory_size, s.type()->alignment());
        _shared_memory_size += s.type()->size();
    }
    _shared_memory_size = luisa::align(_shared_memory_size, 16u);

    LUISA_VERBOSE_WITH_LOCATION(
        "Generating kernel '{}' with {} bytes of "
        "argument buffer and {} bytes of shared memory.",
        _name, _argument_buffer_size, _shared_memory_size);

    // codegen
    std::error_code ec;
    _context = luisa::make_unique<::llvm::LLVMContext>();
    _context->setDiagnosticHandlerCallBack([](const ::llvm::DiagnosticInfo &info, void *) noexcept {
        if (auto severity = info.getSeverity();
            severity == ::llvm::DS_Error || severity == ::llvm::DS_Warning) {
            ::llvm::DiagnosticPrinterRawOStream printer{::llvm::errs()};
            info.print(printer);
            printer << '\n';
        }
    });
    auto file_path = device->context().cache_directory() /
                     luisa::format("kernel.llvm.{:016x}.opt.{:016x}.ll",
                                   func.hash(), hash64(LLVM_VERSION_STRING LC_LLVM_CODEGEN_MAGIC));
    ::llvm::SMDiagnostic diagnostic;
    auto module = ::llvm::parseIRFile(file_path.string(), diagnostic, *_context);
    Clock clk;
    auto machine = device->target_machine();
    if (module == nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to load LLVM IR from cache: {}.",
            diagnostic.getMessage().str());
        LLVMCodegen codegen{*_context};
        module = codegen.emit(func);
        LUISA_INFO("Codegen: {} ms.", clk.toc());
        if (::llvm::verifyModule(*module, &::llvm::errs())) {
            auto error_file_path = device->context().cache_directory() /
                                   luisa::format("kernel.llvm.{:016x}.ll", func.hash());
            auto error_file_path_string = file_path.string();
            ::llvm::raw_fd_ostream file{error_file_path_string, ec};
            if (ec) {
                LUISA_WARNING_WITH_LOCATION(
                    "Failed to create file '{}': {}.",
                    error_file_path_string, ec.message());
            } else {
                LUISA_INFO("Saving LLVM kernel to '{}'.",
                           error_file_path_string);
                module->print(file, nullptr);
            }
            LUISA_ERROR_WITH_LOCATION("Failed to verify module.");
        }
        module->setDataLayout(machine->createDataLayout());
        module->setTargetTriple(machine->getTargetTriple().str());

        // optimize with the new pass manager
        ::llvm::LoopAnalysisManager LAM;
        ::llvm::FunctionAnalysisManager FAM;
        ::llvm::CGSCCAnalysisManager CGAM;
        ::llvm::ModuleAnalysisManager MAM;
        ::llvm::PipelineTuningOptions PTO;
        PTO.LoopInterleaving = true;
        PTO.LoopVectorization = true;
        PTO.SLPVectorization = true;
        PTO.LoopUnrolling = true;
        PTO.MergeFunctions = true;
        ::llvm::PassBuilder PB{machine, PTO};
        FAM.registerPass([&] { return PB.buildDefaultAAPipeline(); });
        PB.registerModuleAnalyses(MAM);
        PB.registerCGSCCAnalyses(CGAM);
        PB.registerFunctionAnalyses(FAM);
        PB.registerLoopAnalyses(LAM);
        PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
        machine->registerPassBuilderCallbacks(PB);
        clk.tic();
        auto MPM = PB.buildPerModuleDefaultPipeline(::llvm::OptimizationLevel::O3);
#if LLVM_VERSION_MAJOR >= 15 // Why not taking effect?
        ::llvm::ModulePassManager CoroPM;
        CoroPM.addPass(::llvm::CoroEarlyPass{});
        ::llvm::CGSCCPassManager CGPM;
        ::llvm::FunctionPassManager FPM;
        CGPM.addPass(::llvm::CoroSplitPass{true});
        FPM.addPass(::llvm::CoroElidePass{});
        CoroPM.addPass(::llvm::createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
        CoroPM.addPass(::llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
        CoroPM.addPass(::llvm::CoroCleanupPass{});
        CoroPM.addPass(::llvm::GlobalDCEPass{});
        MPM.addPass(::llvm::CoroConditionalWrapper{std::move(CoroPM)});
#endif
        MPM.run(*module, MAM);

        // optimize with the legacy pass manager
        //        ::llvm::PassManagerBuilder pass_manager_builder;
        //        pass_manager_builder.OptLevel = ::llvm::CodeGenOpt::Aggressive;
        //        pass_manager_builder.Inliner = ::llvm::createFunctionInliningPass(
        //            pass_manager_builder.OptLevel, 0, false);
        //        pass_manager_builder.LoopsInterleaved = true;
        //        pass_manager_builder.LoopVectorize = true;
        //        pass_manager_builder.SLPVectorize = true;
        //        pass_manager_builder.MergeFunctions = true;
        //        pass_manager_builder.NewGVN = true;
        //        machine->adjustPassManager(pass_manager_builder);
        //        module->setDataLayout(machine->createDataLayout());
        //        module->setTargetTriple(machine->getTargetTriple().str());
        //        ::llvm::legacy::PassManager module_pass_manager;
        //        module_pass_manager.add(
        //            ::llvm::createTargetTransformInfoWrapperPass(
        //                machine->getTargetIRAnalysis()));
        //        pass_manager_builder.populateModulePassManager(module_pass_manager);
        //        module_pass_manager.run(*module);
        LUISA_INFO("Optimize: {} ms.", clk.toc());
        if (::llvm::verifyModule(*module, &::llvm::errs())) {
            LUISA_ERROR_WITH_LOCATION("Failed to verify module.");
        }

        // dump optimized ir for debugging
        {
            auto file_path_string = file_path.string();
            ::llvm::raw_fd_ostream file_opt{file_path_string, ec};
            if (ec) {
                LUISA_ERROR_WITH_LOCATION(
                    "Failed to create file '{}': {}.",
                    file_path_string, ec.message());
            } else {
                LUISA_INFO("Saving optimized LLVM kernel to '{}'.",
                           file_path_string);
                module->print(file_opt, nullptr);
            }
        }
    }

    // compile to machine code
    clk.tic();
    std::string err;
    static std::mutex mutex;
    std::unique_lock lock{mutex};
    _engine = ::llvm::EngineBuilder{std::move(module)}
                  .setErrorStr(&err)
                  .setOptLevel(::llvm::CodeGenOpt::Aggressive)
                  .setEngineKind(::llvm::EngineKind::JIT)
                  .create(machine);
    _engine->DisableLazyCompilation(true);
    _engine->DisableSymbolSearching(false);
    _engine->InstallLazyFunctionCreator([](auto &&name) noexcept -> void * {
        using namespace std::string_view_literals;
        static const luisa::unordered_map<luisa::string_view, void *> symbols{
            {"texture.read.2d.int"sv, reinterpret_cast<void *>(&texture_read_2d_int)},
            {"texture.read.3d.int"sv, reinterpret_cast<void *>(&texture_read_3d_int)},
            {"texture.read.2d.uint"sv, reinterpret_cast<void *>(&texture_read_2d_uint)},
            {"texture.read.3d.uint"sv, reinterpret_cast<void *>(&texture_read_3d_uint)},
            {"texture.read.2d.float"sv, reinterpret_cast<void *>(&texture_read_2d_float)},
            {"texture.read.3d.float"sv, reinterpret_cast<void *>(&texture_read_3d_float)},
            {"texture.write.2d.int"sv, reinterpret_cast<void *>(&texture_write_2d_int)},
            {"texture.write.3d.int"sv, reinterpret_cast<void *>(&texture_write_3d_int)},
            {"texture.write.2d.uint"sv, reinterpret_cast<void *>(&texture_write_2d_uint)},
            {"texture.write.3d.uint"sv, reinterpret_cast<void *>(&texture_write_3d_uint)},
            {"texture.write.2d.float"sv, reinterpret_cast<void *>(&texture_write_2d_float)},
            {"texture.write.3d.float"sv, reinterpret_cast<void *>(&texture_write_3d_float)},
            {"accel.trace.closest"sv, reinterpret_cast<void *>(&accel_trace_closest)},
            {"accel.trace.any"sv, reinterpret_cast<void *>(&accel_trace_any)},
            {"bindless.texture.2d.read"sv, reinterpret_cast<void *>(&bindless_texture_2d_read)},
            {"bindless.texture.3d.read"sv, reinterpret_cast<void *>(&bindless_texture_3d_read)},
            {"bindless.texture.2d.sample"sv, reinterpret_cast<void *>(&bindless_texture_2d_sample)},
            {"bindless.texture.3d.sample"sv, reinterpret_cast<void *>(&bindless_texture_3d_sample)},
            {"bindless.texture.2d.sample.level"sv, reinterpret_cast<void *>(&bindless_texture_2d_sample_level)},
            {"bindless.texture.3d.sample.level"sv, reinterpret_cast<void *>(&bindless_texture_3d_sample_level)},
            {"bindless.texture.2d.sample.grad"sv, reinterpret_cast<void *>(&bindless_texture_2d_sample_grad)},
            {"bindless.texture.3d.sample.grad"sv, reinterpret_cast<void *>(&bindless_texture_3d_sample_grad)}};
        auto name_view = luisa::string_view{name};
        if (name_view.starts_with('_')) { name_view = name_view.substr(1u); }
        LUISA_INFO("Searching for symbol '{}' in JIT.", name_view);
        if (auto iter = symbols.find(name_view); iter != symbols.end()) {
            return iter->second;
        }
        LUISA_WARNING_WITH_LOCATION("Failed to find symbol '{}' in JIT.", name_view);
        return nullptr;
    });
    LUISA_ASSERT(_engine != nullptr, "Failed to create execution engine: {}.", err);
    auto main_name = luisa::format("kernel.{:016x}.main", func.hash());
    _kernel_entry = reinterpret_cast<kernel_entry_t *>(
        _engine->getFunctionAddress(std::string{main_name}));
    LUISA_ASSERT(_kernel_entry != nullptr, "Failed to find kernel entry.");
    LUISA_INFO("Compile: {} ms.", clk.toc());
}

LLVMShader::~LLVMShader() noexcept = default;

size_t LLVMShader::argument_offset(uint uid) const noexcept {
    if (auto iter = _argument_offsets.find(uid);
        iter != _argument_offsets.cend()) [[likely]] {
        return iter->second;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid argument uid {}.", uid);
}

void LLVMShader::invoke(const std::byte *args, std::byte *shared_mem,
                        uint3 dispatch_size, uint3 block_id) const noexcept {
    _kernel_entry(args, shared_mem,
                  dispatch_size.x, dispatch_size.y, dispatch_size.z,
                  block_id.x, block_id.y, block_id.z);
}

}// namespace luisa::compute::llvm
