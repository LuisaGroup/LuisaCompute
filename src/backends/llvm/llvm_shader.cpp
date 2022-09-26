//
// Created by Mike Smith on 2022/2/11.
//

#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/IR/DiagnosticPrinter.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
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
    auto context = std::make_unique<::llvm::LLVMContext>();
    context->setDiagnosticHandlerCallBack([](const ::llvm::DiagnosticInfo &info, void *) noexcept {
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
    auto module = ::llvm::parseIRFile(file_path.string(), diagnostic, *context);
    Clock clk;
    auto machine = device->target_machine();
    if (module == nullptr) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to load LLVM IR from cache: {}.",
            diagnostic.getMessage().str());
        LLVMCodegen codegen{*context};
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
#if LLVM_VERSION_MAJOR >= 15// Why not taking effect?
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
    if (auto expected_jit = ::llvm::orc::LLJITBuilder{}.create()) {
        _jit = std::move(expected_jit.get());
        ::llvm::orc::ThreadSafeModule m{std::move(module), std::move(context)};
        if (auto error = _jit->addIRModule(std::move(m))) {
            ::llvm::handleAllErrors(std::move(error), [](std::unique_ptr<::llvm::ErrorInfoBase> err) {
                LUISA_WARNING_WITH_LOCATION("LLJIT::addIRModule(): {}", err->message());
            });
            LUISA_ERROR_WITH_LOCATION("Failed to add IR module.");
        }
    } else {
        ::llvm::handleAllErrors(expected_jit.takeError(), [](std::unique_ptr<::llvm::ErrorInfoBase> err) {
            LUISA_WARNING_WITH_LOCATION("LLJITBuilder::create(): {}", err->message());
        });
        LUISA_ERROR_WITH_LOCATION("Failed to create LLJIT.");
    }

    // map symbols
    if (auto generator = ::llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            _jit->getDataLayout().getGlobalPrefix())) {
        _jit->getMainJITDylib().addGenerator(std::move(generator.get()));
    } else {
        ::llvm::handleAllErrors(generator.takeError(), [](std::unique_ptr<::llvm::ErrorInfoBase> err) {
            LUISA_WARNING_WITH_LOCATION("DynamicLibrarySearchGenerator::GetForCurrentProcess(): {}", err->message());
        });
        LUISA_ERROR_WITH_LOCATION("Failed to add generator.");
    }
    ::llvm::orc::SymbolMap symbol_map{
        {_jit->mangleAndIntern("texture.read.2d.int"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_read_2d_int)},
        {_jit->mangleAndIntern("texture.read.3d.int"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_read_3d_int)},
        {_jit->mangleAndIntern("texture.read.2d.uint"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_read_2d_uint)},
        {_jit->mangleAndIntern("texture.read.3d.uint"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_read_3d_uint)},
        {_jit->mangleAndIntern("texture.read.2d.float"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_read_2d_float)},
        {_jit->mangleAndIntern("texture.read.3d.float"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_read_3d_float)},
        {_jit->mangleAndIntern("texture.write.2d.int"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_write_2d_int)},
        {_jit->mangleAndIntern("texture.write.3d.int"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_write_3d_int)},
        {_jit->mangleAndIntern("texture.write.2d.uint"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_write_2d_uint)},
        {_jit->mangleAndIntern("texture.write.3d.uint"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_write_3d_uint)},
        {_jit->mangleAndIntern("texture.write.2d.float"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_write_2d_float)},
        {_jit->mangleAndIntern("texture.write.3d.float"), ::llvm::JITEvaluatedSymbol::fromPointer(&texture_write_3d_float)},
        {_jit->mangleAndIntern("accel.trace.closest"), ::llvm::JITEvaluatedSymbol::fromPointer(&accel_trace_closest)},
        {_jit->mangleAndIntern("accel.trace.any"), ::llvm::JITEvaluatedSymbol::fromPointer(&accel_trace_any)},
        {_jit->mangleAndIntern("bindless.texture.2d.read"), ::llvm::JITEvaluatedSymbol::fromPointer(&bindless_texture_2d_read)},
        {_jit->mangleAndIntern("bindless.texture.3d.read"), ::llvm::JITEvaluatedSymbol::fromPointer(&bindless_texture_3d_read)},
        {_jit->mangleAndIntern("bindless.texture.2d.sample"), ::llvm::JITEvaluatedSymbol::fromPointer(&bindless_texture_2d_sample)},
        {_jit->mangleAndIntern("bindless.texture.3d.sample"), ::llvm::JITEvaluatedSymbol::fromPointer(&bindless_texture_3d_sample)},
        {_jit->mangleAndIntern("bindless.texture.2d.sample.level"), ::llvm::JITEvaluatedSymbol::fromPointer(&bindless_texture_2d_sample_level)},
        {_jit->mangleAndIntern("bindless.texture.3d.sample.level"), ::llvm::JITEvaluatedSymbol::fromPointer(&bindless_texture_3d_sample_level)},
        {_jit->mangleAndIntern("bindless.texture.2d.sample.grad"), ::llvm::JITEvaluatedSymbol::fromPointer(&bindless_texture_2d_sample_grad)},
        {_jit->mangleAndIntern("bindless.texture.3d.sample.grad"), ::llvm::JITEvaluatedSymbol::fromPointer(&bindless_texture_3d_sample_grad)}};
    if (auto error = _jit->getMainJITDylib().define(
            ::llvm::orc::absoluteSymbols(std::move(symbol_map)))) {
        ::llvm::handleAllErrors(std::move(error), [](std::unique_ptr<::llvm::ErrorInfoBase> err) {
            LUISA_WARNING_WITH_LOCATION("LLJIT::define(): {}", err->message());
        });
        LUISA_ERROR_WITH_LOCATION("Failed to define symbols.");
    }
    auto main_name = luisa::format("kernel.{:016x}.main", func.hash());
    if (auto addr = _jit->lookup(::llvm::StringRef{main_name.data(), main_name.size()})) {
#if LLVM_VERSION_MAJOR >= 15
        _kernel_entry = addr->toPtr<kernel_entry_t>();
#else
        _kernel_entry = reinterpret_cast<kernel_entry_t *>(addr->getAddress());
#endif
    } else {
        LUISA_ERROR_WITH_LOCATION("Failed to lookup symbol '{}'.", main_name);
    }
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
