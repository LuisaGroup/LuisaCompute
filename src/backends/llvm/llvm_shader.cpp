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
    _argument_buffer_size += 16u; // (trampoline, callbacks)
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

    auto jit = device->jit();
    auto main_name = luisa::format("kernel.{:016x}.main", func.hash());
    auto lookup_kernel_entry = [name = luisa::string_view{main_name}, jit, device]() noexcept -> ::llvm::Expected<kernel_entry_t *> {
        std::scoped_lock lock{device->jit_mutex()};
        auto addr = jit->lookup(::llvm::StringRef{name.data(), name.size()});
        if (addr) {
#if LLVM_VERSION_MAJOR >= 15
            return addr->toPtr<kernel_entry_t>();
#else
            return reinterpret_cast<kernel_entry_t *>(addr->getAddress());
#endif
        }
        return addr.takeError();
    };

    // try to find the kernel entry in the JIT
    if (auto entry = lookup_kernel_entry()) {
        LUISA_INFO("Found kernel '{}' in JIT.", _name);
        _kernel_entry = *entry;
        return;
    }

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
        MPM.run(*module, MAM);
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
    if (auto error = [jit, device, m = ::llvm::orc::ThreadSafeModule{std::move(module), std::move(context)}]() mutable noexcept {
            std::scoped_lock lock{device->jit_mutex()};
            return jit->addIRModule(std::move(m));
        }()) {
        ::llvm::handleAllErrors(std::move(error), [](const ::llvm::ErrorInfoBase &err) {
            LUISA_WARNING_WITH_LOCATION("LLJIT::addIRModule(): {}", err.message());
        });
    }
    if (auto entry = lookup_kernel_entry()) {
        _kernel_entry = *entry;
    } else {
        ::llvm::handleAllErrors(entry.takeError(), [](const ::llvm::ErrorInfoBase &err) {
            LUISA_WARNING_WITH_LOCATION("LLJIT::lookup(): {}", err.message());
        });
        LUISA_ERROR_WITH_LOCATION("Failed to find kernel entry.");
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
