#include "metal_builtin_air.h"

#include <array>
#include <string>
#include <vector>

#include <llvm/IR/Verifier.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/raw_ostream.h>

#include <luisa/core/logging.h>

#include "llvm_codegen/metal_codegen_llvm_builtin.h"
#include "metal_metallib.h"
#include "../../ext/llvm_downgrade.h"

namespace luisa::compute::metal {
namespace {

struct BuiltinEntry {
    MetalBuiltinLLVMProgram program;
    luisa::string_view name;
    MetalLibProgramType type;
};

constexpr std::array builtin_entries{
    BuiltinEntry{
        MetalBuiltinLLVMProgram::UPDATE_ACCEL_INSTANCES,
        "update_accel_instances", MetalLibProgramType::KERNEL},
    BuiltinEntry{
        MetalBuiltinLLVMProgram::UPDATE_BINDLESS_ARRAY,
        "update_bindless_array", MetalLibProgramType::KERNEL},
    BuiltinEntry{
        MetalBuiltinLLVMProgram::PREPARE_INDIRECT_DISPATCHES,
        "prepare_indirect_dispatches", MetalLibProgramType::KERNEL},
    BuiltinEntry{
        MetalBuiltinLLVMProgram::SWAPCHAIN_VERTEX,
        "swapchain_vertex_shader", MetalLibProgramType::VERTEX},
    BuiltinEntry{
        MetalBuiltinLLVMProgram::SWAPCHAIN_FRAGMENT,
        "swapchain_fragment_shader", MetalLibProgramType::FRAGMENT}};

void verify_module(
    const llvm::Module &module,
    luisa::string_view entry,
    luisa::string_view phase) noexcept {
    std::string error;
    llvm::raw_string_ostream stream{error};
    if (llvm::verifyModule(module, &stream)) {
        stream.flush();
        LUISA_ERROR_WITH_LOCATION(
            "Invalid Metal runtime builtin '{}' after {}: {}",
            entry, phase, error);
    }
}

void optimize_module(llvm::Module &module) noexcept {
    llvm::LoopAnalysisManager loop_analysis;
    llvm::FunctionAnalysisManager function_analysis;
    llvm::CGSCCAnalysisManager cgscc_analysis;
    llvm::ModuleAnalysisManager module_analysis;
    llvm::PipelineTuningOptions tuning;
    tuning.LoopUnrolling = false;
    llvm::PassBuilder pass_builder{nullptr, tuning};
    pass_builder.registerModuleAnalyses(module_analysis);
    pass_builder.registerCGSCCAnalyses(cgscc_analysis);
    pass_builder.registerFunctionAnalyses(function_analysis);
    pass_builder.registerLoopAnalyses(loop_analysis);
    pass_builder.crossRegisterProxies(
        loop_analysis, function_analysis,
        cgscc_analysis, module_analysis);
    auto pipeline = pass_builder.buildPerModuleDefaultPipeline(
        llvm::OptimizationLevel::O2);
    pipeline.run(module, module_analysis);
}

[[nodiscard]] MetalLibTarget library_target(
    const MetalAIRTarget &target) noexcept {
    auto version = target.operating_system_version;
    switch (target.platform) {
        case MetalAIRPlatform::MACOS:
            return metallib_target_for_macos(
                static_cast<uint16_t>(version.major),
                static_cast<uint16_t>(version.minor),
                static_cast<uint16_t>(version.patch));
        case MetalAIRPlatform::IOS:
            return metallib_target_for_ios(
                static_cast<uint16_t>(version.major),
                static_cast<uint16_t>(version.minor),
                static_cast<uint16_t>(version.patch));
    }
    LUISA_ERROR_WITH_LOCATION("Invalid Metal runtime builtin AIR platform.");
}

}// namespace

luisa::vector<std::byte> metal_codegen_builtin_air(
    const MetalAIRTarget &target) noexcept {
    std::array<std::vector<std::byte>, builtin_entries.size()> modules;
    for (auto i = 0u; i < builtin_entries.size(); i++) {
        auto entry = builtin_entries[i];
        auto config = metal_air_codegen_config(
            target, luisa::format("metal_builtin/{}.metal", entry.name));
        config.enable_fast_math = true;
        auto result = luisa_compute_metal_codegen_builtin_llvm(
            entry.program, config);
        LUISA_ASSERT(result.module,
                     "Metal runtime builtin '{}' LLVM generation failed.",
                     entry.name);
        verify_module(*result.module, entry.name, "LLVM generation");
        optimize_module(*result.module);
        verify_module(*result.module, entry.name, "LLVM optimization");
        modules[i] = llvm_downgrade_to_14(std::move(result.module));
    }

    std::array<MetalLibFunction, builtin_entries.size()> functions;
    std::array<luisa::string_view, builtin_entries.size()> entry_points;
    std::array<MetalLibProgramType, builtin_entries.size()> program_types;
    for (auto i = 0u; i < builtin_entries.size(); i++) {
        auto entry = builtin_entries[i];
        functions[i] = MetalLibFunction{
            entry.name, modules[i], entry.type};
        entry_points[i] = entry.name;
        program_types[i] = entry.type;
    }
    auto library = make_metallib(library_target(target), functions);
    LUISA_ASSERT(
        validate_metallib(library, entry_points, program_types),
        "Generated Metal runtime builtin library failed validation.");
    return library;
}

}// namespace luisa::compute::metal
