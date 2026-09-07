#include "metal_air_pipeline.h"

#include <array>
#include <cstdlib>
#include <fstream>
#include <string>

#include <llvm/IR/Verifier.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <luisa/core/logging.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/raster/raster_state.h>
#include <luisa/xir/module.h>
#include <luisa/xir/verifier.h>

#include "llvm_codegen/metal_codegen_llvm.h"
#include "metal_api.h"
#include "metal_metallib.h"
#include "metal_raster_archive.h"
#include "../../ext/llvm_downgrade.h"

namespace luisa::compute::metal {

namespace {

[[nodiscard]] MetalAIRVersion host_macos_version() noexcept {
    auto version = NS::ProcessInfo::processInfo()->operatingSystemVersion();
    auto major = static_cast<uint32_t>(version.majorVersion);
    if (major >= 16u && major < 26u) { major += 10u; }
    return {major,
            static_cast<uint32_t>(version.minorVersion),
            static_cast<uint32_t>(version.patchVersion)};
}

[[nodiscard]] MetalAIRVersion host_ios_version() noexcept {
    auto version = NS::ProcessInfo::processInfo()->operatingSystemVersion();
    return {static_cast<uint32_t>(version.majorVersion),
            static_cast<uint32_t>(version.minorVersion),
            static_cast<uint32_t>(version.patchVersion)};
}

[[nodiscard]] MetalAIRVersion parse_macos_version(luisa::string_view text) noexcept {
    MetalAIRVersion version{};
    auto component = 0u;
    for (auto c : text) {
        if (c == '.') {
            component++;
            if (component > 2u) { break; }
        } else if (c >= '0' && c <= '9') {
            auto *value = component == 0u ? &version.major :
                          component == 1u ? &version.minor :
                                            &version.patch;
            *value = *value * 10u + static_cast<uint32_t>(c - '0');
        }
    }
    if (version.major >= 16u && version.major < 26u) {
        version.major += 10u;
    }
    return version;
}

[[nodiscard]] MetalAIRVersion sdk_macos_version() noexcept {
#ifdef LUISA_METAL_AIR_SDK_VERSION
    auto text = luisa::string_view{LUISA_METAL_AIR_SDK_VERSION};
    auto version = parse_macos_version(text);
    LUISA_ASSERT(version.major >= 13u,
                 "Invalid Metal AIR SDK version '{}'.", text);
    return version;
#else
    return host_macos_version();
#endif
}

[[nodiscard]] MetalAIRVersion sdk_ios_version() noexcept {
#ifdef LUISA_METAL_AIR_SDK_VERSION
    auto text = luisa::string_view{LUISA_METAL_AIR_SDK_VERSION};
    MetalAIRVersion version{};
    auto component = 0u;
    for (auto c : text) {
        if (c == '.') {
            component++;
            if (component > 2u) { break; }
        } else if (c >= '0' && c <= '9') {
            auto *value = component == 0u ? &version.major :
                          component == 1u ? &version.minor :
                                            &version.patch;
            *value = *value * 10u + static_cast<uint32_t>(c - '0');
        }
    }
    LUISA_ASSERT(version.major >= 16u,
                 "Invalid Metal AIR iOS SDK version '{}'.", text);
    return version;
#else
    return host_ios_version();
#endif
}

[[nodiscard]] MetalAIRVersion air_version_for_target(
    MetalAIRPlatform platform, uint32_t major) noexcept {
    if (major >= 27u) { return {2u, 9u, 0u}; }
    if (major >= 26u ||
        (platform == MetalAIRPlatform::MACOS && major >= 16u && major < 26u)) {
        return {2u, 8u, 0u};
    }
    if ((platform == MetalAIRPlatform::MACOS && major >= 15u) ||
        (platform == MetalAIRPlatform::IOS && major >= 18u)) {
        return {2u, 7u, 0u};
    }
    if ((platform == MetalAIRPlatform::MACOS && major >= 14u) ||
        (platform == MetalAIRPlatform::IOS && major >= 17u)) {
        return {2u, 6u, 0u};
    }
    return {2u, 5u, 0u};
}

[[nodiscard]] MetalAIRVersion metal_version_for_target(
    MetalAIRPlatform platform, uint32_t major) noexcept {
    auto air = air_version_for_target(platform, major);
    if (air.major == 2u && air.minor >= 9u) { return {4u, 1u, 0u}; }
    if (air.major == 2u && air.minor >= 8u) { return {4u, 0u, 0u}; }
    if (air.major == 2u && air.minor >= 7u) { return {3u, 2u, 0u}; }
    if (air.major == 2u && air.minor >= 6u) { return {3u, 1u, 0u}; }
    return {3u, 0u, 0u};
}

[[nodiscard]] bool version_less(
    MetalAIRVersion lhs, MetalAIRVersion rhs) noexcept {
    if (lhs.major != rhs.major) { return lhs.major < rhs.major; }
    if (lhs.minor != rhs.minor) { return lhs.minor < rhs.minor; }
    return lhs.patch < rhs.patch;
}

[[nodiscard]] constexpr bool current_device_sdk_compatible(
    MetalAIRVersion operating_system_version,
    MetalAIRVersion sdk_version) noexcept {
    return sdk_version.major >= operating_system_version.major;
}

static_assert(current_device_sdk_compatible(
    MetalAIRVersion{26u, 6u, 0u},
    MetalAIRVersion{26u, 4u, 0u}));
static_assert(!current_device_sdk_compatible(
    MetalAIRVersion{27u, 0u, 0u},
    MetalAIRVersion{26u, 4u, 0u}));

[[nodiscard]] MetalLibTarget metallib_target_for_air(
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
    LUISA_ERROR_WITH_LOCATION("Invalid Metal AIR target platform.");
}

void optimize_llvm_module(llvm::Module &module) noexcept {
    llvm::LoopAnalysisManager loop_analysis;
    llvm::FunctionAnalysisManager function_analysis;
    llvm::CGSCCAnalysisManager cgscc_analysis;
    llvm::ModuleAnalysisManager module_analysis;
    llvm::PipelineTuningOptions tuning;
    // Apple metalfe keeps ordinary constant-trip loops in AIR and leaves the
    // final GPU compiler to choose its machine-level policy. Full LLVM loop
    // unrolling substantially increased AIR size and register pressure for
    // arithmetic kernels, so keep the same division of responsibility here.
    tuning.LoopUnrolling = false;
    llvm::PassBuilder pass_builder{nullptr, tuning};
    pass_builder.registerModuleAnalyses(module_analysis);
    pass_builder.registerCGSCCAnalyses(cgscc_analysis);
    pass_builder.registerFunctionAnalyses(function_analysis);
    pass_builder.registerLoopAnalyses(loop_analysis);
    pass_builder.crossRegisterProxies(
        loop_analysis, function_analysis, cgscc_analysis, module_analysis);
    auto pipeline = pass_builder.buildPerModuleDefaultPipeline(
        llvm::OptimizationLevel::O2);
    pipeline.run(module, module_analysis);
}

void verify_llvm_module(const llvm::Module &module, luisa::string_view phase) noexcept {
    std::string error;
    llvm::raw_string_ostream stream{error};
    if (llvm::verifyModule(module, &stream)) {
        stream.flush();
        LUISA_ERROR_WITH_LOCATION("Invalid Metal LLVM IR after {}: {}", phase, error);
    }
}

[[nodiscard]] bool dump_llvm_enabled() noexcept {
    if (auto value = std::getenv("LUISA_DUMP_LLVM_IR")) {
        return luisa::string_view{value} != "0";
    }
    return false;
}

void dump_llvm_module(const llvm::Module &module, luisa::string_view path) noexcept {
    std::error_code error;
    llvm::raw_fd_ostream stream{llvm::StringRef{path.data(), path.size()}, error};
    if (error) {
        LUISA_WARNING_WITH_LOCATION("Failed to dump Metal LLVM IR to '{}': {}.",
                                    path, error.message());
        return;
    }
    module.print(stream, nullptr, false, true);
}

struct AIRCodegenEntry {
    std::vector<std::byte> bitcode;
    luisa::vector<std::vector<std::byte>> intersection_bitcodes;
    luisa::vector<std::pair<luisa::string, luisa::string>> format_types;
    luisa::vector<luisa::string> intersection_functions;
    size_t root_argument_size;
};

[[nodiscard]] AIRCodegenEntry codegen_entry(
    const xir::Module &module, const MetalCodegenLLVMConfig &base_config,
    MetalAIRKernelEntry entry, luisa::string_view dump_suffix) noexcept {
    auto config = base_config;
    config.entry = entry;
    auto result = luisa_compute_metal_codegen_llvm(module, config);
    LUISA_ASSERT(result.module, "Metal XIR to LLVM code generation failed.");
    verify_llvm_module(*result.module, "XIR translation");
    optimize_llvm_module(*result.module);
    verify_llvm_module(*result.module, "LLVM optimization");
    if (dump_llvm_enabled()) {
        auto path = luisa::format("{}.{}.ll", base_config.source_file, dump_suffix);
        dump_llvm_module(*result.module, path);
    }
    luisa::vector<std::vector<std::byte>> intersection_bitcodes;
    intersection_bitcodes.reserve(result.intersection_functions.size());
    for (auto &&entry_name : result.intersection_functions) {
        auto intersection_module = llvm::CloneModule(*result.module);
        if (auto metadata = intersection_module->getNamedMetadata(
                "air.intersection")) {
            llvm::MDNode *selected_operand = nullptr;
            for (auto i = 0u; i < metadata->getNumOperands(); i++) {
                auto operand = metadata->getOperand(i);
                auto value = llvm::dyn_cast<llvm::ValueAsMetadata>(
                    operand->getOperand(0u));
                auto function = value == nullptr ? nullptr :
                                                     llvm::dyn_cast<llvm::Function>(
                                                         value->getValue());
                if (function != nullptr &&
                    function->getName() == entry_name) {
                    selected_operand = operand;
                    break;
                }
            }
            metadata->clearOperands();
            LUISA_ASSERT(selected_operand != nullptr,
                         "Missing AIR intersection metadata for '{}'.",
                         entry_name);
            metadata->addOperand(selected_operand);
        }
        if (auto metadata = intersection_module->getNamedMetadata(
                "air.kernel")) {
            metadata->clearOperands();
        }
        for (auto &function : *intersection_module) {
            if (!function.isDeclaration() &&
                function.getName() != entry_name) {
                function.setLinkage(llvm::GlobalValue::InternalLinkage);
            }
        }
        optimize_llvm_module(*intersection_module);
        verify_llvm_module(*intersection_module,
                           "intersection module extraction");
        if (dump_llvm_enabled()) {
            auto path = luisa::format(
                "{}.{}.{}.ll", base_config.source_file,
                dump_suffix, entry_name);
            dump_llvm_module(*intersection_module, path);
        }
        intersection_bitcodes.emplace_back(
            llvm_downgrade_to_14(std::move(intersection_module)));
    }
    if (auto metadata = result.module->getNamedMetadata(
            "air.intersection")) {
        metadata->clearOperands();
    }
    for (auto &&entry_name : result.intersection_functions) {
        if (auto intersection = result.module->getFunction(entry_name)) {
            intersection->eraseFromParent();
        }
    }
    return {.bitcode = llvm_downgrade_to_14(std::move(result.module)),
            .intersection_bitcodes = std::move(intersection_bitcodes),
            .format_types = std::move(result.format_types),
            .intersection_functions =
                std::move(result.intersection_functions),
            .root_argument_size = result.root_argument_size};
}

struct RasterAIRCodegenEntry {
    std::vector<std::byte> bitcode;
    size_t root_argument_size;
    uint32_t fragment_output_count;
};

[[nodiscard]] RasterAIRCodegenEntry codegen_raster_entry(
    const xir::Module &module, MetalCodegenLLVMConfig config,
    luisa::string_view dump_suffix) noexcept {
    luisa::string unsupported_reason;
    LUISA_ASSERT(
        luisa_compute_metal_codegen_llvm_supported(
            module, config, &unsupported_reason),
        "Metal raster XIR is unsupported by the AIR emitter: {}",
        unsupported_reason);
    auto result = luisa_compute_metal_codegen_llvm(module, config);
    LUISA_ASSERT(result.module,
                 "Metal raster XIR to LLVM code generation failed.");
    verify_llvm_module(*result.module, "raster XIR translation");
    optimize_llvm_module(*result.module);
    verify_llvm_module(*result.module, "raster LLVM optimization");
    if (dump_llvm_enabled()) {
        auto path = luisa::format("{}.{}.ll", config.source_file, dump_suffix);
        dump_llvm_module(*result.module, path);
    }
    return {
        .bitcode = llvm_downgrade_to_14(std::move(result.module)),
        .root_argument_size = result.root_argument_size,
        .fragment_output_count = result.fragment_output_count};
}

[[nodiscard]] const xir::RasterStageFunction *find_raster_stage(
    const xir::Module &module, xir::RasterStage expected) noexcept {
    const xir::RasterStageFunction *result = nullptr;
    for (auto function : module.function_list()) {
        if (!function->isa<xir::RasterStageFunction>()) { continue; }
        auto stage = static_cast<const xir::RasterStageFunction *>(function);
        LUISA_ASSERT(result == nullptr,
                     "Metal raster AIR module contains multiple stage entries.");
        LUISA_ASSERT(stage->stage() == expected,
                     "Metal raster AIR module has the wrong stage identity.");
        result = stage;
    }
    LUISA_ASSERT(result != nullptr,
                 "Metal raster AIR module has no stage entry.");
    return result;
}

}// namespace

MetalAIRTarget metal_air_target_for_ios(
    MetalAIRVersion operating_system_version,
    MetalAIRVersion sdk_version) noexcept {
    LUISA_ASSERT(operating_system_version.major >= 16u,
                 "Metal AIR iOS target requires iOS 16 or newer.");
    LUISA_ASSERT(!version_less(sdk_version, operating_system_version),
                 "Metal AIR iOS SDK {}.{}.{} is older than deployment target {}.{}.{}.",
                 sdk_version.major, sdk_version.minor, sdk_version.patch,
                 operating_system_version.major,
                 operating_system_version.minor,
                 operating_system_version.patch);
    return {
        .platform = MetalAIRPlatform::IOS,
        .operating_system_version = operating_system_version,
        .sdk_version = sdk_version};
}

MetalAIRTarget metal_air_target_for_current_device() noexcept {
#if defined(LUISA_PLATFORM_IOS) && LUISA_PLATFORM_IOS
    auto operating_system_version = host_ios_version();
    auto sdk_version = sdk_ios_version();
    // A signed application built with (for example) the iOS 26.4 SDK must
    // continue to generate AIR after the device receives an iOS 26.6 update.
    // The current device proves that its own minor/patch target is executable;
    // reject only a newer OS major, whose AIR/language ABI may be unknown to
    // this SDK and emitter. Explicit cross-target AOT retains the stricter
    // full-version ordering in metal_air_target_for_ios().
    LUISA_ASSERT(
        current_device_sdk_compatible(
            operating_system_version, sdk_version),
        "Metal AIR generation on iOS {}.{}.{} requires an SDK with major "
        "version {} or newer, but SDK {}.{}.{} is linked.",
        operating_system_version.major,
        operating_system_version.minor,
        operating_system_version.patch,
        operating_system_version.major,
        sdk_version.major,
        sdk_version.minor,
        sdk_version.patch);
    return {
        .platform = MetalAIRPlatform::IOS,
        .operating_system_version = operating_system_version,
        .sdk_version = sdk_version};
#else
    return {
        .platform = MetalAIRPlatform::MACOS,
        .operating_system_version = host_macos_version(),
        .sdk_version = sdk_macos_version()};
#endif
}

MetalCodegenLLVMConfig metal_air_codegen_config(
    const MetalAIRTarget &target,
    luisa::string source_file) noexcept {
    auto platform_version = target.operating_system_version;
    return {
        .platform = target.platform,
        .platform_version = platform_version,
        .sdk_version = target.sdk_version,
        .air_version = air_version_for_target(
            target.platform, platform_version.major),
        .metal_version = metal_version_for_target(
            target.platform, platform_version.major),
        .source_file = std::move(source_file)};
}

MetalAIRCodegenResult
metal_codegen_air(const xir::Module &module, const ShaderOption &option) noexcept {
    return metal_codegen_air(
        module, option, metal_air_target_for_current_device());
}

MetalAIRCodegenResult metal_codegen_air(
    const xir::Module &module, const ShaderOption &option,
    const MetalAIRTarget &target) noexcept {
    auto verification = xir::xir_verify_module(
        &module, {.require_reachable_blocks = true});
    if (!verification.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid XIR at Metal LLVM handoff: {} ({} error(s) total).",
            verification.errors.front().message,
            verification.errors.size());
    }
    auto source_file = option.name;
    if (source_file.empty()) {
        source_file = module.name().value_or("kernel");
    }
    auto config = metal_air_codegen_config(
        target, std::move(source_file));
    config.native_include = option.native_include;
    config.enable_fast_math = option.enable_fast_math;
    config.enable_extended_accel_limits =
        option.enable_extended_accel_limits;
    config.entry = MetalAIRKernelEntry::DIRECT;
    luisa::string unsupported_reason;
    LUISA_ASSERT(
        luisa_compute_metal_codegen_llvm_supported(
            module, config, &unsupported_reason),
        "Metal compute XIR is unsupported by the AIR emitter: {}",
        unsupported_reason);
    auto direct = codegen_entry(
        module, config, MetalAIRKernelEntry::DIRECT, "direct.air");
    auto indirect = codegen_entry(
        module, config, MetalAIRKernelEntry::INDIRECT, "indirect.air");
    LUISA_ASSERT(direct.format_types == indirect.format_types,
                 "Metal AIR direct and indirect printer format tables differ.");
    LUISA_ASSERT(direct.root_argument_size == indirect.root_argument_size,
                 "Metal AIR direct and indirect root argument layouts differ.");
    LUISA_ASSERT(
        direct.intersection_functions == indirect.intersection_functions,
        "Metal AIR direct and indirect intersection function tables differ.");

    auto library_target = metallib_target_for_air(target);
    luisa::vector<MetalLibFunction> functions;
    functions.reserve(2u + direct.intersection_functions.size());
    functions.emplace_back(MetalLibFunction{
        "kernel_main", direct.bitcode, MetalLibProgramType::KERNEL});
    functions.emplace_back(MetalLibFunction{
        "kernel_main_indirect", indirect.bitcode,
        MetalLibProgramType::KERNEL});
    for (auto i = 0u; i < direct.intersection_functions.size(); i++) {
        auto &&name = direct.intersection_functions[i];
        functions.emplace_back(MetalLibFunction{
            name, direct.intersection_bitcodes[i],
            MetalLibProgramType::INTERSECTION});
    }
    auto library = make_metallib(library_target, functions);
    luisa::vector<luisa::string_view> entry_points;
    luisa::vector<MetalLibProgramType> program_types;
    entry_points.reserve(functions.size());
    program_types.reserve(functions.size());
    for (auto &&function : functions) {
        entry_points.emplace_back(function.name);
        program_types.emplace_back(function.type);
    }
    LUISA_ASSERT(validate_metallib(library, entry_points, program_types),
                 "Generated Metal library failed structural validation.");
    return {.library = std::move(library),
            .format_types = std::move(direct.format_types),
            .intersection_functions =
                std::move(direct.intersection_functions),
            .root_argument_size = direct.root_argument_size};
}

MetalAIRRasterCodegenResult metal_codegen_air(
    const xir::Module &vertex_module,
    const xir::Module &fragment_module,
    const MeshFormat &mesh_format,
    const ShaderOption &option) noexcept {
    luisa::string mesh_format_reason;
    LUISA_ASSERT(validate_metal_raster_mesh_format(
                     mesh_format, &mesh_format_reason),
                 "Invalid Metal raster mesh format: {}", mesh_format_reason);
    auto verify_stage = [](const xir::Module &module,
                           luisa::string_view name) noexcept {
        auto verification = xir::xir_verify_module(
            &module, {.require_reachable_blocks = true});
        if (!verification.succeeded()) {
            LUISA_ERROR_WITH_LOCATION(
                "Invalid {} XIR at Metal raster LLVM handoff: {} "
                "({} error(s) total).",
                name, verification.errors.front().message,
                verification.errors.size());
        }
    };
    verify_stage(vertex_module, "vertex");
    verify_stage(fragment_module, "fragment");
    auto vertex = find_raster_stage(
        vertex_module, xir::RasterStage::VERTEX);
    auto fragment = find_raster_stage(
        fragment_module, xir::RasterStage::FRAGMENT);
    luisa::vector<const xir::Argument *> vertex_arguments;
    luisa::vector<const xir::Argument *> fragment_arguments;
    for (auto argument : vertex->arguments()) {
        vertex_arguments.emplace_back(argument);
    }
    for (auto argument : fragment->arguments()) {
        fragment_arguments.emplace_back(argument);
    }
    LUISA_ASSERT(!vertex_arguments.empty() && !fragment_arguments.empty(),
                 "Metal raster stages require payload arguments.");
    LUISA_ASSERT(
        vertex->type() == fragment_arguments.front()->type(),
        "Metal raster vertex return and fragment payload types differ.");

    MetalAIRRasterConfig raster_config;
    raster_config.root_arguments.reserve(
        vertex_arguments.size() + fragment_arguments.size() - 2u);
    for (auto argument : luisa::span{vertex_arguments}.subspan(1u)) {
        raster_config.root_arguments.emplace_back(argument);
    }
    auto fragment_root_offset = raster_config.root_arguments.size();
    for (auto argument : luisa::span{fragment_arguments}.subspan(1u)) {
        raster_config.root_arguments.emplace_back(argument);
    }
    raster_config.vertex_attributes.reserve(
        mesh_format.vertex_attribute_count());
    for (auto stream : mesh_format) {
        for (auto attribute : stream) {
            raster_config.vertex_attributes.emplace_back(
                MetalAIRRasterVertexAttribute{
                    .semantic = attribute.type,
                    .format = attribute.format});
        }
    }

    auto target = metal_air_target_for_current_device();
    auto source_file = option.name;
    if (source_file.empty()) { source_file = "raster"; }
    auto base_config = metal_air_codegen_config(target, source_file);
    base_config.native_include = option.native_include;
    base_config.enable_fast_math = option.enable_fast_math;
    base_config.entry = MetalAIRKernelEntry::DIRECT;
    base_config.program = MetalAIRProgram::RASTER_VERTEX;
    base_config.raster = raster_config;
    auto vertex_config = base_config;
    vertex_config.source_file.append(".vertex");
    auto vertex_air = codegen_raster_entry(
        vertex_module, std::move(vertex_config), "air");
    auto fragment_config = base_config;
    fragment_config.source_file.append(".fragment");
    fragment_config.program = MetalAIRProgram::RASTER_FRAGMENT;
    fragment_config.raster.stage_root_argument_offset =
        fragment_root_offset;
    auto fragment_air = codegen_raster_entry(
        fragment_module, std::move(fragment_config), "air");
    LUISA_ASSERT(
        vertex_air.root_argument_size == fragment_air.root_argument_size,
        "Metal raster stage root argument layouts differ.");
    LUISA_ASSERT(vertex_air.fragment_output_count == 0u,
                 "Metal AIR vertex stage unexpectedly reported fragment outputs.");
    LUISA_ASSERT(fragment_air.fragment_output_count <= 8u,
                 "Metal AIR fragment stage reported an invalid output count {}.",
                 fragment_air.fragment_output_count);

    auto library_target = metallib_target_for_air(target);
    std::array functions{
        MetalLibFunction{
            "vertex_main", vertex_air.bitcode,
            MetalLibProgramType::VERTEX},
        MetalLibFunction{
            "fragment_main", fragment_air.bitcode,
            MetalLibProgramType::FRAGMENT}};
    auto library = make_metallib(library_target, functions);
    std::array<luisa::string_view, 2u> entry_points{
        "vertex_main", "fragment_main"};
    constexpr std::array program_types{
        MetalLibProgramType::VERTEX,
        MetalLibProgramType::FRAGMENT};
    LUISA_ASSERT(validate_metallib(library, entry_points, program_types),
                 "Generated Metal raster library failed structural validation.");
    return {
        .library = std::move(library),
        .root_argument_size = vertex_air.root_argument_size,
        .fragment_output_count = fragment_air.fragment_output_count,
        .vertex_entry = "vertex_main",
        .fragment_entry = "fragment_main"};
}

}// namespace luisa::compute::metal
