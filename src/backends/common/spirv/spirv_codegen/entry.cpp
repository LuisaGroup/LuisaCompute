#include "entry.h"
#include "atomic_target_contract.h"
#include "dialect.h"
#include "optimizer.h"
#include "texture_sampling.h"
#include "utils.h"
#include "../../backend_print_code.h"
#include "../../env_flag.h"
#include <SPIRV/disassemble.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>

namespace lc::spirv {

namespace {

[[nodiscard]] bool profile_native_spirv() noexcept {
    return luisa::compute::detail::env_flag(
        "LUISA_VULKAN_PROFILE_COMPILATION");
}

[[nodiscard]] bool is_constant_ubo_element_layout_supported(
    const Type *type) noexcept {
    // The upload path below serializes one host value into each std140 array
    // element. That is exact for scalar, vector, and matrix values (the outer
    // array supplies the required 16-byte-rounded stride), except for logical
    // bools, which have no physical representation in a Uniform block.
    //
    // Structures and nested arrays require a recursive std140 layout and a
    // corresponding recursive serializer. Keep those as ordinary SPIR-V
    // constants until both halves of that ABI exist; memcpy-ing their host
    // layout into a differently decorated type is incorrect.
    if (type == nullptr) { return false; }
    switch (type->tag()) {
        case Type::Tag::BOOL: return false;
        case Type::Tag::VECTOR:
            return type->element() != nullptr &&
                   !type->element()->is_bool();
        case Type::Tag::MATRIX:
            return type->element() != nullptr;
        case Type::Tag::FLOAT8_E4M3:
        case Type::Tag::FLOAT8_E5M2:
        case Type::Tag::FLOAT16:
        case Type::Tag::FLOAT32:
        case Type::Tag::FLOAT64:
        case Type::Tag::INT8:
        case Type::Tag::UINT8:
        case Type::Tag::INT16:
        case Type::Tag::UINT16:
        case Type::Tag::INT32:
        case Type::Tag::UINT32:
        case Type::Tag::INT64:
        case Type::Tag::UINT64: return true;
        default: return false;
    }
}

}// namespace

void SpirvCodegenEntry::_require_target_feature(
    SpirvTargetFeatureMask feature, bool supported) noexcept {
    LUISA_ASSERT(
        feature != 0u &&
            (feature & (feature - 1u)) == 0u &&
            (feature & target_feature::known_mask) != 0u,
        "SPIR-V codegen tried to require an invalid target-feature bit 0x{:016x}.",
        feature);
    // One-to-one capability requirements are deliberately deferred until
    // after optimization. A feature use may be dead, and SPIRV-Tools can then
    // remove both the instruction and its capability. Runtime/layout and
    // lowering-owned requirements cannot be reconstructed that way, so keep
    // their immediate availability check. Planners that choose a physical
    // representation (float atomics and narrow constant UBOs) or validate a
    // runtime sampler contract may also select a fallback or reject before
    // this recorder is reached.
    if (!supported &&
        !spirv_target_feature_is_capability_owned(feature)) [[unlikely]] {
        LUISA_ERROR(
            "Vulkan XIR-to-SPIR-V codegen requires target feature '{}', "
            "but it is not enabled for this logical device.",
            spirv_target_feature_name(feature));
    }
    _required_target_features |= feature;
}

void SpirvCodegenEntry::_require_sampled_image_array_indexing(
    bool nonuniform) noexcept {
    // Bindless slots contain descriptor indices in runtime metadata. Even a
    // literal slot therefore produces a dynamically-uniform descriptor index;
    // only divergent slot selection needs the stronger nonuniform contract.
    if (nonuniform) {
        _require_target_feature(
            target_feature::sampled_image_array_non_uniform_indexing,
            _target_features.sampled_image_array_non_uniform_indexing);
        _builder.addIncorporatedExtension(
            "SPV_EXT_descriptor_indexing", spv::Spv_1_5);
        _builder.addCapability(spv::Capability::ShaderNonUniformEXT);
        _builder.addCapability(
            spv::Capability::SampledImageArrayNonUniformIndexingEXT);
    } else {
        _require_target_feature(
            target_feature::sampled_image_array_dynamic_indexing,
            _target_features.sampled_image_array_dynamic_indexing);
        _builder.addCapability(
            spv::Capability::SampledImageArrayDynamicIndexing);
    }
}

void SpirvCodegenEntry::_require_storage_buffer_array_indexing(
    bool nonuniform) noexcept {
    if (nonuniform) {
        _require_target_feature(
            target_feature::storage_buffer_array_non_uniform_indexing,
            _target_features.storage_buffer_array_non_uniform_indexing);
        _builder.addIncorporatedExtension(
            "SPV_EXT_descriptor_indexing", spv::Spv_1_5);
        _builder.addCapability(spv::Capability::ShaderNonUniformEXT);
        _builder.addCapability(
            spv::Capability::StorageBufferArrayNonUniformIndexingEXT);
    } else {
        _require_target_feature(
            target_feature::storage_buffer_array_dynamic_indexing,
            _target_features.storage_buffer_array_dynamic_indexing);
        _builder.addCapability(
            spv::Capability::StorageBufferArrayDynamicIndexing);
    }
}

void SpirvCodegenEntry::_require_subgroup_type(
    const Type *type, luisa::string_view operation) noexcept {
    LUISA_ASSERT(type != nullptr,
                 "SPIR-V subgroup operation '{}' has no value type.", operation);
    while (type->is_vector() || type->is_matrix()) {
        type = type->element();
    }
    LUISA_ASSERT(type->is_scalar(),
                 "SPIR-V subgroup operation '{}' does not support XIR type {}.",
                 operation, type->description());
    if (type->is_float8() || type->is_float64()) [[unlikely]] {
        LUISA_ERROR(
            "Vulkan subgroup operation '{}' does not support {} values. "
            "Vulkan subgroup extended types cover 8/16/64-bit integers and "
            "16-bit floats, but not 8-bit or 64-bit floats.",
            operation, type->description());
    }
    auto requires_extended_types =
        spirv_subgroup_type_requires_extended_types(type);
    if (requires_extended_types) {
        LUISA_ASSERT(
            _runtime_target_plan_installed &&
                _runtime_target_plan.uses_subgroup_extended_types &&
                _target_features.subgroup_extended_types,
            "SPIR-V subgroup extended type escaped runtime target preflight.");
        _require_target_feature(
            target_feature::subgroup_extended_types,
            _target_features.subgroup_extended_types);
    }
}

static void luisa_spirv_validate(luisa::span<const uint32_t> words, luisa::string_view stage) {
    auto report = validate_spirv(words.data(), words.size());
    if (!report.valid) {
        LUISA_ERROR("SPIR-V validation failed at {} stage:\n{}",
                    stage, report.diagnostics);
    }
    if (report.has_warning && !report.diagnostics.empty()) {
        LUISA_WARNING("SPIR-V validation diagnostics at {} stage:\n{}",
                      stage, report.diagnostics);
    }
}

vstd::vector<std::pair<Variable, Usage>>
SpirvCodegenEntry::_collect_kernel_argument_usages(Function kernel, const xir::Module *module) const noexcept {
    const xir::KernelFunction *xir_kernel = nullptr;
    for (auto f : module->function_list()) {
        if (f->derived_function_tag() == xir::DerivedFunctionTag::KERNEL) {
            xir_kernel = static_cast<const xir::KernelFunction *>(f);
            break;
        }
    }
    auto ast_args = kernel.arguments();
    vstd::vector<std::pair<Variable, Usage>> result;
    result.reserve(ast_args.size());
    luisa::vector<const xir::Argument *> xir_args;
    if (xir_kernel != nullptr) {
        for (auto arg : xir_kernel->arguments()) {
            xir_args.emplace_back(arg);
        }
    }
    for (auto i = 0u; i < ast_args.size(); i++) {
        auto ast_arg = ast_args[i];
        auto usage = kernel.variable_usage(ast_arg.uid());
        if (i < xir_args.size()) {
            auto xir_usage = spirv_function_argument_usage_of(
                _function_argument_usage, xir_kernel, xir_args[i]);
            if (ast_arg.type()->is_accel()) {
                // Native accel descriptors are an exact optimized-XIR plan.
                // Keeping dead AST reads here would make Usage disagree with
                // the persisted zero-role mask and manufacture a descriptor.
                usage = xir_usage;
            } else {
                // Other public resource usages remain conservative for runtime
                // synchronization while their optional descriptor roles are
                // planned independently from exact XIR.
                usage = static_cast<Usage>(
                    luisa::to_underlying(usage) |
                    luisa::to_underlying(xir_usage));
            }
        }
        result.emplace_back(ast_arg, usage);
    }
    return result;
}

vstd::vector<SpirvKernelArgumentRoleMask>
SpirvCodegenEntry::_collect_kernel_argument_roles(
    Function kernel, const xir::Module *module) const noexcept {
    const xir::KernelFunction *xir_kernel = nullptr;
    for (auto *function : module->function_list()) {
        if (function->derived_function_tag() ==
            xir::DerivedFunctionTag::KERNEL) {
            xir_kernel = static_cast<const xir::KernelFunction *>(function);
            break;
        }
    }
    auto ast_arguments = kernel.arguments();
    luisa::vector<const xir::Argument *> xir_arguments;
    if (xir_kernel != nullptr) {
        for (auto *argument : xir_kernel->arguments()) {
            xir_arguments.emplace_back(argument);
        }
    }
    LUISA_ASSERT(
        xir_kernel != nullptr &&
            xir_arguments.size() == ast_arguments.size(),
        "SPIR-V native argument-role planning requires an exact AST/XIR "
        "kernel ABI ({} AST arguments, {} XIR arguments).",
        ast_arguments.size(), xir_arguments.size());
    vstd::vector<SpirvKernelArgumentRoleMask> roles(
        ast_arguments.size(), kernel_argument_role::none);
    for (auto i = 0u; i < ast_arguments.size(); ++i) {
        auto ast_argument = ast_arguments[i];
        auto *xir_argument = xir_arguments[i];
        if (ast_argument.type()->is_accel()) {
            LUISA_ASSERT(
                xir_argument->type() != nullptr &&
                    xir_argument->type()->is_accel(),
                "SPIR-V native accel role at argument {} has no matching "
                "XIR accel argument.",
                i);
            if (spirv_function_argument_requires_accel_traversal_descriptor(
                    _function_argument_usage, xir_kernel, xir_argument)) {
                roles[i] |= kernel_argument_role::accel_traversal;
            }
            if (spirv_function_argument_requires_accel_instance_buffer(
                    _function_argument_usage, xir_kernel, xir_argument)) {
                roles[i] |= kernel_argument_role::accel_instance;
            }
        } else if (ast_argument.type()->is_buffer() ||
                   ast_argument.type()->is_bindless_array()) {
            if (spirv_function_argument_requires_buffer_device_address(
                    _function_argument_usage, xir_kernel, xir_argument)) {
                roles[i] |= kernel_argument_role::buffer_device_address;
            }
        }
    }
    return roles;
}

SpirvResult SpirvCodegenEntry::compile_spirv(
    Function kernel, const ShaderOption &opt,
    SpirvTargetFeatures target_features) {
    auto profile = profile_native_spirv();
    if (profile) {
        LUISA_INFO(
            "Vulkan native AST-to-XIR begin for kernel '{}'",
            kernel.name());
    }
    auto xir_module = luisa::compute::spirv::luisa_spirv_backend_translate_ast_to_xir(kernel, opt);
    if (profile) {
        LUISA_INFO(
            "Vulkan native AST-to-XIR finished for kernel '{}'",
            kernel.name());
    }
    return compile_spirv_xir(
        kernel, xir_module.get(), opt, target_features);
}

SpirvResult SpirvCodegenEntry::compile_spirv_xir(
    Function kernel, const xir::Module *xir_module,
    const ShaderOption &opt,
    SpirvTargetFeatures target_features) {
    auto profile = profile_native_spirv();
    Clock phase_clock;
    auto report_phase = [&](const char *phase) noexcept {
        if (profile) {
            LUISA_INFO(
                "Vulkan native SPIR-V phase '{}' kernel '{}': {:.3f} ms",
                phase, kernel.name(), phase_clock.toc());
        }
        phase_clock.tic();
    };
    if (profile) {
        LUISA_INFO("Vulkan native SPIR-V compile begin for kernel '{}'",
                   kernel.name());
    }
    LUISA_ASSERT(xir_module != nullptr,
                 "Cannot compile a null XIR module to SPIR-V.");
    auto kernel_abi = validate_spirv_xir_kernel_abi(kernel, xir_module);
    if (!kernel_abi.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid AST/XIR kernel ABI at direct SPIR-V codegen handoff: {}.",
            kernel_abi.diagnostic);
    }
    auto dialect = validate_spirv_xir_codegen_dialect(
        xir_module,
        {.release_assertions_are_no_op =
             !opt.enable_debug_info});
    if (!dialect.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid XIR at direct SPIR-V codegen handoff: {} "
            "({} diagnostic(s) total).",
            dialect.diagnostics.front().message,
            dialect.diagnostics.size());
    }
    report_phase("handoff validation");
    StringScratch scratch;
    SpirvCodegenEntry codegen{scratch, true};
    codegen._enable_fast_math = opt.enable_fast_math;
    codegen._enable_debug_info = opt.enable_debug_info;
    codegen._target_features = target_features;
    auto analysis = codegen._analyze_module_usage(xir_module);
    auto atomic_buffers = plan_spirv_atomic_buffers(
        luisa::span<const xir::Function *const>{
            analysis.used_functions_post_order.data(),
            analysis.used_functions_post_order.size()},
        {.target_features = &target_features});
    if (!atomic_buffers.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid atomic-buffer representation at direct SPIR-V codegen "
            "handoff: {} ({} diagnostic(s) total).",
            atomic_buffers.diagnostics.front().message,
            atomic_buffers.diagnostics.size());
    }
    auto atomic_target = validate_spirv_atomic_target_contract(
        luisa::span<const xir::Function *const>{
            analysis.used_functions_post_order.data(),
            analysis.used_functions_post_order.size()},
        target_features);
    if (!atomic_target.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid atomic target contract at direct SPIR-V codegen "
            "handoff: {} ({} diagnostic(s) total).",
            atomic_target.diagnostics.front().message,
            atomic_target.diagnostics.size());
    }
    auto sampler_target =
        validate_spirv_sampler_target_contract(
            luisa::span<const xir::Function *const>{
                analysis.used_functions_post_order.data(),
                analysis.used_functions_post_order.size()},
            target_features.sampler_anisotropy);
    if (!sampler_target.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid configured sampler at direct SPIR-V target handoff: "
            "{} ({} diagnostic(s) total).",
            sampler_target.diagnostics.front().message,
            sampler_target.diagnostics.size());
    }
    auto runtime_target = plan_spirv_runtime_target_contract(
        luisa::span<const xir::Function *const>{
            analysis.used_functions_post_order.data(),
            analysis.used_functions_post_order.size()},
        analysis.bindless_resources, target_features);
    if (!runtime_target.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid runtime target contract at direct SPIR-V codegen "
            "handoff: {} ({} diagnostic(s) total, missing mask "
            "0x{:016x}).",
            runtime_target.diagnostics.front().message,
            runtime_target.diagnostics.size(),
            runtime_target.missing_features);
    }
    codegen._install_runtime_target_plan(runtime_target.plan);
    codegen._install_atomic_buffer_plan(atomic_buffers);
    codegen._analyze_function_argument_usage(xir_module);
    auto argument_usages = codegen._collect_kernel_argument_usages(
        kernel, xir_module);
    auto argument_roles = codegen._collect_kernel_argument_roles(
        kernel, xir_module);
    report_phase("module and target analysis");

    for (auto c : analysis.used_constants) {
        if (auto t = c->type();
            t != nullptr && t->is_array() &&
            is_constant_ubo_element_layout_supported(t->element())) {
            codegen._ubo_array_constants.emplace_back(c);
        }
    }
    std::sort(
        codegen._ubo_array_constants.begin(),
        codegen._ubo_array_constants.end(),
        [](const xir::Constant *lhs, const xir::Constant *rhs) noexcept {
            auto lhs_type = lhs->type();
            auto rhs_type = rhs->type();
            if (lhs_type->description() != rhs_type->description()) {
                return lhs_type->description() < rhs_type->description();
            }
            if (lhs->hash() != rhs->hash()) {
                return lhs->hash() < rhs->hash();
            }
            auto size = lhs_type->size();
            auto order = std::memcmp(lhs->data(), rhs->data(), size);
            return order < 0;
        });

    auto *xir_kernel = static_cast<const xir::KernelFunction *>(
        analysis.used_functions_post_order.back());
    codegen.generate_binding(kernel, argument_usages, xir_kernel);
    report_phase("binding planning");
    codegen.emit(xir_module, kernel.bound_arguments(), {}, opt.native_include);
    report_phase("SPIR-V emission");
    std::vector<uint32_t> words;
    codegen._builder.dump(words);
    report_phase("SPIR-V serialization");
    if (luisa::compute::backend_print_code_enabled()) {
        std::ostringstream disasm;
        spv::Disassemble(disasm, words);
        LUISA_VERBOSE("=== PRE-VALIDATION SPIR-V for {} (size={}) ===\n{}", kernel.name(), words.size(), disasm.str());
    }
    // Keep the exact emitter output available when the mandatory validator
    // rejects it. Dumping after validation made LUISA_DUMP_SPV ineffective for
    // precisely the failures it is intended to diagnose.
    if (std::getenv("LUISA_DUMP_SPV")) {
        auto filename = luisa::format("/tmp/opencode/kernel_{:016x}.spv", kernel.hash());
        std::ofstream file(filename.c_str(), std::ios::binary);
        file.write(reinterpret_cast<const char *>(words.data()), words.size() * sizeof(uint32_t));
    }
    luisa_spirv_validate(words, "pre-optimization");
    report_phase("pre-optimization validation");
    auto optimizer_report = optimize_spirv(
        words, spirv_optimizer_options_from_environment());
    report_phase("SPIR-V optimization");
    if (!optimizer_report.attempted) {
        LUISA_INFO("SPIR-V optimization skipped (preset={})",
                   optimizer_report.effective_preset);
    } else if (!optimizer_report.succeeded) {
        LUISA_WARNING(
            "SPIR-V optimization preset '{}' failed; using the validated "
            "unoptimized binary.\n{}",
            optimizer_report.effective_preset,
            optimizer_report.diagnostics);
    } else {
        if (!optimizer_report.diagnostics.empty()) {
            LUISA_WARNING("SPIR-V optimizer diagnostics:\n{}",
                          optimizer_report.diagnostics);
        }
        LUISA_INFO(
            "SPIR-V optimization preset '{}' completed: {} -> {} words{}",
            optimizer_report.effective_preset,
            optimizer_report.input_word_count,
            optimizer_report.output_word_count,
            optimizer_report.changed ? " (changed)" : " (unchanged)");
    }
    luisa_spirv_validate(words, "post-optimization");
    report_phase("post-optimization validation");
    codegen._required_target_features = reconcile_spirv_target_features(
        words.data(), words.size(), codegen._required_target_features);
    auto feature_check = check_spirv_target_feature_requirements(
        codegen._required_target_features,
        target_features.enabled_mask());
    LUISA_ASSERT(feature_check.unknown_required_bits == 0u,
                 "SPIR-V codegen produced unknown target-feature requirements 0x{:016x}.",
                 feature_check.unknown_required_bits);
    if (feature_check.missing_required_bits != 0u) [[unlikely]] {
        auto missing = list_spirv_target_features(
            feature_check.missing_required_bits);
        LUISA_ASSERT(missing.count != 0u,
                     "SPIR-V target-feature check reported no named missing feature.");
        LUISA_ERROR(
            "Vulkan XIR-to-SPIR-V final optimized artifact requires target "
            "feature '{}', but it is not enabled for this logical device.",
            missing.features.front().name);
    }
    report_phase("target-feature reconciliation");
    if (profile) {
        LUISA_INFO("Vulkan native SPIR-V compile complete for kernel '{}'",
                   kernel.name());
    }
    LUISA_INFO("SPIR-V compilation successful, binary size: {} words, properties: {} binds",
               words.size(), codegen._properties.size());
    if (luisa::compute::backend_print_code_enabled()) {
        std::ostringstream disasm;
        spv::Disassemble(disasm, words);
        LUISA_INFO("=== Kernel: {} (size={}) ===\n{}", kernel.name(), words.size(), disasm.str());
    }
    auto printers = std::move(codegen).move_print_formats();
    auto props = std::move(codegen._properties);
    auto use_tex2d = codegen._use_tex2d_bindless;
    auto use_tex3d = codegen._use_tex3d_bindless;
    auto use_buffer = codegen._use_buffer_bindless;
    auto constant_ubo_data = std::move(codegen._constant_ubo_data);
    auto required_target_features = codegen._required_target_features;
    return SpirvResult{
        .spv_bin = std::move(words),
        .properties = std::move(props),
        .argument_usages = std::move(argument_usages),
        .argument_roles = std::move(argument_roles),
        .printers = std::move(printers),
        .constant_ubo_data = std::move(constant_ubo_data),
        .required_target_features = required_target_features,
        .useTex2DBindless = use_tex2d,
        .useTex3DBindless = use_tex3d,
        .useBufferBindless = use_buffer};
}
}// namespace lc::spirv
