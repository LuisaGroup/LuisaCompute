#include "metal_xir_pipeline.h"

#include <cstdlib>
#include <fstream>

#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/xir/passes/autodiff.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/inline.h>
#include <luisa/xir/passes/lower_ray_query_to_loop.h>
#include <luisa/xir/passes/lower_ray_query_to_pipeline.h>
#include <luisa/xir/passes/mem2reg.h>
#include <luisa/xir/passes/pass_pipeline.h>
#include <luisa/xir/passes/reg2mem.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <luisa/xir/passes/unused_callable_removal.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/ray_query.h>
#include <luisa/xir/instructions/resource.h>
#include <luisa/xir/metadata/curve_basis.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2text.h>
#include <luisa/xir/verifier.h>
#include <luisa/xir/instructions/alloca.h>

namespace luisa::compute::metal {

namespace {

[[nodiscard]] bool dump_xir_enabled() noexcept {
    if (auto value = std::getenv("LUISA_DUMP_XIR")) {
        return luisa::string_view{value} != "0";
    }
    return false;
}

void dump_xir(const xir::Module *module, luisa::string_view name) noexcept {
    std::ofstream file{luisa::string{name}.c_str()};
    file << xir::xir_to_text_translate(module, true);
}

void verify_xir_or_error(
    const xir::Module *module, luisa::string_view stage,
    const xir::XIRVerificationOptions &options = {}) noexcept {
    auto verification = xir::xir_verify_module(module, options);
    if (!verification.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Invalid XIR at Metal {}: {} ({} error(s) total).",
            stage, verification.errors.front().message,
            verification.errors.size());
    }
}

[[nodiscard]] bool has_autodiff_scope(xir::Module *module) noexcept {
    auto found = false;
    for (auto function : module->function_list()) {
        if (auto definition = function->definition()) {
            definition->traverse_instructions([&](xir::Instruction *instruction) noexcept {
                found |= instruction->derived_instruction_tag() == xir::DerivedInstructionTag::AUTODIFF_SCOPE;
            });
        }
    }
    return found;
}

[[nodiscard]] bool destructure_cfg(xir::Module *module, xir::PassReport &report) noexcept {
    auto result = xir::destructure_cfg_pass_run_on_module(module, &report);
    if (!result.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Metal XIR CFG destructuring failed (errors={}, leaked_blocks={}).",
            result.error_count, result.leaked_block_count);
    }
    return result.changed();
}

[[nodiscard]] bool lower_ray_query_loops(
    xir::Module *module, xir::PassReport &report) noexcept {
    auto result = xir::lower_ray_query_to_loop_pass_run_on_module(module, &report);
    if (!result.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Metal XIR rejected {} ray-query loop(s).", result.error_count);
    }
    return result.lowered_ray_query_loop_count > 0u;
}

[[nodiscard]] bool metal_ray_payload_type_supported(
    const Type *type) noexcept {
    if (type == nullptr) { return false; }
    switch (type->tag()) {
        case Type::Tag::BOOL: [[fallthrough]];
        case Type::Tag::INT8: [[fallthrough]];
        case Type::Tag::UINT8: [[fallthrough]];
        case Type::Tag::INT16: [[fallthrough]];
        case Type::Tag::UINT16: [[fallthrough]];
        case Type::Tag::INT32: [[fallthrough]];
        case Type::Tag::UINT32: [[fallthrough]];
        case Type::Tag::INT64: [[fallthrough]];
        case Type::Tag::UINT64: [[fallthrough]];
        case Type::Tag::FLOAT16: [[fallthrough]];
        case Type::Tag::FLOAT32: return true;
        case Type::Tag::VECTOR: [[fallthrough]];
        case Type::Tag::MATRIX: [[fallthrough]];
        case Type::Tag::ARRAY:
            return metal_ray_payload_type_supported(type->element());
        case Type::Tag::STRUCTURE:
            for (auto member : type->members()) {
                if (!metal_ray_payload_type_supported(member)) {
                    return false;
                }
            }
            return true;
        // MSL explicitly permits device pointers in ray_data. Buffers are a
        // {device pointer, size} value, and bindless arrays are an argument-
        // buffer device pointer; both therefore have a lossless payload ABI.
        case Type::Tag::BUFFER:
            return type->element() == nullptr ||
                   metal_ray_payload_type_supported(type->element());
        case Type::Tag::BINDLESS_ARRAY: return true;
        default: return false;
    }
}

[[nodiscard]] bool metal_ray_payload_capture_supported(
    const xir::Value *value, bool) noexcept {
    if (value == nullptr ||
        !metal_ray_payload_type_supported(value->type())) {
        return false;
    }
    // Copying a captured reference into its own ray-data field preserves one
    // local allocation's state across candidates. Reject arbitrary lvalues,
    // whose aliases could otherwise be split into independent payload fields.
    if (value->is_lvalue()) {
        return value->isa<xir::AllocaInst>() &&
               static_cast<const xir::AllocaInst *>(value)->is_local();
    }
    return true;
}

[[nodiscard]] size_t metal_ray_payload_capture_cost(
    const xir::Value *value, bool) noexcept {
    LUISA_ASSERT(value != nullptr && value->type() != nullptr,
                 "Invalid Metal ray-query payload capture.");
    auto type = value->type();
    if (!metal_ray_payload_type_supported(type)) {
        return std::numeric_limits<size_t>::max();
    }
    switch (type->tag()) {
        // AIR stores a buffer capture as {device pointer, uint64 size} and a
        // bindless capture as its argument-buffer device pointer.
        case Type::Tag::BUFFER: return 16u;
        case Type::Tag::BINDLESS_ARRAY: return 8u;
        default:
            LUISA_ASSERT(!type->is_resource() && !type->is_custom(),
                         "Unsupported Metal ray-query payload cost type '{}'.",
                         type->description());
            return type->size();
    }
}

[[nodiscard]] bool metal_ray_query_loop_supported(
    const xir::RayQueryLoopInst *loop) noexcept {
    if (loop == nullptr || loop->dispatch_block() == nullptr ||
        !loop->dispatch_block()->is_terminated()) {
        return false;
    }
    auto terminator = loop->dispatch_block()->terminator();
    if (!terminator->isa<xir::RayQueryDispatchInst>()) { return false; }
    auto dispatch = static_cast<const xir::RayQueryDispatchInst *>(terminator);
    auto query_object = dispatch->query_object();
    if (query_object == nullptr || !query_object->isa<xir::AllocaInst>() ||
        !static_cast<const xir::AllocaInst *>(query_object)->is_local()) {
        return false;
    }
    const xir::ResourceQueryInst *constructor = nullptr;
    for (auto use : query_object->use_list()) {
        auto user = use->user();
        if (user == nullptr || !user->isa<xir::StoreInst>()) { continue; }
        auto store = static_cast<const xir::StoreInst *>(user);
        if (store->variable() != query_object) { continue; }
        if (constructor != nullptr ||
            !store->value()->isa<xir::ResourceQueryInst>()) {
            return false;
        }
        constructor = static_cast<const xir::ResourceQueryInst *>(
            store->value());
    }
    if (constructor == nullptr) { return false; }
    auto is_ray_query =
        constructor->op() == xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL ||
        constructor->op() == xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY ||
        constructor->op() ==
            xir::ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR ||
        constructor->op() ==
            xir::ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR;
    if (!is_ray_query) { return false; }
    // Curve traversal requires Metal's curve_data intersector/query path; keep
    // curve ray queries on the stateful intersection_query loop per query while
    // allowing triangle and procedural ray queries in the same module to outline.
    auto basis = constructor->find_metadata<xir::CurveBasisMD>();
    if (basis != nullptr && basis->curve_basis_set().any()) {
        return false;
    }
    return true;
}

[[nodiscard]] xir::CallableFunction *clone_ray_query_handler_callable(
    xir::Module *module, const xir::CallableFunction *source) noexcept {
    struct MapCloneResolver final : xir::InstructionCloneValueResolver {
        luisa::unordered_map<const xir::Value *, xir::Value *> map;
        [[nodiscard]] xir::Value *resolve(const xir::Value *value) noexcept override {
            if (auto iter = map.find(value); iter != map.end()) {
                return iter->second;
            }
            return const_cast<xir::Value *>(value);
        }
    };
    auto cloned = module->create_callable(source->type());
    if (auto name = source->name()) {
        cloned->set_name(luisa::format("{}_clone", *name));
    }
    for (auto *metadata : source->metadata_list()) {
        cloned->metadata_list().push_front(metadata->clone());
    }
    MapCloneResolver resolver;
    for (auto arg : source->arguments()) {
        xir::Argument *new_arg = nullptr;
        if (arg->is_reference()) {
            new_arg = cloned->create_reference_argument(arg->type());
        } else if (arg->is_resource()) {
            new_arg = cloned->create_resource_argument(arg->type());
        } else {
            new_arg = cloned->create_value_argument(arg->type());
        }
        if (auto name = arg->name()) { new_arg->set_name(*name); }
        for (auto *metadata : arg->metadata_list()) {
            new_arg->metadata_list().push_front(metadata->clone());
        }
        resolver.map.try_emplace(arg, new_arg);
    }
    for (auto bb : source->basic_blocks()) {
        auto new_bb = cloned->create_basic_block();
        if (auto name = bb->name()) { new_bb->set_name(*name); }
        for (auto *metadata : bb->metadata_list()) {
            new_bb->metadata_list().push_front(metadata->clone());
        }
        resolver.map.try_emplace(bb, new_bb);
    }
    cloned->set_body_block(static_cast<xir::BasicBlock *>(
        resolver.resolve(source->body_block())));
    xir::XIRBuilder builder;
    for (auto bb : source->basic_blocks()) {
        auto new_bb = static_cast<xir::BasicBlock *>(resolver.resolve(bb));
        builder.set_insertion_point(new_bb);
        for (auto inst : bb->instructions()) {
            auto new_inst = inst->clone_with_metadata(builder, resolver);
            if (auto name = inst->name()) { new_inst->set_name(*name); }
            resolver.map.try_emplace(inst, new_inst);
        }
    }
    return cloned;
}

[[nodiscard]] bool deduplicate_ray_query_pipeline_handlers(
    xir::Module *module, xir::PassReport &report) noexcept {
    luisa::unordered_set<const xir::Function *> claimed_handlers;
    luisa::vector<xir::RayQueryPipelineInst *> pipelines;
    for (auto function : module->function_list()) {
        if (auto definition = function->definition()) {
            definition->traverse_instructions(
                [&pipelines](xir::Instruction *instruction) noexcept {
                    if (instruction->isa<xir::RayQueryPipelineInst>()) {
                        pipelines.emplace_back(
                            static_cast<xir::RayQueryPipelineInst *>(instruction));
                    }
                });
        }
    }
    size_t cloned_count = 0u;
    for (auto *pipeline : pipelines) {
        if (auto *surface = pipeline->on_surface_function();
            surface != nullptr && surface->isa<xir::CallableFunction>() &&
            !claimed_handlers.emplace(surface).second) {
            auto *cloned = clone_ray_query_handler_callable(
                module, static_cast<const xir::CallableFunction *>(surface));
            claimed_handlers.emplace(cloned);
            pipeline->set_on_surface_function(cloned);
            ++cloned_count;
        }
        if (auto *procedural = pipeline->on_procedural_function();
            procedural != nullptr && procedural->isa<xir::CallableFunction>() &&
            !claimed_handlers.emplace(procedural).second) {
            auto *cloned = clone_ray_query_handler_callable(
                module, static_cast<const xir::CallableFunction *>(procedural));
            claimed_handlers.emplace(cloned);
            pipeline->set_on_procedural_function(cloned);
            ++cloned_count;
        }
    }
    if (cloned_count > 0u) {
        report.set("cloned_shared_ray_query_handler", cloned_count);
    }
    return cloned_count > 0u;
}

[[nodiscard]] bool outline_ray_query_pipelines(
    xir::Module *module, xir::PassReport &report,
    MetalRayQueryPipelinePolicy policy) noexcept {
    size_t skipped_loop_count = 0u;
    xir::LowerRayQueryToPipelineOptions options;
    options.loop_filter = metal_ray_query_loop_supported;
    options.captured_argument_filter =
        metal_ray_payload_capture_supported;
    options.captured_argument_cost =
        metal_ray_payload_capture_cost;
    options.max_captured_argument_cost =
        policy.max_captured_payload_bytes;
    options.skipped_loop_count = &skipped_loop_count;
    auto result = xir::lower_ray_query_to_pipeline_pass_run_on_module(
        module, &report, options);
    if (!result.succeeded()) {
        LUISA_ERROR_WITH_LOCATION(
            "Metal XIR rejected {} ray-query pipeline candidate(s).",
            result.error_count);
    }
    if (skipped_loop_count > 0u) {
        report.set("retained_ray_query_loop", skipped_loop_count);
    }
    return result.lowered_loop_count > 0u;
}

[[nodiscard]] bool reg2mem(xir::Module *module, xir::PassReport &report) noexcept {
    auto result = xir::reg2mem_pass_run_on_module(module, &report);
    return result.lowered_phi_count > 0u ||
           result.lowered_cross_block_value_count > 0u;
}

[[nodiscard]] bool inline_all(
    xir::Module *module, xir::PassReport &report,
    xir::InlineOptions options = {}) noexcept {
    auto result = xir::inline_all_pass_run_on_module(module, options, &report);
    if (result.rejected_malformed_call_count != 0u) {
        LUISA_ERROR_WITH_LOCATION(
            "Metal XIR inlining rejected {} malformed call(s).",
            result.rejected_malformed_call_count);
    }
    return result.inlined_call_count > 0u;
}

[[nodiscard]] bool hoist_allocas_after_inlining(
    xir::Module *module, xir::PassReport &report) noexcept {
    auto hoisted_count = 0u;
    for (auto function : module->function_list()) {
        auto definition = function->definition();
        if (definition == nullptr || definition->body_block() == nullptr) {
            continue;
        }
        luisa::vector<xir::AllocaInst *> allocas;
        definition->traverse_instructions(
            [&allocas](xir::Instruction *instruction) noexcept {
                if (instruction->isa<xir::AllocaInst>()) {
                    allocas.emplace_back(
                        static_cast<xir::AllocaInst *>(instruction));
                }
            });
        auto insertion_point =
            definition->body_block()->instructions().head_sentinel();
        xir::XIRBuilder builder;
        builder.set_insertion_point(insertion_point);
        for (auto alloca : allocas) {
            if (alloca->parent_block() == definition->body_block() &&
                alloca->prev() == insertion_point) {
                insertion_point = alloca;
                builder.set_insertion_point(insertion_point);
                continue;
            }
            insertion_point = builder.append(alloca->remove_self());
            hoisted_count++;
        }
    }
    report.set("hoisted_alloca", hoisted_count);
    return hoisted_count > 0u;
}

void optimize_xir_for_air(
    xir::Module *module, const ShaderOption &option,
    MetalRayQueryPipelinePolicy ray_query_policy) noexcept {
    verify_xir_or_error(module, "AST translation");

    auto optimization_options = xir::OptimizationPipelineOptions{
        .enable_fast_math = option.enable_fast_math};

    auto basic = xir::create_basic_optimization_pipeline(optimization_options);
    auto basic_stats = basic.run(module);
    basic_stats.log("Metal XIR basic optimization");

    if (has_autodiff_scope(module)) {
        xir::PassPipeline pre_autodiff;
        if (option.enable_ray_query_pipeline && ray_query_policy.enabled) {
            pre_autodiff.add("outline-ray-query-pipelines",
                             [ray_query_policy](xir::Module *m, xir::PassReport &report) {
                                 return outline_ray_query_pipelines(
                                     m, report, ray_query_policy);
                             });
        }
        pre_autodiff.add("lower-ray-query-to-loop", lower_ray_query_loops);
        pre_autodiff.add("destructure-cfg-before-inline", destructure_cfg);
        pre_autodiff.add("inline-all-autodiff-callables", [](xir::Module *m, xir::PassReport &report) {
            return inline_all(
                m, report,
                {.allow_autodiff_scope_in_caller = true});
        });
        pre_autodiff.add("deduplicate-ray-query-handlers",
                         deduplicate_ray_query_pipeline_handlers);
        pre_autodiff.add_fixed_point(
            "post-inline-cleanup",
            xir::create_post_inline_cleanup_pipeline(optimization_options), 1u);
        pre_autodiff.add("simplify-cfg", [](xir::Module *m, xir::PassReport &report) {
            auto result = xir::simplify_cfg_pass_run_on_module(m, &report);
            return result.folded_constant_cond_br_count > 0u ||
                   result.folded_switch_count > 0u ||
                   result.threaded_empty_block_count > 0u ||
                   result.merged_straight_line_count > 0u ||
                   result.removed_unreachable_block_count > 0u;
        });
        pre_autodiff.add("reg2mem-before-restructure", reg2mem);
        pre_autodiff.add("restructure-cfg-before-autodiff", [](xir::Module *m, xir::PassReport &report) {
            auto result = xir::restructure_cfg_pass_run_on_module(m, &report);
            if (!result.succeeded()) {
                LUISA_ERROR_WITH_LOCATION(
                    "Metal XIR pre-autodiff restructuring failed "
                    "(irreducible={}, unstructured={}, invalid={}, iteration_limit={}).",
                    result.irreducible_region_count,
                    result.unstructured_branch_count,
                    result.invalid_construct_count,
                    result.iteration_limit_count);
            }
            return result.restructured_if_count > 0u ||
                   result.restructured_loop_count > 0u;
        });
        auto pre_autodiff_stats = pre_autodiff.run(module);
        pre_autodiff_stats.log("Metal XIR pre-autodiff normalization");
        verify_xir_or_error(
            module, "pre-autodiff normalization",
            {.require_no_phi = true, .require_unique_merge_blocks = true});

        xir::PassPipeline autodiff;
        autodiff.add("autodiff", [](xir::Module *m, xir::PassReport &report) {
            auto result = xir::autodiff_pass_run_on_module(m);
            report.set("transformed_scope", result.transformed_scope_count);
            report.set("removed_instruction", result.removed_instruction_count);
            return result.transformed_scope_count > 0u ||
                   result.removed_instruction_count > 0u;
        });
        autodiff.add("reg2mem-after-autodiff", reg2mem);
        auto autodiff_stats = autodiff.run(module);
        autodiff_stats.log("Metal XIR autodiff lowering");
        verify_xir_or_error(
            module, "autodiff lowering",
            {.require_no_phi = true, .require_unique_merge_blocks = true});
    }

    xir::PassPipeline lowering;
    if (option.enable_ray_query_pipeline && ray_query_policy.enabled) {
        lowering.add("outline-ray-query-pipelines",
                     [ray_query_policy](xir::Module *m, xir::PassReport &report) {
                         return outline_ray_query_pipelines(
                             m, report, ray_query_policy);
                     });
    }
    lowering.add("lower-ray-query-to-loop", lower_ray_query_loops);
    lowering.add("destructure-cfg", destructure_cfg);
    lowering.add("inline-all", [](xir::Module *m, xir::PassReport &report) {
        return inline_all(m, report);
    });
    lowering.add("deduplicate-ray-query-handlers",
                 deduplicate_ray_query_pipeline_handlers);
    // Multi-block callable inlining clones its function-local allocations into
    // the cloned region. AIR stateful intersection-query storage has function
    // lifetime and must be allocated/deallocated on every function exit, so
    // canonicalize all local allocations into the entry prefix before
    // mem2reg. This also keeps ordinary callable-local temporaries on the same
    // lifetime model already used by reg2mem and the IFT outliner.
    lowering.add("hoist-allocas-after-inline",
                 hoist_allocas_after_inlining);
    lowering.add("mem2reg", [](xir::Module *m, xir::PassReport &report) {
        auto result = xir::mem2reg_pass_run_on_module(m, &report);
        return result.promoted_alloca_count > 0u;
    });
    auto lowering_stats = lowering.run(module);
    lowering_stats.log("Metal XIR lowering");

    auto ssa = xir::create_ssa_optimization_pipeline(optimization_options);
    auto ssa_stats = ssa.run(module);
    ssa_stats.log("Metal XIR SSA optimization");

    xir::PassPipeline cleanup;
    cleanup.add("unused-callable-removal", [](xir::Module *m, xir::PassReport &report) {
        auto result = xir::unused_callable_removal_pass_run_on_module(m, &report);
        return result.removed_callable_count > 0u;
    });
    cleanup.add("simplify-cfg", [](xir::Module *m, xir::PassReport &report) {
        auto result = xir::simplify_cfg_pass_run_on_module(m, &report);
        return result.folded_constant_cond_br_count > 0u ||
               result.folded_switch_count > 0u ||
               result.threaded_empty_block_count > 0u ||
               result.merged_straight_line_count > 0u ||
               result.removed_unreachable_block_count > 0u;
    });
    auto cleanup_stats = cleanup.run(module);
    cleanup_stats.log("Metal XIR cleanup");
    verify_xir_or_error(
        module, "LLVM handoff",
        {.require_reachable_blocks = true});
}

}// namespace

luisa::unique_ptr<xir::Module>
metal_translate_ast_to_xir(
    Function kernel, const ShaderOption &option,
    MetalRayQueryPipelinePolicy ray_query_policy) noexcept {
    Clock translate_clock;
    auto module = xir::ast_to_xir_translate(kernel, {});
    module->set_name(luisa::format("kernel_{:016x}", kernel.hash()));
    if (!option.name.empty()) { module->set_location(option.name); }
    LUISA_VERBOSE("Metal AST to XIR translation done in {} ms.",
                  translate_clock.toc());

    if (dump_xir_enabled()) {
        dump_xir(module.get(), luisa::format("kernel.{:016x}.metal.xir", kernel.hash()));
    }
    optimize_xir_for_air(module.get(), option, ray_query_policy);

    if (dump_xir_enabled()) {
        dump_xir(module.get(), luisa::format("kernel.{:016x}.metal.opt.xir", kernel.hash()));
    }
    return module;
}

luisa::unique_ptr<xir::Module>
metal_translate_raster_ast_to_xir(
    Function stage_function, xir::RasterStage stage,
    const ShaderOption &option) noexcept {
    LUISA_ASSERT(
        stage_function.tag() == Function::Tag::RASTER_STAGE,
        "Metal raster AIR translation requires a raster-stage AST function.");
    LUISA_ASSERT(
        xir::RasterStageFunction::is_valid_stage(stage),
        "Metal raster AIR translation received an invalid stage role.");
    Clock translate_clock;
    auto module = xir::ast_to_xir_translate(
        stage_function, {.raster_stage = stage});
    auto stage_name = xir::to_string(stage);
    module->set_name(luisa::format(
        "raster_{}_{:016x}", stage_name, stage_function.hash()));
    if (!option.name.empty()) {
        module->set_location(luisa::format("{}.{}", option.name, stage_name));
    }
    LUISA_VERBOSE(
        "Metal raster {} AST to XIR translation done in {} ms.",
        stage_name, translate_clock.toc());

    if (dump_xir_enabled()) {
        dump_xir(
            module.get(),
            luisa::format(
                "raster.{}.{:016x}.metal.xir",
                stage_name, stage_function.hash()));
    }
    optimize_xir_for_air(module.get(), option, {});
    if (dump_xir_enabled()) {
        dump_xir(
            module.get(),
            luisa::format(
                "raster.{}.{:016x}.metal.opt.xir",
                stage_name, stage_function.hash()));
    }
    return module;
}

}// namespace luisa::compute::metal
