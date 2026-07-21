// Tests for the immutable native-XIR to SPIR-V control-flow plan, including
// physical loop boundaries, native OpPhi forwarding, and Switch layout rules.

#include "ut/ut.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

#include <spirv-tools/libspirv.hpp>

#include <luisa/ast/type_registry.h>
#include <luisa/dsl/sugar.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/mem2reg.h>
#include <luisa/xir/verifier.h>

#include "spirv_codegen/argument_usage.h"
#include "spirv_codegen/call_graph_validation.h"
#include "spirv_codegen/control_flow_plan.h"
#include "spirv_codegen/dialect.h"
#include "spirv_codegen/entry.h"
#include "spirv_codegen/structural_closure.h"
#include "spirv_codegen/utils.h"

using namespace boost::ut;
using namespace boost::ut::literals;
using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;

namespace {

void set_environment_variable(const char *name,
                              const char *value) noexcept {
#ifdef _WIN32
    _putenv_s(name, value == nullptr ? "" : value);
#else
    if (value == nullptr) {
        unsetenv(name);
    } else {
        setenv(name, value, 1);
    }
#endif
}

class ScopedEnvironmentVariable {
private:
    const char *_name;
    std::optional<std::string> _previous;

public:
    ScopedEnvironmentVariable(const char *name,
                              const char *value) noexcept
        : _name{name} {
        if (auto previous = std::getenv(name)) {
            _previous.emplace(previous);
        }
        set_environment_variable(name, value);
    }
    ~ScopedEnvironmentVariable() noexcept {
        set_environment_variable(
            _name, _previous ? _previous->c_str() : nullptr);
    }
    ScopedEnvironmentVariable(
        const ScopedEnvironmentVariable &) = delete;
    ScopedEnvironmentVariable &operator=(
        const ScopedEnvironmentVariable &) = delete;
};

[[nodiscard]] size_t count_opcode(
    luisa::span<const uint32_t> words, spv::Op expected) noexcept {
    auto count = size_t{0u};
    for (auto offset = size_t{5u}; offset < words.size();) {
        auto word_count = words[offset] >> 16u;
        if (word_count == 0u || word_count > words.size() - offset) {
            break;
        }
        auto opcode = static_cast<spv::Op>(words[offset] & 0xffffu);
        count += opcode == expected ? 1u : 0u;
        offset += word_count;
    }
    return count;
}

[[nodiscard]] std::optional<spv::Op> opcode_after_first(
    luisa::span<const uint32_t> words, spv::Op expected) noexcept {
    for (auto offset = size_t{5u}; offset < words.size();) {
        auto word_count = static_cast<size_t>(words[offset] >> 16u);
        if (word_count == 0u || word_count > words.size() - offset) {
            return std::nullopt;
        }
        auto opcode = static_cast<spv::Op>(words[offset] & 0xffffu);
        auto next = offset + word_count;
        if (opcode == expected) {
            if (next >= words.size()) { return std::nullopt; }
            auto next_word_count =
                static_cast<size_t>(words[next] >> 16u);
            if (next_word_count == 0u ||
                next_word_count > words.size() - next) {
                return std::nullopt;
            }
            return static_cast<spv::Op>(words[next] & 0xffffu);
        }
        offset = next;
    }
    return std::nullopt;
}

[[nodiscard]] bool validates(
    luisa::span<const uint32_t> words) noexcept {
    spvtools::SpirvTools tools{SPV_ENV_VULKAN_1_2};
    return tools.Validate(words.data(), words.size());
}

struct SpirvPhiRecord {
    uint32_t result_id{0u};
    uint32_t block_id{0u};
    luisa::vector<std::pair<uint32_t, uint32_t>> incomings;
};

struct SimpleLoopPhiRoutingOracle {
    size_t loop_merge_count{0u};
    size_t header_phi_count{0u};
    size_t header_incoming_count{0u};
    bool entry_incoming_is_initial{false};
    bool continue_incoming_found{false};
    bool continue_incoming_resolves_to_carried{false};
    size_t continue_forwarding_phi_depth{0u};
};

// The hand-built XIR shape below cannot be preserved reliably through the AST
// runtime path. Inspect the exact opt0 binary instead, including the physical
// predecessor IDs and every value carried through the synthetic Phi chain.
[[nodiscard]] SimpleLoopPhiRoutingOracle
inspect_simple_loop_phi_routing(
    luisa::span<const uint32_t> words,
    uint32_t initial_literal,
    uint32_t carried_literal) noexcept {
    SimpleLoopPhiRoutingOracle oracle;
    auto uint_type = uint32_t{0u};
    auto initial = uint32_t{0u};
    auto carried = uint32_t{0u};
    auto current_block = uint32_t{0u};
    auto loop_header = uint32_t{0u};
    auto continue_target = uint32_t{0u};
    luisa::vector<SpirvPhiRecord> phis;
    for (auto offset = size_t{5u}; offset < words.size();) {
        auto word_count = static_cast<size_t>(words[offset] >> 16u);
        if (word_count == 0u || word_count > words.size() - offset) {
            return {};
        }
        auto opcode = static_cast<spv::Op>(words[offset] & 0xffffu);
        switch (opcode) {
            case spv::Op::OpTypeInt:
                if (word_count == 4u && words[offset + 2u] == 32u &&
                    words[offset + 3u] == 0u) {
                    uint_type = words[offset + 1u];
                }
                break;
            case spv::Op::OpConstant:
                if (word_count == 4u && words[offset + 1u] == uint_type) {
                    if (words[offset + 3u] == initial_literal) {
                        initial = words[offset + 2u];
                    }
                    if (words[offset + 3u] == carried_literal) {
                        carried = words[offset + 2u];
                    }
                }
                break;
            case spv::Op::OpLabel:
                if (word_count == 2u) {
                    current_block = words[offset + 1u];
                }
                break;
            case spv::Op::OpFunctionEnd:
                current_block = 0u;
                break;
            case spv::Op::OpLoopMerge:
                oracle.loop_merge_count++;
                if (oracle.loop_merge_count == 1u && word_count >= 4u) {
                    loop_header = current_block;
                    continue_target = words[offset + 2u];
                }
                break;
            case spv::Op::OpPhi: {
                if (word_count < 5u || (word_count - 3u) % 2u != 0u) {
                    return {};
                }
                SpirvPhiRecord phi{
                    .result_id = words[offset + 2u],
                    .block_id = current_block};
                phi.incomings.reserve((word_count - 3u) / 2u);
                for (auto operand = size_t{3u}; operand < word_count;
                     operand += 2u) {
                    phi.incomings.emplace_back(
                        words[offset + operand],
                        words[offset + operand + 1u]);
                }
                phis.emplace_back(std::move(phi));
                break;
            }
            default: break;
        }
        offset += word_count;
    }

    auto find_phi = [&](uint32_t result_id) noexcept
        -> const SpirvPhiRecord * {
        auto iter = std::find_if(
            phis.begin(), phis.end(),
            [result_id](auto &&phi) noexcept {
                return phi.result_id == result_id;
            });
        return iter == phis.end() ? nullptr : &*iter;
    };
    auto forwarding_depth = [&](auto &&self, uint32_t value,
                                uint32_t expected,
                                luisa::vector<uint32_t> &active)
        -> std::optional<size_t> {
        if (value == expected) { return 0u; }
        if (std::find(active.begin(), active.end(), value) != active.end()) {
            return std::nullopt;
        }
        auto *phi = find_phi(value);
        if (phi == nullptr || phi->incomings.empty()) {
            return std::nullopt;
        }
        active.emplace_back(value);
        auto depth = size_t{0u};
        for (auto &&incoming_pair : phi->incomings) {
            auto incoming_depth = self(
                self, incoming_pair.first, expected, active);
            if (!incoming_depth) {
                active.pop_back();
                return std::nullopt;
            }
            depth = std::max(depth, *incoming_depth);
        }
        active.pop_back();
        return depth + 1u;
    };

    if (oracle.loop_merge_count != 1u || loop_header == 0u ||
        continue_target == 0u || initial == 0u || carried == 0u) {
        return oracle;
    }
    for (auto &&phi : phis) {
        if (phi.block_id != loop_header) { continue; }
        oracle.header_phi_count++;
        oracle.header_incoming_count = phi.incomings.size();
        for (auto &&[value, predecessor] : phi.incomings) {
            if (predecessor == continue_target) {
                oracle.continue_incoming_found = true;
                luisa::vector<uint32_t> active;
                if (auto depth = forwarding_depth(
                        forwarding_depth, value, carried, active)) {
                    oracle.continue_incoming_resolves_to_carried = true;
                    oracle.continue_forwarding_phi_depth = *depth;
                }
            } else if (value == initial) {
                oracle.entry_incoming_is_initial = true;
            }
        }
    }
    return oracle;
}

[[nodiscard]] luisa::vector<uint32_t> first_u32_switch_literals(
    luisa::span<const uint32_t> words) noexcept {
    for (auto offset = size_t{5u}; offset < words.size();) {
        auto word_count = words[offset] >> 16u;
        if (word_count == 0u || word_count > words.size() - offset) {
            break;
        }
        auto opcode = static_cast<spv::Op>(words[offset] & 0xffffu);
        if (opcode == spv::Op::OpSwitch) {
            luisa::vector<uint32_t> literals;
            for (auto operand = size_t{3u}; operand + 1u < word_count;
                 operand += 2u) {
                literals.emplace_back(words[offset + operand]);
            }
            return literals;
        }
        offset += word_count;
    }
    return {};
}

[[nodiscard]] lc::spirv::SpirvResult compile_exact_xir(
    luisa::compute::Function kernel, const Module *module) {
    ScopedEnvironmentVariable disable_spirv_optimization{
        "LUISA_SPIRV_OPT_LEVEL", "0"};
    ScopedEnvironmentVariable clear_spirv_pass_override{
        "LUISA_SPIRV_OPT_PASSES", nullptr};
    return lc::spirv::SpirvCodegenEntry::compile_spirv_xir(
        kernel, module, ShaderOption{.enable_cache = false});
}

[[nodiscard]] const lc::spirv::ControlFlowPlan::PhiIncomingPlan *
find_incoming(const lc::spirv::ControlFlowPlan::PhiPlan &plan,
              const BasicBlock *predecessor) noexcept {
    auto iter = std::find_if(
        plan.incomings.begin(), plan.incomings.end(),
        [predecessor](auto &&incoming) noexcept {
            return incoming.predecessor == predecessor;
        });
    return iter == plan.incomings.end() ? nullptr : &*iter;
}

void expect_single_trampoline_to(
    const lc::spirv::ControlFlowPlan &plan,
    lc::spirv::ControlFlowPlan::Target target,
    lc::spirv::ControlFlowPlan::Target continuation) {
    expect(target.kind ==
           lc::spirv::ControlFlowPlan::Target::Kind::SYNTHETIC_BLOCK);
    if (target.kind !=
        lc::spirv::ControlFlowPlan::Target::Kind::SYNTHETIC_BLOCK) {
        return;
    }
    expect(target.synthetic_index < plan.synthetic_blocks().size());
    if (target.synthetic_index >= plan.synthetic_blocks().size()) { return; }
    auto &&synthetic = plan.synthetic_blocks()[target.synthetic_index];
    expect(synthetic.kind ==
           lc::spirv::ControlFlowPlan::SyntheticBlockKind::EDGE_TRAMPOLINE);
    expect(synthetic.continuation == continuation);
}

}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    "spirv_plan_function_entry_boundary_is_testable"_test = [] {
        Module valid_module;
        auto *valid_function =
            valid_module.create_callable(Type::of<int32_t>());
        auto *valid_entry = valid_function->create_body_block();
        auto *one = valid_module.create_constant_one(Type::of<int32_t>());
        XIRBuilder builder;
        builder.set_insertion_point(valid_entry);
        builder.return_(one);
        auto valid =
            lc::spirv::ControlFlowPlan::validate_function_entry_boundary(
                valid_function);
        expect(valid.succeeded());
        expect(valid.logical_predecessor_count == 0u);
        expect(valid.phi_count == 0u);
        auto valid_plan =
            lc::spirv::ControlFlowPlan::create(valid_function);
        expect(valid_plan.block(valid_entry).has_role(lc::spirv::ControlFlowPlan::BlockRole::FUNCTION_ENTRY));

        // Generic XIR can represent a cyclic body entry, but the body is bound
        // to the backend-owned first SPIR-V block (or the kernel prologue's
        // continuation), so both the predecessor and OpPhi are illegal at this
        // narrower handoff.
        Module invalid_module;
        auto *invalid_function =
            invalid_module.create_callable(Type::of<int32_t>());
        auto *invalid_entry = invalid_function->create_body_block();
        auto *latch = invalid_function->create_basic_block();
        auto *initial =
            invalid_module.create_constant_one(Type::of<int32_t>());
        builder.set_insertion_point(invalid_entry);
        builder.phi(Type::of<int32_t>(), {{initial, latch}});
        builder.br(latch);
        builder.set_insertion_point(latch);
        builder.br(invalid_entry);
        auto invalid =
            lc::spirv::ControlFlowPlan::validate_function_entry_boundary(
                invalid_function);
        expect(!invalid.succeeded());
        expect(invalid.logical_predecessor_count == 1u);
        expect(invalid.phi_count == 1u);

        // A true orphan is not part of the emitted executable/structural
        // closure. Its dead edge must not make codegen legality depend on an
        // optional DCE pass.
        Module orphan_module;
        auto *orphan_function =
            orphan_module.create_callable(Type::of<int32_t>());
        auto *orphan_entry = orphan_function->create_body_block();
        auto *orphan = orphan_function->create_basic_block();
        auto *orphan_one =
            orphan_module.create_constant_one(Type::of<int32_t>());
        builder.set_insertion_point(orphan_entry);
        builder.return_(orphan_one);
        builder.set_insertion_point(orphan);
        builder.br(orphan_entry);
        expect(xir_verify_module(&orphan_module).succeeded());
        auto orphan_boundary =
            lc::spirv::ControlFlowPlan::validate_function_entry_boundary(
                orphan_function);
        expect(orphan_boundary.succeeded());
        auto orphan_plan =
            lc::spirv::ControlFlowPlan::create(orphan_function);
        expect(orphan_plan.blocks().size() == 1u);
        expect(orphan_plan.blocks().front().block == orphan_entry);
    };

    "spirv_plan_physical_loop_boundary_rejects_post_merge_backedge"_test = [] {
        using Facts =
            lc::spirv::ControlFlowPlan::PhysicalLoopPredecessorFacts;
        auto valid_predecessors = std::array{
            Facts{.dominated_by_header = false},
            Facts{.dominated_by_header = true,
                  .dominated_by_continue_target = true,
                  .dominated_by_merge_target = false}};
        auto valid =
            lc::spirv::ControlFlowPlan::validate_physical_loop_boundary(
                luisa::span<const Facts>{valid_predecessors.data(),
                                         valid_predecessors.size()});
        expect(valid.succeeded());
        expect(valid.entry_edge_count == 1u);
        expect(valid.backedge_edge_count == 1u);

        // Merely being dominated by P makes this a natural backedge. It is not
        // a legal SPIR-V backedge when it comes from after the declared merge
        // instead of from the declared continue construct.
        auto post_merge_predecessors = std::array{
            Facts{.dominated_by_header = false},
            Facts{.dominated_by_header = true,
                  .dominated_by_continue_target = false,
                  .dominated_by_merge_target = true}};
        auto post_merge =
            lc::spirv::ControlFlowPlan::validate_physical_loop_boundary(
                luisa::span<const Facts>{post_merge_predecessors.data(),
                                         post_merge_predecessors.size()});
        expect(!post_merge.succeeded());
        expect(post_merge.entry_edge_count == 1u);
        expect(post_merge.backedge_edge_count == 1u);
        expect(!post_merge.backedge_dominated_by_continue_target);
        expect(post_merge.backedge_dominated_by_merge_target);

        // Exercise the same classifier through the planner's resolved physical
        // graph. The declared update flows into the declared merge, and a block
        // after that merge branches back to prepare. This has exactly one
        // natural backedge, but it is outside the SPIR-V continue construct.
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *entry = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *loop = builder.loop();
        auto *prepare = loop->create_prepare_block();
        auto *body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        auto *after_merge = kernel->create_basic_block();
        builder.set_insertion_point(prepare);
        builder.cond_br(condition, body, merge);
        builder.set_insertion_point(body);
        builder.br(update);
        builder.set_insertion_point(update);
        builder.br(merge);
        builder.set_insertion_point(merge);
        builder.br(after_merge);
        builder.set_insertion_point(after_merge);
        builder.br(prepare);
        expect(xir_verify_module(
                   &module, {.require_unique_merge_blocks = true})
                   .succeeded());
        auto graph_validation = lc::spirv::ControlFlowPlan::
            validate_function_physical_loop_boundaries(kernel);
        expect(!graph_validation.succeeded());
        expect(graph_validation.loops.size() == 1u);
        if (graph_validation.loops.size() == 1u) {
            auto &&invalid_loop = graph_validation.loops.front();
            expect(!invalid_loop.succeeded());
            expect(invalid_loop.entry_edge_count == 1u);
            expect(invalid_loop.backedge_edge_count == 1u);
            expect(!invalid_loop.backedge_dominated_by_continue_target);
            expect(invalid_loop.backedge_dominated_by_merge_target);
        }
        expect(!lc::spirv::validate_spirv_xir_codegen_dialect(&module)
                    .succeeded());
    };

    "spirv_plan_physical_loop_boundary_reports_missing_explicit_backedge"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *entry = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *loop = builder.loop();
        auto *prepare = loop->create_prepare_block();
        auto *body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        builder.set_insertion_point(prepare);
        builder.cond_br(condition, body, merge);
        builder.set_insertion_point(body);
        builder.break_(merge);
        // The declared continue role exists and is well-formed, but no
        // reachable path enters it. The query must report the missing physical
        // backedge rather than asserting on role reachability.
        builder.set_insertion_point(update);
        builder.br(prepare);
        builder.set_insertion_point(merge);
        builder.return_void();

        expect(xir_verify_module(
                   &module,
                   {.require_unique_merge_blocks = true,
                    .require_canonical_break_continue_targets = true})
                   .succeeded());
        auto validation = lc::spirv::ControlFlowPlan::
            validate_function_physical_loop_boundaries(kernel);
        expect(!validation.succeeded());
        expect(validation.loops.size() == 1u);
        if (validation.loops.size() == 1u) {
            auto &&invalid_loop = validation.loops.front();
            expect(!invalid_loop.succeeded());
            expect(invalid_loop.entry_edge_count == 1u);
            expect(invalid_loop.backedge_edge_count == 0u);
            expect(!invalid_loop.backedge_dominated_by_continue_target);
            expect(!invalid_loop.backedge_dominated_by_merge_target);
        }
        auto dialect =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(!dialect.succeeded())
            << "the nonfatal codegen boundary must reject the missing physical backedge before ControlFlowPlan::create";
    };

    "spirv_plan_simple_loop_boundary_rejects_external_body_entry"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *entry = kernel->create_body_block();
        XIRBuilder builder;

        // Both arms eventually execute the logical SimpleLoop body. Only the
        // true arm passes through the SimpleLoop owner; the false arm is an
        // illegal external entry. Physical edge normalization redirects that
        // false-arm edge to the synthetic continue block, so the synthetic
        // loop header has two entry edges and no header-dominated backedge.
        builder.set_insertion_point(entry);
        auto *outer = builder.if_(condition);
        auto *loop_owner = outer->create_true_block();
        auto *bypass = outer->create_false_block();
        auto *outer_merge = outer->create_merge_block();

        builder.set_insertion_point(loop_owner);
        auto *loop = builder.simple_loop();
        auto *body = loop->create_body_block();
        auto *loop_merge = loop->create_merge_block();
        builder.set_insertion_point(body);
        builder.br(loop_merge);
        builder.set_insertion_point(loop_merge);
        builder.br(outer_merge);
        builder.set_insertion_point(bypass);
        builder.br(body);
        builder.set_insertion_point(outer_merge);
        builder.return_void();

        expect(xir_verify_module(
                   &module,
                   {.require_unique_merge_blocks = true,
                    .require_canonical_break_continue_targets = true})
                   .succeeded())
            << "generic XIR permits the external body edge; the SPIR-V "
               "physical-boundary validator must reject it";
        auto validation = lc::spirv::ControlFlowPlan::
            validate_function_physical_loop_boundaries(kernel);
        expect(!validation.succeeded());
        expect(validation.loops.size() == 1u);
        if (validation.loops.size() == 1u) {
            auto &&invalid_loop = validation.loops.front();
            expect(!invalid_loop.succeeded());
            expect(invalid_loop.reachable_predecessor_count == 2u);
            expect(invalid_loop.entry_edge_count == 2u);
            expect(invalid_loop.backedge_edge_count == 0u);
            expect(!invalid_loop.backedge_dominated_by_continue_target);
            expect(!invalid_loop.backedge_dominated_by_merge_target);
        }
        expect(!lc::spirv::validate_spirv_xir_codegen_dialect(&module)
                    .succeeded());
    };

    "spirv_plan_separates_selection_merge_from_loop_update"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *lhs = kernel->create_value_argument(Type::of<int32_t>());
        auto *rhs = kernel->create_value_argument(Type::of<int32_t>());
        auto *entry = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *loop = builder.loop();
        auto *prepare = loop->create_prepare_block();
        auto *body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();

        builder.set_insertion_point(prepare);
        builder.cond_br(condition, body, loop_merge);
        builder.set_insertion_point(body);
        auto *selection = builder.if_(condition);
        auto *true_block = selection->create_true_block();
        auto *false_block = selection->create_false_block();
        selection->set_merge_block(update);
        builder.set_insertion_point(true_block);
        auto *true_exit = builder.br(update);
        builder.set_insertion_point(false_block);
        auto *false_exit = builder.br(update);
        builder.set_insertion_point(update);
        auto *phi = builder.phi(
            Type::of<int32_t>(),
            {{lhs, true_block}, {rhs, false_block}});
        builder.br(prepare);
        builder.set_insertion_point(loop_merge);
        builder.return_void();

        expect(xir_verify_module(
                   &module, {.require_unique_merge_blocks = true})
                   .succeeded());
        auto plan = lc::spirv::ControlFlowPlan::create(kernel);
        auto update_target =
            lc::spirv::ControlFlowPlan::Target::xir(update);
        auto &&selection_region = plan.if_region(selection);
        expect_single_trampoline_to(
            plan, selection_region.merge_target, update_target);
        expect(plan.edge_target(true_exit) ==
               selection_region.merge_target);
        expect(plan.edge_target(false_exit) ==
               selection_region.merge_target);
        expect(plan.loop_region(loop).continue_target == update_target)
            << "OpLoopMerge must retain the intrinsic update block";
        expect(plan.loop_region(loop).physical_header_predecessor_count == 2u);
        expect(plan.loop_region(loop).physical_boundary.succeeded());
        expect(plan.loop_region(loop)
                   .physical_boundary
                   .backedge_dominated_by_continue_target);
        expect(!plan.loop_region(loop)
                    .physical_boundary
                    .backedge_dominated_by_merge_target);

        auto &&phi_plan = plan.phi_plan(phi);
        expect(phi_plan.result_target == update_target);
        for (auto *predecessor : {true_block, false_block}) {
            auto *incoming = find_incoming(phi_plan, predecessor);
            expect(incoming != nullptr);
            if (incoming != nullptr) {
                expect(incoming->forwarding_synthetic_indices.size() == 1u);
                expect(incoming->forwarding_synthetic_indices.front() ==
                       selection_region.merge_target.synthetic_index);
            }
        }
    };

    "spirv_plan_separates_selection_merge_from_loop_prepare"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *initial = kernel->create_value_argument(Type::of<int32_t>());
        auto *lhs = kernel->create_value_argument(Type::of<int32_t>());
        auto *rhs = kernel->create_value_argument(Type::of<int32_t>());
        auto *entry = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *loop = builder.loop();
        auto *prepare = loop->create_prepare_block();
        auto *body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();

        builder.set_insertion_point(prepare);
        auto *phi = builder.phi(
            Type::of<int32_t>(),
            {{initial, entry}});
        builder.cond_br(condition, body, loop_merge);
        builder.set_insertion_point(body);
        builder.br(update);
        builder.set_insertion_point(update);
        auto *selection = builder.if_(condition);
        auto *true_block = selection->create_true_block();
        auto *false_block = selection->create_false_block();
        selection->set_merge_block(prepare);
        builder.set_insertion_point(true_block);
        auto *true_exit = builder.br(prepare);
        builder.set_insertion_point(false_block);
        auto *false_exit = builder.br(prepare);
        phi->add_incoming(lhs, true_block);
        phi->add_incoming(rhs, false_block);
        builder.set_insertion_point(loop_merge);
        builder.return_void();

        expect(xir_verify_module(
                   &module, {.require_unique_merge_blocks = true})
                   .succeeded());
        auto plan = lc::spirv::ControlFlowPlan::create(kernel);
        auto prepare_target =
            lc::spirv::ControlFlowPlan::Target::xir(prepare);
        auto &&loop_region = plan.loop_region(loop);
        auto &&selection_region = plan.if_region(selection);
        expect(loop_region.entry_target == prepare_target)
            << "loop entry must not be stolen by a nested selection merge";
        expect(loop_region.physical_header_predecessor_count == 2u)
            << "the structured continue region must coalesce to one backedge";
        expect(loop_region.physical_boundary.succeeded());
        expect_single_trampoline_to(
            plan, selection_region.merge_target, prepare_target);
        expect(plan.edge_target(true_exit) ==
               selection_region.merge_target);
        expect(plan.edge_target(false_exit) ==
               selection_region.merge_target);

        auto &&phi_plan = plan.phi_plan(phi);
        auto *entry_incoming = find_incoming(phi_plan, entry);
        expect(entry_incoming != nullptr);
        if (entry_incoming != nullptr) {
            expect(entry_incoming->forwarding_synthetic_indices.empty());
        }
        for (auto *predecessor : {true_block, false_block}) {
            auto *incoming = find_incoming(phi_plan, predecessor);
            expect(incoming != nullptr);
            if (incoming != nullptr) {
                expect(incoming->forwarding_synthetic_indices.size() == 1u);
                expect(incoming->forwarding_synthetic_indices.front() ==
                       selection_region.merge_target.synthetic_index);
            }
        }
    };

    "spirv_plan_isolates_merge_from_external_bypass_predecessor"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *outer_condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *inner_condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *lhs = kernel->create_value_argument(Type::of<int32_t>());
        auto *rhs = kernel->create_value_argument(Type::of<int32_t>());
        auto *bypass = kernel->create_value_argument(Type::of<int32_t>());
        auto *entry = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *outer = builder.if_(outer_condition);
        auto *inner_header = outer->create_true_block();
        auto *outer_false = outer->create_false_block();
        auto *outer_merge = outer->create_merge_block();
        auto *join = kernel->create_basic_block();

        builder.set_insertion_point(inner_header);
        auto *inner = builder.if_(inner_condition);
        auto *inner_true = inner->create_true_block();
        auto *inner_false = inner->create_false_block();
        inner->set_merge_block(join);
        builder.set_insertion_point(inner_true);
        auto *true_exit = builder.br(join);
        builder.set_insertion_point(inner_false);
        auto *false_exit = builder.br(join);
        builder.set_insertion_point(outer_false);
        auto *bypass_exit = builder.br(join);
        builder.set_insertion_point(join);
        auto *phi = builder.phi(
            Type::of<int32_t>(),
            {{lhs, inner_true},
             {rhs, inner_false},
             {bypass, outer_false}});
        builder.br(outer_merge);
        builder.set_insertion_point(outer_merge);
        builder.return_void();

        expect(xir_verify_module(
                   &module, {.require_unique_merge_blocks = true})
                   .succeeded());
        auto plan = lc::spirv::ControlFlowPlan::create(kernel);
        auto join_target =
            lc::spirv::ControlFlowPlan::Target::xir(join);
        auto &&inner_region = plan.if_region(inner);
        expect_single_trampoline_to(
            plan, inner_region.merge_target, join_target);
        expect(plan.edge_target(true_exit) == inner_region.merge_target);
        expect(plan.edge_target(false_exit) == inner_region.merge_target);
        expect(plan.edge_target(bypass_exit) == join_target)
            << "a bypass predecessor must retain the logical merge target";

        auto &&phi_plan = plan.phi_plan(phi);
        for (auto *predecessor : {inner_true, inner_false}) {
            auto *incoming = find_incoming(phi_plan, predecessor);
            expect(incoming != nullptr);
            if (incoming != nullptr) {
                expect(incoming->forwarding_synthetic_indices.size() == 1u);
                expect(incoming->forwarding_synthetic_indices.front() ==
                       inner_region.merge_target.synthetic_index);
            }
        }
        auto *bypass_incoming = find_incoming(phi_plan, outer_false);
        expect(bypass_incoming != nullptr);
        if (bypass_incoming != nullptr) {
            expect(bypass_incoming->forwarding_synthetic_indices.empty());
        }
    };

    "spirv_nested_exit_merge_dependency_stays_reachable"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *outer_condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *inner_condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *entry = kernel->create_body_block();
        XIRBuilder builder;

        // The inner true arm is also the enclosing selection's merge. It then
        // flows to the inner merge. glslang's structured traversal used to
        // delay the enclosing merge, misclassify the dominated inner merge as
        // dead, rewrite its live payload to OpUnreachable, and serialize the
        // inner merge before its dominator.
        builder.set_insertion_point(entry);
        auto *outer = builder.if_(outer_condition);
        auto *outer_true = outer->create_true_block();
        auto *outer_false = outer->create_false_block();
        auto *outer_merge = outer->create_merge_block();
        builder.set_insertion_point(outer_true);
        builder.return_void();
        builder.set_insertion_point(outer_false);
        auto *inner = builder.if_(inner_condition);
        inner->set_true_target(outer_merge);
        auto *inner_false = inner->create_false_block();
        auto *inner_merge = inner->create_merge_block();
        builder.set_insertion_point(inner_false);
        builder.return_void();
        builder.set_insertion_point(outer_merge);
        builder.br(inner_merge);
        builder.set_insertion_point(inner_merge);
        builder.return_void();

        expect(xir_verify_module(
                   &module, {.require_unique_merge_blocks = true})
                   .succeeded());
        expect(lc::spirv::validate_spirv_xir_codegen_dialect(&module)
                   .succeeded());
        auto plan = lc::spirv::ControlFlowPlan::create(kernel);
        expect(plan.nested_selection_merge_rotations().size() == 1u);
        expect(plan.if_region(outer).merge_target ==
               lc::spirv::ControlFlowPlan::Target::xir(inner_merge));
        expect(plan.if_region(inner).merge_target ==
               lc::spirv::ControlFlowPlan::Target::xir(outer_merge));

        Kernel1D ast_kernel = [](Bool, Bool) noexcept {};
        kernel->set_block_size(
            ast_kernel.function()->function().block_size());
        auto compiled = compile_exact_xir(
            ast_kernel.function()->function(), &module);
        auto words = luisa::span{compiled.spv_bin};
        expect(validates(words));
        expect(count_opcode(words, spv::Op::OpUnreachable) == 0u)
            << "both nested merges are reachable on the inner true path";
    };

    "spirv_plan_separates_selection_merge_from_simple_loop_body"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *initial = kernel->create_value_argument(Type::of<int32_t>());
        auto *lhs = kernel->create_value_argument(Type::of<int32_t>());
        auto *rhs = kernel->create_value_argument(Type::of<int32_t>());
        auto *entry = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *loop = builder.simple_loop();
        auto *body = loop->create_body_block();
        auto *loop_merge = loop->create_merge_block();
        auto *selection_header = kernel->create_basic_block();

        builder.set_insertion_point(body);
        auto *phi = builder.phi(
            Type::of<int32_t>(),
            {{initial, entry}});
        builder.br(selection_header);
        builder.set_insertion_point(selection_header);
        auto *selection = builder.if_(condition);
        auto *true_block = selection->create_true_block();
        auto *false_block = selection->create_false_block();
        selection->set_merge_block(body);
        builder.set_insertion_point(true_block);
        auto *true_exit = builder.br(body);
        builder.set_insertion_point(false_block);
        auto *false_exit = builder.br(body);
        phi->add_incoming(lhs, true_block);
        phi->add_incoming(rhs, false_block);
        builder.set_insertion_point(loop_merge);
        builder.return_void();

        expect(xir_verify_module(
                   &module, {.require_unique_merge_blocks = true})
                   .succeeded());
        auto plan = lc::spirv::ControlFlowPlan::create(kernel);
        auto &&loop_region = plan.simple_loop_region(loop);
        auto continue_target =
            lc::spirv::ControlFlowPlan::Target::synthetic(
                loop_region.continue_synthetic_index);
        auto header_target =
            lc::spirv::ControlFlowPlan::Target::synthetic(
                loop_region.header_synthetic_index);
        expect(loop_region.physical_header_predecessor_count == 2u);
        expect(loop_region.physical_boundary.succeeded());
        expect(loop_region.physical_boundary.entry_edge_count == 1u);
        expect(loop_region.physical_boundary.backedge_edge_count == 1u);
        expect(loop_region.physical_boundary
                   .backedge_dominated_by_continue_target);
        expect(!loop_region.physical_boundary
                    .backedge_dominated_by_merge_target);
        auto &&selection_region = plan.if_region(selection);
        expect_single_trampoline_to(
            plan, selection_region.merge_target, continue_target);
        expect(plan.edge_target(true_exit) ==
               selection_region.merge_target);
        expect(plan.edge_target(false_exit) ==
               selection_region.merge_target);

        auto &&phi_plan = plan.phi_plan(phi);
        expect(phi_plan.result_target == header_target);
        auto *entry_incoming = find_incoming(phi_plan, entry);
        expect(entry_incoming != nullptr);
        if (entry_incoming != nullptr) {
            expect(entry_incoming->forwarding_synthetic_indices.empty());
        }
        for (auto *predecessor : {true_block, false_block}) {
            auto *incoming = find_incoming(phi_plan, predecessor);
            expect(incoming != nullptr);
            if (incoming != nullptr) {
                expect(incoming->forwarding_synthetic_indices.size() == 2u);
                expect(incoming->forwarding_synthetic_indices[0u] ==
                       selection_region.merge_target.synthetic_index);
                expect(incoming->forwarding_synthetic_indices[1u] ==
                       loop_region.continue_synthetic_index);
            }
        }
    };

    "spirv_post_restructure_boundary_preserves_constant_loop_prepare"_test = [] {
        for (auto condition_value : std::array{false, true}) {
            Module module;
            auto *function =
                module.create_callable(Type::of<int32_t>());
            auto *initial =
                function->create_value_argument(Type::of<int32_t>());
            auto *entry = function->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(entry);
            auto *loop = builder.loop();
            auto *prepare = loop->create_prepare_block();
            auto *body = loop->create_body_block();
            auto *update = loop->create_update_block();
            auto *merge = loop->create_merge_block();
            auto *condition = condition_value ?
                                  module.create_constant_one(Type::of<bool>()) :
                                  module.create_constant_zero(Type::of<bool>());
            builder.set_insertion_point(prepare);
            auto *prepare_branch =
                builder.cond_br(condition, body, merge);
            builder.set_insertion_point(body);
            auto *body_branch = builder.br(update);
            builder.set_insertion_point(update);
            auto *update_branch = builder.br(prepare);
            builder.set_insertion_point(merge);
            auto *merge_return = builder.return_(initial);
            expect(xir_verify_module(
                       &module,
                       {.require_unique_merge_blocks = true,
                        .require_canonical_break_continue_targets = true})
                       .succeeded());

            auto boundary =
                spirv::create_spirv_codegen_post_restructure_pipeline();
            [[maybe_unused]] auto stats = boundary.run(&module);

            expect(entry->terminator() == loop);
            expect(loop->prepare_block() == prepare);
            expect(loop->body_block() == body);
            expect(loop->update_block() == update);
            expect(loop->merge_block() == merge);
            expect(prepare->terminator() == prepare_branch);
            expect(prepare->terminator()->isa<ConditionalBranchInst>());
            expect(body->terminator() == body_branch);
            expect(update->terminator() == update_branch);
            expect(merge->terminator() == merge_return);
            expect(xir_verify_module(
                       &module,
                       {.require_unique_merge_blocks = true,
                        .require_canonical_break_continue_targets = true})
                       .succeeded());

            auto plan =
                lc::spirv::ControlFlowPlan::create(function);
            expect(plan.loop_region(loop).physical_boundary.succeeded());
        }
    };

    "spirv_inactive_payload_cleanup_preserves_live_loop_structure"_test = [] {
        Module module;
        auto *function = module.create_callable(Type::of<int32_t>());
        auto *initial = function->create_value_argument(Type::of<int32_t>());
        auto *entry = function->create_body_block();
        auto *orphan = function->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *slot = builder.alloca_local(Type::of<int32_t>());
        builder.store(slot, initial);
        auto *loop = builder.loop();
        auto *prepare = loop->create_prepare_block();
        auto *body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        auto *always = module.create_constant_one(Type::of<bool>());
        builder.set_insertion_point(prepare);
        auto *condition = builder.cond_br(always, body, merge);
        builder.set_insertion_point(body);
        builder.br(update);
        builder.set_insertion_point(update);
        builder.br(prepare);
        builder.set_insertion_point(merge);
        auto *result = builder.load(Type::of<int32_t>(), slot);
        builder.return_(result);

        builder.set_insertion_point(orphan);
        auto *dead_load = builder.load(Type::of<int32_t>(), slot);
        builder.store(slot, dead_load);
        builder.unreachable_();
        expect(xir_verify_module(&module).succeeded());

        auto cleared =
            spirv::clear_spirv_codegen_inactive_block_payloads(&module);
        expect(cleared.cleared_block_count == 1u);
        expect(cleared.cleared_true_orphan_block_count == 1u);
        expect(cleared.cleared_disconnected_role_block_count == 0u);
        expect(cleared.removed_instruction_count == 3u);
        expect(prepare->terminator() == condition)
            << "inactive-payload cleanup must not fold the canonical Loop.prepare edge";
        expect(prepare->terminator()->isa<ConditionalBranchInst>());
        expect(loop->body_block() == body);
        expect(loop->update_block() == update);
        expect(loop->merge_block() == merge);
        expect(orphan->instructions().count_size() == 1u);
        expect(orphan->terminator()->isa<UnreachableInst>());

        auto promoted = mem2reg_pass_run_on_function(function);
        expect(promoted.promoted_alloca_count == 1u)
            << "dead true-orphan users must not pin a reg2mem spill slot";
        expect(xir_verify_module(&module).succeeded());
    };

    "spirv_inactive_payload_cleanup_clears_dead_structural_merge_payload"_test = [] {
        Module module;
        auto *function = module.create_callable(Type::of<int32_t>());
        auto *condition =
            function->create_value_argument(Type::of<bool>());
        auto *initial =
            function->create_value_argument(Type::of<int32_t>());
        auto *entry = function->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *slot = builder.alloca_local(Type::of<int32_t>());
        builder.store(slot, initial);
        auto *selection = builder.if_(condition);
        auto *true_block = selection->create_true_block();
        auto *false_block = selection->create_false_block();
        auto *dead_merge = selection->create_merge_block();
        builder.set_insertion_point(true_block);
        auto *true_result = builder.load(Type::of<int32_t>(), slot);
        builder.return_(true_result);
        builder.set_insertion_point(false_block);
        auto *false_result = builder.load(Type::of<int32_t>(), slot);
        builder.return_(false_result);

        // The merge identity is required by IfInst, but both arms return, so
        // this payload has no ordinary predecessor and must not keep the spill
        // slot in memory form.
        builder.set_insertion_point(dead_merge);
        auto *dead_load = builder.load(Type::of<int32_t>(), slot);
        builder.store(slot, dead_load);
        builder.return_(dead_load);
        expect(xir_verify_module(&module).succeeded());

        auto cleared =
            spirv::clear_spirv_codegen_inactive_block_payloads(&module);
        expect(cleared.cleared_block_count == 1u);
        expect(cleared.cleared_true_orphan_block_count == 0u);
        expect(cleared.cleared_disconnected_role_block_count == 1u);
        expect(cleared.removed_instruction_count == 3u);
        expect(selection->merge_block() == dead_merge)
            << "the structural merge block identity must survive pruning";
        expect(dead_merge->instructions().count_size() == 1u);
        expect(dead_merge->terminator()->isa<UnreachableInst>());

        auto promoted = mem2reg_pass_run_on_function(function);
        expect(promoted.promoted_alloca_count == 1u);
        expect(xir_verify_module(&module).succeeded());
    };

    "spirv_post_restructure_cleanup_distinguishes_dead_role_and_true_orphan_payloads"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *buffer = kernel->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *zero =
            module.create_constant_zero(Type::of<uint32_t>());
        auto *one =
            module.create_constant_one(Type::of<uint32_t>());
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *selection = builder.if_(condition);
        auto *true_block = selection->create_true_block();
        auto *false_block = selection->create_false_block();
        auto *dead_merge = selection->create_merge_block();
        builder.set_insertion_point(true_block);
        builder.return_void();
        builder.set_insertion_point(false_block);
        builder.return_void();
        builder.set_insertion_point(dead_merge);
        builder.call(ResourceWriteOp::BUFFER_WRITE,
                     {buffer, zero, one});
        builder.return_void();

        auto *true_orphan = kernel->create_basic_block();
        builder.set_insertion_point(true_orphan);
        builder.call(ResourceWriteOp::BUFFER_WRITE,
                     {buffer, zero, one});
        builder.unreachable_();
        expect(xir_verify_module(&module).succeeded());

        auto boundary =
            spirv::create_spirv_codegen_post_restructure_pipeline();
        [[maybe_unused]] auto stats = boundary.run(&module);

        // Both writes are unreachable after restructuring, but the block
        // identities have different backend meanings: the dead merge remains
        // in exact emission's structural closure, while the true orphan does
        // not. Cleanup may erase both payloads, never either identity.
        expect(dead_merge->instructions().count_size() == 1u);
        expect(dead_merge->terminator()->isa<UnreachableInst>());
        expect(true_orphan->instructions().count_size() == 1u);
        expect(true_orphan->terminator()->isa<UnreachableInst>());
        auto closure =
            lc::spirv::plan_spirv_codegen_structural_closure(kernel);
        expect(closure.succeeded());
        expect(std::find(closure.blocks.begin(), closure.blocks.end(),
                         dead_merge) != closure.blocks.end());
        expect(std::find(closure.blocks.begin(), closure.blocks.end(),
                         true_orphan) == closure.blocks.end());
        expect(lc::spirv::validate_spirv_xir_codegen_dialect(&module)
                   .succeeded());
    };

    "spirv_exact_xir_analyzes_dead_structural_payload_before_spirv_canonicalization"_test = [] {
        Module module;
        auto *callable = module.create_callable(Type::of<void>());
        callable->set_name("dead_merge_callee");
        auto *callable_condition =
            callable->create_value_argument(Type::of<bool>());
        auto *dispatch_size = module.create_dispatch_size();
        auto *zero3 = module.create_constant_zero(Type::of<uint3>());
        XIRBuilder builder;
        builder.set_insertion_point(callable->create_body_block());
        auto *callable_selection = builder.if_(callable_condition);
        auto *callable_true = callable_selection->create_true_block();
        auto *callable_false = callable_selection->create_false_block();
        auto *callable_dead_merge =
            callable_selection->create_merge_block();
        builder.set_insertion_point(callable_true);
        builder.return_void();
        builder.set_insertion_point(callable_false);
        builder.return_void();
        builder.set_insertion_point(callable_dead_merge);
        builder.call(Type::of<uint3>(), ArithmeticOp::BINARY_ADD,
                     {dispatch_size, zero3});
        builder.return_void();

        auto *kernel = module.create_kernel();
        auto *buffer = kernel->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *zero =
            module.create_constant_zero(Type::of<uint32_t>());
        auto *one =
            module.create_constant_one(Type::of<uint32_t>());
        auto *entry = kernel->create_body_block();
        builder.set_insertion_point(entry);
        auto *selection = builder.if_(condition);
        auto *true_block = selection->create_true_block();
        auto *false_block = selection->create_false_block();
        auto *dead_merge = selection->create_merge_block();
        builder.set_insertion_point(true_block);
        builder.return_void();
        builder.set_insertion_point(false_block);
        builder.return_void();
        builder.set_insertion_point(dead_merge);
        builder.call(nullptr, callable, {condition});
        std::array<Value *, 1u> atomic_indices{zero};
        builder.atomic_fetch_add(
            Type::of<uint32_t>(), buffer,
            luisa::span<Value *const>{atomic_indices}, one);
        builder.return_void();

        expect(xir_verify_module(
                   &module,
                   {.require_unique_merge_blocks = true,
                    .require_canonical_break_continue_targets = true})
                   .succeeded());
        auto dialect =
            lc::spirv::validate_spirv_xir_codegen_dialect(&module);
        expect(dialect.succeeded());

        auto closure =
            lc::spirv::plan_spirv_codegen_structural_closure(kernel);
        expect(closure.succeeded());
        auto dead_merge_is_disconnected_role = false;
        for (auto i = closure.ordinary_block_count;
             i < closure.blocks.size(); ++i) {
            dead_merge_is_disconnected_role |=
                closure.blocks[i] == dead_merge;
        }
        expect(dead_merge_is_disconnected_role);

        auto call_graph =
            lc::spirv::validate_spirv_reachable_call_graph(&module);
        expect(call_graph.succeeded());
        expect(eq(call_graph.functions_post_order.size(), 2u));
        if (call_graph.functions_post_order.size() == 2u) {
            expect(call_graph.functions_post_order[0u] == callable);
            expect(call_graph.functions_post_order[1u] == kernel);
        }

        auto argument_usage =
            lc::spirv::analyze_spirv_function_argument_usage(&module);
        expect(lc::spirv::spirv_function_argument_usage_of(
                   argument_usage, kernel, buffer) ==
               Usage::READ_WRITE);

        Kernel1D ast_kernel = [](BufferUInt, Bool) noexcept {};
        kernel->set_block_size(
            ast_kernel.function()->function().block_size());
        auto compiled = compile_exact_xir(
            ast_kernel.function()->function(), &module);
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpFunction) == 2u)
            << "the dead structural call must participate in callable discovery";
        expect(eq(compiled.argument_usages.size(), 2u));
        if (compiled.argument_usages.size() == 2u) {
            expect(compiled.argument_usages.front().second ==
                   Usage::READ_WRITE)
                << "the dead structural atomic must participate in resource-usage planning";
        }
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpFunctionCall) == 0u)
            << "glslang must canonicalize an ordinary-unreachable merge to OpUnreachable";
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpAtomicIAdd) == 0u)
            << "glslang must remove dead structural payload instructions during CFG post-processing";
    };

    "spirv_unconditional_loop_prepare_preserves_phi_and_validates"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *buffer = kernel->create_resource_argument(
            Type::buffer(Type::of<uint32_t>()));
        auto *initial =
            kernel->create_value_argument(Type::of<uint32_t>());
        auto *carried =
            kernel->create_value_argument(Type::of<uint32_t>());
        auto *zero =
            module.create_constant_zero(Type::of<uint32_t>());
        auto *entry = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *loop = builder.loop();
        auto *prepare = loop->create_prepare_block();
        auto *body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();

        builder.set_insertion_point(prepare);
        auto *phi = builder.phi(
            Type::of<uint32_t>(),
            {{initial, entry}, {carried, update}});
        auto *prepare_branch = builder.br(body);
        builder.set_insertion_point(body);
        builder.call(ResourceWriteOp::BUFFER_WRITE,
                     {buffer, zero, phi});
        builder.br(update);
        builder.set_insertion_point(update);
        builder.br(prepare);
        builder.set_insertion_point(merge);
        builder.return_void();

        expect(xir_verify_module(
                   &module, {.require_unique_merge_blocks = true})
                   .succeeded());
        auto prepare_plan = lc::spirv::plan_spirv_loop_prepare(loop);
        expect(prepare_plan.succeeded());
        expect(prepare_plan.kind ==
               lc::spirv::SpirvLoopPrepareKind::UNCONDITIONAL);
        auto plan = lc::spirv::ControlFlowPlan::create(kernel);
        auto &&region = plan.loop_region(loop);
        expect(region.prepare_kind ==
               lc::spirv::SpirvLoopPrepareKind::UNCONDITIONAL);
        expect(plan.edge_target(prepare_branch) == region.body_target);
        expect(region.physical_header_predecessor_count == 2u);
        expect(region.physical_boundary.succeeded());
        expect(plan.phi_plan(phi).incomings.size() == 2u);

        Kernel1D ast_kernel = [](BufferUInt, UInt, UInt) noexcept {};
        kernel->set_block_size(
            ast_kernel.function()->function().block_size());
        auto compiled = compile_exact_xir(
            ast_kernel.function()->function(), &module);
        auto words = luisa::span{compiled.spv_bin};
        expect(validates(words));
        expect(count_opcode(words, spv::Op::OpLoopMerge) == 1u);
        expect(opcode_after_first(words, spv::Op::OpLoopMerge) ==
               spv::Op::OpBranch)
            << "an unconditional Loop.prepare must emit OpLoopMerge "
               "immediately followed by OpBranch";
        expect(count_opcode(words, spv::Op::OpPhi) >= 1u)
            << "the loop-carried value must remain native SSA";
    };

    "spirv_unconditional_loop_prepare_wrong_target_is_nonfatal"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        XIRBuilder builder;
        builder.set_insertion_point(kernel->create_body_block());
        auto *loop = builder.loop();
        auto *prepare = loop->create_prepare_block();
        auto *body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *merge = loop->create_merge_block();
        builder.set_insertion_point(prepare);
        builder.br(merge);
        builder.set_insertion_point(body);
        builder.br(update);
        builder.set_insertion_point(update);
        builder.br(prepare);
        builder.set_insertion_point(merge);
        builder.return_void();

        auto validation = lc::spirv::ControlFlowPlan::
            validate_function_physical_loop_boundaries(kernel);
        expect(!validation.planning_succeeded());
        expect(validation.planning_diagnostic.find(
                   "Branch(Loop.body)") != luisa::string::npos);
        expect(validation.loops.empty());
    };

    "spirv_cyclic_switch_preserves_native_phi_and_validates"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *selector =
            kernel->create_value_argument(Type::of<uint32_t>());
        auto *initial =
            kernel->create_value_argument(Type::of<uint32_t>());
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *entry = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *outer = builder.if_(condition);
        auto *outer_true = outer->create_true_block();
        auto *outer_false = outer->create_false_block();
        auto *header = outer->create_merge_block();
        builder.set_insertion_point(outer_true);
        builder.br(header);
        builder.set_insertion_point(outer_false);
        builder.br(header);

        builder.set_insertion_point(header);
        auto *phi = builder.phi(
            Type::of<uint32_t>(),
            {{initial, outer_true}, {initial, outer_false}});
        auto *switch_inst = builder.switch_(selector);
        switch_inst->add_case(0u, header);
        auto *backedge_case =
            switch_inst->create_case_block(1u);
        auto *default_block =
            switch_inst->create_default_block();
        auto *merge = switch_inst->create_merge_block();
        builder.set_insertion_point(backedge_case);
        builder.br(header);
        phi->add_incoming(initial, header);
        phi->add_incoming(initial, backedge_case);
        builder.set_insertion_point(default_block);
        builder.br(merge);
        builder.set_insertion_point(merge);
        builder.return_void();

        expect(xir_verify_module(
                   &module, {.require_unique_merge_blocks = true})
                   .succeeded());
        auto plan = lc::spirv::ControlFlowPlan::create(kernel);
        auto &&region = plan.switch_region(switch_inst);
        expect(region.loop_wrapped);
        expect(region.has_header_case_target);
        expect(region.physical_header_predecessor_count == 2u);
        expect(region.physical_boundary.succeeded());
        expect(region.merge_target != region.loop_merge_target);
        auto &&outer_region = plan.if_region(outer);
        expect_single_trampoline_to(
            plan, outer_region.merge_target,
            lc::spirv::ControlFlowPlan::Target::xir(header));
        auto *backedge_incoming =
            find_incoming(plan.phi_plan(phi), backedge_case);
        auto *self_incoming = find_incoming(plan.phi_plan(phi), header);
        expect(backedge_incoming != nullptr);
        expect(self_incoming != nullptr);
        if (backedge_incoming != nullptr) {
            expect(backedge_incoming
                       ->forwarding_synthetic_indices.size() == 1u);
            expect(backedge_incoming
                       ->forwarding_synthetic_indices.front() ==
                   region.continue_synthetic_index);
        }
        if (self_incoming != nullptr) {
            expect(self_incoming
                       ->forwarding_synthetic_indices.size() == 3u);
            expect(self_incoming
                       ->forwarding_synthetic_indices[0u] ==
                   region.dispatch_synthetic_index);
            expect(self_incoming
                       ->forwarding_synthetic_indices[1u] ==
                   region.header_case_synthetic_index);
            expect(self_incoming
                       ->forwarding_synthetic_indices[2u] ==
                   region.continue_synthetic_index);
        }

        Kernel1D ast_kernel = [](UInt, UInt, Bool) noexcept {};
        kernel->set_block_size(
            ast_kernel.function()->function().block_size());
        auto compiled = compile_exact_xir(
            ast_kernel.function()->function(), &module);
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpSwitch) == 1u);
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpLoopMerge) >= 1u);
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpPhi) >= 4u)
            << "the header and synthetic forwarding Phis must remain native";
    };

    "spirv_switch_fallthrough_operands_are_reordered_and_validate"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *selector =
            kernel->create_value_argument(Type::of<uint32_t>());
        auto *header = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(header);
        auto *switch_inst = builder.switch_(selector);
        auto *case_a = switch_inst->create_case_block(0u);
        auto *case_b = switch_inst->create_case_block(1u);
        auto *case_c = switch_inst->create_case_block(2u);
        auto *default_block =
            switch_inst->create_default_block();
        auto *merge = switch_inst->create_merge_block();
        auto *one =
            module.create_constant_one(Type::of<uint32_t>());
        auto two_value = uint32_t{2u};
        auto *two = module.create_constant(
            Type::of<uint32_t>(), &two_value);

        builder.set_insertion_point(case_a);
        builder.br(case_c);
        builder.set_insertion_point(case_b);
        builder.br(merge);
        builder.set_insertion_point(case_c);
        builder.phi(Type::of<uint32_t>(),
                    {{one, header}, {two, case_a}});
        builder.br(merge);
        builder.set_insertion_point(default_block);
        builder.br(merge);
        builder.set_insertion_point(merge);
        builder.return_void();

        expect(xir_verify_module(
                   &module, {.require_unique_merge_blocks = true})
                   .succeeded());
        auto plan = lc::spirv::ControlFlowPlan::create(kernel);
        auto &&region = plan.switch_region(switch_inst);
        expect(!region.loop_wrapped);
        expect(region.case_operand_order ==
               luisa::vector<size_t>{0u, 2u, 1u});

        Kernel1D ast_kernel = [](UInt) noexcept {};
        kernel->set_block_size(
            ast_kernel.function()->function().block_size());
        auto compiled = compile_exact_xir(
            ast_kernel.function()->function(), &module);
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpSwitch) == 1u);
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpLoopMerge) == 0u);
        expect(first_u32_switch_literals(
                   luisa::span{compiled.spv_bin}) ==
               luisa::vector<uint32_t>{0u, 2u, 1u});
    };

    "spirv_switch_direct_loop_continue_uses_case_exit_trampoline"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *selector =
            kernel->create_value_argument(Type::of<uint32_t>());
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *initial =
            kernel->create_value_argument(Type::of<uint32_t>());
        auto *entry = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *loop = builder.loop();
        auto *prepare = loop->create_prepare_block();
        auto *body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        builder.set_insertion_point(prepare);
        builder.cond_br(condition, body, loop_merge);

        builder.set_insertion_point(body);
        auto *switch_inst = builder.switch_(selector);
        switch_inst->add_case(0u, update);
        auto *default_block =
            switch_inst->create_default_block();
        auto *switch_merge =
            switch_inst->create_merge_block();
        builder.set_insertion_point(default_block);
        builder.br(switch_merge);
        builder.set_insertion_point(switch_merge);
        builder.br(update);
        builder.set_insertion_point(update);
        auto *phi = builder.phi(
            Type::of<uint32_t>(),
            {{initial, body}, {initial, switch_merge}});
        builder.br(prepare);
        builder.set_insertion_point(loop_merge);
        builder.return_void();

        expect(xir_verify_module(
                   &module, {.require_unique_merge_blocks = true})
                   .succeeded());
        auto plan = lc::spirv::ControlFlowPlan::create(kernel);
        auto &&region = plan.switch_region(switch_inst);
        expect(!region.loop_wrapped);
        expect(region.direct_exit_targets.contains(update));
        expect_single_trampoline_to(
            plan, region.case_targets.front(),
            lc::spirv::ControlFlowPlan::Target::xir(update));
        auto *body_incoming = find_incoming(plan.phi_plan(phi), body);
        expect(body_incoming != nullptr);
        if (body_incoming != nullptr) {
            expect(body_incoming
                       ->forwarding_synthetic_indices.size() == 1u);
            expect(body_incoming
                       ->forwarding_synthetic_indices.front() ==
                   region.case_targets.front().synthetic_index);
        }

        Kernel1D ast_kernel = [](UInt, Bool, UInt) noexcept {};
        kernel->set_block_size(
            ast_kernel.function()->function().block_size());
        auto compiled = compile_exact_xir(
            ast_kernel.function()->function(), &module);
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpSwitch) == 1u);
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpLoopMerge) == 1u);
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpPhi) >= 2u)
            << "the case-exit forwarding Phi and update Phi must stay native";
    };

    "spirv_switch_direct_simple_loop_body_preserves_phi_boundary"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *selector =
            kernel->create_value_argument(Type::of<uint32_t>());
        constexpr auto initial_value = uint32_t{0x13579bdfu};
        constexpr auto carried_value = uint32_t{0x2468ace0u};
        auto *initial = module.create_constant(
            Type::of<uint32_t>(), &initial_value);
        auto *carried = module.create_constant(
            Type::of<uint32_t>(), &carried_value);
        auto *entry = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        auto *loop = builder.simple_loop();
        auto *body = loop->create_body_block();
        auto *loop_merge = loop->create_merge_block();

        builder.set_insertion_point(body);
        auto *phi = builder.phi(
            Type::of<uint32_t>(), {{initial, entry}});
        auto *switch_inst = builder.switch_(selector);
        switch_inst->add_case(0u, body);
        auto *default_block =
            switch_inst->create_default_block();
        auto *switch_merge =
            switch_inst->create_merge_block();
        phi->add_incoming(carried, body);
        builder.set_insertion_point(default_block);
        builder.br(switch_merge);
        builder.set_insertion_point(switch_merge);
        builder.br(loop_merge);
        builder.set_insertion_point(loop_merge);
        builder.return_void();

        expect(xir_verify_module(
                   &module, {.require_unique_merge_blocks = true})
                   .succeeded());
        auto plan = lc::spirv::ControlFlowPlan::create(kernel);
        auto &&region = plan.switch_region(switch_inst);
        auto &&loop_region = plan.simple_loop_region(loop);
        auto continue_target =
            lc::spirv::ControlFlowPlan::Target::synthetic(
                loop_region.continue_synthetic_index);
        expect(loop_region.physical_header_predecessor_count == 2u);
        expect(loop_region.physical_boundary.succeeded());
        expect(loop_region.physical_boundary.entry_edge_count == 1u);
        expect(loop_region.physical_boundary.backedge_edge_count == 1u);
        expect(loop_region.physical_boundary
                   .backedge_dominated_by_continue_target);
        expect(!loop_region.physical_boundary
                    .backedge_dominated_by_merge_target);
        expect(!region.loop_wrapped);
        expect(region.direct_exit_targets.contains(body));
        expect_single_trampoline_to(
            plan, region.case_targets.front(), continue_target);
        auto &&phi_plan = plan.phi_plan(phi);
        auto *entry_incoming = find_incoming(phi_plan, entry);
        expect(entry_incoming != nullptr);
        if (entry_incoming != nullptr) {
            expect(entry_incoming->value == initial);
            expect(entry_incoming->forwarding_synthetic_indices.empty());
        }
        auto *self_incoming = find_incoming(phi_plan, body);
        expect(self_incoming != nullptr);
        if (self_incoming != nullptr) {
            expect(self_incoming->value == carried);
            expect(self_incoming
                       ->forwarding_synthetic_indices.size() == 2u);
            expect(self_incoming
                       ->forwarding_synthetic_indices[0u] ==
                   region.case_targets.front().synthetic_index);
            expect(self_incoming
                       ->forwarding_synthetic_indices[1u] ==
                   loop_region.continue_synthetic_index);
        }

        Kernel1D ast_kernel = [](UInt) noexcept {};
        kernel->set_block_size(
            ast_kernel.function()->function().block_size());
        auto compiled = compile_exact_xir(
            ast_kernel.function()->function(), &module);
        expect(validates(luisa::span{compiled.spv_bin}))
            << "accepted SimpleLoop boundary must validate as exact opt0 "
               "Vulkan SPIR-V";
        auto routing = inspect_simple_loop_phi_routing(
            luisa::span{compiled.spv_bin}, initial_value, carried_value);
        expect(routing.loop_merge_count == 1u);
        expect(routing.header_phi_count == 1u);
        expect(routing.header_incoming_count == 2u);
        expect(routing.entry_incoming_is_initial)
            << "the SimpleLoop owner edge must carry the initial value";
        expect(routing.continue_incoming_found)
            << "the SimpleLoop header Phi must name the declared continue block";
        expect(routing.continue_incoming_resolves_to_carried)
            << "the continue-edge forwarding Phi chain must carry the distinct "
               "loop value";
        expect(routing.continue_forwarding_phi_depth == 2u)
            << "the carried value must cross the case-exit and SimpleLoop "
               "continue Phis at opt0";
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpSwitch) == 1u);
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpLoopMerge) == 1u);
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpPhi) >= 3u)
            << "the loop-header and case-exit forwarding Phis must stay native";
    };

    "spirv_switch_direct_outer_cyclic_switch_continue_uses_case_exit_trampoline"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *outer_selector =
            kernel->create_value_argument(Type::of<uint32_t>());
        auto *inner_selector =
            kernel->create_value_argument(Type::of<uint32_t>());
        auto *initial =
            kernel->create_value_argument(Type::of<uint32_t>());
        auto *entry = kernel->create_body_block();
        auto *outer_header = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(entry);
        builder.br(outer_header);

        builder.set_insertion_point(outer_header);
        auto *phi = builder.phi(
            Type::of<uint32_t>(), {{initial, entry}});
        auto *outer_switch = builder.switch_(outer_selector);
        auto *inner_header =
            outer_switch->create_case_block(0u);
        auto *outer_default =
            outer_switch->create_default_block();
        auto *outer_merge =
            outer_switch->create_merge_block();

        builder.set_insertion_point(inner_header);
        auto *inner_switch = builder.switch_(inner_selector);
        inner_switch->add_case(0u, outer_header);
        auto *inner_default =
            inner_switch->create_default_block();
        auto *inner_merge =
            inner_switch->create_merge_block();
        phi->add_incoming(initial, inner_header);
        builder.set_insertion_point(inner_default);
        builder.br(inner_merge);
        builder.set_insertion_point(inner_merge);
        builder.br(outer_merge);
        builder.set_insertion_point(outer_default);
        builder.br(outer_merge);
        builder.set_insertion_point(outer_merge);
        builder.return_void();

        expect(xir_verify_module(
                   &module, {.require_unique_merge_blocks = true})
                   .succeeded());
        auto plan = lc::spirv::ControlFlowPlan::create(kernel);
        auto &&outer_region = plan.switch_region(outer_switch);
        auto &&inner_region = plan.switch_region(inner_switch);
        expect(outer_region.loop_wrapped);
        expect(!inner_region.loop_wrapped);
        expect(inner_region.direct_exit_targets.contains(outer_header));
        auto outer_continue =
            lc::spirv::ControlFlowPlan::Target::synthetic(
                outer_region.continue_synthetic_index);
        expect_single_trampoline_to(
            plan, inner_region.case_targets.front(), outer_continue);
        auto *backedge_incoming =
            find_incoming(plan.phi_plan(phi), inner_header);
        expect(backedge_incoming != nullptr);
        if (backedge_incoming != nullptr) {
            expect(backedge_incoming
                       ->forwarding_synthetic_indices.size() == 2u);
            expect(backedge_incoming
                       ->forwarding_synthetic_indices[0u] ==
                   inner_region.case_targets.front().synthetic_index);
            expect(backedge_incoming
                       ->forwarding_synthetic_indices[1u] ==
                   outer_region.continue_synthetic_index);
        }

        Kernel1D ast_kernel = [](UInt, UInt, UInt) noexcept {};
        kernel->set_block_size(
            ast_kernel.function()->function().block_size());
        auto compiled = compile_exact_xir(
            ast_kernel.function()->function(), &module);
        expect(validates(luisa::span{compiled.spv_bin}));
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpSwitch) == 2u);
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpLoopMerge) == 1u);
        expect(count_opcode(luisa::span{compiled.spv_bin},
                            spv::Op::OpPhi) >= 3u)
            << "the outer header and nested case-exit forwarding Phis must stay native";
    };
}
