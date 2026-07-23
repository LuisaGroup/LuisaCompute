#include <luisa/core/logging.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>
#include <algorithm>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/if.h>
#include <luisa/xir/instructions/phi.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/module.h>
#include <luisa/xir/op.h>
#include <luisa/xir/passes/lower_switch.h>
#include <luisa/xir/passes/pass_pipeline.h>

namespace luisa::compute::xir {

namespace {

[[nodiscard]] Constant *create_case_constant(Module *module, const Type *type,
                                             SwitchInst::case_value_type value) noexcept {
    switch (type->tag()) {
        case Type::Tag::BOOL: {
            auto v = static_cast<bool>(value);
            return module->create_constant(type, &v);
        }
        case Type::Tag::INT8: {
            auto v = luisa::bit_cast<int8_t>(static_cast<uint8_t>(value));
            return module->create_constant(type, &v);
        }
        case Type::Tag::UINT8: {
            auto v = static_cast<uint8_t>(value);
            return module->create_constant(type, &v);
        }
        case Type::Tag::INT16: {
            auto v = luisa::bit_cast<int16_t>(static_cast<uint16_t>(value));
            return module->create_constant(type, &v);
        }
        case Type::Tag::UINT16: {
            auto v = static_cast<uint16_t>(value);
            return module->create_constant(type, &v);
        }
        case Type::Tag::INT32: {
            auto v = luisa::bit_cast<int32_t>(static_cast<uint32_t>(value));
            return module->create_constant(type, &v);
        }
        case Type::Tag::UINT32: {
            auto v = static_cast<uint32_t>(value);
            return module->create_constant(type, &v);
        }
        case Type::Tag::INT64: {
            auto v = luisa::bit_cast<int64_t>(value);
            return module->create_constant(type, &v);
        }
        case Type::Tag::UINT64: {
            auto v = static_cast<uint64_t>(value);
            return module->create_constant(type, &v);
        }
        default: LUISA_ERROR_WITH_LOCATION(
            "Invalid switch selector type {}.", type->description());
    }
}

[[nodiscard]] luisa::vector<SwitchInst *> collect_switches(FunctionDefinition *def) noexcept {
    luisa::vector<SwitchInst *> switches;
    if (def == nullptr) { return switches; }
    // Inspect every owned block. A disconnected structured switch is still an
    // input to this representation-changing pass and may become reachable after
    // a later CFG rewrite.
    for (auto *block : def->basic_blocks()) {
        if (block != nullptr && block->is_terminated() &&
            block->terminator()->isa<SwitchInst>()) {
            switches.emplace_back(static_cast<SwitchInst *>(block->terminator()));
        }
    }
    return switches;
}

[[nodiscard]] bool is_supported_switch_selector_type(const Type *type) noexcept {
    if (type == nullptr) { return false; }
    switch (type->tag()) {
        case Type::Tag::BOOL:
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

[[nodiscard]] size_t count_rejected_switches(
    FunctionDefinition *def, luisa::span<SwitchInst *const> switches) noexcept {
    size_t count = 0u;
    for (auto *sw : switches) {
        auto rejected = sw == nullptr || sw->parent_block() == nullptr ||
                        sw->parent_block()->parent_function() != def ||
                        sw->value() == nullptr ||
                        !is_supported_switch_selector_type(sw->value()->type()) ||
                        sw->default_block() == nullptr ||
                        sw->default_block()->parent_function() != def;
        if (!rejected && sw->merge_block() != nullptr) {
            rejected = sw->merge_block()->parent_function() != def;
        }
        if (!rejected && sw->case_count() == 0u && sw->merge_block() != nullptr) {
            rejected = true;
        }
        for (size_t i = 0u; i < sw->case_count() && !rejected; ++i) {
            auto *case_block = sw->case_block(i);
            rejected = case_block == nullptr || case_block->parent_function() != def;
        }
        count += rejected ? 1u : 0u;
    }
    return count;
}

struct LowerSwitchPreflight {
    luisa::vector<SwitchInst *> switches;
    size_t rejected_switch_count{0u};
};

[[nodiscard]] LowerSwitchPreflight preflight_lower_switch(
    FunctionDefinition *def) noexcept {
    LowerSwitchPreflight preflight;
    if (def == nullptr) { return preflight; }
    preflight.switches = collect_switches(def);
    preflight.rejected_switch_count =
        count_rejected_switches(def, preflight.switches);
    return preflight;
}

[[nodiscard]] LowerSwitchInfo lower_switch_on_definition(FunctionDefinition *def) noexcept {
    LowerSwitchInfo info{};
    if (def == nullptr) { return info; }

    auto preflight = preflight_lower_switch(def);
    info.rejected_switch_count = preflight.rejected_switch_count;
    if (info.rejected_switch_count != 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "lower_switch: refusing to erase {} structured zero-case merge frame(s); "
            "the function was left unchanged.",
            info.rejected_switch_count);
        return info;
    }

    for (auto *sw : preflight.switches) {
        auto *parent_bb = sw->parent_block();
        if (parent_bb == nullptr) { continue; }

        auto *value = sw->value();
        auto *default_bb = sw->default_block();
        auto *merge_bb = sw->merge_block();
        auto *mod = def->parent_module();
        size_t n = sw->case_count();

        if (n == 0) {
            sw->remove_self();
            XIRBuilder b;
            b.set_insertion_point(parent_bb);
            b.br(default_bb);
            ++info.lowered_switch_count;
            continue;
        }

        // Build a cascaded if-else chain.
        // Each case creates: if(value == case_i) { case_block_i } else { next }
        // where "next" is the header of the next case (or default for the last).
        // All IfInsts share the original switch's merge block.
        struct SuccessorPredecessors {
            BasicBlock *successor;
            luisa::vector<BasicBlock *> predecessors;
        };
        luisa::vector<SuccessorPredecessors> successor_predecessors;
        auto record_predecessor = [&](BasicBlock *successor, BasicBlock *predecessor) noexcept {
            auto iter = std::find_if(
                successor_predecessors.begin(), successor_predecessors.end(),
                [successor](const SuccessorPredecessors &item) noexcept {
                    return item.successor == successor;
                });
            if (iter == successor_predecessors.end()) {
                successor_predecessors.emplace_back(SuccessorPredecessors{successor, {predecessor}});
            } else if (std::find(iter->predecessors.begin(), iter->predecessors.end(), predecessor) ==
                       iter->predecessors.end()) {
                iter->predecessors.emplace_back(predecessor);
            }
        };

        BasicBlock *next_else = default_bb;
        BasicBlock *default_predecessor = nullptr;
        for (size_t i = n; i-- > 0;) {
            auto case_val = sw->case_value(i);
            auto *case_bb = sw->case_block(i);

            auto *if_header = (i == 0) ? parent_bb : def->create_basic_block();
            XIRBuilder b;
            b.set_insertion_point(if_header);

            auto *case_const = create_case_constant(mod, value->type(), case_val);
            auto *eq = b.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {value, case_const});
            auto *if_inst = b.if_(eq);
            if_inst->set_true_target(case_bb);
            if_inst->set_false_target(next_else);
            if_inst->set_merge_block(merge_bb);

            record_predecessor(case_bb, if_header);
            if (i == n - 1u) { default_predecessor = if_header; }
            next_else = if_header;
        }
        record_predecessor(default_bb, default_predecessor);

        // Switch PHIs name the original switch header as their predecessor.
        // The cascade introduces a distinct header for every case after the
        // first (and for the default edge), so rewrite/duplicate that incoming
        // value for the exact predecessor set before deleting the switch.
        for (auto &&[successor, predecessors] : successor_predecessors) {
            if (successor == nullptr) { continue; }
            for (auto *inst : successor->instructions()) {
                if (!inst->isa<PhiInst>()) { continue; }
                auto *phi = static_cast<PhiInst *>(inst);
                Value *incoming_value = nullptr;
                auto found_incoming = false;
                for (size_t i = phi->incoming_count(); i-- > 0u;) {
                    auto incoming = phi->incoming(i);
                    if (incoming.block == parent_bb) {
                        if (!found_incoming) {
                            incoming_value = incoming.value;
                            found_incoming = true;
                        }
                        phi->remove_incoming(i);
                    }
                }
                if (found_incoming) {
                    for (auto *predecessor : predecessors) {
                        phi->add_incoming(incoming_value, predecessor);
                    }
                }
            }
        }

        sw->remove_self();
        ++info.lowered_switch_count;
    }

    return info;
}

}// namespace

LowerSwitchInfo lower_switch_pass_run_on_function(Function *function) noexcept {
    if (function == nullptr) { return {}; }
    auto *def = function->definition();
    if (def == nullptr) { return {}; }
    return lower_switch_on_definition(def);
}

LowerSwitchInfo lower_switch_pass_preflight_function(Function *function) noexcept {
    LowerSwitchInfo info;
    if (function == nullptr) { return info; }
    auto *def = function->definition();
    if (def == nullptr) { return info; }
    info.rejected_switch_count =
        preflight_lower_switch(def).rejected_switch_count;
    return info;
}

LowerSwitchInfo lower_switch_pass_run_on_module(Module *module, PassReport *report) noexcept {
    LowerSwitchInfo total{};
    if (module == nullptr) { return total; }
    // Keep the module entry point atomic as well: do not lower an earlier
    // function before discovering an unsupported structured switch later.
    for (auto *f : module->function_list()) {
        total.rejected_switch_count +=
            lower_switch_pass_preflight_function(f)
                .rejected_switch_count;
    }
    if (total.rejected_switch_count != 0u) {
        LUISA_WARNING_WITH_LOCATION(
            "lower_switch: refusing to erase {} structured zero-case merge frame(s); "
            "the module was left unchanged.",
            total.rejected_switch_count);
        if (report != nullptr) {
            report->set("lowered_switch", 0u);
            report->set("rejected_switch", total.rejected_switch_count);
        }
        return total;
    }
    for (auto *f : module->function_list()) {
        auto info = lower_switch_pass_run_on_function(f);
        total.lowered_switch_count += info.lowered_switch_count;
    }
    if (report != nullptr) {
        report->set("lowered_switch", total.lowered_switch_count);
        report->set("rejected_switch", total.rejected_switch_count);
    }
    return total;
}

}// namespace luisa::compute::xir
