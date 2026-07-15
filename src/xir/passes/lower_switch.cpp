#include <luisa/core/logging.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/if.h>
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
            auto v = static_cast<int8_t>(value);
            return module->create_constant(type, &v);
        }
        case Type::Tag::UINT8: {
            auto v = static_cast<uint8_t>(value);
            return module->create_constant(type, &v);
        }
        case Type::Tag::INT16: {
            auto v = static_cast<int16_t>(value);
            return module->create_constant(type, &v);
        }
        case Type::Tag::UINT16: {
            auto v = static_cast<uint16_t>(value);
            return module->create_constant(type, &v);
        }
        case Type::Tag::INT32: {
            auto v = static_cast<int32_t>(value);
            return module->create_constant(type, &v);
        }
        case Type::Tag::UINT32: {
            auto v = static_cast<uint32_t>(value);
            return module->create_constant(type, &v);
        }
        case Type::Tag::INT64: {
            auto v = static_cast<int64_t>(value);
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

[[nodiscard]] LowerSwitchInfo lower_switch_on_definition(FunctionDefinition *def) noexcept {
    LowerSwitchInfo info{};
    if (def == nullptr) { return info; }

    luisa::vector<SwitchInst *> switches;
    def->traverse_basic_blocks([&](BasicBlock *bb) noexcept {
        if (!bb->is_terminated()) { return; }
        if (bb->is_terminated() && bb->terminator()->isa<SwitchInst>()) {
            switches.push_back(static_cast<SwitchInst *>(bb->terminator()));
        }
    });

    for (auto *sw : switches) {
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
        BasicBlock *next_else = default_bb;
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

            next_else = if_header;
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

LowerSwitchInfo lower_switch_pass_run_on_module(Module *module, PassReport *report) noexcept {
    LowerSwitchInfo total{};
    if (module == nullptr) { return total; }
    for (auto *f : module->function_list()) {
        auto info = lower_switch_pass_run_on_function(f);
        total.lowered_switch_count += info.lowered_switch_count;
    }
    if (report != nullptr) {
        report->set("lowered_switch", total.lowered_switch_count);
    }
    return total;
}

}// namespace luisa::compute::xir
