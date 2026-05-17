#include <luisa/xir/passes/uniformity_analysis.h>
#include <luisa/xir/function.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/special_register.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/instruction.h>

namespace luisa::compute::xir {

void UniformityAnalysis::clear() noexcept {
    _uniform.clear();
    _function = nullptr;
}

void UniformityAnalysis::analyze(const Function *function) noexcept {
    clear();
    _function = function;
    if (function == nullptr) { return; }
    auto def = function->definition();
    if (def == nullptr) { return; }

    bool is_kernel = function->isa<KernelFunction>();
    for (auto a : function->arguments()) {
        if (is_kernel) { _uniform[a] = true; }
    }

    for (;;) {
        bool changed = false;
        def->traverse_instructions([&](const Instruction *inst) noexcept {
            if (auto it = _uniform.find(inst); it != _uniform.end() && it->second) { return; }
            bool can_be_uniform = false;
            switch (inst->derived_instruction_tag()) {
                case DerivedInstructionTag::CAST:
                case DerivedInstructionTag::ARITHMETIC:
                case DerivedInstructionTag::GEP: {
                    can_be_uniform = true;
                    for (size_t i = 0u, n = inst->operand_count(); i < n; ++i) {
                        if (!is_uniform(inst->operand(i))) {
                            can_be_uniform = false;
                            break;
                        }
                    }
                    break;
                }
                default: break;
            }
            if (can_be_uniform) {
                _uniform[inst] = true;
                changed = true;
            }
        });
        if (!changed) { break; }
    }
}

bool UniformityAnalysis::is_uniform(const Value *value) const noexcept {
    if (value == nullptr) { return false; }
    if (auto it = _uniform.find(value); it != _uniform.end()) { return it->second; }
    switch (value->derived_value_tag()) {
        case DerivedValueTag::CONSTANT: return true;
        case DerivedValueTag::ARGUMENT: {
            auto arg = static_cast<const Argument *>(value);
            return _function != nullptr && _function->isa<KernelFunction>() &&
                   arg->parent_function() == _function;
        }
        case DerivedValueTag::SPECIAL_REGISTER: {
            using T = DerivedSpecialRegisterTag;
            switch (static_cast<const SpecialRegister *>(value)->derived_special_register_tag()) {
                case T::BLOCK_ID:
                case T::DISPATCH_SIZE:
                case T::KERNEL_ID:
                case T::BLOCK_SIZE:
                    return true;
                default:
                    return false;
            }
        }
        default: return false;
    }
}

}// namespace luisa::compute::xir
