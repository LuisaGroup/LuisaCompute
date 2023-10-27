#include <luisa/ir_v2/analysis/usedef.h>
namespace luisa::compute::ir_v2 {
UseDefAnalysis::UseDefAnalysis(Module &module) : module{module} {
}
void UseDefAnalysis::visit_block(const BasicBlock *block) noexcept {
    block->for_each([&](Node *n) {
        visit_node(n);
    });
}
void UseDefAnalysis::visit_node(const Node *n) noexcept {
    auto inst = &n->inst;
    auto tag = inst->tag();
    switch (tag) {
        case Instruction::Tag::ACCEL: [[fallthrough]];
        case Instruction::Tag::BUFFER: [[fallthrough]];
        case Instruction::Tag::TEXTURE2D: [[fallthrough]];
        case Instruction::Tag::TEXTURE3D: [[fallthrough]];
        case Instruction::Tag::BINDLESS_ARRAY: [[fallthrough]];
        case Instruction::Tag::UNIFORM: [[fallthrough]];
        case Instruction::Tag::ARGUMENT: [[fallthrough]];
        case Instruction::Tag::BREAK: [[fallthrough]];
        case Instruction::Tag::CONTINUE: [[fallthrough]];
        case Instruction::Tag::RETURN: [[fallthrough]];
        case Instruction::Tag::SHARED:
            this->add_to_root(n);
            break;
        case Instruction::Tag::CALL: {
            auto call = inst->as<CallInst>();
            auto f = &call->func;
            if (f->has_side_effects()) {
                this->add_to_root(n);
            }
            auto &args = call->args;
            for (auto arg : args) {
                _used_by[arg].insert(n);
            }
        } break;
        case Instruction::Tag::UPDATE: {
            // this is very special
            this->add_to_root(n);
        } break;
        case Instruction::Tag::IF: {
            auto if_ = inst->as<IfInst>();
            this->add_to_root(if_->cond);
            this->add_to_root(n);
            this->visit_block(if_->true_branch);
            this->visit_block(if_->false_branch);
        } break;
        case Instruction::Tag::GENERIC_LOOP: {
            auto loop = inst->as<GenericLoopInst>();
            this->add_to_root(n);
            this->add_to_root(loop->cond);
            this->visit_block(loop->prepare);
            this->visit_block(loop->body);
            this->visit_block(loop->update);
        }
        case Instruction::Tag::CONSTANT: break;
        case InstructionTag::PHI: {
            auto phi = inst->as<PhiInst>();
            this->add_to_root(n);
            for (auto &i : phi->incomings) {
                _used_by[i.value].insert(n);
            }
        } break;

        default:
            LUISA_ERROR_WITH_LOCATION("unhandled instruction: {}", (int)tag);
            break;
    }
}
void UseDefAnalysis::run() noexcept {
    this->_used_by.clear();
    for (auto arg : module.args) {
        this->add_to_root(arg);
    }
    this->visit_block(module.entry);
}
}// namespace luisa::compute::ir_v2