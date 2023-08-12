#include <luisa/ir/transforms/ref2ret.h>

namespace luisa::compute::ir_v2 {

bool Ref2Ret::_needs_transform(CallableModule *m) noexcept {
    // if any argument is a reference, we need to transform the function
    // TODO: identify the usage of the argument; const reference does not need to be transformed
    return std::any_of(
        m->args().cbegin(), m->args().cend(),
        [](const NodeRef &node) noexcept {
            auto instr = node->instruction().get();
            if (auto arg = instr->as<Instruction::Argument>()) {
                return !arg->by_value();
            }
            return false;
        });
}

void Ref2Ret::_transform(const CArc<KernelModule> &m) noexcept {
    _transform(m->module().entry().get());
}

void Ref2Ret::_transform(const CArc<CallableModule> &m) noexcept {
    //    // already transformed
    //    if (_processed.contains(m.get())) { return; }
    //    // transform
    //    if (_needs_transform(m.get())) {
    //        auto copy = raw::luisa_compute_ir_copy_callable_module(m->raw());
    //        auto ctx = _prepare_context(reinterpret_cast<CallableModule *>(copy->ptr));
    //        auto old_ctx = std::exchange(_current, &ctx);
    //        _transform(copy->ptr->module.entry.get());
    //        _processed.emplace(m, Metadata{copy._0.clone(), std::move(ctx.ref_arg_indices)});
    //        _current = old_ctx;
    //    } else {
    //        auto old_ctx = std::exchange(_current, nullptr);
    //        _transform(m->module().entry().get());
    //        _processed.emplace(m, Metadata{.module = m.clone()});// empty metadata means no transformation
    //        _current = old_ctx;
    //    }
}

void Ref2Ret::_transform(BasicBlock *bb) noexcept {
    if (bb == nullptr) { return; }
    for (auto node : *bb) {
        using Tag = Instruction::Tag;
        switch (auto instr = node->instruction(); instr->tag()) {
            case Tag::Call: {
                auto call = instr->as<Instruction::Call>();
                //                _transform_call(call);
                break;
            }
            case Tag::Return: {
                auto ret = instr->as<Instruction::Return>();
                //                _transform_return(ret);
                break;
            }
            case Tag::Loop: {
                auto loop = instr->as<Instruction::Loop>();
                _transform(loop->body().get());
                break;
            }
            case Tag::GenericLoop: {
                auto loop = instr->as<Instruction::GenericLoop>();
                _transform(loop->prepare().get());
                _transform(loop->body().get());
                _transform(loop->update().get());
                break;
            }
            case Tag::If: {
                auto if_ = instr->as<Instruction::If>();
                _transform(if_->true_branch().get());
                _transform(if_->false_branch().get());
                break;
            }
            case Tag::Switch: {
                auto switch_ = instr->as<Instruction::Switch>();
                for (auto &c : switch_->cases()) { _transform(c.block().get()); }
                _transform(switch_->default_().get());
                break;
            }
            case Tag::AdScope: {
                auto ad = instr->as<Instruction::AdScope>();
                _transform(ad->body().get());
                break;
            }
            case Tag::RayQuery: {
                auto rq = instr->as<Instruction::RayQuery>();
                _transform(rq->on_triangle_hit().get());
                _transform(rq->on_procedural_hit().get());
                break;
            }
            default: break;
        }
    }
}

Ref2Ret::Context Ref2Ret::_prepare_context(CallableModule *m) noexcept {
    Context ctx;// collect the arguments passed by reference and their types
    luisa::vector<CArc<Type>> ret_types;
    for (auto i = 0u; i < m->args().size(); i++) {
        auto node = m->args()[i];
        if (auto arg = node->instruction()->as<Instruction::Argument>();
            arg != nullptr && !arg->by_value()) {
            // change the node to mark that the argument is passed by value
            node->raw()->instruction->argument.by_value = true;
            // add the argument to the context
            ctx.ref_args.emplace_back(node);
            ctx.ref_arg_indices.emplace_back(i);
            ret_types.emplace_back(node->type_());
        }
    }
    // add return type if any
    if (auto &&t = m->ret_type();
        !t.is_null() && !t->isa<Type::Void>()) {
        ret_types.emplace_back(t);
    }
    LUISA_ASSERT(!ret_types.empty(), "No return type.");
    // create the return type
    auto ret_type = [&]() noexcept -> CArc<Type> {
        auto ret_fields = raw::create_boxed_slice<CArc<raw::Type>>(ret_types.size());
        auto ret_size = static_cast<size_t>(0u);
        auto ret_alignment = static_cast<size_t>(0u);
        // compute type alignment and size
        for (auto i = 0u; i < ret_types.size(); i++) {
            auto t = reinterpret_cast<const CArc<raw::Type> *>(&ret_types[i]);
            auto size = raw::luisa_compute_ir_type_size(t);
            auto alignment = raw::luisa_compute_ir_type_alignment(t);
            ret_alignment = std::max(ret_alignment, alignment);
            ret_size = luisa::align(ret_size, alignment);
            ret_size += size;
            ret_fields.ptr[i] = *t;
        }
        ret_size = luisa::align(ret_size, ret_alignment);
        raw::Type ret_type{.tag = raw::Type::Tag::Struct};
        ret_type.struct_ = {{
            .fields = ret_fields,
            .alignment = ret_alignment,
            .size = ret_size,
        }};
        auto t = raw::luisa_compute_ir_register_type(&ret_type);
        raw::destroy_boxed_slice(ret_fields);
        return reinterpret_cast<raw::CArcSharedBlock<Type> *>(t);
    }();
    // change the return type of the function
    m->raw()->ret_type.release();
//    m->raw()->ret_type = ret_type.clone();
    // create a local variable to store the return value
    IrBuilder::with(m->pools().clone(), [&](IrBuilder &b) noexcept {
        b.set_insert_point(m->module().entry()->first());
        //        b.local(ret_type.clone());
    });
    auto bb = m->module().entry();
    // zero init
//    raw::Instruction ret_init_instr{.tag = raw::Instruction::Tag::Call};
//    ret_init_instr.call._0.tag = raw::Func::Tag::ZeroInitializer;
//    ret_init_instr.call._1 = raw::create_boxed_slice<raw::NodeRef>(0u);
//    raw::Node ret_init_node{.type_ = ret_type.clone(),
//                            .instruction = raw::luisa_compute_ir_new_instruction(ret_init_instr)};
//    auto ret_init = raw::luisa_compute_ir_new_node(m->pools().clone(), ret_init_node);
//    raw::luisa_compute_ir_node_insert_before_self(bb->first().raw(), ret_init);// first
//    // local variable
//    raw::Instruction ret_var_instr{.tag = raw::Instruction::Tag::Local};
//    ret_var_instr.local.init = ret_init;
//    raw::Node ret_var_node{.type_ = ret_type.clone(),
//                           .instruction = raw::luisa_compute_ir_new_instruction(ret_var_instr)};
//    auto ret_var = raw::luisa_compute_ir_new_node(m->pools().clone(), ret_var_node);
//    raw::luisa_compute_ir_node_insert_after_self(ret_init, ret_var);// second
//    // put the return value into the context
//    ctx.ret = NodeRef::from_raw(ret_var);
    return ctx;
}

}// namespace luisa::compute::ir_v2
