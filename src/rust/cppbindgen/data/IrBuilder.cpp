NodeRef IrBuilder::call(const Func &f, luisa::span<const NodeRef> args, const CArc<Type> &type) noexcept {
    auto f_inner = f._inner;
    auto slice = raw::CSlice{reinterpret_cast<const raw::NodeRef *>(args.data()), args.size()};
    auto node = luisa_compute_ir_build_call(&_inner, f_inner, slice, luisa::bit_cast<CArc<raw::Type>>(type.clone()));
    return NodeRef::from_raw(node);
}
NodeRef IrBuilder::phi(luisa::span<const PhiIncoming> incoming, const CArc<Type> &type) noexcept {
    auto incoming_ = raw::CSlice{reinterpret_cast<const raw::PhiIncoming *>(incoming.data()), incoming.size()};
    auto node = luisa_compute_ir_build_phi(&_inner, incoming_, luisa::bit_cast<CArc<raw::Type>>(type.clone()));
    return NodeRef::from_raw(node);
}
NodeRef IrBuilder::local(NodeRef init) noexcept {
    auto node = luisa_compute_ir_build_local(&_inner, init._inner);
    return NodeRef::from_raw(node);
}
NodeRef IrBuilder::local(const CppOwnedCArc<Type> &type) noexcept {
    auto node = luisa_compute_ir_build_local_zero_init(&_inner, luisa::bit_cast<CArc<raw::Type>>(type.clone()));
    return NodeRef::from_raw(node);
}
NodeRef IrBuilder::if_(const NodeRef &cond, const Pooled<BasicBlock> &true_branch, const Pooled<BasicBlock> &false_branch) noexcept {
    auto node = luisa_compute_ir_build_if(&_inner,
                                          cond._inner,
                                          reinterpret_cast<const Pooled<raw::BasicBlock> &>(true_branch),
                                          reinterpret_cast<const Pooled<raw::BasicBlock> &>(false_branch));
    return NodeRef::from_raw(node);
}
NodeRef IrBuilder::switch_(const NodeRef &value, luisa::span<const SwitchCase> cases, const Pooled<BasicBlock> &default_) noexcept {
    auto cases_ = raw::CSlice{reinterpret_cast<const raw::SwitchCase *>(cases.data()), cases.size()};
    auto node = luisa_compute_ir_build_switch(&_inner, value._inner, cases_, reinterpret_cast<const Pooled<raw::BasicBlock> &>(default_));
    return NodeRef::from_raw(node);
}
// NodeRef IrBuilder::loop(const Pooled<BasicBlock> &body, const NodeRef &cond) noexcept;
NodeRef IrBuilder::generic_loop(const Pooled<BasicBlock> &prepare, const NodeRef &cond, const Pooled<BasicBlock> &body, const Pooled<BasicBlock> &update) noexcept {
    auto node = luisa_compute_ir_build_generic_loop(&_inner,
                                                    reinterpret_cast<const Pooled<raw::BasicBlock> &>(prepare),
                                                    cond._inner,
                                                    reinterpret_cast<const Pooled<raw::BasicBlock> &>(body),
                                                    reinterpret_cast<const Pooled<raw::BasicBlock> &>(update));
    return NodeRef::from_raw(node);
}
NodeRef IrBuilder::loop(const Pooled<BasicBlock> &body, const NodeRef &cond) noexcept {
    auto node = luisa_compute_ir_build_loop(&_inner,
                                            reinterpret_cast<const Pooled<raw::BasicBlock> &>(body),
                                            cond._inner);
    return NodeRef::from_raw(node);
}
Pooled<BasicBlock> IrBuilder::finish(IrBuilder &&builder) noexcept {
    auto block = luisa_compute_ir_build_finish(builder._inner);
    return luisa::bit_cast<Pooled<BasicBlock>>(block);
}

void IrBuilder::set_insert_point(const NodeRef &node) noexcept {
    raw::luisa_compute_ir_builder_set_insert_point(&_inner, node._inner);
}