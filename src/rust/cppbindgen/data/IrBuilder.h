NodeRef call(const Func &f, luisa::span<const NodeRef> args, const CArc<Type> &type) noexcept;
NodeRef phi(luisa::span<const PhiIncoming> incoming, const CArc<Type> &type) noexcept;
NodeRef local(const CArc<Type> &type) noexcept;
NodeRef local(NodeRef init) noexcept;
NodeRef if_(const NodeRef &cond, const Pooled<BasicBlock> &true_branch, const Pooled<BasicBlock> &false_branch) noexcept;
NodeRef switch_(const NodeRef &value, luisa::span<const SwitchCase> cases, const Pooled<BasicBlock> &default_) noexcept;
NodeRef loop(const Pooled<BasicBlock> &body, const NodeRef &cond) noexcept;
NodeRef generic_loop(const Pooled<BasicBlock> &prepare, const NodeRef &cond, const Pooled<BasicBlock> &body, const Pooled<BasicBlock> &update) noexcept;
static Pooled<BasicBlock> finish(IrBuilder &&builder) noexcept;