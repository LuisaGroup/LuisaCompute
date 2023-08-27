    IrBuilder(raw::IrBuilder inner) noexcept : _inner{inner} {}

    NodeRef call(const Func &f, luisa::span<const NodeRef> args, const CArc<Type> &type) noexcept;
    NodeRef phi(luisa::span<const PhiIncoming> incoming, const CArc<Type> &type) noexcept;
    NodeRef local(const CppOwnedCArc<Type> &type) noexcept;
    NodeRef local(NodeRef init) noexcept;
    NodeRef if_(const NodeRef &cond, const Pooled<BasicBlock> &true_branch, const Pooled<BasicBlock> &false_branch) noexcept;
    NodeRef switch_(const NodeRef &value, luisa::span<const SwitchCase> cases, const Pooled<BasicBlock> &default_) noexcept;
    NodeRef loop(const Pooled<BasicBlock> &body, const NodeRef &cond) noexcept;
    NodeRef generic_loop(const Pooled<BasicBlock> &prepare, const NodeRef &cond, const Pooled<BasicBlock> &body, const Pooled<BasicBlock> &update) noexcept;
    static Pooled<BasicBlock> finish(IrBuilder &&builder) noexcept;
    void set_insert_point(const NodeRef &node) noexcept;

    template<class F>
    static Pooled<BasicBlock> with(const CppOwnedCArc<ModulePools> &pools, F &&f) {
        static_assert(std::is_invocable_v<F, IrBuilder &>);
        static_assert(std::is_same_v<std::invoke_result_t<F, IrBuilder &>, void>);
        auto _inner = luisa_compute_ir_new_builder(pools.clone());
        auto builder = IrBuilder{_inner};
        f(builder);
        return IrBuilder::finish(std::move(builder));
    }