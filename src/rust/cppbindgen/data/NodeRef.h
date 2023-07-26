    [[nodiscard]] const Instruction *operator->() const noexcept;
    static NodeRef from_raw(raw::NodeRef raw) noexcept {
        auto ret = NodeRef{};
        ret._inner = raw;
        return ret;
    }