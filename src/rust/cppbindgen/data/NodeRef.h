    [[nodiscard]] const Node *operator->() const noexcept;
    [[nodiscard]] const Node *get() const noexcept;
    static NodeRef from_raw(raw::NodeRef raw) noexcept {
        auto ret = NodeRef{};
        ret._inner = raw;
        return ret;
    }