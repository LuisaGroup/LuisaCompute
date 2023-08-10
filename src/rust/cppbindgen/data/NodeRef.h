    [[nodiscard]] const Node *operator->() const noexcept;
    [[nodiscard]] const Node *get() const noexcept;
    static NodeRef from_raw(raw::NodeRef raw) noexcept {
        auto ret = NodeRef{};
        ret._inner = raw;
        return ret;
    }
    [[nodiscard]] auto raw() const noexcept { return _inner; }
    [[nodiscard]] auto operator==(const NodeRef &rhs) const noexcept { return raw() == rhs.raw(); }
    [[nodiscard]] auto valid() const noexcept { return raw() != raw::INVALID_REF; }