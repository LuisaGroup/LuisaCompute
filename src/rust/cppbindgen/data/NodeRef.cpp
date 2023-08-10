[[nodiscard]] const Node *NodeRef::operator->() const noexcept {
    return get();
}
[[nodiscard]] const Node *NodeRef::get() const noexcept {
    auto node = luisa_compute_ir_node_get(_inner);
    return reinterpret_cast<const Node *>(node);
}