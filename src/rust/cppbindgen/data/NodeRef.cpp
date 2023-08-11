[[nodiscard]] const Node *NodeRef::operator->() const noexcept {
    return get();
}

[[nodiscard]] const Node *NodeRef::get() const noexcept {
    auto node = raw::luisa_compute_ir_node_get(_inner);
    return reinterpret_cast<const Node *>(node);
}

void NodeRef::insert_before_self(NodeRef node) noexcept {
    raw::luisa_compute_ir_node_insert_before_self(_inner, node.raw());
}

void NodeRef::insert_after_self(NodeRef node) noexcept {
    raw::luisa_compute_ir_node_insert_after_self(_inner, node.raw());
}

void NodeRef::replace_with(NodeRef node) noexcept {
    raw::luisa_compute_ir_node_replace_with(_inner, reinterpret_cast<const raw::Node *>(node.get()));
}

void NodeRef::remove() noexcept {
    raw::luisa_compute_ir_node_remove(_inner);
}