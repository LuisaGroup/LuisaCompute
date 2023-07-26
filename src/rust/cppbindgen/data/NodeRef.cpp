[[nodiscard]] const Instruction *NodeRef::operator->() const noexcept {
    auto node = luisa_compute_ir_node_get(_inner);
    auto inst = node->instruction.get();
    return reinterpret_cast<const Instruction *>(inst);
}