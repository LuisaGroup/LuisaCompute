#include <luisa/runtime/graph/graph_builder.h>
#include <luisa/runtime/graph/memory_node_direction.h>

namespace luisa::compute::graph {
template<typename T>
inline GraphNode& GraphSubVar<BufferView<T>>::copy_from(const GraphVar<BufferView<T>> &view) noexcept {
    return *GraphBuilder::add_memory_node(view.sub_var_id(), this->sub_var_id(), MemoryNodeDirection::DeviceToDevice);
}

template<typename T>
inline GraphNode &GraphSubVar<BufferView<T>>::copy_to(const GraphVar<void *> &host) noexcept {
    return *GraphBuilder::add_memory_node(this->sub_var_id(), host.sub_var_id(), MemoryNodeDirection::DeviceToHost);
}

template<typename T>
inline GraphNode &GraphSubVar<BufferView<T>>::copy_from(const GraphVar<void *> &view) noexcept {
    return *GraphBuilder::add_memory_node(view.sub_var_id(), this->sub_var_id(), MemoryNodeDirection::HostToDevice);
}

template<typename T>
inline GraphVar<BufferView<T>>::sub_var_type GraphVar<BufferView<T>>::view(const GraphUInt &offset, const GraphUInt &size) const noexcept {
    this->is_input_var();
    LUISA_ASSERT(this->is_input_var(), "can't create view from a sub var");
    return *this->emplace_sub_var<sub_var_type>(offset.sub_var_id(), size.sub_var_id());
}
}// namespace luisa::compute::graph