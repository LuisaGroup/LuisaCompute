#include <luisa/runtime/graph/graph_buffer_var.h>
#include <luisa/runtime/graph/graph_builder.h>
#include <luisa/runtime/graph/sparse_2d_array.h>
namespace luisa::compute::graph {
uint64_t GraphBufferVarBase::eval_offset(GraphBuilder *builder) const noexcept {
    // LUISA_ASSERT(is_sub_var(), "this function can only be invoked when it is a sub var");
    auto offset_var = builder->sub_var(_buffer_offset_var_id)->cast<GraphUInt>();
    return offset_var->eval();
}

uint64_t GraphBufferVarBase::eval_size(GraphBuilder *builder) const noexcept {
    // LUISA_ASSERT(is_sub_var(), "this function can only be invoked when it is a sub var");
    auto size_var = builder->sub_var(_buffer_size_var_id)->cast<GraphUInt>();
    return size_var->eval();
}

GraphBufferVarBase::BufferViewBase GraphBufferVarBase::eval_buffer_view_base(GraphBuilder *builder) const noexcept {
    if (is_input_var()) return _buffer_view_base;
    if (is_sub_var()) {
        auto input_var = builder->input_var(input_var_id())->cast<GraphBufferVarBase>();
        // copy and modify
        auto current_buffer_view_base = input_var->_buffer_view_base;

        auto this_offset = eval_offset(builder) + current_buffer_view_base.offset();
        current_buffer_view_base.set_offset(this_offset);
        auto this_size = eval_size(builder);
        current_buffer_view_base.set_size(this_size);
        return current_buffer_view_base;
    }
}
}// namespace luisa::compute::graph