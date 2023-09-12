#pragma once
#include <luisa/runtime/graph/graph_var.h>
#include <luisa/runtime/graph/graph_basic_var.h>
#include <luisa/runtime/graph/graph_host_memory_var.h>
#include <luisa/runtime/graph/input_sub_var_collection.h>

namespace luisa::compute::graph {
class LC_RUNTIME_API GraphBufferVarBase : public GraphVarBase {
public:
    class BufferViewBase {
    public:
        BufferViewBase() = default;
        template<typename T>
        BufferViewBase(const BufferView<T> &view) noexcept
            : _native_handle{view.native_handle()},
              _handle{view.handle()},
              _offset_bytes{view.offset_bytes()},
              _element_stride{view.stride()},
              _size{view.size()},
              _total_size{view.size()} {}

        void *_native_handle;
        uint64_t _handle;
        size_t _offset_bytes;
        size_t _element_stride;
        size_t _size;
        size_t _total_size;

        [[nodiscard]] auto handle() const noexcept { return _handle; }
        [[nodiscard]] auto native_handle() const noexcept { return _native_handle; }
        [[nodiscard]] constexpr auto stride() const noexcept { return _element_stride; }
        [[nodiscard]] auto size() const noexcept { return _size; }
        [[nodiscard]] auto offset() const noexcept { return _offset_bytes / _element_stride; }
        [[nodiscard]] auto offset_bytes() const noexcept { return _offset_bytes; }
        [[nodiscard]] auto size_bytes() const noexcept { return _size * _element_stride; }
        void set_offset(uint64_t offset) noexcept { _offset_bytes = offset * _element_stride; }
        void set_size(uint64_t size) noexcept { _size = size; }

        friend bool operator==(const BufferViewBase &lhs, const BufferViewBase &rhs) noexcept {
            return lhs._handle == rhs._handle &&
                   lhs._offset_bytes == rhs._offset_bytes &&
                   lhs._element_stride == rhs._element_stride &&
                   lhs._size == rhs._size;
        }

        friend bool operator!=(const BufferViewBase &lhs, const BufferViewBase &rhs) noexcept {
            return !(lhs == rhs);
        }

        template<typename T>
        BufferView<T> as() const noexcept {
            return BufferView<T>{_native_handle, _handle, _element_stride, _offset_bytes, _size, _total_size};
        }
    };
    using clone_ptr = U<GraphBufferVarBase>;
    GraphBufferVarBase(GraphInputVarId input_var_id) noexcept
        : GraphVarBase{input_var_id, GraphResourceTag::Buffer} {}

    GraphBufferVarBase(GraphSubVarId id, const GraphBufferVarBase &src, GraphSubVarId offset_id, GraphSubVarId size_id) noexcept
        : GraphVarBase{id, src, {offset_id, size_id}},
          _buffer_offset_var_id{offset_id},
          _buffer_size_var_id{size_id} {
    }

    virtual U<GraphBufferVarBase> clone() const noexcept = 0;

    BufferViewBase eval_buffer_view_base(GraphBuilder *builder) const noexcept;
protected:
    friend class GraphBuilder;
    friend class InputSubVarCollection<GraphBufferVarBase>;
    using GraphVarBase::GraphVarBase;

    GraphSubVarId _buffer_offset_var_id{GraphSubVarId::invalid_id};
    GraphSubVarId _buffer_size_var_id{GraphSubVarId::invalid_id};

    BufferViewBase _buffer_view_base{};

    template<typename T>
    void input_var_update_check(const BufferView<T> &input_view) noexcept {
        LUISA_ASSERT(this->is_input_var(), "this function can only be invoked when it is a input var");
        _need_update = _need_update || _buffer_view_base != BufferViewBase{input_view};
        if (_need_update) {
            LUISA_INFO("input_var {} need update: new buffer view handle = {}", this->input_var_id().value(), input_view.handle());
            _buffer_view_base = input_view;
        }
    }

    void sub_var_update_check(GraphBuilder *builder) noexcept {
        LUISA_ASSERT(this->is_sub_var(), "this function can only be invoked when it is a sub var");
        auto new_view = eval_buffer_view_base(builder);
        _need_update = _need_update || _buffer_view_base != new_view;
        if (_need_update) {
            LUISA_INFO("sub_var {} need update: new buffer view handle = {}", this->input_var_id().value(), new_view.native_handle());
            _buffer_view_base = new_view;
        }
    }

    uint64_t eval_offset(GraphBuilder *builder) const noexcept;
    uint64_t eval_size(GraphBuilder *builder) const noexcept;

    virtual void update_kernel_node_cmd_encoder(
        size_t arg_idx_in_kernel_parms, KernelNodeCmdEncoder *encoder) const noexcept override {
        encoder->update_buffer(arg_idx_in_kernel_parms,
                               _buffer_view_base.handle(),
                               _buffer_view_base.offset_bytes(),
                               _buffer_view_base.size_bytes());
    }
};

template<typename T>
class GraphSubVar<BufferView<T>> : public GraphBufferVarBase {
    friend class GraphBuilder;
    template<typename... Args>
    friend class Graph;
public:
    using this_type = GraphVar<BufferView<T>>;
    using value_type = T;

    using GraphBufferVarBase::GraphBufferVarBase;

    GraphNode &copy_from(const GraphVar<BufferView<T>> &view) noexcept;
    GraphNode &copy_to(const GraphVar<void *> &host) noexcept;
    GraphNode &copy_from(const GraphVar<void *> &host) noexcept;
    auto eval() const noexcept { return _buffer_view_base.as<T>(); }
    virtual clone_ptr clone() const noexcept override { return make_unique<GraphSubVar<BufferView<T>>>(*this); }
};

template<typename T>
class GraphVar<BufferView<T>> : public GraphSubVar<BufferView<T>> {
    friend class GraphBuilder;
    template<typename... Args>
    friend class Graph;
public:
    using this_type = GraphVar<BufferView<T>>;
    using sub_var_type = GraphSubVar<BufferView<T>>;
    using clone_ptr = typename GraphSubVar<BufferView<T>>::clone_ptr;
    using value_type = T;

    using sub_var_type::sub_var_type;

    sub_var_type view(const GraphUInt &offset, const GraphUInt &size) const noexcept;
    virtual clone_ptr clone() const noexcept override { return make_unique<GraphVar<BufferView<T>>>(*this); }
};

template<typename T>
using GraphBuffer = GraphVar<BufferView<T>>;
}// namespace luisa::compute::graph

#include <luisa/runtime/graph/graph_buffer_var.inl>