#pragma once
#include <luisa/vstl/unique_ptr.h>
#include <luisa/core/logging.h>
#include <luisa/vstl/hash_map.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/graph/graph_deps.h>

namespace luisa::compute::graph {
enum class GraphNodeType {
    None = 0,
    Kernel = 1,
    MemoryCopy = 2,
    EventRecord = 3,
    EventWait = 4
};

class GraphBuilder;

class LC_RUNTIME_API GraphNode {
    friend class GraphBuilder;
public:
    static constexpr uint64_t invalid_node_id() noexcept { return std::numeric_limits<uint64_t>::max(); }

    GraphNode(GraphBuilder *builder, GraphNodeType type) noexcept
        : _builder{builder},
          _node_id{_builder->graph_nodes().size()},
          _type{type} {
        _arg_usage.clear();
    }
    GraphNode(GraphNode &&) noexcept = default;
    GraphNode &operator=(GraphNode &&) noexcept = default;
    virtual ~GraphNode() noexcept {}
    const unordered_map<uint64_t, Usage> &arg_usage() const noexcept { return _arg_usage; }
    uint64_t node_id() const noexcept { return _node_id; }
    GraphNodeType type() const noexcept { return _type; }
    span<GraphDependency> deps() const noexcept { return span{builder()->_deps}.subspan(_dep_begin, _dep_count); }
protected:
    template<typename T>
    using U = unique_ptr<T>;
    // virtual U<GraphNode> clone() const noexcept = 0;
    GraphBuilder *builder() const noexcept { return _builder; }
    void add_arg_usage(uint64_t arg_id, Usage usage) noexcept { _arg_usage.emplace(arg_id, usage); }
private:
    void set_dep_range(size_t begin, size_t count) noexcept {
        _dep_begin = begin;
        _dep_count = count;
    }
    unordered_map<uint64_t, Usage> _arg_usage;
    friend class GraphBuilder;
    GraphBuilder *_builder{nullptr};
    uint64_t _node_id{invalid_node_id()};
    GraphNodeType _type{GraphNodeType::None};
    size_t _dep_begin{0};
    size_t _dep_count{0};
    void *_native_handle{nullptr};
    
};
}// namespace luisa::compute::graph