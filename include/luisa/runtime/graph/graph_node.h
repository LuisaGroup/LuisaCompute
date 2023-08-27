#pragma once
#include <luisa/vstl/unique_ptr.h>
#include <luisa/core/logging.h>
#include <luisa/vstl/hash_map.h>
namespace luisa::compute::graph {
class GraphBuilder;

class GraphNode {
public:
    GraphNode(GraphBuilder *builder, span<uint64_t> arg_ids) noexcept : _builder{builder} {
        _arg_set.clear();
        for (auto id : arg_ids) _arg_set.insert(id);
    }
    GraphNode(GraphNode &&) noexcept = default;
    GraphNode &operator=(GraphNode &&) noexcept = default;
    virtual ~GraphNode() noexcept {}
    const unordered_set<uint64_t> &arg_set() const noexcept { return _arg_set; }
protected:
    template<typename T>
    using U = unique_ptr<T>;
    
    GraphBuilder *builder() const noexcept { return _builder; }
private:
    unordered_set<uint64_t> _arg_set;
    friend class GraphBuilder;
    GraphBuilder *_builder{nullptr};
};
}// namespace luisa::compute::graph