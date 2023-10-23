#pragma
#include <luisa/ir_v2/ir_v2.h>

namespace luisa::compute::ir_v2 {
class UseDefAnalysis {
    // _use[a] = {b, c, d} means a is used by b, c, d
    luisa::unordered_map<Node *, luisa::unordered_set<Node *>> _used_by;

    // nodes that are implicitly used
    luisa::unordered_set<Node *> _root;
    Module &module;
    void visit_block(BasicBlock *block) noexcept;
    void visit_node(Node *node) noexcept;
public:
    UseDefAnalysis(Module &module);
    void add_to_root(Node *node) noexcept {
        _root.insert(node);
    }
    void run() noexcept;
    void clear() noexcept {
        _used_by.clear();
        _root.clear();
    }
    [[nodiscard]] const luisa::unordered_set<Node *> &used_by(Node *n) const noexcept {
        auto it = _used_by.find(n);
        LUISA_ASSERT(it != _used_by.end(), "{} not in use-def analysis", (void*)n);
        return it->second;
    }
    [[nodiscard]] const luisa::unordered_set<Node *> &root() const noexcept {
        return _root;
    }
    [[nodiscard]] bool in_use(Node *node) const noexcept {
        {
            auto it = _used_by.find(node);
            LUISA_ASSERT(it != _used_by.end(), "{} not in use-def analysis", (void*)node);
        }
        // check if n can be reached from root
        luisa::unordered_set<Node *> visited;
        luisa::vector<Node *> queue;
        queue.reserve(_root.size());
        for (auto r : _root) {
            queue.push_back(r);
        }
        while (!queue.empty()) {
            auto n = queue.back();
            queue.pop_back();
            if (visited.contains(n)) { continue; }
            visited.insert(n);
            if (n == node) { return true; }
            auto &us = _used_by.at(n);
            if (us.contains(node)) { return true; }
            for (auto u : us) {
                queue.push_back(u);
            }
        }
        return false;
    }
};
}// namespace luisa::compute::ir_v2