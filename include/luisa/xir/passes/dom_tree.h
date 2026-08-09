#pragma once

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/function.h>

namespace luisa::compute::xir {

class DomTree;

class LUISA_XIR_API DomTreeNode : public concepts::Noncopyable {

private:
    friend class DomTree;

    BasicBlock *_block;
    const DomTreeNode *_parent;
    luisa::vector<const DomTreeNode *> _children;
    luisa::vector<const DomTreeNode *> _frontiers;
    size_t _preorder_index;
    size_t _subtree_end_index;

public:
    explicit DomTreeNode(BasicBlock *block) noexcept;
    void add_child(DomTreeNode *child) noexcept;
    void add_frontier(DomTreeNode *frontier) noexcept;

public:
    [[nodiscard]] auto parent() const noexcept { return _parent; }
    [[nodiscard]] auto block() const noexcept { return _block; }
    [[nodiscard]] auto children() const noexcept { return luisa::span{_children}; }
    [[nodiscard]] auto frontiers() const noexcept { return luisa::span{_frontiers}; }
    [[nodiscard]] bool dominates(const DomTreeNode *other) const noexcept;
};

class LUISA_XIR_API DomTree : public concepts::Noncopyable {

private:
    luisa::unordered_map<BasicBlock *, luisa::unique_ptr<DomTreeNode>> _nodes;
    const DomTreeNode *_root;

public: /* for internal usage only */
    DomTree() noexcept;
    DomTreeNode *add_or_get_node(BasicBlock *block) noexcept;
    void set_root(DomTreeNode *root) noexcept;
    void compute_ancestry_intervals() noexcept;
    void compute_dominance_frontiers() noexcept;

public:
    [[nodiscard]] auto root() const noexcept { return _root; }
    [[nodiscard]] auto &nodes() const noexcept { return _nodes; }
    [[nodiscard]] auto node(BasicBlock *block) const noexcept -> const DomTreeNode *;
    [[nodiscard]] auto node_or_null(BasicBlock *block) const noexcept -> const DomTreeNode *;
    [[nodiscard]] bool contains(BasicBlock *block) const noexcept;
    [[nodiscard]] bool dominates(BasicBlock *src, BasicBlock *dst) const noexcept;
    [[nodiscard]] bool strictly_dominates(BasicBlock *src, BasicBlock *dst) const noexcept;
    [[nodiscard]] auto immediate_dominator(BasicBlock *block) const noexcept -> BasicBlock *;
};

struct DomTreeBuildOptions {
    bool compute_dominance_frontiers{true};
};

/// Null and declaration-only functions yield an empty tree.
[[nodiscard]] LUISA_XIR_API DomTree compute_dom_tree(
    Function *function) noexcept;
[[nodiscard]] LUISA_XIR_API DomTree compute_dom_tree(
    Function *function,
    DomTreeBuildOptions options) noexcept;

}// namespace luisa::compute::xir
