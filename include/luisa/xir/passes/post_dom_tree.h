#pragma once

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/xir/function.h>

namespace luisa::compute::xir {

class PostDomTree;

class LUISA_XIR_API PostDomTreeNode : public concepts::Noncopyable {

private:
    BasicBlock *_block;
    const PostDomTreeNode *_parent;
    luisa::vector<const PostDomTreeNode *> _children;
    luisa::vector<const PostDomTreeNode *> _frontiers;

public:
    explicit PostDomTreeNode(BasicBlock *block) noexcept;
    void add_child(PostDomTreeNode *child) noexcept;
    void add_frontier(PostDomTreeNode *frontier) noexcept;

public:
    [[nodiscard]] auto parent() const noexcept { return _parent; }
    [[nodiscard]] auto block() const noexcept { return _block; }
    [[nodiscard]] auto children() const noexcept { return luisa::span{_children}; }
    [[nodiscard]] auto frontiers() const noexcept { return luisa::span{_frontiers}; }
};

class LUISA_XIR_API PostDomTree : public concepts::Noncopyable {

private:
    luisa::unordered_map<BasicBlock *, luisa::unique_ptr<PostDomTreeNode>> _nodes;
    luisa::unique_ptr<PostDomTreeNode> _virtual_root;
    const PostDomTreeNode *_root;

public: /* for internal usage only */
    PostDomTree() noexcept;
    PostDomTreeNode *add_or_get_node(BasicBlock *block) noexcept;
    void set_root(PostDomTreeNode *root) noexcept;
    void compute_post_dominance_frontiers() noexcept;

public:
    [[nodiscard]] auto root() const noexcept { return _root; }
    [[nodiscard]] auto &nodes() const noexcept { return _nodes; }
    [[nodiscard]] auto node(BasicBlock *block) const noexcept -> const PostDomTreeNode *;
    [[nodiscard]] auto node_or_null(BasicBlock *block) const noexcept -> const PostDomTreeNode *;
    [[nodiscard]] bool contains(BasicBlock *block) const noexcept;
    [[nodiscard]] bool post_dominates(BasicBlock *a, BasicBlock *b) const noexcept;
    [[nodiscard]] bool strictly_post_dominates(BasicBlock *a, BasicBlock *b) const noexcept;
    [[nodiscard]] auto immediate_post_dominator(BasicBlock *block) const noexcept -> BasicBlock *;
};

/// Null and declaration-only functions yield an empty tree.
[[nodiscard]] LUISA_XIR_API PostDomTree compute_post_dom_tree(Function *function) noexcept;

}// namespace luisa::compute::xir
