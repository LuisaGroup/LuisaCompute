#include "restructure_cfg_construct_boundary.h"

#include <luisa/core/stl/unordered_map.h>
#include <luisa/core/stl/vector.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/passes/dom_tree.h>

namespace luisa::compute::xir {

class EnclosingConstructBoundaryAnalysis::Impl {
private:
    struct Owner {
        BasicBlock *header{nullptr};
        BasicBlock *merge{nullptr};
    };

    luisa::unordered_map<BasicBlock *, luisa::vector<Owner>>
        _owners;

public:
    explicit Impl(FunctionDefinition *definition) noexcept {
        if (definition == nullptr) { return; }
        auto add = [&](BasicBlock *boundary,
                       BasicBlock *header,
                       BasicBlock *merge) noexcept {
            if (boundary == nullptr || header == nullptr ||
                merge == nullptr) {
                return;
            }
            auto &list = _owners[boundary];
            for (auto owner : list) {
                if (owner.header == header) { return; }
            }
            list.emplace_back(Owner{
                .header = header, .merge = merge});
        };
        definition->traverse_basic_blocks(
            [&](BasicBlock *header) noexcept {
                if (header == nullptr ||
                    !header->is_terminated()) {
                    return;
                }
                auto *terminator = header->terminator();
                auto *control_flow_merge =
                    terminator->control_flow_merge();
                auto *merge = control_flow_merge == nullptr ?
                                  nullptr :
                                  control_flow_merge->merge_block();
                add(merge, header, merge);
                if (terminator->isa<LoopInst>()) {
                    auto *loop =
                        static_cast<LoopInst *>(terminator);
                    add(loop->prepare_block(), header, merge);
                    add(loop->update_block(), header, merge);
                } else if (terminator->isa<SimpleLoopInst>()) {
                    add(static_cast<SimpleLoopInst *>(terminator)
                            ->body_block(),
                        header, merge);
                }
            });
    }

    [[nodiscard]] bool contains(
        BasicBlock *construct_header,
        BasicBlock *entry,
        const DomTree &dominance) const noexcept {
        if (construct_header == nullptr || entry == nullptr ||
            !dominance.contains(construct_header)) {
            return false;
        }
        auto iter = _owners.find(entry);
        if (iter == _owners.end()) { return false; }
        for (auto owner : iter->second) {
            if (owner.header == nullptr ||
                owner.header == construct_header ||
                owner.merge == nullptr ||
                !dominance.contains(owner.header) ||
                !dominance.contains(owner.merge) ||
                !dominance.strictly_dominates(
                    owner.header, construct_header)) {
                continue;
            }
            // H is inside owner C iff C.header strictly dominates H and
            // C.merge does not strictly dominate H. This is the sparse
            // dominator-tree form of the lexical active-scope predicate.
            if (!dominance.strictly_dominates(
                    owner.merge, construct_header)) {
                return true;
            }
        }
        return false;
    }
};

EnclosingConstructBoundaryAnalysis::
    EnclosingConstructBoundaryAnalysis(
        FunctionDefinition *definition) noexcept
    : _impl{luisa::make_unique<Impl>(definition)} {}

EnclosingConstructBoundaryAnalysis::
    ~EnclosingConstructBoundaryAnalysis() noexcept = default;

EnclosingConstructBoundaryAnalysis::
    EnclosingConstructBoundaryAnalysis(
        EnclosingConstructBoundaryAnalysis &&) noexcept = default;

EnclosingConstructBoundaryAnalysis &
EnclosingConstructBoundaryAnalysis::operator=(
    EnclosingConstructBoundaryAnalysis &&) noexcept = default;

bool EnclosingConstructBoundaryAnalysis::contains(
    BasicBlock *construct_header,
    BasicBlock *entry,
    const DomTree &dominance) const noexcept {
    return _impl->contains(
        construct_header, entry, dominance);
}

}// namespace luisa::compute::xir
