#pragma once

#include <luisa/core/stl/memory.h>

namespace luisa::compute::xir {

class BasicBlock;
class DomTree;
class FunctionDefinition;

// Immutable index from structural boundary blocks to their owning constructs.
// One instance serves every entry query for one CFG version.
class EnclosingConstructBoundaryAnalysis {
private:
    class Impl;
    luisa::unique_ptr<Impl> _impl;

public:
    explicit EnclosingConstructBoundaryAnalysis(
        FunctionDefinition *definition) noexcept;
    ~EnclosingConstructBoundaryAnalysis() noexcept;
    EnclosingConstructBoundaryAnalysis(
        EnclosingConstructBoundaryAnalysis &&) noexcept;
    EnclosingConstructBoundaryAnalysis &operator=(
        EnclosingConstructBoundaryAnalysis &&) noexcept;
    EnclosingConstructBoundaryAnalysis(
        const EnclosingConstructBoundaryAnalysis &) = delete;
    EnclosingConstructBoundaryAnalysis &operator=(
        const EnclosingConstructBoundaryAnalysis &) = delete;

    [[nodiscard]] bool contains(
        BasicBlock *construct_header,
        BasicBlock *entry,
        const DomTree &dominance) const noexcept;
};

}// namespace luisa::compute::xir
