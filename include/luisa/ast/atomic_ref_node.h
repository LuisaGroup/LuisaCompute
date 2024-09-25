#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/ast/expression.h>

namespace luisa::compute::detail {

class LC_AST_API AtomicRefNode {

private:
    const AtomicRefNode *_parent;
    const Expression *_value;

private:
    explicit AtomicRefNode(const RefExpr *self) noexcept;
    AtomicRefNode(const AtomicRefNode *parent, const Expression *index) noexcept;

public:
    [[nodiscard]] static const AtomicRefNode *create(const RefExpr *ref) noexcept;
    [[nodiscard]] const AtomicRefNode *access(const Expression *index) const noexcept;
    [[nodiscard]] const AtomicRefNode *access(size_t member_index) const noexcept;
    [[nodiscard]] const Expression *operate(CallOp op, luisa::span<const Expression *const> values) const noexcept;
    [[nodiscard]] const Expression *operate(CallOp op, std::initializer_list<const Expression *> values) const noexcept;
};

}// namespace luisa::compute::detail