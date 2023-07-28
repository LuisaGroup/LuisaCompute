#include <luisa/core/magic_enum.h>
#include <luisa/core/logging.h>
#include <luisa/ast/atomic_ref_node.h>
#include <luisa/ast/function_builder.h>

namespace luisa::compute::detail {

AtomicRefNode::AtomicRefNode(const RefExpr *self) noexcept
    : _parent{nullptr}, _value{self} {
    LUISA_ASSERT(self->variable().tag() == Variable::Tag::BUFFER ||
                     self->variable().is_shared(),
                 "Atomic operation is only allowed on buffers or shared memory.");
}

AtomicRefNode::AtomicRefNode(const AtomicRefNode *parent,
                             const Expression *index) noexcept
    : _parent{parent}, _value{index} {
    LUISA_ASSERT(parent != nullptr, "Null parent for non-root AtomicRefNode.");
    LUISA_ASSERT(index->type()->is_int32() ||
                     index->type()->is_uint32(),
                 "Only integral types are allowed as "
                 "AtomicRefNode indices (got {}).",
                 index->type()->description());
}

const Expression *AtomicRefNode::operate(
    CallOp op, luisa::span<const Expression *const> values) const noexcept {

    LUISA_ASSERT(is_atomic_operation(op),
                 "Only atomic operations are allowed "
                 "on AtomicRefNode (got {}).",
                 to_string(op));
    LUISA_ASSERT((op == CallOp::ATOMIC_COMPARE_EXCHANGE && values.size() == 2u) ||
                     (op != CallOp::ATOMIC_COMPARE_EXCHANGE && values.size() == 1u),
                 "Invalid number of arguments for atomic operation {} (got {}).",
                 to_string(op), values.size());
    luisa::fixed_vector<const Expression *, 16u> args;
    for (auto node = this; node != nullptr; node = node->_parent) {
        args.emplace_back(node->_value);
    }
    std::reverse(args.begin(), args.end());

    auto access_chain_string = [&] {
        auto s = luisa::string{args.front()->type()->description()};
        for (auto index : luisa::span{args}.subspan(1)) {
            s.append(luisa::format(" -> {}", index->type()->description()));
        }
        return s;
    };

    auto type = args.front()->type();
    LUISA_ASSERT(type->is_buffer() || type->is_array(),
                 "Atomic operation is only allowed on "
                 "buffers or shared-memory arrays (got {}).",
                 type->description());
    for (auto index : luisa::span{args}.subspan(1)) {
        LUISA_ASSERT(index->type()->is_int32() ||
                         index->type()->is_uint32(),
                     "Only integral types are allowed as "
                     "AtomicRefNode indices (got {}).",
                     index->type()->description());
        switch (type->tag()) {
            case Type::Tag::VECTOR: type = type->element(); break;
            case Type::Tag::MATRIX: type = Type::vector(type->element(), type->dimension()); break;
            case Type::Tag::ARRAY: type = type->element(); break;
            case Type::Tag::STRUCTURE: {
                LUISA_ASSERT(index->tag() == Expression::Tag::LITERAL,
                             "Only literal indices are allowed for "
                             "AtomicRefNode on structures (got {}).",
                             index->type()->description());
                auto literal = static_cast<const LiteralExpr *>(index)->value();
                auto member_index = luisa::holds_alternative<int>(literal) ?
                                        static_cast<uint>(luisa::get<int>(literal)) :
                                        luisa::get<uint>(literal);
                LUISA_ASSERT(member_index < type->members().size(),
                             "Invalid member index {} for "
                             "atomic operation {} on {}.",
                             member_index, to_string(op),
                             type->description());
                type = type->members()[member_index];
                break;
            }
            case Type::Tag::BUFFER: type = type->element(); break;
            default: LUISA_ERROR_WITH_LOCATION(
                "Invalid node type {} in access chain "
                "for atomic operation {} ({}).",
                type->description(), to_string(op),
                access_chain_string());
        }
    }

    // append extra parameters
    for (auto value : values) {
        LUISA_ASSERT(value->type() == type,
                     "Type mismatch for atomic operation {} (got {}, expected {}).",
                     to_string(op),
                     value->type()->description(),
                     type->description());
        args.emplace_back(value);
    }

    // create atomic call
    return FunctionBuilder::current()->call(type, op, luisa::span{args});
}

const Expression *AtomicRefNode::operate(
    CallOp op, std::initializer_list<const Expression *> values) const noexcept {
    return operate(op, luisa::span{values.begin(), values.end()});
}

const AtomicRefNode *AtomicRefNode::access(const Expression *index) const noexcept {
    AtomicRefNode node{this, index};
    return FunctionBuilder::current()
        ->create_temporary<AtomicRefNode>(node);
}

const AtomicRefNode *AtomicRefNode::access(size_t member_index) const noexcept {
    return access(FunctionBuilder::current()->literal(
        Type::of<uint>(), static_cast<uint>(member_index)));
}

const AtomicRefNode *AtomicRefNode::create(const RefExpr *ref) noexcept {
    AtomicRefNode node{ref};
    return FunctionBuilder::current()
        ->create_temporary<AtomicRefNode>(node);
}

}// namespace luisa::compute::detail

