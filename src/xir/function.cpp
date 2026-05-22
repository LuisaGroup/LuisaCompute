#include "luisa/core/stl/unordered_map.h"

#include <luisa/ast/type.h>
#include <luisa/core/logging.h>
#include <luisa/xir/function.h>

namespace luisa::compute::xir {

Function::Function(Module *parent_module, const Type *type) noexcept
    : Super{parent_module, type}, _arguments{this}, _basic_blocks{this} {}

Argument *Function::create_argument(const Type *type, bool by_ref) noexcept {
    if (type->is_resource()) {
        LUISA_ASSERT(!by_ref, "Resource argument must not be passed by reference.");
        return create_resource_argument(type);
    }
    return by_ref ? static_cast<Argument *>(create_reference_argument(type)) :
                    static_cast<Argument *>(create_value_argument(type));
}

ValueArgument *Function::create_value_argument(const Type *type) noexcept {
    LUISA_ASSERT(!type->is_resource(), "Resource argument must be created with create_resource_argument.");
    LUISA_ASSERT(!type->is_custom(), "Opaque argument must be created with create_reference_argument.");
    auto argument = luisa::make_managed<ValueArgument>(this, type);
    return static_cast<ValueArgument *>(_arguments.push_back(std::move(argument)));
}

ReferenceArgument *Function::create_reference_argument(const Type *type) noexcept {
    LUISA_ASSERT(!type->is_resource(), "Resource argument must be created with create_resource_argument.");
    auto argument = luisa::make_managed<ReferenceArgument>(this, type);
    return static_cast<ReferenceArgument *>(_arguments.push_back(std::move(argument)));
}

ResourceArgument *Function::create_resource_argument(const Type *type) noexcept {
    LUISA_ASSERT(type->is_resource(), "Resource argument must be created with create_resource_argument.");
    auto argument = luisa::make_managed<ResourceArgument>(this, type);
    return static_cast<ResourceArgument *>(_arguments.push_back(std::move(argument)));
}

BasicBlock *Function::create_basic_block() noexcept {
    auto block = luisa::make_managed<BasicBlock>(this);
    return _basic_blocks.push_back(std::move(block));
}

SentinelFunction::SentinelFunction(Module *parent_module) noexcept
    : Function{parent_module, nullptr} {}

DerivedFunctionTag SentinelFunction::derived_function_tag() const noexcept {
    LUISA_ERROR_WITH_LOCATION("Sentinel function should not be used.");
}

void FunctionDefinition::set_body_block(BasicBlock *block) noexcept {
    LUISA_DEBUG_ASSERT(block != nullptr, "Invalid body block.");
    block->_set_parent_function(this);
    _body_block = block;
}

BasicBlock *FunctionDefinition::create_body_block(bool overwrite_existing) noexcept {
    LUISA_ASSERT(_body_block == nullptr || overwrite_existing, "Body block already exists.");
    auto new_block = create_basic_block();
    set_body_block(new_block);
    return new_block;
}

namespace detail {

void traverse_basic_block_pre_order(luisa::unordered_set<BasicBlock *> &visited, BasicBlock *block,
                                    void *visit_ctx, void (*visit)(void *, BasicBlock *)) noexcept {
    if (visited.emplace(block).second) {
        visit(visit_ctx, block);
        if (!block->is_terminated()) { return; }
        auto terminator = block->terminator();
        for (auto use : terminator->operand_uses()) {
            if (auto v = use->value(); v != nullptr && v->isa<BasicBlock>()) {
                traverse_basic_block_pre_order(visited, static_cast<BasicBlock *>(v), visit_ctx, visit);
            }
        }
    }
}

void traverse_basic_block_post_order(luisa::unordered_set<BasicBlock *> &visited, BasicBlock *block,
                                     void *visit_ctx, void (*visit)(void *, BasicBlock *)) noexcept {
    if (visited.emplace(block).second) {
        if (block->is_terminated()) {
            auto terminator = block->terminator();
            for (auto use : terminator->operand_uses()) {
                if (auto v = use->value(); v != nullptr && v->isa<BasicBlock>()) {
                    traverse_basic_block_post_order(visited, static_cast<BasicBlock *>(v), visit_ctx, visit);
                }
            }
        }
        visit(visit_ctx, block);
    }
}

}// namespace detail

void FunctionDefinition::_traverse_basic_block_pre_order(BasicBlock *block, void *visit_ctx,
                                                         void (*visit)(void *, BasicBlock *)) noexcept {
    luisa::unordered_set<BasicBlock *> visited;
    detail::traverse_basic_block_pre_order(visited, block, visit_ctx, visit);
}

void FunctionDefinition::_traverse_basic_block_post_order(BasicBlock *block, void *visit_ctx, void (*visit)(void *, BasicBlock *)) noexcept {
    luisa::unordered_set<BasicBlock *> visited;
    detail::traverse_basic_block_post_order(visited, block, visit_ctx, visit);
}

void FunctionDefinition::_traverse_basic_block_reverse_pre_order(BasicBlock *block, void *visit_ctx, void (*visit)(void *, BasicBlock *)) noexcept {
    luisa::vector<BasicBlock *> stack;
    _traverse_basic_block_pre_order(block, &stack, [](void *ctx, BasicBlock *bb) noexcept {
        static_cast<luisa::vector<BasicBlock *> *>(ctx)->emplace_back(bb);
    });
    for (auto iter = stack.rbegin(); iter != stack.rend(); ++iter) {
        visit(visit_ctx, *iter);
    }
}

void FunctionDefinition::_traverse_basic_block_reverse_post_order(BasicBlock *block, void *visit_ctx, void (*visit)(void *, BasicBlock *)) noexcept {
    luisa::vector<BasicBlock *> stack;
    _traverse_basic_block_post_order(block, &stack, [](void *ctx, BasicBlock *bb) noexcept {
        static_cast<luisa::vector<BasicBlock *> *>(ctx)->emplace_back(bb);
    });
    for (auto iter = stack.rbegin(); iter != stack.rend(); ++iter) {
        visit(visit_ctx, *iter);
    }
}

KernelFunction::KernelFunction(Module *parent_module, luisa::uint3 block_size) noexcept
    : Super{parent_module}, _block_size{} { set_block_size(block_size); }

void KernelFunction::set_block_size(luisa::uint3 size) noexcept {
    auto thread_count = size.x * size.y * size.z;
    LUISA_ASSERT(thread_count >= 32u &&
                     thread_count <= 1024u &&
                     thread_count % 32u == 0u,
                 "Invalid block size: {}.", size);
    _block_size = {size.x, size.y, size.z};
}

luisa::uint3 KernelFunction::block_size() const noexcept {
    return luisa::make_uint3(_block_size[0], _block_size[1], _block_size[2]);
}

}// namespace luisa::compute::xir
