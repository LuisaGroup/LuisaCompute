#pragma once

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/ast/type.h>
#include <luisa/ast/type_registry.h>
#include <luisa/ir_v2/ir_v2_fwd.h>
#include <luisa/ir_v2/ir_v2_defs.h>
#include <luisa/core/logging.h>

namespace luisa::compute::ir_v2 {
class Pool;
struct Node {
    Node *prev = nullptr;
    Node *next = nullptr;
    BasicBlock *scope = nullptr;
    Instruction *inst = nullptr;
    const Type *ty = Type::of<void>();
    Node() noexcept = default;
    Node(Instruction *inst, const Type *ty) : inst{inst}, ty(ty) {}

    void insert_after_this(Node *n) noexcept {
        LUISA_ASSERT(n != nullptr, "bad node");
        LUISA_ASSERT(this != nullptr, "bad node");
        LUISA_ASSERT(next != nullptr, "bad node");
        n->prev = this;
        n->next = next;
        next->prev = n;
        next = n;
    }
    void insert_before_this(Node *n) noexcept {
        LUISA_ASSERT(n != nullptr, "bad node");
        LUISA_ASSERT(this != nullptr, "bad node");
        LUISA_ASSERT(prev != nullptr, "bad node");
        n->prev = prev;
        n->next = this;
        prev->next = n;
        prev = n;
    }
};

class BasicBlock {
    Node *_first;
    Node *_last;
public:
    BasicBlock(Pool &pool) noexcept;
    template<class F>
    void for_each(F &&f) const noexcept {
        auto n = _first.next;
        while (n != _last) {
            f(n);
            n = n->next;
        }
    }
};

class Pool {
    using Deleter = void (*)(void *) noexcept;
public:
    template<typename T, typename... Args>
    [[nodiscard]] T *alloc(Args &&...args) noexcept {
        auto ptr = luisa::new_with_allocator<T>(std::forward<Args>(args)...);
        _deleters.emplace_back([](void *p) noexcept {
            luisa::delete_with_allocator<T>(static_cast<T *>(p));
        });
        return ptr;
    }
    ~Pool() noexcept {
        for (auto d : _deleters) { d(nullptr); }
    }
private:
    luisa::vector<Deleter> _deleters;
};

class IrBuilder {
    luisa::shared_ptr<Pool> _pool;
    Node *_insert_point = nullptr;
    BasicBlock *_current_bb = nullptr;
    Node *append(Node *n) {
        LUISA_ASSERT(n != nullptr, "bad node");
        LUISA_ASSERT(_current_bb != nullptr, "bad node");
        LUISA_ASSERT(n->scope == nullptr, "bad node");
        n->scope = _current_bb;
        _insert_point->insert_after_this(n);
        _insert_point = n;
        return n;
    }
public:
    static IrBuilder create_without_bb(luisa::shared_ptr<Pool> pool) noexcept {
        auto builder = IrBuilder{pool};
        return builder;
    }
    IrBuilder(luisa::shared_ptr<Pool> pool) noexcept : _pool{pool} {
    }
    [[nodiscard]] Pool &pool() const noexcept {
        return *_pool;
    }
    void set_insert_point(Node *n) noexcept {
        LUISA_ASSERT(n != nullptr, "bad node");
        if (_current_bb != nullptr) {
            LUISA_ASSERT(n->scope == _current_bb, "bad node");
        }
        _insert_point = n;
    }
    [[nodiscard]] Node *insert_point() const noexcept {
        return _insert_point;
    }

    [[nodiscard]] Node *call(const Func *f, luisa::span<Node *> args, const Type *ty) noexcept;
    Node *if_(Node *cond, BasicBlock *true_branch, BasicBlock *false_branch) noexcept;
    Node *generic_loop(BasicBlock *perpare, Node *cond, BasicBlock *body, BasicBlock *after) noexcept;
    Node *switch_(Node *value, luisa::span<SwitchCase> cases, BasicBlock *default_branch) noexcept;
    const BasicBlock *finish() && noexcept {
        LUISA_ASSERT(_current_bb != nullptr, "IrBuilder is not configured to produce a basic block");
        return _current_bb;
    }
    Node *return_(Node *value) noexcept {
        auto ret = _pool->alloc<Return>(value);
        return append(_pool->alloc<Node>(ret, Type::of<void>()));
    }
    Node *break_() noexcept {
        auto br = _pool->alloc<Break>();
        return append(_pool->alloc<Node>(br, Type::of<void>()));
    }
    Node *continue_() noexcept {
        auto cont = _pool->alloc<Continue>();
        return append(_pool->alloc<Node>(cont, Type::of<void>()));
    }
};

struct Module {
    BasicBlock *entry = nullptr;
};

class Transform {
    virtual void run(Module &module) noexcept = 0;
};

}// namespace luisa::compute::ir_v2