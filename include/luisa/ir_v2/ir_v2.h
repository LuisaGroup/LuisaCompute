#pragma once

#include <luisa/core/stl/vector.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/ast/type.h>
#include <luisa/ast/type_registry.h>
#include <luisa/ir_v2/ir_v2_fwd.h>
#include <luisa/ir_v2/ir_v2_defs.h>
#include <luisa/core/logging.h>

namespace luisa::compute::ir_v2 {
class Pool;

struct LC_IR_API Node {
    Node *prev = nullptr;
    Node *next = nullptr;
    BasicBlock *scope = nullptr;
    Instruction inst;
    const Type *ty = Type::of<void>();
    Node(Instruction inst, const Type *ty) : inst{std::move(inst)}, ty(ty) {}
    // insert all nodes in bb after this node
    // bb is set to empty
    void insert_block_after(BasicBlock *bb) noexcept;
    void insert_after_this(Node *n) noexcept {
        LUISA_ASSERT(n != nullptr, "bad node");

        LUISA_ASSERT(next != nullptr, "bad node");
        n->prev = this;
        n->next = next;
        next->prev = n;
        next = n;
    }
    void insert_before_this(Node *n) noexcept {
        LUISA_ASSERT(n != nullptr, "bad node");

        LUISA_ASSERT(prev != nullptr, "bad node");
        n->prev = prev;
        n->next = this;
        prev->next = n;
        prev = n;
    }
    void remove_this() noexcept {

        LUISA_ASSERT(prev != nullptr, "bad node");
        LUISA_ASSERT(next != nullptr, "bad node");
        prev->next = next;
        next->prev = prev;
        prev = nullptr;
        next = nullptr;
    }
    luisa::optional<int32_t> get_index() const noexcept {
        _check();
        if (auto cst = inst.as<ConstantInst>()) {
            auto ty = cst->ty;
            if (ty == Type::of<int32_t>()) {
                return *reinterpret_cast<const int32_t *>(cst->value.data());
            }
        }
        return luisa::nullopt;
    }
    bool is_const() const noexcept {
        _check();
        return inst.isa<ConstantInst>();
    }
    bool is_call() const noexcept {
        _check();
        return inst.isa<CallInst>();
    }
    bool is_local() const noexcept {
        _check();
        return inst.isa<LocalInst>();
    }
    bool is_argument() const noexcept {
        _check();
        return inst.isa<ArgumentInst>();
    }
    bool is_gep() const noexcept {
        _check();
        if (auto call = inst.as<CallInst>()) {
            return call->func.isa(FuncTag::GET_ELEMENT_PTR);
        }
        return false;
    }
    bool is_lvalue() const noexcept;
private:
    void _check() const noexcept {
        LUISA_ASSERT(scope != nullptr, "bad node");
    }
};

class LC_IR_API BasicBlock {
    // NEVER set _first and _last to nullptr!!!
    Node *_first;
    Node *_last;
public:
    BasicBlock(Pool &pool) noexcept;
    template<class F>
    void for_each(F &&f) const noexcept {
        auto n = _first->next;
        while (n != _last) {
            f(n);
            n = n->next;
        }
    }
    [[nodiscard]] Node *first() const noexcept {
        return _first;
    }
    [[nodiscard]] Node *last() const noexcept {
        return _last;
    }
    void clear() noexcept {
        _first->next = _last;
        _last->prev = _first;
    }
    [[nodiscard]] bool is_empty() const noexcept {
        return _first->next == _last;
    }
};

class LC_IR_API Pool : public luisa::enable_shared_from_this<Pool> {
    using Deleter = void (*)(void *) noexcept;
public:
    template<typename T, typename... Args>
        requires std::constructible_from<T, Args...>
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

class LC_IR_API IrBuilder {
    luisa::shared_ptr<Pool> _pool;
    Node *_insert_point = nullptr;
    BasicBlock *_current_bb = nullptr;
    [[nodiscard]] Node *append(Node *n) {
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
        _current_bb = _pool->alloc<BasicBlock>(*pool);
        _insert_point = _current_bb->first();
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
    auto &pool() noexcept {
        return *_pool;
    }
    [[nodiscard]] Node *call(Func f, luisa::span<Node *> args, const Type *ty) noexcept;
    [[nodiscard]] Node *call(FuncTag tag, luisa::span<Node *> args, const Type *ty) noexcept {
        return call(Func(tag), args, ty);
    }
    template<class T>
    [[nodiscard]] Node *const_(T v) noexcept {
        luisa::vector<uint8_t> data(sizeof(T));
        std::memcpy(data.data(), &v, sizeof(T));
        auto cst = ConstantInst();
        cst.ty = Type::of<T>();
        cst.value = std::move(data);
        return append(_pool->alloc<Node>(Instruction(cst), Type::of<T>()));
    }
    [[nodiscard]] Node *extract_element(Node *value, luisa::span<uint32_t> indices, const Type *ty) noexcept {
        luisa::vector<Node *> args;
        args.push_back(value);
        for (auto i : indices) {
            args.push_back(const_(i));
        }
        return call(FuncTag::EXTRACT_ELEMENT, args, ty);
    }
    [[nodiscard]] Node *insert_element(Node *agg, Node *el, luisa::span<uint32_t> indices, const Type *ty) noexcept {
        luisa::vector<Node *> args;
        args.push_back(agg);
        args.push_back(el);
        for (auto i : indices) {
            args.push_back(const_(i));
        }
        return call(FuncTag::INSERT_ELEMENT, args, ty);
    }
    [[nodiscard]] Node *gep(Node *agg, luisa::span<uint32_t> indices, const Type *ty) noexcept {
        luisa::vector<Node *> args;
        if (agg->is_gep()) {
            auto call = agg->inst.as<CallInst>();
            LUISA_ASSERT(!call->args.empty(), "bad gep");
            LUISA_ASSERT(!call->args[0]->is_gep(), "bad gep");
            for (auto a : call->args) {
                args.push_back(a);
            }
        } else {
            args.push_back(agg);
        }
        for (auto i : indices) {
            args.push_back(const_(i));
        }
        return call(FuncTag::GET_ELEMENT_PTR, args, ty);
    }
    Node *if_(Node *cond, BasicBlock *true_branch, BasicBlock *false_branch) noexcept;
    Node *generic_loop(BasicBlock *perpare, Node *cond, BasicBlock *body, BasicBlock *after) noexcept;
    Node *switch_(Node *value, luisa::span<SwitchCase> cases, BasicBlock *default_branch) noexcept;
    const BasicBlock *finish() && noexcept {
        LUISA_ASSERT(_current_bb != nullptr, "IrBuilder is not configured to produce a basic block");
        return _current_bb;
    }
    Node *return_(Node *value) noexcept {
        auto ret = Instruction(ReturnInst(value));
        return append(_pool->alloc<Node>(std::move(ret), Type::of<void>()));
    }
    Node *break_() noexcept {
        auto br = Instruction(InstructionTag::BREAK);
        return append(_pool->alloc<Node>(std::move(br), Type::of<void>()));
    }
    Node *continue_() noexcept {
        auto cont = Instruction(InstructionTag::CONTINUE);
        return append(_pool->alloc<Node>(std::move(cont), Type::of<void>()));
    }
    Node *rev_autodiff(BasicBlock *body) noexcept {
        auto rev = Instruction(RevAutodiffInst(body));
        return append(_pool->alloc<Node>(std::move(rev), Type::of<void>()));
    }
    Node *forward_autodiff(BasicBlock *body) noexcept {
        auto fwd = Instruction(FwdAutodiffInst(body));
        return append(_pool->alloc<Node>(std::move(fwd), Type::of<void>()));
    }
    Node *update(Node *var, Node *value) noexcept {
        LUISA_ASSERT(var->is_lvalue(), "bad update");
        auto update = Instruction(UpdateInst(var, value));
        return append(_pool->alloc<Node>(std::move(update), Type::of<void>()));
    }
    [[nodiscard]] Node *local(Node *init) noexcept {
        auto local = Instruction(LocalInst(init));
        return append(_pool->alloc<Node>(std::move(local), Type::of<void>()));
    }
    [[nodiscard]] Node *zero(const Type *ty) noexcept {
        return call(FuncTag::ZERO, {}, ty);
    }
    [[nodiscard]] Node *one(const Type *ty) noexcept {
        return call(FuncTag::ONE, {}, ty);
    }
    [[nodiscard]] Node *local_zeroed(const Type *ty) noexcept {
        return this->local(this->zero(ty));
    }
    [[nodiscard]] Node *phi(luisa::span<const PhiIncoming> incomings, const Type *ty) noexcept {
        luisa::vector<PhiIncoming> args{incomings.begin(), incomings.end()};
        auto phi = Instruction(PhiInst(args));
        return append(_pool->alloc<Node>(std::move(phi), ty));
    }
};
class UseDefAnalysis;

struct Module : luisa::enable_shared_from_this<Module> {
public:
    enum class Kind {
        CALLABLE,
        KERNEL
    };
    luisa::vector<Node *> args;
    BasicBlock *entry = nullptr;
    luisa::shared_ptr<UseDefAnalysis> use_def_analysis;
    luisa::shared_ptr<Pool> pool;
    virtual ~Module() noexcept = default;
    [[nodiscard]] virtual Kind kind() const noexcept = 0;
};

struct Capture {
    Node *node = nullptr;
    Binding binding;
};

struct CallableModule : Module {
    [[nodiscard]] virtual Kind kind() const noexcept override {
        return Kind::CALLABLE;
    }
};

struct KernelModule : Module {
    luisa::vector<Capture> captures;
    std::array<uint32_t, 3> block_size;
    [[nodiscard]] virtual Kind kind() const noexcept override {
        return Kind::KERNEL;
    }
};

class Transform {
    virtual void run(Module &module) noexcept = 0;
};

void validate(Module &module) noexcept;
void normalize(Module &module) noexcept;

}// namespace luisa::compute::ir_v2