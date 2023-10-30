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
inline void validate(const Type *ty) noexcept;
struct CpuExternFn : CpuExternFnData, luisa::enable_shared_from_this<CpuExternFn> {
    CpuExternFn(CpuExternFnData data) noexcept : CpuExternFnData{std::move(data)} {}
};
struct LC_IR_API Node {
    mutable Node *prev = nullptr;
    mutable Node *next = nullptr;
    BasicBlock *scope = nullptr;
    Instruction inst;
    const Type *ty = Type::of<void>();
    Node() = delete;
    Node(Instruction inst, const Type *ty) : inst{std::move(inst)}, ty(ty) {
#ifndef NDEBUG
        validate(ty);
#endif
    }
    void replace_with(Node *other) noexcept {
        LUISA_ASSERT(other->next == nullptr, "bad node");
        LUISA_ASSERT(other->prev == nullptr, "bad node");
        inst = std::move(other->inst);
    }
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
    void unlink() noexcept {

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
template<class NodeType>
class BasicBlockIterator {
    NodeType _begin;
    NodeType _end;
    friend class BasicBlock;
    BasicBlockIterator(NodeType begin, NodeType end) noexcept : _begin{begin}, _end{end} {}
public:
    [[nodiscard]] bool operator==(const BasicBlockIterator &rhs) const noexcept {
        return _begin == rhs._begin;
    }
    [[nodiscard]] bool operator!=(const BasicBlockIterator &rhs) const noexcept {
        return _begin != rhs._begin;
    }
    BasicBlockIterator &operator++() noexcept {
        LUISA_ASSERT(_begin != _end, "bad iterator");
        _begin = _begin->next;
        return *this;
    }
    BasicBlockIterator operator++(int) noexcept {
        auto copy = *this;
        ++(*this);
        return copy;
    }
    [[nodiscard]] auto &operator*() const noexcept {
        LUISA_ASSERT(_begin != _end, "bad iterator");
        return *_begin;
    }
};
class LC_IR_API BasicBlock {
    // NEVER set _first and _last to nullptr!!!
    Node *_first;
    Node *_last;
public:
    [[nodiscard]] BasicBlockIterator<Node *> begin() noexcept {
        return {_first, _last};
    }
    [[nodiscard]] BasicBlockIterator<const Node *> cbegin() const noexcept {
        return {_first, _last};
    }
    [[nodiscard]] BasicBlockIterator<Node *> end() noexcept {
        return {_last, _last};
    }
    [[nodiscard]] BasicBlockIterator<const Node *> cend() const noexcept {
        return {_last, _last};
    }
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
    [[nodiscard]] Node *call(Func f, luisa::span<const Node *const> args, const Type *ty) noexcept;
    [[nodiscard]] Node *call(FuncTag tag, luisa::span<const Node *const> args, const Type *ty) noexcept {
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
    template<class I>
        requires std::is_integral_v<I>
    [[nodiscard]] Node *extract_element(const Node *value, luisa::span<I> indices, const Type *ty) noexcept {
        luisa::vector<const Node *> args;
        args.push_back(value);
        for (auto i : indices) {
            args.push_back(const_((int32_t)i));
        }
        return call(FuncTag::EXTRACT_ELEMENT, args, ty);
    }
    template<class I>
        requires std::is_integral_v<I>
    [[nodiscard]] Node *insert_element(const Node *agg, Node *el, luisa::span<I> indices, const Type *ty) noexcept {
        luisa::vector<const Node *> args;
        args.push_back(agg);
        args.push_back(el);
        for (auto i : indices) {
            args.push_back(const_((int32_t)i));
        }
        return call(FuncTag::INSERT_ELEMENT, args, ty);
    }
    template<class I>
        requires std::is_integral_v<I>
    [[nodiscard]] Node *gep(const Node *agg, luisa::span<I> indices, const Type *ty) noexcept {
        luisa::vector<const Node *> args;
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
            args.push_back(const_((int32_t)i));
        }
        return call(FuncTag::GET_ELEMENT_PTR, args, ty);
    }
    Node *if_(const Node *cond, const BasicBlock *true_branch, const BasicBlock *false_branch) noexcept;
    Node *generic_loop(const BasicBlock *perpare, const Node *cond, const BasicBlock *body, const BasicBlock *after) noexcept;
    Node *switch_(const Node *value, luisa::span<const SwitchCase> cases, const BasicBlock *default_branch) noexcept;
    BasicBlock *finish() && noexcept {
        LUISA_ASSERT(_current_bb != nullptr, "IrBuilder is not configured to produce a basic block");
        return _current_bb;
    }
    Node *return_(const Node *value) noexcept {
        auto ret = Instruction(ReturnInst(const_cast<Node *>(value)));
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
    Node *rev_autodiff(const BasicBlock *body) noexcept {
        auto rev = Instruction(RevAutodiffInst(body));
        return append(_pool->alloc<Node>(std::move(rev), Type::of<void>()));
    }
    Node *forward_autodiff(const BasicBlock *body) noexcept {
        auto fwd = Instruction(FwdAutodiffInst(body));
        return append(_pool->alloc<Node>(std::move(fwd), Type::of<void>()));
    }
    Node *update(const Node *var, const Node *value) noexcept {
        LUISA_ASSERT(var->is_lvalue(), "bad update");
        auto update = Instruction(UpdateInst(const_cast<Node *>(var), const_cast<Node *>(value)));
        return append(_pool->alloc<Node>(std::move(update), Type::of<void>()));
    }
    [[nodiscard]] Node *local(const Node *init) noexcept {
        auto local = Instruction(LocalInst(const_cast<Node *>(init)));
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

struct Module : luisa::enable_shared_from_this<Module>, luisa::concepts::Noncopyable {
public:
    enum class Kind {
        CALLABLE,
        KERNEL
    };
    luisa::vector<Node *> args;
    BasicBlock *entry = nullptr;
    luisa::shared_ptr<UseDefAnalysis> use_def_analysis;
    luisa::shared_ptr<Pool> pool;
    Module() noexcept : pool{luisa::make_shared<Pool>()} {}
    virtual ~Module() noexcept = default;
    [[nodiscard]] virtual Kind kind() const noexcept = 0;
};

struct Capture {
    Node *node = nullptr;
    Binding binding;
    Capture() noexcept = delete;
    Capture(const Capture &) = delete;
    Capture(Node *node, Binding binding) noexcept : node{node}, binding{std::move(binding)} {}
};

struct CallableModule : Module {
    [[nodiscard]] virtual Kind kind() const noexcept override {
        return Kind::CALLABLE;
    }
};

struct KernelModule : Module {
    luisa::vector<Capture> captures;
    std::array<uint32_t, 3> block_size = {64, 1, 1};
    KernelModule() = default;
    [[nodiscard]] virtual Kind kind() const noexcept override {
        return Kind::KERNEL;
    }
};

class Transform {
    virtual void run(Module &module) noexcept = 0;
};

LC_IR_API void validate(Module &module) noexcept;
LC_IR_API void normalize(Module &module) noexcept;
inline void validate(const Type *ty) noexcept {
    bool bad = false;
    if (!ty) {
    } else if (ty->is_structure()) {
        for (auto e : ty->members()) {
            validate(e);
        }
    } else if (ty->is_array()) {
        validate(ty->element());
    } else if (ty->is_vector()) {
        validate(ty->element());
        if (ty->dimension() > 4) {
            bad = true;
        }
    } else if (ty->is_matrix()) {
        validate(ty->element());
        if (ty->dimension() > 4) {
            bad = true;
        }
    } else if (ty->is_scalar()) {
        // very good
    } else {
        bad = true;
    }
    if (bad) {
        LUISA_ERROR_WITH_LOCATION("type: `{}` is not a valid type for IR!!!", ty->description());
    }
}
LC_IR_API luisa::string dump_human_readable(Module &module) noexcept;
}// namespace luisa::compute::ir_v2