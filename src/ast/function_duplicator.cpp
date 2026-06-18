//
// Created by Mike on 10/12/2023.
//

#include <luisa/ast/function_builder.h>
#include <luisa/core/logging.h>

namespace luisa::compute::detail {

class FunctionDuplicator {

private:
    struct DupCtx {
        const FunctionBuilder &original;
        luisa::unordered_map<
            uint32_t /* original uid */,
            const RefExpr * /* copy */>
            var_map{};
        luisa::unordered_map<
            const Expression * /* original */,
            const Expression * /* copy */>
            expr_map{};
    };

private:
    luisa::unordered_map<
        const FunctionBuilder * /* original */,
        luisa::shared_ptr<const FunctionBuilder> /* copy */>
        _duplicated;
    luisa::vector<DupCtx *> _contexts;

    // Keyed by (source FunctionBuilder*, Variable uid) so that all RefExpr
    // nodes referring to the same leaked source variable share a single
    // kernel-local bridge. Pointer identity of RefExpr is NOT a robust
    // alias key because FunctionBuilder::_ref() may create multiple
    // RefExpr nodes for the same Variable.
    struct LeakedRefKey {
        const FunctionBuilder *builder;
        uint32_t uid;
        [[nodiscard]] bool operator==(const LeakedRefKey &rhs) const noexcept {
            return builder == rhs.builder && uid == rhs.uid;
        }
    };
    struct LeakedRefKeyHash {
        [[nodiscard]] uint64_t operator()(LeakedRefKey k) const noexcept {
            return luisa::hash_combine({luisa::hash_value(k.builder),
                                        luisa::hash_value(k.uid)});
        }
    };
    struct LeakedRefInfo {
        LeakedRefKey key;
        const Type *type;
    };
    luisa::unordered_map<
        LeakedRefKey,
        const RefExpr * /* hoisted kernel-local bridge */,
        LeakedRefKeyHash>
        _hoisted_refs;

private:
    [[nodiscard]] static luisa::optional<LeakedRefInfo>
    _try_make_leaked_ref(const Expression *e) noexcept {
        if (e == nullptr || e->tag() != Expression::Tag::REF) { return luisa::nullopt; }
        auto ref = static_cast<const RefExpr *>(e);
        auto v = ref->variable();
        if (v.tag() == Variable::Tag::LOCAL ||
            v.tag() == Variable::Tag::REFERENCE) {
            return LeakedRefInfo{
                .key = {.builder = e->builder(), .uid = v.uid()},
                .type = e->type()};
        }
        return luisa::nullopt;
    }

    static void _collect_leaked_refs(
        luisa::unordered_map<LeakedRefKey, LeakedRefInfo, LeakedRefKeyHash> &collected,
        luisa::unordered_set<const FunctionBuilder *> &visited,
        const FunctionBuilder &kernel,
        const FunctionBuilder &f) noexcept {
        if (!visited.emplace(&f).second) { return; }
        traverse_expressions<true>(
            f.body(),
            [&](const Expression *e) noexcept {
                if (e->builder() != &f) {
                    // Only hoist foreign REF leaves whose source builder is
                    // ANOTHER callable. Refs to the kernel's own variables
                    // are already handled by var_map + _internalize lvalue path.
                    if (e->tag() == Expression::Tag::REF &&
                        e->builder() != &kernel) {
                        if (auto leaked = _try_make_leaked_ref(e)) {
                            auto [iter, inserted] = collected.emplace(leaked->key, *leaked);
                            if (!inserted) {
                                LUISA_ASSERT(*iter->second.type == *leaked->type,
                                             "Inconsistent leaked ref type.");
                            }
                        }
                    }
                }
                if (e->tag() == Expression::Tag::CALL) {
                    auto call = static_cast<const CallExpr *>(e);
                    if (call->is_custom()) {
                        _collect_leaked_refs(collected, visited, kernel, *call->custom().builder());
                    }
                }
            },
            [](auto) noexcept {},
            [](auto) noexcept {});
    }

    void _hoist_leaked_refs(
        const luisa::unordered_map<LeakedRefKey, LeakedRefInfo, LeakedRefKeyHash> &leaked) noexcept {
        auto fb = FunctionBuilder::current();
        for (auto &&[key, info] : leaked) {
            if (_hoisted_refs.find(key) == _hoisted_refs.end()) {
                auto bridge = fb->local(info.type);
                _hoisted_refs.emplace(key, bridge);
            }
        }
    }

private:
    void _dup_function(const FunctionBuilder &f) noexcept {
        auto fb = FunctionBuilder::current();
        fb->mark_required_curve_basis_set(f.required_curve_bases());
        fb->set_name(f.name());
        if (f.tag() == Function::Tag::KERNEL) {
            fb->set_block_size(f.block_size());
            luisa::unordered_map<LeakedRefKey, LeakedRefInfo, LeakedRefKeyHash> collected;
            luisa::unordered_set<const FunctionBuilder *> visited;
            _collect_leaked_refs(collected, visited, f, f);
            _hoist_leaked_refs(collected);
        }
        auto dup_arg = [this, fb](Variable original) noexcept {
            auto dup = [&] {
                switch (original.tag()) {
                    case Variable::Tag::REFERENCE: return fb->reference(original.type());
                    case Variable::Tag::BUFFER: return fb->buffer(original.type());
                    case Variable::Tag::TEXTURE: return fb->texture(original.type());
                    case Variable::Tag::BINDLESS_ARRAY: return fb->bindless_array();
                    case Variable::Tag::ACCEL: return fb->accel();
                    default: return fb->argument(original.type());
                }
            }();
            _contexts.back()->var_map.emplace(original.uid(), dup);
        };
        if (f.tag() == Function::Tag::CALLABLE) {// captures are already lowered to arguments
            for (auto &arg : f.arguments()) { dup_arg(arg); }
        } else {
            for (auto i = 0u; i < f.bound_arguments().size(); i++) {
                auto &&a = f.arguments()[i];
                auto &&b = f.bound_arguments()[i];
                auto copy = luisa::visit(
                    [&]<typename B>(B &&bb) noexcept -> const RefExpr * {
                        using T = std::remove_cvref_t<B>;
                        if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                            return fb->buffer_binding(a.type(), bb.handle, bb.offset, bb.size);
                        } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                            return fb->texture_binding(a.type(), bb.handle, bb.level);
                        } else if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                            return fb->bindless_array_binding(bb.handle);
                        } else if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                            return fb->accel_binding(bb.handle);
                        } else {
                            LUISA_ERROR_WITH_LOCATION("Unbound captured argument.");
                        }
                    },
                    b);
                _contexts.back()->var_map.emplace(a.uid(), copy);
            }
            for (auto &arg : f.unbound_arguments()) { dup_arg(arg); }
            for (auto &b : f.builtin_variables()) {
                auto copy = fb->_builtin(b.type(), b.tag());
                _contexts.back()->var_map.emplace(b.uid(), copy);
            }
        }
        for (auto &shared : f.shared_variables()) {
            auto s = fb->shared(shared.type());
            _contexts.back()->var_map.emplace(shared.uid(), s);
        }
        for (auto &local : f.local_variables()) {
            auto l = fb->local(local.type());
            _contexts.back()->var_map.emplace(local.uid(), l);
        }
        _dup_scope(f.body(), fb->body());
    }

    [[nodiscard]] const Expression *_dup_expr(const Expression *original) noexcept {
        if (original == nullptr) { return nullptr; }
        if (auto iter = _contexts.back()->expr_map.find(original);
            iter != _contexts.back()->expr_map.end()) {
            return iter->second;
        }
        auto fb = FunctionBuilder::current();
        auto cache_key = original;
        if (auto leaked = _try_make_leaked_ref(original)) {
            if (auto iter = _hoisted_refs.find(leaked->key);
                iter != _hoisted_refs.end()) {
                auto copy = fb->_internalize(iter->second);
                _contexts.back()->expr_map.emplace(cache_key, copy);
                return copy;
            }
        }
        auto copy = static_cast<const Expression *>(nullptr);
        switch (original->tag()) {
            case Expression::Tag::UNARY: {
                auto e = static_cast<const UnaryExpr *>(original);
                auto operand = _dup_expr(e->operand());
                copy = fb->unary(e->type(), e->op(), operand);
                break;
            }
            case Expression::Tag::BINARY: {
                auto e = static_cast<const BinaryExpr *>(original);
                auto lhs = _dup_expr(e->lhs());
                auto rhs = _dup_expr(e->rhs());
                copy = fb->binary(e->type(), e->op(), lhs, rhs);
                break;
            }
            case Expression::Tag::MEMBER: {
                auto e = static_cast<const MemberExpr *>(original);
                auto self = _dup_expr(e->self());
                copy = e->is_swizzle() ?
                           fb->swizzle(e->type(), self,
                                       e->swizzle_size(),
                                       e->swizzle_code()) :
                           fb->member(e->type(), self,
                                      e->member_index());
                break;
            }
            case Expression::Tag::ACCESS: {
                auto e = static_cast<const AccessExpr *>(original);
                auto range = _dup_expr(e->range());
                auto index = _dup_expr(e->index());
                copy = fb->access(e->type(), range, index);
                break;
            }
            case Expression::Tag::LITERAL: {
                auto e = static_cast<const LiteralExpr *>(original);
                copy = fb->literal(e->type(), e->value());
                break;
            }
            case Expression::Tag::REF: {
                if (original->builder() == fb) {
                    copy = original;
                } else {
                    auto e = static_cast<const RefExpr *>(original);
                    auto iter = _contexts.back()->var_map.find(e->variable().uid());
                    LUISA_ASSERT(iter != _contexts.back()->var_map.end(),
                                 "Variable not found in context.");
                    copy = iter->second;
                }
                break;
            }
            case Expression::Tag::CONSTANT: {
                auto e = static_cast<const ConstantExpr *>(original);
                copy = fb->constant(e->data());
                break;
            }
            case Expression::Tag::CALL: {
                auto e = static_cast<const CallExpr *>(original);
                luisa::vector<const Expression *> args;
                args.reserve(e->arguments().size());
                for (auto arg : e->arguments()) { args.emplace_back(_dup_expr(arg)); }
                if (e->is_builtin()) {
                    copy = fb->call(e->type(), e->op(), args, e->curve_basis_set());
                } else if (e->is_custom()) {
                    auto callee = _duplicate(*e->custom().builder());
                    copy = fb->call(e->type(), callee->function(), args);
                } else if (e->is_external()) {
                    auto &o = _contexts.back()->original;
                    auto iter = std::find_if(
                        o.external_callables().begin(),
                        o.external_callables().end(),
                        [ext = e->external()](auto &&f) noexcept { return *f == *ext; });
                    LUISA_ASSERT(iter != o.external_callables().end(),
                                 "External function not found in context.");
                    copy = fb->call(e->type(), *iter, args);
                } else {
                    LUISA_ASSERT(e->is_external(), "Unknown call type.");
                }
                break;
            }
            case Expression::Tag::CAST: {
                auto e = static_cast<const CastExpr *>(original);
                auto src = _dup_expr(e->expression());
                copy = fb->cast(e->type(), e->op(), src);
                break;
            }
            case Expression::Tag::TYPE_ID: {
                auto e = static_cast<const TypeIDExpr *>(original);
                copy = fb->type_id(e->type());
                break;
            }
            case Expression::Tag::STRING_ID: {
                auto e = static_cast<const StringIDExpr *>(original);
                copy = fb->string_id(luisa::string{e->data()});
                break;
            }
            case Expression::Tag::CPUCUSTOM: LUISA_NOT_IMPLEMENTED();
            case Expression::Tag::GPUCUSTOM: LUISA_NOT_IMPLEMENTED();
            case Expression::Tag::FUNC_REF: {
                auto e = static_cast<const FuncRefExpr *>(original);
                auto f = _duplicate(*e->func());
                copy = fb->func_ref(f->function());
                break;
            }
        }
        _contexts.back()->expr_map.emplace(original, copy);
        return copy;
    }

    void _dup_scope(const ScopeStmt *original, ScopeStmt *copy) noexcept {
        FunctionBuilder::current()->with(copy, [original, this] {
            for (auto s : original->statements()) {
                _dup_stmt(s);
                if (auto tag = s->tag();
                    tag == Statement::Tag::BREAK ||
                    tag == Statement::Tag::CONTINUE ||
                    tag == Statement::Tag::RETURN) {
                    break;
                }
            }
        });
    }

    void _dup_stmt(const Statement *stmt) noexcept {
        auto fb = FunctionBuilder::current();
        switch (stmt->tag()) {
            case Statement::Tag::BREAK: {
                fb->break_();
                break;
            }
            case Statement::Tag::CONTINUE: {
                fb->continue_();
                break;
            }
            case Statement::Tag::RETURN: {
                auto s = static_cast<const ReturnStmt *>(stmt);
                auto e = _dup_expr(s->expression());
                fb->return_(e);
                break;
            }
            case Statement::Tag::SCOPE: {
                LUISA_ERROR_WITH_LOCATION(
                    "ScopeStmt should have been "
                    "handled in parent statements.");
            }
            case Statement::Tag::IF: {
                auto s = static_cast<const IfStmt *>(stmt);
                auto cond = _dup_expr(s->condition());
                auto if_ = fb->if_(cond);
                _dup_scope(s->true_branch(), if_->true_branch());
                _dup_scope(s->false_branch(), if_->false_branch());
                break;
            }
            case Statement::Tag::LOOP: {
                auto s = static_cast<const LoopStmt *>(stmt);
                auto loop = fb->loop_();
                _dup_scope(s->body(), loop->body());
                break;
            }
            case Statement::Tag::EXPR: {
                auto s = static_cast<const ExprStmt *>(stmt);
                auto e = _dup_expr(s->expression());
                // _dup_expr uses the rvalue call() overloads which only
                // construct the expression; they never insert it into the
                // body. We must explicitly call _void_expr() here for ALL
                // expression statements, including void custom-callable
                // calls. Skipping this silently drops the statement.
                fb->_void_expr(e);
                break;
            }
            case Statement::Tag::SWITCH: {
                auto s = static_cast<const SwitchStmt *>(stmt);
                auto e = _dup_expr(s->expression());
                auto sw = fb->switch_(e);
                _dup_scope(s->body(), sw->body());
                break;
            }
            case Statement::Tag::SWITCH_CASE: {
                auto s = static_cast<const SwitchCaseStmt *>(stmt);
                auto e = _dup_expr(s->expression());
                auto sw = fb->case_(e);
                _dup_scope(s->body(), sw->body());
                break;
            }
            case Statement::Tag::SWITCH_DEFAULT: {
                auto s = static_cast<const SwitchDefaultStmt *>(stmt);
                auto sw = fb->default_();
                _dup_scope(s->body(), sw->body());
                break;
            }
            case Statement::Tag::ASSIGN: {
                auto s = static_cast<const AssignStmt *>(stmt);
                auto lhs = _dup_expr(s->lhs());
                auto rhs = _dup_expr(s->rhs());
                fb->assign(lhs, rhs);
                break;
            }
            case Statement::Tag::FOR: {
                auto s = static_cast<const ForStmt *>(stmt);
                auto var = _dup_expr(s->variable());
                auto cond = _dup_expr(s->condition());
                auto step = _dup_expr(s->step());
                auto for_ = fb->for_(var, cond, step);
                _dup_scope(s->body(), for_->body());
                break;
            }
            case Statement::Tag::COMMENT: {
                auto s = static_cast<const CommentStmt *>(stmt);
                fb->comment_(luisa::string{s->comment()});
                break;
            }
            case Statement::Tag::SUSPEND: {
                auto s = static_cast<const SuspendStmt *>(stmt);
                fb->suspend_(s->token(), luisa::string{s->name()});
                break;
            }
            case Statement::Tag::RAY_QUERY: {
                auto s = static_cast<const RayQueryStmt *>(stmt);
                auto q = _dup_expr(s->query());
                LUISA_ASSERT(q->tag() == Expression::Tag::REF,
                             "RayQueryExpr should be a reference.");
                auto rq = fb->ray_query_(static_cast<const RefExpr *>(q));
                _dup_scope(s->on_triangle_candidate(),
                           rq->on_triangle_candidate());
                _dup_scope(s->on_procedural_candidate(),
                           rq->on_procedural_candidate());
                break;
            }
            case Statement::Tag::AUTO_DIFF: {
                auto s = static_cast<const AutoDiffStmt *>(stmt);
                auto ad = fb->autodiff_();
                _dup_scope(s->body(), ad->body());
                break;
            }
            case Statement::Tag::PRINT: {
                auto s = static_cast<const PrintStmt *>(stmt);
                luisa::vector<const Expression *> args;
                args.reserve(s->arguments().size());
                for (auto arg : s->arguments()) { args.emplace_back(_dup_expr(arg)); }
                fb->print_(luisa::string{s->format()}, args);
                break;
            }
            case Statement::Tag::DEBUG_BREAK: {
                auto s = static_cast<const DebugBreakStmt *>(stmt);
                luisa::vector<const Expression *> watches;
                watches.reserve(s->watches().size());
                for (auto watch : s->watches()) {
                    watches.emplace_back(_dup_expr(watch));
                }
                fb->debug_break_(s->wrapper(), watches);
                break;
            }
        }
    }

private:
    [[nodiscard]] luisa::shared_ptr<const FunctionBuilder>
    _duplicate(const FunctionBuilder &f) noexcept {
        if (auto iter = _duplicated.find(&f);
            iter != _duplicated.end()) {
            return iter->second;
        }
        auto dup = FunctionBuilder::_define(f.tag(), [&f, this] {
            DupCtx ctx{.original = f, .var_map = {}, .expr_map = {}};
            _contexts.emplace_back(&ctx);
            _dup_function(f);
            LUISA_ASSERT(!_contexts.empty() && _contexts.back() == &ctx,
                         "Corrupted context stack.");
            _contexts.pop_back();
        });
        auto not_predefined = _duplicated.emplace(&f, dup).second;
        LUISA_ASSERT(not_predefined, "FunctionBuilder::duplicate() called recursively.");
        return dup;
    }

private:
    static void _deduplicate_custom_callables_impl(
        luisa::unordered_map<uint64_t, luisa::shared_ptr<const FunctionBuilder>> &unique,
        const FunctionBuilder *const_builder) noexcept {
        auto builder = const_cast<FunctionBuilder *>(const_builder);
        luisa::unordered_set<const FunctionBuilder *> used;
        traverse_expressions<true>(
            builder->body(),
            [&unique, &used](const Expression *expr) noexcept {
                if (expr->tag() == Expression::Tag::CALL) {
                    auto call = static_cast<const CallExpr *>(expr);
                    if (call->is_custom()) {
                        auto custom = call->custom();
                        auto [iter, is_new] = unique.try_emplace(
                            custom.hash(), custom.shared_builder());
                        auto f = iter->second.get();
                        used.emplace(f);
                        if (is_new) {
                            _deduplicate_custom_callables_impl(unique, f);
                        } else {
                            call->_unsafe_set_custom(f);
                        }
                    }
                }
            },
            [](auto) noexcept {},
            [](auto) noexcept {});
        builder->_used_custom_callables.clear();
        builder->_used_custom_callables.reserve(used.size());
        for (auto f : used) { builder->_used_custom_callables.emplace(f->shared_from_this()); }
    }

public:
    static void deduplicate_custom_callables(const FunctionBuilder *const_builder) noexcept {
        luisa::unordered_map<uint64_t, luisa::shared_ptr<const FunctionBuilder>> unique;
        _deduplicate_custom_callables_impl(unique, const_builder);
    }

public:
    [[nodiscard]] static luisa::shared_ptr<const FunctionBuilder>
    duplicate(const FunctionBuilder &f) noexcept {
        FunctionDuplicator d;
        return d._duplicate(f);
    }
};

luisa::shared_ptr<const FunctionBuilder> FunctionBuilder::duplicate() const noexcept {
    return FunctionDuplicator::duplicate(*this);
}

luisa::shared_ptr<const FunctionBuilder> FunctionBuilder::_duplicate_if_necessary() const noexcept {
    auto check = [](auto &&check, auto f) noexcept -> bool {
        auto necessary = false;
        traverse_expressions<true>(
            f->body(),
            [&](const Expression *e) noexcept {
                necessary |= e->builder() != f;
                if (e->tag() == Expression::Tag::CALL) {
                    auto call = static_cast<const CallExpr *>(e);
                    if (call->is_custom()) {
                        necessary |= check(check, call->custom().builder());
                    }
                }
            },
            [](auto) noexcept {},
            [](auto) noexcept {});
        return necessary;
    };
    if (check(check, this)) {
        auto f = duplicate();
        FunctionDuplicator::deduplicate_custom_callables(f.get());
        return f;
    }
    FunctionDuplicator::deduplicate_custom_callables(this);
    return shared_from_this();
}

}// namespace luisa::compute::detail
