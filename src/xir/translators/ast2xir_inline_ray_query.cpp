#include "ast2xir_inline_ray_query.h"

#include <utility>

#include <luisa/ast/type.h>

namespace luisa::compute::xir::detail {

namespace {

class InlineRayQueryASTAnalyzer {

private:
    InlineRayQueryASTLoopMap _matches;

private:
    enum class InlineRayQueryASTLoopResult {
        ignored,
        accepted,
        rejected,
    };

    [[nodiscard]] static const RefExpr *_match_query_call(
        const Expression *expression, CallOp op) noexcept {
        if (expression == nullptr ||
            expression->tag() != Expression::Tag::CALL) {
            return nullptr;
        }
        auto call = static_cast<const CallExpr *>(expression);
        if (call->op() != op || call->arguments().size() != 1u ||
            call->arguments().front()->tag() != Expression::Tag::REF) {
            return nullptr;
        }
        return static_cast<const RefExpr *>(call->arguments().front());
    }

    [[nodiscard]] static bool _same_query(
        const RefExpr *lhs, const RefExpr *rhs) noexcept {
        return lhs != nullptr && rhs != nullptr &&
               lhs->variable() == rhs->variable() &&
               lhs->type() == rhs->type();
    }

    [[nodiscard]] static bool _is_ray_query_object(
        const RefExpr *query) noexcept {
        if (query == nullptr) { return false; }
        auto type = query->type();
        return type == Type::custom("LC_RayQueryAll") ||
               type == Type::custom("LC_RayQueryAny");
    }

    [[nodiscard]] static bool _expression_contains_call(
        const Expression *expression, CallOp op) noexcept {
        auto found = false;
        if (expression != nullptr) {
            traverse_subexpressions(
                expression,
                [&](const Expression *candidate) noexcept {
                    if (!found &&
                        candidate->tag() == Expression::Tag::CALL) {
                        found = static_cast<const CallExpr *>(candidate)->op() ==
                                op;
                    }
                },
                [](const Expression *) noexcept {});
        }
        return found;
    }

    [[nodiscard]] static bool _statement_contains_call(
        const Statement *statement, CallOp op) noexcept {
        auto found = false;
        if (statement != nullptr) {
            traverse_expressions<true>(
                statement,
                [&](const Expression *expression) noexcept {
                    if (!found &&
                        expression->tag() == Expression::Tag::CALL) {
                        found = static_cast<const CallExpr *>(expression)->op() ==
                                op;
                    }
                },
                [](const Statement *) noexcept {},
                [](const Statement *) noexcept {});
        }
        return found;
    }

    [[nodiscard]] static luisa::vector<const Statement *>
    _semantic_statements(
        const ScopeStmt *scope,
        luisa::vector<const CommentStmt *> *comments = nullptr) noexcept {
        luisa::vector<const Statement *> statements;
        if (scope == nullptr) { return statements; }
        statements.reserve(scope->statements().size());
        for (auto statement : scope->statements()) {
            if (statement->tag() == Statement::Tag::COMMENT) {
                if (comments != nullptr) {
                    comments->emplace_back(
                        static_cast<const CommentStmt *>(statement));
                }
            } else {
                statements.emplace_back(statement);
            }
        }
        return statements;
    }

    [[nodiscard]] static bool _scope_contains_ray_query_proceed(
        const ScopeStmt *scope) noexcept {
        auto found = false;
        if (scope != nullptr) {
            traverse_expressions<true>(
                scope,
                [&](const Expression *expression) noexcept {
                    if (!found &&
                        expression->tag() == Expression::Tag::CALL) {
                        found = static_cast<const CallExpr *>(expression)->op() ==
                                CallOp::RAY_QUERY_PROCEED;
                    }
                },
                [](const Statement *) noexcept {},
                [](const Statement *) noexcept {});
        }
        return found;
    }

    [[nodiscard]] static bool _handler_scope_is_structurable(
        const ScopeStmt *scope, uint32_t loop_depth = 0u,
        uint32_t break_depth = 0u,
        bool check_for_proceed = true) noexcept {
        if (scope == nullptr ||
            (check_for_proceed &&
             _scope_contains_ray_query_proceed(scope))) {
            return false;
        }
        for (auto statement : scope->statements()) {
            switch (statement->tag()) {
                case Statement::Tag::BREAK: {
                    if (break_depth == 0u) { return false; }
                    break;
                }
                case Statement::Tag::CONTINUE: {
                    if (loop_depth == 0u) { return false; }
                    break;
                }
                case Statement::Tag::RETURN:
                case Statement::Tag::RAY_QUERY: return false;
                case Statement::Tag::SCOPE: {
                    if (!_handler_scope_is_structurable(
                            static_cast<const ScopeStmt *>(statement),
                            loop_depth, break_depth, false)) {
                        return false;
                    }
                    break;
                }
                case Statement::Tag::IF: {
                    auto ast_if = static_cast<const IfStmt *>(statement);
                    if (!_handler_scope_is_structurable(
                            ast_if->true_branch(), loop_depth, break_depth,
                            false) ||
                        !_handler_scope_is_structurable(
                            ast_if->false_branch(), loop_depth, break_depth,
                            false)) {
                        return false;
                    }
                    break;
                }
                case Statement::Tag::LOOP: {
                    auto ast_loop = static_cast<const LoopStmt *>(statement);
                    if (!_handler_scope_is_structurable(
                            ast_loop->body(), loop_depth + 1u,
                            break_depth + 1u, false)) {
                        return false;
                    }
                    break;
                }
                case Statement::Tag::SWITCH: {
                    auto ast_switch = static_cast<const SwitchStmt *>(statement);
                    if (!_handler_scope_is_structurable(
                            ast_switch->body(), loop_depth,
                            break_depth + 1u, false)) {
                        return false;
                    }
                    break;
                }
                case Statement::Tag::SWITCH_CASE: {
                    auto ast_case =
                        static_cast<const SwitchCaseStmt *>(statement);
                    if (!_handler_scope_is_structurable(
                            ast_case->body(), loop_depth, break_depth,
                            false)) {
                        return false;
                    }
                    break;
                }
                case Statement::Tag::SWITCH_DEFAULT: {
                    auto ast_default =
                        static_cast<const SwitchDefaultStmt *>(statement);
                    if (!_handler_scope_is_structurable(
                            ast_default->body(), loop_depth, break_depth,
                            false)) {
                        return false;
                    }
                    break;
                }
                case Statement::Tag::FOR: {
                    auto ast_for = static_cast<const ForStmt *>(statement);
                    if (!_handler_scope_is_structurable(
                            ast_for->body(), loop_depth + 1u,
                            break_depth + 1u, false)) {
                        return false;
                    }
                    break;
                }
                case Statement::Tag::AUTO_DIFF: {
                    auto ast_autodiff =
                        static_cast<const AutoDiffStmt *>(statement);
                    if (!_handler_scope_is_structurable(
                            ast_autodiff->body(), loop_depth, break_depth,
                            false)) {
                        return false;
                    }
                    break;
                }
                case Statement::Tag::EXPR:
                case Statement::Tag::ASSIGN:
                case Statement::Tag::COMMENT:
                case Statement::Tag::SUSPEND:
                case Statement::Tag::PRINT:
                case Statement::Tag::DEBUG_BREAK: break;
            }
        }
        return true;
    }

    [[nodiscard]] static InlineRayQueryASTLoopResult
    _match_inline_ray_query_loop(
        const LoopStmt *loop, InlineRayQueryASTLoop &match) noexcept {
        auto condition = loop->while_condition();
        if (condition == nullptr) {
            return InlineRayQueryASTLoopResult::ignored;
        }
        auto condition_statement_count =
            loop->while_condition_statement_count();
        auto statements = loop->body()->statements();
        if (condition_statement_count > statements.size()) {
            return InlineRayQueryASTLoopResult::rejected;
        }
        auto condition_prefix_contains_proceed = false;
        for (auto i = 0u; i < condition_statement_count; ++i) {
            condition_prefix_contains_proceed |=
                _statement_contains_call(
                    statements[i], CallOp::RAY_QUERY_PROCEED);
        }
        auto query = _match_query_call(
            condition, CallOp::RAY_QUERY_PROCEED);
        if (query == nullptr) {
            return _expression_contains_call(
                       condition, CallOp::RAY_QUERY_PROCEED) ||
                           condition_prefix_contains_proceed ?
                       InlineRayQueryASTLoopResult::rejected :
                       InlineRayQueryASTLoopResult::ignored;
        }
        // A structured query loop replaces condition evaluation with the
        // implicit dispatch advance. Any statement materialized while
        // evaluating the condition, even if it does not contain PROCEED,
        // could be observable and therefore makes direct preservation unsafe.
        if (condition_statement_count != 0u ||
            !_is_ray_query_object(query)) {
            return InlineRayQueryASTLoopResult::rejected;
        }

        auto body = _semantic_statements(loop->body());
        if ((body.size() != 2u && body.size() != 3u) ||
            body.back()->tag() != Statement::Tag::IF) {
            return InlineRayQueryASTLoopResult::rejected;
        }

        const IfStmt *guard = nullptr;
        const RefExpr *guard_query = nullptr;
        if (body.size() == 2u &&
            body.front()->tag() == Statement::Tag::IF) {
            guard = static_cast<const IfStmt *>(body.front());
            auto guard_condition = guard->condition();
            if (guard_condition->tag() == Expression::Tag::UNARY) {
                auto guard_not =
                    static_cast<const UnaryExpr *>(guard_condition);
                if (guard_not->op() == UnaryOp::NOT) {
                    guard_query = _match_query_call(
                        guard_not->operand(),
                        CallOp::RAY_QUERY_PROCEED);
                }
            }
        } else if (body.size() == 3u &&
                   body[0]->tag() == Statement::Tag::ASSIGN &&
                   body[1]->tag() == Statement::Tag::IF) {
            auto guard_assignment =
                static_cast<const AssignStmt *>(body[0]);
            guard = static_cast<const IfStmt *>(body[1]);
            auto assigned = guard_assignment->lhs();
            auto assigned_value = guard_assignment->rhs();
            if (assigned->tag() == Expression::Tag::REF &&
                assigned_value->tag() == Expression::Tag::UNARY &&
                guard->condition()->tag() == Expression::Tag::REF) {
                auto guard_not =
                    static_cast<const UnaryExpr *>(assigned_value);
                auto assigned_ref =
                    static_cast<const RefExpr *>(assigned);
                auto condition_ref =
                    static_cast<const RefExpr *>(guard->condition());
                if (guard_not->op() == UnaryOp::NOT &&
                    _same_query(assigned_ref, condition_ref)) {
                    guard_query = _match_query_call(
                        guard_not->operand(),
                        CallOp::RAY_QUERY_PROCEED);
                }
            }
        }
        if (guard == nullptr) {
            return InlineRayQueryASTLoopResult::rejected;
        }
        auto guard_true = _semantic_statements(guard->true_branch());
        auto guard_false = _semantic_statements(guard->false_branch());
        if (!_same_query(query, guard_query) ||
            guard_true.size() != 1u ||
            guard_true.front()->tag() != Statement::Tag::BREAK ||
            !guard_false.empty()) {
            return InlineRayQueryASTLoopResult::rejected;
        }

        auto dispatch = static_cast<const IfStmt *>(body.back());
        auto surface_query = _match_query_call(
            dispatch->condition(),
            CallOp::RAY_QUERY_IS_TRIANGLE_CANDIDATE);
        auto procedural_query = _match_query_call(
            dispatch->condition(),
            CallOp::RAY_QUERY_IS_PROCEDURAL_CANDIDATE);
        if (surface_query == nullptr && procedural_query == nullptr) {
            return InlineRayQueryASTLoopResult::rejected;
        }
        auto dispatch_query = surface_query != nullptr ?
                                  surface_query :
                                  procedural_query;
        if (!_same_query(query, dispatch_query)) {
            return InlineRayQueryASTLoopResult::rejected;
        }
        auto surface_handler = surface_query != nullptr ?
                                   dispatch->true_branch() :
                                   dispatch->false_branch();
        auto procedural_handler = procedural_query != nullptr ?
                                      dispatch->true_branch() :
                                      dispatch->false_branch();
        if (!_handler_scope_is_structurable(surface_handler) ||
            !_handler_scope_is_structurable(procedural_handler)) {
            return InlineRayQueryASTLoopResult::rejected;
        }

        luisa::vector<const CommentStmt *> dispatch_comments;
        auto after_guard = false;
        for (auto statement : loop->body()->statements()) {
            if (statement == guard) {
                after_guard = true;
            } else if (statement == dispatch) {
                break;
            } else if (after_guard &&
                       statement->tag() == Statement::Tag::COMMENT) {
                dispatch_comments.emplace_back(
                    static_cast<const CommentStmt *>(statement));
            }
        }

        match.query = query;
        match.surface_handler = surface_handler;
        match.procedural_handler = procedural_handler;
        match.dispatch_comments = std::move(dispatch_comments);
        return InlineRayQueryASTLoopResult::accepted;
    }

    [[nodiscard]] bool _collect_inline_ray_query_loops(
        const ScopeStmt *scope, bool inside_ray_query = false) noexcept {
        for (auto statement : scope->statements()) {
            switch (statement->tag()) {
                case Statement::Tag::SCOPE: {
                    if (!_collect_inline_ray_query_loops(
                            static_cast<const ScopeStmt *>(statement),
                            inside_ray_query)) {
                        return false;
                    }
                    break;
                }
                case Statement::Tag::IF: {
                    auto ast_if = static_cast<const IfStmt *>(statement);
                    if (!_collect_inline_ray_query_loops(
                            ast_if->true_branch(), inside_ray_query) ||
                        !_collect_inline_ray_query_loops(
                            ast_if->false_branch(), inside_ray_query)) {
                        return false;
                    }
                    break;
                }
                case Statement::Tag::LOOP: {
                    auto ast_loop = static_cast<const LoopStmt *>(statement);
                    InlineRayQueryASTLoop match;
                    switch (_match_inline_ray_query_loop(ast_loop, match)) {
                        case InlineRayQueryASTLoopResult::ignored: {
                            if (!_collect_inline_ray_query_loops(
                                    ast_loop->body(), inside_ray_query)) {
                                return false;
                            }
                            break;
                        }
                        case InlineRayQueryASTLoopResult::accepted: {
                            if (inside_ray_query) { return false; }
                            _matches.emplace(
                                ast_loop, std::move(match));
                            break;
                        }
                        case InlineRayQueryASTLoopResult::rejected:
                            return false;
                    }
                    break;
                }
                case Statement::Tag::SWITCH: {
                    if (!_collect_inline_ray_query_loops(
                            static_cast<const SwitchStmt *>(statement)->body(),
                            inside_ray_query)) {
                        return false;
                    }
                    break;
                }
                case Statement::Tag::SWITCH_CASE: {
                    if (!_collect_inline_ray_query_loops(
                            static_cast<const SwitchCaseStmt *>(statement)->body(),
                            inside_ray_query)) {
                        return false;
                    }
                    break;
                }
                case Statement::Tag::SWITCH_DEFAULT: {
                    if (!_collect_inline_ray_query_loops(
                            static_cast<const SwitchDefaultStmt *>(statement)->body(),
                            inside_ray_query)) {
                        return false;
                    }
                    break;
                }
                case Statement::Tag::FOR: {
                    if (!_collect_inline_ray_query_loops(
                            static_cast<const ForStmt *>(statement)->body(),
                            inside_ray_query)) {
                        return false;
                    }
                    break;
                }
                case Statement::Tag::RAY_QUERY: {
                    auto ast_query =
                        static_cast<const RayQueryStmt *>(statement);
                    if (!_collect_inline_ray_query_loops(
                            ast_query->on_triangle_candidate(), true) ||
                        !_collect_inline_ray_query_loops(
                            ast_query->on_procedural_candidate(), true)) {
                        return false;
                    }
                    break;
                }
                case Statement::Tag::AUTO_DIFF: {
                    if (!_collect_inline_ray_query_loops(
                            static_cast<const AutoDiffStmt *>(statement)->body(),
                            inside_ray_query)) {
                        return false;
                    }
                    break;
                }
                case Statement::Tag::BREAK:
                case Statement::Tag::CONTINUE:
                case Statement::Tag::RETURN:
                case Statement::Tag::EXPR:
                case Statement::Tag::ASSIGN:
                case Statement::Tag::COMMENT:
                case Statement::Tag::SUSPEND:
                case Statement::Tag::PRINT:
                case Statement::Tag::DEBUG_BREAK: break;
            }
        }
        return true;
    }


public:
    [[nodiscard]] bool analyze(
        const ScopeStmt *function_body,
        InlineRayQueryASTLoopMap &matches) noexcept {
        matches.clear();
        if (function_body == nullptr ||
            !_collect_inline_ray_query_loops(function_body)) {
            _matches.clear();
            return false;
        }
        matches = std::move(_matches);
        return true;
    }
};

}// namespace

bool analyze_inline_ray_query_ast(
    const ScopeStmt *function_body,
    InlineRayQueryASTLoopMap &matches) noexcept {
    InlineRayQueryASTAnalyzer analyzer;
    return analyzer.analyze(function_body, matches);
}

}// namespace luisa::compute::xir::detail
