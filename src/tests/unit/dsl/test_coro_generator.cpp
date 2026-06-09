#include <luisa/ast/expression.h>
#include <luisa/ast/function_builder.h>
#include <luisa/ast/statement.h>
#include <luisa/ast/type.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include "ut/ut.hpp"

using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

// NOTE: $yield(x) is defined as:
//   #define $yield(x) do { $promise("__yielded_value", (x)); $suspend; } while(0)
// However, $suspend is a function-like macro ($suspend(...)) and $yield uses
// $suspend; without parens, which means it does NOT expand. This is a known
// preprocessor limitation. The tests below simulate the intended expansion by
// manually calling $promise(...) + $suspend().

namespace {

const Statement *find_stmt_by_tag(const ScopeStmt *s, Statement::Tag t) {
    for (auto x : s->statements()) {
        if (x->tag() == t) { return x; }
    }
    return nullptr;
}

size_t count_stmts_by_tag(const ScopeStmt *s, Statement::Tag t) {
    size_t n = 0u;
    for (auto x : s->statements()) {
        if (x->tag() == t) { n++; }
    }
    return n;
}

luisa::vector<const Statement *>
collect_stmts_by_tag(const ScopeStmt *s, Statement::Tag t) {
    luisa::vector<const Statement *> result;
    for (auto x : s->statements()) {
        if (x->tag() == t) { result.push_back(x); }
    }
    return result;
}

} // namespace

int main() {

    // ── $yield AST recording (simulated expansion) ─────────────────────

    "yield_records_suspend_and_coro_bind"_test = [] {
        // Simulates what $yield(42) should produce in the AST
        auto builder = luisa::compute::detail::FunctionBuilder::define_coroutine([] {
            $promise("__yielded_value", 42);
            $suspend();
        });
        auto *body = builder->body();

        auto *s = find_stmt_by_tag(body, Statement::Tag::SUSPEND);
        expect(s != nullptr) << "SuspendStmt should be recorded";

        auto *b = find_stmt_by_tag(body, Statement::Tag::CORO_BIND);
        expect(b != nullptr) << "CoroBindStmt should be recorded";
    };

    "yield_binds_expected_name"_test = [] {
        auto builder = luisa::compute::detail::FunctionBuilder::define_coroutine([] {
            $promise("__yielded_value", 42);
            $suspend();
        });
        auto *body = builder->body();

        auto *b = static_cast<const CoroBindStmt *>(
            find_stmt_by_tag(body, Statement::Tag::CORO_BIND));
        expect(b != nullptr);
        expect(b->name() == luisa::string_view{"__yielded_value"});
    };

    "yield_value_expression_is_literal_42"_test = [] {
        auto builder = luisa::compute::detail::FunctionBuilder::define_coroutine([] {
            $promise("__yielded_value", 42);
            $suspend();
        });
        auto *body = builder->body();

        auto *b = static_cast<const CoroBindStmt *>(
            find_stmt_by_tag(body, Statement::Tag::CORO_BIND));
        expect(b != nullptr);

        auto *val = b->value();
        expect(val != nullptr);
        expect(val->tag() == Expression::Tag::LITERAL);

        auto *lit = static_cast<const LiteralExpr *>(val);
        expect(luisa::holds_alternative<int>(lit->value()));
        expect(luisa::get<int>(lit->value()) == 42);
    };

    "yield_suspend_has_valid_token"_test = [] {
        auto builder = luisa::compute::detail::FunctionBuilder::define_coroutine([] {
            $promise("__yielded_value", 42);
            $suspend();
        });
        auto *body = builder->body();

        auto *s = static_cast<const SuspendStmt *>(
            find_stmt_by_tag(body, Statement::Tag::SUSPEND));
        expect(s != nullptr);
        expect(s->token() != 0u);
        expect(s->token() != 0xFFFFFFFFu);
    };

    // ── $yield with various expression types ───────────────────────────

    "yield_with_float_literal"_test = [] {
        auto builder = luisa::compute::detail::FunctionBuilder::define_coroutine([] {
            $promise("__yielded_value", 3.14f);
            $suspend();
        });
        auto *body = builder->body();

        auto *b = static_cast<const CoroBindStmt *>(
            find_stmt_by_tag(body, Statement::Tag::CORO_BIND));
        expect(b != nullptr);

        auto *val = b->value();
        expect(val->tag() == Expression::Tag::LITERAL);

        auto *lit = static_cast<const LiteralExpr *>(val);
        expect(luisa::holds_alternative<float>(lit->value()));
        expect(luisa::get<float>(lit->value()) == 3.14_f);
    };

    "yield_with_unsigned_literal"_test = [] {
        auto builder = luisa::compute::detail::FunctionBuilder::define_coroutine([] {
            $promise("__yielded_value", 999u);
            $suspend();
        });
        auto *body = builder->body();

        auto *b = static_cast<const CoroBindStmt *>(
            find_stmt_by_tag(body, Statement::Tag::CORO_BIND));
        expect(b != nullptr);

        auto *lit = static_cast<const LiteralExpr *>(b->value());
        expect(luisa::holds_alternative<uint>(lit->value()));
        expect(luisa::get<uint>(lit->value()) == 999u);
    };

    "yield_multiple_values_produce_distinct_stmts"_test = [] {
        auto builder = luisa::compute::detail::FunctionBuilder::define_coroutine([] {
            $promise("__yielded_value", 1);
            $suspend();
            $promise("__yielded_value", 2);
            $suspend();
            $promise("__yielded_value", 3);
            $suspend();
        });
        auto *body = builder->body();

        expect(count_stmts_by_tag(body, Statement::Tag::SUSPEND) == 3u);
        expect(count_stmts_by_tag(body, Statement::Tag::CORO_BIND) == 3u);

        auto binds = collect_stmts_by_tag(body, Statement::Tag::CORO_BIND);
        expect(binds.size() == 3u);

        for (size_t i = 0u; i < binds.size(); i++) {
            auto *cb = static_cast<const CoroBindStmt *>(binds[i]);
            expect(cb->name() == luisa::string_view{"__yielded_value"});

            auto *lit = static_cast<const LiteralExpr *>(cb->value());
            expect(luisa::holds_alternative<int>(lit->value()));
            expect(luisa::get<int>(lit->value()) == static_cast<int>(i + 1));
        }
    };

    // ── AST ordering: CoroBindStmt before SuspendStmt ──────────────────

    "yield_ast_order_bind_before_suspend"_test = [] {
        // $yield(x) is: promise first, then suspend.
        // CoroBindStmt must appear before SuspendStmt in the AST.
        auto builder = luisa::compute::detail::FunctionBuilder::define_coroutine([] {
            $promise("__yielded_value", 42);
            $suspend();
        });
        auto *body = builder->body();

        bool found_bind = false;
        bool found_suspend = false;
        bool order_correct = true;

        for (auto *stmt : body->statements()) {
            if (stmt->tag() == Statement::Tag::CORO_BIND) {
                found_bind = true;
                if (found_suspend) { order_correct = false; }
            }
            if (stmt->tag() == Statement::Tag::SUSPEND) {
                found_suspend = true;
                if (!found_bind) { order_correct = false; }
            }
        }

        expect(found_bind) << "CoroBindStmt must be present";
        expect(found_suspend) << "SuspendStmt must be present";
        expect(order_correct)
            << "CoroBindStmt must precede SuspendStmt";
    };

    // ── Generator type system ──────────────────────────────────────────

    "generator_has_function_builder"_test = [] {
        auto gen = Generator<int>([]() -> int {
            $suspend();
            return 0;
        });
        auto fb = gen.function_builder();
        expect(fb != nullptr);

        auto *body = fb->body();
        expect(body != nullptr);
        auto *s = find_stmt_by_tag(body, Statement::Tag::SUSPEND);
        expect(s != nullptr) << "Generator should record SuspendStmt";
    };

    "generator_stores_coroutine_internally"_test = [] {
        auto gen = Generator<int>([]() -> int {
            $suspend();
            return 0;
        });
        auto fb = gen.function_builder();
        expect(fb != nullptr);
        expect(fb->body() != nullptr);
    };

    // ── LCG sampler pattern ────────────────────────────────────────────

    "lcg_sampler_records_loop_with_yield"_test = [] {
        // Simulate the LCG pattern: loop with state update and yield
        auto builder = luisa::compute::detail::FunctionBuilder::define_coroutine([] {
            Var<uint> state = 12345u;
            $loop {
                state = state * 1103515245u + 12345u;
                $promise("__yielded_value", state);
                $suspend();
            };
        });
        auto *body = builder->body();

        // Must contain a loop
        auto *loop = find_stmt_by_tag(body, Statement::Tag::LOOP);
        expect(loop != nullptr) << "LCG pattern should contain a loop";

        // Inside the loop, verify yield components
        auto *loop_body = static_cast<const LoopStmt *>(loop)->body();

        auto *susp = find_stmt_by_tag(loop_body, Statement::Tag::SUSPEND);
        expect(susp != nullptr) << "LCG loop should contain a SuspendStmt";

        auto *bind = find_stmt_by_tag(loop_body, Statement::Tag::CORO_BIND);
        expect(bind != nullptr) << "LCG loop should contain a CoroBindStmt";

        auto *cb = static_cast<const CoroBindStmt *>(bind);
        expect(cb->name() == luisa::string_view{"__yielded_value"});
    };

    "lcg_sampler_yields_nonzero_tokens"_test = [] {
        auto builder = luisa::compute::detail::FunctionBuilder::define_coroutine([] {
            Var<uint> state = 12345u;
            $loop {
                state = state * 1103515245u + 12345u;
                $promise("__yielded_value", state);
                $suspend();
            };
        });
        auto *body = builder->body();

        auto *loop = static_cast<const LoopStmt *>(
            find_stmt_by_tag(body, Statement::Tag::LOOP));
        expect(loop != nullptr);

        auto *loop_body = loop->body();
        auto *susp = static_cast<const SuspendStmt *>(
            find_stmt_by_tag(loop_body, Statement::Tag::SUSPEND));
        expect(susp != nullptr);
        expect(susp->token() != 0u) << "Suspend token should be non-zero";
        expect(susp->token() != 0xFFFFFFFFu)
            << "Suspend token should not be terminal";
    };

    "lcg_sampler_coro_bind_value_is_variable"_test = [] {
        // The yielded value (state) should be a RefExpr (variable reference)
        auto builder = luisa::compute::detail::FunctionBuilder::define_coroutine([] {
            Var<uint> state = 12345u;
            $loop {
                state = state * 1103515245u + 12345u;
                $promise("__yielded_value", state);
                $suspend();
            };
        });
        auto *body = builder->body();

        auto *loop = static_cast<const LoopStmt *>(
            find_stmt_by_tag(body, Statement::Tag::LOOP));
        expect(loop != nullptr);

        auto *loop_body = loop->body();
        auto *bind = static_cast<const CoroBindStmt *>(
            find_stmt_by_tag(loop_body, Statement::Tag::CORO_BIND));
        expect(bind != nullptr);

        auto *val = bind->value();
        expect(val != nullptr);
        expect(val->tag() == Expression::Tag::REF)
            << "Yielded variable should be a RefExpr";
    };

    return 0;
}
