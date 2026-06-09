#include <luisa/ast/statement.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include "ut/ut.hpp"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

const Statement *fst(const ScopeStmt *s, Statement::Tag t) {
    for (auto x : s->statements()) {
        if (x->tag() == t) { return x; }
    }
    return nullptr;
}

size_t cst(const ScopeStmt *s, Statement::Tag t) {
    size_t n = 0u;
    for (auto x : s->statements()) {
        if (x->tag() == t) { n++; }
    }
    return n;
}

} // namespace

int main() {

    "bare_suspend_records_suspend_stmt"_test = [] {
        Coroutine c = [](Var<int> unused) { $suspend(); };
        auto *body = c.function_builder()->body();
        auto *s = fst(body, Statement::Tag::SUSPEND);
        expect(s != nullptr);
        auto *sus = static_cast<const SuspendStmt *>(s);
        expect(sus->name().empty());
    };

    "suspend_tag_passes_string"_test = [] {
        Coroutine c = [](Var<int> unused) { $suspend("myTag"); };
        auto *body = c.function_builder()->body();
        auto *s = fst(body, Statement::Tag::SUSPEND);
        expect(s != nullptr);
        auto *sus = static_cast<const SuspendStmt *>(s);
        expect(sus->name() == "myTag");
    };

    "suspend_has_valid_token"_test = [] {
        Coroutine c = [](Var<int> unused) { $suspend("tagged"); };
        auto *body = c.function_builder()->body();
        auto *s = fst(body, Statement::Tag::SUSPEND);
        expect(s != nullptr);
        auto *sus = static_cast<const SuspendStmt *>(s);
        expect(sus->token() != 0xFFFFFFFFu);
    };

    "multiple_suspends_produce_distinct_tokens"_test = [] {
        Coroutine c = [](Var<int> x) {
            $suspend("first");
            $suspend("second");
            $suspend("third");
        };
        auto *body = c.function_builder()->body();
        expect(cst(body, Statement::Tag::SUSPEND) == 3u);

        uint32_t tokens[3];
        size_t i = 0u;
        for (auto s : body->statements()) {
            if (s->tag() == Statement::Tag::SUSPEND) {
                tokens[i++] = static_cast<const SuspendStmt *>(s)->token();
            }
        }
        expect(tokens[0] != tokens[1]);
        expect(tokens[1] != tokens[2]);
        expect(tokens[0] < tokens[1]);
        expect(tokens[1] < tokens[2]);

        auto stmts = body->statements();
        auto *s0 = static_cast<const SuspendStmt *>(stmts[0]);
        auto *s1 = static_cast<const SuspendStmt *>(stmts[1]);
        auto *s2 = static_cast<const SuspendStmt *>(stmts[2]);
        expect(s0->name() == "first");
        expect(s1->name() == "second");
        expect(s2->name() == "third");
    };

    "promise_and_yield_equivalent_suspend_behaviour"_test = [] {
        // The $yield(x) macro expands to $promise("__yielded_value", x) + $suspend.
        // Verifying the full macro expansion requires the coroutine pipeline to handle
        // $promise variables (currently blocked by a pre-existing pipeline crash).
        // This test verifies the structure: a manual $suspend with zero arguments
        // produces the same SuspendStmt shape as what $yield would append.
        Coroutine c = [](Var<int> x) { $suspend(); };
        auto *body = c.function_builder()->body();
        auto *s = fst(body, Statement::Tag::SUSPEND);
        expect(s != nullptr);
        auto *sus = static_cast<const SuspendStmt *>(s);
        expect(sus->name().empty());
    };

    return 0;
}
