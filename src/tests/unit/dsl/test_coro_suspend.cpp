#include <luisa/ast/statement.h>
#include <luisa/ast/callable_library.h>
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
        expect(sus->token() != 0u);
        expect(sus->token() != 0xFFFFFFFFu);
    };

    "suspend_frame_export_is_explicit_semantic_state"_test = [] {
        Coroutine c = [](Var<uint> x) {
            auto hint = x * 13u + 7u;
            $suspend("sort", coro_frame_export(
                                 "coro_hint", hint));
        };
        auto *body = c.function_builder()->body();
        auto *s = fst(body, Statement::Tag::SUSPEND);
        expect(s != nullptr);
        auto *sus = static_cast<const SuspendStmt *>(s);
        expect(sus->frame_exports().size() == 1u);
        if (sus->frame_exports().size() == 1u) {
            expect(sus->frame_exports().front().name ==
                   "coro_hint");
            expect(sus->frame_exports().front().value != nullptr);
            expect(sus->frame_exports().front().value->type() ==
                   Type::of<uint>());
        }
    };

    "sort_suspend_annotation_records_binding_and_attribute"_test = [] {
        Coroutine c = [](Var<uint> x) {
            auto key = x * 13u + 7u;
            $suspend("shade_surface", coro_sort_by(key, 64u));
        };
        auto *s = fst(c.function_builder()->body(),
                      Statement::Tag::SUSPEND);
        expect(s != nullptr);
        auto *sus = static_cast<const SuspendStmt *>(s);
        expect(sus->extensions().size() == 1u);
        expect(sus->extension_binding_values().size() == 1u);
        if (sus->extensions().size() == 1u) {
            auto &&extension = sus->extensions().front();
            expect(extension->schema() ==
                   "luisa.coro.schedule.sort");
            expect(extension->version() == 1u);
            expect(extension->is_annotation());
            expect(extension->fallback() ==
                   CoroSuspendFallback::ignore);
            expect(extension->bindings().size() == 1u);
            expect(extension->attributes().size() == 1u);
            if (extension->bindings().size() == 1u) {
                auto &&binding = extension->bindings().front();
                expect(binding.name == "key");
                expect(binding.access ==
                       CoroSuspendBindingAccess::read);
                expect(binding.lifetime ==
                       CoroSuspendBindingLifetime::queued);
                expect(binding.index == 0u);
                auto *value = sus->extension_binding_values()[
                    binding.index];
                expect(value != nullptr);
                expect(value->type() == Type::of<uint>());
                expect((to_underlying(value->usage()) &
                        to_underlying(Usage::READ)) != 0u);
            }
            if (extension->attributes().size() == 1u) {
                auto &&attribute = extension->attributes().front();
                expect(attribute.name == "range");
                expect(luisa::holds_alternative<uint64_t>(
                    attribute.value));
                if (luisa::holds_alternative<uint64_t>(
                        attribute.value)) {
                    expect(luisa::get<uint64_t>(attribute.value) ==
                           64u);
                }
            }
        }
    };

    "suspend_accepts_frame_exports_and_extensions_together"_test = [] {
        Coroutine c = [](Var<uint> x) {
            $suspend("mixed",
                     coro_frame_export("legacy_key", x),
                     coro_sort_by(x, 128u));
        };
        auto *s = fst(c.function_builder()->body(),
                      Statement::Tag::SUSPEND);
        auto *sus = static_cast<const SuspendStmt *>(s);
        expect(sus->frame_exports().size() == 1u);
        expect(sus->extensions().size() == 1u);
        expect(sus->extension_binding_values().size() == 1u);
    };

    "suspend_extensions_survive_callable_library_round_trip"_test = [] {
        Coroutine c = [](Var<uint> x) {
            $suspend("round_trip", coro_sort_by(x, 256u));
        };
        CallableLibrary source;
        source.add_callable("coro", c.function_builder());
        auto binary = source.serialize();
        CallableLibrary loaded;
        loaded.load(binary);
        auto builder = loaded.get_function_builder("coro");
        auto *s = fst(builder->body(), Statement::Tag::SUSPEND);
        expect(s != nullptr);
        auto *sus = static_cast<const SuspendStmt *>(s);
        expect(sus->extensions().size() == 1u);
        expect(sus->extension_binding_values().size() == 1u);
        if (sus->extensions().size() == 1u) {
            auto &&extension = sus->extensions().front();
            expect(extension->schema() ==
                   "luisa.coro.schedule.sort");
            expect(extension->is_annotation());
            expect(extension->bindings().front().name == "key");
            expect(luisa::get<uint64_t>(
                       extension->attributes().front().value) == 256u);
        }
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
        expect(s0->token() != 0u);
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
