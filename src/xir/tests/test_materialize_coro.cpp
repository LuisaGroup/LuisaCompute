#include <algorithm>
#include <utility>

#include <luisa/ast/function_builder.h>
#include <luisa/coro/coro_func.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/canonicalize_control_flow.h>
#include <luisa/xir/passes/coro/coro_graph.h>
#include <luisa/xir/passes/coro/materialize_coro.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2ast.h>
#include <luisa/xir/translators/xir2text.h>

using namespace luisa;
using namespace luisa::compute;

namespace {

[[noreturn]] void fail(luisa::string_view message) {
    LUISA_ERROR_WITH_LOCATION("{}", message);
}

void require(bool condition, luisa::string_view message) {
    if (!condition) { fail(message); }
}

xir::CallableFunction *first_callable(xir::Module *module) {
    for (auto f : module->function_list()) {
        if (f->isa<xir::CallableFunction>()) { return static_cast<xir::CallableFunction *>(f); }
    }
    return nullptr;
}

size_t function_count(const xir::Module *module) {
    auto count = size_t{0u};
    for (auto _ : module->function_list()) {
        static_cast<void>(_);
        count++;
    }
    return count;
}

size_t count_tag(const xir::FunctionDefinition *function, xir::DerivedInstructionTag tag) {
    auto count = size_t{0u};
    function->traverse_instructions([&](const xir::Instruction *inst) noexcept {
        if (inst->derived_instruction_tag() == tag) { count++; }
    });
    return count;
}

size_t count_source_only_coro_markers(const xir::CallableFunction *function) {
    auto def = const_cast<xir::CallableFunction *>(function)->definition();
    require(def != nullptr, "Materialized callable must have a definition.");
    return count_tag(def, xir::DerivedInstructionTag::CORO_ID) +
           count_tag(def, xir::DerivedInstructionTag::CORO_TOKEN) +
           count_tag(def, xir::DerivedInstructionTag::CORO_REGISTER) +
           count_tag(def, xir::DerivedInstructionTag::SUSPEND);
}

bool has_loaded_uninitialized_local(const xir::FunctionDefinition *function) {
    struct Usage {
        size_t loads{0u};
        size_t stores{0u};
    };
    luisa::unordered_map<const xir::AllocaInst *, Usage> usage;
    function->traverse_instructions([&](const xir::Instruction *inst) noexcept {
        if (auto alloca = inst->isa<xir::AllocaInst>() ? static_cast<const xir::AllocaInst *>(inst) : nullptr;
            alloca != nullptr && alloca->is_local()) {
            usage.try_emplace(alloca);
        }
        if (auto load = inst->isa<xir::LoadInst>() ? static_cast<const xir::LoadInst *>(inst) : nullptr) {
            if (auto alloca = load->variable()->isa<xir::AllocaInst>() ? static_cast<const xir::AllocaInst *>(load->variable()) : nullptr;
                alloca != nullptr && alloca->is_local()) {
                usage[alloca].loads++;
            }
        }
        if (auto store = inst->isa<xir::StoreInst>() ? static_cast<const xir::StoreInst *>(inst) : nullptr) {
            if (auto alloca = store->variable()->isa<xir::AllocaInst>() ? static_cast<const xir::AllocaInst *>(store->variable()) : nullptr;
                alloca != nullptr && alloca->is_local()) {
                usage[alloca].stores++;
            }
        }
    });
    for (auto &&[alloca, counts] : usage) {
        if (counts.loads > 0u && counts.stores == 0u) { return true; }
    }
    return false;
}

bool ends_with(luisa::string_view value, luisa::string_view suffix) {
    return value.size() >= suffix.size() &&
           value.substr(value.size() - suffix.size()) == suffix;
}

const xir::MaterializedCoroScope *find_scope(const xir::MaterializeCoroResult &result, uint32_t token) {
    for (auto &&scope : result.scopes) {
        if (scope.token == token) { return &scope; }
    }
    return nullptr;
}

luisa::string sanitize_ast_name(luisa::string_view name) {
    auto result = luisa::string{name};
    for (auto &c : result) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') { c = '_'; }
    }
    return result;
}

luisa::string require_name(const xir::CallableFunction *function) {
    auto name = function->name();
    require(name.has_value(), "Expected materialized callable to carry a name.");
    return luisa::string{*name};
}

template<typename Body>
luisa::unique_ptr<xir::Module> translate_coroutine(Body &&body, luisa::string_view name) {
    auto builder = luisa::compute::detail::FunctionBuilder::define_coroutine([fn = std::forward<Body>(body)]() mutable noexcept {
        fn();
        luisa::compute::detail::FunctionBuilder::current()->return_();
    });
    builder->set_name(name);
    return xir::ast_to_xir_translate(builder->function(), {});
}

void verify_roundtrip_markers() {
    auto module = translate_coroutine([]() noexcept {
        Var<uint> x = 1u;
        auto id = coro_id();
        auto token = coro_token();
        promise("named_value", x + token);
        x += id.x;
        suspend("marker_scope", std::make_pair(x, "__yielded_value"));
        x += coro_token();
    }, "xir_test_roundtrip_markers");
    auto function = first_callable(module.get());
    require(function != nullptr, "Expected a translated callable for marker roundtrip.");
    auto dump = xir::xir_to_text_translate(module.get(), false);
    require(dump.find("coro_id") != luisa::string::npos, "AST -> XIR dump lost coro_id.");
    require(dump.find("coro_token") != luisa::string::npos, "AST -> XIR dump lost coro_token.");
    require(dump.find("coro_register") != luisa::string::npos, "AST -> XIR dump lost coro_register.");
    require(dump.find("suspend") != luisa::string::npos, "AST -> XIR dump lost suspend.");

    auto roundtrip_ast = xir::XIR2AST::build(function);
    require(roundtrip_ast != nullptr, "XIR2AST failed for marker coroutine.");
    auto roundtrip_module = xir::ast_to_xir_translate(roundtrip_ast->function(), {});
    auto roundtrip_dump = xir::xir_to_text_translate(roundtrip_module.get(), false);
    require(roundtrip_dump.find("coro_id") != luisa::string::npos, "AST -> XIR -> AST -> XIR dump lost coro_id.");
    require(roundtrip_dump.find("coro_token") != luisa::string::npos, "AST -> XIR -> AST -> XIR dump lost coro_token.");
    require(roundtrip_dump.find("coro_register") != luisa::string::npos, "AST -> XIR -> AST -> XIR dump lost coro_register.");
    require(roundtrip_dump.find("suspend") != luisa::string::npos, "AST -> XIR -> AST -> XIR dump lost suspend.");
}

void verify_materialization_pipeline() {
    auto module = translate_coroutine([]() noexcept {
        Var<uint> x = 0u;
        auto id = coro_id();
        auto token = coro_token();
        x += id.x + token;
        $if (x == 0u) {
            promise("if_value", x);
            suspend("if_scope", std::make_pair(x, "__yielded_value"));
            x += 1u;
        };
        $switch (x) {
            $case (1u) {
                promise("switch_value", x);
                suspend("switch_scope", std::make_pair(x, "__yielded_value"));
                x += 2u;
            };
            $default {
                x += 3u;
            };
        };
        $for (i, 0, 4) {
            x += cast<uint>(i) + 1u;
            $if (x < 8u) {
                promise("loop_value", x);
                suspend("loop_scope", std::make_pair(x, "__yielded_value"));
            };
        };
    }, "xir_test_materialization_pipeline");
    auto function = first_callable(module.get());
    require(function != nullptr, "Expected a translated callable for coroutine materialization.");
    auto pre_materialize_function_count = function_count(module.get());
    auto canonical_info = xir::Canoinicalize_Control_Flow_pass_run_on_Function(function);
    require(canonical_info.lowered_loop_count >= 1u,
            "Coroutine materialization test expected loop canonicalization to lower at least one loop.");

    auto graph = xir::compute_coro_graph(function);
    require(graph.entry != xir::invalid_coro_scope_ref, "Coroutine graph did not record an entry scope.");
    require(graph.tokens.size() == 3u, "Expected three suspend tokens in the coroutine graph.");

    auto result = xir::materialize_coro_pass_run_on_function(function);
    require(result.entry != nullptr, "Materialization did not generate an entry callable.");
    require(result.scopes.size() == 3u, "Materialization did not generate all resume scopes.");
    require(result.frame_interface_type != nullptr, "Materialization did not compute a frame interface type.");
    require(!result.frame_fields.empty(), "Expected coroutine frame fields for the test coroutine.");
    require(result.named_tokens.find("if_scope") != result.named_tokens.end(), "Missing named token for if_scope.");
    require(result.named_tokens.find("switch_scope") != result.named_tokens.end(), "Missing named token for switch_scope.");
    require(result.named_tokens.find("loop_scope") != result.named_tokens.end(), "Missing named token for loop_scope.");

    auto post_materialize_function_count = function_count(module.get());
    require(post_materialize_function_count >= pre_materialize_function_count + 4u,
            "Materialization did not append entry/resume callables into the original module.");

    auto entry_name = require_name(result.entry);
    require(ends_with(entry_name, ".coro.entry"), "Materialized entry callable has an unexpected name.");
    require(count_source_only_coro_markers(result.entry) == 0u,
            "Materialized entry callable still contains source-only coroutine markers.");
    require(xir::XIR2AST::build(result.entry) != nullptr, "XIR2AST failed on the materialized entry callable.");

    for (auto &&scope : result.scopes) {
        auto name = require_name(scope.function);
        auto suffix = luisa::string{".coro.resume."}.append(std::to_string(scope.token));
        require(ends_with(name, suffix), "Materialized resume callable has an unexpected name.");
        require(count_source_only_coro_markers(scope.function) == 0u,
                "Materialized resume callable still contains source-only coroutine markers.");
        require(std::find(scope.output_fields.begin(), scope.output_fields.end(), 1u) != scope.output_fields.end(),
                "Materialized resume callable must always write the target token field.");
        require(xir::XIR2AST::build(scope.function) != nullptr, "XIR2AST failed on a materialized resume callable.");
    }

    auto materialized_dump = xir::xir_to_text_translate(module.get(), false);
    require(materialized_dump.find(".coro.entry") != luisa::string::npos, "Module dump does not contain the materialized entry callable.");
    require(materialized_dump.find(".coro.resume.") != luisa::string::npos, "Module dump does not contain the materialized resume callables.");
}

void verify_legacy_coro_graph_adapter() {
    auto coro = coroutine::Coroutine<void()>{[]() noexcept {
        Var<uint> x = 0u;
        auto id = coro_id();
        auto token = coro_token();
        x += id.x + token;
        $if (x == 0u) {
            promise("if_value", x);
            suspend("if_scope", std::make_pair(x, "__yielded_value"));
            x += 1u;
        };
        $switch (x) {
            $case (1u) {
                promise("switch_value", x);
                suspend("switch_scope", std::make_pair(x, "__yielded_value"));
                x += 2u;
            };
            $default {
                x += 3u;
            };
        };
        $for (i, 0, 4) {
            x += cast<uint>(i) + 1u;
            $if (x < 8u) {
                promise("loop_value", x);
                suspend("loop_scope", std::make_pair(x, "__yielded_value"));
            };
        };
    }};
    auto graph = coro.graph();
    require(graph != nullptr, "Legacy CoroGraph adapter did not produce a graph.");
    require(coro.subroutine_count() == 4u, "Legacy CoroGraph adapter returned an unexpected subroutine count.");
    require(graph->entry().cc().tag() == Function::Tag::CALLABLE, "Legacy CoroGraph entry is not a callable AST function.");
    require(ends_with(graph->entry().cc().name(), sanitize_ast_name(".coro.entry")), "Legacy CoroGraph entry callable name is unexpected.");
    require(std::find(graph->entry().output_fields().begin(), graph->entry().output_fields().end(), 1u) != graph->entry().output_fields().end(),
            "Legacy CoroGraph entry must write the target token field.");
    require(graph->named_tokens().find("if_scope") != graph->named_tokens().end(), "Legacy CoroGraph is missing the if_scope token.");
    require(graph->named_tokens().find("switch_scope") != graph->named_tokens().end(), "Legacy CoroGraph is missing the switch_scope token.");
    require(graph->named_tokens().find("loop_scope") != graph->named_tokens().end(), "Legacy CoroGraph is missing the loop_scope token.");
    require(graph->frame()->designated_fields().find("__yielded_value") != graph->frame()->designated_fields().end(),
            "Legacy CoroGraph frame is missing __yielded_value.");
    require(graph->frame()->designated_fields().find("if_value") != graph->frame()->designated_fields().end(),
            "Legacy CoroGraph frame is missing if_value.");
    require(graph->frame()->designated_fields().find("switch_value") != graph->frame()->designated_fields().end(),
            "Legacy CoroGraph frame is missing switch_value.");
    require(graph->frame()->designated_fields().find("loop_value") != graph->frame()->designated_fields().end(),
            "Legacy CoroGraph frame is missing loop_value.");
    for (auto &&name : {"if_scope", "switch_scope", "loop_scope"}) {
        auto iter = graph->named_tokens().find(name);
        require(iter != graph->named_tokens().end(),
                luisa::format("Legacy CoroGraph named token '{}' is missing.", name));
        auto token = iter->second;
        auto suffix = sanitize_ast_name(luisa::string{".coro.resume."}.append(std::to_string(token)));
        require(ends_with(graph->node(token).cc().name(), suffix),
                luisa::format("Legacy CoroGraph resume callable for '{}' has an unexpected name.", name));
        require(std::find(graph->node(token).output_fields().begin(), graph->node(token).output_fields().end(), 1u) != graph->node(token).output_fields().end(),
                luisa::format("Legacy CoroGraph resume callable for '{}' must write the target token field.", name));
    }
}

void verify_nested_coroutine_adapter() {
    auto nested2 = coroutine::Coroutine<void(uint)>{[](UInt n) noexcept {
        $for (i, n) {
            device_log("nested2: {}", i);
            $suspend();
        };
    }};
    auto nested1 = coroutine::Coroutine<void(uint)>{[&](UInt n) noexcept {
        $for (i, n) {
            $await nested2(i);
            device_log("nested1: {}", i);
        };
    }};
    auto top_level = coroutine::Coroutine<void()>{[&]() noexcept {
        $await nested1(4u);
    }};

    require(nested1.graph() != nullptr, "Nested coroutine adapter did not produce a graph for nested1.");
    require(top_level.graph() != nullptr, "Nested coroutine adapter did not produce a graph for top_level.");
    require(nested1.subroutine_count() >= 2u, "Nested coroutine adapter did not materialize nested1 resume scopes.");
    require(top_level.subroutine_count() >= 2u, "Nested coroutine adapter did not materialize top_level resume scopes.");
}

void verify_loop_carried_state_is_saved() {
    auto module = translate_coroutine([]() noexcept {
        Var<uint> x = 1u;
        Var<uint> i = 0u;
        $while (i < 3u) {
            suspend("loop_scope");
            x += i + 1u;
            i += 1u;
        };
        promise("final_value", x);
        suspend("done_scope", std::make_pair(x, "__yielded_value"));
    }, "xir_test_loop_carried_state");
    auto function = first_callable(module.get());
    require(function != nullptr, "Expected a translated callable for loop-carried-state materialization.");

    auto result = xir::materialize_coro_pass_run_on_function(function);
    for (auto &&scope : result.scopes) {
        luisa::string targets;
        for (auto i = 0u; i < scope.target_tokens.size(); i++) {
            if (i != 0u) { targets.append(", "); }
            targets.append(luisa::format("{}", scope.target_tokens[i]));
        }
        LUISA_INFO("materialized scope token={} inputs={} outputs={} targets=[{}]",
                   scope.token,
                   scope.input_fields.size(),
                   scope.output_fields.size(),
                   targets);
    }
    auto loop_token_iter = result.named_tokens.find("loop_scope");
    require(loop_token_iter != result.named_tokens.end(), "Missing loop_scope token in loop-carried-state test.");
    auto loop_scope = find_scope(result, loop_token_iter->second);
    require(loop_scope != nullptr, "Missing loop_scope materialized resume callable.");
    LUISA_INFO("loop_scope token={} input_fields={} output_fields={} targets={} [{}]",
               loop_token_iter->second,
               loop_scope->input_fields.size(),
               loop_scope->output_fields.size(),
               loop_scope->target_tokens.size(),
               [&] {
                   luisa::string s;
                   for (auto i = 0u; i < loop_scope->target_tokens.size(); i++) {
                       if (i != 0u) { s.append(", "); }
                       s.append(luisa::format("{}", loop_scope->target_tokens[i]));
                   }
                   return s;
               }());
    require(loop_scope->target_tokens.size() == 2u, "Loop-carried-state test expected a self-loop and an exit transition.");
    LUISA_INFO("loop_scope input_fields = {}, output_fields = {}, targets = {}",
               loop_scope->input_fields.size(),
               loop_scope->output_fields.size(),
               loop_scope->target_tokens.size());
    require(loop_scope->input_fields.size() == loop_scope->output_fields.size(),
            "Loop-carried-state resume scope should save every live frame field it loads.");
    require(loop_scope->output_fields.size() >= 3u,
            "Loop-carried-state resume scope should save user state in addition to the target token.");
}

void verify_materialized_roundtrip_has_no_uninitialized_locals() {
    auto module = translate_coroutine([]() noexcept {
        Var<uint> value = 0u;
        $for (depth, 6) {
            suspend("loop_scope");
            value += cast<uint>(depth) + 1u;
        };
        promise("final_value", value);
        suspend("done_scope", std::make_pair(value, "__yielded_value"));
    }, "xir_test_materialized_roundtrip_no_uninit");
    auto function = first_callable(module.get());
    require(function != nullptr, "Expected a translated callable for materialized roundtrip validation.");

    auto result = xir::materialize_coro_pass_run_on_function(function);
    require(!has_loaded_uninitialized_local(result.entry->definition()),
            "Materialized entry contains a local that is loaded before any store.");
    auto entry_ast = xir::XIR2AST::build(result.entry);
    require(entry_ast != nullptr, "XIR2AST failed on materialized entry during roundtrip validation.");
    auto entry_roundtrip_module = xir::ast_to_xir_translate(entry_ast->function(), {});
    auto entry_roundtrip = first_callable(entry_roundtrip_module.get());
    require(entry_roundtrip != nullptr, "Round-tripped materialized entry did not produce a callable.");
    require(!has_loaded_uninitialized_local(entry_roundtrip->definition()),
            "Round-tripped materialized entry contains a local that is loaded before any store.");

    for (auto &&scope : result.scopes) {
        require(!has_loaded_uninitialized_local(scope.function->definition()),
                "Materialized resume contains a local that is loaded before any store.");
        auto scope_ast = xir::XIR2AST::build(scope.function);
        require(scope_ast != nullptr, "XIR2AST failed on a materialized resume callable during roundtrip validation.");
        auto scope_roundtrip_module = xir::ast_to_xir_translate(scope_ast->function(), {});
        auto scope_roundtrip = first_callable(scope_roundtrip_module.get());
        require(scope_roundtrip != nullptr, "Round-tripped materialized resume did not produce a callable.");
        require(!has_loaded_uninitialized_local(scope_roundtrip->definition()),
                "Round-tripped materialized resume contains a local that is loaded before any store.");
    }
}

}// namespace

int main() {
    luisa::log_level_verbose();
    verify_roundtrip_markers();
    verify_materialization_pipeline();
    verify_legacy_coro_graph_adapter();
    verify_nested_coroutine_adapter();
    verify_loop_carried_state_is_saved();
    verify_materialized_roundtrip_has_no_uninitialized_locals();
    return 0;
}
