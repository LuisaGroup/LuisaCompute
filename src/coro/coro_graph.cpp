//
// Created by Mike on 2024/5/8.
//

#include <fstream>

#include <algorithm>

#include <luisa/core/logging.h>
#include <luisa/ast/function_builder.h>
#include <luisa/coro/coro_frame_desc.h>
#include <luisa/coro/coro_graph.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/coro/id.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro/coro_frame.h>
#include <luisa/xir/passes/coro/coro_graph.h>
#include <luisa/xir/passes/coro/materialize_coro.h>
#include <luisa/xir/translators/ast2xir.h>
#include <luisa/xir/translators/xir2ast.h>
#include <luisa/xir/translators/xir2text.h>

namespace luisa::compute::coroutine {
namespace {

[[nodiscard]] xir::CallableFunction *find_translated_coroutine(xir::Module *module, Function coroutine) noexcept {
    LUISA_ASSERT(module != nullptr, "XIR coroutine bridge requires a translated module.");
    auto target_name = coroutine.name();
    xir::CallableFunction *first_callable = nullptr;
    for (auto function : module->function_list()) {
        auto callable = function->isa<xir::CallableFunction>() ? static_cast<xir::CallableFunction *>(function) : nullptr;
        if (callable == nullptr) { continue; }
        if (first_callable == nullptr) { first_callable = callable; }
        if (!target_name.empty()) {
            auto name = callable->name();
            if (name.has_value() && *name == target_name) { return callable; }
        }
    }
    return first_callable;
}

[[nodiscard]] luisa::vector<uint> convert_indices(luisa::span<const uint32_t> indices) noexcept {
    return luisa::vector<uint>{indices.begin(), indices.end()};
}

[[nodiscard]] luisa::unordered_map<luisa::string, uint>
convert_named_tokens(const luisa::unordered_map<luisa::string, uint32_t> &named_tokens) noexcept {
    luisa::unordered_map<luisa::string, uint> result;
    for (auto &&[name, token] : named_tokens) { result.emplace(name, token); }
    return result;
}

[[nodiscard]] CoroFrameDesc::DesignatedFieldDict
collect_designated_fields(luisa::span<const xir::CoroDesignatedFieldInfo> designated_fields) noexcept {
    CoroFrameDesc::DesignatedFieldDict result;
    for (auto &&field : designated_fields) { result.emplace(field.name, field.frame_index); }
    return result;
}

void dump_roundtrip_xir_if_requested(luisa::string_view stem,
                                     Function function) noexcept {
    if (auto env = std::getenv("LUISA_CORO_DEBUG_XIR");
        env == nullptr || std::string_view{env} != "1") {
        return;
    }
    auto module = xir::ast_to_xir_translate(function, {});
    auto callable = find_translated_coroutine(module.get(), function);
    if (callable == nullptr) { return; }
    auto dump = xir::xir_to_text_translate(module.get(), true);
    auto filename = luisa::string{"coro-debug."}.append(stem).append(".xir");
    std::ofstream out{filename.c_str()};
    out << dump.c_str();
}

void dump_materialized_module_if_requested(xir::CallableFunction *function) noexcept {
    if (auto env = std::getenv("LUISA_CORO_DEBUG_XIR");
        env == nullptr || std::string_view{env} != "1") {
        return;
    }
    if (function == nullptr || function->parent_module() == nullptr) { return; }
    auto dump = xir::xir_to_text_translate(function->parent_module(), true);
    std::ofstream out{"coro-debug.materialized-module.xir"};
    out << dump.c_str();
}

[[nodiscard]] luisa::unordered_map<CoroToken, CoroGraph::Node>
collect_nodes(Function coroutine,
              const xir::MaterializeCoroResult &materialized) noexcept {
    auto wrap_materialized_node = [&](luisa::shared_ptr<const compute::detail::FunctionBuilder> callable) noexcept {
        auto inner = callable->function();
        return compute::detail::FunctionBuilder::define_callable([&] {
            auto fb = compute::detail::FunctionBuilder::current();
            if (auto name = inner.name(); !name.empty()) { fb->set_name(name); }
            LUISA_ASSERT(coroutine.arguments().size() == coroutine.bound_arguments().size(),
                         "Invalid coroutine capture list size (expected {}, got {}).",
                         coroutine.arguments().size(), coroutine.bound_arguments().size());
            luisa::vector<const Expression *> args;
            args.reserve(1u + coroutine.arguments().size());
            auto inner_args = inner.arguments();
            LUISA_ASSERT(!inner_args.empty() && inner_args.front().is_reference(),
                         "Materialized coroutine node '{}' must start with a frame reference argument.",
                         inner.debug_name());
            args.emplace_back(fb->reference(inner_args.front().type()));
            for (auto i = 0u; i < coroutine.arguments().size(); i++) {
                auto def_arg = coroutine.arguments()[i];
                auto internal_arg = luisa::visit(
                    [&]<typename T>(T binding) noexcept -> const Expression * {
                        if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                            return fb->buffer_binding(def_arg.type(), binding.handle, binding.offset, binding.size);
                        } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                            return fb->texture_binding(def_arg.type(), binding.handle, binding.level);
                        } else if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                            return fb->bindless_array_binding(binding.handle);
                        } else if constexpr (std::is_same_v<T, Function::AccelBinding>) {
                            return fb->accel_binding(binding.handle);
                        } else {
                            static_assert(std::is_same_v<T, luisa::monostate>);
                            switch (def_arg.tag()) {
                                case Variable::Tag::REFERENCE: return fb->reference(def_arg.type());
                                case Variable::Tag::BUFFER: return fb->buffer(def_arg.type());
                                case Variable::Tag::TEXTURE: return fb->texture(def_arg.type());
                                case Variable::Tag::BINDLESS_ARRAY: return fb->bindless_array();
                                case Variable::Tag::ACCEL: return fb->accel();
                                default: return fb->argument(def_arg.type());
                            }
                        }
                    },
                    coroutine.bound_arguments()[i]);
                args.emplace_back(internal_arg);
            }
            fb->call(inner, luisa::span{args});
        });
    };
    luisa::unordered_map<CoroToken, CoroGraph::Node> nodes;
    auto entry = xir::XIR2AST::build(materialized.entry);
    LUISA_ASSERT(entry != nullptr, "Failed to rebuild AST for materialized coroutine entry.");
    entry = wrap_materialized_node(std::move(entry));
    nodes.emplace(coro_token_entry, CoroGraph::Node{
                                      convert_indices(materialized.entry_input_fields),
                                      convert_indices(materialized.entry_output_fields),
                                      convert_indices(materialized.entry_target_tokens),
                                      std::move(entry)});
    auto expected_token = 1u;
    for (auto &&scope : materialized.scopes) {
        LUISA_ASSERT(scope.token == expected_token,
                     "Legacy coroutine schedulers require dense resume tokens, but token {} appeared at position {}.",
                     scope.token, expected_token);
        auto callable = xir::XIR2AST::build(scope.function);
        LUISA_ASSERT(callable != nullptr, "Failed to rebuild AST for materialized coroutine resume {}.", scope.token);
        callable = wrap_materialized_node(std::move(callable));
        nodes.emplace(scope.token, CoroGraph::Node{
                                       convert_indices(scope.input_fields),
                                       convert_indices(scope.output_fields),
                                       convert_indices(scope.target_tokens),
                                       std::move(callable)});
        expected_token++;
    }
    return nodes;
}

}// namespace

CoroGraph::Node::Node(luisa::vector<uint> input_fields,
                      luisa::vector<uint> output_fields,
                      luisa::vector<CoroToken> targets,
                      CC current_continuation) noexcept
    : _input_fields{std::move(input_fields)},
      _output_fields{std::move(output_fields)},
      _targets{std::move(targets)},
      _cc{std::move(current_continuation)} {}

CoroGraph::Node::~Node() noexcept = default;

Function CoroGraph::Node::cc() const noexcept { return _cc->function(); }

luisa::string CoroGraph::Node::dump() const noexcept {
    luisa::string s;
    s.append("  Input Fields: [");
    for (auto i : _input_fields) {
        s.append(luisa::format("{}, ", i));
    }
    if (!_input_fields.empty()) {
        s.pop_back();
        s.pop_back();
    }
    s.append("]\n");
    s.append("  Output Fields: [");
    for (auto i : _output_fields) {
        s.append(luisa::format("{}, ", i));
    }
    if (!_output_fields.empty()) {
        s.pop_back();
        s.pop_back();
    }
    s.append("]\n");
    s.append("  Transition Targets: [");
    for (auto i : _targets) {
        s.append(luisa::format("{}, ", i));
    }
    if (!_targets.empty()) {
        s.pop_back();
        s.pop_back();
    }
    s.append("]\n");
    return s;
}

CoroGraph::CoroGraph(luisa::shared_ptr<const CoroFrameDesc> frame_desc,
                     luisa::unordered_map<CoroToken, Node> nodes,
                     luisa::unordered_map<luisa::string, CoroToken> named_tokens) noexcept
    : _frame{std::move(frame_desc)},
      _nodes{std::move(nodes)},
      _named_tokens{std::move(named_tokens)} {}

CoroGraph::~CoroGraph() noexcept = default;

const CoroGraph::Node &CoroGraph::entry() const noexcept {
    return node(coro_token_entry);
}

const CoroGraph::Node &CoroGraph::node(CoroToken token) const noexcept {
    auto iter = _nodes.find(token);
    LUISA_ASSERT(iter != _nodes.end(),
                 "Coroutine node with token {} not found.",
                 token);
    return iter->second;
}

const CoroGraph::Node &CoroGraph::node(luisa::string_view name) const noexcept {
    auto iter = _named_tokens.find(name);
    LUISA_ASSERT(iter != _named_tokens.end(),
                 "Coroutine node with name '{}' not found.",
                 name);
    return node(iter->second);
}

luisa::string CoroGraph::dump() const noexcept {
    luisa::string s;
    s.append("Arguments:\n");
    auto args = entry().cc().arguments();
    for (auto i = 0u; i < args.size(); i++) {
        s.append(luisa::format("  Argument {}: ", i));
        s.append(args[i].type()->description());
        if (args[i].is_reference()) { s.append(" &"); }
        s.append("\n");
    }
    s.append("Frame:\n").append(_frame->dump());
    for (auto &&[token, node] : _nodes) {
        if (token == coro_token_entry) {
            s.append("Entry:\n");
        } else {
            s.append(luisa::format("Node {}:\n", token));
        }
        s.append(node.dump());
    }
    if (!_named_tokens.empty()) {
        s.append("Named Tokens:\n");
        for (auto &&[name, token] : _named_tokens) {
            s.append(luisa::format("  {} -> \"{}\"\n", token, name));
        }
    }
    return s;
}

luisa::shared_ptr<const CoroGraph> CoroGraph::create(Function coroutine) noexcept {
    LUISA_ASSERT(coroutine, "Coroutine graph creation requires a valid function.");
    LUISA_ASSERT(coroutine.tag() == Function::Tag::COROUTINE,
                 "CoroGraph::create expects a coroutine function, got tag {}.",
                 static_cast<uint>(coroutine.tag()));
    auto module = xir::ast_to_xir_translate(coroutine, {});
    auto callable = find_translated_coroutine(module.get(), coroutine);
    LUISA_ASSERT(callable != nullptr, "Failed to locate translated XIR coroutine callable.");
    auto materialized = xir::materialize_coro_pass_run_on_function(callable);
    LUISA_ASSERT(materialized.entry != nullptr, "XIR coroutine materialization did not produce an entry callable.");
    LUISA_ASSERT(materialized.frame_interface_type != nullptr, "XIR coroutine materialization did not produce a frame interface.");
    dump_materialized_module_if_requested(materialized.entry);
    auto frame_desc = CoroFrameDesc::create(materialized.frame_interface_type,
                                            collect_designated_fields(materialized.designated_fields));
    auto nodes = collect_nodes(coroutine, materialized);
    dump_roundtrip_xir_if_requested("entry", nodes.at(coro_token_entry).cc());
    for (auto &&[token, node] : nodes) {
        if (token == coro_token_entry) { continue; }
        dump_roundtrip_xir_if_requested(luisa::string{"resume."}.append(std::to_string(token)), node.cc());
    }
    return luisa::make_shared<CoroGraph>(std::move(frame_desc),
                                         std::move(nodes),
                                         convert_named_tokens(materialized.named_tokens));
}

}// namespace luisa::compute::coroutine
