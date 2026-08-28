#include "ut/ut.hpp"
#include <luisa/coro/coro_graph.h>
#include <luisa/dsl/coro_frame.h>
#include <luisa/dsl/func.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/clock.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/passes/coro_materialize.h>
#include <luisa/xir/passes/coro_split.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace luisa::compute::coro;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] KernelFunction *make_kernel_with_body(Module &m, BasicBlock *&body_out) noexcept {
    auto *k = m.create_kernel();
    body_out = k->create_body_block();
    return k;
}

}// namespace

void reg_coro_graph() {

    "one_suspend_two_nodes_one_edge"_test = [] {
        // given: a kernel with one suspend point
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());

        auto *suspend_bb = k->create_basic_block();
        auto *resume_bb = k->create_basic_block();

        b.set_insertion_point(body);
        b.cond_br(cond, suspend_bb, resume_bb);

        b.set_insertion_point(suspend_bb);
        b.coro_suspend(1u, "checkpoint", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(1u, nullptr);
        b.return_void();

        // Run cfg-distill (read-only) on the kernel before splitting
        auto cfg = coro_cfg_distill_pass_run_on_function(k);

        // Run the coroutine pipeline
        auto split = coro_split_pass_run_on_module_with_cfg_and_frame_info(&m, cfg, nullptr);
        auto info = coro_materialize_pass_run_on_module_with_cfg(&m, cfg, split);

        // when: build CoroGraph
        auto graph = CoroGraph::from_module(m, info, cfg, split);

        // then: 2 nodes, 1 edge
        expect(graph.node_count() == 2u);
        expect(graph.edge_count() == 1u);

        // entry node (index 0)
        auto &n0 = graph.node(0u);
        expect(n0.index == 0u);
        expect(n0.token == 0u);// entry token
        expect(n0.name.empty());
        expect(!n0.is_terminal);
        expect(n0.callable != nullptr);

        // scope 1 node
        auto &n1 = graph.node(1u);
        expect(n1.index == 1u);
        expect(n1.token == 1u);
        expect(n1.name == "checkpoint");
        expect(!n1.is_terminal);
        expect(n1.callable != nullptr);

        // edge from 0 → 1
        auto *e = graph.edge(0u, 1u);
        expect(e != nullptr);
        expect(e->from_index == 0u);
        expect(e->to_index == 1u);

        // token lookup
        auto *t0 = graph.node_by_token(0u);
        expect(t0 != nullptr);
        expect(t0->index == 0u);

        auto *t1 = graph.node_by_token(1u);
        expect(t1 != nullptr);
        expect(t1->index == 1u);

        // name lookup
        auto *nm = graph.node_by_name("checkpoint");
        expect(nm != nullptr);
        expect(nm->index == 1u);
    };

    "graph_preserves_complete_extensions_and_shared_slot_accesses"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *state = b.clock();
        state->set_name("shared_state");
        auto *output = b.alloca_local(Type::of<float>());
        output->set_name("stage_output");
        luisa::vector<CoroSuspendExtensionPtr> extensions;
        extensions.emplace_back(make_coro_suspend_extension_data(
            "com.example.nn-shade", 3u,
            CoroSuspendFallback::reject,
            {{.name = "input",
              .access = CoroSuspendBindingAccess::read,
              .lifetime = CoroSuspendBindingLifetime::resumed,
              .index = 0u},
             {.name = "output",
              .access = CoroSuspendBindingAccess::write,
              .lifetime = CoroSuspendBindingLifetime::resumed,
              .index = 1u}},
            {{.name = "network", .value = luisa::string{"bunny"}}}));
        extensions.emplace_back(make_coro_suspend_annotation_data(
            "luisa.coro.debug.watch", 1u,
            CoroSuspendFallback::ignore,
            {{.name = "value",
              .access = CoroSuspendBindingAccess::read,
              .lifetime = CoroSuspendBindingLifetime::queued,
              .index = 2u}},
            {{.name = "label", .value = luisa::string{"state"}}}));
        luisa::vector<luisa::string> export_names{"legacy"};
        luisa::vector<Value *> export_values{state};
        luisa::vector<Value *> binding_values{state, output, state};
        b.coro_suspend(
            5u, "extension-boundary", nullptr,
            luisa::span{export_names}, luisa::span{export_values},
            std::move(extensions), luisa::span{binding_values});
        b.set_insertion_point(resume);
        b.coro_resume(5u, nullptr);
        static_cast<void>(b.call(
            Type::of<uint64_t>(), ArithmeticOp::BINARY_ADD,
            {state, m.create_constant_one(Type::of<uint64_t>())}));
        static_cast<void>(b.load(Type::of<float>(), output));
        b.return_void();

        auto cfg = coro_cfg_distill_pass_run_on_function(kernel);
        expect(cfg.succeeded());
        auto split = coro_split_pass_run_on_module_with_cfg_and_frame_info(
            &m, cfg, nullptr);
        auto info = coro_materialize_pass_run_on_module_with_cfg(
            &m, cfg, split);
        auto graph = CoroGraph::from_module(m, info, cfg, split);

        expect(graph.boundary_count() == 1u);
        if (graph.boundary_count() != 1u) { return; }
        auto &boundary = graph.boundary(0u);
        expect(boundary.index == 0u);
        expect(boundary.from_index == 0u);
        expect(boundary.to_index == 1u);
        expect(boundary.token == 5u);
        expect(boundary.extensions.size() == 2u);
        expect(boundary.bindings.size() == 3u);
        if (boundary.extensions.size() == 2u) {
            expect(boundary.extensions[0u]->schema() ==
                   "com.example.nn-shade");
            expect(boundary.extensions[0u]->version() == 3u);
            expect(!boundary.extensions[0u]->is_annotation());
            expect(boundary.extensions[0u]->fallback() ==
                   CoroSuspendFallback::reject);
            expect(boundary.extensions[0u]->attributes().size() == 1u);
            expect(boundary.extensions[1u]->schema() ==
                   "luisa.coro.debug.watch");
            expect(boundary.extensions[1u]->is_annotation());
        }
        if (boundary.bindings.size() == 3u) {
            auto &stage_input = boundary.bindings[0u];
            auto &stage_output = boundary.bindings[1u];
            auto &debug_watch = boundary.bindings[2u];
            expect(stage_input.type() == Type::of<uint64_t>());
            expect(stage_input.readable());
            expect(!stage_input.writable());
            expect(stage_input.materialized());
            expect(stage_output.type() == Type::of<float>());
            expect(!stage_output.readable());
            expect(stage_output.writable());
            expect(stage_output.materialized());
            expect(debug_watch.type() == Type::of<uint64_t>());
            expect(debug_watch.readable());
            expect(debug_watch.materialized());
            expect(stage_input.pieces().size() == 1u);
            expect(debug_watch.pieces().size() == 1u);
            if (stage_input.pieces().size() == 1u &&
                debug_watch.pieces().size() == 1u) {
                // Both descriptors reference the exact same logical atom and
                // physical slot; graph materialization only creates views.
                expect(stage_input.pieces()[0u].frame_value_index ==
                       debug_watch.pieces()[0u].frame_value_index);
                expect(stage_input.pieces()[0u].field_index ==
                       debug_watch.pieces()[0u].field_index);
            }
        }
        expect(cfg.frame_values.size() == 2u);
        expect(cfg.frame_slots.size() == 2u);
        if (boundary.bindings.size() == 3u) {
            CoroFrameDesc desc;
            for (auto &slot : cfg.frame_slots) {
                desc.add_field(slot.name, slot.type);
            }
            auto &stage_input = boundary.bindings[0u];
            auto &stage_output = boundary.bindings[1u];
            auto &debug_watch = boundary.bindings[2u];
            Kernel1D access_kernel = [&]() noexcept {
                auto frame = CoroFrame::create(&desc);
                auto input_user_field =
                    stage_input.pieces()[0u].field_index -
                    CoroFrameDesc::reserved_field_count;
                auto input_field =
                    frame.get<uint64_t>(input_user_field);
                input_field = uint64_t{17u};
                auto input_snapshot =
                    stage_input.read<uint64_t>(frame);
                auto watched = debug_watch.read<uint64_t>(frame);
                stage_output.write<float>(frame, Expr<float>{0.75f});
                static_cast<void>(input_snapshot);
                static_cast<void>(watched);
            };
            expect(access_kernel.function() != nullptr);
            expect(access_kernel.function()->hash() != 0u);
        }
    };

    "slot_access_projects_aggregate_lvalue_relative_to_binding"_test = [] {
        Module m;
        auto *state_type = Type::structure(
            {Type::of<float>(), Type::of<float3>()});
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume = kernel->create_basic_block();
        XIRBuilder b;
        uint32_t member_index = 1u;
        auto *member = m.create_constant(
            Type::of<uint32_t>(), &member_index);
        b.set_insertion_point(entry);
        auto *state = b.alloca_local(state_type);
        state->set_name("aggregate_stage_state");
        auto *output = b.gep(Type::of<float3>(), state, {member});
        luisa::vector<CoroSuspendExtensionPtr> extensions;
        extensions.emplace_back(make_coro_suspend_extension_data(
            "com.example.aggregate-stage", 1u,
            CoroSuspendFallback::reject,
            {{.name = "output",
              .access = CoroSuspendBindingAccess::write,
              .lifetime = CoroSuspendBindingLifetime::resumed,
              .index = 0u}},
            {}));
        luisa::vector<Value *> binding_values{output};
        b.coro_suspend(
            7u, "aggregate-stage", nullptr, {}, {},
            std::move(extensions), luisa::span{binding_values});
        b.set_insertion_point(resume);
        b.coro_resume(7u, nullptr);
        auto *resumed_output =
            b.gep(Type::of<float3>(), state, {member});
        static_cast<void>(b.load(Type::of<float3>(), resumed_output));
        b.return_void();

        auto cfg = coro_cfg_distill_pass_run_on_function(kernel);
        expect(cfg.succeeded());
        expect(cfg.transition_edges.size() == 1u);
        if (cfg.transition_edges.size() == 1u) {
            expect(cfg.transition_edges[0u]
                       .extension_binding_access_chains ==
                   luisa::vector<luisa::vector<uint32_t>>{{1u}});
        }
        auto split = coro_split_pass_run_on_module_with_cfg_and_frame_info(
            &m, cfg, nullptr);
        auto info = coro_materialize_pass_run_on_module_with_cfg(
            &m, cfg, split);
        auto graph = CoroGraph::from_module(m, info, cfg, split);

        expect(graph.boundary_count() == 1u);
        if (graph.boundary_count() != 1u ||
            graph.boundary(0u).bindings.size() != 1u) {
            return;
        }
        auto &access = graph.boundary(0u).bindings[0u];
        expect(access.type() == Type::of<float3>());
        expect(access.writable());
        expect(!access.readable());
        expect(access.pieces().size() == 3u);
        if (access.pieces().size() == 3u) {
            luisa::vector<luisa::vector<uint32_t>> paths;
            for (auto &piece : access.pieces()) {
                paths.emplace_back(piece.access_chain);
                expect(piece.logical_type == Type::of<float>());
            }
            std::sort(paths.begin(), paths.end());
            expect(paths ==
                   luisa::vector<luisa::vector<uint32_t>>{
                       {0u}, {1u}, {2u}});
        }

        CoroFrameDesc desc;
        for (auto &slot : cfg.frame_slots) {
            desc.add_field(slot.name, slot.type);
        }
        Kernel1D write_kernel = [&]() noexcept {
            auto frame = CoroFrame::create(&desc);
            access.write<float3>(
                frame, Expr<float3>{float3{1.f, 2.f, 3.f}});
        };
        expect(write_kernel.function() != nullptr);
        expect(write_kernel.function()->hash() != 0u);
    };

    "slot_access_preserves_packed_boolean_neighbors"_test = [] {
        Module m;
        BasicBlock *entry;
        auto *kernel = make_kernel_with_body(m, entry);
        auto *resume = kernel->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(entry);
        auto *first = b.alloca_local(Type::of<bool>());
        auto *second = b.alloca_local(Type::of<bool>());
        b.store(first, m.create_constant_one(Type::of<bool>()));
        b.store(second, m.create_constant_zero(Type::of<bool>()));
        luisa::vector<CoroSuspendExtensionPtr> extensions;
        extensions.emplace_back(make_coro_suspend_extension_data(
            "luisa.coro.debug.watch-edit", 1u,
            CoroSuspendFallback::reject,
            {{.name = "first",
              .access = CoroSuspendBindingAccess::read_write,
              .lifetime = CoroSuspendBindingLifetime::resumed,
              .index = 0u},
             {.name = "second",
              .access = CoroSuspendBindingAccess::read_write,
              .lifetime = CoroSuspendBindingLifetime::resumed,
              .index = 1u}},
            {}));
        luisa::vector<Value *> binding_values{first, second};
        b.coro_suspend(
            9u, "packed-bools", nullptr, {}, {},
            std::move(extensions), luisa::span{binding_values});
        b.set_insertion_point(resume);
        b.coro_resume(9u, nullptr);
        static_cast<void>(b.load(Type::of<bool>(), first));
        static_cast<void>(b.load(Type::of<bool>(), second));
        b.return_void();

        auto cfg = coro_cfg_distill_pass_run_on_function(kernel);
        expect(cfg.succeeded());
        expect(cfg.frame_values.size() == 2u);
        expect(cfg.frame_slots.size() == 1u);
        auto split = coro_split_pass_run_on_module_with_cfg_and_frame_info(
            &m, cfg, nullptr);
        auto info = coro_materialize_pass_run_on_module_with_cfg(
            &m, cfg, split);
        auto graph = CoroGraph::from_module(m, info, cfg, split);

        expect(graph.boundary_count() == 1u);
        if (graph.boundary_count() != 1u ||
            graph.boundary(0u).bindings.size() != 2u) {
            return;
        }
        auto &first_access = graph.boundary(0u).bindings[0u];
        auto &second_access = graph.boundary(0u).bindings[1u];
        expect(first_access.pieces().size() == 1u);
        expect(second_access.pieces().size() == 1u);
        if (first_access.pieces().size() == 1u &&
            second_access.pieces().size() == 1u) {
            auto &a = first_access.pieces()[0u];
            auto &b_piece = second_access.pieces()[0u];
            expect(a.field_index == b_piece.field_index);
            expect(a.physical_type == Type::of<uint>());
            expect(b_piece.physical_type == Type::of<uint>());
            expect(a.bit_offset.has_value());
            expect(b_piece.bit_offset.has_value());
            if (a.bit_offset && b_piece.bit_offset) {
                expect(*a.bit_offset != *b_piece.bit_offset);
            }
        }

        CoroFrameDesc desc;
        for (auto &slot : cfg.frame_slots) {
            desc.add_field(slot.name, slot.type);
        }
        Kernel1D packed_kernel = [&]() noexcept {
            auto frame = CoroFrame::create(&desc);
            first_access.write<bool>(frame, Expr<bool>{true});
            second_access.write<bool>(frame, Expr<bool>{false});
            auto first_value = first_access.read<bool>(frame);
            auto second_value = second_access.read<bool>(frame);
            static_cast<void>(first_value);
            static_cast<void>(second_value);
        };
        expect(packed_kernel.function() != nullptr);
        expect(packed_kernel.function()->hash() != 0u);
    };

    "three_suspends_four_nodes"_test = [] {
        // given: a kernel with three suspend points (linear chain)
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());

        // suspend 1
        auto *s1 = k->create_basic_block();
        auto *r1 = k->create_basic_block();
        b.set_insertion_point(body);
        b.cond_br(cond, s1, r1);
        b.set_insertion_point(s1);
        b.coro_suspend(10u, "alpha", nullptr);
        b.set_insertion_point(r1);
        b.coro_resume(10u, nullptr);

        // suspend 2
        auto *s2 = k->create_basic_block();
        auto *r2 = k->create_basic_block();
        b.cond_br(cond, s2, r2);
        b.set_insertion_point(s2);
        b.coro_suspend(20u, "beta", nullptr);
        b.set_insertion_point(r2);
        b.coro_resume(20u, nullptr);

        // suspend 3
        auto *s3 = k->create_basic_block();
        auto *r3 = k->create_basic_block();
        b.cond_br(cond, s3, r3);
        b.set_insertion_point(s3);
        b.coro_suspend(30u, "gamma", nullptr);
        b.set_insertion_point(r3);
        b.coro_resume(30u, nullptr);
        b.return_void();

        // Run passes
        auto cfg = coro_cfg_distill_pass_run_on_function(k);
        auto split = coro_split_pass_run_on_module_with_cfg_and_frame_info(&m, cfg, nullptr);
        auto info = coro_materialize_pass_run_on_module_with_cfg(&m, cfg, split);

        // when
        auto graph = CoroGraph::from_module(m, info, cfg, split);

        // then: 4 nodes, 3 edges
        expect(graph.node_count() == 4u);
        expect(graph.edge_count() == 3u);

        // node 0: entry
        auto &n0 = graph.node(0u);
        expect(n0.index == 0u);
        expect(n0.token == 0u);
        expect(!n0.is_terminal);

        // node 1: alpha
        auto &n1 = graph.node(1u);
        expect(n1.index == 1u);
        expect(n1.token == 10u);
        expect(n1.name == "alpha");
        expect(!n1.is_terminal);

        // node 2: beta
        auto &n2 = graph.node(2u);
        expect(n2.index == 2u);
        expect(n2.token == 20u);
        expect(n2.name == "beta");
        expect(!n2.is_terminal);

        // node 3: gamma (last, non-terminal because no CoroTerminateInst)
        auto &n3 = graph.node(3u);
        expect(n3.index == 3u);
        expect(n3.token == 30u);
        expect(n3.name == "gamma");
        expect(!n3.is_terminal);

        // Verify all edges exist
        expect(graph.edge(0u, 1u) != nullptr);
        expect(graph.edge(1u, 2u) != nullptr);
        expect(graph.edge(2u, 3u) != nullptr);

        // Name lookup for all
        expect(graph.node_by_name("alpha") != nullptr);
        expect(graph.node_by_name("beta") != nullptr);
        expect(graph.node_by_name("gamma") != nullptr);
        expect(graph.node_by_name("nonexistent") == nullptr);

        // Token lookup for all
        expect(graph.node_by_token(0u) != nullptr);
        expect(graph.node_by_token(10u) != nullptr);
        expect(graph.node_by_token(20u) != nullptr);
        expect(graph.node_by_token(30u) != nullptr);
        expect(graph.node_by_token(999u) == nullptr);
    };

    "relocation_payload_uses_distilled_live_begin"_test = [] {
        // `late` is resident at the first continuation but is not evaluated
        // there. Immediate callable inputs may therefore omit its physical
        // field, while relocation must retain it until the second resume.
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        auto *first_resume = k->create_basic_block();
        auto *second_resume = k->create_basic_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *early = b.clock();
        auto *late = b.clock();
        b.coro_suspend(301u, "early", nullptr);
        b.set_insertion_point(first_resume);
        b.coro_resume(301u, nullptr);
        static_cast<void>(b.call(
            Type::of<uint64_t>(), ArithmeticOp::BINARY_ADD,
            {early, m.create_constant_one(Type::of<uint64_t>())}));
        b.coro_suspend(302u, "late", nullptr);
        b.set_insertion_point(second_resume);
        b.coro_resume(302u, nullptr);
        static_cast<void>(b.call(
            Type::of<uint64_t>(), ArithmeticOp::BINARY_ADD,
            {late, m.create_constant_one(Type::of<uint64_t>())}));
        b.return_void();

        auto cfg = coro_cfg_distill_pass_run_on_function(k);
        auto split = coro_split_pass_run_on_module_with_cfg_and_frame_info(
            &m, cfg, nullptr);
        auto info = coro_materialize_pass_run_on_module_with_cfg(
            &m, cfg, split);
        auto graph = CoroGraph::from_module(m, info, cfg, split);

        auto *early_node = graph.node_by_name("early");
        auto *late_node = graph.node_by_name("late");
        expect(early_node != nullptr);
        expect(late_node != nullptr);
        if (early_node != nullptr && late_node != nullptr) {
            auto late_only = std::find_if(
                late_node->input_fields.begin(),
                late_node->input_fields.end(),
                [&](auto field) noexcept {
                    return std::find(
                               early_node->input_fields.begin(),
                               early_node->input_fields.end(), field) ==
                           early_node->input_fields.end();
                });
            expect(late_only != late_node->input_fields.end());
            if (late_only != late_node->input_fields.end()) {
                expect(std::find(
                           early_node->relocation_fields.begin(),
                           early_node->relocation_fields.end(),
                           *late_only) !=
                       early_node->relocation_fields.end())
                    << "live-through state must be projected from the "
                       "distilled liveness certificate";
            }
        }
    };

    "terminal_scope_has_terminal_flag"_test = [] {
        // given: a kernel that ends with CoroTerminateInst
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());

        auto *suspend_bb = k->create_basic_block();
        auto *resume_bb = k->create_basic_block();

        b.set_insertion_point(body);
        b.cond_br(cond, suspend_bb, resume_bb);

        b.set_insertion_point(suspend_bb);
        b.coro_suspend(1u, "middle", nullptr);

        b.set_insertion_point(resume_bb);
        b.coro_resume(1u, nullptr);

        // terminal block
        auto *term_bb = k->create_basic_block();
        b.br(term_bb);

        b.set_insertion_point(term_bb);
        b.coro_terminate();

        // Run passes
        auto cfg = coro_cfg_distill_pass_run_on_function(k);
        auto split = coro_split_pass_run_on_module_with_cfg_and_frame_info(&m, cfg, nullptr);
        auto info = coro_materialize_pass_run_on_module_with_cfg(&m, cfg, split);

        // when
        auto graph = CoroGraph::from_module(m, info, cfg, split);

        // then: last scope is terminal
        expect(graph.node_count() >= 1u);
        auto &last = graph.node(graph.node_count() - 1u);
        expect(last.is_terminal);

        // entry is not terminal
        auto &entry = graph.node(0u);
        expect(!entry.is_terminal);
    };

    "named_token_lookup_by_string_name"_test = [] {
        // given: a kernel with two suspends, each with distinct names
        Module m;
        BasicBlock *body;
        auto *k = make_kernel_with_body(m, body);
        XIRBuilder b;
        auto *cond = m.create_constant_one(Type::of<bool>());

        // suspend "first_half"
        auto *s1 = k->create_basic_block();
        auto *r1 = k->create_basic_block();
        b.set_insertion_point(body);
        b.cond_br(cond, s1, r1);
        b.set_insertion_point(s1);
        b.coro_suspend(100u, "first_half", nullptr);
        b.set_insertion_point(r1);
        b.coro_resume(100u, nullptr);

        // suspend "second_half"
        auto *s2 = k->create_basic_block();
        auto *r2 = k->create_basic_block();
        b.cond_br(cond, s2, r2);
        b.set_insertion_point(s2);
        b.coro_suspend(200u, "second_half", nullptr);
        b.set_insertion_point(r2);
        b.coro_resume(200u, nullptr);
        b.return_void();

        // Run passes
        auto cfg = coro_cfg_distill_pass_run_on_function(k);
        auto split = coro_split_pass_run_on_module_with_cfg_and_frame_info(&m, cfg, nullptr);
        auto info = coro_materialize_pass_run_on_module_with_cfg(&m, cfg, split);

        // when
        auto graph = CoroGraph::from_module(m, info, cfg, split);

        // then: find nodes by their suspend names
        auto *first = graph.node_by_name("first_half");
        expect(first != nullptr);
        expect(first->token == 100u);
        expect(first->name == "first_half");

        auto *second = graph.node_by_name("second_half");
        expect(second != nullptr);
        expect(second->token == 200u);
        expect(second->name == "second_half");

        // Entry node should not be findable by name (empty name)
        // node_by_name with empty string should not return entry
        auto *entry = graph.node_by_name("");
        expect(entry == nullptr);

        // Nonexistent name returns nullptr
        auto *missing = graph.node_by_name("third_half");
        expect(missing == nullptr);
    };
}

int main(int /*argc*/, char * /*argv*/[]) {
    reg_coro_graph();
    return 0;
}
