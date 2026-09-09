#include "ut/ut.hpp"

#include <algorithm>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/module.h>
#include <luisa/xir/verifier.h>

#include "coro_scalar_relation_liveness.h"
#include "coro_semantic_graph.h"

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {
bool contains(span<xir::Value *const> values, xir::Value *value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}
}// namespace

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    "a_live_overlapping_use_and_definition_is_not_retired"_test = [] {
        xir::Module module;
        auto *kernel = module.create_kernel();
        auto *entry = kernel->create_body_block();
        xir::XIRBuilder b;
        b.set_insertion_point(entry);
        auto *slot = b.alloca_local(Type::of<bool>());
        auto *update = b.store(slot, module.create_constant_one(Type::of<bool>()));
        auto *read = b.load(Type::of<bool>(), slot);
        b.return_void();
        expect(xir::xir_verify_module(&module).succeeded());
        xir::detail::CoroSemanticGraph graph{kernel};
        expect(graph.valid());
        vector<uint8_t> active(graph.block_count(), 1u);
        vector<xir::Value *> predicates{slot};
        // Conservative semantic annotations may both observe and redefine a
        // root. Adding an old-value use is safe even for this simple store.
        xir::detail::CoroBooleanSemanticValues uses{{update, {slot}}, {read, {slot}}};
        xir::detail::CoroBooleanSemanticValues definitions{{update, {slot}}};
        xir::detail::CoroBooleanPredicateLiveness live{
            graph, active, graph.block_count(), predicates, uses, definitions};
        expect(!contains(live.dead_after(update), slot));
        expect(contains(live.dead_after(read), slot));
    };
    "a_never_used_definition_is_retired_after_its_transfer"_test = [] {
        xir::Module module;
        auto *kernel = module.create_kernel();
        auto *entry = kernel->create_body_block();
        xir::XIRBuilder b;
        b.set_insertion_point(entry);
        auto *slot = b.alloca_local(Type::of<bool>());
        auto *definition = b.store(slot, module.create_constant_one(Type::of<bool>()));
        b.return_void();
        expect(xir::xir_verify_module(&module).succeeded());
        xir::detail::CoroSemanticGraph graph{kernel};
        expect(graph.valid());
        vector<uint8_t> active(graph.block_count(), 1u);
        vector<xir::Value *> predicates{slot};
        xir::detail::CoroBooleanSemanticValues uses;
        xir::detail::CoroBooleanSemanticValues definitions{{definition, {slot}}};
        xir::detail::CoroBooleanPredicateLiveness live{
            graph, active, graph.block_count(), predicates, uses, definitions};
        expect(contains(live.dead_after(definition), slot));
        expect(live.live_in(graph.block_id(entry)).empty());
    };
    "a_raw_successor_use_is_live_even_when_its_condition_is_false"_test = [] {
        xir::Module module;
        auto *kernel = module.create_kernel();
        auto *entry = kernel->create_body_block();
        auto *selected = kernel->create_basic_block();
        auto *raw_only = kernel->create_basic_block();
        xir::XIRBuilder b;
        b.set_insertion_point(entry);
        auto *slot = b.alloca_local(Type::of<bool>());
        auto *definition = b.store(slot, module.create_constant_one(Type::of<bool>()));
        b.cond_br(module.create_constant_one(Type::of<bool>()), selected, raw_only);
        b.set_insertion_point(selected);
        b.return_void();
        b.set_insertion_point(raw_only);
        auto *read = b.load(Type::of<bool>(), slot);
        b.return_void();
        expect(xir::xir_verify_module(&module).succeeded());
        xir::detail::CoroSemanticGraph graph{kernel};
        expect(graph.valid());
        expect(graph.successors(graph.block_id(entry)).size() == 2u);
        vector<uint8_t> active(graph.block_count(), 1u);
        vector<xir::Value *> predicates{slot};
        xir::detail::CoroBooleanSemanticValues uses{{read, {slot}}};
        xir::detail::CoroBooleanSemanticValues definitions{{definition, {slot}}};
        xir::detail::CoroBooleanPredicateLiveness live{
            graph, active, graph.block_count(), predicates, uses, definitions};
        expect(!contains(live.dead_after(definition), slot));
        expect(contains(live.live_in(graph.block_id(raw_only)), slot));
        expect(contains(live.dead_after(read), slot));
    };
}
