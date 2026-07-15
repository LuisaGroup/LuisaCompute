#include "ut/ut.hpp"

#include <luisa/ast/type.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/passes/alias_analysis.h>

using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] Constant *uint_constant(Module &m, uint32_t value) noexcept {
    return m.create_constant(Type::of<uint>(), &value);
}

}// namespace

int main() {

    "alias_nested_gep_offsets_are_not_comparable"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *inner = Type::array(Type::of<float>(), 2u);
        auto *outer = Type::array(inner, 2u);
        auto *base = b.alloca_local(outer);
        auto *row = b.gep(inner, base, {uint_constant(m, 0u)});
        auto *element = b.gep(Type::of<float>(), row, {uint_constant(m, 1u)});
        auto *whole_row_load = b.load(inner, row);
        auto *element_store = b.store(element, m.create_constant_zero(Type::of<float>()));
        b.return_void();

        expect(alias_analysis_query(whole_row_load, element_store) == AliasResult::MayAlias);
    };

    "alias_direct_sibling_geps_are_disjoint"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *inner = Type::array(Type::of<float>(), 2u);
        auto *base = b.alloca_local(Type::array(inner, 2u));
        auto *row0 = b.gep(inner, base, {uint_constant(m, 0u)});
        auto *row1 = b.gep(inner, base, {uint_constant(m, 1u)});
        auto *load0 = b.load(inner, row0);
        auto *load1 = b.load(inner, row1);
        b.return_void();

        expect(alias_analysis_query(load0, load1) == AliasResult::NoAlias);
    };

    "alias_same_local_pointer_is_must_alias"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *local = b.alloca_local(Type::of<int>());
        auto *load = b.load(Type::of<int>(), local);
        auto *store = b.store(local, m.create_constant_zero(Type::of<int>()));
        b.return_void();

        expect(alias_analysis_query(load, store) == AliasResult::MustAlias);
    };

    "alias_distinct_resource_arguments_may_alias"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *a = k->create_resource_argument(Type::buffer(Type::of<int>()));
        auto *b_arg = k->create_resource_argument(Type::buffer(Type::of<int>()));
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *index = uint_constant(m, 0u);
        auto *read_a = b.call(Type::of<int>(), ResourceReadOp::BUFFER_READ, {a, index});
        auto *read_b = b.call(Type::of<int>(), ResourceReadOp::BUFFER_READ, {b_arg, index});
        b.return_void();

        expect(alias_analysis_query(read_a, read_b) == AliasResult::MayAlias);
    };

    "alias_query_observes_gep_retargeting"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *array_type = Type::array(Type::of<int>(), 2u);
        auto *a = b.alloca_local(array_type);
        auto *b_local = b.alloca_local(array_type);
        auto *index = uint_constant(m, 0u);
        auto *p = b.gep(Type::of<int>(), a, {index});
        auto *q = b.gep(Type::of<int>(), b_local, {index});
        auto *store = b.store(p, m.create_constant_zero(Type::of<int>()));
        auto *load = b.load(Type::of<int>(), q);
        b.return_void();

        static_cast<void>(alias_analysis_pass_run_on_function(k));
        expect(alias_analysis_query(store, load) == AliasResult::NoAlias);
        p->set_operand(0u, b_local);
        expect(alias_analysis_query(store, load) == AliasResult::MayAlias);
    };
}
