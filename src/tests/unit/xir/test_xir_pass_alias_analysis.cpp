// Test for XIR alias classification and conservative invalid-input handling.

#include "ut/ut.hpp"

#include <luisa/ast/type.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/passes/alias_analysis.h>
#include <luisa/xir/verifier.h>

#include "../../../xir/passes/helpers.h"

#include <array>

using namespace luisa;
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

    "alias_direct_sibling_geps_decode_mixed_integer_widths"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *base = b.alloca_local(Type::array(Type::of<float>(), 2u));
        int8_t zero_value = 0;
        uint64_t one_value = 1u;
        auto *zero = m.create_constant(Type::of<int8_t>(), &zero_value);
        auto *one = m.create_constant(Type::of<uint64_t>(), &one_value);
        auto *element0 = b.gep(Type::of<float>(), base, {zero});
        auto *element1 = b.gep(Type::of<float>(), base, {one});
        auto *load0 = b.load(Type::of<float>(), element0);
        auto *load1 = b.load(Type::of<float>(), element1);
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

    "alias_overlapping_byte_buffer_ranges_may_alias"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *buffer = k->create_resource_argument(Type::buffer(Type::of<uint>()));
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *read0 = b.call(Type::of<float>(), ResourceReadOp::BYTE_BUFFER_READ,
                             {buffer, uint_constant(m, 0u)});
        auto *read1 = b.call(Type::of<float>(), ResourceReadOp::BYTE_BUFFER_READ,
                             {buffer, uint_constant(m, 1u)});
        b.return_void();

        expect(alias_analysis_query(read0, read1) == AliasResult::MayAlias);
    };

    "alias_distinct_bindless_slots_may_reference_same_resource"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *bindless = k->create_resource_argument(Type::from("bindless_array"));
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *element = uint_constant(m, 0u);
        auto *read0 = b.call(Type::of<int>(), ResourceReadOp::BINDLESS_BUFFER_READ,
                             {bindless, uint_constant(m, 0u), element});
        auto *read1 = b.call(Type::of<int>(), ResourceReadOp::BINDLESS_BUFFER_READ,
                             {bindless, uint_constant(m, 1u), element});
        b.return_void();

        expect(alias_analysis_query(read0, read1) == AliasResult::MayAlias);
    };

    "alias_shared_load_and_atomic_are_classified_in_shared_scope"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *shared = b.alloca_shared(Type::array(Type::of<int>(), 2u));
        auto *zero = uint_constant(m, 0u);
        auto *one = uint_constant(m, 1u);
        auto *p0 = b.gep(Type::of<int>(), shared, {zero});
        auto *p1 = b.gep(Type::of<int>(), shared, {one});
        auto *load = b.load(Type::of<int>(), p0);
        auto *atomic_same = b.atomic_fetch_add(
            Type::of<int>(), p0, {}, m.create_constant_one(Type::of<int>()));
        auto *atomic_other = b.atomic_fetch_add(
            Type::of<int>(), p1, {}, m.create_constant_one(Type::of<int>()));
        b.return_void();

        auto load_memory = get_memory_info(load);
        auto atomic_memory = get_memory_info(atomic_same);
        expect(load_memory.scope == MemoryScope::SHARED);
        expect(load_memory.effects == MemoryEffects::READ);
        expect(atomic_memory.scope == MemoryScope::SHARED);
        expect(atomic_memory.effects == MemoryEffects::READ_WRITE);
        expect(alias_analysis_query(load, atomic_same) == AliasResult::MustAlias);
        expect(alias_analysis_query(load, atomic_other) == AliasResult::NoAlias);
    };

    "resource_query_memory_effects_distinguish_stable_and_mutable_state"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *buffer =
            k->create_resource_argument(Type::buffer(Type::of<float>()));
        auto *accel = k->create_resource_argument(Type::of<Accel>());
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *instance = uint_constant(m, 0u);
        auto *new_user_id = uint_constant(m, 1u);
        auto *size = b.call(Type::of<uint>(), ResourceQueryOp::BUFFER_SIZE,
                            {buffer});
        auto *user_id = b.call(
            Type::of<uint>(), ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID,
            {accel, instance});
        auto *write = b.call(
            ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID,
            {accel, instance, new_user_id});
        b.return_void();

        auto stable_info = get_memory_info(size);
        auto mutable_info = get_memory_info(user_id);
        expect(stable_info.scope == MemoryScope::GLOBAL);
        expect(stable_info.effects == MemoryEffects::NONE);
        expect(stable_info.is_safe_to_value_number());
        expect(mutable_info.scope == MemoryScope::GLOBAL);
        expect(mutable_info.effects == MemoryEffects::READ);
        expect(!mutable_info.is_safe_to_value_number());
        expect(mutable_info.is_removable_if_unused());
        expect(alias_analysis_query(user_id, write) == AliasResult::MayAlias);
    };

    "reference_argument_address_space_is_not_fabricated"_test = [] {
        Module m;
        auto *callable = m.create_callable(nullptr);
        auto *reference =
            callable->create_reference_argument(Type::of<int>());
        auto *body = callable->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *load = b.load(Type::of<int>(), reference);
        auto *atomic = b.atomic_fetch_add(
            Type::of<int>(), reference, {},
            m.create_constant_one(Type::of<int>()));
        b.return_void();

        auto load_info = get_memory_info(load);
        auto atomic_info = get_memory_info(atomic);
        expect(load_info.scope == MemoryScope::NONE);
        expect(load_info.effects == MemoryEffects::READ);
        expect(atomic_info.scope == MemoryScope::NONE);
        expect(atomic_info.effects == MemoryEffects::READ_WRITE);
        // The two operations even have the same base, but without a declared
        // reference address space the conservative public answer is MayAlias.
        expect(alias_analysis_query(load, atomic) ==
               AliasResult::MayAlias);
    };

    "alias_call_through_local_reference_is_may_alias"_test = [] {
        Module m;
        auto *callee = m.create_callable(nullptr);
        auto *reference =
            callee->create_reference_argument(Type::of<int>());
        auto *callee_body = callee->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        b.store(reference, m.create_constant_one(Type::of<int>()));
        b.return_void();

        auto *kernel = m.create_kernel();
        auto *body = kernel->create_body_block();
        b.set_insertion_point(body);
        auto *local = b.alloca_local(Type::of<int>());
        auto *load = b.load(Type::of<int>(), local);
        auto *call = b.call(nullptr, callee, {local});
        b.return_void();

        expect(alias_analysis_query(load, call) ==
               AliasResult::MayAlias);
        expect(alias_analysis_query(call, load) ==
               AliasResult::MayAlias);
    };

    "alias_call_through_shared_reference_is_may_alias"_test = [] {
        Module m;
        auto *callee = m.create_callable(nullptr);
        auto *reference =
            callee->create_reference_argument(Type::of<int>());
        auto *callee_body = callee->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(callee_body);
        b.store(reference, m.create_constant_one(Type::of<int>()));
        b.return_void();

        auto *kernel = m.create_kernel();
        auto *body = kernel->create_body_block();
        b.set_insertion_point(body);
        auto *shared = b.alloca_shared(Type::of<int>());
        auto *load = b.load(Type::of<int>(), shared);
        auto *call = b.call(nullptr, callee, {shared});
        b.return_void();

        expect(alias_analysis_query(load, call) ==
               AliasResult::MayAlias);
        expect(alias_analysis_query(call, load) ==
               AliasResult::MayAlias);
    };

    "alias_local_indexed_atomics_compare_full_index_paths"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *dynamic_a = k->create_value_argument(Type::of<uint>());
        auto *dynamic_b = k->create_value_argument(Type::of<uint>());
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *local = b.alloca_local(Type::array(Type::of<int>(), 2u));
        auto *zero = uint_constant(m, 0u);
        auto *one = uint_constant(m, 1u);
        auto *increment = m.create_constant_one(Type::of<int>());
        std::array<Value *, 1u> zero_index{zero};
        std::array<Value *, 1u> one_index{one};
        std::array<Value *, 1u> dynamic_a_index{dynamic_a};
        std::array<Value *, 1u> dynamic_b_index{dynamic_b};
        auto *atomic_zero_a = b.atomic_fetch_add(Type::of<int>(), local, zero_index, increment);
        auto *atomic_zero_b = b.atomic_fetch_add(Type::of<int>(), local, zero_index, increment);
        auto *atomic_one = b.atomic_fetch_add(Type::of<int>(), local, one_index, increment);
        auto *atomic_dynamic_a = b.atomic_fetch_add(Type::of<int>(), local, dynamic_a_index, increment);
        auto *atomic_dynamic_a_again = b.atomic_fetch_add(Type::of<int>(), local, dynamic_a_index, increment);
        auto *atomic_dynamic_b = b.atomic_fetch_add(Type::of<int>(), local, dynamic_b_index, increment);
        auto *whole_array = b.load(local->type(), local);
        b.return_void();

        expect(alias_analysis_query(atomic_zero_a, atomic_zero_b) == AliasResult::MustAlias);
        expect(alias_analysis_query(atomic_zero_a, atomic_one) == AliasResult::NoAlias);
        expect(alias_analysis_query(atomic_dynamic_a, atomic_dynamic_a_again) == AliasResult::MustAlias);
        expect(alias_analysis_query(atomic_dynamic_a, atomic_dynamic_b) == AliasResult::MayAlias);
        expect(alias_analysis_query(atomic_zero_a, whole_array) == AliasResult::MayAlias);
    };

    "alias_shared_indexed_atomics_compare_full_index_paths"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *dynamic = k->create_value_argument(Type::of<uint>());
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *shared = b.alloca_shared(Type::array(Type::of<int>(), 2u));
        auto *zero = uint_constant(m, 0u);
        auto *one = uint_constant(m, 1u);
        auto *increment = m.create_constant_one(Type::of<int>());
        std::array<Value *, 1u> zero_index{zero};
        std::array<Value *, 1u> one_index{one};
        std::array<Value *, 1u> dynamic_index{dynamic};
        auto *atomic_zero_a = b.atomic_fetch_add(Type::of<int>(), shared, zero_index, increment);
        auto *atomic_zero_b = b.atomic_fetch_add(Type::of<int>(), shared, zero_index, increment);
        auto *atomic_one = b.atomic_fetch_add(Type::of<int>(), shared, one_index, increment);
        auto *atomic_dynamic = b.atomic_fetch_add(Type::of<int>(), shared, dynamic_index, increment);
        b.return_void();

        expect(alias_analysis_query(atomic_zero_a, atomic_zero_b) == AliasResult::MustAlias);
        expect(alias_analysis_query(atomic_zero_a, atomic_one) == AliasResult::NoAlias);
        expect(alias_analysis_query(atomic_zero_a, atomic_dynamic) == AliasResult::MayAlias);
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

    "alias_indirect_dispatch_uses_only_record_offset_as_address"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *indirect = k->create_reference_argument(
            Type::custom("LC_IndirectDispatchBuffer"));
        auto *body = k->create_body_block();
        auto *offset0 = uint_constant(m, 0u);
        auto *offset1 = uint_constant(m, 1u);
        auto *block_one = m.create_constant_one(Type::of<uint3>());
        auto *block_zero = m.create_constant_zero(Type::of<uint3>());
        auto *dispatch_size = m.create_constant_one(Type::of<uint3>());
        auto *kernel_id = uint_constant(m, 0u);
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *same_address_a = b.call(
            ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL,
            {indirect, offset0, block_one, dispatch_size, kernel_id});
        auto *same_address_b = b.call(
            ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL,
            {indirect, offset0, block_zero, dispatch_size, kernel_id});
        auto *different_address = b.call(
            ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL,
            {indirect, offset1, block_one, dispatch_size, kernel_id});
        b.return_void();

        // block_size/dispatch_size/kernel_id are record payloads. A payload
        // difference cannot prove that two writes target disjoint storage.
        expect(alias_analysis_query(same_address_a, same_address_b) ==
               AliasResult::MayAlias);
        expect(alias_analysis_query(same_address_a, different_address) ==
               AliasResult::NoAlias);
        auto verification = xir_verify_module(&m);
        expect(verification.succeeded())
            << (verification.errors.empty() ?
                    "unknown XIR verification error" :
                    verification.errors.front().message.c_str());
    };
}
