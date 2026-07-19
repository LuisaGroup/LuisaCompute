#include "ut/ut.hpp"

#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/verifier.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/cast.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/passes/slp_vectorization.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;

namespace {

[[nodiscard]] Value *gep_array_element(XIRBuilder &b, Module &m, Value *array_alloca,
                                       int32_t index, const Type *elem_type) noexcept {
    auto *c = m.create_constant(Type::of<int>(), &index);
    return b.gep(elem_type, array_alloca, {c});
}

[[nodiscard]] Value *gep_array_element_u8(XIRBuilder &b, Module &m, Value *array_alloca,
                                          uint8_t index, const Type *elem_type) noexcept {
    auto *c = m.create_constant(Type::of<uint8_t>(), &index);
    return b.gep(elem_type, array_alloca, {c});
}

[[nodiscard]] Constant *float_const(Module &m, float v) noexcept {
    return m.create_constant(Type::of<float>(), &v);
}

[[nodiscard]] Constant *int_const(Module &m, int32_t v) noexcept {
    return m.create_constant(Type::of<int32_t>(), &v);
}

[[nodiscard]] bool is_lane_extract(Value *value, Value *vector, uint32_t lane) noexcept {
    if (value == nullptr || !value->isa<ArithmeticInst>()) { return false; }
    auto *extract = static_cast<ArithmeticInst *>(value);
    if (extract->op() != ArithmeticOp::EXTRACT ||
        extract->operand_count() != 2u || extract->operand(0) != vector) {
        return false;
    }
    auto *index = extract->operand(1);
    return index->isa<Constant>() && index->type()->is_uint32() &&
           static_cast<Constant *>(index)->as<uint32_t>() == lane;
}

[[nodiscard]] bool verify(Module &module) noexcept {
    return xir_verify_module(&module).succeeded();
}

}// namespace

void reg_slp_vectorization() {

    "slp_vectorization_vectorizes_arithmetic_and_preserves_scalar_memory"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);

        auto *arr = b.alloca_local(Type::array(Type::of<float>(), 4));
        auto *x = b.alloca_local(Type::of<float>());
        b.store(x, m.create_constant_zero(Type::of<float>()));
        auto *xv = b.load(Type::of<float>(), x);
        auto *c0 = float_const(m, 1.0f);
        auto *c1 = float_const(m, 2.0f);
        auto *c2 = float_const(m, 3.0f);
        auto *c3 = float_const(m, 4.0f);
        auto *v0 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {xv, c0});
        auto *v1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {xv, c1});
        auto *v2 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {xv, c2});
        auto *v3 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {xv, c3});
        auto *p0 = gep_array_element(b, m, arr, 0, Type::of<float>());
        auto *p1 = gep_array_element(b, m, arr, 1, Type::of<float>());
        auto *p2 = gep_array_element(b, m, arr, 2, Type::of<float>());
        auto *p3 = gep_array_element(b, m, arr, 3, Type::of<float>());
        auto *s0 = b.store(p0, v0);
        auto *s1 = b.store(p1, v1);
        auto *s2 = b.store(p2, v2);
        auto *s3 = b.store(p3, v3);
        b.return_void();

        expect(verify(m));
        auto info = slp_vectorization_pass_run_on_function(k);
        expect(info.vectorized_tree_count == 1u);
        expect(info.vectorized_inst_count == 4u);
        expect(info.rejected_candidate_count == 0u);
        expect(verify(m));

        ArithmeticInst *vector_add = nullptr;
        size_t scalar_add_count = 0u;
        size_t vector_store_count = 0u;
        size_t vector_gep_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>()) {
                auto *arithmetic = static_cast<ArithmeticInst *>(inst);
                if (arithmetic->op() == ArithmeticOp::BINARY_ADD &&
                    arithmetic->type() == Type::of<float>()) {
                    scalar_add_count++;
                }
                if (arithmetic->op() == ArithmeticOp::BINARY_ADD &&
                    arithmetic->type() == Type::of<float4>()) {
                    vector_add = arithmetic;
                }
            } else if (inst->isa<StoreInst>() &&
                       static_cast<StoreInst *>(inst)->value()->type() == Type::of<float4>()) {
                vector_store_count++;
            } else if (inst->isa<GEPInst>() && inst->type() == Type::of<float4>()) {
                vector_gep_count++;
            }
        });
        expect(vector_add != nullptr);
        expect(scalar_add_count == 0u);
        expect(vector_store_count == 0u);
        expect(vector_gep_count == 0u);
        expect(vector_add->operand_count() == 2u);
        expect(vector_add->operand(0)->isa<ArithmeticInst>());
        expect(vector_add->operand(1)->isa<ArithmeticInst>());
        auto *lhs = static_cast<ArithmeticInst *>(vector_add->operand(0));
        auto *rhs = static_cast<ArithmeticInst *>(vector_add->operand(1));
        expect(lhs->op() == ArithmeticOp::AGGREGATE);
        expect(rhs->op() == ArithmeticOp::AGGREGATE);
        expect(lhs->type() == Type::of<float4>());
        expect(rhs->type() == Type::of<float4>());
        expect(lhs->operand_count() == 4u);
        expect(rhs->operand_count() == 4u);
        expect(lhs->operand(0) == xv);
        expect(lhs->operand(1) == xv);
        expect(lhs->operand(2) == xv);
        expect(lhs->operand(3) == xv);
        expect(rhs->operand(0) == c0);
        expect(rhs->operand(1) == c1);
        expect(rhs->operand(2) == c2);
        expect(rhs->operand(3) == c3);
        expect(is_lane_extract(s0->value(), vector_add, 0u));
        expect(is_lane_extract(s1->value(), vector_add, 1u));
        expect(is_lane_extract(s2->value(), vector_add, 2u));
        expect(is_lane_extract(s3->value(), vector_add, 3u));
        expect(s0->variable() == p0);
        expect(s1->variable() == p1);
        expect(s2->variable() == p2);
        expect(s3->variable() == p3);
    };

    "slp_vectorization_vectorizes_casts_with_lane_order"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);

        auto *arr = b.alloca_local(Type::array(Type::of<float>(), 4));
        auto *c0 = int_const(m, 5);
        auto *c1 = int_const(m, 6);
        auto *c2 = int_const(m, 7);
        auto *c3 = int_const(m, 8);
        auto *v0 = b.static_cast_(Type::of<float>(), c0);
        auto *v1 = b.static_cast_(Type::of<float>(), c1);
        auto *v2 = b.static_cast_(Type::of<float>(), c2);
        auto *v3 = b.static_cast_(Type::of<float>(), c3);
        auto *p0 = gep_array_element(b, m, arr, 0, Type::of<float>());
        auto *p1 = gep_array_element(b, m, arr, 1, Type::of<float>());
        auto *p2 = gep_array_element(b, m, arr, 2, Type::of<float>());
        auto *p3 = gep_array_element(b, m, arr, 3, Type::of<float>());
        auto *s0 = b.store(p0, v0);
        auto *s1 = b.store(p1, v1);
        auto *s2 = b.store(p2, v2);
        auto *s3 = b.store(p3, v3);
        b.return_void();

        expect(verify(m));
        auto info = slp_vectorization_pass_run_on_function(k);
        expect(info.vectorized_tree_count == 1u);
        expect(info.vectorized_inst_count == 4u);
        expect(info.rejected_candidate_count == 0u);
        expect(verify(m));

        CastInst *vector_cast = nullptr;
        size_t scalar_cast_count = 0u;
        size_t vector_store_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<CastInst>()) {
                if (inst->type() == Type::of<float>()) { scalar_cast_count++; }
                if (inst->type() == Type::of<float4>()) {
                    vector_cast = static_cast<CastInst *>(inst);
                }
            } else if (inst->isa<StoreInst>() &&
                       static_cast<StoreInst *>(inst)->value()->type() == Type::of<float4>()) {
                vector_store_count++;
            }
        });
        expect(vector_cast != nullptr);
        expect(vector_cast->op() == CastOp::STATIC_CAST);
        expect(scalar_cast_count == 0u);
        expect(vector_store_count == 0u);
        expect(vector_cast->value()->isa<ArithmeticInst>());
        auto *gather = static_cast<ArithmeticInst *>(vector_cast->value());
        expect(gather->op() == ArithmeticOp::AGGREGATE);
        expect(gather->type() == Type::of<int4>());
        expect(gather->operand_count() == 4u);
        expect(gather->operand(0) == c0);
        expect(gather->operand(1) == c1);
        expect(gather->operand(2) == c2);
        expect(gather->operand(3) == c3);
        expect(is_lane_extract(s0->value(), vector_cast, 0u));
        expect(is_lane_extract(s1->value(), vector_cast, 1u));
        expect(is_lane_extract(s2->value(), vector_cast, 2u));
        expect(is_lane_extract(s3->value(), vector_cast, 3u));
    };

    "slp_vectorization_accepts_narrow_unsigned_gep_indices"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);

        auto *arr = b.alloca_local(Type::array(Type::of<int32_t>(), 4));
        auto *v0 = b.call(Type::of<int32_t>(), ArithmeticOp::UNARY_MINUS, {int_const(m, 1)});
        auto *v1 = b.call(Type::of<int32_t>(), ArithmeticOp::UNARY_MINUS, {int_const(m, 2)});
        auto *v2 = b.call(Type::of<int32_t>(), ArithmeticOp::UNARY_MINUS, {int_const(m, 3)});
        auto *v3 = b.call(Type::of<int32_t>(), ArithmeticOp::UNARY_MINUS, {int_const(m, 4)});
        auto *p0 = gep_array_element_u8(b, m, arr, 0u, Type::of<int32_t>());
        auto *p1 = gep_array_element_u8(b, m, arr, 1u, Type::of<int32_t>());
        auto *p2 = gep_array_element_u8(b, m, arr, 2u, Type::of<int32_t>());
        auto *p3 = gep_array_element_u8(b, m, arr, 3u, Type::of<int32_t>());
        auto *s0 = b.store(p0, v0);
        auto *s1 = b.store(p1, v1);
        auto *s2 = b.store(p2, v2);
        auto *s3 = b.store(p3, v3);
        b.return_void();

        expect(verify(m));
        auto info = slp_vectorization_pass_run_on_function(k);
        expect(info.vectorized_tree_count == 1u);
        expect(info.vectorized_inst_count == 4u);
        expect(info.rejected_candidate_count == 0u);
        expect(verify(m));

        ArithmeticInst *vector_negate = nullptr;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->isa<ArithmeticInst>()) {
                auto *arithmetic = static_cast<ArithmeticInst *>(inst);
                if (arithmetic->op() == ArithmeticOp::UNARY_MINUS &&
                    arithmetic->type() == Type::of<int4>()) {
                    vector_negate = arithmetic;
                }
            }
        });
        expect(vector_negate != nullptr);
        expect(is_lane_extract(s0->value(), vector_negate, 0u));
        expect(is_lane_extract(s1->value(), vector_negate, 1u));
        expect(is_lane_extract(s2->value(), vector_negate, 2u));
        expect(is_lane_extract(s3->value(), vector_negate, 3u));
    };

    "slp_vectorization_rejects_mixed_producers_without_mutation"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);

        auto *arr = b.alloca_local(Type::array(Type::of<float>(), 4));
        auto *x = b.alloca_local(Type::of<float>());
        b.store(x, m.create_constant_zero(Type::of<float>()));
        auto *xv = b.load(Type::of<float>(), x);
        auto *v0 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {xv, float_const(m, 1.0f)});
        auto *v1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_SUB, {xv, float_const(m, 2.0f)});
        auto *v2 = b.call(Type::of<float>(), ArithmeticOp::BINARY_MUL, {xv, float_const(m, 3.0f)});
        auto *v3 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {xv, float_const(m, 4.0f)});
        auto *p0 = gep_array_element(b, m, arr, 0, Type::of<float>());
        auto *p1 = gep_array_element(b, m, arr, 1, Type::of<float>());
        auto *p2 = gep_array_element(b, m, arr, 2, Type::of<float>());
        auto *p3 = gep_array_element(b, m, arr, 3, Type::of<float>());
        auto *s0 = b.store(p0, v0);
        auto *s1 = b.store(p1, v1);
        auto *s2 = b.store(p2, v2);
        auto *s3 = b.store(p3, v3);
        b.return_void();

        auto info = slp_vectorization_pass_run_on_function(k);
        expect(info.vectorized_tree_count == 0u);
        expect(info.vectorized_inst_count == 0u);
        expect(info.rejected_candidate_count == 1u);
        expect(verify(m));
        expect(s0->value() == v0);
        expect(s1->value() == v1);
        expect(s2->value() == v2);
        expect(s3->value() == v3);
        size_t vector_instruction_count = 0u;
        k->traverse_instructions([&](Instruction *inst) noexcept {
            if (inst->type() != nullptr && inst->type()->is_vector()) {
                vector_instruction_count++;
            }
        });
        expect(vector_instruction_count == 0u);
    };

    "slp_vectorization_rejects_multi_use_producers"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *arr = b.alloca_local(Type::array(Type::of<float>(), 4));
        auto *zero = float_const(m, 0.0f);
        auto *v0 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {zero, float_const(m, 1.0f)});
        auto *v1 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {zero, float_const(m, 2.0f)});
        auto *v2 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {zero, float_const(m, 3.0f)});
        auto *v3 = b.call(Type::of<float>(), ArithmeticOp::BINARY_ADD, {zero, float_const(m, 4.0f)});
        auto *p0 = gep_array_element(b, m, arr, 0, Type::of<float>());
        auto *p1 = gep_array_element(b, m, arr, 1, Type::of<float>());
        auto *p2 = gep_array_element(b, m, arr, 2, Type::of<float>());
        auto *p3 = gep_array_element(b, m, arr, 3, Type::of<float>());
        auto *s0 = b.store(p0, v0);
        b.store(p1, v1);
        b.store(p2, v2);
        b.store(p3, v3);
        auto *extra_use = b.call(Type::of<float>(), ArithmeticOp::UNARY_MINUS, {v0});
        b.return_void();

        auto info = slp_vectorization_pass_run_on_function(k);
        expect(info.vectorized_tree_count == 0u);
        expect(info.vectorized_inst_count == 0u);
        expect(info.rejected_candidate_count == 1u);
        expect(s0->value() == v0);
        expect(extra_use->operand(0) == v0);
        expect(verify(m));
    };

    "slp_vectorization_side_effect_barriers_break_store_runs"_test = [] {
        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        auto *arr = b.alloca_local(Type::array(Type::of<float>(), 4));
        auto *x = b.alloca_local(Type::of<float>());
        b.store(x, m.create_constant_zero(Type::of<float>()));
        auto *xv = b.load(Type::of<float>(), x);
        luisa::vector<Value *> values;
        for (int32_t i = 0; i < 4; ++i) {
            values.emplace_back(b.call(
                Type::of<float>(), ArithmeticOp::BINARY_ADD,
                {xv, float_const(m, static_cast<float>(i + 1))}));
        }
        for (int32_t i = 0; i < 4; ++i) {
            b.store(gep_array_element(b, m, arr, i, Type::of<float>()),
                    values[static_cast<size_t>(i)]);
            b.print("barrier", {});
        }
        b.return_void();

        auto info = slp_vectorization_pass_run_on_function(k);
        expect(info.vectorized_tree_count == 0u);
        expect(info.vectorized_inst_count == 0u);
        expect(info.rejected_candidate_count == 0u);
        expect(verify(m));

        Module control;
        auto *control_kernel = control.create_kernel();
        auto *control_body = control_kernel->create_body_block();
        XIRBuilder control_builder;
        control_builder.set_insertion_point(control_body);
        auto *control_arr = control_builder.alloca_local(
            Type::array(Type::of<float>(), 4));
        auto *control_x = control_builder.alloca_local(Type::of<float>());
        control_builder.store(
            control_x, control.create_constant_zero(Type::of<float>()));
        auto *control_xv = control_builder.load(Type::of<float>(), control_x);
        luisa::vector<Value *> control_values;
        luisa::vector<Value *> control_pointers;
        for (int32_t i = 0; i < 4; ++i) {
            control_values.emplace_back(control_builder.call(
                Type::of<float>(), ArithmeticOp::BINARY_ADD,
                {control_xv,
                 float_const(control, static_cast<float>(i + 1))}));
            control_pointers.emplace_back(
                gep_array_element(control_builder, control, control_arr, i,
                                  Type::of<float>()));
        }
        for (int32_t i = 0; i < 4; ++i) {
            control_builder.store(
                control_pointers[static_cast<size_t>(i)],
                control_values[static_cast<size_t>(i)]);
        }
        control_builder.return_void();
        auto control_info = slp_vectorization_pass_run_on_function(
            control_kernel);
        expect(control_info.vectorized_tree_count == 1u);
        expect(control_info.vectorized_inst_count == 4u);
        expect(control_info.rejected_candidate_count == 0u);
        expect(verify(control));
    };

    "slp_vectorization_empty_and_no_store_inputs"_test = [] {
        Module empty;
        auto empty_info = slp_vectorization_pass_run_on_module(&empty);
        expect(empty_info.vectorized_tree_count == 0u);
        expect(empty_info.vectorized_inst_count == 0u);
        expect(empty_info.rejected_candidate_count == 0u);

        Module m;
        auto *k = m.create_kernel();
        auto *body = k->create_body_block();
        XIRBuilder b;
        b.set_insertion_point(body);
        b.return_void();
        auto info = slp_vectorization_pass_run_on_function(k);
        expect(info.vectorized_tree_count == 0u);
        expect(info.vectorized_inst_count == 0u);
        expect(info.rejected_candidate_count == 0u);
        expect(verify(m));
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    reg_slp_vectorization();
    return 0;
}
