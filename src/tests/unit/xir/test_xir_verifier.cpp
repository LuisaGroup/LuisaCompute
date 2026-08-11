// Test for XIR structural verification.
// This test covers:
// - valid modules
// - ownership, dominance, termination, and structured-control requirements

#include "ut/ut.hpp"

#include <limits>

#include <luisa/ast/type_registry.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/module.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] bool has_verification_error(
    const XIRVerificationResult &result,
    const BasicBlock *block,
    const Instruction *instruction,
    const char *message) noexcept {
    for (auto &&error : result.errors) {
        if (error.block == block &&
            error.instruction == instruction &&
            error.message.find(message) != luisa::string::npos) {
            return true;
        }
    }
    return false;
}

}// namespace

void reg_xir_verifier() {
    "xir_verifier_accepts_valid_module"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.return_void();
        auto result = xir_verify_module(&module);
        expect(result.succeeded());
    };

    "xir_verifier_checks_one_bounded_function_set"_test = [] {
        Module module;
        auto *valid = module.create_kernel();
        auto *valid_body = valid->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(valid_body);
        builder.return_void();

        auto *invalid = module.create_kernel();
        invalid->create_body_block();
        expect(!xir_verify_module(&module).succeeded());

        luisa::vector<const Function *> selected{valid};
        expect(xir_verify_functions(selected).succeeded());
        selected.emplace_back(invalid);
        expect(!xir_verify_functions(selected).succeeded());
    };

    "xir_verifier_use_list_membership_is_constant_in_fanout"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *one = module.create_constant_one(Type::of<int32_t>());
        auto *zero =
            module.create_constant_zero(Type::of<int32_t>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        // Put a user of `zero` before the high-fanout users of `one`. In the
        // wrong-owner case below this makes the verifier materialize zero's
        // use-list first, proving that cached ownership still rejects a Use
        // node linked into a different Value's list.
        builder.call(
            Type::of<int32_t>(), ArithmeticOp::BINARY_ADD,
            {zero, zero});
        constexpr auto fanout = size_t{8192u};
        ArithmeticInst *first_fanout = nullptr;
        for (auto i = 0u; i < fanout; ++i) {
            auto *call = builder.call(
                Type::of<int32_t>(), ArithmeticOp::BINARY_ADD,
                {one, one});
            if (first_fanout == nullptr) { first_fanout = call; }
        }
        builder.return_void();

        auto result = xir_verify_module(&module);
        expect(result.succeeded());
        expect(
            result.statistics.use_list_owner_checks ==
            fanout * 2u + 2u);
        expect(
            result.statistics.use_list_membership_traversal_steps == 0u)
            << "exact membership must use the intrusive owner identity, not "
               "walk any fraction of the high-fanout list";
        expect(one->use_list().contains(first_fanout->operand_use(0u)));
        expect(!zero->use_list().contains(first_fanout->operand_use(0u)));

        // Caching the exact Use-node identities must preserve the verifier's
        // linkage semantics: a non-null operand detached from its Value's
        // use-list remains invalid.
        auto detached =
            first_fanout->operand_use(0u)->remove_self();
        expect(!one->use_list().contains(first_fanout->operand_use(0u)));
        expect(!zero->use_list().contains(first_fanout->operand_use(0u)));
        auto invalid = xir_verify_module(&module);
        expect(!invalid.succeeded());
        expect(has_verification_error(
            invalid, body, first_fanout,
            "Operand use-list linkage is inconsistent."));

        zero->use_list().push_front(std::move(detached));
        expect(!one->use_list().contains(first_fanout->operand_use(0u)));
        expect(zero->use_list().contains(first_fanout->operand_use(0u)));
        auto wrong_owner = xir_verify_module(&module);
        expect(!wrong_owner.succeeded());
        expect(has_verification_error(
            wrong_owner, body, first_fanout,
            "Operand use-list linkage is inconsistent."));
        detached =
            first_fanout->operand_use(0u)->remove_self();
        one->use_list().push_front(std::move(detached));
        expect(one->use_list().contains(first_fanout->operand_use(0u)));
        expect(!zero->use_list().contains(first_fanout->operand_use(0u)));
        expect(xir_verify_module(&module).succeeded());
    };

    "xir_verifier_dominance_storage_is_sparse_in_cfg_size"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition =
            kernel->create_value_argument(Type::of<bool>());
        auto *current = kernel->create_body_block();
        auto *one =
            module.create_constant_one(Type::of<int32_t>());
        XIRBuilder builder;

        // A long diamond chain is deliberately large enough that the old
        // per-block set of all dominators required quadratic storage. The
        // numbered sparse representation must retain only the CFG edges and
        // one idom edge per block.
        constexpr auto diamond_count = size_t{2048u};
        for (auto i = size_t{0u}; i < diamond_count; ++i) {
            auto *left = kernel->create_basic_block();
            auto *right = kernel->create_basic_block();
            auto *merge = kernel->create_basic_block();

            builder.set_insertion_point(current);
            auto *dominating_value = builder.call(
                Type::of<int32_t>(), ArithmeticOp::BINARY_ADD,
                {one, one});
            builder.cond_br(condition, left, right);
            builder.set_insertion_point(left);
            builder.br(merge);
            builder.set_insertion_point(right);
            builder.br(merge);
            builder.set_insertion_point(merge);
            builder.call(
                Type::of<int32_t>(), ArithmeticOp::BINARY_ADD,
                {dominating_value, one});
            current = merge;
        }
        builder.return_void();

        auto result = xir_verify_module(&module);
        expect(result.succeeded());
        constexpr auto expected_blocks =
            size_t{1u} + diamond_count * 3u;
        constexpr auto expected_cfg_edges = diamond_count * 4u;
        expect(
            result.statistics.dominance_tree_nodes ==
            expected_blocks);
        expect(
            result.statistics.dominance_tree_edges ==
            expected_blocks - 1u);
        expect(
            result.statistics.dominance_cfg_edges ==
            expected_cfg_edges);
        expect(
            result.statistics.dominance_fixed_point_iterations <= 3u)
            << "reverse-postorder CHK should converge in a constant number "
               "of sweeps on a diamond chain";
        expect(
            result.statistics.dominance_queries >= diamond_count);
    };

    "xir_verifier_accepts_valid_type_and_category_paths"_test = [] {
        Module module;
        auto int_type = Type::of<int32_t>();
        auto uint_type = Type::of<uint32_t>();
        auto float_type = Type::of<float>();
        auto uint2_type = Type::vector(uint_type, 2u);
        auto ushort4_type = Type::vector(Type::of<uint16_t>(), 4u);
        auto float2_type = Type::vector(float_type, 2u);
        auto float_array_type = Type::array(float_type, 2u);
        auto aggregate_type = Type::structure(
            {int_type, float_array_type});
        auto zero_float = module.create_constant_zero(float_type);
        auto one = module.create_constant_one(int_type);
        auto wide_one = module.create_constant_one(Type::of<uint64_t>());
        auto narrow_zero = module.create_constant_zero(Type::of<int16_t>());
        auto float2_zero = module.create_constant_zero(float2_type);
        auto matrix2_zero = module.create_constant_zero(Type::matrix(2u));
        auto field_index = module.create_constant_one(uint_type);
        auto element_index = module.create_constant_zero(uint_type);

        auto *operations = module.create_callable(nullptr);
        auto *float_array = operations->create_value_argument(float_array_type);
        auto *operations_body = operations->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(operations_body);
        auto *storage = builder.alloca_local(aggregate_type);
        auto *leaf = builder.gep(
            float_type, storage, {field_index, element_index});
        auto *equality = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
            {zero_float, zero_float});
        auto *static_cast_inst = builder.static_cast_(uint_type, one);
        auto *scalar_bit_cast = builder.bit_cast_(float_type, one);
        auto *cross_shape_bit_cast = builder.bit_cast_(ushort4_type, wide_one);
        auto *clz = builder.call(uint_type, ArithmeticOp::CLZ, {field_index});
        auto *reverse = builder.call(
            uint2_type, ArithmeticOp::REVERSE,
            {module.create_constant_zero(uint2_type)});
        auto *shuffle = builder.call(
            float2_type, ArithmeticOp::SHUFFLE,
            {float2_zero, narrow_zero, wide_one});
        auto *extract = builder.call(
            float_type, ArithmeticOp::EXTRACT,
            {float_array, narrow_zero});
        auto *insert = builder.call(
            float_array_type, ArithmeticOp::INSERT,
            {float_array, zero_float, wide_one});
        auto *saturate = builder.call(
            float_type, ArithmeticOp::SATURATE, {zero_float});
        auto *acos = builder.call(
            float_type, ArithmeticOp::ACOS, {zero_float});
        auto *dot = builder.call(
            float_type, ArithmeticOp::DOT, {float2_zero, float2_zero});
        auto *reduce_sum = builder.call(
            float_type, ArithmeticOp::REDUCE_SUM, {float2_zero});
        auto *determinant = builder.call(
            float_type, ArithmeticOp::MATRIX_DETERMINANT, {matrix2_zero});
        auto *aggregate = builder.call(
            float2_type, ArithmeticOp::AGGREGATE,
            {zero_float, zero_float});
        builder.store(leaf, zero_float);
        builder.load(float_type, leaf);
        builder.return_void();
        expect(equality != nullptr);
        expect(static_cast_inst != nullptr);
        expect(scalar_bit_cast != nullptr);
        expect(cross_shape_bit_cast != nullptr);
        expect(clz != nullptr);
        expect(reverse != nullptr);
        expect(shuffle != nullptr);
        expect(extract != nullptr);
        expect(insert != nullptr);
        expect(saturate != nullptr);
        expect(acos != nullptr);
        expect(dot != nullptr);
        expect(reduce_sum != nullptr);
        expect(determinant != nullptr);
        expect(aggregate != nullptr);

        auto *returning = module.create_callable(int_type);
        builder.set_insertion_point(returning->create_body_block());
        builder.return_(one);

        auto buffer_type = Type::buffer(int_type);
        auto *callee = module.create_callable(nullptr);
        callee->create_value_argument(int_type);
        callee->create_reference_argument(int_type);
        callee->create_resource_argument(buffer_type);
        builder.set_insertion_point(callee->create_body_block());
        builder.return_void();

        auto *caller = module.create_callable(nullptr);
        auto *value_argument = caller->create_value_argument(int_type);
        auto *reference_argument = caller->create_reference_argument(int_type);
        auto *resource_argument = caller->create_resource_argument(buffer_type);
        builder.set_insertion_point(caller->create_body_block());
        auto *call = builder.call(
            nullptr, callee,
            {value_argument, reference_argument, resource_argument});
        builder.return_void();
        expect(call != nullptr);

        for (auto selector_type : {int_type, uint_type}) {
            auto *switch_function = module.create_callable(nullptr);
            auto *selector = switch_function->create_value_argument(selector_type);
            auto *body = switch_function->create_body_block();
            builder.set_insertion_point(body);
            auto *switch_inst = builder.switch_(selector);
            auto *default_block = switch_inst->create_default_block();
            auto *case_block = switch_inst->create_case_block(1);
            auto *merge_block = switch_inst->create_merge_block();
            builder.set_insertion_point(default_block);
            builder.br(merge_block);
            builder.set_insertion_point(case_block);
            builder.br(merge_block);
            builder.set_insertion_point(merge_block);
            builder.return_void();
        }

        auto result = xir_verify_module(&module);
        expect(result.succeeded());
        expect(result.errors.empty());
    };

    "xir_verifier_accepts_custom_rvalue_load_store"_test = [] {
        Module module;
        auto query_type = Type::custom("LC_RayQueryAll");
        auto *callable = module.create_callable(nullptr);
        auto *query = callable->create_reference_argument(query_type);
        auto *body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *source = builder.load(query_type, query);
        auto *local = builder.alloca_local(query_type);
        auto *store = builder.store(local, source);
        auto *loaded = builder.load(query_type, local);
        builder.return_void();
        expect(store != nullptr);
        expect(loaded != nullptr);
        auto result = xir_verify_module(&module);
        expect(result.succeeded());
    };

    "xir_kernel_block_size_validation"_test = [] {
        for (auto size : {
                 luisa::make_uint3(1u, 1u, 1u),
                 luisa::make_uint3(4u, 1u, 1u),
                 luisa::make_uint3(8u, 1u, 1u),
                 luisa::make_uint3(31u, 1u, 1u),
                 luisa::make_uint3(32u, 1u, 1u),
                 luisa::make_uint3(33u, 1u, 1u),
                 luisa::make_uint3(32u, 2u, 1u),
                 luisa::make_uint3(1024u, 1u, 1u)}) {
            expect(KernelFunction::is_valid_block_size(size));
        }
        for (auto size : {
                 luisa::make_uint3(0u, 32u, 1u),
                 luisa::make_uint3(33u, 32u, 1u),
                 luisa::make_uint3(1025u, 1u, 1u),
                 luisa::make_uint3(0x80000001u, 32u, 1u),
                 luisa::make_uint3(0xffffffffu, 0xffffffffu, 0xffffffffu)}) {
            expect(!KernelFunction::is_valid_block_size(size));
        }
    };

    "xir_verifier_rejects_invalid_external_argument"_test = [] {
        Module module;
        auto *external = module.create_external_function(nullptr);
        external->arguments().push_back(
            luisa::make_managed<ValueArgument>(external, nullptr));
        auto result = xir_verify_module(&module);
        expect(!result.succeeded());
        expect(has_verification_error(
            result, nullptr, nullptr,
            "Function argument ownership or type is invalid."));
    };

    "xir_verifier_rejects_argument_kind_type_mismatches"_test = [] {
        Module module;
        auto buffer_type = Type::buffer(Type::of<int32_t>());
        auto custom_type = Type::custom("VerifierOpaque");
        auto int_type = Type::of<int32_t>();

        auto *value_resource = module.create_external_function(nullptr);
        value_resource->arguments().push_back(
            luisa::make_managed<ValueArgument>(value_resource, buffer_type));
        auto *value_custom = module.create_external_function(nullptr);
        value_custom->arguments().push_back(
            luisa::make_managed<ValueArgument>(value_custom, custom_type));
        auto *reference_resource = module.create_external_function(nullptr);
        reference_resource->arguments().push_back(
            luisa::make_managed<ReferenceArgument>(
                reference_resource, buffer_type));
        auto *resource_data = module.create_external_function(nullptr);
        resource_data->arguments().push_back(
            luisa::make_managed<ResourceArgument>(resource_data, int_type));

        auto result = xir_verify_module(&module);
        expect(!result.succeeded());
        size_t category_error_count = 0u;
        for (auto &&error : result.errors) {
            category_error_count +=
                error.message.find(
                    "Function argument ownership or type is invalid.") !=
                        luisa::string::npos ?
                    1u :
                    0u;
        }
        expect(category_error_count == 4u);
    };

    "xir_verifier_rejects_unterminated_block"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        kernel->create_body_block();
        auto result = xir_verify_module(&module);
        expect(!result.succeeded());
    };

    "xir_verifier_sanitizes_cross_function_cfg_edges_before_dominance"_test = [] {
        Module module;
        auto *source = module.create_kernel();
        auto *source_body = source->create_body_block();
        auto *foreign = module.create_kernel();
        auto *foreign_body = foreign->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(source_body);
        auto *invalid_branch = builder.br(foreign_body);
        builder.set_insertion_point(foreign_body);
        builder.return_void();

        // The malformed edge is diagnosed, but it must never enter the
        // numbered dominance CFG of either function.
        auto result = xir_verify_module(&module);
        expect(!result.succeeded());
        expect(has_verification_error(
            result, source_body, invalid_branch,
            "Branch has an invalid target."));
        expect(result.statistics.dominance_tree_nodes == 2u);
        expect(result.statistics.dominance_tree_edges == 0u);
        expect(result.statistics.dominance_cfg_edges == 0u);
    };

    "xir_verifier_rejects_use_before_definition"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *one = module.create_constant_one(Type::of<int>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *use = builder.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {one, one});
        auto *definition = builder.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {one, one});
        use->set_operand(0u, definition);
        builder.return_void();
        auto result = xir_verify_module(&module);
        expect(!result.succeeded());
    };

    "xir_verifier_rejects_cross_branch_use"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *true_block = kernel->create_basic_block();
        auto *false_block = kernel->create_basic_block();
        auto *merge_block = kernel->create_basic_block();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *one = module.create_constant_one(Type::of<int>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.cond_br(condition, true_block, false_block);
        builder.set_insertion_point(true_block);
        auto *definition = builder.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {one, one});
        builder.br(merge_block);
        builder.set_insertion_point(false_block);
        builder.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {definition, one});
        builder.br(merge_block);
        builder.set_insertion_point(merge_block);
        builder.return_void();
        auto result = xir_verify_module(&module);
        expect(!result.succeeded());
    };

    "xir_verifier_rejects_reachable_use_of_orphan_definition"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *orphan = kernel->create_basic_block();
        auto *one = module.create_constant_one(Type::of<int>());
        XIRBuilder builder;
        builder.set_insertion_point(orphan);
        auto *definition = builder.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {one, one});
        builder.return_void();
        builder.set_insertion_point(body);
        auto *use = builder.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD, {definition, one});
        builder.return_void();

        auto result = xir_verify_module(&module);
        expect(!result.succeeded());
        expect(has_verification_error(
            result, body, use,
            "Instruction operand does not dominate its use."));
    };

    "xir_verifier_enforces_structured_control_flow"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *true_block = kernel->create_basic_block();
        auto *false_block = kernel->create_basic_block();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.cond_br(condition, true_block, false_block);
        builder.set_insertion_point(true_block);
        builder.return_void();
        builder.set_insertion_point(false_block);
        builder.return_void();
        auto permissive = xir_verify_module(&module);
        expect(permissive.succeeded());
        auto structured = xir_verify_module(
            &module, {.require_no_unstructured_control_flow = true});
        expect(!structured.succeeded());
    };

    "xir_verifier_distinguishes_structured_switch_from_indexed_branch"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *selector = kernel->create_value_argument(Type::of<int>());
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *if_inst = builder.if_(condition);
        auto *true_block = if_inst->create_true_block();
        auto *false_block = if_inst->create_false_block();
        builder.set_insertion_point(true_block);
        builder.return_void();
        builder.set_insertion_point(false_block);
        auto *switch_inst = builder.switch_(selector);
        auto *case_block = switch_inst->create_case_block(1);
        auto *default_block = switch_inst->create_default_block();
        builder.set_insertion_point(case_block);
        builder.return_void();
        builder.set_insertion_point(default_block);
        builder.return_void();

        expect(if_inst->merge_block() == nullptr);
        expect(switch_inst->merge_block() == nullptr);
        auto invalid_switch = xir_verify_module(&module);
        expect(!invalid_switch.succeeded());
        expect(has_verification_error(
            invalid_switch, false_block, switch_inst,
            "Structured control flow has an invalid merge block."));

        switch_inst->remove_self();
        builder.set_insertion_point(false_block);
        auto *indexed_branch = builder.indexed_branch(selector);
        indexed_branch->add_case(1, case_block);
        indexed_branch->set_default_block(default_block);

        auto valid_raw_cfg = xir_verify_module(&module);
        expect(valid_raw_cfg.succeeded());
        auto structured_only = xir_verify_module(
            &module, {.require_no_unstructured_control_flow = true});
        expect(!structured_only.succeeded());
        expect(has_verification_error(
            structured_only, false_block, indexed_branch,
            "Unstructured control flow is not allowed."));
    };

    "xir_verifier_accepts_nearest_structured_break_continue_targets"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *selector = kernel->create_value_argument(Type::of<int32_t>());
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *loop = builder.loop();
        auto *prepare = loop->create_prepare_block();
        auto *loop_body = loop->create_body_block();
        auto *update = loop->create_update_block();
        auto *loop_merge = loop->create_merge_block();
        builder.set_insertion_point(prepare);
        builder.cond_br(condition, loop_body, loop_merge);
        builder.set_insertion_point(loop_body);
        auto *switch_inst = builder.switch_(selector);
        auto *case_block = switch_inst->create_case_block(0);
        auto *default_block = switch_inst->create_default_block();
        auto *switch_merge = switch_inst->create_merge_block();
        builder.set_insertion_point(case_block);
        builder.break_(switch_merge);
        builder.set_insertion_point(default_block);
        builder.continue_(update);
        builder.set_insertion_point(switch_merge);
        builder.break_(loop_merge);
        builder.set_insertion_point(update);
        builder.br(prepare);
        builder.set_insertion_point(loop_merge);
        builder.return_void();

        auto result = xir_verify_module(
            &module, {.require_canonical_break_continue_targets = true});
        expect(result.succeeded());
    };

    "xir_verifier_rejects_non_nearest_structured_break_continue_targets"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *outer_loop = builder.loop();
        auto *outer_prepare = outer_loop->create_prepare_block();
        auto *outer_body = outer_loop->create_body_block();
        auto *outer_update = outer_loop->create_update_block();
        auto *outer_merge = outer_loop->create_merge_block();
        builder.set_insertion_point(outer_prepare);
        builder.cond_br(condition, outer_body, outer_merge);
        builder.set_insertion_point(outer_body);
        auto *inner_loop = builder.loop();
        auto *inner_prepare = inner_loop->create_prepare_block();
        auto *inner_body = inner_loop->create_body_block();
        auto *inner_update = inner_loop->create_update_block();
        auto *inner_merge = inner_loop->create_merge_block();
        builder.set_insertion_point(inner_prepare);
        builder.cond_br(condition, inner_body, inner_merge);
        builder.set_insertion_point(inner_body);
        auto *if_inst = builder.if_(condition);
        auto *break_block = if_inst->create_true_block();
        auto *continue_block = if_inst->create_false_block();
        builder.set_insertion_point(break_block);
        auto *break_inst = builder.break_(outer_merge);
        builder.set_insertion_point(continue_block);
        auto *continue_inst = builder.continue_(outer_update);
        builder.set_insertion_point(inner_update);
        builder.br(inner_prepare);
        builder.set_insertion_point(inner_merge);
        builder.br(outer_update);
        builder.set_insertion_point(outer_update);
        builder.br(outer_prepare);
        builder.set_insertion_point(outer_merge);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto result = xir_verify_module(
            &module, {.require_canonical_break_continue_targets = true});
        expect(has_verification_error(
            result, break_block, break_inst,
            "Break target is not the nearest enclosing structured break target."));
        expect(has_verification_error(
            result, continue_block, continue_inst,
            "Continue target is not the nearest enclosing structured loop target."));
    };

    "xir_verifier_rejects_break_continue_without_structured_scope"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *body = kernel->create_body_block();
        auto *break_block = kernel->create_basic_block();
        auto *continue_block = kernel->create_basic_block();
        auto *exit = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.cond_br(condition, break_block, continue_block);
        builder.set_insertion_point(break_block);
        auto *break_inst = builder.break_(exit);
        builder.set_insertion_point(continue_block);
        auto *continue_inst = builder.continue_(exit);
        builder.set_insertion_point(exit);
        builder.return_void();

        expect(xir_verify_module(&module).succeeded());
        auto result = xir_verify_module(
            &module, {.require_canonical_break_continue_targets = true});
        expect(has_verification_error(
            result, break_block, break_inst,
            "Break target is not the nearest enclosing structured break target."));
        expect(has_verification_error(
            result, continue_block, continue_inst,
            "Continue target is not the nearest enclosing structured loop target."));
    };

    "xir_verifier_rejects_call_result_type_mismatch"_test = [] {
        Module module;
        auto *callee = module.create_callable(nullptr);
        auto *callee_body = callee->create_body_block();
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(callee_body);
        builder.return_void();
        builder.set_insertion_point(body);
        builder.call(Type::of<int>(), callee, {});
        builder.return_void();
        auto result = xir_verify_module(&module);
        expect(!result.succeeded());
    };

    "xir_verifier_rejects_return_type_mismatch"_test = [] {
        Module module;
        auto *callable = module.create_callable(Type::of<int>());
        auto *body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.return_void();
        auto result = xir_verify_module(&module);
        expect(!result.succeeded());
    };

    "xir_verifier_rejects_phi_after_non_phi"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *merge = kernel->create_basic_block();
        auto *one = module.create_constant_one(Type::of<int>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.br(merge);
        builder.set_insertion_point(merge);
        builder.call(Type::of<int>(), ArithmeticOp::BINARY_ADD, {one, one});
        builder.phi(Type::of<int>(), {{one, body}});
        builder.return_void();
        auto result = xir_verify_module(&module);
        expect(!result.succeeded());
    };

    "xir_verifier_require_no_phi_rejects_valid_phi"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *true_block = kernel->create_basic_block();
        auto *false_block = kernel->create_basic_block();
        auto *merge_block = kernel->create_basic_block();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        auto *one = module.create_constant_one(Type::of<int>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.cond_br(condition, true_block, false_block);
        builder.set_insertion_point(true_block);
        builder.br(merge_block);
        builder.set_insertion_point(false_block);
        builder.br(merge_block);
        builder.set_insertion_point(merge_block);
        auto *phi = builder.phi(Type::of<int>(), {{one, true_block}, {one, false_block}});
        builder.return_void();

        auto permissive = xir_verify_module(&module);
        expect(permissive.succeeded());
        auto strict = xir_verify_module(&module, {.require_no_phi = true});
        expect(!strict.succeeded());
        expect(has_verification_error(
            strict, merge_block, phi, "PHI instruction is not allowed."));
    };

    "xir_verifier_require_unique_merge_blocks_rejects_shared_merge"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *left_header = kernel->create_basic_block();
        auto *right_header = kernel->create_basic_block();
        auto *shared_merge = kernel->create_basic_block();
        auto *condition = kernel->create_value_argument(Type::of<bool>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.cond_br(condition, left_header, right_header);

        builder.set_insertion_point(left_header);
        auto *left_if = builder.if_(condition);
        auto *left_true = left_if->create_true_block();
        auto *left_false = left_if->create_false_block();
        left_if->set_merge_block(shared_merge);
        builder.set_insertion_point(left_true);
        builder.br(shared_merge);
        builder.set_insertion_point(left_false);
        builder.br(shared_merge);

        builder.set_insertion_point(right_header);
        auto *right_if = builder.if_(condition);
        auto *right_true = right_if->create_true_block();
        auto *right_false = right_if->create_false_block();
        right_if->set_merge_block(shared_merge);
        builder.set_insertion_point(right_true);
        builder.br(shared_merge);
        builder.set_insertion_point(right_false);
        builder.br(shared_merge);

        builder.set_insertion_point(shared_merge);
        builder.return_void();

        auto permissive = xir_verify_module(&module);
        expect(permissive.succeeded());
        auto strict = xir_verify_module(
            &module, {.require_unique_merge_blocks = true});
        expect(!strict.succeeded());
        expect(has_verification_error(
            strict, right_header, right_if,
            "Structured merge block is owned by multiple instructions."));
    };

    "xir_verifier_require_reachable_blocks_rejects_orphan"_test = [] {
        Module module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *orphan = kernel->create_basic_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        builder.return_void();
        builder.set_insertion_point(orphan);
        builder.return_void();

        auto permissive = xir_verify_module(&module);
        expect(permissive.succeeded());
        auto strict = xir_verify_module(
            &module, {.require_reachable_blocks = true});
        expect(!strict.succeeded());
        expect(has_verification_error(
            strict, orphan, nullptr, "Basic block is unreachable."));
    };

    "xir_verifier_rejects_cross_module_constant_with_context"_test = [] {
        Module module;
        Module foreign_module;
        auto *kernel = module.create_kernel();
        auto *body = kernel->create_body_block();
        auto *foreign_one = foreign_module.create_constant_one(Type::of<int>());
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *use = builder.call(
            Type::of<int>(), ArithmeticOp::BINARY_ADD,
            {foreign_one, foreign_one});
        builder.return_void();

        auto result = xir_verify_module(&module);
        expect(!result.succeeded());
        expect(has_verification_error(
            result, body, use,
            "Instruction references a constant from another module."));
    };

    "xir_verifier_rejects_invalid_arithmetic_types"_test = [] {
        Module module;
        auto *callable = module.create_callable(nullptr);
        auto *lhs = callable->create_value_argument(Type::of<int32_t>());
        auto *rhs = callable->create_value_argument(Type::of<int32_t>());
        auto *body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *arithmetic = builder.call(
            Type::of<float>(), ArithmeticOp::BINARY_ADD, {lhs, rhs});
        builder.return_void();

        auto result = xir_verify_module(&module);
        expect(!result.succeeded());
        expect(has_verification_error(
            result, body, arithmetic,
            "Arithmetic operands or result type are invalid."));
    };

    "xir_verifier_rejects_extended_arithmetic_family_errors"_test = [] {
        Module module;
        auto int_type = Type::of<int32_t>();
        auto uint_type = Type::of<uint32_t>();
        auto float_type = Type::of<float>();
        auto float2_type = Type::vector(float_type, 2u);
        auto matrix2_type = Type::matrix(2u);
        auto array_type = Type::array(int_type, 2u);
        auto *callable = module.create_callable(nullptr);
        auto *int_value = callable->create_value_argument(int_type);
        auto *float_value = callable->create_value_argument(float_type);
        auto *float2_value = callable->create_value_argument(float2_type);
        auto *matrix_value = callable->create_value_argument(matrix2_type);
        auto *array_value = callable->create_value_argument(array_type);
        auto *index = callable->create_value_argument(uint_type);
        auto *narrow_value = callable->create_value_argument(Type::of<uint16_t>());
        auto *bool3_value = callable->create_value_argument(
            Type::vector(Type::of<bool>(), 3u));
        auto *body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto negative_index_value = int32_t{-1};
        auto out_of_range_index_value = uint32_t{2u};
        auto *negative_index = module.create_constant(
            int_type, &negative_index_value);
        auto *out_of_range_index = module.create_constant(
            uint_type, &out_of_range_index_value);
        auto *zero_index = module.create_constant_zero(uint_type);

        luisa::vector<ArithmeticInst *> malformed;
        malformed.emplace_back(builder.call(
            int_type, ArithmeticOp::SATURATE, {int_value}));
        malformed.emplace_back(builder.call(
            int_type, ArithmeticOp::ACOS, {int_value}));
        malformed.emplace_back(builder.call(
            int_type, ArithmeticOp::CLZ, {int_value}));
        malformed.emplace_back(builder.call(
            Type::of<uint16_t>(), ArithmeticOp::POPCOUNT, {narrow_value}));
        malformed.emplace_back(builder.call(
            float_type, ArithmeticOp::SELECT,
            {float_value, float_value, int_value}));
        malformed.emplace_back(builder.call(
            float2_type, ArithmeticOp::SELECT,
            {float2_value, float2_value, bool3_value}));
        malformed.emplace_back(builder.call(
            float2_type, ArithmeticOp::DOT, {float2_value, float2_value}));
        malformed.emplace_back(builder.call(
            float2_type, ArithmeticOp::REDUCE_SUM, {float2_value}));
        malformed.emplace_back(builder.call(
            matrix2_type, ArithmeticOp::MATRIX_DETERMINANT, {matrix_value}));
        malformed.emplace_back(builder.call(
            float2_type, ArithmeticOp::AGGREGATE, {float_value}));
        malformed.emplace_back(builder.call(
            float2_type, ArithmeticOp::SHUFFLE, {float2_value, index}));
        malformed.emplace_back(builder.call(
            float2_type, ArithmeticOp::SHUFFLE,
            {float2_value, out_of_range_index, zero_index}));
        malformed.emplace_back(builder.call(
            float_type, ArithmeticOp::EXTRACT, {array_value, index}));
        malformed.emplace_back(builder.call(
            int_type, ArithmeticOp::EXTRACT,
            {array_value, negative_index}));
        malformed.emplace_back(builder.call(
            array_type, ArithmeticOp::INSERT,
            {array_value, float_value, index}));
        malformed.emplace_back(builder.call(
            array_type, ArithmeticOp::INSERT,
            {array_value, int_value, out_of_range_index}));
        auto *wrong_arity = builder.call(
            float_type, ArithmeticOp::SATURATE, {float_value});
        wrong_arity->add_operand(float_value);
        auto *storage = builder.alloca_local(float_type);
        malformed.emplace_back(builder.call(
            float_type, ArithmeticOp::SATURATE, {storage}));
        builder.return_void();

        auto result = xir_verify_module(&module);
        expect(!result.succeeded());
        for (auto *instruction : malformed) {
            expect(has_verification_error(
                result, body, instruction,
                "Arithmetic operands or result type are invalid."));
        }
        expect(has_verification_error(
            result, body, wrong_arity,
            "Instruction operand count is invalid."));
    };

    "xir_verifier_rejects_invalid_instruction_opcodes"_test = [] {
        Module module;
        auto int_type = Type::of<int32_t>();
        auto *one = module.create_constant_one(int_type);
        auto *callable = module.create_callable(nullptr);
        auto *query = callable->create_reference_argument(
            Type::custom("LC_RayQueryAll"));
        auto *body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);

        auto *alloca = builder.alloca_local(int_type);
        alloca->set_op(static_cast<AllocaOp>(999));
        auto *atomic = builder.atomic_fetch_add(
            int_type, alloca, luisa::span<Value *const>{}, one);
        atomic->set_op(static_cast<AtomicOp>(999));
        auto *ray_query_read = builder.call(
            Type::of<bool>(),
            RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED,
            {query});
        ray_query_read->set_op(static_cast<RayQueryObjectReadOp>(999));
        auto *ray_query_write = builder.call(
            RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE,
            {query});
        ray_query_write->set_op(static_cast<RayQueryObjectWriteOp>(999));
        builder.return_void();

        auto result = xir_verify_module(&module);
        expect(!result.succeeded());
        for (auto *instruction : std::array<const Instruction *, 4u>{
                 alloca, atomic, ray_query_read, ray_query_write}) {
            expect(has_verification_error(
                result, body, instruction,
                "Instruction opcode is invalid."));
        }
    };

    "xir_verifier_accepts_integer_64_bit_atomics"_test = [] {
        for (auto type : {Type::of<luisa::slong>(),
                          Type::of<luisa::ulong>()}) {
            Module module;
            auto *index = module.create_constant_zero(Type::of<uint>());
            auto *value = module.create_constant_one(type);
            auto *callable = module.create_callable(nullptr);
            auto *buffer = callable->create_resource_argument(
                Type::buffer(type));
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *shared = builder.alloca_shared(Type::array(type, 1u));
            std::array<Value *, 1u> indices{index};
            builder.atomic_fetch_add(
                type, buffer, luisa::span<Value *const>{indices}, value);
            builder.atomic_fetch_add(
                type, shared, luisa::span<Value *const>{indices}, value);
            builder.return_void();

            expect(xir_verify_module(&module).succeeded())
                << "signed and unsigned 64-bit atomics must be valid XIR for both buffer and shared storage";
        }
    };

    "xir_verifier_rejects_unsupported_atomic_scalar_types"_test = [] {
        for (auto type : {
                 Type::of<int8_t>(), Type::of<uint8_t>(),
                 Type::of<int16_t>(), Type::of<uint16_t>(),
                 Type::of<double>()}) {
            Module module;
            auto *index = module.create_constant_zero(Type::of<uint>());
            auto *value = module.create_constant_one(type);
            auto *callable = module.create_callable(nullptr);
            auto *buffer = callable->create_resource_argument(
                Type::buffer(type));
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            std::array<Value *, 1u> indices{index};
            auto *atomic = builder.atomic_fetch_add(
                type, buffer, luisa::span<Value *const>{indices}, value);
            builder.return_void();

            auto result = xir_verify_module(&module);
            expect(!result.succeeded());
            expect(has_verification_error(
                result, body, atomic,
                "Instruction operands or result type are invalid."));
        }
    };

    "xir_verifier_rejects_invalid_resource_atomic_and_thread_group_semantics"_test = [] {
        Module module;
        auto int_type = Type::of<int32_t>();
        auto uint_type = Type::of<uint32_t>();
        auto float_type = Type::of<float>();
        auto buffer_type = Type::buffer(int_type);
        auto *index = module.create_constant_zero(uint_type);
        auto *float_value = module.create_constant_zero(float_type);
        auto *int_value = module.create_constant_zero(int_type);
        auto *callable = module.create_callable(nullptr);
        auto *buffer = callable->create_resource_argument(buffer_type);
        auto *body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);

        auto *query = builder.call(
            float_type, ResourceQueryOp::BUFFER_SIZE, {buffer});
        auto *read = builder.call(
            float_type, ResourceReadOp::BUFFER_READ, {buffer, index});
        auto *write = builder.call(
            ResourceWriteOp::BUFFER_WRITE, {buffer, index, float_value});
        std::array<Value *, 1u> indices{index};
        auto *atomic = builder.atomic_fetch_add(
            float_type, buffer, luisa::span<Value *const>{indices}, float_value);
        auto *thread_group = builder.call(
            Type::of<bool>(), ThreadGroupOp::WARP_ACTIVE_SUM, {int_value});
        builder.return_void();

        auto result = xir_verify_module(&module);
        expect(!result.succeeded());
        for (auto *instruction : std::array<const Instruction *, 5u>{
                 query, read, write, atomic, thread_group}) {
            expect(has_verification_error(
                result, body, instruction,
                "Instruction operands or result type are invalid."));
        }
    };

    "xir_verifier_rejects_invalid_ray_query_and_autodiff_semantics"_test = [] {
        Module module;
        auto query_type = Type::custom("LC_RayQueryAll");
        auto int_type = Type::of<int32_t>();
        auto float_type = Type::of<float>();
        auto *int_value = module.create_constant_zero(int_type);
        auto *float_value = module.create_constant_zero(float_type);

        auto *procedural = module.create_callable(nullptr);
        procedural->create_reference_argument(query_type);
        auto *procedural_body = procedural->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(procedural_body);
        builder.return_void();

        auto *external_surface = module.create_external_function(nullptr);
        auto *callable = module.create_callable(nullptr);
        auto *query = callable->create_reference_argument(query_type);
        auto *body = callable->create_body_block();
        builder.set_insertion_point(body);
        auto *read = builder.call(
            int_type,
            RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED,
            {query});
        auto *write = builder.call(
            RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL,
            {query, int_value});
        auto *pipeline = builder.ray_query_pipeline(
            query, external_surface, procedural);
        auto *autodiff = builder.call(
            int_type, AutodiffIntrinsicOp::AUTODIFF_GRADIENT,
            {float_value});
        builder.return_void();

        auto result = xir_verify_module(&module);
        expect(!result.succeeded());
        for (auto *instruction : std::array<const Instruction *, 4u>{
                 read, write, pipeline, autodiff}) {
            expect(has_verification_error(
                result, body, instruction,
                "Instruction operands or result type are invalid."));
        }
    };

    "xir_verifier_rejects_invalid_coro_and_diagnostic_semantics"_test = [] {
        Module module;
        auto int_type = Type::of<int32_t>();
        auto buffer_type = Type::buffer(int_type);
        auto *int_value = module.create_constant_zero(int_type);
        XIRBuilder builder;

        auto *diagnostics = module.create_callable(nullptr);
        auto *reference = diagnostics->create_reference_argument(int_type);
        auto *resource = diagnostics->create_resource_argument(buffer_type);
        auto *diagnostics_body = diagnostics->create_body_block();
        builder.set_insertion_point(diagnostics_body);
        auto *print = builder.print("{}", {reference});
        auto *debug_break = builder.debug_break();
        debug_break->add_operand(resource);
        auto *assert_inst = builder.assert_(int_value);
        auto *assume_inst = builder.assume_(int_value);
        builder.return_void();

        auto *resume_function = module.create_callable(nullptr);
        auto *resume_body = resume_function->create_body_block();
        builder.set_insertion_point(resume_body);
        auto *resume = builder.coro_resume(1u, resume_body);
        builder.return_void();

        auto *suspend_function = module.create_callable(nullptr);
        auto *suspend_body = suspend_function->create_body_block();
        builder.set_insertion_point(suspend_body);
        auto *suspend = builder.coro_suspend(2u, "suspend", suspend_body);

        auto result = xir_verify_module(&module);
        expect(!result.succeeded());
        for (auto *instruction : std::array<const Instruction *, 4u>{
                 print, debug_break, assert_inst, assume_inst}) {
            expect(has_verification_error(
                result, diagnostics_body, instruction,
                "Instruction operands or result type are invalid."));
        }
        expect(has_verification_error(
            result, resume_body, resume,
            "Instruction operands or result type are invalid."));
        expect(has_verification_error(
            result, suspend_body, suspend,
            "Instruction operands or result type are invalid."));
    };

    "xir_verifier_rejects_invalid_cast_types"_test = [] {
        auto verify_invalid_cast = [](const Type *source_type,
                                      const Type *target_type) {
            Module module;
            auto *callable = module.create_callable(nullptr);
            auto *value = callable->create_value_argument(source_type);
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *cast = builder.bit_cast_(target_type, value);
            builder.return_void();

            auto result = xir_verify_module(&module);
            expect(!result.succeeded());
            expect(has_verification_error(
                result, body, cast,
                "Cast operands or result type are invalid."));
        };
        verify_invalid_cast(Type::of<int32_t>(), Type::of<uint64_t>());
        verify_invalid_cast(
            Type::vector(Type::of<float>(), 3u),
            Type::vector(Type::of<uint32_t>(), 4u));
        verify_invalid_cast(Type::of<bool>(), Type::of<bool>());
        verify_invalid_cast(
            Type::vector(Type::of<bool>(), 2u),
            Type::vector(Type::of<bool>(), 2u));
    };

    "xir_verifier_rejects_invalid_value_categories"_test = [] {
        {
            Module module;
            auto *callable = module.create_callable(nullptr);
            auto *value = callable->create_value_argument(Type::of<int32_t>());
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *storage = builder.alloca_local(Type::of<int32_t>());
            auto *load = builder.load(Type::of<int32_t>(), storage);
            load->set_variable(value);
            builder.return_void();
            auto result = xir_verify_module(&module);
            expect(has_verification_error(
                result, body, load,
                "Load variable or result type is invalid."));
        }
        {
            Module module;
            auto *callable = module.create_callable(nullptr);
            auto *destination = callable->create_reference_argument(
                Type::of<int32_t>());
            auto *source = callable->create_reference_argument(
                Type::of<int32_t>());
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *store = builder.store(
                destination,
                module.create_constant_zero(Type::of<int32_t>()));
            store->set_value(source);
            builder.return_void();
            auto result = xir_verify_module(&module);
            expect(has_verification_error(
                result, body, store,
                "Store variable or value type is invalid."));
        }
        {
            Module module;
            auto *callable = module.create_callable(Type::of<int32_t>());
            auto *value = callable->create_reference_argument(
                Type::of<int32_t>());
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *return_inst = builder.return_(value);
            auto result = xir_verify_module(&module);
            expect(has_verification_error(
                result, body, return_inst,
                "Return value does not match the function return type."));
        }
        {
            Module module;
            auto *callable = module.create_callable(nullptr);
            auto *condition = callable->create_value_argument(Type::of<bool>());
            auto *value = callable->create_reference_argument(
                Type::of<int32_t>());
            auto *body = callable->create_body_block();
            auto *left = callable->create_basic_block();
            auto *right = callable->create_basic_block();
            auto *merge = callable->create_basic_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            builder.cond_br(condition, left, right);
            builder.set_insertion_point(left);
            builder.br(merge);
            builder.set_insertion_point(right);
            builder.br(merge);
            builder.set_insertion_point(merge);
            auto *phi = builder.phi(
                Type::of<int32_t>(), {{value, left}, {value, right}});
            builder.return_void();
            auto result = xir_verify_module(&module);
            expect(has_verification_error(
                result, merge, phi,
                "PHI incoming edge or value is invalid."));
        }
        {
            Module module;
            auto *callable = module.create_callable(nullptr);
            auto *condition = callable->create_reference_argument(Type::of<bool>());
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *if_inst = builder.if_(condition);
            auto *true_block = if_inst->create_true_block();
            auto *false_block = if_inst->create_false_block();
            auto *merge = if_inst->create_merge_block();
            builder.set_insertion_point(true_block);
            builder.br(merge);
            builder.set_insertion_point(false_block);
            builder.br(merge);
            builder.set_insertion_point(merge);
            builder.return_void();
            auto result = xir_verify_module(&module);
            expect(has_verification_error(
                result, body, if_inst,
                "Conditional branch condition is not a boolean rvalue."));
        }
        {
            Module module;
            auto *callable = module.create_callable(nullptr);
            auto *condition = callable->create_reference_argument(Type::of<bool>());
            auto *body = callable->create_body_block();
            auto *target = callable->create_basic_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *branch = builder.cond_br(condition, target, target);
            builder.set_insertion_point(target);
            builder.return_void();
            auto result = xir_verify_module(&module);
            expect(has_verification_error(
                result, body, branch,
                "Conditional branch condition is not a boolean rvalue."));
        }
    };

    "xir_verifier_rejects_non_block_branch_and_switch_targets"_test = [] {
        {
            Module module;
            auto *zero = module.create_constant_zero(Type::of<int32_t>());
            auto *callable = module.create_callable(nullptr);
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *branch = builder.br(body);
            branch->set_operand(
                BranchTerminatorInstruction::operand_index_target, zero);
            auto result = xir_verify_module(&module);
            expect(has_verification_error(
                result, body, branch, "Branch has an invalid target."));
        }
        {
            Module module;
            auto *zero = module.create_constant_zero(Type::of<int32_t>());
            auto *condition = module.create_constant_one(Type::of<bool>());
            auto *callable = module.create_callable(nullptr);
            auto *body = callable->create_body_block();
            auto *target = callable->create_basic_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *branch = builder.cond_br(condition, target, target);
            branch->set_operand(
                ConditionalBranchTerminatorInstruction::operand_index_true_target,
                zero);
            builder.set_insertion_point(target);
            builder.return_void();
            auto result = xir_verify_module(&module);
            expect(has_verification_error(
                result, body, branch,
                "Conditional branch has an invalid target."));
        }
        {
            Module module;
            auto *zero = module.create_constant_zero(Type::of<int32_t>());
            auto *condition = module.create_constant_one(Type::of<bool>());
            auto *callable = module.create_callable(nullptr);
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *if_inst = builder.if_(condition);
            auto *true_block = if_inst->create_true_block();
            auto *false_block = if_inst->create_false_block();
            auto *merge = if_inst->create_merge_block();
            if_inst->set_operand(
                ConditionalBranchTerminatorInstruction::operand_index_true_target,
                zero);
            builder.set_insertion_point(true_block);
            builder.br(merge);
            builder.set_insertion_point(false_block);
            builder.br(merge);
            builder.set_insertion_point(merge);
            builder.return_void();
            auto result = xir_verify_module(&module);
            expect(has_verification_error(
                result, body, if_inst,
                "Conditional branch has an invalid target."));
        }
        {
            Module module;
            auto *zero = module.create_constant_zero(Type::of<int32_t>());
            auto *selector = module.create_constant_zero(Type::of<uint32_t>());
            auto *callable = module.create_callable(nullptr);
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *switch_inst = builder.switch_(selector);
            auto *default_block = switch_inst->create_default_block();
            auto *case_block = switch_inst->create_case_block(1);
            auto *merge = switch_inst->create_merge_block();
            switch_inst->set_operand(
                SwitchInst::operand_index_default_block, zero);
            switch_inst->set_operand(
                SwitchInst::operand_index_case_block_offset, zero);
            builder.set_insertion_point(default_block);
            builder.br(merge);
            builder.set_insertion_point(case_block);
            builder.br(merge);
            builder.set_insertion_point(merge);
            builder.return_void();
            auto result = xir_verify_module(&module);
            expect(has_verification_error(
                result, body, switch_inst,
                "Switch value or default block is invalid."));
            expect(has_verification_error(
                result, body, switch_inst,
                "Switch case block is invalid."));
        }
    };

    "xir_verifier_rejects_extra_branch_operands"_test = [] {
        {
            Module module;
            auto *callable = module.create_callable(nullptr);
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *return_inst = builder.return_void();
            return_inst->remove_operand(0u);
            auto result = xir_verify_module(&module);
            expect(has_verification_error(
                result, body, return_inst,
                "Instruction operand count is invalid."));
        }
        {
            Module module;
            auto *callable = module.create_callable(nullptr);
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *storage = builder.alloca_local(Type::of<int32_t>());
            auto *load = builder.load(Type::of<int32_t>(), storage);
            load->remove_operand(0u);
            builder.return_void();
            auto result = xir_verify_module(&module);
            expect(has_verification_error(
                result, body, load,
                "Instruction operand count is invalid."));
        }
        {
            Module module;
            auto *callable = module.create_callable(nullptr);
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *return_inst = builder.return_void();
            return_inst->add_operand(body);
            auto result = xir_verify_module(&module);
            expect(has_verification_error(
                result, body, return_inst,
                "Instruction operand count is invalid."));
        }
        {
            Module module;
            auto *callable = module.create_callable(nullptr);
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *branch = builder.br(body);
            branch->add_operand(body);
            auto result = xir_verify_module(&module);
            expect(has_verification_error(
                result, body, branch,
                "Instruction operand count is invalid."));
        }
        {
            Module module;
            auto *condition = module.create_constant_one(Type::of<bool>());
            auto *callable = module.create_callable(nullptr);
            auto *body = callable->create_body_block();
            auto *target = callable->create_basic_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *branch = builder.cond_br(condition, target, target);
            branch->add_operand(target);
            builder.set_insertion_point(target);
            builder.return_void();
            auto result = xir_verify_module(&module);
            expect(has_verification_error(
                result, body, branch,
                "Instruction operand count is invalid."));
        }
        {
            Module module;
            auto *condition = module.create_constant_one(Type::of<bool>());
            auto *callable = module.create_callable(nullptr);
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *if_inst = builder.if_(condition);
            auto *true_block = if_inst->create_true_block();
            auto *false_block = if_inst->create_false_block();
            auto *merge = if_inst->create_merge_block();
            if_inst->add_operand(merge);
            builder.set_insertion_point(true_block);
            builder.br(merge);
            builder.set_insertion_point(false_block);
            builder.br(merge);
            builder.set_insertion_point(merge);
            builder.return_void();
            auto result = xir_verify_module(&module);
            expect(has_verification_error(
                result, body, if_inst,
                "Instruction operand count is invalid."));
        }
    };

    "xir_verifier_enforces_call_argument_categories"_test = [] {
        Module module;
        auto int_type = Type::of<int32_t>();
        auto buffer_type = Type::buffer(int_type);
        auto *callee = module.create_callable(nullptr);
        callee->create_value_argument(int_type);
        callee->create_reference_argument(int_type);
        callee->create_resource_argument(buffer_type);
        XIRBuilder builder;
        builder.set_insertion_point(callee->create_body_block());
        builder.return_void();

        auto *caller = module.create_callable(nullptr);
        auto *value = caller->create_value_argument(int_type);
        auto *reference = caller->create_reference_argument(int_type);
        auto *resource = caller->create_resource_argument(buffer_type);
        auto *body = caller->create_body_block();
        builder.set_insertion_point(body);
        auto *bad_value = builder.call(
            nullptr, callee, {reference, reference, resource});
        auto *bad_reference = builder.call(
            nullptr, callee, {value, value, resource});
        auto *bad_resource = builder.call(
            nullptr, callee,
            {value, reference, module.create_undefined(buffer_type)});
        builder.return_void();

        auto result = xir_verify_module(&module);
        expect(!result.succeeded());
        for (auto *call : {bad_value, bad_reference, bad_resource}) {
            expect(has_verification_error(
                result, body, call,
                "Call argument type or value category is invalid."));
        }
    };

    "xir_verifier_rejects_invalid_gep_type_paths"_test = [] {
        {
            Module module;
            auto *callable = module.create_callable(nullptr);
            auto *index = callable->create_value_argument(Type::of<uint32_t>());
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *storage = builder.alloca_local(
                Type::array(Type::of<int32_t>(), 2u));
            auto *gep = builder.gep(Type::of<float>(), storage, {index});
            builder.return_void();
            auto result = xir_verify_module(&module);
            expect(!result.succeeded());
            expect(has_verification_error(result, body, gep, "GEP is invalid."));
        }
        {
            Module module;
            auto *index = module.create_constant_one(Type::of<uint32_t>());
            auto *callable = module.create_callable(nullptr);
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *storage = builder.alloca_local(
                Type::structure({Type::of<int32_t>()}));
            auto *gep = builder.gep(Type::of<int32_t>(), storage, {index});
            builder.return_void();
            auto result = xir_verify_module(&module);
            expect(!result.succeeded());
            expect(has_verification_error(result, body, gep, "GEP is invalid."));
        }
        {
            Module module;
            auto *callable = module.create_callable(nullptr);
            auto *index = callable->create_value_argument(Type::of<float>());
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *storage = builder.alloca_local(
                Type::array(Type::of<int32_t>(), 2u));
            auto *gep = builder.gep(Type::of<int32_t>(), storage, {index});
            builder.return_void();
            auto result = xir_verify_module(&module);
            expect(!result.succeeded());
            expect(has_verification_error(result, body, gep, "GEP is invalid."));
        }
        {
            Module module;
            auto *callable = module.create_callable(nullptr);
            auto *index = callable->create_value_argument(Type::of<uint32_t>());
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *storage = builder.alloca_local(
                Type::structure({Type::of<int32_t>(), Type::of<uint32_t>()}));
            auto *gep = builder.gep(Type::of<int32_t>(), storage, {index});
            builder.return_void();
            auto result = xir_verify_module(&module);
            expect(!result.succeeded());
            expect(has_verification_error(result, body, gep, "GEP is invalid."));
        }
    };

    "xir_verifier_accepts_integer_and_bool_switch_selectors"_test = [] {
        for (auto selector_type : {
                 Type::of<bool>(),
                 Type::of<int8_t>(), Type::of<uint8_t>(),
                 Type::of<int16_t>(), Type::of<uint16_t>(),
                 Type::of<int32_t>(), Type::of<uint32_t>(),
                 Type::of<int64_t>(), Type::of<uint64_t>()}) {
            Module module;
            auto *callable = module.create_callable(nullptr);
            auto *selector = callable->create_value_argument(selector_type);
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *switch_inst = builder.switch_(selector);
            auto *default_block = switch_inst->create_default_block();
            auto *merge_block = switch_inst->create_merge_block();
            builder.set_insertion_point(default_block);
            builder.br(merge_block);
            builder.set_insertion_point(merge_block);
            builder.return_void();
            auto result = xir_verify_module(&module);
            expect(result.succeeded());
        }
    };

    "xir_verifier_rejects_switch_case_aliases_after_width_normalization"_test = [] {
        auto verify_alias = [](const Type *selector_type,
                               SwitchInst::case_value_type lhs,
                               SwitchInst::case_value_type rhs) noexcept {
            Module module;
            auto *callable = module.create_callable(nullptr);
            auto *selector = callable->create_value_argument(selector_type);
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *switch_inst = builder.switch_(selector);
            auto *lhs_block = switch_inst->create_case_block(lhs);
            auto *rhs_block = switch_inst->create_case_block(rhs);
            auto *default_block = switch_inst->create_default_block();
            auto *merge_block = switch_inst->create_merge_block();
            for (auto *block : {lhs_block, rhs_block, default_block}) {
                builder.set_insertion_point(block);
                builder.br(merge_block);
            }
            builder.set_insertion_point(merge_block);
            builder.return_void();
            auto result = xir_verify_module(&module);
            expect(!result.succeeded());
            expect(has_verification_error(
                result, body, switch_inst,
                "Switch case values alias after selector-width normalization."));
        };
        verify_alias(Type::of<int8_t>(),
                     std::numeric_limits<uint64_t>::max(), uint64_t{0xffu});
        verify_alias(Type::of<uint8_t>(), uint64_t{0x1ffu}, uint64_t{0xffu});
    };

    "xir_verifier_rejects_noncanonical_switch_case_width"_test = [] {
        Module module;
        auto *callable = module.create_callable(nullptr);
        auto *wide_selector = callable->create_value_argument(Type::of<uint64_t>());
        auto *narrow_selector = callable->create_value_argument(Type::of<uint8_t>());
        auto *body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *switch_inst = builder.switch_(wide_selector);
        auto *case_block = switch_inst->create_case_block(0x1ffu);
        auto *default_block = switch_inst->create_default_block();
        auto *merge_block = switch_inst->create_merge_block();
        switch_inst->set_operand(
            SwitchInst::operand_index_value, narrow_selector);
        builder.set_insertion_point(case_block);
        builder.br(merge_block);
        builder.set_insertion_point(default_block);
        builder.br(merge_block);
        builder.set_insertion_point(merge_block);
        builder.return_void();
        auto result = xir_verify_module(&module);
        expect(!result.succeeded());
        expect(has_verification_error(
            result, body, switch_inst,
            "Switch case value is outside the selector bit width."));
    };

    "xir_verifier_rejects_invalid_switch_selectors"_test = [] {
        auto verify_selector = [](const Type *type, bool by_reference) {
            Module module;
            auto *callable = module.create_callable(nullptr);
            Value *selector = by_reference ?
                                  static_cast<Value *>(callable->create_reference_argument(type)) :
                                  static_cast<Value *>(callable->create_value_argument(type));
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *switch_inst = builder.switch_(selector);
            auto *default_block = switch_inst->create_default_block();
            auto *merge_block = switch_inst->create_merge_block();
            builder.set_insertion_point(default_block);
            builder.br(merge_block);
            builder.set_insertion_point(merge_block);
            builder.return_void();
            auto result = xir_verify_module(&module);
            expect(!result.succeeded());
            expect(has_verification_error(
                result, body, switch_inst,
                "Switch selector is not an integer/bool scalar rvalue."));
        };
        verify_selector(Type::of<float>(), false);
        verify_selector(Type::vector(Type::of<int32_t>(), 2u), false);
        verify_selector(Type::of<int32_t>(), true);
    };

    "xir_verifier_validates_ray_query_dispatch_operands"_test = [] {
        {
            Module module;
            auto *callable = module.create_callable(nullptr);
            auto *query = callable->create_reference_argument(
                Type::custom("LC_RayQueryAll"));
            auto *wrong_query = callable->create_reference_argument(
                Type::of<int32_t>());
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *dispatch = builder.ray_query_dispatch(query);
            auto *exit = callable->create_basic_block();
            dispatch->set_exit_block(exit);
            auto *surface = dispatch->create_on_surface_candidate_block();
            auto *procedural = dispatch->create_on_procedural_candidate_block();
            builder.set_insertion_point(surface);
            builder.br(body);
            builder.set_insertion_point(procedural);
            builder.br(body);
            builder.set_insertion_point(exit);
            builder.return_void();
            expect(xir_verify_module(&module).succeeded());

            dispatch->set_query_object(wrong_query);
            auto result = xir_verify_module(&module);
            expect(has_verification_error(
                result, body, dispatch,
                "Ray-query dispatch operands are invalid."));
        }
        {
            Module module;
            auto *query = module.create_undefined(
                Type::custom("LC_RayQueryAny"));
            auto *callable = module.create_callable(nullptr);
            auto *query_storage = callable->create_reference_argument(
                Type::custom("LC_RayQueryAny"));
            auto *body = callable->create_body_block();
            XIRBuilder builder;
            builder.set_insertion_point(body);
            auto *dispatch = builder.ray_query_dispatch(query_storage);
            auto *exit = callable->create_basic_block();
            dispatch->set_exit_block(exit);
            auto *surface = dispatch->create_on_surface_candidate_block();
            auto *procedural = dispatch->create_on_procedural_candidate_block();
            dispatch->set_operand(
                RayQueryDispatchInst::operand_index_exit_block, query);
            builder.set_insertion_point(surface);
            builder.br(body);
            builder.set_insertion_point(procedural);
            builder.br(body);
            builder.set_insertion_point(exit);
            builder.return_void();
            auto result = xir_verify_module(&module);
            expect(has_verification_error(
                result, body, dispatch,
                "Ray-query dispatch operands are invalid."));
        }
    };

    "xir_verifier_rejects_incomplete_ray_query_dispatch"_test = [] {
        Module module;
        auto *callable = module.create_callable(nullptr);
        auto *query = callable->create_reference_argument(
            Type::custom("LC_RayQueryAny"));
        auto *body = callable->create_body_block();
        XIRBuilder builder;
        builder.set_insertion_point(body);
        auto *dispatch = builder.ray_query_dispatch(query);
        auto *exit = callable->create_basic_block();
        dispatch->set_exit_block(exit);
        auto *surface = dispatch->create_on_surface_candidate_block();
        builder.set_insertion_point(surface);
        builder.br(body);
        builder.set_insertion_point(exit);
        builder.return_void();
        auto result = xir_verify_module(&module);
        expect(has_verification_error(
            result, body, dispatch,
            "Ray-query dispatch operands are invalid."));
    };
}

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    reg_xir_verifier();
    return 0;
}
