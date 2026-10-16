// Regression tests for discriminated counted-array lifetime contraction.

#include "ut/ut.hpp"

#include <algorithm>
#include <luisa/core/logging.h>

#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/instructions/coro.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coro_alloca_scope.h>
#include <luisa/xir/passes/coro_cfg_distill.h>
#include <luisa/xir/verifier.h>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::xir;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

enum class ReadMode {
    matching_tag,
    wrong_tag,
    unguarded,
    overwritten_tag
};

enum class RangeGuardMode {
    direct,
    conjunction,
    disjunction
};

[[nodiscard]] bool frame_contains(
    const CoroCfgDistillResult &cfg, Value *value) noexcept {
    return std::any_of(
        cfg.frame_values.begin(), cfg.frame_values.end(),
        [value](auto &&field) noexcept { return field.value == value; });
}

void check_discriminated_prefix(
    ReadMode mode, bool expect_contraction,
    bool reset_before_suspend = false,
    bool overwrite_after_resume = false,
    bool counter_before_arrays = false,
    RangeGuardMode range_guard_mode = RangeGuardMode::direct,
    bool copy_guarded_index = false,
    bool add_unrelated_tag_read = false) {
    Module module;
    auto *kernel = module.create_kernel();
    auto *kind = kernel->create_value_argument(Type::of<uint>());
    auto *read_index = kernel->create_value_argument(Type::of<uint>());
    auto *extra_guard = kernel->create_value_argument(Type::of<bool>());
    auto *entry = kernel->create_body_block();
    auto *resume = kernel->create_basic_block();
    auto *define_first = kernel->create_basic_block();
    auto *skip_first = kernel->create_basic_block();
    auto *append_second = kernel->create_basic_block();
    auto *inspect = kernel->create_basic_block();
    auto *copy_index = copy_guarded_index ?
                           kernel->create_basic_block() :
                           nullptr;
    auto *consume = kernel->create_basic_block();
    auto *done = kernel->create_basic_block();
    auto *array_type = Type::array(Type::of<uint>(), 4u);
    XIRBuilder builder;

    builder.set_insertion_point(entry);
    AllocaInst *count = nullptr;
    if (counter_before_arrays) {
        count = builder.alloca_local(Type::of<uint>());
        count->set_name("record_count");
    }
    auto *tag = builder.alloca_local(array_type);
    tag->set_name("discriminant");
    auto *payload = builder.alloca_local(array_type);
    payload->set_name("conditional_payload");
    if (count == nullptr) {
        count = builder.alloca_local(Type::of<uint>());
        count->set_name("record_count");
    }
    auto *ticket = builder.alloca_local(Type::of<uint>());
    ticket->set_name("allocation_ticket");
    auto *cursor = builder.alloca_local(Type::of<uint>());
    cursor->set_name("read_cursor");
    auto *copied_cursor = builder.alloca_local(Type::of<uint>());
    copied_cursor->set_name("copied_read_cursor");
    if (reset_before_suspend) {
        builder.store(
            count, module.create_constant_zero(Type::of<uint>()));
    }
    builder.coro_suspend(41u, "discriminated-prefix", nullptr);

    builder.set_insertion_point(resume);
    builder.coro_resume(41u, nullptr);
    if (!reset_before_suspend) {
        builder.store(
            count, module.create_constant_zero(Type::of<uint>()));
    } else if (overwrite_after_resume) {
        builder.store(
            count, module.create_constant_one(Type::of<uint>()));
    }

    // Append record 0. The tag is defined before publishing the record by
    // incrementing count; the payload is initialized afterwards through the
    // saved pre-increment ticket, and only for tag 1.
    auto *first_index = builder.load(Type::of<uint>(), count);
    builder.store(ticket, first_index);
    auto *first_tag = builder.gep(Type::of<uint>(), tag, {first_index});
    builder.store(first_tag, kind);
    auto *first_count = builder.load(Type::of<uint>(), count);
    auto *after_first = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {first_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, after_first);
    auto *is_payload_kind = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
        {kind, module.create_constant_one(Type::of<uint>())});
    builder.cond_br(is_payload_kind, define_first, skip_first);

    builder.set_insertion_point(define_first);
    auto *saved_ticket = builder.load(Type::of<uint>(), ticket);
    auto *first_payload = builder.gep(
        Type::of<uint>(), payload, {saved_ticket});
    builder.store(
        first_payload, module.create_constant_one(Type::of<uint>()));
    builder.br(append_second);

    builder.set_insertion_point(skip_first);
    builder.br(append_second);

    builder.set_insertion_point(append_second);
    if (mode == ReadMode::overwritten_tag) {
        // This destroys the discriminant/payload invariant on kind != 1.
        auto *old_ticket = builder.load(Type::of<uint>(), ticket);
        auto *old_tag = builder.gep(Type::of<uint>(), tag, {old_ticket});
        builder.store(old_tag, module.create_constant_one(Type::of<uint>()));
    }
    // Append record 1 with tag 2 and no payload. This forces the proof to
    // summarize an older record rather than tracking only the current tail.
    auto *second_index = builder.load(Type::of<uint>(), count);
    auto *second_tag = builder.gep(Type::of<uint>(), tag, {second_index});
    std::uint32_t tag_two = 2u;
    builder.store(
        second_tag, module.create_constant(Type::of<uint>(), &tag_two));
    auto *second_count = builder.load(Type::of<uint>(), count);
    auto *after_second = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {second_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, after_second);
    builder.store(cursor, read_index);
    auto *current_cursor = builder.load(Type::of<uint>(), cursor);
    auto *published_count = builder.load(Type::of<uint>(), count);
    auto *in_range = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_LESS,
        {current_cursor, published_count});
    Value *range_guard = in_range;
    if (range_guard_mode != RangeGuardMode::direct) {
        range_guard = builder.call(
            Type::of<bool>(),
            range_guard_mode == RangeGuardMode::conjunction ?
                ArithmeticOp::BINARY_BIT_AND :
                ArithmeticOp::BINARY_BIT_OR,
            {in_range, extra_guard});
    }
    builder.cond_br(range_guard, inspect, done);

    builder.set_insertion_point(inspect);
    if (mode == ReadMode::unguarded) {
        builder.br(consume);
    } else {
        auto *tag_index = builder.load(Type::of<uint>(), cursor);
        auto *selected_tag = builder.gep(
            Type::of<uint>(), tag, {tag_index});
        auto *tag_value = builder.load(Type::of<uint>(), selected_tag);
        auto expected_tag = mode == ReadMode::wrong_tag ? 2u : 1u;
        auto *matches = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
            {tag_value,
             module.create_constant(Type::of<uint>(), &expected_tag)});
        builder.cond_br(
            matches, copy_guarded_index ? copy_index : consume, done);
    }

    if (copy_guarded_index) {
        builder.set_insertion_point(copy_index);
        auto *guarded_cursor = builder.load(Type::of<uint>(), cursor);
        builder.store(copied_cursor, guarded_cursor);
        builder.br(consume);
    }

    builder.set_insertion_point(consume);
    auto *payload_index = builder.load(
        Type::of<uint>(),
        copy_guarded_index ? copied_cursor : cursor);
    auto *selected_payload = builder.gep(
        Type::of<uint>(), payload, {payload_index});
    static_cast<void>(builder.load(Type::of<uint>(), selected_payload));
    builder.br(done);

    builder.set_insertion_point(done);
    if (add_unrelated_tag_read) {
        // This read is deliberately not range-guarded. Moving `payload`
        // leaves the tag array and this pre-existing operation untouched, so
        // it is not a proof obligation for the payload's lifetime.
        auto *unrelated_tag = builder.gep(
            Type::of<uint>(), tag, {read_index});
        static_cast<void>(
            builder.load(Type::of<uint>(), unrelated_tag));
    }
    builder.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto before = coro_cfg_distill_pass_run_on_function(kernel);
    expect(before.succeeded());
    expect(frame_contains(before, payload));

    auto original_block = payload->parent_block();
    auto info = coro_alloca_scope_pass_run_on_function(kernel);
    expect(info.discriminated_prefix_proof_count ==
           (expect_contraction ? 1u : 0u));
    expect(info.rejected_prior_lifetime_observation_count >=
           (expect_contraction ? 0u : 1u));
    expect(payload->parent_block() ==
           (expect_contraction ? resume : original_block));
    expect(xir_verify_module(&module).succeeded());

    auto after = coro_cfg_distill_pass_run_on_function(kernel);
    expect(after.succeeded());
    expect(frame_contains(after, payload) == !expect_contraction);
}

void check_stale_uninitialized_discriminator_is_not_evidence() {
    Module module;
    auto *kernel = module.create_kernel();
    auto *entry = kernel->create_body_block();
    auto *resume = kernel->create_basic_block();
    auto *publish_unsafe = kernel->create_basic_block();
    auto *consume = kernel->create_basic_block();
    auto *done = kernel->create_basic_block();
    auto *array_type = Type::array(Type::of<uint>(), 4u);
    XIRBuilder builder;

    builder.set_insertion_point(entry);
    auto *tag = builder.alloca_local(array_type);
    tag->set_name("stale_discriminator_tag");
    auto *payload = builder.alloca_local(array_type);
    payload->set_name("stale_discriminator_payload");
    auto *count = builder.alloca_local(Type::of<uint>());
    count->set_name("stale_discriminator_count");
    auto *ticket = builder.alloca_local(Type::of<uint>());
    ticket->set_name("stale_discriminator_ticket");
    builder.coro_suspend(61u, "stale-discriminator", nullptr);

    builder.set_insertion_point(resume);
    builder.coro_resume(61u, nullptr);
    builder.store(count, module.create_constant_zero(Type::of<uint>()));

    // Publish one completely initialized record so `payload` is a genuine
    // counted/discriminated candidate.
    auto *first_index = builder.load(Type::of<uint>(), count);
    auto *first_payload = builder.gep(
        Type::of<uint>(), payload, {first_index});
    builder.store(
        first_payload, module.create_constant_one(Type::of<uint>()));
    auto *first_tag = builder.gep(Type::of<uint>(), tag, {first_index});
    std::uint32_t tag_two = 2u;
    builder.store(
        first_tag, module.create_constant(Type::of<uint>(), &tag_two));
    auto *first_count = builder.load(Type::of<uint>(), count);
    auto *after_first = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {first_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, after_first);

    // Read T[C] before that slot is initialized. The selected edge must not
    // turn this stale/undefined value into a durable tag constraint.
    auto *second_index = builder.load(Type::of<uint>(), count);
    builder.store(ticket, second_index);
    auto *uninitialized_tag = builder.gep(
        Type::of<uint>(), tag, {second_index});
    auto *stale_tag = builder.load(Type::of<uint>(), uninitialized_tag);
    auto *stale_is_not_one = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_NOT_EQUAL,
        {stale_tag, module.create_constant_one(Type::of<uint>())});
    builder.cond_br(stale_is_not_one, publish_unsafe, done);

    builder.set_insertion_point(publish_unsafe);
    auto *saved_ticket = builder.load(Type::of<uint>(), ticket);
    auto *second_tag = builder.gep(
        Type::of<uint>(), tag, {saved_ticket});
    builder.store(
        second_tag, module.create_constant_one(Type::of<uint>()));
    auto *old_count = builder.load(Type::of<uint>(), count);
    auto *after_second = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {old_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, after_second);
    builder.br(consume);

    builder.set_insertion_point(consume);
    auto *payload_index = builder.load(Type::of<uint>(), ticket);
    auto *undefined_payload = builder.gep(
        Type::of<uint>(), payload, {payload_index});
    static_cast<void>(
        builder.load(Type::of<uint>(), undefined_payload));
    builder.br(done);

    builder.set_insertion_point(done);
    builder.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto before = coro_cfg_distill_pass_run_on_function(kernel);
    expect(before.succeeded());
    expect(frame_contains(before, payload));

    auto original_block = payload->parent_block();
    auto info = coro_alloca_scope_pass_run_on_function(kernel);
    expect(info.discriminated_prefix_proof_count == 0u);
    expect(payload->parent_block() == original_block);
    expect(xir_verify_module(&module).succeeded());

    auto after = coro_cfg_distill_pass_run_on_function(kernel);
    expect(after.succeeded());
    expect(frame_contains(after, payload));
}

void check_guarded_select_fallback(bool initialize_fallback) {
    Module module;
    auto *kernel = module.create_kernel();
    auto *read_index = kernel->create_value_argument(Type::of<uint>());
    auto *entry = kernel->create_body_block();
    auto *resume = kernel->create_basic_block();
    auto *inspect = kernel->create_basic_block();
    auto *consume = kernel->create_basic_block();
    auto *done = kernel->create_basic_block();
    auto *array_type = Type::array(Type::of<uint>(), 4u);
    XIRBuilder builder;

    builder.set_insertion_point(entry);
    auto *tag = builder.alloca_local(array_type);
    tag->set_name("select_discriminant");
    auto *payload = builder.alloca_local(array_type);
    payload->set_name("select_payload");
    auto *count = builder.alloca_local(Type::of<uint>());
    count->set_name("select_count");
    auto *cursor = builder.alloca_local(Type::of<uint>());
    cursor->set_name("select_cursor");
    auto *selected = builder.alloca_local(Type::of<uint>());
    selected->set_name("selected_cursor");
    builder.coro_suspend(73u, "select-prefix", nullptr);

    builder.set_insertion_point(resume);
    builder.coro_resume(73u, nullptr);
    builder.store(count, module.create_constant_zero(Type::of<uint>()));

    // Index 3 is outside the published prefix. It models the permanent
    // fallback row selected by P[select(3, i, i<C)]. The negative form omits
    // exactly this definition while leaving the counted record transaction
    // unchanged.
    std::uint32_t fallback_index_value = 3u;
    auto *fallback_index = module.create_constant(
        Type::of<uint>(), &fallback_index_value);
    auto *fallback_tag = builder.gep(
        Type::of<uint>(), tag, {fallback_index});
    builder.store(
        fallback_tag, module.create_constant_one(Type::of<uint>()));
    if (initialize_fallback) {
        auto *fallback_payload = builder.gep(
            Type::of<uint>(), payload, {fallback_index});
        builder.store(
            fallback_payload,
            module.create_constant_one(Type::of<uint>()));
    }

    // Record zero has tag 1 and a payload. Record one has tag 2 but no
    // payload, so an ordinary initialized-prefix proof must fail while the
    // discriminated proof may still admit reads selected by tag 1.
    auto *first_index = builder.load(Type::of<uint>(), count);
    auto *first_tag = builder.gep(
        Type::of<uint>(), tag, {first_index});
    builder.store(
        first_tag, module.create_constant_one(Type::of<uint>()));
    auto *first_payload = builder.gep(
        Type::of<uint>(), payload, {first_index});
    builder.store(
        first_payload,
        module.create_constant_one(Type::of<uint>()));
    auto *first_count = builder.load(Type::of<uint>(), count);
    auto *after_first = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {first_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, after_first);

    std::uint32_t tag_two_value = 2u;
    auto *second_index = builder.load(Type::of<uint>(), count);
    auto *second_tag = builder.gep(
        Type::of<uint>(), tag, {second_index});
    builder.store(
        second_tag,
        module.create_constant(Type::of<uint>(), &tag_two_value));
    auto *second_count = builder.load(Type::of<uint>(), count);
    auto *after_second = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {second_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, after_second);

    builder.store(cursor, read_index);
    auto *current_cursor = builder.load(Type::of<uint>(), cursor);
    auto *published_count = builder.load(Type::of<uint>(), count);
    auto *in_range = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_LESS,
        {current_cursor, published_count});
    auto *safe_index = builder.call(
        Type::of<uint>(), ArithmeticOp::SELECT,
        {fallback_index, current_cursor, in_range});
    builder.store(selected, safe_index);
    auto *tag_index = builder.load(Type::of<uint>(), selected);
    auto *selected_tag = builder.gep(
        Type::of<uint>(), tag, {tag_index});
    auto *tag_value = builder.load(Type::of<uint>(), selected_tag);
    auto *matches = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
        {tag_value, module.create_constant_one(Type::of<uint>())});
    builder.cond_br(matches, inspect, done);

    builder.set_insertion_point(inspect);
    builder.br(consume);

    builder.set_insertion_point(consume);
    auto *payload_index = builder.load(Type::of<uint>(), selected);
    auto *selected_payload = builder.gep(
        Type::of<uint>(), payload, {payload_index});
    static_cast<void>(builder.load(Type::of<uint>(), selected_payload));
    builder.br(done);

    builder.set_insertion_point(done);
    builder.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto before = coro_cfg_distill_pass_run_on_function(kernel);
    expect(before.succeeded());
    expect(frame_contains(before, payload));

    auto original_block = payload->parent_block();
    auto info = coro_alloca_scope_pass_run_on_function(kernel);
    expect(info.discriminated_prefix_proof_count ==
           (initialize_fallback ? 1u : 0u));
    expect(payload->parent_block() ==
           (initialize_fallback ? resume : original_block));
    expect(xir_verify_module(&module).succeeded());

    auto after = coro_cfg_distill_pass_run_on_function(kernel);
    expect(after.succeeded());
    expect(frame_contains(after, payload) == !initialize_fallback);
}

void check_moved_preheader_insertion_snapshot() {
    Module module;
    auto *kernel = module.create_kernel();
    auto *kind = kernel->create_value_argument(Type::of<uint>());
    auto *read_index = kernel->create_value_argument(Type::of<uint>());
    auto *entry = kernel->create_body_block();
    auto *resume = kernel->create_basic_block();
    auto *loop_header = kernel->create_basic_block();
    auto *append = kernel->create_basic_block();
    auto *define_payload = kernel->create_basic_block();
    auto *skip_payload = kernel->create_basic_block();
    auto *test_range = kernel->create_basic_block();
    auto *inspect = kernel->create_basic_block();
    auto *consume = kernel->create_basic_block();
    auto *latch = kernel->create_basic_block();
    auto *done = kernel->create_basic_block();
    auto *array_type = Type::array(Type::of<uint>(), 4u);
    XIRBuilder builder;

    builder.set_insertion_point(entry);
    // This unrelated single-definition local is processed before payload.
    // Its lifetime start and store move into the resume preheader, thereby
    // changing the block location of the first legal insertion instruction.
    auto *preheader_value = builder.alloca_local(Type::of<uint>());
    auto *preheader_definition = builder.store(
        preheader_value, module.create_constant_one(Type::of<uint>()));
    auto *tag = builder.alloca_local(array_type);
    tag->set_name("discriminant");
    auto *payload = builder.alloca_local(array_type);
    payload->set_name("conditional_payload");
    auto *count = builder.alloca_local(Type::of<uint>());
    count->set_name("record_count");
    auto *ticket = builder.alloca_local(Type::of<uint>());
    auto *cursor = builder.alloca_local(Type::of<uint>());
    builder.coro_suspend(43u, "moved-preheader-insertion", nullptr);

    builder.set_insertion_point(resume);
    builder.coro_resume(43u, nullptr);
    static_cast<void>(builder.load(Type::of<uint>(), preheader_value));
    auto *reset = builder.store(
        count, module.create_constant_zero(Type::of<uint>()));
    builder.br(loop_header);

    builder.set_insertion_point(loop_header);
    auto *published = builder.load(Type::of<uint>(), count);
    auto *has_capacity = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_LESS,
        {published, module.create_constant_one(Type::of<uint>())});
    builder.cond_br(has_capacity, append, done);

    builder.set_insertion_point(append);
    auto *record_index = builder.load(Type::of<uint>(), count);
    builder.store(ticket, record_index);
    auto *record_tag = builder.gep(Type::of<uint>(), tag, {record_index});
    builder.store(record_tag, kind);
    auto *old_count = builder.load(Type::of<uint>(), count);
    auto *new_count = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {old_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, new_count);
    auto *is_payload_kind = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
        {kind, module.create_constant_one(Type::of<uint>())});
    builder.cond_br(is_payload_kind, define_payload, skip_payload);

    builder.set_insertion_point(define_payload);
    auto *saved_ticket = builder.load(Type::of<uint>(), ticket);
    auto *record_payload = builder.gep(
        Type::of<uint>(), payload, {saved_ticket});
    builder.store(
        record_payload, module.create_constant_one(Type::of<uint>()));
    builder.br(test_range);

    builder.set_insertion_point(skip_payload);
    builder.br(test_range);

    builder.set_insertion_point(test_range);
    builder.store(cursor, read_index);
    auto *current_cursor = builder.load(Type::of<uint>(), cursor);
    auto *current_count = builder.load(Type::of<uint>(), count);
    auto *in_range = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_LESS,
        {current_cursor, current_count});
    builder.cond_br(in_range, inspect, latch);

    builder.set_insertion_point(inspect);
    auto *tag_index = builder.load(Type::of<uint>(), cursor);
    auto *selected_tag = builder.gep(Type::of<uint>(), tag, {tag_index});
    auto *tag_value = builder.load(Type::of<uint>(), selected_tag);
    auto *matches = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
        {tag_value, module.create_constant_one(Type::of<uint>())});
    builder.cond_br(matches, consume, latch);

    builder.set_insertion_point(consume);
    auto *payload_index = builder.load(Type::of<uint>(), cursor);
    auto *selected_payload = builder.gep(
        Type::of<uint>(), payload, {payload_index});
    static_cast<void>(builder.load(Type::of<uint>(), selected_payload));
    builder.br(latch);

    builder.set_insertion_point(latch);
    builder.br(loop_header);

    builder.set_insertion_point(done);
    builder.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto before = coro_cfg_distill_pass_run_on_function(kernel);
    expect(before.succeeded());
    expect(frame_contains(before, payload));

    auto info = coro_alloca_scope_pass_run_on_function(kernel);
    expect(info.delayed_first_definition_count >= 1u);
    expect(preheader_value->parent_block() == resume);
    expect(preheader_definition->parent_block() == resume);
    expect(reset->parent_block() == resume);
    expect(info.discriminated_prefix_proof_count == 1u);
    expect(payload->parent_block() == resume);
    expect(xir_verify_module(&module).succeeded());

    auto after = coro_cfg_distill_pass_run_on_function(kernel);
    expect(after.succeeded());
    expect(!frame_contains(after, payload));
}

void check_compound_rollback_guard(
    bool conjunction, bool expect_contraction) {
    Module module;
    auto *kernel = module.create_kernel();
    auto *kind = kernel->create_value_argument(Type::of<uint>());
    auto *append_record = kernel->create_value_argument(Type::of<bool>());
    auto *extra_guard = kernel->create_value_argument(Type::of<bool>());
    auto *read_index = kernel->create_value_argument(Type::of<uint>());
    auto *entry = kernel->create_body_block();
    auto *resume = kernel->create_basic_block();
    auto *append = kernel->create_basic_block();
    auto *skip_append = kernel->create_basic_block();
    auto *define_payload = kernel->create_basic_block();
    auto *skip_payload = kernel->create_basic_block();
    auto *join = kernel->create_basic_block();
    auto *rollback = kernel->create_basic_block();
    auto *test_range = kernel->create_basic_block();
    auto *inspect = kernel->create_basic_block();
    auto *consume = kernel->create_basic_block();
    auto *done = kernel->create_basic_block();
    auto *array_type = Type::array(Type::of<uint>(), 4u);
    XIRBuilder builder;

    builder.set_insertion_point(entry);
    auto *tag = builder.alloca_local(array_type);
    auto *payload = builder.alloca_local(array_type);
    auto *count = builder.alloca_local(Type::of<uint>());
    auto *ticket = builder.alloca_local(Type::of<uint>());
    auto *cursor = builder.alloca_local(Type::of<uint>());
    builder.coro_suspend(47u, "compound-rollback-guard", nullptr);

    builder.set_insertion_point(resume);
    builder.coro_resume(47u, nullptr);
    builder.store(count, module.create_constant_zero(Type::of<uint>()));
    builder.cond_br(append_record, append, skip_append);

    builder.set_insertion_point(append);
    auto *record_index = builder.load(Type::of<uint>(), count);
    builder.store(ticket, record_index);
    auto *record_tag = builder.gep(Type::of<uint>(), tag, {record_index});
    builder.store(record_tag, kind);
    auto *old_count = builder.load(Type::of<uint>(), count);
    auto *new_count = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {old_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, new_count);
    auto *is_payload_kind = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
        {kind, module.create_constant_one(Type::of<uint>())});
    builder.cond_br(is_payload_kind, define_payload, skip_payload);

    builder.set_insertion_point(define_payload);
    auto *saved_ticket = builder.load(Type::of<uint>(), ticket);
    auto *record_payload = builder.gep(
        Type::of<uint>(), payload, {saved_ticket});
    builder.store(
        record_payload, module.create_constant_one(Type::of<uint>()));
    builder.br(join);

    builder.set_insertion_point(skip_payload);
    builder.br(join);

    builder.set_insertion_point(skip_append);
    builder.br(join);

    builder.set_insertion_point(join);
    auto *joined_count = builder.load(Type::of<uint>(), count);
    auto *nonzero = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_NOT_EQUAL,
        {joined_count, module.create_constant_zero(Type::of<uint>())});
    auto *rollback_guard = builder.call(
        Type::of<bool>(),
        conjunction ? ArithmeticOp::BINARY_BIT_AND :
                      ArithmeticOp::BINARY_BIT_OR,
        {nonzero, extra_guard});
    builder.cond_br(rollback_guard, rollback, test_range);

    builder.set_insertion_point(rollback);
    auto *rollback_count = builder.load(Type::of<uint>(), count);
    auto *decremented = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_SUB,
        {rollback_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, decremented);
    builder.br(test_range);

    builder.set_insertion_point(test_range);
    builder.store(cursor, read_index);
    auto *current_cursor = builder.load(Type::of<uint>(), cursor);
    auto *current_count = builder.load(Type::of<uint>(), count);
    auto *in_range = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_LESS,
        {current_cursor, current_count});
    builder.cond_br(in_range, inspect, done);

    builder.set_insertion_point(inspect);
    auto *tag_index = builder.load(Type::of<uint>(), cursor);
    auto *selected_tag = builder.gep(Type::of<uint>(), tag, {tag_index});
    auto *tag_value = builder.load(Type::of<uint>(), selected_tag);
    auto *matches = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
        {tag_value, module.create_constant_one(Type::of<uint>())});
    builder.cond_br(matches, consume, done);

    builder.set_insertion_point(consume);
    auto *payload_index = builder.load(Type::of<uint>(), cursor);
    auto *selected_payload = builder.gep(
        Type::of<uint>(), payload, {payload_index});
    static_cast<void>(builder.load(Type::of<uint>(), selected_payload));
    builder.br(done);

    builder.set_insertion_point(done);
    builder.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto before = coro_cfg_distill_pass_run_on_function(kernel);
    expect(before.succeeded());
    expect(frame_contains(before, payload));

    auto original_block = payload->parent_block();
    auto info = coro_alloca_scope_pass_run_on_function(kernel);
    expect(info.discriminated_prefix_proof_count ==
           (expect_contraction ? 1u : 0u));
    expect(payload->parent_block() ==
           (expect_contraction ? resume : original_block));
    expect(xir_verify_module(&module).succeeded());

    auto after = coro_cfg_distill_pass_run_on_function(kernel);
    expect(after.succeeded());
    expect(frame_contains(after, payload) == !expect_contraction);
}

void check_conditional_allocation_ticket(
    bool correlated_valid, bool expect_contraction) {
    Module module;
    auto *kernel = module.create_kernel();
    auto *kind = kernel->create_value_argument(Type::of<uint>());
    auto *has_capacity = kernel->create_value_argument(Type::of<bool>());
    auto *unrelated_valid = kernel->create_value_argument(Type::of<bool>());
    auto *read_index = kernel->create_value_argument(Type::of<uint>());
    auto *entry = kernel->create_body_block();
    auto *resume = kernel->create_basic_block();
    auto *allocate = kernel->create_basic_block();
    auto *exhausted = kernel->create_basic_block();
    auto *allocation_join = kernel->create_basic_block();
    auto *define_payload = kernel->create_basic_block();
    auto *test_range = kernel->create_basic_block();
    auto *consume = kernel->create_basic_block();
    auto *done = kernel->create_basic_block();
    auto *array_type = Type::array(Type::of<uint>(), 4u);
    XIRBuilder builder;

    builder.set_insertion_point(entry);
    auto *tag = builder.alloca_local(array_type);
    tag->set_name("conditional_allocation_tag");
    auto *payload = builder.alloca_local(array_type);
    payload->set_name("conditional_allocation_payload");
    auto *count = builder.alloca_local(Type::of<uint>());
    count->set_name("conditional_allocation_count");
    auto *ticket = builder.alloca_local(Type::of<uint>());
    ticket->set_name("conditional_allocation_ticket");
    auto *valid = builder.alloca_local(Type::of<bool>());
    valid->set_name("conditional_allocation_valid");
    auto *returned_ticket = builder.alloca_local(Type::of<uint>());
    returned_ticket->set_name("returned_allocation_ticket");
    auto *returned_valid = builder.alloca_local(Type::of<bool>());
    returned_valid->set_name("returned_allocation_valid");
    auto *cursor = builder.alloca_local(Type::of<uint>());
    cursor->set_name("conditional_allocation_cursor");
    builder.coro_suspend(53u, "conditional-allocation-ticket", nullptr);

    builder.set_insertion_point(resume);
    builder.coro_resume(53u, nullptr);
    builder.store(count, module.create_constant_zero(Type::of<uint>()));
    builder.store(valid, module.create_constant_zero(Type::of<bool>()));
    builder.store(ticket, module.create_constant_zero(Type::of<uint>()));
    builder.cond_br(has_capacity, allocate, exhausted);

    builder.set_insertion_point(allocate);
    auto *allocated_ticket = builder.load(Type::of<uint>(), count);
    builder.store(ticket, allocated_ticket);
    auto *record_tag = builder.gep(
        Type::of<uint>(), tag, {allocated_ticket});
    builder.store(record_tag, kind);
    auto *old_count = builder.load(Type::of<uint>(), count);
    auto *new_count = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {old_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, new_count);
    builder.store(valid, module.create_constant_one(Type::of<bool>()));
    builder.br(allocation_join);

    builder.set_insertion_point(exhausted);
    builder.br(allocation_join);

    // Model returning Allocation by value: both fields pass through fresh
    // scalar locals after the success/failure join. The Boolean field must
    // carry the path correlation that makes the returned ticket a published
    // record exactly when valid is true.
    builder.set_insertion_point(allocation_join);
    auto *joined_ticket = builder.load(Type::of<uint>(), ticket);
    builder.store(returned_ticket, joined_ticket);
    auto *joined_valid = builder.load(Type::of<bool>(), valid);
    builder.store(
        returned_valid,
        correlated_valid ? static_cast<Value *>(joined_valid) :
                           static_cast<Value *>(unrelated_valid));
    auto *selected_valid = builder.load(Type::of<bool>(), returned_valid);
    builder.cond_br(selected_valid, define_payload, test_range);

    builder.set_insertion_point(define_payload);
    auto *payload_ticket = builder.load(Type::of<uint>(), returned_ticket);
    auto *record_payload = builder.gep(
        Type::of<uint>(), payload, {payload_ticket});
    builder.store(
        record_payload, module.create_constant_one(Type::of<uint>()));
    builder.br(test_range);

    builder.set_insertion_point(test_range);
    builder.store(cursor, read_index);
    auto *current_cursor = builder.load(Type::of<uint>(), cursor);
    auto *published_count = builder.load(Type::of<uint>(), count);
    auto *in_range = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_LESS,
        {current_cursor, published_count});
    builder.cond_br(in_range, consume, done);

    builder.set_insertion_point(consume);
    auto *payload_index = builder.load(Type::of<uint>(), cursor);
    auto *selected_payload = builder.gep(
        Type::of<uint>(), payload, {payload_index});
    static_cast<void>(builder.load(Type::of<uint>(), selected_payload));
    builder.br(done);

    builder.set_insertion_point(done);
    builder.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto before = coro_cfg_distill_pass_run_on_function(kernel);
    expect(before.succeeded());
    expect(frame_contains(before, payload));

    auto original_block = payload->parent_block();
    auto info = coro_alloca_scope_pass_run_on_function(kernel);
    expect(info.discriminated_prefix_proof_count ==
           (expect_contraction ? 1u : 0u));
    expect(payload->parent_block() ==
           (expect_contraction ? resume : original_block));
    expect(xir_verify_module(&module).succeeded());

    auto after = coro_cfg_distill_pass_run_on_function(kernel);
    expect(after.succeeded());
    expect(frame_contains(after, payload) == !expect_contraction);
}

void check_conditional_extra_allocation_transaction(
    bool rollback_on_extra_failure, bool exact_rollback_guard,
    bool expect_contraction) {
    Module module;
    auto *kernel = module.create_kernel();
    auto *owner_capacity = kernel->create_value_argument(Type::of<bool>());
    auto *extra_capacity = kernel->create_value_argument(Type::of<bool>());
    auto *unrelated_rollback_guard =
        kernel->create_value_argument(Type::of<bool>());
    auto *read_index = kernel->create_value_argument(Type::of<uint>());
    auto *entry = kernel->create_body_block();
    auto *resume = kernel->create_basic_block();
    auto *allocate_owner = kernel->create_basic_block();
    auto *owner_exhausted = kernel->create_basic_block();
    auto *owner_join = kernel->create_basic_block();
    auto *test_extra = kernel->create_basic_block();
    auto *extra_success = kernel->create_basic_block();
    auto *extra_failure = kernel->create_basic_block();
    auto *rollback_owner = kernel->create_basic_block();
    auto *after_extra = kernel->create_basic_block();
    auto *define_owner = kernel->create_basic_block();
    auto *append_second = kernel->create_basic_block();
    auto *inspect = kernel->create_basic_block();
    auto *consume = kernel->create_basic_block();
    auto *done = kernel->create_basic_block();
    auto *array_type = Type::array(Type::of<uint>(), 4u);
    XIRBuilder builder;

    builder.set_insertion_point(entry);
    auto *tag = builder.alloca_local(array_type);
    tag->set_name("extra_transaction_tag");
    auto *payload = builder.alloca_local(array_type);
    payload->set_name("extra_transaction_payload");
    auto *count = builder.alloca_local(Type::of<uint>());
    count->set_name("extra_transaction_count");
    auto *owner_ticket = builder.alloca_local(Type::of<uint>());
    owner_ticket->set_name("extra_transaction_owner_ticket");
    auto *owner_valid = builder.alloca_local(Type::of<bool>());
    owner_valid->set_name("extra_transaction_owner_valid");
    auto *returned_ticket = builder.alloca_local(Type::of<uint>());
    returned_ticket->set_name("extra_transaction_returned_ticket");
    auto *returned_valid = builder.alloca_local(Type::of<bool>());
    returned_valid->set_name("extra_transaction_returned_valid");
    auto *extra_valid = builder.alloca_local(Type::of<bool>());
    extra_valid->set_name("extra_transaction_extra_valid");
    auto *cursor = builder.alloca_local(Type::of<uint>());
    cursor->set_name("extra_transaction_cursor");
    builder.coro_suspend(59u, "conditional-extra-transaction", nullptr);

    builder.set_insertion_point(resume);
    builder.coro_resume(59u, nullptr);
    builder.store(count, module.create_constant_zero(Type::of<uint>()));
    builder.store(owner_ticket,
                  module.create_constant_zero(Type::of<uint>()));
    builder.store(owner_valid,
                  module.create_constant_zero(Type::of<bool>()));
    builder.store(extra_valid,
                  module.create_constant_zero(Type::of<bool>()));
    builder.cond_br(owner_capacity, allocate_owner, owner_exhausted);

    builder.set_insertion_point(allocate_owner);
    auto *ticket = builder.load(Type::of<uint>(), count);
    builder.store(owner_ticket, ticket);
    auto *owner_tag = builder.gep(Type::of<uint>(), tag, {ticket});
    builder.store(owner_tag, module.create_constant_zero(Type::of<uint>()));
    auto *old_count = builder.load(Type::of<uint>(), count);
    auto *published_count = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {old_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, published_count);
    builder.store(owner_valid,
                  module.create_constant_one(Type::of<bool>()));
    builder.br(owner_join);

    builder.set_insertion_point(owner_exhausted);
    builder.br(owner_join);

    // Model returning Allocation by value before allocate_extra consumes it.
    builder.set_insertion_point(owner_join);
    auto *joined_ticket = builder.load(Type::of<uint>(), owner_ticket);
    builder.store(returned_ticket, joined_ticket);
    auto *joined_valid = builder.load(Type::of<bool>(), owner_valid);
    builder.store(returned_valid, joined_valid);
    auto *has_owner = builder.load(Type::of<bool>(), returned_valid);
    builder.cond_br(has_owner, test_extra, after_extra);

    builder.set_insertion_point(test_extra);
    builder.cond_br(extra_capacity, extra_success, extra_failure);

    builder.set_insertion_point(extra_success);
    builder.store(extra_valid,
                  module.create_constant_one(Type::of<bool>()));
    builder.br(after_extra);

    builder.set_insertion_point(extra_failure);
    if (rollback_on_extra_failure) {
        auto *current_count = builder.load(Type::of<uint>(), count);
        auto *count_positive = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_NOT_EQUAL,
            {current_count, module.create_constant_zero(Type::of<uint>())});
        auto *saved_owner = builder.load(Type::of<uint>(), returned_ticket);
        auto *owner_end = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_ADD,
            {saved_owner, module.create_constant_one(Type::of<uint>())});
        auto *published_end = builder.load(Type::of<uint>(), count);
        auto *owner_is_tail = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
            {owner_end, published_end});
        auto *returned_owner_is_valid =
            builder.load(Type::of<bool>(), returned_valid);
        auto *valid_owner_with_positive_count = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_BIT_AND,
            {returned_owner_is_valid, count_positive});
        auto *can_rollback = builder.call(
            Type::of<bool>(), ArithmeticOp::BINARY_BIT_AND,
            {valid_owner_with_positive_count,
             exact_rollback_guard ?
                 static_cast<Value *>(owner_is_tail) :
                 static_cast<Value *>(unrelated_rollback_guard)});
        builder.cond_br(can_rollback, rollback_owner, after_extra);
    } else {
        builder.br(after_extra);
    }

    builder.set_insertion_point(rollback_owner);
    if (rollback_on_extra_failure) {
        auto *current_count = builder.load(Type::of<uint>(), count);
        auto *rolled_back = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_SUB,
            {current_count, module.create_constant_one(Type::of<uint>())});
        builder.store(count, rolled_back);
    }
    builder.br(after_extra);

    builder.set_insertion_point(after_extra);
    auto *has_extra = builder.load(Type::of<bool>(), extra_valid);
    builder.cond_br(has_extra, define_owner, append_second);

    builder.set_insertion_point(define_owner);
    auto *owner_index = builder.load(Type::of<uint>(), returned_ticket);
    auto *owner_payload = builder.gep(
        Type::of<uint>(), payload, {owner_index});
    builder.store(owner_payload,
                  module.create_constant_one(Type::of<uint>()));
    auto *initialized_owner_tag = builder.gep(
        Type::of<uint>(), tag, {owner_index});
    builder.store(initialized_owner_tag,
                  module.create_constant_one(Type::of<uint>()));
    builder.br(append_second);

    // Append one fully initialized record so a surviving owner becomes an
    // older record rather than the specially tracked current tail.
    builder.set_insertion_point(append_second);
    auto *second_index = builder.load(Type::of<uint>(), count);
    auto *second_payload = builder.gep(
        Type::of<uint>(), payload, {second_index});
    builder.store(second_payload,
                  module.create_constant_one(Type::of<uint>()));
    auto *second_tag = builder.gep(Type::of<uint>(), tag, {second_index});
    std::uint32_t tag_two = 2u;
    builder.store(second_tag,
                  module.create_constant(Type::of<uint>(), &tag_two));
    auto *before_second = builder.load(Type::of<uint>(), count);
    auto *after_second = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {before_second, module.create_constant_one(Type::of<uint>())});
    builder.store(count, after_second);
    builder.store(cursor, read_index);
    auto *selected_index = builder.load(Type::of<uint>(), cursor);
    auto *final_count = builder.load(Type::of<uint>(), count);
    auto *in_range = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_LESS,
        {selected_index, final_count});
    builder.cond_br(in_range, inspect, done);

    builder.set_insertion_point(inspect);
    builder.br(consume);

    builder.set_insertion_point(consume);
    auto *payload_index = builder.load(Type::of<uint>(), cursor);
    auto *selected_payload = builder.gep(
        Type::of<uint>(), payload, {payload_index});
    static_cast<void>(builder.load(Type::of<uint>(), selected_payload));
    builder.br(done);

    builder.set_insertion_point(done);
    builder.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto before = coro_cfg_distill_pass_run_on_function(kernel);
    expect(before.succeeded());
    expect(frame_contains(before, payload));

    auto original_block = payload->parent_block();
    auto info = coro_alloca_scope_pass_run_on_function(kernel);
    expect(info.discriminated_prefix_proof_count ==
           (expect_contraction ? 1u : 0u));
    expect(payload->parent_block() ==
           (expect_contraction ? resume : original_block));
    expect(xir_verify_module(&module).succeeded());

    auto after = coro_cfg_distill_pass_run_on_function(kernel);
    expect(after.succeeded());
    expect(frame_contains(after, payload) == !expect_contraction);
}

void check_masked_flag_implies_nonempty_prefix(
    bool set_flag_only_after_publication,
    bool expect_contraction) {
    Module module;
    auto *kernel = module.create_kernel();
    auto *has_capacity = kernel->create_value_argument(Type::of<bool>());
    auto *alternate_index = kernel->create_value_argument(Type::of<uint>());
    auto *entry = kernel->create_body_block();
    auto *resume = kernel->create_basic_block();
    auto *allocate = kernel->create_basic_block();
    auto *exhausted = kernel->create_basic_block();
    auto *after_payload = kernel->create_basic_block();
    auto *test_sample = kernel->create_basic_block();
    auto *test_alternate = kernel->create_basic_block();
    auto *select_alternate = kernel->create_basic_block();
    auto *use_default = kernel->create_basic_block();
    auto *inspect = kernel->create_basic_block();
    auto *consume = kernel->create_basic_block();
    auto *done = kernel->create_basic_block();
    auto *array_type = Type::array(Type::of<uint>(), 4u);
    XIRBuilder builder;

    builder.set_insertion_point(entry);
    auto *tag = builder.alloca_local(array_type);
    tag->set_name("masked_flag_tag");
    auto *payload = builder.alloca_local(array_type);
    payload->set_name("masked_flag_payload");
    auto *count = builder.alloca_local(Type::of<uint>());
    count->set_name("masked_flag_count");
    auto *flags = builder.alloca_local(Type::of<uint>());
    flags->set_name("masked_flag_bits");
    auto *sampled = builder.alloca_local(Type::of<uint>());
    sampled->set_name("masked_flag_sampled");
    auto *ticket = builder.alloca_local(Type::of<uint>());
    ticket->set_name("masked_flag_ticket");
    auto *candidate = builder.alloca_local(Type::of<uint>());
    candidate->set_name("masked_flag_candidate");
    // The flag invariant intentionally starts before the semantic lifetime
    // candidate. The XIR pass must recover it through ordinary whole-CFG
    // dataflow; an application-side scratch-lifetime marker would merely
    // hide this compiler obligation.
    builder.store(flags, module.create_constant_zero(Type::of<uint>()));
    builder.coro_suspend(67u, "masked-flag-prefix", nullptr);

    builder.set_insertion_point(resume);
    builder.coro_resume(67u, nullptr);
    builder.store(count, module.create_constant_zero(Type::of<uint>()));
    if (!set_flag_only_after_publication) {
        builder.store(flags, module.create_constant_one(Type::of<uint>()));
    }
    builder.cond_br(has_capacity, allocate, exhausted);

    builder.set_insertion_point(allocate);
    auto *record_index = builder.load(Type::of<uint>(), count);
    builder.store(ticket, record_index);
    auto *record_tag = builder.gep(Type::of<uint>(), tag, {record_index});
    builder.store(
        record_tag, module.create_constant_one(Type::of<uint>()));
    auto *old_count = builder.load(Type::of<uint>(), count);
    auto *new_count = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {old_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, new_count);
    auto *saved_ticket = builder.load(Type::of<uint>(), ticket);
    auto *defined_payload = builder.gep(
        Type::of<uint>(), payload, {saved_ticket});
    builder.store(
        defined_payload, module.create_constant_one(Type::of<uint>()));
    if (set_flag_only_after_publication) {
        auto *old_flags = builder.load(Type::of<uint>(), flags);
        auto *new_flags = builder.call(
            Type::of<uint>(), ArithmeticOp::BINARY_BIT_OR,
            {old_flags, module.create_constant_one(Type::of<uint>())});
        builder.store(flags, new_flags);
    }
    builder.br(after_payload);

    builder.set_insertion_point(after_payload);
    builder.br(test_sample);

    builder.set_insertion_point(exhausted);
    builder.br(test_sample);

    builder.set_insertion_point(test_sample);
    auto *current_flags = builder.load(Type::of<uint>(), flags);
    auto *masked_flags = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_BIT_AND,
        {current_flags, module.create_constant_one(Type::of<uint>())});
    auto *has_scatter = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_NOT_EQUAL,
        {masked_flags, module.create_constant_zero(Type::of<uint>())});
    builder.cond_br(has_scatter, test_alternate, done);

    // This models the picker join: the default index is zero, while a
    // selected alternate is copied only from an edge proving I < C.
    builder.set_insertion_point(test_alternate);
    builder.store(sampled, module.create_constant_zero(Type::of<uint>()));
    builder.store(candidate, alternate_index);
    auto *candidate_value = builder.load(Type::of<uint>(), candidate);
    auto *published_count = builder.load(Type::of<uint>(), count);
    auto *candidate_in_range = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_LESS,
        {candidate_value, published_count});
    builder.cond_br(candidate_in_range, select_alternate, use_default);

    builder.set_insertion_point(select_alternate);
    auto *selected_candidate = builder.load(Type::of<uint>(), candidate);
    builder.store(sampled, selected_candidate);
    builder.br(inspect);

    builder.set_insertion_point(use_default);
    builder.br(inspect);

    builder.set_insertion_point(inspect);
    auto *tag_index = builder.load(Type::of<uint>(), sampled);
    auto *selected_tag = builder.gep(Type::of<uint>(), tag, {tag_index});
    auto *selected_tag_value = builder.load(Type::of<uint>(), selected_tag);
    auto *has_payload = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
        {selected_tag_value,
         module.create_constant_one(Type::of<uint>())});
    builder.cond_br(has_payload, consume, done);

    builder.set_insertion_point(consume);
    auto *payload_index = builder.load(Type::of<uint>(), sampled);
    auto *selected_payload = builder.gep(
        Type::of<uint>(), payload, {payload_index});
    static_cast<void>(
        builder.load(Type::of<uint>(), selected_payload));
    builder.br(done);

    builder.set_insertion_point(done);
    builder.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto before = coro_cfg_distill_pass_run_on_function(kernel);
    expect(before.succeeded());
    expect(frame_contains(before, payload));

    auto original_block = payload->parent_block();
    auto info = coro_alloca_scope_pass_run_on_function(kernel);
    expect(info.discriminated_prefix_proof_count ==
           (expect_contraction ? 1u : 0u));
    expect(payload->parent_block() ==
           (expect_contraction ? resume : original_block));
    expect(xir_verify_module(&module).succeeded());

    auto after = coro_cfg_distill_pass_run_on_function(kernel);
    expect(after.succeeded());
    expect(frame_contains(after, payload) == !expect_contraction);
}

void check_rolled_back_record_remains_physically_defined(
    bool write_payload, bool expect_contraction) {
    Module module;
    auto *kernel = module.create_kernel();
    auto *has_capacity = kernel->create_value_argument(Type::of<bool>());
    auto *entry = kernel->create_body_block();
    auto *resume = kernel->create_basic_block();
    auto *allocate = kernel->create_basic_block();
    auto *exhausted = kernel->create_basic_block();
    auto *test_sample = kernel->create_basic_block();
    auto *inspect = kernel->create_basic_block();
    auto *consume = kernel->create_basic_block();
    auto *done = kernel->create_basic_block();
    auto *array_type = Type::array(Type::of<uint>(), 4u);
    XIRBuilder builder;

    builder.set_insertion_point(entry);
    auto *tag = builder.alloca_local(array_type);
    tag->set_name("rolled_back_physical_tag");
    auto *payload = builder.alloca_local(array_type);
    payload->set_name("rolled_back_physical_payload");
    auto *count = builder.alloca_local(Type::of<uint>());
    count->set_name("rolled_back_physical_count");
    auto *flags = builder.alloca_local(Type::of<uint>());
    flags->set_name("rolled_back_physical_flags");
    auto *ticket = builder.alloca_local(Type::of<uint>());
    ticket->set_name("rolled_back_physical_ticket");
    auto *sampled = builder.alloca_local(Type::of<uint>());
    sampled->set_name("rolled_back_physical_sampled");
    builder.store(flags, module.create_constant_zero(Type::of<uint>()));
    builder.coro_suspend(71u, "rolled-back-physical-record", nullptr);

    builder.set_insertion_point(resume);
    builder.coro_resume(71u, nullptr);
    builder.store(count, module.create_constant_zero(Type::of<uint>()));
    builder.cond_br(has_capacity, allocate, exhausted);

    // Publish record zero, set a runtime closure flag, then model a failed
    // extra allocation rolling the semantic counter back to zero. The arrays
    // are ordinary local storage: rollback changes membership in [0, C), but
    // does not erase the physical record at P[C].
    builder.set_insertion_point(allocate);
    auto *record_index = builder.load(Type::of<uint>(), count);
    builder.store(ticket, record_index);
    auto *record_tag = builder.gep(Type::of<uint>(), tag, {record_index});
    builder.store(
        record_tag, module.create_constant_one(Type::of<uint>()));
    auto *old_count = builder.load(Type::of<uint>(), count);
    auto *published_count = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_ADD,
        {old_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, published_count);
    if (write_payload) {
        auto *payload_index = builder.load(Type::of<uint>(), ticket);
        auto *record_payload = builder.gep(
            Type::of<uint>(), payload, {payload_index});
        builder.store(
            record_payload,
            module.create_constant_one(Type::of<uint>()));
    }
    auto *old_flags = builder.load(Type::of<uint>(), flags);
    auto *new_flags = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_BIT_OR,
        {old_flags, module.create_constant_one(Type::of<uint>())});
    builder.store(flags, new_flags);
    auto *nonempty_count = builder.load(Type::of<uint>(), count);
    auto *rolled_back_count = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_SUB,
        {nonempty_count, module.create_constant_one(Type::of<uint>())});
    builder.store(count, rolled_back_count);
    builder.br(test_sample);

    builder.set_insertion_point(exhausted);
    builder.br(test_sample);

    builder.set_insertion_point(test_sample);
    auto *current_flags = builder.load(Type::of<uint>(), flags);
    auto *masked_flags = builder.call(
        Type::of<uint>(), ArithmeticOp::BINARY_BIT_AND,
        {current_flags, module.create_constant_one(Type::of<uint>())});
    auto *has_scatter = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_NOT_EQUAL,
        {masked_flags, module.create_constant_zero(Type::of<uint>())});
    builder.cond_br(has_scatter, inspect, done);

    builder.set_insertion_point(inspect);
    builder.store(sampled, module.create_constant_zero(Type::of<uint>()));
    auto *tag_index = builder.load(Type::of<uint>(), sampled);
    auto *selected_tag = builder.gep(Type::of<uint>(), tag, {tag_index});
    auto *selected_tag_value = builder.load(Type::of<uint>(), selected_tag);
    auto *has_payload = builder.call(
        Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
        {selected_tag_value,
         module.create_constant_one(Type::of<uint>())});
    builder.cond_br(has_payload, consume, done);

    builder.set_insertion_point(consume);
    auto *payload_index = builder.load(Type::of<uint>(), sampled);
    auto *selected_payload = builder.gep(
        Type::of<uint>(), payload, {payload_index});
    static_cast<void>(
        builder.load(Type::of<uint>(), selected_payload));
    builder.br(done);

    builder.set_insertion_point(done);
    builder.return_void();

    expect(xir_verify_module(&module).succeeded());
    auto before = coro_cfg_distill_pass_run_on_function(kernel);
    expect(before.succeeded());
    expect(frame_contains(before, payload));

    auto *original_block = payload->parent_block();
    auto info = coro_alloca_scope_pass_run_on_function(kernel);
    if ((payload->parent_block() != original_block) != expect_contraction) {
        LUISA_INFO("Physical payload scope diagnostic: discriminated={} "
                   "ordinary={} prefix={} rejected={} moved_to_consume={} "
                   "moved_to_resume={} alloca_intra={} alloca_cross={}.",
                   info.discriminated_prefix_proof_count,
                   info.definite_initialization_proof_count,
                   info.initialized_prefix_proof_count,
                   info.rejected_prior_lifetime_observation_count,
                   payload->parent_block() == consume,
                   payload->parent_block() == resume,
                   info.intra_block_contraction_count,
                   info.cross_block_contraction_count);
    }
    expect(payload->parent_block() ==
           (expect_contraction ? resume : original_block));
    expect(info.rejected_prior_lifetime_observation_count >=
           (expect_contraction ? 0u : 1u));
    expect(xir_verify_module(&module).succeeded());

    auto after = coro_cfg_distill_pass_run_on_function(kernel);
    expect(after.succeeded());
    expect(frame_contains(after, payload) == !expect_contraction);
}

void register_discriminated_prefix_tests() {
    "unwritten_static_payload_is_not_defined_by_bitset_capacity"_test = [] {
        // Exercise the words/bits where a brace-initialized vector of
        // {word_count, 0} would manufacture definition evidence. None of
        // these payload arrays has a store in any lifetime.
        for (auto capacity : {1u, 4u, 64u, 65u, 128u, 255u}) {
            for (auto index = 0u; index < std::min(capacity, 4u); ++index) {
                Module module;
                auto *kernel = module.create_kernel();
                auto *entry = kernel->create_body_block();
                auto *resume = kernel->create_basic_block();
                auto *publish = kernel->create_basic_block();
                auto *array_type = Type::array(Type::of<uint>(), capacity);
                XIRBuilder builder;
                builder.set_insertion_point(entry);
                auto *tag = builder.alloca_local(array_type);
                auto *payload = builder.alloca_local(array_type);
                auto *count = builder.alloca_local(Type::of<uint>());
                auto *zero = module.create_constant_zero(Type::of<uint>());
                auto *one = module.create_constant_one(Type::of<uint>());
                builder.coro_suspend(72u, "unwritten-static-payload", nullptr);
                builder.set_insertion_point(resume);
                builder.coro_resume(72u, nullptr);
                builder.store(count, zero);
                builder.br(publish);
                builder.set_insertion_point(publish);
                auto *ticket = builder.load(Type::of<uint>(), count);
                builder.store(builder.gep(Type::of<uint>(), tag, {ticket}), one);
                builder.store(count, builder.call(
                    Type::of<uint>(), ArithmeticOp::BINARY_ADD, {ticket, one}));
                auto *offset = module.create_constant(Type::of<uint>(), &index);
                auto *address = builder.gep(Type::of<uint>(), payload, {offset});
                static_cast<void>(builder.load(Type::of<uint>(), address));
                builder.return_void();
                expect(xir_verify_module(&module).succeeded());
                auto before = coro_cfg_distill_pass_run_on_function(kernel);
                expect(before.succeeded());
                expect(frame_contains(before, payload));
                auto info = coro_alloca_scope_pass_run_on_function(kernel);
                expect(info.discriminated_prefix_candidate_count != 0u)
                    << "the negative witness must exercise the prefix domain";
                expect(info.discriminated_prefix_proof_count == 0u)
                    << "no initialization may be inferred from bitset capacity";
                expect(payload->parent_block() == entry)
                    << "an unproved payload lifetime must stay unchanged";
                expect(xir_verify_module(&module).succeeded());
            }
        }
    };

    "matching_tag_contracts_conditional_payload_prefix"_test = [] {
        check_discriminated_prefix(ReadMode::matching_tag, true);
    };

    "different_tag_cannot_justify_payload_read"_test = [] {
        check_discriminated_prefix(ReadMode::wrong_tag, false);
    };

    "unguarded_conditional_payload_read_is_rejected"_test = [] {
        check_discriminated_prefix(ReadMode::unguarded, false);
    };

    "tag_overwrite_without_payload_definition_is_rejected"_test = [] {
        check_discriminated_prefix(ReadMode::overwritten_tag, false);
    };

    "dominating_pre_suspend_zero_seeds_empty_prefix"_test = [] {
        check_discriminated_prefix(ReadMode::matching_tag, true, true);
    };

    "post_resume_nonzero_overwrite_rejects_pre_suspend_zero"_test = [] {
        check_discriminated_prefix(
            ReadMode::matching_tag, false, true, true);
    };

    "true_conjunction_exposes_index_bound"_test = [] {
        check_discriminated_prefix(
            ReadMode::matching_tag, true, false, false, false,
            RangeGuardMode::conjunction);
    };

    "true_disjunction_does_not_expose_index_bound"_test = [] {
        check_discriminated_prefix(
            ReadMode::matching_tag, false, false, false, false,
            RangeGuardMode::disjunction);
    };

    "scalar_copy_preserves_guarded_tag_constraint"_test = [] {
        check_discriminated_prefix(
            ReadMode::matching_tag, true, false, false, false,
            RangeGuardMode::direct, true);
    };

    "unrelated_tag_read_is_not_a_payload_lifetime_obligation"_test = [] {
        check_discriminated_prefix(
            ReadMode::matching_tag, true, false, false, false,
            RangeGuardMode::direct, false, true);
    };

    "stale_uninitialized_tag_cannot_justify_payload_read"_test = [] {
        check_stale_uninitialized_discriminator_is_not_evidence();
    };

    "true_conjunction_proves_counter_positive_for_rollback"_test = [] {
        check_compound_rollback_guard(true, true);
    };

    "true_disjunction_does_not_prove_counter_positive_for_rollback"_test = [] {
        check_compound_rollback_guard(false, false);
    };

    "copied_valid_recovers_conditional_allocation_ticket"_test = [] {
        check_conditional_allocation_ticket(true, true);
    };

    "unrelated_valid_cannot_recover_conditional_allocation_ticket"_test = [] {
        check_conditional_allocation_ticket(false, false);
    };

    "extra_failure_rollback_or_successful_definition_closes_transaction"_test = [] {
        check_conditional_extra_allocation_transaction(true, true, true);
    };

    "extra_failure_without_rollback_leaves_undefined_record"_test = [] {
        check_conditional_extra_allocation_transaction(false, false, false);
    };

    "partially_proved_rollback_guard_keeps_false_edge_feasible"_test = [] {
        check_conditional_extra_allocation_transaction(true, false, false);
    };

    "moved_preheader_insertion_is_observed_by_later_payload_proof"_test = [] {
        check_moved_preheader_insertion_snapshot();
    };

    "guarded_select_accepts_initialized_static_fallback"_test = [] {
        check_guarded_select_fallback(true);
    };

    "guarded_select_rejects_uninitialized_static_fallback"_test = [] {
        check_guarded_select_fallback(false);
    };

    "dominating_masked_flag_zero_and_post_publish_set_prove_nonempty_prefix"_test = [] {
        check_masked_flag_implies_nonempty_prefix(true, true);
    };

    "post_target_masked_flag_overwrite_before_publish_kills_zero_fact"_test = [] {
        check_masked_flag_implies_nonempty_prefix(false, false);
    };

    "rollback_preserves_defined_physical_default_record"_test = [] {
        check_rolled_back_record_remains_physically_defined(true, true);
    };

    "rollback_cannot_manufacture_missing_physical_payload"_test = [] {
        check_rolled_back_record_remains_physically_defined(false, false);
    };
}

}// namespace

int main() {
    register_discriminated_prefix_tests();
    return 0;
}
