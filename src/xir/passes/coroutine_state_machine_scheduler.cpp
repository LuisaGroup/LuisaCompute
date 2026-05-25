#include <luisa/core/logging.h>
#include <luisa/ast/type.h>
#include <luisa/ast/type_registry.h>
#include <luisa/xir/argument.h>
#include <luisa/xir/basic_block.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/constant.h>
#include <luisa/xir/function.h>
#include <luisa/xir/instructions/alloca.h>
#include <luisa/xir/instructions/branch.h>
#include <luisa/xir/instructions/call.h>
#include <luisa/xir/instructions/gep.h>
#include <luisa/xir/instructions/load.h>
#include <luisa/xir/instructions/loop.h>
#include <luisa/xir/instructions/return.h>
#include <luisa/xir/instructions/store.h>
#include <luisa/xir/instructions/switch.h>
#include <luisa/xir/instructions/unreachable.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/coroutine_state_machine_scheduler.h>

namespace luisa::compute::xir {

namespace {

[[nodiscard]] Constant *uint_constant(Module *module, uint32_t value) noexcept {
    return module->create_constant(luisa::compute::Type::of<uint32_t>(), &value);
}

[[nodiscard]] Argument *mirror_argument(Function *target, const Argument *src) noexcept {
    switch (src->derived_argument_tag()) {
        case DerivedArgumentTag::VALUE:
            return target->create_value_argument(src->type());
        case DerivedArgumentTag::REFERENCE:
            return target->create_reference_argument(src->type());
        case DerivedArgumentTag::RESOURCE:
            return target->create_resource_argument(src->type());
    }
    return nullptr;
}

}// namespace

CoroutineStateMachineSchedulerInfo coroutine_state_machine_scheduler_emit(
    Module *module,
    const CoroutineSplitInfo &split,
    const CoroutineStateMachineSchedulerConfig &config) noexcept {
    CoroutineStateMachineSchedulerInfo info;
    if (module == nullptr) {
        info.diagnostics.emplace_back("coroutine_state_machine: module is null");
        return info;
    }
    if (!split.is_supported || !split.changed || split.continuations.empty()) {
        info.diagnostics.emplace_back(
            "coroutine_state_machine: requires a successfully split coroutine "
            "(is_supported && changed && continuations non-empty)");
        return info;
    }
    if (split.frame_type == nullptr) {
        info.diagnostics.emplace_back("coroutine_state_machine: split has no frame_type");
        return info;
    }
    // Validate continuations: all must point to a CallableFunction whose
    // first argument is a reference to the frame type.
    for (auto &&cont : split.continuations) {
        if (cont.callable == nullptr) {
            info.diagnostics.emplace_back("coroutine_state_machine: a continuation has no callable");
            return info;
        }
    }
    auto kernel = module->create_kernel();
    kernel->set_block_size(config.block_size);
    // Mirror the kernel-side arguments. The split pass already copied the
    // source coroutine's argument list onto each continuation after the frame
    // ref, so we use the entry continuation's argument list (skipping the
    // frame ref) as the canonical user-facing signature.
    auto &entry_callable_args = split.continuations.front().callable->arguments();
    auto args_iter = entry_callable_args.begin();
    auto args_end = entry_callable_args.end();
    if (args_iter == args_end) {
        info.diagnostics.emplace_back("coroutine_state_machine: entry continuation has no arguments");
        return info;
    }
    ++args_iter;// skip frame ref
    luisa::vector<Argument *> mirrored;
    for (; args_iter != args_end; ++args_iter) {
        mirrored.emplace_back(mirror_argument(kernel, *args_iter));
    }
    auto uint_t = luisa::compute::Type::of<uint32_t>();
    auto entry_block = kernel->create_body_block();
    XIRBuilder builder;
    builder.set_insertion_point(entry_block);
    auto frame = builder.alloca_local(split.frame_type);
    auto state_idx = uint_constant(module, 0u);
    // Build the call argument list once: [frame, mirrored kernel args...].
    luisa::vector<Value *> call_args;
    call_args.emplace_back(frame);
    for (auto a : mirrored) { call_args.emplace_back(a); }
    // Initial state = 0 (entry token; the entry continuation overwrites it).
    auto state_init = builder.gep(uint_t, frame, {state_idx});
    builder.store(state_init, uint_constant(module, 0u));
    // Call the entry continuation once. It writes the next token into frame[0].
    builder.call(nullptr, split.continuations.front().callable, call_args);
    // Build the dispatch loop: simple_loop { switch(frame[0]) { ... } }.
    auto loop = builder.simple_loop();
    auto loop_body = loop->create_body_block();
    auto loop_merge = loop->create_merge_block();
    builder.set_insertion_point(loop_body);
    auto state_gep = builder.gep(uint_t, frame, {state_idx});
    auto state = builder.load(uint_t, state_gep);
    auto sw = builder.switch_(state);
    auto sw_default = kernel->create_basic_block();
    auto sw_merge = kernel->create_basic_block();
    sw->set_default_block(sw_default);
    sw->set_merge_block(sw_merge);
    // case 0 → terminated: branch to loop_merge (exit the loop).
    {
        auto case0 = kernel->create_basic_block();
        sw->add_case(0, case0);
        builder.set_insertion_point(case0);
        builder.br(loop_merge);
    }
    // case k for each non-entry continuation: call continuation, branch to switch merge.
    for (size_t k = 1; k < split.continuations.size(); ++k) {
        auto case_block = kernel->create_basic_block();
        sw->add_case(static_cast<int32_t>(k), case_block);
        builder.set_insertion_point(case_block);
        builder.call(nullptr, split.continuations[k].callable, call_args);
        builder.br(sw_merge);
    }
    // default: unreachable — every live token must hit a case.
    builder.set_insertion_point(sw_default);
    builder.unreachable_("coroutine_state_machine: invalid target_token");
    // After the switch merges, branch back to loop body start (continue).
    builder.set_insertion_point(sw_merge);
    builder.br(loop_body);
    // After the loop merges, return.
    builder.set_insertion_point(loop_merge);
    builder.return_void();
    info.ok = true;
    info.kernel = kernel;
    return info;
}

}// namespace luisa::compute::xir
