//
// Created by Mike on 2024/5/10.
//

#include <luisa/dsl/sugar.h>
#include <luisa/coro/schedulers/state_machine.h>

namespace luisa::compute::coroutine::detail {

void coro_scheduler_state_machine_impl(CoroFrame &frame, uint state_count,
                                       luisa::move_only_function<void(CoroToken)> node) noexcept {
    node(coro_token_entry);
    $loop {
        $switch (frame.target_token) {
            for (auto i = 1u; i < state_count; i++) {
                $case (i) { node(i); };
            }
            $default { $return(); };
        };
    };
}

void coro_scheduler_state_machine_smem_impl(Shared<CoroFrame> &smem, const CoroGraph *graph,
                                            luisa::move_only_function<void(CoroToken, CoroFrame &frame)> node) noexcept {
    auto target_token = def(0u);
    auto frame = CoroFrame::create(graph->shared_frame(), dispatch_id());
    node(coro_token_entry, frame);
    target_token = frame.target_token;
    auto tid = thread_z() * block_size().x * block_size().y + thread_y() * block_size().x + thread_x();
    smem.write(tid, frame, graph->entry().output_fields());
    $loop {
        $switch (target_token) {
            for (auto i = 1u; i < graph->nodes().size(); i++) {
                $case (i) {
                    auto f = smem.read(tid, graph->node(i).input_fields());
                    f.coro_id = dispatch_id();
                    node(i, f);
                    target_token = f.target_token;
                    smem.write(tid, f, graph->node(i).output_fields());
                };
            }
            $default { $return(); };
        };
    };
}

}// namespace luisa::compute::coroutine::detail
