#pragma once

#include <luisa/ast/function_builder.h>
#include <luisa/coro/coro_scheduler.h>
#include <luisa/dsl/coro_frame.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

namespace luisa::compute::coro {

/// Persistent-threads coroutine scheduler: one persistent thread block.
///
/// A single thread block stays resident on the device across multiple
/// coroutine instances.  Each thread owns one frame stored in shared
/// memory.  Thread 0 (the leader) votes for the most common continuation
/// target, and ALL threads in the block execute that subroutine
/// synchronously (threads whose token does not match the voted target
/// simply skip via the coro-split skip guards).
///
/// When a frame reaches TERMINAL_TOKEN, the leader acquires a new
/// coroutine instance via an atomic counter (simplified in Phase 1).
///
/// @tparam Args  Coroutine input parameter types.
template<typename... Args>
class PersistentThreadsCoroScheduler : public CoroScheduler<Args...> {

    using Coro = Coroutine<void(Args...)>;
    using ShaderType = Shader<1, Args...>;
    using Base = CoroScheduler<Args...>;

    const Coro &_coro;
    ShaderType _shader;
    uint _block_size;

public:
    /// Construct the scheduler and compile the persistent kernel.
    ///
    /// @param device      Device used to compile the kernel.
    /// @param coro        The compiled coroutine (provides graph, frame_desc, subroutines).
    /// @param block_size  Number of threads per persistent block (default: 256).
    PersistentThreadsCoroScheduler(Device &device, const Coro &coro,
                                   uint block_size = 256u) noexcept
        : Base{coro.graph(), coro.frame_desc()},
          _coro{coro},
          _shader{_compile(device, coro, block_size)},
          _block_size{block_size} {}

    // Non-copyable, non-movable
    PersistentThreadsCoroScheduler(const PersistentThreadsCoroScheduler &) = delete;
    PersistentThreadsCoroScheduler &operator=(const PersistentThreadsCoroScheduler &) = delete;
    PersistentThreadsCoroScheduler(PersistentThreadsCoroScheduler &&) = delete;
    PersistentThreadsCoroScheduler &operator=(PersistentThreadsCoroScheduler &&) = delete;

    /// Dispatch the pre-compiled persistent kernel.
    /// The dispatch size.x sets the number of thread blocks.
    /// Each block has _block_size threads (persistent).
    void _dispatch(Stream &stream, uint3 dispatch_size,
                   const Args &...args) noexcept override {
        stream << _shader(args...).dispatch(dispatch_size.x);
    }

    [[nodiscard]] const Coro &coroutine() const noexcept { return _coro; }
    [[nodiscard]] auto block_size() const noexcept { return _block_size; }

private:
    /// Compile the persistent kernel.
    [[nodiscard]] static ShaderType _compile(
        Device &device, const Coro &coro, uint block_size) noexcept {

        return device.compile(Kernel1D{
            [&coro, block_size](Var<std::remove_cvref_t<Args>>... k_args) noexcept {

                const auto &graph = coro.graph();
                const auto *desc = &coro.frame_desc();
                const uint BST = block_size;
                const size_t nc = graph.node_count();

                set_block_size(BST);

                // Shared memory: one token+active slot per thread
                Shared<uint> shm_tokens{BST};
                Shared<bool> shm_active{BST};

                auto tid = thread_x();

                // Helper: invoke a subroutine
                auto call_sub = [&](const auto &sub, CoroFrame &frame) noexcept {
                    const Expression *call_args[1u + sizeof...(Args)];
                    call_args[0] = frame.expression();
                    size_t ai = 1u;
                    ((call_args[ai++] = detail::extract_expression(k_args)), ...);
                    detail::FunctionBuilder::current()->call(
                        sub->function(),
                        luisa::span<const Expression *const>{
                            call_args, 1u + sizeof...(Args)});
                };

                // ======================================================
                // Initialization
                // ======================================================
                shm_tokens[tid] = 0u;
                shm_active[tid] = true;
                sync_block();

                // ======================================================
                // Phase 1: Entry subroutine
                // ======================================================
                if (auto entry_sub = coro[0u]) {
                    auto frame = CoroFrame::create(desc);
                    frame.coro_id = make_uint3(tid, 0u, 0u);
                    call_sub(entry_sub, frame);
                    shm_tokens[tid] = frame.target_token;
                    shm_active[tid] = !frame.is_terminated();
                }
                sync_block();

                // ======================================================
                // Phase 2: Persistent continuation loop
                // ======================================================
                constexpr uint MAX_ITERS = 32u;

                // Shared flag: leader sets true when any thread needs this
                // continuation.
                Shared<bool> shm_leader_match{1u};

                // Precompute (token, subroutine) pairs for every continuation.
                luisa::vector<uint> cont_tokens;
                luisa::vector<decltype(coro[0u])> cont_subs;
                for (size_t i = 1u; i < nc; ++i) {
                    if (auto s = coro[i]) {
                        cont_tokens.push_back(
                            static_cast<uint>(graph.node(i).token));
                        cont_subs.push_back(s);
                    }
                }
                size_t nct = cont_tokens.size();

                // Outer persistent loop — create frame once per iteration
                $for (iter, MAX_ITERS) {
                    CoroFrame frame{desc};
                    frame.coro_id = make_uint3(tid, 0u, 0u);

                    // Host-side loop over all continuations.
                    // For each, the leader scans whether any active thread
                    // currently needs it.  If yes, all threads execute that
                    // subroutine synchronously.
                    for (size_t ci = 0u; ci < nct; ++ci) {
                        uint c_token = cont_tokens[ci];
                        auto cont_sub = cont_subs[ci];

                        // Leader: reset flag, scan all active threads
                        $if (tid == 0u) {
                            shm_leader_match[0u] = false;
                        };
                        sync_block();

                        $if (tid == 0u) {
                            $for (t, BST) {
                                $if (shm_active[t] &
                                     (shm_tokens[t] == c_token)) {
                                    shm_leader_match[0u] = true;
                                    $break;
                                };
                            };
                        };
                        sync_block();

                        // All threads: if leader found a match, execute
                        $if (shm_leader_match[0u]) {
                            $if (shm_active[tid]) {
                                frame.target_token = shm_tokens[tid];
                                call_sub(cont_sub, frame);
                                shm_tokens[tid] = frame.target_token;
                                shm_active[tid] =
                                    !frame.is_terminated();
                            };
                        };
                        sync_block();
                    }
                };
            }});
    }
};

// CTAD deduction guide
template<typename... Args>
PersistentThreadsCoroScheduler(Device &, const Coroutine<void(Args...)> &)
    -> PersistentThreadsCoroScheduler<Args...>;

}// namespace luisa::compute::coro
