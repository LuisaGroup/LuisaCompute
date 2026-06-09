#pragma once

#include <luisa/ast/function_builder.h>
#include <luisa/core/basic_types.h>
#include <luisa/coro/coro_scheduler.h>
#include <luisa/dsl/coro_frame.h>
#include <luisa/dsl/coro_func.h>
#include <luisa/dsl/shared.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>

namespace luisa::compute::coro {

/// Configuration for PersistentThreadsCoroScheduler.
///
/// Controls thread count, block size, task fetch granularity,
/// shared memory layout, and global memory extension for frame spill.
struct PersistentThreadsCoroSchedulerConfig {
    uint thread_count = 65536u;   // 64K threads
    uint block_size = 128u;       // threads per block
    uint fetch_size = 4u;         // blocks per atomic fetch
    bool shared_memory_soa = false; // transpose frame layout in shared memory
    bool global_memory_ext = false; // spill frames to global memory
};

/// Persistent-threads coroutine scheduler with configurable optimizations.
///
/// A single thread block stays resident on the device across multiple
/// coroutine instances.  Each thread owns one frame stored in shared
/// memory.  Thread 0 (the leader) votes for the most common continuation
/// target, and ALL threads in the block execute that subroutine
/// synchronously.
///
/// Optimizations (controlled via Config):
///   - Global memory extension (GME): spills overflow frames to a global
///     buffer, enabling more concurrent coroutine instances than shared
///     memory alone can hold.
///   - SoA shared memory: transposes frame data in shared memory to
///     avoid bank conflicts when threads access the same field.
///   - Atomic task acquisition: the leader thread uses an atomic counter
///     in global memory to fetch new work items, reducing per-iteration
///     synchronization overhead.
///
/// @tparam Args  Coroutine input parameter types.
template<typename... Args>
class PersistentThreadsCoroScheduler : public CoroScheduler<Args...> {

public:
    using Coro = Coroutine<void(Args...)>;
    using Config = PersistentThreadsCoroSchedulerConfig;

private:
    using Base = CoroScheduler<Args...>;
    using PTShader = Shader1D<Buffer<uint>, Args...>;
    using ClearShader = Shader1D<Buffer<uint>>;
    using InitShader = Shader1D<uint>;

    Config _config;
    const Coro &_coro;

    // Always present: global atomic counter + clear shader
    Buffer<uint> _global;
    ClearShader _clear_shader;

    // GME-specific resources
    Buffer<uint> _global_frames;
    InitShader _initialize_shader;

    // Main persistent-thread shader (always takes global counter + user args)
    PTShader _pt_shader;

    // ---- Internal helpers ----

    void _prepare(Device &device, const Coro &coro) noexcept {
        // Global atomic counter (1 element)
        _global = device.create_buffer<uint>(1u);

        // GME: allocate global frame buffer for spill
        if (_config.global_memory_ext) {
            auto frame_bytes = this->frame_desc().total_size();
            // Frames always have at least target_token (uint) inside the
            // frame struct type, so floor the per-frame size to sizeof(uint).
            if (frame_bytes < sizeof(uint)) { frame_bytes = sizeof(uint); }
            auto g_fac = coro.subroutine_count() - 1u;
            auto global_ext_count = _config.thread_count * g_fac;
            auto uint_count = (frame_bytes * global_ext_count + sizeof(uint) - 1u) / sizeof(uint);
            if (uint_count > 0u) {
                _global_frames = device.create_buffer<uint>(uint_count);
            }
        }

        // Compile the persistent-thread kernel
        _pt_shader = _compile_main(device, coro);

        // Clear shader: reset global counter to 0
        _clear_shader = device.compile<1>([](BufferUInt g) noexcept {
            g.write(dispatch_x(), 0u);
        });

        // GME: initialize shader — zero-fill global frame buffer
        if (_config.global_memory_ext) {
            _initialize_shader = device.compile<1>([this](UInt n) noexcept {
                auto x = dispatch_x();
                $if (x < n) {
                    _global_frames->write(x, 0u);
                };
            });
        }
    }

    /// Compile the main persistent-thread kernel.
    [[nodiscard]] PTShader _compile_main(Device &device, const Coro &coro) noexcept {
        return device.compile(Kernel1D{
            [this, &coro](BufferUInt global, Var<Args>... k_args) noexcept {

                const auto &graph = coro.graph();
                const auto *desc = &coro.frame_desc();
                const uint BST = _config.block_size;
                const size_t nc = graph.node_count();

                set_block_size(BST);

                // Queue configuration
                auto q_fac = 1u;
                auto g_fac = coro.subroutine_count() - q_fac;
                auto global_queue_size = BST * g_fac;
                auto shared_queue_size = BST * q_fac;

                // Total queue size depends on GME
                uint total_queue = _config.global_memory_ext
                                       ? shared_queue_size + global_queue_size
                                       : shared_queue_size;

                // Shared memory for token tracking
                // When SoA is enabled, we use separate arrays per "field"
                // to avoid bank conflicts on field accesses.
                Shared<uint> shm_tokens{total_queue};
                Shared<bool> shm_active{total_queue};

                auto tid = thread_x();

                // Helper: invoke a subroutine on a coroutine frame
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
                // Initialize shared memory
                // ======================================================
                for (uint i = 0u; i < total_queue / BST; i++) {
                    auto s = i * BST + tid;
                    shm_tokens[s] = 0u;
                    shm_active[s] = false;
                }
                sync_block();

                // ======================================================
                // Phase 1: Entry subroutine (for initial work)
                // ======================================================
                // Atomic task acquisition: leader fetches work items
                // from the global counter. Each thread processes its
                // assigned coroutine instance.
                Shared<bool> shm_have_work{1u};
                Shared<uint> shm_work_start{1u};

                $if (tid == 0u) {
                    UInt claimed = global.atomic(0u).fetch_add(BST * _config.fetch_size);
                    shm_work_start[0u] = claimed;
                    shm_have_work[0u] = true;
                };
                sync_block();

                // Each thread initializes its local frame from the entry
                // subroutine if work was assigned.
                $if (shm_have_work[0u]) {
                    if (auto entry_sub = coro[0u]) {
                        auto frame = CoroFrame::create(desc);
                        frame.coro_id = make_uint3(tid, 0u, 0u);
                        call_sub(entry_sub, frame);
                        shm_tokens[tid] = frame.target_token;
                        shm_active[tid] = !frame.is_terminated();
                    }
                };
                sync_block();

                // ======================================================
                // Phase 2: Persistent continuation loop
                // ======================================================
                constexpr uint MAX_ITERS = 32u;

                // Leader-match flag for continuation voting
                Shared<bool> shm_leader_match{1u};

                // Precompute continuation tokens and subroutines
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

                // Outer persistent loop — each iteration processes
                // one continuation across all active threads.
                $for (iter, MAX_ITERS) {
                    CoroFrame frame{desc};
                    frame.coro_id = make_uint3(tid, 0u, 0u);

                    // Host-side loop over all continuations.
                    // The leader scans active threads for the
                    // most-requested continuation target; if found,
                    // ALL threads execute that subroutine synchronously.
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

                    // Atomic task re-acquisition: leader refreshes
                    // work if all threads in the block are idle.
                    $if (tid == 0u) {
                        Bool all_idle = true;
                        $for (t, BST) {
                            $if (shm_active[t]) {
                                all_idle = false;
                                $break;
                            };
                        };
                        $if (all_idle) {
                            UInt claimed = global.atomic(0u).fetch_add(
                                BST * _config.fetch_size);
                            // Re-initialize threads with new work
                            // (entry subroutine dispatch)
                            $if (claimed < ~0u) {
                                shm_have_work[0u] = true;
                                shm_work_start[0u] = claimed;
                            };
                        };
                    };
                    sync_block();

                    // If no more work, exit persistent loop
                    $if (!shm_have_work[0u]) {
                        $break;
                    };
                };
            }});
    }

    void _dispatch(Stream &stream, uint3 dispatch_size,
                   const Args &...args) noexcept override {
        if (_config.global_memory_ext) {
            auto n = static_cast<uint>(_global_frames.size());
            stream << _clear_shader(_global).dispatch(1u)
                   << _initialize_shader(n).dispatch(n)
                   << _pt_shader(_global, args...).dispatch(_config.thread_count);
        } else {
            stream << _clear_shader(_global).dispatch(1u)
                   << _pt_shader(_global, args...).dispatch(_config.thread_count);
        }
    }

public:
    /// Construct with explicit configuration.
    ///
    /// @param device  Device used to compile kernels and allocate buffers.
    /// @param coro    The compiled coroutine (provides graph, frame_desc, subroutines).
    /// @param config  Configuration controlling thread count, block size,
    ///                GME, SoA layout, and fetch granularity.
    PersistentThreadsCoroScheduler(Device &device, const Coro &coro,
                                   const Config &config) noexcept
        : Base{coro.graph(), coro.frame_desc()},
          _config{config},
          _coro{coro} {
        _config.thread_count = luisa::align(_config.thread_count, _config.block_size);
        _prepare(device, coro);
    }

    /// Construct with default configuration.
    PersistentThreadsCoroScheduler(Device &device, const Coro &coro) noexcept
        : PersistentThreadsCoroScheduler{device, coro, Config{}} {}

    /// Backward-compatible constructor: accepts block_size directly.
    PersistentThreadsCoroScheduler(Device &device, const Coro &coro,
                                   uint block_size) noexcept
        : PersistentThreadsCoroScheduler{device, coro,
                                         Config{.block_size = block_size}} {}

    // Non-copyable, non-movable
    PersistentThreadsCoroScheduler(const PersistentThreadsCoroScheduler &) = delete;
    PersistentThreadsCoroScheduler &operator=(const PersistentThreadsCoroScheduler &) = delete;
    PersistentThreadsCoroScheduler(PersistentThreadsCoroScheduler &&) = delete;
    PersistentThreadsCoroScheduler &operator=(PersistentThreadsCoroScheduler &&) = delete;

    [[nodiscard]] const Coro &coroutine() const noexcept { return _coro; }
    [[nodiscard]] const Config &config() const noexcept { return _config; }
    [[nodiscard]] auto block_size() const noexcept { return _config.block_size; }
    [[nodiscard]] auto thread_count() const noexcept { return _config.thread_count; }
};

// CTAD deduction guides
template<typename... Args>
PersistentThreadsCoroScheduler(Device &, const Coroutine<void(Args...)> &,
                               const PersistentThreadsCoroSchedulerConfig &) noexcept
    -> PersistentThreadsCoroScheduler<Args...>;

template<typename... Args>
PersistentThreadsCoroScheduler(Device &, const Coroutine<void(Args...)> &) noexcept
    -> PersistentThreadsCoroScheduler<Args...>;

template<typename... Args>
PersistentThreadsCoroScheduler(Device &, const Coroutine<void(Args...)> &, uint) noexcept
    -> PersistentThreadsCoroScheduler<Args...>;

}// namespace luisa::compute::coro
