#pragma once
// bge-m3 embedding engine -- continuous-batching scheduler
// (embedding_model_plan.md §3.5, milestone M6).
//
// Structure (all primitives per §3.5, verified in include/luisa/runtime/):
//   - Worker pool of P workers (default 2). Each worker owns: one compute
//     Stream, one Event (done fence), one Executor (its own grow-only
//     activation pool), busy accounting. NOTHING is shared between workers
//     except the read-only WeightStore.
//   - Weights: uploaded once by the Scheduler on a dedicated init stream;
//     the init stream signals the weights_ready TimelineEvent fence 1, and
//     every worker stream issues weights_ready.wait(1) as its first command.
//     After that the common case is event-free (weights immutable).
//   - Weight-conversion lifetime (phase 1b; v1 fp32 package +
//     LUISA_EMBED_WEIGHTS=bf16/fp16): packing the four per-layer GEMM matrices
//     on the device needs a transient fp32 copy of each (~1.1 GiB for bge-m3).
//     Those copies are ONE-SHOT. The Scheduler takes ownership of them from the
//     WeightStore at construction (WeightStore::take_conversion_sources, called
//     after record_upload() and before the command list is committed), keeps
//     them alive while the pack dispatches run -- concurrently with the
//     Executors' shader compilation, which is what makes the conversion overlap
//     instead of serialising start-up -- and then, once the Executors are up,
//     synchronizes the init stream ONCE and destroys them. They are
//     deliberately neither retained by the WeightStore (that would hold the
//     fp32 staging for the whole session) nor freed from a backend callback (the
//     copies have to outlive the list, and a callback is not an ordering
//     guarantee for the init stream). A session that converts nothing does not
//     synchronize here at all.
// - Dispatch: one dispatcher fiber/task. Acquire = non-blocking
//     Event::is_completed() poll; queued requests are coalesced (FIFO per
//     bucket, same padded sequence bucket + same completion kind) into one
//     batch job up to max_batch_size; a group whose head is older than
//     max_queue_delay_ms launches even sub-maximal so short sentences never
//     starve behind a filling long bucket. The queue is bounded
//     (max_queue_size); overflow is a fatal LUISA_ERROR (backpressure is the
//     caller's problem above the API).
//   - Completion: async jobs carry a CommandList::add_callback (lc_optimize
//     §8.8: fires on the backend completion thread after ALL GPU commands in
//     the list finished). Callbacks capture buffers BY VALUE and never touch
//     any stream. Sync jobs wait on the worker Event fence with
//     Event::synchronize(fence) -- a CPU-side wait, never a queue-flush
//     stream::synchronize() -- then assemble results from host staging. The
//     async path calls NO synchronize() of any kind.
//   - encode_batch (§3.4 static path) submits as ONE atomic batch job.
//
// Deviation notes (plan §3.1): EncodeOptions/Embedding are public Session
// types (defined here because the scheduler assembles Embeddings; session.h
// includes this header). std::pair is not an STL container so it is allowed
// inside luisa::vector (the ban covers containers). The plan sketch has no
// caller-supplied Stream -- the session/scheduler own all streams per §3.5.

#include <atomic>
#include <cstdint>
#include <mutex>
#include <utility>

#include <luisa/core/fiber.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/stream.h>

#include "executor.h"

namespace luisa::embed {

// Public output/options types (plan §3.1).
struct EncodeOptions {
    uint32_t max_seq_len = 512u;// tokenizer truncation length
    bool want_sparse = false;
    bool want_colbert = false;
};

struct Embedding {
    luisa::vector<float> dense;                      // [H], L2-normalized CLS
    luisa::vector<std::pair<uint32_t, float>> sparse;// (token_id, weight>0), specials dropped
    luisa::vector<float> colbert;                    // [(n-2) * H], specials dropped, per-row L2-normalized
    bool truncated = false;                          // input exceeded max_seq_len and was truncated
};

struct SchedulerConfig {
    uint32_t num_workers = 2u;       // P (plan §3.5 default)
    uint32_t max_batch_size = 8u;    // coalescing cap per launched job
    uint32_t max_queue_delay_ms = 2u;// staleness that forces a sub-maximal batch
    uint32_t max_queue_size = 512u;  // bounded FIFO (in-flight requests beyond workers)
};

// One tokenized encode request. The scheduler owns the padded upload staging
// and the readback staging inside its Job; finalize() moves the results into
// an Embedding (sparse compaction + colbert special-token drop, plan §4 #10).
struct ScheduledRequest {
    luisa::vector<uint32_t> ids;// tokenized, unpadded (2 <= size <= 8192)
    bool want_sparse{};
    bool want_colbert{};
    bool ids_truncated{};// the tokenizer truncated ids to max_seq_len
    luisa::function<void(Embedding)> callback;// set => async; null => sync caller

    // internal (scheduler-owned) ------------------------------------------------
    uint32_t seq{};// padded seq of this request's row block inside its job
    uint32_t row{};// batch row index inside its job
    void *opaque_job{};
    uint64_t fence{};
    uint32_t worker_index{0xFFFFFFFFu};
    bool sync_ready{};// fence/worker/job visible to the blocking caller
    double submitted_ms{};
};

// The caller must have a bound luisa::fiber::scheduler before constructing
// this object and keep it alive until after the Scheduler is destroyed.
class Scheduler {
public:
    // `weights` is non-const because the construction performs the one-shot
    // handoff of the phase-1b fp32 conversion staging (see the header comment);
    // the workers only ever read the store afterwards.
    Scheduler(compute::Device &device, WeightStore &weights, const SchedulerConfig &cfg) noexcept;
    ~Scheduler() noexcept;// stops the dispatcher and drains in-flight jobs

    Scheduler(const Scheduler &) = delete;
    Scheduler(Scheduler &&) = delete;
    Scheduler &operator=(const Scheduler &) = delete;
    Scheduler &operator=(Scheduler &&) = delete;

    // Async submit: takes ownership; the request's callback runs on the
    // backend completion thread exactly once. Never blocks, never
    // synchronizes. Requires callback != null.
    void submit(luisa::unique_ptr<ScheduledRequest> req) noexcept;

    // Static batch (§3.4): one atomic job, padded to the common bucket, async.
    void submit_batch(luisa::vector<luisa::unique_ptr<ScheduledRequest>> reqs) noexcept;

    // Sync path (plan §3.5 "sync"): enqueues `req` (caller keeps the
    // unique_ptr alive until this call returns), blocks on the WORKER EVENT
    // FENCE (CPU side, no queue flush), then returns the finished embedding.
    Embedding encode_sync(luisa::unique_ptr<ScheduledRequest> &req) noexcept;
    // All `reqs` go up as one atomic batch; waits the same way; caller keeps
    // ownership until the call returns.
    luisa::vector<Embedding> encode_batch_sync(luisa::vector<luisa::unique_ptr<ScheduledRequest>> &reqs) noexcept;

    // Trace hook passthrough (sync-path semantics; see executor.h).
    void set_layer_trace(LayerTraceCallback cb) noexcept;

    [[nodiscard]] uint32_t worker_count() const noexcept { return static_cast<uint32_t>(_workers.size()); }
    // Per-request submit->completion latency samples in ms (drains the buffer).
    [[nodiscard]] luisa::vector<double> take_latency_samples_ms();

private:
    struct Job;
    struct PendingGroup;

    void _dispatch_loop() noexcept;
    bool _try_dispatch() noexcept;// true if a job was launched
    Job *_launch(luisa::vector<luisa::unique_ptr<ScheduledRequest>> owned,
                 luisa::vector<ScheduledRequest *> reqs, bool atomic_batch, uint32_t slot,
                 luisa::fiber::lock &lk) noexcept;
    Embedding _finalize(ScheduledRequest *req, Job &job) noexcept;
    uint32_t _pick_worker_slot() noexcept;// requires mutex; non-blocking poll
    void _wait_assigned(ScheduledRequest *req) noexcept;
    void _record_sample(double ms) noexcept;
    static Embedding _empty_embedding() noexcept { return {}; }

    compute::Device &_device;
    const WeightStore &_weights;
    SchedulerConfig _cfg;

    struct Worker {
        compute::Stream stream;
        compute::Event done_ev;
        luisa::unique_ptr<Executor> executor;
        uint64_t busy_fence{};
        bool busy{};
    };
    luisa::vector<Worker> _workers;

    compute::Stream _init_stream;
    compute::TimelineEvent _weights_ready;

    luisa::fiber::mutex _mutex;
    luisa::fiber::condition_variable _cv;
    luisa::vector<PendingGroup *> _queue;// FIFO of pending groups (heap-owned)
    uint32_t _queue_size{};              // in-flight request count
    std::atomic<bool> _stop{false};
    luisa::fiber::event _dispatcher;// signalled after _dispatch_loop() returns
    uint32_t _outstanding_jobs{};// launched, not yet completed (under _mutex)

    std::mutex _stats_mutex;
    luisa::vector<double> _latency_ms;
    double _now_ms() const noexcept;
};

}// namespace luisa::embed
