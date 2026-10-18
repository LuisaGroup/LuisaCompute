// bge-m3 embedding engine -- continuous-batching scheduler implementation
// (plan §3.5). Design contract: see scheduler.h.
#include "scheduler.h"

// D3 routing (plan §9.1): device log lines on the worker streams are routed
// to the host log via the debug harness. set_log_callback REPLACES the
// callback, so route_stream_logs must be the sole installer per stream
// (debug_utils.h contract). Parent-side integration one-liner (B4 to-do #1).
#include "embedding/embed/debug_utils.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>

#include <luisa/core/logging.h>
#include <luisa/core/stl/string.h>
#include <luisa/runtime/command_list.h>

namespace luisa::embed {

using compute::CommandList;
using compute::Device;

double Scheduler::_now_ms() const noexcept {
    return static_cast<double>(
               std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
                   .count()) /
           1000.0;
}

struct Scheduler::Job {
    luisa::vector<luisa::unique_ptr<ScheduledRequest>> owned;// non-empty for async jobs
    luisa::vector<ScheduledRequest *> reqs;
    uint32_t B{}, S{};
    luisa::vector<uint32_t> ids; // [B*S] padded upload staging (pad id 1)
    luisa::vector<float> mask;   // [B*S] 1.0 real / 0.0 pad
    luisa::vector<float> dense;  // [B*H] readback staging
    luisa::vector<float> sparse; // [B*S] readback staging when any want_sparse
    luisa::vector<float> colbert;// [B*S*H] readback staging when any want_colbert
    uint32_t worker{0xFFFFFFFFu};
    uint64_t fence{};
    uint32_t pending_sync_callers{};// sync waiters still to finish (async jobs: 0)
};

// Hard envelope for ONE encode job's padded rows (B*S), parent-fixed after
// the test-plan wave (T3/T5/T8 evidence): every per-row kernel launches 1D
// grids whose GROUP count scales with rows, and the DX (and, silently, vk)
// grid cap is 65535 groups per dimension. Worst kernels at rows R, H=1024:
//   residual_add groups = R*H/256 = 4R      (kernels_gemm.cpp, block 256)
//   embed / layernorm / heads groups = R
//   softmax groups = B*heads*S = 16R        -> subview-chunked in executor.cpp
//   gemm tiles_m = ceil(R/16); attention scores y = B*heads*ceil(S/16)
// R <= 16383 keeps the worst UNCHUNKED grid at 65532 <= 65535 with margin.
// The scheduler therefore never launches a job with more than kMaxJobRows
// padded rows: atomic encode_batch calls are split (row-local math makes
// per-sentence results chunk-invariant) and coalescing is rows-aware.
inline constexpr uint32_t kMaxJobRows = 16383u;

// Greedy order-preserving split of `reqs` into atomic-chunks whose padded
// row count (chunk size x bucket of the chunk's longest member) fits
// kMaxJobRows. Single requests always fit (8192 tokens < kMaxJobRows).
luisa::vector<luisa::vector<ScheduledRequest *>> split_atomic_chunks(
    const luisa::vector<ScheduledRequest *> &reqs) noexcept {
    luisa::vector<luisa::vector<ScheduledRequest *>> chunks;
    luisa::vector<ScheduledRequest *> cur;
    uint32_t cur_max_len = 0u;
    for (auto *r : reqs) {
        const auto n = static_cast<uint32_t>(r->ids.size());
        const auto m = std::max(cur_max_len, n);
        if (!cur.empty() &&
            (cur.size() + 1u) * embed_bucket_for(m) > kMaxJobRows) {
            chunks.emplace_back(std::move(cur));
            cur = {};
            cur_max_len = n;
        } else {
            cur_max_len = m;
        }
        cur.emplace_back(r);
    }
    if (!cur.empty()) { chunks.emplace_back(std::move(cur)); }
    return chunks;
}

struct Scheduler::PendingGroup {
    luisa::vector<luisa::unique_ptr<ScheduledRequest>> owned;// async ownership transfer
    luisa::vector<ScheduledRequest *> reqs;
    double first_ms{};
    bool atomic_batch{};
    bool is_async{};
};

namespace {
uint32_t group_max_len(const luisa::vector<ScheduledRequest *> &reqs) noexcept {
    uint32_t n = 0u;
    for (const auto *r : reqs) {
        n = std::max(n, static_cast<uint32_t>(r->ids.size()));
    }
    return n;
}

// B-1 workaround toggle (defect investigation, see embed_opt_log B-1 row).
// The sync path waits the WORKER EVENT FENCE (GPU timeline reached the event),
// which does NOT imply that the backend completion worker has finished the
// host-side readback memcpy (DtoH into the Job's host staging) and recycled the
// readback placed-resource. In debug the completion path is slow/synchronous
// enough that it always finishes before the fence wait returns; in release it
// races, and freeing/reusing the Job's host staging under that in-flight read
// produces either a host use-after-free segfault (dx+vk, batch>=8) or a lost
// device surfacing at the next readback CreatePlacedResource (b32). The barrier
// forces a full queue drain -- provably ordering every completion-Worker side
// effect before the caller reads/deletes the Job.
//
// EMBED_B1_READBACK_BARRIER=0 disables it (to re-measure the pre-fix crash /
// compare cost); default ON. It is a pure host-completion barrier: no numeric
// change. On the sync path we already block on the fence, so an extra
// stream::synchronize() is essentially free (the queue is already drained to the
// fence; this only waits for the completion worker's bookkeeping).
bool b1_readback_barrier_enabled() noexcept {
    static const bool on = [] {
        const auto *raw = std::getenv("EMBED_B1_READBACK_BARRIER");
        return raw == nullptr || luisa::string_view{raw} != luisa::string_view{"0"};
    }();
    return on;
}
}// namespace

Scheduler::Scheduler(Device &device, const WeightStore &weights, const SchedulerConfig &cfg) noexcept
    : _device{device}, _weights{weights}, _cfg{cfg} {

    if (_cfg.num_workers == 0u) {
        _cfg.num_workers = 1u;
    }

    // weights_ready fence: upload on the init stream, signal fence 1; every
    // worker stream waits fence 1 once as its first command (§3.5). After
    // this, weights are read-only and the common case stays event-free.
    _weights_ready = _device.create_timeline_event();
    _init_stream = _device.create_stream(compute::StreamTag::COMPUTE);
    {
        auto cl = CommandList::create();
        _weights.record_upload(cl);
        _init_stream << cl.commit();
    }
    _init_stream << _weights_ready.signal(1u);

    _workers.reserve(_cfg.num_workers);// stable addresses for Executor refs
    for (uint32_t w = 0; w < _cfg.num_workers; w++) {
        _workers.emplace_back();
        auto &worker = _workers.back();// construct IN PLACE: the Executor
        // stores a reference to worker.stream
        worker.stream = _device.create_stream(compute::StreamTag::COMPUTE);
        worker.stream.set_name(luisa::format("embed_worker_{}", w));
        luisa::embed::route_stream_logs(worker.stream);
        worker.done_ev = _device.create_event();
        worker.stream << _weights_ready.wait(1u);
        worker.executor = luisa::make_unique<Executor>(_device, _weights, worker.stream);
    }

    _dispatcher = luisa::fiber::async([this] { _dispatch_loop(); });
}

Scheduler::~Scheduler() noexcept {

    _stop.store(true);
    {
        std::scoped_lock lk{_mutex};
        _cv.notify_all();
    }
    _dispatcher.wait();

    // never tear a worker/pool down while jobs are still in flight: async
    // callbacks and sync finishers capture/derive `this`.
    {
        luisa::fiber::lock lk{_mutex};
        _cv.wait(lk, [this] { return _outstanding_jobs == 0u; });
    }

    {
        std::scoped_lock lk{_mutex};
        for (auto *g : _queue) {
            LUISA_WARNING("embedding scheduler destroyed with {} queued requests; dropping.", g->reqs.size());
            delete g;
        }
        _queue.clear();
        _queue_size = 0u;
    }
    // let the weight upload (and any late event waits) retire before the
    // streams themselves go away.
    _init_stream.synchronize();
    for (auto &w : _workers) {
        w.stream.synchronize();
    }
}

void Scheduler::submit(luisa::unique_ptr<ScheduledRequest> req) noexcept {

    if (req->callback == nullptr) [[unlikely]] {
        LUISA_ERROR("embedding scheduler: submit() requires a completion callback (use encode_sync for the blocking path).");
    }
    if (req->ids.size() < 2u || req->ids.size() > 8192u) [[unlikely]] {
        LUISA_ERROR("embedding scheduler: request of {} tokens outside [2, 8192].", req->ids.size());
    }

    auto *group = new PendingGroup{};
    group->first_ms = _now_ms();
    group->is_async = true;
    req->submitted_ms = group->first_ms;
    group->reqs.emplace_back(req.get());
    group->owned.emplace_back(std::move(req));

    {
        std::scoped_lock lk{_mutex};
        if (_queue_size >= _cfg.max_queue_size) [[unlikely]] {
            LUISA_ERROR("embedding scheduler: bounded queue overflow ({} requests in flight; max_queue_size={}).",
                        _queue_size, _cfg.max_queue_size);
        }
        _queue.emplace_back(group);
        _queue_size++;
    }
    _cv.notify_one();
}

void Scheduler::submit_batch(luisa::vector<luisa::unique_ptr<ScheduledRequest>> reqs) noexcept {

    if (reqs.empty()) [[unlikely]] {
        LUISA_ERROR("embedding scheduler: submit_batch with zero requests.");
    }
    const auto is_async = std::all_of(reqs.begin(), reqs.end(), [](const auto &r) { return r->callback != nullptr; });
    if (!is_async) [[unlikely]] {
        LUISA_ERROR("embedding scheduler: submit_batch requires callback requests (use encode_batch_sync otherwise).");
    }
    for (auto &r : reqs) {
        if (r->ids.size() < 2u || r->ids.size() > 8192u) [[unlikely]] {
            LUISA_ERROR("embedding scheduler: batch request of {} tokens outside [2, 8192].", r->ids.size());
        }
    }

    // One atomic job may not exceed kMaxJobRows padded rows (grid-cap
    // envelope above): split into order-preserving chunks. Per-sentence
    // results are row-local, so chunking does not change any output.
    luisa::vector<ScheduledRequest *> flat;
    flat.reserve(reqs.size());
    for (auto &r : reqs) { flat.emplace_back(r.get()); }
    auto chunks = split_atomic_chunks(flat);

    {
        std::scoped_lock lk{_mutex};
        if (_queue_size + reqs.size() > _cfg.max_queue_size) [[unlikely]] {
            LUISA_ERROR("embedding scheduler: batch of {} exceeds bounded queue capacity ({} in flight, max {}).",
                        reqs.size(), _queue_size, _cfg.max_queue_size);
        }
        const auto first_ms = _now_ms();
        size_t owned_i = 0u;// chunks preserve order -> walk reqs once
        for (auto &chunk : chunks) {
            auto *group = new PendingGroup{};
            group->first_ms = first_ms;
            group->atomic_batch = true;
            group->is_async = true;
            for (auto *rp : chunk) {
                rp->submitted_ms = first_ms;
                group->reqs.emplace_back(rp);
                group->owned.emplace_back(std::move(reqs[owned_i++]));
            }
            _queue.emplace_back(group);
        }
        _queue_size += static_cast<uint32_t>(reqs.size());
    }
    _cv.notify_one();
}

uint32_t Scheduler::_pick_worker_slot() noexcept {
    // requires _mutex; non-blocking Event::is_completed() poll (§3.5 acquire)
    for (auto &w : _workers) {
        if (w.busy && w.done_ev.is_completed()) {
            w.busy = false;
        }
    }
    for (uint32_t i = 0; i < _workers.size(); i++) {
        if (!_workers[i].busy) {
            return i;
        }
    }
    return static_cast<uint32_t>(_workers.size());
}

bool Scheduler::_try_dispatch() noexcept {

    luisa::fiber::lock lk{_mutex};
    const uint32_t slot = _pick_worker_slot();
    if (slot >= _workers.size() || _queue.empty()) {
        return false;
    }

    auto *front = _queue[0];
    const auto now = _now_ms();
    const auto bucket = embed_bucket_for(group_max_len(front->reqs));

    // FIFO per bucket: front + every later group with the same key (async vs
    // sync, bucket) that is not itself atomic, capped at max_batch_size AND
    // at kMaxJobRows padded rows (grid-cap envelope: rows = total sentences
    // x this bucket; both caps are "never split a group across jobs").
    luisa::vector<PendingGroup *> selected;
    selected.emplace_back(front);
    auto total = static_cast<uint32_t>(front->reqs.size());
    uint32_t sel_max_len = group_max_len(front->reqs);
    if (!front->atomic_batch) {
        for (size_t i = 1; i < _queue.size() && total < _cfg.max_batch_size; i++) {
            auto *g = _queue[i];
            if (g->atomic_batch || g->is_async != front->is_async ||
                embed_bucket_for(group_max_len(g->reqs)) != bucket) {
                continue;
            }
            const auto g_n = static_cast<uint32_t>(g->reqs.size());
            const auto rows_if_added =
                (total + g_n) * embed_bucket_for(std::max(sel_max_len, group_max_len(g->reqs)));
            if (total + g_n > _cfg.max_batch_size || rows_if_added > kMaxJobRows) {
                break;// never split a group across jobs
            }
            selected.emplace_back(g);
            total += g_n;
            sel_max_len = std::max(sel_max_len, group_max_len(g->reqs));
        }
    }

    // Launch policy: full coalesced batch, stale queue head
    // (max_queue_delay_ms), an explicit atomic batch, or any synchronous
    // group (blocking callers must not pay the coalescing window).
    const bool ready = front->atomic_batch ||
                       total >= _cfg.max_batch_size ||
                       (now - front->first_ms) >= static_cast<double>(_cfg.max_queue_delay_ms) ||
                       !front->is_async;
    if (!ready) {
        return false;
    }

    luisa::vector<ScheduledRequest *> reqs;
    luisa::vector<luisa::unique_ptr<ScheduledRequest>> owned;
    const bool is_async = front->is_async;
    for (auto *g : selected) {
        reqs.insert(reqs.end(), g->reqs.begin(), g->reqs.end());
        if (is_async) {
            for (auto &r : g->owned) {
                owned.emplace_back(std::move(r));
            }
        }
        auto it = std::find(_queue.begin(), _queue.end(), g);
        if (it != _queue.end()) {
            _queue.erase(it);
        }
        _queue_size -= static_cast<uint32_t>(g->reqs.size());
    }
    const bool atomic_batch = front->atomic_batch;
    for (auto *g : selected) {
        delete g;
    }

    _launch(std::move(owned), std::move(reqs), atomic_batch, slot, lk);
    return true;
}

Scheduler::Job *Scheduler::_launch(
    luisa::vector<luisa::unique_ptr<ScheduledRequest>> owned,
    luisa::vector<ScheduledRequest *> reqs, bool atomic_batch, uint32_t slot,
    luisa::fiber::lock &lk) noexcept {

    // Reserve the worker NOW (mutex still held by _try_dispatch); the
    // dispatcher is the only task that launches, so the slot cannot be
    // taken between here and the busy bookkeeping below. Then release the
    // lock for the slow recording/commit phase.
    auto &worker = _workers[slot];
    worker.busy = true;
    _outstanding_jobs++;
    lk.unlock_no_tsa();

    const auto H = _weights.hidden_size();
    auto *job = new Job{};
    job->owned = std::move(owned);
    job->reqs = std::move(reqs);
    const auto B = static_cast<uint32_t>(job->reqs.size());
    const auto max_len = group_max_len(job->reqs);
    // static batch coalesced job pads to the common bucket (§3.4); a lone
    // request runs at its exact length (kernels accept ragged S).
    const auto S = (B == 1u && !atomic_batch) ? max_len : embed_bucket_for(max_len);

    job->B = B;
    job->S = S;
    job->ids.assign(static_cast<size_t>(B) * S, 1u);// pad token id
    job->mask.assign(static_cast<size_t>(B) * S, 0.f);
    bool any_sparse = false, any_colbert = false;
    for (uint32_t b = 0; b < B; b++) {
        auto *r = job->reqs[b];
        const auto n = static_cast<uint32_t>(r->ids.size());
        std::copy_n(r->ids.begin(), n, job->ids.begin() + static_cast<ptrdiff_t>(b) * static_cast<ptrdiff_t>(S));
        std::fill_n(job->mask.begin() + static_cast<ptrdiff_t>(b) * static_cast<ptrdiff_t>(S), n, 1.0f);
        r->seq = S;
        r->row = b;
        r->opaque_job = job;
        r->worker_index = slot;
        any_sparse = any_sparse || r->want_sparse;
        any_colbert = any_colbert || r->want_colbert;
    }
    job->dense.assign(static_cast<size_t>(B) * H, 0.f);
    if (any_sparse) {
        job->sparse.assign(static_cast<size_t>(B) * S, 0.f);
    }
    if (any_colbert) {
        job->colbert.assign(static_cast<size_t>(B) * S * H, 0.f);
    }

    EncodeJob ej;
    ej.batch = B;
    ej.seq = S;
    ej.ids = luisa::span{job->ids};
    ej.mask = luisa::span{job->mask};
    ej.want_sparse = any_sparse;
    ej.want_colbert = any_colbert;
    ej.dense_out = luisa::span{job->dense};
    ej.sparse_out = any_sparse ? luisa::span{job->sparse} : luisa::span<float>{};
    ej.colbert_out = any_colbert ? luisa::span{job->colbert} : luisa::span<float>{};

    job->worker = slot;// MUST be set before the list (and its completion
    // callback) is committed: the async callback reads job->worker.
    auto cl = CommandList::create();
    worker.executor->record_encode(cl, ej);

    const bool has_sync = std::any_of(job->reqs.begin(), job->reqs.end(),
                                      [](const auto *r) { return r->callback == nullptr; });
    if (!has_sync) {
        // §8.8 completion callback: fires on the backend thread after every
        // command in this list (including the readback copies) finished. It
        // captures the job BY VALUE (unique_ptr move) and never touches a
        // stream.
        cl.add_callback([this, job]() noexcept {
            const auto done_ms = _now_ms();
            for (auto *r : job->reqs) {
                _record_sample(done_ms - r->submitted_ms);
                auto emb = _finalize(r, *job);
                if (r->callback) {
                    r->callback(std::move(emb));
                }
            }
            {
                std::scoped_lock lk{_mutex};
                _workers[job->worker].busy = false;// completion implies fence done
                _outstanding_jobs--;
                _cv.notify_all();
            }
            delete job;
        });
    }

    worker.stream << cl.commit();
    auto sig = worker.done_ev.signal();
    const auto sig_fence = sig.fence;
    worker.stream << std::move(sig);// event ops must be fed as rvalues
    job->fence = sig_fence;
    worker.busy_fence = sig_fence;

    if (has_sync) {
        uint32_t sync_waiters = 0u;
        for (auto *r : job->reqs) {
            if (r->callback == nullptr) {
                r->fence = sig_fence;
                sync_waiters++;
            }
        }
        // an atomic batch is driven by exactly one caller; coalesced single
        // sync requests have one caller each. The publish of sync_ready (and
        // with it fence/worker/opaque_job) happens under the mutex so the
        // waiters observe fully-initialized request state.
        job->pending_sync_callers = atomic_batch ? 1u : sync_waiters;
        {
            std::scoped_lock lk2{_mutex};
            for (auto *r : job->reqs) {
                if (r->callback == nullptr) {
                    r->sync_ready = true;
                }
            }
            _cv.notify_all();
        }
    }
    return job;
}

void Scheduler::_wait_assigned(ScheduledRequest *req) noexcept {
    luisa::fiber::lock lk{_mutex};
    _cv.wait(lk, [this, req] { return _stop.load() || req->sync_ready; });
}

Embedding Scheduler::encode_sync(luisa::unique_ptr<ScheduledRequest> &req) noexcept {

    if (req->callback != nullptr) [[unlikely]] {
        LUISA_ERROR("embedding scheduler: encode_sync request must not carry a callback.");
    }
    if (req->ids.size() < 2u || req->ids.size() > 8192u) [[unlikely]] {
        LUISA_ERROR("embedding scheduler: request of {} tokens outside [2, 8192].", req->ids.size());
    }

    auto *group = new PendingGroup{};
    group->first_ms = _now_ms();
    group->is_async = false;
    req->submitted_ms = group->first_ms;
    group->reqs.emplace_back(req.get());
    {
        std::scoped_lock lk{_mutex};
        _queue.emplace_back(group);
        _queue_size++;
    }
    _cv.notify_one();

    _wait_assigned(req.get());
    if (!req->sync_ready) {
        LUISA_WARNING("embedding scheduler: encode_sync aborted by shutdown.");
        return {};
    }

    // CPU-side wait on the WORKER EVENT FENCE -- never a queue-flush
    // stream::synchronize() (plan §3.5 completion-release "sync" path).
    auto &worker = _workers[req->worker_index];
    worker.done_ev.synchronize(req->fence);
    worker.executor->fire_trace_callbacks();// no-op unless tracing is on
    // B-1 barrier (default ON): drain the worker stream so the backend completion
    // worker has finished the readback memcpy into the Job's host staging before
    // it is read below and freed -- see the note in encode_batch_sync.
    if (b1_readback_barrier_enabled()) [[likely]] {
        worker.stream.synchronize();
    }

    auto *job = static_cast<Job *>(req->opaque_job);
    auto emb = _finalize(req.get(), *job);
    _record_sample(_now_ms() - req->submitted_ms);

    {
        std::scoped_lock lk{_mutex};
        if (--job->pending_sync_callers == 0u) {
            _outstanding_jobs--;
            _cv.notify_all();
            delete job;
        }
    }
    return emb;
}

luisa::vector<Embedding> Scheduler::encode_batch_sync(luisa::vector<luisa::unique_ptr<ScheduledRequest>> &reqs) noexcept {

    if (reqs.empty()) [[unlikely]] {
        LUISA_ERROR("embedding scheduler: encode_batch_sync with zero requests.");
    }
    for (auto &r : reqs) {
        if (r->callback != nullptr || r->ids.size() < 2u || r->ids.size() > 8192u) [[unlikely]] {
            LUISA_ERROR("embedding scheduler: encode_batch_sync request invalid (callback set or {} tokens).", r->ids.size());
        }
    }

    // kMaxJobRows envelope (see above): split the atomic batch into
    // order-preserving row-safe chunks, each its own atomic group.
    luisa::vector<ScheduledRequest *> flat;
    flat.reserve(reqs.size());
    for (auto &r : reqs) { flat.emplace_back(r.get()); }
    auto chunks = split_atomic_chunks(flat);

    const auto first_ms = _now_ms();
    for (auto &r : reqs) { r->submitted_ms = first_ms; }
    {
        std::scoped_lock lk{_mutex};
        if (_queue_size + static_cast<uint32_t>(reqs.size()) > _cfg.max_queue_size) [[unlikely]] {
            LUISA_ERROR("embedding scheduler: batch of {} exceeds bounded queue capacity ({} in flight, max {}).",
                        reqs.size(), _queue_size, _cfg.max_queue_size);
        }
        for (auto &chunk : chunks) {
            auto *group = new PendingGroup{};
            group->first_ms = first_ms;
            group->is_async = false;
            group->atomic_batch = true;
            for (auto *rp : chunk) { group->reqs.emplace_back(rp); }
            _queue.emplace_back(group);
        }
        _queue_size += static_cast<uint32_t>(reqs.size());// balances the dequeue
        // (_try_dispatch subtracts g->reqs.size()); the original +1 underflowed
        // the bounded-queue counter after the first dequeued atomic batch.
    }
    _cv.notify_one();

    // B-1 fix (multi-chunk submit path): a batch whose padded row total exceeds
    // kMaxJobRows is split by split_atomic_chunks into SEVERAL atomic jobs, each
    // launched on some worker with its OWN event fence. The previous code only
    // waited on and deleted reqs[0]'s job, so every later chunk was (a)
    // read back before its fence was signalled -- a host data race on the
    // job-owned dense/sparse/colbert staging (surfaced as the release-only
    // crash for large batches), (b) left un-deleted (Job leak) and (c) left
    // _outstanding_jobs inflated (the destructor drain wait never completes).
    // Correctness-only control flow; no numeric change; the single-chunk
    // (common) case behaves exactly as before (all reqs share one fence).

    // 1) Wait until EVERY request in the batch has been assigned (each chunk
    //    launched and its sync_ready published under the mutex). A single
    //    worker runs the chunks back-to-back, so the last chunk only launches
    //    once the dispatcher observes the previous chunk's fence completed --
    //    the dispatcher loop drives that, so this cannot deadlock.
    for (auto &r : reqs) {
        _wait_assigned(r.get());
        if (!r->sync_ready) [[unlikely]] {
            LUISA_WARNING("embedding scheduler: encode_batch_sync aborted by shutdown.");
            return {};
        }
    }

    // 2) Collect the DISTINCT jobs (one per atomic chunk), keyed by the
    //    scheduler-owned Job pointer stored on each request.
    luisa::vector<Job *> jobs;
    jobs.reserve(chunks.size());
    for (auto &r : reqs) {
        auto *job = static_cast<Job *>(r->opaque_job);
        bool seen = false;
        for (auto *j : jobs) {
            if (j == job) {
                seen = true;
                break;
            }
        }
        if (!seen) [[likely]] { jobs.emplace_back(job); }
    }

    // 3) CPU-side wait on each chunk job's fence (never a queue flush) and fire
    //    trace callbacks on each distinct worker once (debug-only hook).
    {
        luisa::vector<Worker *> traced_workers;
        for (auto &r : reqs) {
            auto &worker = _workers[r->worker_index];
            worker.done_ev.synchronize(r->fence);
            bool dup = false;
            for (auto *w : traced_workers) {
                if (w == &worker) {
                    dup = true;
                    break;
                }
            }
            if (!dup) [[likely]] {
                traced_workers.emplace_back(&worker);
                worker.executor->fire_trace_callbacks();
            }
        }
        // B-1 barrier (default ON, EMBED_B1_READBACK_BARRIER=0 to disable):
        // the event fence proves the GPU reached the job's tail, not that the
        // backend completion worker finished the readback memcpy into each
        // Job's host staging. Drain every involved worker stream fully before
        // the caller reads (_finalize) and frees the Jobs, so host staging is
        // never freed under an in-flight DtoH copy. Pure host ordering; no
        // numeric change; near-zero cost on a path that already blocks.
        if (b1_readback_barrier_enabled()) [[likely]] {
            for (auto *w : traced_workers) {
                w->stream.synchronize();
            }
        }
    }

    luisa::vector<Embedding> out;
    out.reserve(reqs.size());
    const auto now = _now_ms();
    for (auto &r : reqs) {
        auto *job = static_cast<Job *>(r->opaque_job);
        out.emplace_back(_finalize(r.get(), *job));
        _record_sample(now - r->submitted_ms);
    }
    {
        // 4) Retire every distinct chunk job exactly once (each atomic batch
        //    sets pending_sync_callers == 1, so a single decrement frees it).
        std::scoped_lock lk{_mutex};
        for (auto *job : jobs) {
            if (--job->pending_sync_callers == 0u) {
                _outstanding_jobs--;
                _cv.notify_all();
                delete job;
            }
        }
    }
    return out;
}

Embedding Scheduler::_finalize(ScheduledRequest *req, Job &job) noexcept {

    const auto H = _weights.hidden_size();
    const auto S = req->seq;
    const auto b = static_cast<ptrdiff_t>(req->row);
    const auto N = static_cast<uint32_t>(req->ids.size());

    Embedding out;
    const auto d0 = b * static_cast<ptrdiff_t>(H);
    out.dense.assign(job.dense.begin() + d0, job.dense.begin() + d0 + static_cast<ptrdiff_t>(H));

    // sparse compaction (plan §4 #10): keep (token_id, weight>0) pairs of the
    // REAL tokens; the template specials at position 0 (<s>) and N-1 (</s>)
    // are dropped, as are the pad rows s >= N.
    if (req->want_sparse) {
        const auto *w = job.sparse.data() + b * static_cast<ptrdiff_t>(S);
        for (uint32_t s = 1u; s + 1u < N; s++) {
            if (w[s] > 0.f) {
                out.sparse.emplace_back(req->ids[s], w[s]);
            }
        }
    }
    // colbert: per-token projected rows with specials dropped (golden
    // convention); already L2-normalized on device (normalize=1).
    if (req->want_colbert && N > 2u) {
        const auto *c = job.colbert.data() + b * static_cast<ptrdiff_t>(S) * static_cast<ptrdiff_t>(H);
        out.colbert.resize(static_cast<size_t>(N - 2u) * H);
        for (uint32_t s = 1u; s + 1u < N; s++) {
            std::copy_n(c + static_cast<ptrdiff_t>(s) * static_cast<ptrdiff_t>(H), static_cast<ptrdiff_t>(H),
                        out.colbert.begin() + static_cast<ptrdiff_t>(s - 1u) * static_cast<ptrdiff_t>(H));
        }
    }
    return out;
}

void Scheduler::set_layer_trace(LayerTraceCallback cb) noexcept {
    for (auto &w : _workers) {
        w.executor->set_layer_trace(cb);
    }
}

luisa::vector<double> Scheduler::take_latency_samples_ms() {
    std::scoped_lock lk{_stats_mutex};
    auto out = std::move(_latency_ms);
    _latency_ms.clear();
    return out;
}

void Scheduler::_record_sample(double ms) noexcept {
    std::scoped_lock lk{_stats_mutex};
    _latency_ms.emplace_back(ms);
}

void Scheduler::_dispatch_loop() noexcept {
    while (!_stop.load()) {
        {
            luisa::fiber::lock lk{_mutex};
            _cv.wait_for(lk, std::chrono::milliseconds{1},
                         [this] { return _stop.load() || !_queue.empty(); });
        }
        while (_try_dispatch()) {
        }
    }
}

}// namespace luisa::embed
