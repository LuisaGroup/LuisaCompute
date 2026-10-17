#pragma once
// bge-m3 embedding engine -- public Session API (embedding_model_plan.md §3.1,
// milestones M3/M6).
//
// A Session composes Tokenizer (A1) + Package loader (B3) + WeightStore +
// Executor pool + continuous-batching Scheduler (all plan §3.4/§3.5) behind
// the three encode entry points. Deviation from the §3.1 sketch: the caller
// does NOT pass a compute::Stream -- the session owns its streams internally
// (§3.5 worker pool; the sketch's Stream parameter predates the scheduler
// design).
//
// Thread-safety: all methods are safe to call concurrently from many host
// threads (the scheduler serializes dispatch internally). create() is
// expensive (weight upload + kernel compilation) and expected to run once.

#include <cstdint>
#include <filesystem>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/device.h>

#include "scheduler.h"// EncodeOptions / Embedding / SchedulerConfig

namespace luisa::embed {

class Package;
class Tokenizer;
class WeightStore;
class Scheduler;

class Session {
public:
    // Loads the packed model (model.bgm, plan §3.3) and tokenizer.json,
    // uploads the weights once behind the weights_ready fence, compiles all
    // kernels, and starts the P-worker continuous-batching scheduler.
    // Fatal LUISA_ERROR on any load/validation failure.
    static luisa::unique_ptr<Session> create(compute::Device &device,
                                             const std::filesystem::path &bgm_package,
                                             const std::filesystem::path &tokenizer_json,
                                             uint32_t num_workers = 2u);

    ~Session() noexcept;

    Session(const Session &) = delete;
    Session(Session &&) = delete;
    Session &operator=(const Session &) = delete;
    Session &operator=(Session &&) = delete;

    // Blocking single-text encode (sync scheduler path: waits on the worker
    // event fence, never a queue flush).
    Embedding encode(luisa::string_view utf8, const EncodeOptions &opt = {});

    // Static batch (§3.4): tokenizes every text, pads to the common bucket,
    // submits as one scheduler job, returns one Embedding per input (order
    // preserved).
    luisa::vector<Embedding> encode_batch(luisa::span<const luisa::string_view> texts,
                                          const EncodeOptions &opt = {});

    // Non-blocking encode (§3.5 async path): completion arrives on the
    // backend callback thread; the async path performs no host-side
    // synchronize() of any kind.
    void encode_async(luisa::string_view utf8, luisa::function<void(Embedding)> callback,
                      const EncodeOptions &opt = {});

    // Layer-trace plumbing for the D4 debug harness (plan §9.1; see the
    // executor.h contract). Broadcast to every worker executor; synchronous
    // firing, intended for single-session debugging. Unset with a null
    // callback (zero overhead).
    void set_layer_trace(LayerTraceCallback cb) noexcept;

    // M6 diagnostics: submit->completion latency samples in ms (drained).
    [[nodiscard]] luisa::vector<double> take_latency_samples_ms();

    [[nodiscard]] uint32_t worker_count() const noexcept;

private:
    Session() noexcept;
    luisa::unique_ptr<ScheduledRequest> _make_request(luisa::string_view utf8, const EncodeOptions &opt);

    compute::Device *_device{};
    luisa::unique_ptr<Package> _package;
    luisa::unique_ptr<Tokenizer> _tokenizer;
    luisa::unique_ptr<WeightStore> _weights;
    luisa::unique_ptr<Scheduler> _scheduler;
};

}// namespace luisa::embed
