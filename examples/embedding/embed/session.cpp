// bge-m3 embedding engine -- Session implementation (plan §3.1/§3.4/§3.5).
// See session.h for the API contract and deviation notes.
#include "session.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <utility>

#include <luisa/core/logging.h>

#include "executor.h"
#include "package.h"
#include "scheduler.h"
#include "tokenizer.h"

namespace luisa::embed {

using compute::Device;

namespace {
// Runtime kill-switches for the host-memory optimizations (H8 tokenizer pool,
// H10 length bucketing). Each is read exactly once from the environment (a
// thread-safe magic-static init) and cached: default ON, disabled by setting
// the variable to "0". Individually toggleable without a rebuild, so a
// regression can be reverted by the parent via environment alone. No mutable
// shared state after first init -> safe for concurrent encode_batch calls.
bool env_flag_on(luisa::string_view name, bool default_on) noexcept {
    const auto *raw = std::getenv(luisa::string{name}.c_str());
    return (raw == nullptr) ? default_on : (luisa::string_view{raw} != luisa::string_view{"0"});
}
bool h8_pool_on() noexcept {
    static const bool on = env_flag_on("EMBED_H8_TOKENIZER_POOL", true);
    return on;
}
bool h10_bucket_on() noexcept {
    static const bool on = env_flag_on("EMBED_H10_LENGTH_BUCKET", true);
    return on;
}
}// namespace

Session::Session() noexcept = default;

luisa::unique_ptr<Session> Session::create(Device &device,
                                           const std::filesystem::path &bgm_package,
                                           const std::filesystem::path &tokenizer_json,
                                           uint32_t num_workers) {

    auto session = luisa::unique_ptr<Session>{new Session{}};
    session->_device = &device;

    session->_package = Package::load(bgm_package);
    if (session->_package == nullptr) [[unlikely]] {
        LUISA_ERROR("embedding Session: failed to load package \"{}\".", bgm_package.string());
    }
    if (session->_package->num_layers() != 24u || session->_package->hidden_size() != 1024u) {
        // Not fatal: the executor derives shapes from the package; this is
        // the bge-m3 assumption of the golden gates only. Log it.
        LUISA_WARNING("embedding Session: package is not the expected 24x1024 bge-m3 "
                      "(layers={}, hidden={}) -- goldens assume bge-m3.",
                      session->_package->num_layers(), session->_package->hidden_size());
    }

    session->_tokenizer = Tokenizer::load(tokenizer_json);

    session->_weights = luisa::make_unique<WeightStore>(device, *session->_package);

    SchedulerConfig cfg;
    cfg.num_workers = num_workers;
    session->_scheduler = luisa::make_unique<Scheduler>(device, *session->_weights, cfg);

    LUISA_INFO("embedding Session ready: {} layers, H={}, {} weights-queue workers, "
               "tensor-path='{}', gemm='{}'.",
               session->_package->num_layers(), session->_package->hidden_size(),
               session->_scheduler->worker_count(),
               tensor_path_name(tensor_path_requested()),
               gemm_resolved_config_source());
    return session;
}

Session::~Session() noexcept = default;

luisa::unique_ptr<ScheduledRequest> Session::_make_request(luisa::string_view utf8, const EncodeOptions &opt) {

    auto req = luisa::make_unique<ScheduledRequest>();
    const auto max_len = std::min<uint32_t>(std::max<uint32_t>(opt.max_seq_len, 2u), 8192u);
    req->ids = _tokenizer->encode(utf8, max_len);
    req->want_sparse = opt.want_sparse;
    req->want_colbert = opt.want_colbert;
    return req;
}

Embedding Session::encode(luisa::string_view utf8, const EncodeOptions &opt) {
    auto req = _make_request(utf8, opt);
    return _scheduler->encode_sync(req);
}

luisa::vector<Embedding> Session::encode_batch(luisa::span<const luisa::string_view> texts, const EncodeOptions &opt) {

    const auto max_len = std::min<uint32_t>(std::max<uint32_t>(opt.max_seq_len, 2u), 8192u);

    // H8: tokenize the whole batch up front. The Tokenizer is reentrant after
    // load, so the batch is encoded by fiber worker tasks; the result is
    // order-preserving and bit-identical to the serial encode loop. Set
    // EMBED_H8_TOKENIZER_POOL=0 to fall back to the serial path. encode() and
    // encode_async() keep using the single-text _make_request path.
    luisa::vector<luisa::vector<uint32_t>> all_ids;
    if (h8_pool_on()) [[likely]] {
        all_ids = _tokenizer->encode_parallel(texts, max_len);
    } else [[unlikely]] {
        all_ids.resize(texts.size());
        for (size_t i = 0u; i < texts.size(); i++) {
            all_ids[i] = _tokenizer->encode(texts[i], max_len);
        }
    }

    const size_t N = texts.size();
    luisa::vector<luisa::unique_ptr<ScheduledRequest>> reqs;
    reqs.reserve(N);
    luisa::vector<uint32_t> orig_of_sorted;// sorted slot -> original input index
    orig_of_sorted.reserve(N);

    // H10: sort by tokenized length (bucket boundaries 128/512/2048/8192 via
    // embed_bucket_for, then exact length, then original index for a total,
    // deterministic order) so each scheduler chunk lands inside one padding
    // bucket. Padding is correctness-neutral (masked residuals), so this only
    // removes wasted GEMM/attention work on ragged batches -- it must NOT change
    // any per-sentence result. Results are un-sorted back to input order below,
    // which test_embed_e2e LEG5 bit-identity hard-gates. Only the request vector
    // is reordered; Scheduler / executor / kMaxJobRows envelope are untouched.
    // Set EMBED_H10_LENGTH_BUCKET=0 to submit in input order (identity perm).
    luisa::vector<uint32_t> order;
    order.reserve(N);
    if (h10_bucket_on()) [[likely]] {
        for (uint32_t i = 0u; i < N; i++) { order.push_back(i); }
        std::sort(order.begin(), order.end(), [&all_ids](uint32_t a, uint32_t b) noexcept {
            const auto sa = all_ids[a].size();
            const auto sb = all_ids[b].size();
            const auto ba = embed_bucket_for(static_cast<uint32_t>(sa));
            const auto bb = embed_bucket_for(static_cast<uint32_t>(sb));
            if (ba != bb) { return ba < bb; }
            if (sa != sb) { return sa < sb; }
            return a < b;
        });
    } else [[unlikely]] {
        for (uint32_t i = 0u; i < N; i++) { order.push_back(i); }
    }
    for (const auto idx : order) {
        auto req = luisa::make_unique<ScheduledRequest>();
        req->ids = std::move(all_ids[idx]);
        req->want_sparse = opt.want_sparse;
        req->want_colbert = opt.want_colbert;
        reqs.emplace_back(std::move(req));
        orig_of_sorted.push_back(idx);
    }

    auto out = _scheduler->encode_batch_sync(reqs);

    // Restore exact input/output ordering. The scheduler returns one Embedding
    // per request in the request-vector order; on the shutdown-abort path it
    // returns an empty vector, which we pass through unchanged (as before).
    if (out.size() == reqs.size()) [[likely]] {
        luisa::vector<Embedding> result;
        result.resize(N);
        for (size_t k = 0u; k < out.size(); k++) {
            result[orig_of_sorted[k]] = std::move(out[k]);
        }
        return result;
    } else [[unlikely]] {
        return out;
    }
}

void Session::encode_async(luisa::string_view utf8, luisa::function<void(Embedding)> callback, const EncodeOptions &opt) {
    auto req = _make_request(utf8, opt);
    req->callback = std::move(callback);
    _scheduler->submit(std::move(req));
}

void Session::set_layer_trace(LayerTraceCallback cb) noexcept {
    _scheduler->set_layer_trace(std::move(cb));
}

luisa::vector<double> Session::take_latency_samples_ms() {
    return _scheduler->take_latency_samples_ms();
}

uint32_t Session::worker_count() const noexcept {
    return _scheduler->worker_count();
}

}// namespace luisa::embed
