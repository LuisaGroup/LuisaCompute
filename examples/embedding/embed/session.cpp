// bge-m3 embedding engine -- Session implementation (plan §3.1/§3.4/§3.5).
// See session.h for the API contract and deviation notes.
#include "session.h"

#include <algorithm>

#include <luisa/core/logging.h>

#include "executor.h"
#include "package.h"
#include "scheduler.h"
#include "tokenizer.h"

namespace luisa::embed {

using compute::Device;

Session::Session() noexcept = default;

luisa::unique_ptr<Session> Session::create(Device &device,
                                           const std::filesystem::path &bgm_package,
                                           const std::filesystem::path &tokenizer_json,
                                           uint32_t num_workers) {

    auto session = luisa::unique_ptr<Session>{new Session{}};
    session->m_device = &device;

    session->m_package = Package::load(bgm_package);
    if(session->m_package == nullptr) [[unlikely]] {
        LUISA_ERROR("embedding Session: failed to load package \"{}\".", bgm_package.string());
    }
    if(session->m_package->num_layers() != 24u || session->m_package->hidden_size() != 1024u) {
        // Not fatal: the executor derives shapes from the package; this is
        // the bge-m3 assumption of the golden gates only. Log it.
        LUISA_WARNING("embedding Session: package is not the expected 24x1024 bge-m3 "
                      "(layers={}, hidden={}) -- goldens assume bge-m3.",
                      session->m_package->num_layers(), session->m_package->hidden_size());
    }

    session->m_tokenizer = Tokenizer::load(tokenizer_json);

    session->m_weights = luisa::make_unique<WeightStore>(device, *session->m_package);

    SchedulerConfig cfg;
    cfg.num_workers = num_workers;
    session->m_scheduler = luisa::make_unique<Scheduler>(device, *session->m_weights, cfg);

    LUISA_INFO("embedding Session ready: {} layers, H={}, {} weights-queue workers.",
               session->m_package->num_layers(), session->m_package->hidden_size(),
               session->m_scheduler->worker_count());
    return session;
}

Session::~Session() noexcept = default;

luisa::unique_ptr<ScheduledRequest> Session::m_make_request(luisa::string_view utf8, const EncodeOptions &opt) {

    auto req = luisa::make_unique<ScheduledRequest>();
    const auto max_len = std::min<uint32_t>(std::max<uint32_t>(opt.max_seq_len, 2u), 8192u);
    req->ids = m_tokenizer->encode(utf8, max_len);
    req->want_sparse = opt.want_sparse;
    req->want_colbert = opt.want_colbert;
    return req;
}

Embedding Session::encode(luisa::string_view utf8, const EncodeOptions &opt) {
    auto req = m_make_request(utf8, opt);
    return m_scheduler->encode_sync(req);
}

luisa::vector<Embedding> Session::encode_batch(luisa::span<const luisa::string_view> texts, const EncodeOptions &opt) {

    luisa::vector<luisa::unique_ptr<ScheduledRequest>> reqs;
    reqs.reserve(texts.size());
    for(const auto &t : texts) {
        reqs.emplace_back(m_make_request(t, opt));
    }
    return m_scheduler->encode_batch_sync(reqs);
}

void Session::encode_async(luisa::string_view utf8, luisa::function<void(Embedding)> callback, const EncodeOptions &opt) {
    auto req = m_make_request(utf8, opt);
    req->callback = std::move(callback);
    m_scheduler->submit(std::move(req));
}

void Session::set_layer_trace(LayerTraceCallback cb) noexcept {
    m_scheduler->set_layer_trace(std::move(cb));
}

luisa::vector<double> Session::take_latency_samples_ms() {
    return m_scheduler->take_latency_samples_ms();
}

uint32_t Session::worker_count() const noexcept {
    return m_scheduler->worker_count();
}

}// namespace luisa::embed
