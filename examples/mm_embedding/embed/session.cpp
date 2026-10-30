// mm_embedding engine -- public Session API implementation.
#include "mm_embedding/embed/session.h"

#include <algorithm>

#include <luisa/core/logging.h>
#include <luisa/runtime/command_list.h>

#include "mm_embedding/embed/executor.h"
#include "mm_embedding/embed/package.h"
#include "mm_embedding/embed/tokenizer.h"

namespace luisa::mm_embed {

Session::Session() noexcept = default;
Session::~Session() noexcept = default;

const Tokenizer &Session::tokenizer() const noexcept { return *_tokenizer; }
const Package &Session::package() const noexcept { return *_package; }

luisa::unique_ptr<Session> Session::create(
    compute::Device &device,
    const std::filesystem::path &egm_package,
    const std::filesystem::path &tokenizer_json) {

    auto sess = luisa::make_unique<Session>();
    sess->_device = &device;
    sess->_stream = device.create_stream(compute::StreamTag::COMPUTE);

    LUISA_INFO("Session: loading package from '{}'...", egm_package.string());
    sess->_package = Package::load(egm_package);

    LUISA_INFO("Session: loading tokenizer from '{}'...", tokenizer_json.string());
    sess->_tokenizer = Tokenizer::load(tokenizer_json);

    LUISA_INFO("Session: initializing executor and uploading weights to GPU...");
    sess->_executor = Executor::create(device, *sess->_package);

    LUISA_INFO("Session: ready for inference.");
    return sess;
}

Embedding Session::encode(luisa::string_view text, const EncodeOptions &opt) {
    luisa::string_view text_arr[1] = {text};
    auto batch_res = encode_batch(luisa::span{text_arr, 1u}, opt);
    return std::move(batch_res[0]);
}

luisa::vector<Embedding> Session::encode_batch(
    luisa::span<const luisa::string_view> texts,
    const EncodeOptions &opt) {

    if (texts.empty()) {
        return {};
    }

    uint32_t B = static_cast<uint32_t>(texts.size());
    auto tokenized = _tokenizer->encode_batch(texts, opt.max_length);

    // Determine common padded sequence length S
    uint32_t max_seq = 0;
    for (const auto &toks : tokenized) {
        max_seq = std::max(max_seq, static_cast<uint32_t>(toks.size()));
    }
    max_seq = std::max(max_seq, 1u);
    max_seq = std::min(max_seq, opt.max_length);

    uint32_t S = max_seq;
    uint32_t target_dim = opt.target_dimension;
    size_t total_tokens = static_cast<size_t>(B) * S;

    // Prepare host input_ids and attention_mask using persistent host vectors
    _host_ids.assign(total_tokens, _tokenizer->pad_id());
    _host_mask.assign(total_tokens, 0.0f);

    for (uint32_t b = 0; b < B; b++) {
        const auto &toks = tokenized[b];
        uint32_t len = std::min(static_cast<uint32_t>(toks.size()), S);
        for (uint32_t s = 0; s < len; s++) {
            _host_ids[b * S + s] = toks[s];
            _host_mask[b * S + s] = 1.0f;
        }
    }

    // Reuse persistent device buffers; only reallocate if required capacity expands
    if (!_dev_ids || _dev_ids.size() < total_tokens) {
        _dev_ids = _device->create_buffer<uint32_t>(total_tokens);
        _dev_mask = _device->create_buffer<float>(total_tokens);
    }
    size_t total_out = static_cast<size_t>(B) * target_dim;
    if (!_dev_out || _dev_out.size() < total_out) {
        _dev_out = _device->create_buffer<float>(total_out);
    }
    _host_out.resize(total_out);

    auto dev_ids_view = _dev_ids.view(0, total_tokens);
    auto dev_mask_view = _dev_mask.view(0, total_tokens);
    auto dev_out_view = _dev_out.view(0, total_out);

    compute::CommandList cmd_list;
    cmd_list << dev_ids_view.copy_from(_host_ids.data())
             << dev_mask_view.copy_from(_host_mask.data());

    _executor->record_encode(cmd_list, dev_ids_view, dev_mask_view, dev_out_view, B, S, target_dim);

    cmd_list << dev_out_view.copy_to(_host_out.data());

    _stream << cmd_list.commit() << compute::synchronize();

    // Package output embeddings
    luisa::vector<Embedding> result;
    result.reserve(B);
    for (uint32_t b = 0; b < B; b++) {
        Embedding emb{};
        emb.dimension = target_dim;
        emb.values.assign(_host_out.begin() + b * target_dim,
                          _host_out.begin() + (b + 1) * target_dim);
        result.push_back(std::move(emb));
    }

    return result;
}

}// namespace luisa::mm_embed
