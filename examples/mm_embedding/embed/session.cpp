// mm_embedding engine -- public Session API implementation.
#include "mm_embedding/embed/session.h"

#include <algorithm>

#include <luisa/core/logging.h>
#include <luisa/runtime/command_list.h>

#include "mm_embedding/embed/executor.h"
#include "mm_embedding/embed/image_processor.h"
#include "mm_embedding/embed/package.h"
#include "mm_embedding/embed/tokenizer.h"

namespace luisa::mm_embed {

Session::Session() noexcept = default;
Session::~Session() noexcept = default;

const Tokenizer &Session::tokenizer() const noexcept { return *_tokenizer; }
const Package &Session::package() const noexcept { return *_package; }
bool Session::has_vision() const noexcept { return _has_vision; }

luisa::unique_ptr<Session> Session::create(
    compute::Device &device,
    const std::filesystem::path &egm_package,
    const std::filesystem::path &tokenizer_json,
    const SessionConfig &config) {

    auto sess = luisa::make_unique<Session>();
    sess->_device = &device;
    sess->_stream = device.create_stream(compute::StreamTag::COMPUTE);

    LUISA_INFO("Session: loading package from '{}'...", egm_package.string());
    sess->_package = Package::load(egm_package);

    LUISA_INFO("Session: loading tokenizer from '{}'...", tokenizer_json.string());
    sess->_tokenizer = Tokenizer::load(tokenizer_json);

    LUISA_INFO("Session: initializing executor (enable_vision={})...", config.enable_vision);
    sess->_executor = Executor::create(device, *sess->_package, config.enable_vision);
    sess->_has_vision = sess->_executor->has_vision();

    LUISA_INFO("Session: ready for inference (vision={}).",
               sess->_has_vision ? "active" : "disabled");
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

    // Reuse persistent device buffers
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

Embedding Session::encode_image(const std::filesystem::path &image_path, const EncodeOptions &opt) {
    if (!_has_vision) {
        LUISA_ERROR("Session::encode_image: Vision Encoder is disabled (Selective Encoder Loading)");
    }
    auto processed = ImageProcessor::process_file(image_path);

    // Image tokens: [<bos>, <|image>, <|image|> * num_soft_tokens, <image|>, <eos>]
    // <bos> = 2, <|image> = 255999, <|image|> = 258880, <image|> = 258882, <eos> = 1
    uint32_t num_soft = processed.num_soft_tokens;
    uint32_t S = 1u + 1u + num_soft + 1u + 1u; // 260
    uint32_t target_dim = opt.target_dimension;

    _host_ids.clear();
    _host_ids.reserve(S);
    _host_ids.push_back(_tokenizer->bos_id()); // 2
    _host_ids.push_back(255999u);              // <|image> (BOI)
    for (uint32_t i = 0; i < num_soft; i++) {
        _host_ids.push_back(258880u);          // <|image|> (soft token placeholder)
    }
    _host_ids.push_back(258882u);              // <image|> (EOI)
    _host_ids.push_back(_tokenizer->eos_id()); // 1

    _host_mask.assign(S, 1.0f);

    // Prepare persistent vision buffers
    size_t pixel_elems = processed.max_patches * 768u;
    size_t pos_elems = processed.max_patches * 2u;
    size_t soft_elems = num_soft * 512u;

    if (!_dev_pixel_values || _dev_pixel_values.size() < pixel_elems) {
        _dev_pixel_values = _device->create_buffer<float>(pixel_elems);
        _dev_position_ids = _device->create_buffer<int32_t>(pos_elems);
        _dev_soft_tokens = _device->create_buffer<float>(280u * 512u);
    }
    if (!_dev_ids || _dev_ids.size() < S) {
        _dev_ids = _device->create_buffer<uint32_t>(S);
        _dev_mask = _device->create_buffer<float>(S);
    }
    if (!_dev_out || _dev_out.size() < target_dim) {
        _dev_out = _device->create_buffer<float>(target_dim);
    }
    _host_out.resize(target_dim);

    auto dev_px_view = _dev_pixel_values.view(0, pixel_elems);
    auto dev_pos_view = _dev_position_ids.view(0, pos_elems);
    auto dev_soft_view = _dev_soft_tokens.view(0, soft_elems);
    auto dev_ids_view = _dev_ids.view(0, S);
    auto dev_mask_view = _dev_mask.view(0, S);
    auto dev_out_view = _dev_out.view(0, target_dim);

    compute::CommandList cmd_list;
    // 1. Upload image pixels and position coordinates
    cmd_list << dev_px_view.copy_from(processed.pixel_values.data())
             << dev_pos_view.copy_from(processed.position_ids.data());

    // 2. Vision forward pass -> soft tokens [256, 512]
    _executor->record_vision_encode(
        cmd_list, dev_px_view, dev_pos_view, dev_soft_view,
        1u, processed.num_patches_w, processed.num_patches_h, processed.max_patches, num_soft);

    // 3. Upload prompt token IDs and attention mask
    cmd_list << dev_ids_view.copy_from(_host_ids.data())
             << dev_mask_view.copy_from(_host_mask.data());

    // 4. Text core forward pass with soft tokens scattered at index 2
    MultimodalInputs mm_inputs{};
    mm_inputs.vision_embeds = dev_soft_view;
    mm_inputs.vision_start_idx = 2u;
    mm_inputs.num_vision_tokens = num_soft;

    _executor->record_encode(cmd_list, dev_ids_view, dev_mask_view, dev_out_view, 1u, S, target_dim, mm_inputs);

    // 5. Read back embedding
    cmd_list << dev_out_view.copy_to(_host_out.data());

    _stream << cmd_list.commit() << compute::synchronize();

    Embedding emb{};
    emb.dimension = target_dim;
    emb.values = std::move(_host_out);
    return emb;
}

Embedding Session::encode_image_memory(
    const uint8_t *rgb_bytes,
    int width,
    int height,
    int channels,
    const EncodeOptions &opt) {

    if (!_has_vision) {
        LUISA_ERROR("Session::encode_image_memory: Vision Encoder is disabled (Selective Encoder Loading)");
    }
    auto processed = ImageProcessor::process_memory(rgb_bytes, width, height, channels);

    uint32_t num_soft = processed.num_soft_tokens;
    uint32_t S = 1u + 1u + num_soft + 1u + 1u;
    uint32_t target_dim = opt.target_dimension;

    _host_ids.clear();
    _host_ids.reserve(S);
    _host_ids.push_back(_tokenizer->bos_id());
    _host_ids.push_back(255999u);
    for (uint32_t i = 0; i < num_soft; i++) {
        _host_ids.push_back(258880u);
    }
    _host_ids.push_back(258882u);
    _host_ids.push_back(_tokenizer->eos_id());

    _host_mask.assign(S, 1.0f);

    size_t pixel_elems = processed.max_patches * 768u;
    size_t pos_elems = processed.max_patches * 2u;
    size_t soft_elems = num_soft * 512u;

    if (!_dev_pixel_values || _dev_pixel_values.size() < pixel_elems) {
        _dev_pixel_values = _device->create_buffer<float>(pixel_elems);
        _dev_position_ids = _device->create_buffer<int32_t>(pos_elems);
        _dev_soft_tokens = _device->create_buffer<float>(280u * 512u);
    }
    if (!_dev_ids || _dev_ids.size() < S) {
        _dev_ids = _device->create_buffer<uint32_t>(S);
        _dev_mask = _device->create_buffer<float>(S);
    }
    if (!_dev_out || _dev_out.size() < target_dim) {
        _dev_out = _device->create_buffer<float>(target_dim);
    }
    _host_out.resize(target_dim);

    auto dev_px_view = _dev_pixel_values.view(0, pixel_elems);
    auto dev_pos_view = _dev_position_ids.view(0, pos_elems);
    auto dev_soft_view = _dev_soft_tokens.view(0, soft_elems);
    auto dev_ids_view = _dev_ids.view(0, S);
    auto dev_mask_view = _dev_mask.view(0, S);
    auto dev_out_view = _dev_out.view(0, target_dim);

    compute::CommandList cmd_list;
    cmd_list << dev_px_view.copy_from(processed.pixel_values.data())
             << dev_pos_view.copy_from(processed.position_ids.data());

    _executor->record_vision_encode(
        cmd_list, dev_px_view, dev_pos_view, dev_soft_view,
        1u, processed.num_patches_w, processed.num_patches_h, processed.max_patches, num_soft);

    cmd_list << dev_ids_view.copy_from(_host_ids.data())
             << dev_mask_view.copy_from(_host_mask.data());

    MultimodalInputs mm_inputs{};
    mm_inputs.vision_embeds = dev_soft_view;
    mm_inputs.vision_start_idx = 2u;
    mm_inputs.num_vision_tokens = num_soft;

    _executor->record_encode(cmd_list, dev_ids_view, dev_mask_view, dev_out_view, 1u, S, target_dim, mm_inputs);
    cmd_list << dev_out_view.copy_to(_host_out.data());

    _stream << cmd_list.commit() << compute::synchronize();

    Embedding emb{};
    emb.dimension = target_dim;
    emb.values = std::move(_host_out);
    return emb;
}

Embedding Session::encode_multimodal(
    const std::filesystem::path &image_path,
    luisa::string_view prompt_text,
    const EncodeOptions &opt) {

    if (!_has_vision) {
        LUISA_ERROR("Session::encode_multimodal: Vision Encoder is disabled (Selective Encoder Loading)");
    }
    auto processed = ImageProcessor::process_file(image_path);
    auto text_toks = _tokenizer->encode(prompt_text, opt.max_length);

    // Image prefix: [<bos>, <|image>, <|image|> * num_soft, <image|>]
    uint32_t num_soft = processed.num_soft_tokens;
    _host_ids.clear();
    _host_ids.push_back(_tokenizer->bos_id()); // 2
    _host_ids.push_back(255999u);              // <|image>
    for (uint32_t i = 0; i < num_soft; i++) {
        _host_ids.push_back(258880u);          // <|image|>
    }
    _host_ids.push_back(258882u);              // <image|>

    // Append text tokens (skip leading <bos> from text_toks if present)
    size_t text_start = (!text_toks.empty() && text_toks.front() == _tokenizer->bos_id()) ? 1u : 0u;
    for (size_t i = text_start; i < text_toks.size(); i++) {
        _host_ids.push_back(text_toks[i]);
    }

    uint32_t S = static_cast<uint32_t>(_host_ids.size());
    uint32_t target_dim = opt.target_dimension;
    _host_mask.assign(S, 1.0f);

    size_t pixel_elems = processed.max_patches * 768u;
    size_t pos_elems = processed.max_patches * 2u;
    size_t soft_elems = num_soft * 512u;

    if (!_dev_pixel_values || _dev_pixel_values.size() < pixel_elems) {
        _dev_pixel_values = _device->create_buffer<float>(pixel_elems);
        _dev_position_ids = _device->create_buffer<int32_t>(pos_elems);
        _dev_soft_tokens = _device->create_buffer<float>(280u * 512u);
    }
    if (!_dev_ids || _dev_ids.size() < S) {
        _dev_ids = _device->create_buffer<uint32_t>(S);
        _dev_mask = _device->create_buffer<float>(S);
    }
    if (!_dev_out || _dev_out.size() < target_dim) {
        _dev_out = _device->create_buffer<float>(target_dim);
    }
    _host_out.resize(target_dim);

    auto dev_px_view = _dev_pixel_values.view(0, pixel_elems);
    auto dev_pos_view = _dev_position_ids.view(0, pos_elems);
    auto dev_soft_view = _dev_soft_tokens.view(0, soft_elems);
    auto dev_ids_view = _dev_ids.view(0, S);
    auto dev_mask_view = _dev_mask.view(0, S);
    auto dev_out_view = _dev_out.view(0, target_dim);

    compute::CommandList cmd_list;
    cmd_list << dev_px_view.copy_from(processed.pixel_values.data())
             << dev_pos_view.copy_from(processed.position_ids.data());

    _executor->record_vision_encode(
        cmd_list, dev_px_view, dev_pos_view, dev_soft_view,
        1u, processed.num_patches_w, processed.num_patches_h, processed.max_patches, num_soft);

    cmd_list << dev_ids_view.copy_from(_host_ids.data())
             << dev_mask_view.copy_from(_host_mask.data());

    MultimodalInputs mm_inputs{};
    mm_inputs.vision_embeds = dev_soft_view;
    mm_inputs.vision_start_idx = 2u;
    mm_inputs.num_vision_tokens = num_soft;

    _executor->record_encode(cmd_list, dev_ids_view, dev_mask_view, dev_out_view, 1u, S, target_dim, mm_inputs);
    cmd_list << dev_out_view.copy_to(_host_out.data());

    _stream << cmd_list.commit() << compute::synchronize();

    Embedding emb{};
    emb.dimension = target_dim;
    emb.values = std::move(_host_out);
    return emb;
}

}// namespace luisa::mm_embed
