// embedding_mm engine -- public Session API implementation.
#include "embedding_mm/embed/session.h"

#include <algorithm>

#include <luisa/core/logging.h>
#include <luisa/runtime/command_list.h>

#include "embedding_mm/embed/audio_processor.h"
#include "embedding_mm/embed/executor.h"
#include "embedding_mm/embed/image_processor.h"
#include "embedding_mm/embed/video_processor.h"
#include "embedding_mm/embed/package.h"
#include "embedding_mm/embed/tokenizer.h"

namespace luisa::embed_mm {

Session::Session() noexcept = default;
Session::~Session() noexcept = default;

const Tokenizer &Session::tokenizer() const noexcept { return *_tokenizer; }
const Package &Session::package() const noexcept { return *_package; }
bool Session::has_vision() const noexcept { return _has_vision; }
bool Session::has_audio() const noexcept { return _has_audio; }

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

    LUISA_INFO("Session: initializing executor (enable_vision={}, enable_audio={})...",
               config.enable_vision, config.enable_audio);
    sess->_executor = Executor::create(device, *sess->_package, config.enable_vision, config.enable_audio);
    sess->_has_vision = sess->_executor->has_vision();
    sess->_has_audio = sess->_executor->has_audio();

    LUISA_INFO("Session: ready for inference (vision={}, audio={}).",
               sess->_has_vision ? "active" : "disabled",
               sess->_has_audio ? "active" : "disabled");
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
    const std::filesystem::path &media_path,
    luisa::string_view prompt_text,
    const EncodeOptions &opt) {

    if (!_has_vision) {
        LUISA_ERROR("Session::encode_multimodal: Vision Encoder is disabled (Selective Encoder Loading)");
    }

    std::string prompt_str(prompt_text.data(), prompt_text.size());
    auto ext = media_path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    // If media_path is a video container or prompt contains <|video|>
    if (ext == ".mp4" || ext == ".m4v" || ext == ".mov" || ext == ".wmv" || ext == ".avi" || ext == ".webm" ||
        prompt_str.find("<|video|>") != std::string::npos) {
        return encode_multimodal_video(media_path, prompt_text, {}, opt);
    }

    auto processed = ImageProcessor::process_file(media_path);
    uint32_t num_soft = processed.num_soft_tokens;
    uint32_t vision_start_idx = 2u;

    _host_ids.clear();
    size_t img_token_pos = prompt_str.find("<|image|>");
    if (img_token_pos != std::string::npos) {
        std::string text_before = prompt_str.substr(0, img_token_pos);
        std::string text_after = prompt_str.substr(img_token_pos + 9u);
        auto prefix_ids = _tokenizer->encode(text_before);
        auto suffix_ids = _tokenizer->encode(text_after);

        _host_ids.push_back(_tokenizer->bos_id());
        size_t pre_start = (!prefix_ids.empty() && prefix_ids.front() == _tokenizer->bos_id()) ? 1u : 0u;
        for (size_t i = pre_start; i < prefix_ids.size(); i++) {
            if (prefix_ids[i] != _tokenizer->eos_id()) {
                _host_ids.push_back(prefix_ids[i]);
            }
        }
        vision_start_idx = static_cast<uint32_t>(_host_ids.size() + 1u);
        _host_ids.push_back(255999u); // <|image>
        for (uint32_t i = 0; i < num_soft; i++) {
            _host_ids.push_back(258880u); // <|image|>
        }
        _host_ids.push_back(258882u); // <image|>
        size_t suf_start = (!suffix_ids.empty() && suffix_ids.front() == _tokenizer->bos_id()) ? 1u : 0u;
        for (size_t i = suf_start; i < suffix_ids.size(); i++) {
            if (suffix_ids[i] != _tokenizer->eos_id()) {
                _host_ids.push_back(suffix_ids[i]);
            }
        }
        _host_ids.push_back(_tokenizer->eos_id());
    } else {
        auto text_toks = _tokenizer->encode(prompt_text, opt.max_length);
        // Image prefix: [<bos>, <|image>, <|image|> * num_soft, <image|>]
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
    mm_inputs.vision_start_idx = vision_start_idx;
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
    luisa::span<const std::filesystem::path> frame_paths,
    luisa::string_view prompt_text,
    const VideoConfig &cfg,
    const EncodeOptions &opt) {
    return encode_multimodal_video(frame_paths, prompt_text, cfg, opt);
}

Embedding Session::encode_multimodal(
    const std::filesystem::path &media_path,
    luisa::string_view prompt_text,
    const VideoConfig &cfg,
    const EncodeOptions &opt) {
    return encode_multimodal_video(media_path, prompt_text, cfg, opt);
}

Embedding Session::encode_audio(
    const std::filesystem::path &audio_path,
    const EncodeOptions &opt) {

    if (!_has_audio) {
        LUISA_ERROR("Session::encode_audio: Audio Encoder is disabled (Selective Encoder Loading)");
    }
    auto processed = AudioProcessor::process_file(audio_path, _package.get());

    uint32_t num_soft = processed.num_soft_tokens;
    _host_ids.clear();
    _host_ids.push_back(_tokenizer->bos_id()); // 2 (<bos>)
    _host_ids.push_back(256000u);              // <|audio>
    for (uint32_t i = 0; i < num_soft; i++) {
        _host_ids.push_back(258881u);          // <|audio|>
    }
    _host_ids.push_back(258883u);              // <audio|>
    _host_ids.push_back(1u);                   // <eos>

    uint32_t S = static_cast<uint32_t>(_host_ids.size());
    uint32_t target_dim = opt.target_dimension;
    _host_mask.assign(S, 1.0f);

    size_t feat_elems = processed.features.size();
    size_t soft_elems = num_soft * 512u;

    if (!_dev_audio_features || _dev_audio_features.size() < feat_elems) {
        _dev_audio_features = _device->create_buffer<float>(feat_elems);
    }
    if (!_dev_audio_soft_tokens || _dev_audio_soft_tokens.size() < soft_elems) {
        _dev_audio_soft_tokens = _device->create_buffer<float>(soft_elems);
    }
    if (!_dev_ids || _dev_ids.size() < S) {
        _dev_ids = _device->create_buffer<uint32_t>(S);
        _dev_mask = _device->create_buffer<float>(S);
    }
    if (!_dev_out || _dev_out.size() < target_dim) {
        _dev_out = _device->create_buffer<float>(target_dim);
    }
    _host_out.resize(target_dim);

    auto dev_feat_view = _dev_audio_features.view(0, feat_elems);
    auto dev_soft_view = _dev_audio_soft_tokens.view(0, soft_elems);
    auto dev_ids_view = _dev_ids.view(0, S);
    auto dev_mask_view = _dev_mask.view(0, S);
    auto dev_out_view = _dev_out.view(0, target_dim);

    compute::CommandList cmd_list;
    cmd_list << dev_feat_view.copy_from(processed.features.data());

    _executor->record_audio_encode(
        cmd_list, dev_feat_view, dev_soft_view,
        processed.num_frames, processed.num_features, num_soft);

    cmd_list << dev_ids_view.copy_from(_host_ids.data())
             << dev_mask_view.copy_from(_host_mask.data());

    MultimodalInputs mm_inputs{};
    mm_inputs.audio_embeds = dev_soft_view;
    mm_inputs.audio_start_idx = 2u;
    mm_inputs.num_audio_tokens = num_soft;

    _executor->record_encode(cmd_list, dev_ids_view, dev_mask_view, dev_out_view, 1u, S, target_dim, mm_inputs);
    cmd_list << dev_out_view.copy_to(_host_out.data());

    _stream << cmd_list.commit() << compute::synchronize();

    Embedding emb{};
    emb.dimension = target_dim;
    emb.values = std::move(_host_out);
    return emb;
}

Embedding Session::encode_audio_memory(
    const float *samples,
    size_t count,
    int sample_rate,
    const EncodeOptions &opt) {

    if (!_has_audio) {
        LUISA_ERROR("Session::encode_audio_memory: Audio Encoder is disabled (Selective Encoder Loading)");
    }
    auto processed = AudioProcessor::process_memory(samples, count, static_cast<uint32_t>(sample_rate), _package.get());

    uint32_t num_soft = processed.num_soft_tokens;
    _host_ids.clear();
    _host_ids.push_back(_tokenizer->bos_id()); // 2 (<bos>)
    _host_ids.push_back(256000u);              // <|audio>
    for (uint32_t i = 0; i < num_soft; i++) {
        _host_ids.push_back(258881u);          // <|audio|>
    }
    _host_ids.push_back(258883u);              // <audio|>
    _host_ids.push_back(1u);                   // <eos>

    uint32_t S = static_cast<uint32_t>(_host_ids.size());
    uint32_t target_dim = opt.target_dimension;
    _host_mask.assign(S, 1.0f);

    size_t feat_elems = processed.features.size();
    size_t soft_elems = num_soft * 512u;

    if (!_dev_audio_features || _dev_audio_features.size() < feat_elems) {
        _dev_audio_features = _device->create_buffer<float>(feat_elems);
    }
    if (!_dev_audio_soft_tokens || _dev_audio_soft_tokens.size() < soft_elems) {
        _dev_audio_soft_tokens = _device->create_buffer<float>(soft_elems);
    }
    if (!_dev_ids || _dev_ids.size() < S) {
        _dev_ids = _device->create_buffer<uint32_t>(S);
        _dev_mask = _device->create_buffer<float>(S);
    }
    if (!_dev_out || _dev_out.size() < target_dim) {
        _dev_out = _device->create_buffer<float>(target_dim);
    }
    _host_out.resize(target_dim);

    auto dev_feat_view = _dev_audio_features.view(0, feat_elems);
    auto dev_soft_view = _dev_audio_soft_tokens.view(0, soft_elems);
    auto dev_ids_view = _dev_ids.view(0, S);
    auto dev_mask_view = _dev_mask.view(0, S);
    auto dev_out_view = _dev_out.view(0, target_dim);

    compute::CommandList cmd_list;
    cmd_list << dev_feat_view.copy_from(processed.features.data());

    _executor->record_audio_encode(
        cmd_list, dev_feat_view, dev_soft_view,
        processed.num_frames, processed.num_features, num_soft);

    cmd_list << dev_ids_view.copy_from(_host_ids.data())
             << dev_mask_view.copy_from(_host_mask.data());

    MultimodalInputs mm_inputs{};
    mm_inputs.audio_embeds = dev_soft_view;
    mm_inputs.audio_start_idx = 2u;
    mm_inputs.num_audio_tokens = num_soft;

    _executor->record_encode(cmd_list, dev_ids_view, dev_mask_view, dev_out_view, 1u, S, target_dim, mm_inputs);
    cmd_list << dev_out_view.copy_to(_host_out.data());

    _stream << cmd_list.commit() << compute::synchronize();

    Embedding emb{};
    emb.dimension = target_dim;
    emb.values = std::move(_host_out);
    return emb;
}

Embedding Session::encode_multimodal_audio(
    const std::filesystem::path &audio_path,
    luisa::string_view prompt_text,
    const EncodeOptions &opt) {

    if (!_has_audio) {
        LUISA_ERROR("Session::encode_multimodal_audio: Audio Encoder is disabled (Selective Encoder Loading)");
    }
    auto processed = AudioProcessor::process_file(audio_path, _package.get());
    uint32_t num_soft = processed.num_soft_tokens;

    uint32_t audio_start_idx = 2u;
    _host_ids.clear();

    auto audio_tag_pos = prompt_text.find("<|audio|>");
    if (audio_tag_pos != luisa::string_view::npos) {
        auto prefix = prompt_text.substr(0, audio_tag_pos);
        auto suffix = prompt_text.substr(audio_tag_pos + 9);
        auto prefix_toks = _tokenizer->encode(prefix, opt.max_length);
        auto suffix_toks = !suffix.empty() ? _tokenizer->encode(suffix, opt.max_length) : luisa::vector<uint32_t>{};

        size_t pref_end = prefix_toks.size();
        if (pref_end > 0 && prefix_toks.back() == _tokenizer->eos_id()) {
            pref_end--;
        }
        for (size_t i = 0; i < pref_end; i++) {
            _host_ids.push_back(prefix_toks[i]);
        }
        _host_ids.push_back(256000u); // <|audio>
        audio_start_idx = static_cast<uint32_t>(_host_ids.size());
        for (uint32_t i = 0; i < num_soft; i++) {
            _host_ids.push_back(258881u); // <|audio|>
        }
        _host_ids.push_back(258883u); // <audio|>

        size_t suf_start = (!suffix_toks.empty() && suffix_toks.front() == _tokenizer->bos_id()) ? 1u : 0u;
        for (size_t i = suf_start; i < suffix_toks.size(); i++) {
            _host_ids.push_back(suffix_toks[i]);
        }
        if (_host_ids.empty() || _host_ids.back() != _tokenizer->eos_id()) {
            _host_ids.push_back(_tokenizer->eos_id()); // <eos>
        }
    } else {
        auto text_toks = _tokenizer->encode(prompt_text, opt.max_length);
        _host_ids.push_back(_tokenizer->bos_id()); // 2 (<bos>)
        _host_ids.push_back(256000u);              // <|audio>
        audio_start_idx = static_cast<uint32_t>(_host_ids.size());
        for (uint32_t i = 0; i < num_soft; i++) {
            _host_ids.push_back(258881u);          // <|audio|>
        }
        _host_ids.push_back(258883u);              // <audio|>

        size_t text_start = (!text_toks.empty() && text_toks.front() == _tokenizer->bos_id()) ? 1u : 0u;
        for (size_t i = text_start; i < text_toks.size(); i++) {
            _host_ids.push_back(text_toks[i]);
        }
        if (_host_ids.empty() || _host_ids.back() != 1u) {
            _host_ids.push_back(1u); // <eos>
        }
    }

    uint32_t S = static_cast<uint32_t>(_host_ids.size());
    uint32_t target_dim = opt.target_dimension;
    _host_mask.assign(S, 1.0f);

    size_t feat_elems = processed.features.size();
    size_t soft_elems = num_soft * 512u;

    if (!_dev_audio_features || _dev_audio_features.size() < feat_elems) {
        _dev_audio_features = _device->create_buffer<float>(feat_elems);
    }
    if (!_dev_audio_soft_tokens || _dev_audio_soft_tokens.size() < soft_elems) {
        _dev_audio_soft_tokens = _device->create_buffer<float>(soft_elems);
    }
    if (!_dev_ids || _dev_ids.size() < S) {
        _dev_ids = _device->create_buffer<uint32_t>(S);
        _dev_mask = _device->create_buffer<float>(S);
    }
    if (!_dev_out || _dev_out.size() < target_dim) {
        _dev_out = _device->create_buffer<float>(target_dim);
    }
    _host_out.resize(target_dim);

    auto dev_feat_view = _dev_audio_features.view(0, feat_elems);
    auto dev_soft_view = _dev_audio_soft_tokens.view(0, soft_elems);
    auto dev_ids_view = _dev_ids.view(0, S);
    auto dev_mask_view = _dev_mask.view(0, S);
    auto dev_out_view = _dev_out.view(0, target_dim);

    compute::CommandList cmd_list;
    cmd_list << dev_feat_view.copy_from(processed.features.data());

    _executor->record_audio_encode(
        cmd_list, dev_feat_view, dev_soft_view,
        processed.num_frames, processed.num_features, num_soft);

    cmd_list << dev_ids_view.copy_from(_host_ids.data())
             << dev_mask_view.copy_from(_host_mask.data());

    MultimodalInputs mm_inputs{};
    mm_inputs.audio_embeds = dev_soft_view;
    mm_inputs.audio_start_idx = audio_start_idx;
    mm_inputs.num_audio_tokens = num_soft;

    _executor->record_encode(cmd_list, dev_ids_view, dev_mask_view, dev_out_view, 1u, S, target_dim, mm_inputs);
    cmd_list << dev_out_view.copy_to(_host_out.data());

    _stream << cmd_list.commit() << compute::synchronize();

    Embedding emb{};
    emb.dimension = target_dim;
    emb.values = std::move(_host_out);
    return emb;
}

namespace {

Embedding encode_video_internal(
    Session &session,
    compute::Device *device,
    compute::Stream &stream,
    Executor *executor,
    const Tokenizer *tokenizer,
    compute::Buffer<float> &dev_video_pixel_values,
    compute::Buffer<int32_t> &dev_video_position_ids,
    compute::Buffer<float> &dev_video_soft_tokens,
    compute::Buffer<uint32_t> &dev_video_token_indices,
    luisa::vector<uint32_t> &host_video_token_indices,
    compute::Buffer<uint32_t> &dev_ids,
    compute::Buffer<float> &dev_mask,
    compute::Buffer<float> &dev_out,
    luisa::vector<uint32_t> &host_ids,
    luisa::vector<float> &host_mask,
    luisa::vector<float> &host_out,
    const ProcessedVideo &video,
    luisa::string_view prompt_text,
    const EncodeOptions &opt) {

    uint32_t num_frames = video.num_frames;
    uint32_t num_soft_per_frame = video.num_soft_tokens_per_frame;
    uint32_t total_soft = video.total_soft_tokens;

    host_ids.clear();
    host_video_token_indices.clear();
    host_video_token_indices.reserve(total_soft);

    std::string prompt_str(prompt_text.data(), prompt_text.size());
    size_t video_token_pos = prompt_str.find("<|video|>");

    if (video_token_pos != std::string::npos) {
        std::string text_before = prompt_str.substr(0, video_token_pos);
        std::string text_after = prompt_str.substr(video_token_pos + 9u);
        auto prefix_ids = tokenizer->encode(text_before);
        auto suffix_ids = tokenizer->encode(text_after);

        host_ids.push_back(tokenizer->bos_id());
        size_t pre_start = (!prefix_ids.empty() && prefix_ids.front() == tokenizer->bos_id()) ? 1u : 0u;
        for (size_t i = pre_start; i < prefix_ids.size(); i++) {
            if (prefix_ids[i] != tokenizer->eos_id()) {
                host_ids.push_back(prefix_ids[i]);
            }
        }
        for (uint32_t f = 0u; f < num_frames; f++) {
            host_ids.push_back(255999u); // <|image>
            for (uint32_t k = 0u; k < num_soft_per_frame; k++) {
                host_ids.push_back(258884u); // <|video|>
                host_video_token_indices.push_back(static_cast<uint32_t>(host_ids.size() - 1u));
            }
            host_ids.push_back(258882u); // <image|>
        }
        size_t suf_start = (!suffix_ids.empty() && suffix_ids.front() == tokenizer->bos_id()) ? 1u : 0u;
        for (size_t i = suf_start; i < suffix_ids.size(); i++) {
            if (suffix_ids[i] != tokenizer->eos_id()) {
                host_ids.push_back(suffix_ids[i]);
            }
        }
        host_ids.push_back(tokenizer->eos_id());
    } else if (!prompt_text.empty()) {
        host_ids.push_back(tokenizer->bos_id());
        auto txt_toks = tokenizer->encode(prompt_text, opt.max_length);
        size_t t_start = (!txt_toks.empty() && txt_toks.front() == tokenizer->bos_id()) ? 1u : 0u;
        for (size_t i = t_start; i < txt_toks.size(); i++) {
            if (txt_toks[i] != tokenizer->eos_id()) {
                host_ids.push_back(txt_toks[i]);
            }
        }
        for (uint32_t f = 0u; f < num_frames; f++) {
            host_ids.push_back(255999u);
            for (uint32_t k = 0u; k < num_soft_per_frame; k++) {
                host_ids.push_back(258884u);
                host_video_token_indices.push_back(static_cast<uint32_t>(host_ids.size() - 1u));
            }
            host_ids.push_back(258882u);
        }
        host_ids.push_back(tokenizer->eos_id());
    } else {
        host_ids.push_back(tokenizer->bos_id());
        for (uint32_t f = 0u; f < num_frames; f++) {
            host_ids.push_back(255999u);
            for (uint32_t k = 0u; k < num_soft_per_frame; k++) {
                host_ids.push_back(258884u);
                host_video_token_indices.push_back(static_cast<uint32_t>(host_ids.size() - 1u));
            }
            host_ids.push_back(258882u);
        }
        host_ids.push_back(tokenizer->eos_id());
    }

    uint32_t S = static_cast<uint32_t>(host_ids.size());
    uint32_t target_dim = opt.target_dimension;
    host_mask.assign(S, 1.0f);

    size_t frame_px_elems = video.max_patches * 768u;
    size_t frame_pos_elems = video.max_patches * 2u;
    size_t total_soft_elems = total_soft * 512u;

    if (!dev_video_pixel_values || dev_video_pixel_values.size() < frame_px_elems) {
        dev_video_pixel_values = device->create_buffer<float>(frame_px_elems);
    }
    if (!dev_video_position_ids || dev_video_position_ids.size() < frame_pos_elems) {
        dev_video_position_ids = device->create_buffer<int32_t>(frame_pos_elems);
    }
    if (!dev_video_soft_tokens || dev_video_soft_tokens.size() < total_soft_elems) {
        dev_video_soft_tokens = device->create_buffer<float>(total_soft_elems);
    }
    if (!dev_video_token_indices || dev_video_token_indices.size() < total_soft) {
        dev_video_token_indices = device->create_buffer<uint32_t>(total_soft);
    }
    if (!dev_ids || dev_ids.size() < S) {
        dev_ids = device->create_buffer<uint32_t>(S);
        dev_mask = device->create_buffer<float>(S);
    }
    if (!dev_out || dev_out.size() < target_dim) {
        dev_out = device->create_buffer<float>(target_dim);
    }
    host_out.resize(target_dim);

    auto dev_pos_view = dev_video_position_ids.view(0, frame_pos_elems);
    auto dev_indices_view = dev_video_token_indices.view(0, total_soft);
    auto dev_ids_view = dev_ids.view(0, S);
    auto dev_mask_view = dev_mask.view(0, S);
    auto dev_out_view = dev_out.view(0, target_dim);

    compute::CommandList cmd_list;
    cmd_list << dev_pos_view.copy_from(video.frames[0].position_ids.data())
             << dev_indices_view.copy_from(host_video_token_indices.data());

    for (uint32_t f = 0u; f < num_frames; f++) {
        auto px_view = dev_video_pixel_values.view(0, frame_px_elems);
        auto soft_view = dev_video_soft_tokens.view(f * num_soft_per_frame * 512u, num_soft_per_frame * 512u);
        cmd_list << px_view.copy_from(video.frames[f].pixel_values.data());
        executor->record_vision_encode(
            cmd_list, px_view, dev_pos_view, soft_view,
            1u, video.num_patches_w, video.num_patches_h, video.max_patches, num_soft_per_frame);
    }

    cmd_list << dev_ids_view.copy_from(host_ids.data())
             << dev_mask_view.copy_from(host_mask.data());

    MultimodalInputs mm_inputs{};
    mm_inputs.vision_embeds = dev_video_soft_tokens.view(0, total_soft_elems);
    mm_inputs.num_vision_tokens = total_soft;
    mm_inputs.vision_token_indices = dev_indices_view;

    executor->record_encode(cmd_list, dev_ids_view, dev_mask_view, dev_out_view, 1u, S, target_dim, mm_inputs);
    cmd_list << dev_out_view.copy_to(host_out.data());

    stream << cmd_list.commit() << compute::synchronize();

    Embedding emb{};
    emb.dimension = target_dim;
    emb.values = std::move(host_out);
    return emb;
}

}// namespace

Embedding Session::encode_video(
    const std::filesystem::path &video_path,
    const VideoConfig &cfg,
    const EncodeOptions &opt) {

    if (!_has_vision) {
        LUISA_ERROR("Session::encode_video: Vision Encoder is disabled (Selective Encoder Loading)");
    }
    auto processed = VideoProcessor::process_file(video_path, cfg);
    return encode_video_internal(
        *this, _device, _stream, _executor.get(), _tokenizer.get(),
        _dev_video_pixel_values, _dev_video_position_ids, _dev_video_soft_tokens, _dev_video_token_indices,
        _host_video_token_indices, _dev_ids, _dev_mask, _dev_out, _host_ids, _host_mask, _host_out,
        processed, "", opt);
}

Embedding Session::encode_video(
    luisa::span<const std::filesystem::path> frame_paths,
    const VideoConfig &cfg,
    const EncodeOptions &opt) {

    if (!_has_vision) {
        LUISA_ERROR("Session::encode_video: Vision Encoder is disabled (Selective Encoder Loading)");
    }
    auto processed = VideoProcessor::process_frames(frame_paths, cfg);
    return encode_video_internal(
        *this, _device, _stream, _executor.get(), _tokenizer.get(),
        _dev_video_pixel_values, _dev_video_position_ids, _dev_video_soft_tokens, _dev_video_token_indices,
        _host_video_token_indices, _dev_ids, _dev_mask, _dev_out, _host_ids, _host_mask, _host_out,
        processed, "", opt);
}

Embedding Session::encode_video_memory(
    luisa::span<const RawVideoFrame> frames,
    const VideoConfig &cfg,
    const EncodeOptions &opt) {

    if (!_has_vision) {
        LUISA_ERROR("Session::encode_video_memory: Vision Encoder is disabled (Selective Encoder Loading)");
    }
    auto processed = VideoProcessor::process_memory(frames, cfg);
    return encode_video_internal(
        *this, _device, _stream, _executor.get(), _tokenizer.get(),
        _dev_video_pixel_values, _dev_video_position_ids, _dev_video_soft_tokens, _dev_video_token_indices,
        _host_video_token_indices, _dev_ids, _dev_mask, _dev_out, _host_ids, _host_mask, _host_out,
        processed, "", opt);
}

Embedding Session::encode_multimodal_video(
    const std::filesystem::path &video_path,
    luisa::string_view prompt_text,
    const VideoConfig &cfg,
    const EncodeOptions &opt) {

    if (!_has_vision) {
        LUISA_ERROR("Session::encode_multimodal_video: Vision Encoder is disabled (Selective Encoder Loading)");
    }
    auto processed = VideoProcessor::process_file(video_path, cfg);
    return encode_video_internal(
        *this, _device, _stream, _executor.get(), _tokenizer.get(),
        _dev_video_pixel_values, _dev_video_position_ids, _dev_video_soft_tokens, _dev_video_token_indices,
        _host_video_token_indices, _dev_ids, _dev_mask, _dev_out, _host_ids, _host_mask, _host_out,
        processed, prompt_text, opt);
}

Embedding Session::encode_multimodal_video(
    luisa::span<const std::filesystem::path> frame_paths,
    luisa::string_view prompt_text,
    const VideoConfig &cfg,
    const EncodeOptions &opt) {

    if (!_has_vision) {
        LUISA_ERROR("Session::encode_multimodal_video: Vision Encoder is disabled (Selective Encoder Loading)");
    }
    auto processed = VideoProcessor::process_frames(frame_paths, cfg);
    return encode_video_internal(
        *this, _device, _stream, _executor.get(), _tokenizer.get(),
        _dev_video_pixel_values, _dev_video_position_ids, _dev_video_soft_tokens, _dev_video_token_indices,
        _host_video_token_indices, _dev_ids, _dev_mask, _dev_out, _host_ids, _host_mask, _host_out,
        processed, prompt_text, opt);
}

}// namespace luisa::embed_mm
