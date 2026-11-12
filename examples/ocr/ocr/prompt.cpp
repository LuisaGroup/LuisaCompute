// OCR engine -- recipe-ID-specific prompt builders (Phase 1).
// No general Jinja interpreter; unknown template IDs fail closed.
#include "ocr/prompt.h"

#include <utility>

namespace luisa::ocr {
namespace {

// The Phase 1 contract template set. Each builder is a fixed, explicit layout
// (no template engine). Official per-checkpoint templates are pinned in later
// phases; these ids are the stable registry the session/CLI select against.
constexpr luisa::string_view kTemplates[] = {
    "qwen3_vl_ocr",
    "fire_red_ocr",
    "glm_ocr",
    "got_ocr2",
    "pp_ocrv6_plain",
    "generic_ocr",
};

[[nodiscard]] bool is_known_template(luisa::string_view id) noexcept {
    for (auto t : kTemplates) {
        if (t == id) { return true; }
    }
    return false;
}

[[nodiscard]] Status validate_common(const PromptRequest &request) {
    if (request.template_id.empty()) {
        return Status::error(StatusCode::invalid_argument, "prompt template_id must not be empty", "template_id");
    }
    if (!is_known_template(request.template_id)) {
        return Status::error(StatusCode::unsupported,
                             luisa::string{"unknown prompt template '"} + request.template_id + "' (no general Jinja fallback)", "template_id");
    }
    if (request.mode == RecognizeMode::extract && request.schema.empty()) {
        return Status::error(StatusCode::invalid_argument, "extract mode requires a JSON schema", "schema");
    }
    return Status::success();
}

// The OCR instruction per mode (fixed contract text).
[[nodiscard]] luisa::string mode_instruction(const PromptRequest &request) {
    switch (request.mode) {
        case RecognizeMode::plain:
            return "Read all the text in the image and output it as plain text.";
        case RecognizeMode::spotting:
            return "Read all the text in the image and output each line with its bounding box.";
        case RecognizeMode::markdown:
            return "Convert the document in the image to Markdown, preserving structure.";
        case RecognizeMode::extract:
            return "Extract information from the image as JSON conforming to this schema:\n" + request.schema;
        default:
            return {};
    }
}

}// namespace

[[nodiscard]] luisa::span<const luisa::string_view> known_prompt_template_ids() noexcept {
    return kTemplates;
}

[[nodiscard]] Result<luisa::string> build_prompt(const PromptRequest &request) {
    using R = Result<luisa::string>;
    auto common = validate_common(request);
    if (!common.ok()) { return R::failure(std::move(common)); }

    const auto &id = request.template_id;
    luisa::string image_part;
    for (uint32_t i = 0; i < request.num_images; i++) {
        image_part += request.image_token.empty() ? luisa::string{"<image>"} : request.image_token;
        image_part += "\n";
    }
    const auto instruction = request.user_text.empty() ? mode_instruction(request) : request.user_text;

    if (id == "qwen3_vl_ocr") {
        // Qwen3-VL chat layout: system + user (image + instruction) + assistant
        // generation prompt. Control tokens are literal text for the tokenizer.
        luisa::string out;
        out += "<|im_start|>system\nYou are a helpful OCR assistant.<|im_end|>\n";
        out += "<|im_start|>user\n";
        out += image_part;
        out += instruction;
        out += "<|im_end|>\n";
        out += "<|im_start|>assistant\n";
        return R::success(std::move(out));
    }
    if (id == "fire_red_ocr") {
        // FireRed-OCR layout (SYNTHETIC contract; the official template is an
        // unmet gate): system + user (image + instruction) + assistant, with
        // a document-conversion system instruction. Distinct from the
        // Qwen3-VL template so the two recipes never share a prompt by
        // accident.
        luisa::string out;
        out += "<|im_start|>system\nYou are a document OCR engine. Output structurally valid Markdown.<|im_end|>\n";
        out += "<|im_start|>user\n";
        out += image_part;
        out += instruction;
        out += "<|im_end|>\n";
        out += "<|im_start|>assistant\n";
        return R::success(std::move(out));
    }
    if (id == "glm_ocr") {
        // GLM layout: [gMASK]<sop> then user/assistant turns.
        luisa::string out;
        out += "[gMASK]<sop>";
        out += "<|user|>\n";
        out += image_part;
        out += instruction;
        out += "<|assistant|>\n";
        return R::success(std::move(out));
    }
    if (id == "got_ocr2") {
        // GOT-OCR2: a plain instruction prompt with the image placeholder.
        luisa::string out;
        out += image_part;
        out += instruction;
        out += "\n";
        return R::success(std::move(out));
    }
    if (id == "pp_ocrv6_plain") {
        // Conventional CNN OCR has no chat prompt; the recognizer consumes the
        // cropped line image directly. An empty prompt is the contract.
        if (request.mode != RecognizeMode::plain) {
            return R::failure(StatusCode::unsupported, "pp_ocrv6_plain supports only plain mode", "mode");
        }
        return R::success(luisa::string{});
    }
    if (id == "generic_ocr") {
        luisa::string out;
        out += image_part;
        out += instruction;
        return R::success(std::move(out));
    }
    return R::failure(StatusCode::unsupported,
                      luisa::string{"unknown prompt template '"} + id + "'", "template_id");
}

}// namespace luisa::ocr