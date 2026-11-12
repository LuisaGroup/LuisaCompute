#pragma once
// OCR engine -- recipe-ID-specific prompt builders (Phase 1). Host-only.
//
// Prompts are built ONLY for explicitly registered recipe template IDs. There
// is NO general Jinja interpreter: an unknown template ID fails closed with
// StatusCode::unsupported rather than silently approximating a chat template.
// Each builder reproduces the exact control-token layout of its recipe's
// official apply_chat_template for the OCR task; the frozen fixtures pin the
// expected strings.
#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/model_recipe.h"
#include "ocr/status.h"

namespace luisa::ocr {

// One chat message. Only the user turn carries the OCR instruction/image
// placeholder; the builders place the recipe's control tokens around it.
struct PromptMessage {
    luisa::string role;// "system" | "user" | "assistant"
    luisa::string content;
};

struct PromptRequest {
    luisa::string template_id;// recipe prompt template id (required)
    RecognizeMode mode{RecognizeMode::plain};
    luisa::string user_text;  // task instruction / question
    luisa::string schema;     // JSON schema text for extract
    luisa::string image_token;// recipe image placeholder (e.g. "<image>")
    uint32_t num_images{0};
};

// Registered template IDs (Phase 1 synthetic/contract set; official checkpoint
// templates are pinned per recipe in later phases). These are the ONLY ids
// build_prompt accepts.
[[nodiscard]] luisa::span<const luisa::string_view> known_prompt_template_ids() noexcept;

// Build the prompt string for a recipe template. Fails closed on an unknown
// template id, an extract request without a schema where required, or an
// unsupported role/mode combination. The result is the exact text to feed the
// tokenizer (control tokens included as literal text).
[[nodiscard]] Result<luisa::string> build_prompt(const PromptRequest &request);

}// namespace luisa::ocr