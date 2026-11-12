// OCR engine -- FireRed-OCR recipe contract (Phase 4). Host-only.
// Synthetic recipe-driven path over the shared Qwen3-VL graph; real
// artifacts fail closed. See fire_red_ocr.h.
#include "ocr/models/fire_red_ocr.h"

#include <luisa/core/stl/format.h>

namespace luisa::ocr {

[[nodiscard]] Status FireRedOcrConfig::validate() const {
    if (auto s = graph.validate(); !s.ok()) { return s; }
    if (auto s = names.validate(); !s.ok()) { return s; }
    if (prompt_template_id != kFireRedPromptTemplateId) {
        return Status::error(StatusCode::invalid_argument,
                             "FireRed-OCR prompt template must be the registered fire_red_ocr id", "prompt_template_id");
    }
    if (output_parser_id != kFireRedOutputParserId) {
        return Status::error(StatusCode::invalid_argument,
                             "FireRed-OCR output parser must be the fire_red_markdown contract", "output_parser_id");
    }
    // S9: the FireRed tensor map must be the EXACT declared FireRed prefix
    // pair. Any Qwen3-VL map combination (model./visual., language_model./
    // visual., model./model.visual., or any other) is refused — a FireRed
    // package is never loaded through a different name contract.
    if (names.text.prefix != "language_model." || names.visual_prefix != "model.visual.") {
        return Status::error(StatusCode::invalid_argument,
                             "FireRed-OCR requires the exact declared tensor-map prefixes "
                             "(text 'language_model.', vision 'model.visual.'); any Qwen3-VL map combination is refused",
                             "names");
    }
    return Status::success();
}

[[nodiscard]] FireRedOcrConfig fire_red_ocr_synthetic_config() {
    FireRedOcrConfig c;
    c.graph = qwen3_vl_synthetic_config();
    // FireRed-specific canonical prefixes (synthetic contract): the text
    // decoder nests under language_model.*, vision under model.visual.*.
    c.names.text.prefix = "language_model.";
    c.names.visual_prefix = "model.visual.";
    c.prompt_template_id = luisa::string{kFireRedPromptTemplateId};
    c.output_parser_id = luisa::string{kFireRedOutputParserId};
    return c;
}

[[nodiscard]] Status fire_red_ocr_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                         luisa::string_view config_fingerprint) {
    // Fail closed unconditionally: no official FireRed-OCR artifact has been
    // qualified. Even a non-empty revision/fingerprint is NOT evidence of
    // qualification (the registry requires a pinned, validated recipe).
    (void)checkpoint_revision;
    (void)config_fingerprint;
    return Status::error(StatusCode::unvalidated,
                         "FireRed-OCR has no qualified official artifact: real checkpoints are refused "
                         "(synthetic recipe-driven path only; see examples/ocr/MODELS.md)",
                         "fire_red_ocr");
}

[[nodiscard]] Status fire_red_ocr_check_no_training_nodes(luisa::span<const luisa::string> component_roles) {
    for (const auto &role : component_roles) {
        // Forbidden training/GRPO roles (case-insensitive substring match).
        luisa::string lower;
        lower.reserve(role.size());
        for (char c : role) {
            lower += (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
        }
        for (const char *forbidden : {"grpo", "training", "train", "reward", "policy", "rollout", "loss"}) {
            if (lower.find(forbidden) != luisa::string::npos) {
                return Status::error(StatusCode::unsupported,
                                     luisa::format("FireRed-OCR contract forbids training/GRPO component '{}'", role),
                                     role);
            }
        }
    }
    return Status::success();
}

}// namespace luisa::ocr
