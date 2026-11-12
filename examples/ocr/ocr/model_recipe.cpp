// OCR engine -- recipe declarations and contract validation (Phase 0).
//
// Every recipe below is PLANNED: graph contracts are written down from the
// requirement (plan §3) as contracts to verify against pinned official
// checkpoints/configs/reference implementations. No official artifact has
// been inspected yet, so revisions/fingerprints/token ids stay empty and no
// dimension is claimed as verified. Nothing here downloads or loads weights.
#include "ocr/model_recipe.h"

#include <algorithm>

namespace luisa::ocr {

[[nodiscard]] luisa::string_view pp_ocr_size_name(PpOcrSize size) noexcept {
    switch (size) {
        case PpOcrSize::none: return "none";
        case PpOcrSize::tiny: return "tiny";
        case PpOcrSize::small: return "small";
        case PpOcrSize::medium: return "medium";
    }
    return "unknown";
}

[[nodiscard]] luisa::string_view model_id_name(ModelId id) noexcept {
    switch (id) {
        case ModelId::unknown: return "unknown";
        case ModelId::pp_ocrv6: return "pp_ocrv6";
        case ModelId::paddle_ocr_vl_1_6: return "paddle_ocr_vl_1_6";
        case ModelId::ovis_ocr2: return "ovis_ocr2";
        case ModelId::tele_ocr: return "tele_ocr";
        case ModelId::glm_ocr: return "glm_ocr";
        case ModelId::mineru_2_5_pro_2605_1_2b: return "mineru_2_5_pro_2605_1_2b";
        case ModelId::hunyuan_ocr_1_5: return "hunyuan_ocr_1_5";
        case ModelId::surya_2_5: return "surya_2_5";
        case ModelId::open_doc_0_1b: return "open_doc_0_1b";
        case ModelId::fire_red_ocr: return "fire_red_ocr";
        case ModelId::qwen3_vl_2b_instruct: return "qwen3_vl_2b_instruct";
        case ModelId::deepseek_ocr_2: return "deepseek_ocr_2";
        case ModelId::dots_mocr: return "dots_mocr";
        case ModelId::got_ocr2_0: return "got_ocr2_0";
        case ModelId::olm_ocr_2: return "olm_ocr_2";
    }
    return "unknown";
}

[[nodiscard]] ModelId model_id_from_name(luisa::string_view name) noexcept {
    constexpr ModelId ids[] = {
        ModelId::pp_ocrv6, ModelId::paddle_ocr_vl_1_6, ModelId::ovis_ocr2,
        ModelId::tele_ocr, ModelId::glm_ocr, ModelId::mineru_2_5_pro_2605_1_2b,
        ModelId::hunyuan_ocr_1_5, ModelId::surya_2_5, ModelId::open_doc_0_1b,
        ModelId::fire_red_ocr, ModelId::qwen3_vl_2b_instruct, ModelId::deepseek_ocr_2,
        ModelId::dots_mocr, ModelId::got_ocr2_0, ModelId::olm_ocr_2,
    };
    for (auto id : ids) {
        if (model_id_name(id) == name) { return id; }
    }
    return ModelId::unknown;
}

[[nodiscard]] luisa::string_view architecture_family_name(ArchitectureFamily family) noexcept {
    switch (family) {
        case ArchitectureFamily::unknown: return "unknown";
        case ArchitectureFamily::cnn: return "cnn";
        case ArchitectureFamily::dense_vlm: return "dense_vlm";
        case ArchitectureFamily::layout_vlm: return "layout_vlm";
        case ArchitectureFamily::hybrid_gdn: return "hybrid_gdn";
        case ArchitectureFamily::custom_vlm: return "custom_vlm";
        case ArchitectureFamily::moe_vlm: return "moe_vlm";
    }
    return "unknown";
}

[[nodiscard]] luisa::string_view recognize_mode_name(RecognizeMode mode) noexcept {
    switch (mode) {
        case RecognizeMode::plain: return "plain";
        case RecognizeMode::spotting: return "spotting";
        case RecognizeMode::markdown: return "markdown";
        case RecognizeMode::extract: return "extract";
    }
    return "unknown";
}

[[nodiscard]] luisa::optional<RecognizeMode> recognize_mode_from_name(luisa::string_view name) noexcept {
    constexpr RecognizeMode modes[] = {
        RecognizeMode::plain, RecognizeMode::spotting,
        RecognizeMode::markdown, RecognizeMode::extract,
    };
    for (auto mode : modes) {
        if (recognize_mode_name(mode) == name) { return mode; }
    }
    return luisa::nullopt;
}

[[nodiscard]] luisa::string_view precision_name(Precision precision) noexcept {
    switch (precision) {
        case Precision::fp32: return "fp32";
        case Precision::fp16: return "fp16";
        case Precision::bf16: return "bf16";
        case Precision::int8: return "int8";
        case Precision::int4: return "int4";
        case Precision::automatic: return "automatic";
    }
    return "unknown";
}

[[nodiscard]] luisa::optional<Precision> precision_from_name(luisa::string_view name) noexcept {
    constexpr Precision precisions[] = {
        Precision::fp32, Precision::fp16, Precision::bf16,
        Precision::int8, Precision::int4, Precision::automatic,
    };
    for (auto precision : precisions) {
        if (precision_name(precision) == name) { return precision; }
    }
    return luisa::nullopt;
}

[[nodiscard]] Precision resolve_precision(Precision requested, Precision recipe_default) noexcept {
    // A concrete request passes through; `automatic` (or any unrecognized
    // sentinel) resolves to the recipe default. The default itself must be
    // concrete; if a caller mistakenly passes `automatic` as the default we
    // fall back to the portable FP32 baseline rather than propagating it.
    if (requested != Precision::automatic && precision_name(requested) != "unknown") {
        return requested;
    }
    if (recipe_default != Precision::automatic && precision_name(recipe_default) != "unknown") {
        return recipe_default;
    }
    return Precision::fp32;
}

[[nodiscard]] luisa::string_view support_state_name(SupportState state) noexcept {
    switch (state) {
        case SupportState::planned: return "planned";
        case SupportState::load_only: return "load_only";
        case SupportState::experimental: return "experimental";
        case SupportState::validated: return "validated";
    }
    return "unknown";
}

[[nodiscard]] luisa::string_view attention_mask_kind_name(AttentionMaskKind kind) noexcept {
    switch (kind) {
        case AttentionMaskKind::unspecified: return "unspecified";
        case AttentionMaskKind::bidirectional: return "bidirectional";
        case AttentionMaskKind::causal: return "causal";
        case AttentionMaskKind::windowed: return "windowed";
        case AttentionMaskKind::prefix_causal: return "prefix_causal";
        case AttentionMaskKind::custom_block_mask: return "custom_block_mask";
    }
    return "unknown";
}

[[nodiscard]] luisa::string_view positional_encoding_kind_name(PositionalEncodingKind kind) noexcept {
    switch (kind) {
        case PositionalEncodingKind::unspecified: return "unspecified";
        case PositionalEncodingKind::none: return "none";
        case PositionalEncodingKind::rope_1d: return "rope_1d";
        case PositionalEncodingKind::rope_2d: return "rope_2d";
        case PositionalEncodingKind::mrope: return "mrope";
        case PositionalEncodingKind::interleaved_mrope: return "interleaved_mrope";
        case PositionalEncodingKind::xd_rope: return "xd_rope";
        case PositionalEncodingKind::learned_absolute: return "learned_absolute";
        case PositionalEncodingKind::sinusoidal_absolute: return "sinusoidal_absolute";
        case PositionalEncodingKind::relative_bias: return "relative_bias";
    }
    return "unknown";
}

[[nodiscard]] luisa::string_view cache_layout_kind_name(CacheLayoutKind kind) noexcept {
    switch (kind) {
        case CacheLayoutKind::unspecified: return "unspecified";
        case CacheLayoutKind::none: return "none";
        case CacheLayoutKind::contiguous_kv: return "contiguous_kv";
        case CacheLayoutKind::paged_kv: return "paged_kv";
        case CacheLayoutKind::gdn_state: return "gdn_state";
    }
    return "unknown";
}

[[nodiscard]] luisa::string ModelRecipe::target_name() const {
    luisa::string name{model_id_name(id)};
    if (id == ModelId::pp_ocrv6 && pp_size != PpOcrSize::none) {
        name += ":";
        name += pp_ocr_size_name(pp_size);
    }
    return name;
}

[[nodiscard]] Status validate_recipe(const ModelRecipe &recipe) {
    if (recipe.id == ModelId::unknown) {
        return Status::error(StatusCode::invalid_argument, "recipe has no model id", "id");
    }
    if (recipe.family == ArchitectureFamily::unknown) {
        return Status::error(StatusCode::invalid_argument, "recipe has no architecture family", "family");
    }
    if (recipe.id == ModelId::pp_ocrv6) {
        if (recipe.pp_size == PpOcrSize::none) {
            return Status::error(StatusCode::invalid_argument, "pp_ocrv6 recipe requires an explicit size", "pp_size");
        }
        if (recipe.family != ArchitectureFamily::cnn) {
            return Status::error(StatusCode::invalid_argument, "pp_ocrv6 must be a cnn recipe", "family");
        }
    } else if (recipe.pp_size != PpOcrSize::none) {
        return Status::error(StatusCode::invalid_argument, "pp_size is only valid for pp_ocrv6", "pp_size");
    }
    if (recipe.identity.recipe_version.empty()) {
        return Status::error(StatusCode::invalid_argument, "recipe_version must be set", "identity.recipe_version");
    }
    if (recipe.capabilities.modes.empty()) {
        return Status::error(StatusCode::invalid_argument, "recipe declares no modes", "capabilities.modes");
    }
    for (size_t i = 0; i < recipe.capabilities.modes.size(); i++) {
        const auto mode = recipe.capabilities.modes[i];
        if (recognize_mode_name(mode) == "unknown") {
            return Status::error(StatusCode::invalid_argument, "recipe declares an unknown mode", "capabilities.modes");
        }
        for (size_t j = 0; j < i; j++) {
            if (recipe.capabilities.modes[j] == mode) {
                return Status::error(StatusCode::invalid_argument, "recipe declares a duplicate mode", "capabilities.modes");
            }
        }
    }
    const auto has_mode = [&recipe](RecognizeMode mode) {
        return std::find(recipe.capabilities.modes.begin(), recipe.capabilities.modes.end(), mode) !=
               recipe.capabilities.modes.end();
    };
    if (has_mode(RecognizeMode::spotting) && !recipe.capabilities.has_source_coordinates) {
        return Status::error(StatusCode::invalid_argument,
                             "spotting mode requires declared source coordinates", "capabilities.has_source_coordinates");
    }
    if (recipe.cache_layout == CacheLayoutKind::unspecified) {
        return Status::error(StatusCode::invalid_argument, "recipe must declare a cache layout", "cache_layout");
    }
    // Family/cache consistency: CNN pipelines have no decode cache; hybrid GDN
    // models carry recurrent/conv state, never a KV-only layout.
    if (recipe.family == ArchitectureFamily::cnn && recipe.cache_layout != CacheLayoutKind::none) {
        return Status::error(StatusCode::invalid_argument, "cnn recipes must use cache layout 'none'", "cache_layout");
    }
    if (recipe.family == ArchitectureFamily::hybrid_gdn && recipe.cache_layout != CacheLayoutKind::gdn_state) {
        return Status::error(StatusCode::invalid_argument, "hybrid_gdn recipes must use cache layout 'gdn_state'", "cache_layout");
    }
    if (recipe.components.empty()) {
        return Status::error(StatusCode::invalid_argument, "recipe declares no components", "components");
    }
    for (const auto &component : recipe.components) {
        if (component.role.empty() || component.kind.empty()) {
            return Status::error(StatusCode::invalid_argument, "component role/kind must be set", "components");
        }
    }
    if (recipe.prompt.output_parser_id.empty()) {
        return Status::error(StatusCode::invalid_argument, "recipe must declare an output parser", "prompt.output_parser_id");
    }
    // Loadable/validated recipes must pin identity and declare their tensor map.
    if (recipe.support == SupportState::load_only || recipe.support == SupportState::validated) {
        if (!recipe.identity_pinned()) {
            return Status::error(StatusCode::invalid_argument,
                                 "load_only/validated recipes must pin checkpoint revision and config fingerprint", "identity");
        }
        if (recipe.identity.package_format_version.empty()) {
            return Status::error(StatusCode::invalid_argument,
                                 "load_only/validated recipes must declare a package format version", "identity.package_format_version");
        }
        if (recipe.tensor_mappings.empty()) {
            return Status::error(StatusCode::invalid_argument,
                                 "load_only/validated recipes must declare tensor mappings", "tensor_mappings");
        }
        for (const auto &mapping : recipe.tensor_mappings) {
            if (mapping.name.empty()) {
                return Status::error(StatusCode::invalid_argument, "tensor mapping name must be set", "tensor_mappings");
            }
        }
    }
    return Status::success();
}

[[nodiscard]] Status validate_request_capabilities(
    const ModelRecipe &recipe, RecognizeMode mode,
    luisa::string_view language, bool has_schema_or_prompt,
    bool requests_region_guidance, bool requests_color_guidance) {
    if (recognize_mode_name(mode) == "unknown") {
        return Status::error(StatusCode::invalid_argument, "unknown recognize mode", "mode");
    }
    const auto &caps = recipe.capabilities;
    const auto has_mode = std::find(caps.modes.begin(), caps.modes.end(), mode) != caps.modes.end();
    if (!has_mode) {
        return Status::error(StatusCode::unsupported,
                             luisa::string{"model does not declare mode '"} + luisa::string{recognize_mode_name(mode)} + "'",
                             "mode");
    }
    if (mode == RecognizeMode::spotting && !caps.has_source_coordinates) {
        return Status::error(StatusCode::unsupported,
                             "spotting requires declared source coordinates for this model", "mode");
    }
    if (mode == RecognizeMode::extract && caps.extract_requires_schema && !has_schema_or_prompt) {
        return Status::error(StatusCode::invalid_argument,
                             "extract mode requires a schema or prompt for this model", "schema");
    }
    if (!language.empty() && !caps.languages.empty()) {
        // EXACT match only: "multi" is a literal declared tag, never a wildcard.
        const auto found = std::find(caps.languages.begin(), caps.languages.end(), luisa::string{language});
        if (found == caps.languages.end()) {
            return Status::error(StatusCode::unsupported,
                                 luisa::string{"model does not declare language '"} + luisa::string{language} + "'",
                                 "language");
        }
    }
    if (requests_region_guidance && !caps.supports_region_guidance) {
        return Status::error(StatusCode::unsupported,
                             "model does not declare region guidance support", "region_guidance");
    }
    if (requests_color_guidance && !caps.supports_color_guidance) {
        return Status::error(StatusCode::unsupported,
                             "model does not declare color guidance support", "color_guidance");
    }
    return Status::success();
}

}// namespace luisa::ocr
