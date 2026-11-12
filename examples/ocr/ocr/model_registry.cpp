// OCR engine -- planned recipe table and fail-closed signature matching
// (Phase 0). Every recipe is `planned`; graph contracts are transcribed from
// the requirement (plan §3) as contracts TO VERIFY against pinned official
// checkpoints, not verified facts. Revisions, fingerprints, tensor maps and
// token ids intentionally stay empty until an official artifact is inspected.
#include "ocr/model_registry.h"

#include <utility>

namespace luisa::ocr {
namespace {

// Shared Phase 0 recipe identity: format version 0, nothing pinned.
[[nodiscard]] RecipeIdentity planned_identity() {
    RecipeIdentity identity;
    identity.recipe_version = "0";
    identity.package_format_version = "0";
    return identity;
}

[[nodiscard]] ComponentContract component(luisa::string role, luisa::string kind) {
    ComponentContract c;
    c.role = std::move(role);
    c.kind = std::move(kind);
    return c;
}

[[nodiscard]] ModelRecipe base_recipe(ModelId id, ArchitectureFamily family,
                                      CacheLayoutKind cache, luisa::string parser) {
    ModelRecipe recipe;
    recipe.id = id;
    recipe.family = family;
    recipe.identity = planned_identity();
    recipe.support = SupportState::planned;
    recipe.cache_layout = cache;
    recipe.prompt.output_parser_id = std::move(parser);
    recipe.preprocessing.version = "0";
    return recipe;
}

[[nodiscard]] ModelRecipe make_pp_ocrv6(PpOcrSize size) {
    auto recipe = base_recipe(ModelId::pp_ocrv6, ArchitectureFamily::cnn,
                              CacheLayoutKind::none, "ctc");
    recipe.pp_size = size;
    recipe.preprocessing.resize_semantics = "aspect_resize_pad";
    recipe.preprocessing.color_order = "rgb";
    auto detector = component("detector", "lcnetv4_replkfpn_db");
    detector.notes = "LCNetV4 backbone + RepLKFPN + DB head; fold RepLK/Conv+BN at pack time; remove NRTR/training branches";
    auto recognizer = component("recognizer", "lcnetv4_lightsvtr_ctc");
    recognizer.notes = "LCNetV4 + LightSVTR + CTC; per-size shapes must be validated separately";
    recipe.components.emplace_back(std::move(detector));
    recipe.components.emplace_back(std::move(recognizer));
    recipe.capabilities.modes = {RecognizeMode::plain, RecognizeMode::spotting};
    recipe.capabilities.languages = {"zh", "en", "multi"};
    recipe.capabilities.has_source_coordinates = true;
    recipe.capabilities.has_calibrated_confidence = true;
    recipe.notes = "CNN pipeline: line polygons + CTC confidence; tiny/small/medium validated independently (plan §3)";
    return recipe;
}

[[nodiscard]] luisa::vector<ModelRecipe> planned_recipes() {
    luisa::vector<ModelRecipe> recipes;
    recipes.emplace_back(make_pp_ocrv6(PpOcrSize::tiny));
    recipes.emplace_back(make_pp_ocrv6(PpOcrSize::small));
    recipes.emplace_back(make_pp_ocrv6(PpOcrSize::medium));

    {
        auto recipe = base_recipe(ModelId::paddle_ocr_vl_1_6, ArchitectureFamily::layout_vlm,
                                  CacheLayoutKind::contiguous_kv, "markdown");
        recipe.preprocessing.resize_semantics = "native_resolution";
        recipe.preprocessing.color_order = "rgb";
        recipe.components.emplace_back(component("layout", "pp_doclayoutv3"));
        auto vision = component("vision", "navit");
        vision.attention_mask = AttentionMaskKind::bidirectional;
        vision.positional_encoding = PositionalEncodingKind::rope_2d;
        recipe.components.emplace_back(std::move(vision));
        recipe.components.emplace_back(component("connector", "adaptive_mlp"));
        auto decoder = component("decoder", "ernie_4_5_0_3b");
        decoder.attention_mask = AttentionMaskKind::causal;
        recipe.components.emplace_back(std::move(decoder));
        recipe.capabilities.modes = {RecognizeMode::plain, RecognizeMode::spotting, RecognizeMode::markdown, RecognizeMode::extract};
        recipe.capabilities.languages = {"zh", "en", "multi"};
        recipe.capabilities.has_source_coordinates = true;
        recipe.capabilities.supports_multi_page = true;
        recipe.notes = "Full layout->Markdown/JSON/spotting pipeline required; running only its VLM is insufficient (plan §3)";
        recipes.emplace_back(std::move(recipe));
    }
    {
        auto recipe = base_recipe(ModelId::ovis_ocr2, ArchitectureFamily::hybrid_gdn,
                                  CacheLayoutKind::gdn_state, "markdown");
        recipe.preprocessing.resize_semantics = "dynamic";
        recipe.preprocessing.color_order = "rgb";
        recipe.components.emplace_back(component("vision", "ovis_vision_merger"));
        auto decoder = component("decoder", "qwen3_5_0_8b_hybrid_gdn");
        decoder.attention_mask = AttentionMaskKind::causal;
        decoder.notes = "GDN layers + periodic full GQA layers; recurrent/conv ring state + full-attn KV";
        recipe.components.emplace_back(std::move(decoder));
        recipe.capabilities.modes = {RecognizeMode::plain, RecognizeMode::markdown};
        recipe.capabilities.languages = {"zh", "en", "multi"};
        recipe.notes = "Hybrid GDN prefill/decode; never treat as KV-only transformer (plan §3)";
        recipes.emplace_back(std::move(recipe));
    }
    {
        auto recipe = base_recipe(ModelId::tele_ocr, ArchitectureFamily::dense_vlm,
                                  CacheLayoutKind::contiguous_kv, "markdown");
        recipe.preprocessing.resize_semantics = "dynamic";
        recipe.preprocessing.color_order = "rgb";
        auto vision = component("vision", "qwen2_5_vl_derived");
        vision.attention_mask = AttentionMaskKind::windowed;
        vision.positional_encoding = PositionalEncodingKind::mrope;
        vision.notes = "Window/global attention; actual tensors/config must be inspected -- outer model_type is NOT identity";
        recipe.components.emplace_back(std::move(vision));
        recipe.components.emplace_back(component("connector", "mlp_aligner"));
        auto decoder = component("decoder", "qwen3_0_6b");
        decoder.attention_mask = AttentionMaskKind::causal;
        decoder.normalization = "rmsnorm_qk_norm";
        decoder.positional_encoding = PositionalEncodingKind::mrope;
        recipe.components.emplace_back(std::move(decoder));
        recipe.capabilities.modes = {RecognizeMode::plain, RecognizeMode::markdown};
        recipe.capabilities.languages = {"zh", "en", "multi"};
        recipe.notes = "Camera/deformed document modes; QK-Norm + M-RoPE must be verified against the real checkpoint (plan §3/§6.1)";
        recipes.emplace_back(std::move(recipe));
    }
    {
        auto recipe = base_recipe(ModelId::glm_ocr, ArchitectureFamily::layout_vlm,
                                  CacheLayoutKind::contiguous_kv, "markdown");
        recipe.preprocessing.color_order = "rgb";
        recipe.components.emplace_back(component("layout", "pp_doclayoutv3"));
        auto vision = component("vision", "cogvit_0_4b");
        vision.attention_mask = AttentionMaskKind::bidirectional;
        recipe.components.emplace_back(std::move(vision));
        recipe.components.emplace_back(component("connector", "downsampling"));
        auto decoder = component("decoder", "glm_0_5b");
        decoder.attention_mask = AttentionMaskKind::causal;
        decoder.notes = "GLM positional/layer conventions to verify; MTP is a later optional accelerator";
        recipe.components.emplace_back(std::move(decoder));
        recipe.capabilities.modes = {RecognizeMode::plain, RecognizeMode::markdown, RecognizeMode::extract};
        recipe.capabilities.languages = {"zh", "en", "multi"};
        recipe.capabilities.supports_multi_page = true;
        recipes.emplace_back(std::move(recipe));
    }
    {
        auto recipe = base_recipe(ModelId::mineru_2_5_pro_2605_1_2b, ArchitectureFamily::layout_vlm,
                                  CacheLayoutKind::contiguous_kv, "markdown");
        recipe.preprocessing.resize_semantics = "native_resolution";
        recipe.preprocessing.color_order = "rgb";
        recipe.components.emplace_back(component("layout", "mineru_page_layout"));
        auto vision = component("vision", "navit_675m");
        vision.attention_mask = AttentionMaskKind::bidirectional;
        vision.positional_encoding = PositionalEncodingKind::rope_2d;
        recipe.components.emplace_back(std::move(vision));
        recipe.components.emplace_back(component("connector", "pixel_unshuffle_2x2"));
        auto decoder = component("decoder", "qwen2_0_5b");
        decoder.attention_mask = AttentionMaskKind::causal;
        decoder.positional_encoding = PositionalEncodingKind::mrope;
        recipe.components.emplace_back(std::move(decoder));
        recipe.capabilities.modes = {RecognizeMode::plain, RecognizeMode::markdown, RecognizeMode::extract};
        recipe.capabilities.languages = {"zh", "en", "multi"};
        recipe.capabilities.supports_multi_page = true;
        recipe.notes = "Cross-page document composition, reading order, tables/formulas (plan §3)";
        recipes.emplace_back(std::move(recipe));
    }
    {
        auto recipe = base_recipe(ModelId::hunyuan_ocr_1_5, ArchitectureFamily::custom_vlm,
                                  CacheLayoutKind::contiguous_kv, "markdown");
        recipe.preprocessing.color_order = "rgb";
        auto vision = component("vision", "hunyuan_vit");
        vision.attention_mask = AttentionMaskKind::bidirectional;
        recipe.components.emplace_back(std::move(vision));
        recipe.components.emplace_back(component("connector", "adaptive_pool_mlp"));
        auto decoder = component("decoder", "hunyuan_0_5b");
        decoder.attention_mask = AttentionMaskKind::causal;
        decoder.positional_encoding = PositionalEncodingKind::xd_rope;
        decoder.notes = "XD-RoPE over text/height/width/time; DFlash is a later optional accelerator";
        recipe.components.emplace_back(std::move(decoder));
        recipe.capabilities.modes = {RecognizeMode::plain, RecognizeMode::spotting, RecognizeMode::markdown, RecognizeMode::extract};
        recipe.capabilities.languages = {"zh", "en", "multi"};
        recipe.capabilities.has_source_coordinates = true;
        recipe.notes = "Spotting/translation/extraction only as validated per mode (plan §3)";
        recipes.emplace_back(std::move(recipe));
    }
    {
        auto recipe = base_recipe(ModelId::surya_2_5, ArchitectureFamily::hybrid_gdn,
                                  CacheLayoutKind::gdn_state, "markdown");
        recipe.preprocessing.color_order = "rgb";
        recipe.components.emplace_back(component("detector", "efficientvit_segformer"));
        recipe.components.emplace_back(component("vision", "surya_vision_merger"));
        auto decoder = component("decoder", "surya_hybrid_24layer");
        decoder.num_layers = 24;// contract from the requirement; verify at load
        decoder.attention_mask = AttentionMaskKind::causal;
        decoder.notes = "18 GDN + 6 full-attention layers (contract to verify)";
        recipe.components.emplace_back(std::move(decoder));
        recipe.capabilities.modes = {RecognizeMode::plain, RecognizeMode::spotting, RecognizeMode::markdown};
        recipe.capabilities.languages = {"multi"};
        recipe.capabilities.has_source_coordinates = true;
        recipe.notes = "Polygons/blocks/confidence/HTML only where reference outputs support them (plan §3)";
        recipes.emplace_back(std::move(recipe));
    }
    {
        // OpenDoc is a two-stage layout + UniRec recognizer, NOT a generic
        // autoregressive VLM: it has no KV decode cache and uses a UniRec-
        // specific output contract while retaining layout routing.
        auto recipe = base_recipe(ModelId::open_doc_0_1b, ArchitectureFamily::layout_vlm,
                                  CacheLayoutKind::none, "unirec_document");
        recipe.preprocessing.color_order = "rgb";
        recipe.components.emplace_back(component("layout", "pp_doclayoutv2"));
        auto recognizer = component("recognizer", "unirec_0_1b");
        recognizer.notes = "Two-stage text/formula/table recognizer, NOT a generic VLM; no autoregressive KV cache";
        recipe.components.emplace_back(std::move(recognizer));
        recipe.capabilities.modes = {RecognizeMode::plain, RecognizeMode::markdown};
        recipe.capabilities.languages = {"zh", "en", "multi"};
        recipe.capabilities.has_source_coordinates = true;
        recipe.notes = "Layout-routed UniRec recognizer; output parser is UniRec-specific, not a generic KV decoder (plan §3)";
        recipes.emplace_back(std::move(recipe));
    }
    {
        auto recipe = base_recipe(ModelId::fire_red_ocr, ArchitectureFamily::dense_vlm,
                                  CacheLayoutKind::contiguous_kv, "markdown");
        recipe.preprocessing.resize_semantics = "native_resolution";
        recipe.preprocessing.color_order = "rgb";
        auto vision = component("vision", "qwen3_vl_2b_vision_merger");
        vision.attention_mask = AttentionMaskKind::bidirectional;
        vision.positional_encoding = PositionalEncodingKind::interleaved_mrope;
        vision.notes = "DeepStack injection at exact selected decoder layers";
        recipe.components.emplace_back(std::move(vision));
        auto decoder = component("decoder", "qwen3_dense");
        decoder.attention_mask = AttentionMaskKind::causal;
        decoder.normalization = "rmsnorm_qk_norm";
        decoder.positional_encoding = PositionalEncodingKind::interleaved_mrope;
        recipe.components.emplace_back(std::move(decoder));
        recipe.capabilities.modes = {RecognizeMode::plain, RecognizeMode::markdown, RecognizeMode::extract};
        recipe.capabilities.languages = {"zh", "en", "multi"};
        recipe.notes = "Structural Markdown/table/LaTeX validation; no GRPO/training operators (plan §3)";
        recipes.emplace_back(std::move(recipe));
    }
    {
        auto recipe = base_recipe(ModelId::qwen3_vl_2b_instruct, ArchitectureFamily::dense_vlm,
                                  CacheLayoutKind::contiguous_kv, "markdown");
        recipe.preprocessing.resize_semantics = "native_resolution";
        recipe.preprocessing.color_order = "rgb";
        auto vision = component("vision", "qwen3_vl_2b_vision_merger");
        vision.attention_mask = AttentionMaskKind::bidirectional;
        vision.positional_encoding = PositionalEncodingKind::interleaved_mrope;
        recipe.components.emplace_back(std::move(vision));
        auto decoder = component("decoder", "qwen3_dense");
        decoder.attention_mask = AttentionMaskKind::causal;
        decoder.normalization = "rmsnorm_qk_norm";
        decoder.positional_encoding = PositionalEncodingKind::interleaved_mrope;
        recipe.components.emplace_back(std::move(decoder));
        recipe.capabilities.modes = {RecognizeMode::plain, RecognizeMode::markdown, RecognizeMode::extract};
        recipe.capabilities.languages = {"zh", "en", "multi"};
        recipe.notes = "Instruct recipe, NOT Thinking; spotting only if grounded output is validated (plan §3)";
        recipes.emplace_back(std::move(recipe));
    }
    {
        auto recipe = base_recipe(ModelId::deepseek_ocr_2, ArchitectureFamily::moe_vlm,
                                  CacheLayoutKind::contiguous_kv, "markdown");
        recipe.preprocessing.color_order = "rgb";
        // SAM-style local perception is a windowed/local vision block, NOT the
        // custom visual/query mask. Keep it separate from the visual-LM encoder
        // and causal-flow queries that actually carry the custom block mask.
        auto perception = component("vision", "sam_local_perception");
        perception.attention_mask = AttentionMaskKind::windowed;
        perception.notes = "SAM-style local perception + convolutional downsamplers; local/windowed attention (contract to verify)";
        recipe.components.emplace_back(std::move(perception));
        auto visual_lm = component("vision", "deepseek_visual_lm_encoder");
        visual_lm.attention_mask = AttentionMaskKind::custom_block_mask;
        visual_lm.notes = "Visual LM encoder + causal-flow queries: visual tokens bidirectional, queries see all visual tokens and previous queries only (custom block mask, contract to verify)";
        recipe.components.emplace_back(std::move(visual_lm));
        auto decoder = component("decoder", "deepseek_moe");
        decoder.attention_mask = AttentionMaskKind::causal;
        decoder.num_experts = 64;// contract from the requirement; verify at load
        decoder.top_k_experts = 6;
        decoder.num_shared_experts = 2;
        decoder.notes = "FP32 router; 64 routed experts/top-6/two shared (contract to verify)";
        recipe.components.emplace_back(std::move(decoder));
        recipe.capabilities.modes = {RecognizeMode::plain, RecognizeMode::markdown};
        recipe.capabilities.languages = {"zh", "en", "multi"};
        recipe.notes = "Grounded Markdown; custom visual/query mask is mandatory, never a generic causal mask (plan §3)";
        recipes.emplace_back(std::move(recipe));
    }
    {
        auto recipe = base_recipe(ModelId::dots_mocr, ArchitectureFamily::dense_vlm,
                                  CacheLayoutKind::contiguous_kv, "markdown");
        recipe.preprocessing.color_order = "rgb";
        recipe.components.emplace_back(component("vision", "vit_patch_merger"));
        auto decoder = component("decoder", "dots_dense_28layer");
        decoder.num_layers = 28;// contract from the requirement; verify at load
        decoder.num_attention_heads = 12;
        decoder.num_kv_heads = 2;
        decoder.attention_mask = AttentionMaskKind::causal;
        decoder.normalization = "rmsnorm";
        decoder.positional_encoding = PositionalEncodingKind::rope_1d;
        decoder.notes = "GQA 12 query / 2 KV heads, RMSNorm/RoPE/SwiGLU (contract to verify)";
        recipe.components.emplace_back(std::move(decoder));
        recipe.capabilities.modes = {RecognizeMode::plain, RecognizeMode::markdown};
        recipe.capabilities.languages = {"zh", "en", "multi"};
        recipe.notes = "Quantized path required for 8 GB targets (plan §3)";
        recipes.emplace_back(std::move(recipe));
    }
    {
        auto recipe = base_recipe(ModelId::got_ocr2_0, ArchitectureFamily::dense_vlm,
                                  CacheLayoutKind::contiguous_kv, "markdown");
        recipe.preprocessing.color_order = "rgb";
        auto vision = component("vision", "vitdet_sam");
        vision.attention_mask = AttentionMaskKind::windowed;
        vision.positional_encoding = PositionalEncodingKind::relative_bias;
        vision.notes = "Windowed/global attention + relative-position handling";
        recipe.components.emplace_back(std::move(vision));
        recipe.components.emplace_back(component("connector", "linear"));
        auto decoder = component("decoder", "qwen_0_5b");
        decoder.attention_mask = AttentionMaskKind::causal;
        recipe.components.emplace_back(std::move(decoder));
        recipe.capabilities.modes = {RecognizeMode::plain, RecognizeMode::markdown};
        recipe.capabilities.languages = {"zh", "en", "multi"};
        recipe.capabilities.supports_region_guidance = true;
        recipe.capabilities.supports_color_guidance = true;
        recipe.notes = "Region/color prompt modes and multi-crop composition (plan §3)";
        recipes.emplace_back(std::move(recipe));
    }
    {
        auto recipe = base_recipe(ModelId::olm_ocr_2, ArchitectureFamily::dense_vlm,
                                  CacheLayoutKind::contiguous_kv, "markdown");
        recipe.preprocessing.color_order = "rgb";
        auto vision = component("vision", "qwen2_5_vl_7b_vision_merger");
        vision.attention_mask = AttentionMaskKind::windowed;
        vision.positional_encoding = PositionalEncodingKind::mrope;
        recipe.components.emplace_back(std::move(vision));
        auto decoder = component("decoder", "qwen2_5_vl_7b");
        decoder.attention_mask = AttentionMaskKind::causal;
        decoder.positional_encoding = PositionalEncodingKind::mrope;
        recipe.components.emplace_back(std::move(decoder));
        recipe.capabilities.modes = {RecognizeMode::plain, RecognizeMode::markdown};
        recipe.capabilities.languages = {"en"};// English-only; do not claim bilingual
        recipe.notes = "INT4 + strict memory envelope or explicit host offload; English-only (plan §3)";
        recipes.emplace_back(std::move(recipe));
    }
    return recipes;
}

}// namespace

[[nodiscard]] luisa::string ModelTarget::name() const {
    luisa::string text{model_id_name(id)};
    if (id == ModelId::pp_ocrv6 && pp_size != PpOcrSize::none) {
        text += ":";
        text += pp_ocr_size_name(pp_size);
    }
    return text;
}

[[nodiscard]] Result<ModelTarget> parse_model_target(luisa::string_view text) {
    if (text.empty()) {
        return Result<ModelTarget>::failure(StatusCode::invalid_argument, "empty model target", "model");
    }
    const auto colon = text.find(':');
    const auto id_name = colon == luisa::string_view::npos ? text : text.substr(0, colon);
    const auto id = model_id_from_name(id_name);
    if (id == ModelId::unknown) {
        return Result<ModelTarget>::failure(StatusCode::invalid_argument,
                                            luisa::string{"unknown model '"} + luisa::string{id_name} + "'", "model");
    }
    ModelTarget target;
    target.id = id;
    if (id == ModelId::pp_ocrv6) {
        if (colon == luisa::string_view::npos) {
            return Result<ModelTarget>::failure(StatusCode::invalid_argument,
                                                "pp_ocrv6 requires an explicit size: pp_ocrv6:tiny|small|medium", "model");
        }
        const auto size_name = text.substr(colon + 1);
        if (size_name == "tiny") {
            target.pp_size = PpOcrSize::tiny;
        } else if (size_name == "small") {
            target.pp_size = PpOcrSize::small;
        } else if (size_name == "medium") {
            target.pp_size = PpOcrSize::medium;
        } else {
            return Result<ModelTarget>::failure(StatusCode::invalid_argument,
                                                luisa::string{"unknown pp_ocrv6 size '"} + luisa::string{size_name} + "'", "model");
        }
    } else if (colon != luisa::string_view::npos) {
        return Result<ModelTarget>::failure(StatusCode::invalid_argument,
                                            "only pp_ocrv6 accepts a ':size' suffix", "model");
    }
    return Result<ModelTarget>::success(target);
}

ModelRegistry::ModelRegistry()
    : _recipes{planned_recipes()} {}

ModelRegistry::ModelRegistry(luisa::vector<ModelRecipe> recipes)
    : _recipes{std::move(recipes)} {}

namespace {

// Index of the recipe for a target within a table (targets are unique).
[[nodiscard]] size_t recipe_index_of(const luisa::vector<ModelRecipe> &recipes,
                                     const ModelTarget &target) noexcept {
    for (size_t i = 0; i < recipes.size(); i++) {
        if (recipes[i].id == target.id && recipes[i].pp_size == target.pp_size) {
            return i;
        }
    }
    return recipes.size();
}

// Exact canonical tensor-signature comparison. The recipe's tensor map defines
// the known names; the artifact must contain every required name, no duplicate
// names, and no name outside the map. Optional mapped names may be absent.
// Order is irrelevant. Returns a failure Status describing the first mismatch.
[[nodiscard]] Status compare_tensor_signature(const ModelRecipe &recipe,
                                              const ArtifactSignature &signature) {
    if (recipe.tensor_mappings.empty()) {
        return Status::error(StatusCode::unvalidated,
                             "recipe declares no tensor signature to validate against", "tensor_mappings");
    }
    // Detect duplicate artifact names and build the observed set.
    luisa::vector<luisa::string> observed;
    observed.reserve(signature.tensor_names.size());
    for (const auto &name : signature.tensor_names) {
        if (name.empty()) {
            return Status::error(StatusCode::unvalidated, "artifact tensor name is empty", "tensor_names");
        }
        for (const auto &seen : observed) {
            if (seen == name) {
                return Status::error(StatusCode::unvalidated,
                                     luisa::string{"duplicate artifact tensor name '"} + name + "'", "tensor_names");
            }
        }
        observed.emplace_back(name);
    }
    const auto is_known = [&recipe](const luisa::string &name) {
        for (const auto &mapping : recipe.tensor_mappings) {
            if (mapping.name == name) { return true; }
        }
        return false;
    };
    const auto is_observed = [&observed](const luisa::string &name) {
        for (const auto &seen : observed) {
            if (seen == name) { return true; }
        }
        return false;
    };
    // No unexpected names.
    for (const auto &name : observed) {
        if (!is_known(name)) {
            return Status::error(StatusCode::unvalidated,
                                 luisa::string{"unexpected artifact tensor name '"} + name + "'", "tensor_names");
        }
    }
    // Every required name present.
    for (const auto &mapping : recipe.tensor_mappings) {
        if (mapping.required && !is_observed(mapping.name)) {
            return Status::error(StatusCode::unvalidated,
                                 luisa::string{"missing required tensor '"} + mapping.name + "'", "tensor_names");
        }
    }
    return Status::success();
}

}// namespace

[[nodiscard]] luisa::vector<ModelTarget> ModelRegistry::targets() const {
    luisa::vector<ModelTarget> out;
    out.reserve(_recipes.size());
    for (const auto &recipe : _recipes) {
        out.emplace_back(ModelTarget{recipe.id, recipe.pp_size});
    }
    return out;
}

[[nodiscard]] Result<ModelRecipe> ModelRegistry::find_recipe(const ModelTarget &target) const {
    for (const auto &recipe : _recipes) {
        if (recipe.id == target.id && recipe.pp_size == target.pp_size) {
            return Result<ModelRecipe>::success(recipe);
        }
    }
    return Result<ModelRecipe>::failure(StatusCode::not_found,
                                        luisa::string{"no recipe for target '"} + target.name() + "'", "model");
}

[[nodiscard]] Result<RecipeMatch> ModelRegistry::match_signature(
    const ModelTarget &declared_target,
    const ArtifactSignature &signature,
    const RegistryPolicy &policy) const {
    const auto index = recipe_index_of(_recipes, declared_target);
    if (index >= _recipes.size()) {
        return Result<RecipeMatch>::failure(StatusCode::not_found,
                                            luisa::string{"no recipe for target '"} + declared_target.name() + "'", "model");
    }
    const auto &recipe = _recipes[index];
    // An outer model_type alone is NEVER sufficient -- not even with the
    // unsafe override. TeleOCR's outer type collides with other checkpoints.
    const bool has_artifact_identity = !signature.checkpoint_revision.empty() ||
                                       !signature.config_fingerprint.empty() ||
                                       !signature.tensor_names.empty();
    if (!has_artifact_identity) {
        return Result<RecipeMatch>::failure(
            StatusCode::unvalidated,
            "outer model_type alone cannot select a recipe; a checkpoint revision, config fingerprint or tensor signature is required",
            "signature");
    }
    // A validated match requires ALL of: support==validated, a fully pinned
    // recipe identity, exact revision/fingerprint/package-format equality, and
    // an exact canonical tensor signature. Anything short of that fails closed.
    const bool recipe_pin_complete = recipe.support == SupportState::validated &&
                                     recipe.identity_pinned() &&
                                     !recipe.identity.package_format_version.empty();
    const bool identity_exact = recipe_pin_complete &&
                                signature.checkpoint_revision == recipe.identity.checkpoint_revision &&
                                signature.config_fingerprint == recipe.identity.config_fingerprint &&
                                !signature.package_format_version.empty() &&
                                signature.package_format_version == recipe.identity.package_format_version;
    if (identity_exact) {
        const auto tensor_status = compare_tensor_signature(recipe, signature);
        if (tensor_status.ok()) {
            RecipeMatch match;
            match.recipe_index = index;
            match.target = declared_target;
            match.validated = true;
            match.message = "exact pinned signature match (support, identity, package format and tensor signature)";
            return Result<RecipeMatch>::success(std::move(match));
        }
        // Identity matched but the tensor signature did not: fail closed even
        // under the unsafe override, because this is a concrete mismatch, not
        // an unknown artifact.
        return Result<RecipeMatch>::failure(std::move(tensor_status));
    }
    if (!policy.allow_unsafe_experimental) {
        return Result<RecipeMatch>::failure(
            StatusCode::unvalidated,
            "artifact signature does not exactly match a validated pinned recipe; production default is fail-closed (use the explicit unsafe experimental override to inspect it as unvalidated)",
            "signature");
    }
    // Explicit unsafe override: return the declared recipe as an UNVALIDATED
    // experimental candidate. This never marks anything validated/supported.
    RecipeMatch match;
    match.recipe_index = index;
    match.target = declared_target;
    match.validated = false;
    match.unsafe_override_used = true;
    match.message = "unsafe experimental override: recipe remains unvalidated; no inference is permitted";
    return Result<RecipeMatch>::success(std::move(match));
}

}// namespace luisa::ocr
