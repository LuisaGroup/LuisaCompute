// Test for examples/ocr/ocr/model_recipe.h/.cpp and model_registry.h/.cpp --
// the Phase 0 recipe/registry contract.
// Host-only: no device, no backend, no model artifacts, no exceptions.
//
// This test covers:
// - registry coverage of all 15 requirement models (17 selectable targets:
//   PP-OCRv6 tiny/small/medium as one family with explicit sizes);
// - every recipe is `planned`, structurally valid, and unpinned (no invented
//   revisions/fingerprints);
// - architecture families distinguish cnn/dense/layout/hybrid-GDN/custom/MoE;
// - target parsing (explicit PP-OCRv6 sizes, unknown names fail closed);
// - capability/mode negotiation (spotting coordinates, extract schema,
//   languages, region/color guidance);
// - exact-signature fail-closed matching, including the TeleOCR rule that an
//   outer model_type alone never selects a recipe;
// - the unsafe experimental override remaining unvalidated.
#include "ut/ut.hpp"

#include <cstdint>

#include "ocr/model_recipe.h"
#include "ocr/model_registry.h"
#include "ocr/status.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

void test_registry_coverage() {
    const ModelRegistry registry;
    const auto targets = registry.targets();
    // 15 models; PP-OCRv6 contributes three size variants => 17 targets.
    expect(targets.size() == 17u) << "all 15 requirement models must be selectable";
    expect(registry.recipes().size() == 17u);
    const ModelId required_ids[] = {
        ModelId::pp_ocrv6, ModelId::paddle_ocr_vl_1_6, ModelId::ovis_ocr2,
        ModelId::tele_ocr, ModelId::glm_ocr, ModelId::mineru_2_5_pro_2605_1_2b,
        ModelId::hunyuan_ocr_1_5, ModelId::surya_2_5, ModelId::open_doc_0_1b,
        ModelId::fire_red_ocr, ModelId::qwen3_vl_2b_instruct, ModelId::deepseek_ocr_2,
        ModelId::dots_mocr, ModelId::got_ocr2_0, ModelId::olm_ocr_2,
    };
    for (auto id : required_ids) {
        bool found = false;
        for (const auto &target : targets) {
            if (target.id == id) { found = true; }
        }
        expect(found) << "missing model id in registry";
    }
    // PP-OCRv6: exactly the three explicit sizes.
    uint32_t pp_count = 0;
    bool tiny = false, small = false, medium = false;
    for (const auto &target : targets) {
        if (target.id != ModelId::pp_ocrv6) { continue; }
        pp_count++;
        tiny = tiny || target.pp_size == PpOcrSize::tiny;
        small = small || target.pp_size == PpOcrSize::small;
        medium = medium || target.pp_size == PpOcrSize::medium;
    }
    expect(pp_count == 3u);
    expect(tiny && small && medium);
}

void test_all_recipes_planned_and_valid() {
    const ModelRegistry registry;
    for (const auto &recipe : registry.recipes()) {
        expect(recipe.support == SupportState::planned) << recipe.target_name();
        expect(validate_recipe(recipe).ok()) << recipe.target_name();
        // No invented pins: Phase 0 has inspected no official artifact.
        expect(!recipe.identity_pinned()) << recipe.target_name();
        expect(recipe.identity.checkpoint_revision.empty());
        expect(recipe.identity.config_fingerprint.empty());
        expect(!recipe.identity.recipe_version.empty());
        expect(recipe.cache_layout != CacheLayoutKind::unspecified);
        expect(!recipe.prompt.output_parser_id.empty());
        expect(!recipe.capabilities.modes.empty());
        expect(recipe.prompt.greedy_default);
    }
}

void test_architecture_families() {
    const ModelRegistry registry;
    const auto family_of = [&registry](ModelId id) {
        for (const auto &recipe : registry.recipes()) {
            if (recipe.id == id) { return recipe.family; }
        }
        return ArchitectureFamily::unknown;
    };
    expect(family_of(ModelId::pp_ocrv6) == ArchitectureFamily::cnn);
    expect(family_of(ModelId::paddle_ocr_vl_1_6) == ArchitectureFamily::layout_vlm);
    expect(family_of(ModelId::glm_ocr) == ArchitectureFamily::layout_vlm);
    expect(family_of(ModelId::mineru_2_5_pro_2605_1_2b) == ArchitectureFamily::layout_vlm);
    expect(family_of(ModelId::open_doc_0_1b) == ArchitectureFamily::layout_vlm);
    expect(family_of(ModelId::ovis_ocr2) == ArchitectureFamily::hybrid_gdn);
    expect(family_of(ModelId::surya_2_5) == ArchitectureFamily::hybrid_gdn);
    expect(family_of(ModelId::hunyuan_ocr_1_5) == ArchitectureFamily::custom_vlm);
    expect(family_of(ModelId::deepseek_ocr_2) == ArchitectureFamily::moe_vlm);
    expect(family_of(ModelId::tele_ocr) == ArchitectureFamily::dense_vlm);
    expect(family_of(ModelId::fire_red_ocr) == ArchitectureFamily::dense_vlm);
    expect(family_of(ModelId::qwen3_vl_2b_instruct) == ArchitectureFamily::dense_vlm);
    expect(family_of(ModelId::dots_mocr) == ArchitectureFamily::dense_vlm);
    expect(family_of(ModelId::got_ocr2_0) == ArchitectureFamily::dense_vlm);
    expect(family_of(ModelId::olm_ocr_2) == ArchitectureFamily::dense_vlm);
    // Hybrid models must not collapse to KV-only caches.
    for (const auto &recipe : registry.recipes()) {
        if (recipe.family == ArchitectureFamily::hybrid_gdn) {
            expect(recipe.cache_layout == CacheLayoutKind::gdn_state);
        }
        if (recipe.family == ArchitectureFamily::cnn) {
            expect(recipe.cache_layout == CacheLayoutKind::none);
        }
    }
}

void test_target_parsing() {
    {
        const auto parsed = parse_model_target("pp_ocrv6:medium");
        expect(parsed.ok());
        expect(parsed.value->id == ModelId::pp_ocrv6);
        expect(parsed.value->pp_size == PpOcrSize::medium);
        expect(parsed.value->name() == "pp_ocrv6:medium");
    }
    {
        const auto parsed = parse_model_target("glm_ocr");
        expect(parsed.ok());
        expect(parsed.value->id == ModelId::glm_ocr);
        expect(parsed.value->pp_size == PpOcrSize::none);
    }
    expect(!parse_model_target("").ok());
    expect(!parse_model_target("no_such_model").ok());
    expect(!parse_model_target("pp_ocrv6").ok());// size must be explicit
    expect(!parse_model_target("pp_ocrv6:large").ok());
    expect(!parse_model_target("glm_ocr:medium").ok());// suffix only for pp_ocrv6
    expect(parse_model_target("pp_ocrv6:tiny").status.code == StatusCode::ok);
    expect(parse_model_target("pp_ocrv6:").status.code == StatusCode::invalid_argument);
    // Round-trip every registered target name.
    const ModelRegistry registry;
    for (const auto &target : registry.targets()) {
        const auto parsed = parse_model_target(target.name());
        expect(parsed.ok()) << target.name();
        expect(*parsed.value == target) << target.name();
    }
}

void test_find_recipe() {
    const ModelRegistry registry;
    const auto found = registry.find_recipe(ModelTarget{ModelId::pp_ocrv6, PpOcrSize::small});
    expect(found.ok());
    expect(found.value->pp_size == PpOcrSize::small);
    expect(found.value->family == ArchitectureFamily::cnn);
    const auto missing = registry.find_recipe(ModelTarget{ModelId::pp_ocrv6, PpOcrSize::none});
    expect(!missing.ok());
    expect(missing.status.code == StatusCode::not_found);
    const auto unknown = registry.find_recipe(ModelTarget{ModelId::unknown, PpOcrSize::none});
    expect(!unknown.ok());
}

void test_capability_negotiation() {
    const ModelRegistry registry;
    const auto recipe_of = [&registry](ModelId id, PpOcrSize size = PpOcrSize::none) {
        const auto found = registry.find_recipe(ModelTarget{id, size});
        expect(found.ok());
        return *found.value;
    };
    // PP-OCRv6: plain + spotting with coordinates/confidence; no markdown.
    {
        const auto recipe = recipe_of(ModelId::pp_ocrv6, PpOcrSize::medium);
        expect(validate_request_capabilities(recipe, RecognizeMode::plain, "", false, false, false).ok());
        expect(validate_request_capabilities(recipe, RecognizeMode::spotting, "", false, false, false).ok());
        expect(validate_request_capabilities(recipe, RecognizeMode::markdown, "", false, false, false).code == StatusCode::unsupported);
        expect(recipe.capabilities.has_calibrated_confidence);
    }
    // Generative VLM without validated grounding: spotting rejected.
    {
        const auto recipe = recipe_of(ModelId::qwen3_vl_2b_instruct);
        expect(!recipe.capabilities.has_calibrated_confidence);
        expect(validate_request_capabilities(recipe, RecognizeMode::spotting, "", false, false, false).code == StatusCode::unsupported);
        expect(validate_request_capabilities(recipe, RecognizeMode::markdown, "", false, false, false).ok());
    }
    // extract requires schema/prompt.
    {
        const auto recipe = recipe_of(ModelId::glm_ocr);
        expect(validate_request_capabilities(recipe, RecognizeMode::extract, "", false, false, false).code == StatusCode::invalid_argument);
        expect(validate_request_capabilities(recipe, RecognizeMode::extract, "", true, false, false).ok());
    }
    // Language restrictions: olmOCR-2 is English-only.
    {
        const auto recipe = recipe_of(ModelId::olm_ocr_2);
        expect(validate_request_capabilities(recipe, RecognizeMode::plain, "en", false, false, false).ok());
        expect(validate_request_capabilities(recipe, RecognizeMode::plain, "zh", false, false, false).code == StatusCode::unsupported);
        expect(validate_request_capabilities(recipe, RecognizeMode::plain, "", false, false, false).ok());// unspecified allowed
    }
    // Language matching is EXACT: "multi" is a literal tag, not a wildcard.
    // A nonsense hint must be rejected even on multilingual planned recipes.
    {
        const auto recipe = recipe_of(ModelId::qwen3_vl_2b_instruct);
        expect(validate_request_capabilities(recipe, RecognizeMode::plain, "zh", false, false, false).ok());
        expect(validate_request_capabilities(recipe, RecognizeMode::plain, "en", false, false, false).ok());
        expect(validate_request_capabilities(recipe, RecognizeMode::plain, "multi", false, false, false).ok());// literal tag
        expect(validate_request_capabilities(recipe, RecognizeMode::plain, "xx", false, false, false).code == StatusCode::unsupported);
        expect(validate_request_capabilities(recipe, RecognizeMode::plain, "klingon", false, false, false).code == StatusCode::unsupported);
    }
    {
        // Surya declares only the literal "multi" tag: concrete hints that are
        // not declared are rejected, and so is nonsense.
        const auto recipe = recipe_of(ModelId::surya_2_5);
        expect(validate_request_capabilities(recipe, RecognizeMode::plain, "multi", false, false, false).ok());
        expect(validate_request_capabilities(recipe, RecognizeMode::plain, "xx", false, false, false).code == StatusCode::unsupported);
        expect(validate_request_capabilities(recipe, RecognizeMode::plain, "zh", false, false, false).code == StatusCode::unsupported);
    }
    // Region/color guidance only where declared (GOT).
    {
        const auto got = recipe_of(ModelId::got_ocr2_0);
        expect(validate_request_capabilities(got, RecognizeMode::plain, "", false, true, true).ok());
        const auto qwen = recipe_of(ModelId::qwen3_vl_2b_instruct);
        expect(validate_request_capabilities(qwen, RecognizeMode::plain, "", false, true, false).code == StatusCode::unsupported);
        expect(validate_request_capabilities(qwen, RecognizeMode::plain, "", false, false, true).code == StatusCode::unsupported);
    }
    // Unknown mode is invalid, not unsupported.
    {
        const auto recipe = recipe_of(ModelId::glm_ocr);
        expect(validate_request_capabilities(recipe, static_cast<RecognizeMode>(99u), "", false, false, false).code == StatusCode::invalid_argument);
    }
}

void test_recipe_validation_rules() {
    const ModelRegistry registry;
    const auto base = *registry.find_recipe(ModelTarget{ModelId::glm_ocr, PpOcrSize::none}).value;
    {
        auto recipe = base;
        recipe.id = ModelId::unknown;
        expect(validate_recipe(recipe).code == StatusCode::invalid_argument);
    }
    {
        auto recipe = base;
        recipe.pp_size = PpOcrSize::medium;// only pp_ocrv6 may carry a size
        expect(validate_recipe(recipe).code == StatusCode::invalid_argument);
    }
    {
        auto recipe = *registry.find_recipe(ModelTarget{ModelId::pp_ocrv6, PpOcrSize::tiny}).value;
        recipe.pp_size = PpOcrSize::none;
        expect(validate_recipe(recipe).code == StatusCode::invalid_argument);
    }
    {
        auto recipe = base;
        recipe.capabilities.modes.clear();
        expect(validate_recipe(recipe).code == StatusCode::invalid_argument);
    }
    {
        auto recipe = base;
        recipe.capabilities.modes.emplace_back(RecognizeMode::spotting);
        recipe.capabilities.has_source_coordinates = false;
        expect(validate_recipe(recipe).code == StatusCode::invalid_argument);
    }
    {
        // A "validated" recipe without pinned identity is a contract error.
        auto recipe = base;
        recipe.support = SupportState::validated;
        expect(validate_recipe(recipe).code == StatusCode::invalid_argument);
        recipe.identity.checkpoint_revision = "test-revision";
        recipe.identity.config_fingerprint = "test-fingerprint";
        // Still missing tensor mappings for a validated recipe.
        expect(validate_recipe(recipe).code == StatusCode::invalid_argument);
        TensorMapping mapping;
        mapping.name = "decoder.layers.0.self_attn.q_proj.weight";
        recipe.tensor_mappings.emplace_back(std::move(mapping));
        expect(validate_recipe(recipe).ok());
    }
    {
        auto recipe = base;
        recipe.prompt.output_parser_id.clear();
        expect(validate_recipe(recipe).code == StatusCode::invalid_argument);
    }
    {
        // Duplicate modes are rejected.
        auto recipe = base;
        recipe.capabilities.modes.emplace_back(recipe.capabilities.modes.front());
        expect(validate_recipe(recipe).code == StatusCode::invalid_argument);
    }
    {
        // CNN must use cache layout none.
        auto recipe = *registry.find_recipe(ModelTarget{ModelId::pp_ocrv6, PpOcrSize::tiny}).value;
        recipe.cache_layout = CacheLayoutKind::contiguous_kv;
        expect(validate_recipe(recipe).code == StatusCode::invalid_argument);
    }
    {
        // Hybrid GDN must use gdn_state, never a KV-only layout.
        auto recipe = *registry.find_recipe(ModelTarget{ModelId::ovis_ocr2, PpOcrSize::none}).value;
        recipe.cache_layout = CacheLayoutKind::contiguous_kv;
        expect(validate_recipe(recipe).code == StatusCode::invalid_argument);
    }
    {
        // Components must be non-empty with role/kind set.
        auto recipe = base;
        recipe.components.clear();
        expect(validate_recipe(recipe).code == StatusCode::invalid_argument);
    }
    {
        auto recipe = base;
        recipe.components.front().kind.clear();
        expect(validate_recipe(recipe).code == StatusCode::invalid_argument);
    }
    {
        // load_only also requires pinned identity + tensor mappings.
        auto recipe = base;
        recipe.support = SupportState::load_only;
        expect(validate_recipe(recipe).code == StatusCode::invalid_argument);
        recipe.identity.checkpoint_revision = "test-revision";
        recipe.identity.config_fingerprint = "test-fingerprint";
        expect(validate_recipe(recipe).code == StatusCode::invalid_argument);// no tensors yet
        TensorMapping mapping;
        mapping.name = "decoder.layers.0.self_attn.q_proj.weight";
        recipe.tensor_mappings.emplace_back(std::move(mapping));
        expect(validate_recipe(recipe).ok());
    }
}

void test_signature_fail_closed() {
    const ModelRegistry registry;
    const ModelTarget target{ModelId::glm_ocr, PpOcrSize::none};
    // No artifact identity at all -> unvalidated, even with a model_type.
    {
        ArtifactSignature signature;
        signature.outer_model_type = "glm_ocr";
        const auto match = registry.match_signature(target, signature);
        expect(!match.ok());
        expect(match.status.code == StatusCode::unvalidated);
    }
    // Real-looking but unpinned identity -> fail closed by default.
    {
        ArtifactSignature signature;
        signature.outer_model_type = "glm_ocr";
        signature.checkpoint_revision = "some-revision";
        signature.config_fingerprint = "some-fingerprint";
        signature.tensor_names.emplace_back("decoder.layers.0.self_attn.q_proj.weight");
        const auto match = registry.match_signature(target, signature);
        expect(!match.ok());
        expect(match.status.code == StatusCode::unvalidated);
    }
    // Unknown declared target -> not_found regardless of signature.
    {
        ArtifactSignature signature;
        signature.checkpoint_revision = "rev";
        const auto match = registry.match_signature(ModelTarget{ModelId::unknown, PpOcrSize::none}, signature);
        expect(!match.ok());
        expect(match.status.code == StatusCode::not_found);
    }
}

void test_teleocr_outer_model_type_insufficient() {
    const ModelRegistry registry;
    const ModelTarget tele{ModelId::tele_ocr, PpOcrSize::none};
    // TeleOCR's outer model_type collides with other checkpoints: the outer
    // type alone must NEVER select a recipe, with or without the override.
    ArtifactSignature signature;
    signature.outer_model_type = "qwen2_5_vl";// plausible colliding outer type
    {
        const auto match = registry.match_signature(tele, signature);
        expect(!match.ok());
        expect(match.status.code == StatusCode::unvalidated);
    }
    {
        RegistryPolicy policy;
        policy.allow_unsafe_experimental = true;
        const auto match = registry.match_signature(tele, signature, policy);
        expect(!match.ok()) << "outer model_type alone must not select even under the unsafe override";
        expect(match.status.code == StatusCode::unvalidated);
    }
    // Even the model's own name as outer type is insufficient.
    signature.outer_model_type = "tele_ocr";
    {
        const auto match = registry.match_signature(tele, signature);
        expect(!match.ok());
        expect(match.status.code == StatusCode::unvalidated);
    }
}

void test_unsafe_override_stays_unvalidated() {
    const ModelRegistry registry;
    const ModelTarget target{ModelId::tele_ocr, PpOcrSize::none};
    ArtifactSignature signature;
    signature.outer_model_type = "qwen2_5_vl";
    signature.checkpoint_revision = "unpinned-revision";
    signature.config_fingerprint = "unpinned-fingerprint";
    signature.tensor_names.emplace_back("model.layers.0.self_attn.q_proj.weight");
    RegistryPolicy policy;
    policy.allow_unsafe_experimental = true;
    const auto match = registry.match_signature(target, signature, policy);
    expect(match.ok());
    expect(!match.value->validated) << "unsafe override must remain unvalidated";
    expect(match.value->unsafe_override_used);
    expect(match.value->target == target);
    expect(match.value->recipe_index < registry.recipes().size());
    const auto &recipe = registry.recipes()[match.value->recipe_index];
    expect(recipe.id == ModelId::tele_ocr);
    expect(recipe.support == SupportState::planned) << "override never upgrades support state";
    // The same signature without the override still fails closed.
    const auto strict = registry.match_signature(target, signature);
    expect(!strict.ok());
    expect(strict.status.code == StatusCode::unvalidated);
}

// Build a fake pinned, validated recipe for exact-match tests. This is a TEST
// fixture only -- it never enters the default planned registry.
[[nodiscard]] ModelRecipe fake_validated_recipe() {
    ModelRecipe recipe;
    recipe.id = ModelId::glm_ocr;
    recipe.family = ArchitectureFamily::layout_vlm;
    recipe.support = SupportState::validated;
    recipe.identity.recipe_version = "0";
    recipe.identity.checkpoint_revision = "fake-revision-abc123";
    recipe.identity.config_fingerprint = "fake-fingerprint-def456";
    recipe.identity.package_format_version = "1";
    recipe.cache_layout = CacheLayoutKind::contiguous_kv;
    recipe.preprocessing.version = "0";
    recipe.prompt.output_parser_id = "markdown";
    ComponentContract decoder;
    decoder.role = "decoder";
    decoder.kind = "fake_dense";
    decoder.attention_mask = AttentionMaskKind::causal;
    recipe.components.emplace_back(std::move(decoder));    recipe.capabilities.modes = {RecognizeMode::plain, RecognizeMode::markdown};
    recipe.capabilities.languages = {"en"};
    TensorMapping required_a;
    required_a.name = "decoder.layers.0.self_attn.q_proj.weight";
    required_a.required = true;
    TensorMapping required_b;
    required_b.name = "decoder.layers.0.self_attn.k_proj.weight";
    required_b.required = true;
    TensorMapping optional_c;
    optional_c.name = "decoder.layers.0.mlp.gate_proj.weight";
    optional_c.required = false;
    recipe.tensor_mappings.emplace_back(std::move(required_a));
    recipe.tensor_mappings.emplace_back(std::move(required_b));
    recipe.tensor_mappings.emplace_back(std::move(optional_c));
    return recipe;
}

[[nodiscard]] ArtifactSignature exact_signature() {
    ArtifactSignature signature;
    signature.outer_model_type = "glm_ocr";
    signature.checkpoint_revision = "fake-revision-abc123";
    signature.config_fingerprint = "fake-fingerprint-def456";
    signature.package_format_version = "1";
    signature.tensor_names.emplace_back("decoder.layers.0.self_attn.q_proj.weight");
    signature.tensor_names.emplace_back("decoder.layers.0.self_attn.k_proj.weight");
    signature.tensor_names.emplace_back("decoder.layers.0.mlp.gate_proj.weight");
    return signature;
}

void test_exact_match_requires_full_signature() {
    const auto recipe = fake_validated_recipe();
    expect(validate_recipe(recipe).ok());// fixture is structurally valid
    luisa::vector<ModelRecipe> table;
    table.emplace_back(recipe);
    const ModelRegistry registry{std::move(table)};
    const ModelTarget target{ModelId::glm_ocr, PpOcrSize::none};

    // Only the exact full signature validates.
    {
        const auto match = registry.match_signature(target, exact_signature());
        expect(match.ok());
        expect(match.value->validated);
        expect(!match.value->unsafe_override_used);
        expect(match.value->recipe_index == 0u);
    }
    // Optional tensor may be absent; required ones must be present.
    {
        auto signature = exact_signature();
        signature.tensor_names.pop_back();// drop optional gate_proj
        const auto match = registry.match_signature(target, signature);
        expect(match.ok());
        expect(match.value->validated);
    }
    // Wrong revision -> fail closed.
    {
        auto signature = exact_signature();
        signature.checkpoint_revision = "other-revision";
        expect(registry.match_signature(target, signature).status.code == StatusCode::unvalidated);
    }
    // Wrong fingerprint -> fail closed.
    {
        auto signature = exact_signature();
        signature.config_fingerprint = "other-fingerprint";
        expect(registry.match_signature(target, signature).status.code == StatusCode::unvalidated);
    }
    // Wrong package format version -> fail closed.
    {
        auto signature = exact_signature();
        signature.package_format_version = "2";
        expect(registry.match_signature(target, signature).status.code == StatusCode::unvalidated);
    }
    // Missing package format version -> fail closed.
    {
        auto signature = exact_signature();
        signature.package_format_version.clear();
        expect(registry.match_signature(target, signature).status.code == StatusCode::unvalidated);
    }
    // Missing a required tensor -> fail closed.
    {
        auto signature = exact_signature();
        signature.tensor_names.pop_back();// drop optional
        signature.tensor_names.pop_back();// drop required k_proj
        expect(registry.match_signature(target, signature).status.code == StatusCode::unvalidated);
    }
    // Unexpected tensor name -> fail closed.
    {
        auto signature = exact_signature();
        signature.tensor_names.emplace_back("decoder.layers.999.unknown.weight");
        expect(registry.match_signature(target, signature).status.code == StatusCode::unvalidated);
    }
    // Duplicate tensor name -> fail closed.
    {
        auto signature = exact_signature();
        signature.tensor_names.emplace_back("decoder.layers.0.self_attn.q_proj.weight");
        expect(registry.match_signature(target, signature).status.code == StatusCode::unvalidated);
    }
    // Incomplete artifact identity (only outer model_type) -> fail closed.
    {
        ArtifactSignature signature;
        signature.outer_model_type = "glm_ocr";
        expect(registry.match_signature(target, signature).status.code == StatusCode::unvalidated);
    }
}

void test_exact_match_requires_validated_support() {
    const ModelTarget target{ModelId::glm_ocr, PpOcrSize::none};
    // Same pinned identity + exact tensors, but support is planned -> the
    // registry must NOT validate it (planned recipes never validate).
    {
        auto recipe = fake_validated_recipe();
        recipe.support = SupportState::planned;
        luisa::vector<ModelRecipe> table;
        table.emplace_back(std::move(recipe));
        const ModelRegistry registry{std::move(table)};
        const auto match = registry.match_signature(target, exact_signature());
        expect(!match.ok());
        expect(match.status.code == StatusCode::unvalidated);
    }
    // load_only is also not a validated inference match.
    {
        auto recipe = fake_validated_recipe();
        recipe.support = SupportState::load_only;
        luisa::vector<ModelRecipe> table;
        table.emplace_back(std::move(recipe));
        const ModelRegistry registry{std::move(table)};
        const auto match = registry.match_signature(target, exact_signature());
        expect(!match.ok());
        expect(match.status.code == StatusCode::unvalidated);
    }
    // The unsafe override on a planned recipe stays unvalidated.
    {
        auto recipe = fake_validated_recipe();
        recipe.support = SupportState::planned;
        luisa::vector<ModelRecipe> table;
        table.emplace_back(std::move(recipe));
        const ModelRegistry registry{std::move(table)};
        RegistryPolicy policy;
        policy.allow_unsafe_experimental = true;
        const auto match = registry.match_signature(target, exact_signature(), policy);
        expect(match.ok());
        expect(!match.value->validated);
        expect(match.value->unsafe_override_used);
    }
}

void test_default_registry_never_validates() {
    // The default planned registry must never validate any signature, even a
    // fully-populated one, because no recipe is pinned/validated.
    const ModelRegistry registry;
    const ModelTarget target{ModelId::glm_ocr, PpOcrSize::none};
    const auto match = registry.match_signature(target, exact_signature());
    expect(!match.ok());
    expect(match.status.code == StatusCode::unvalidated);
}

void test_deepseek_open_doc_contracts() {
    const ModelRegistry registry;
    // DeepSeek: SAM local perception is windowed, NOT the custom block mask;
    // the visual-LM/causal-flow component carries the custom mask.
    {
        const auto recipe = *registry.find_recipe(ModelTarget{ModelId::deepseek_ocr_2, PpOcrSize::none}).value;
        bool sam_windowed = false;
        bool visual_lm_custom = false;
        for (const auto &component : recipe.components) {
            if (component.kind == "sam_local_perception") {
                sam_windowed = component.attention_mask == AttentionMaskKind::windowed;
            }
            if (component.kind == "deepseek_visual_lm_encoder") {
                visual_lm_custom = component.attention_mask == AttentionMaskKind::custom_block_mask;
            }
        }
        expect(sam_windowed) << "SAM block must not be labeled custom_block_mask";
        expect(visual_lm_custom) << "visual-LM/causal-flow carries the custom block mask";
    }
    // OpenDoc: no generic KV decoder cache; UniRec-specific output parser.
    {
        const auto recipe = *registry.find_recipe(ModelTarget{ModelId::open_doc_0_1b, PpOcrSize::none}).value;
        expect(recipe.cache_layout == CacheLayoutKind::none);
        expect(recipe.prompt.output_parser_id == "unirec_document");
        expect(recipe.family == ArchitectureFamily::layout_vlm);// retains layout routing
    }
}

void test_name_tables() {
    expect(model_id_name(ModelId::unknown) == "unknown");
    expect(model_id_from_name("pp_ocrv6") == ModelId::pp_ocrv6);
    expect(model_id_from_name("olm_ocr_2") == ModelId::olm_ocr_2);
    expect(model_id_from_name("nope") == ModelId::unknown);
    expect(model_id_from_name("") == ModelId::unknown);
    expect(architecture_family_name(ArchitectureFamily::hybrid_gdn) == "hybrid_gdn");
    expect(architecture_family_name(ArchitectureFamily::moe_vlm) == "moe_vlm");
    expect(recognize_mode_name(RecognizeMode::markdown) == "markdown");
    expect(recognize_mode_from_name("extract").has_value());
    expect(!recognize_mode_from_name("translation").has_value());
    expect(precision_name(Precision::bf16) == "bf16");
    expect(precision_from_name("int4").has_value());
    expect(!precision_from_name("fp8").has_value());
    expect(support_state_name(SupportState::planned) == "planned");
    expect(support_state_name(SupportState::validated) == "validated");
    expect(pp_ocr_size_name(PpOcrSize::medium) == "medium");
    expect(attention_mask_kind_name(AttentionMaskKind::custom_block_mask) == "custom_block_mask");
    expect(positional_encoding_kind_name(PositionalEncodingKind::xd_rope) == "xd_rope");
    expect(cache_layout_kind_name(CacheLayoutKind::gdn_state) == "gdn_state");
}

}// namespace

static auto test_ocr_recipe_registration = [] {
    "registry_coverage"_test = [] { test_registry_coverage(); };
    "all_recipes_planned_and_valid"_test = [] { test_all_recipes_planned_and_valid(); };
    "architecture_families"_test = [] { test_architecture_families(); };
    "target_parsing"_test = [] { test_target_parsing(); };
    "find_recipe"_test = [] { test_find_recipe(); };
    "capability_negotiation"_test = [] { test_capability_negotiation(); };
    "recipe_validation_rules"_test = [] { test_recipe_validation_rules(); };
    "signature_fail_closed"_test = [] { test_signature_fail_closed(); };
    "teleocr_outer_model_type_insufficient"_test = [] { test_teleocr_outer_model_type_insufficient(); };
    "unsafe_override_stays_unvalidated"_test = [] { test_unsafe_override_stays_unvalidated(); };
    "exact_match_requires_full_signature"_test = [] { test_exact_match_requires_full_signature(); };
    "exact_match_requires_validated_support"_test = [] { test_exact_match_requires_validated_support(); };
    "default_registry_never_validates"_test = [] { test_default_registry_never_validates(); };
    "deepseek_open_doc_contracts"_test = [] { test_deepseek_open_doc_contracts(); };
    "name_tables"_test = [] { test_name_tables(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
}
