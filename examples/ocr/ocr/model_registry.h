#pragma once
// OCR engine -- fail-closed recipe registry (Phase 0).
//
// The registry maps exact, validated config/tensor signatures to recipes.
// Phase 0 ships the planned recipe table (all 17 targets, all `planned`) and
// the signature/override contract; no signature is pinned yet, so every
// lookup fails closed with `unvalidated` unless the caller passes the
// explicit unsafe experimental override -- and even then the returned recipe
// stays marked unvalidated. An outer HF `model_type` string alone NEVER
// selects a recipe (TeleOCR is the required regression case).
//
// Host-only header: no DSL/runtime device includes.
#include <cstdint>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/optional.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/model_recipe.h"
#include "ocr/status.h"

namespace luisa::ocr {

// One selectable target: a ModelId plus (for PP-OCRv6) an explicit size.
struct ModelTarget {
    ModelId id{ModelId::unknown};
    PpOcrSize pp_size{PpOcrSize::none};

    [[nodiscard]] luisa::string name() const;
    [[nodiscard]] bool operator==(const ModelTarget &other) const noexcept {
        return id == other.id && pp_size == other.pp_size;
    }
};

// Parse "pp_ocrv6:medium", "glm_ocr", ... . A bare "pp_ocrv6" without a size
// is rejected: the size variant must always be explicit.
[[nodiscard]] Result<ModelTarget> parse_model_target(luisa::string_view text);

// Observed identity of a candidate artifact. All fields come from actually
// reading the package/config in later phases; Phase 0 callers (tests, CLI)
// construct them by hand. `outer_model_type` is recorded for diagnostics
// only and is never sufficient for selection.
struct ArtifactSignature {
    luisa::string outer_model_type;// HF config model_type (advisory only)
    luisa::string checkpoint_revision;
    luisa::string config_fingerprint;
    luisa::string package_format_version;
    // Canonical tensor names observed in the package. For an exact validated
    // match this must equal the recipe's canonical tensor signature EXACTLY:
    // every required/known name present, no duplicates, no unexpected names.
    // Order is irrelevant (the comparison is set-based after duplicate check).
    luisa::vector<luisa::string> tensor_names;
};

// Explicit opt-in for unknown/unpinned artifacts. Even with the override the
// match result remains unvalidated/experimental; production default is
// fail-closed. The override never turns a planned recipe into a working one.
struct RegistryPolicy {
    bool allow_unsafe_experimental{false};
};

struct RecipeMatch {
    // Index into the registry's recipe table (stable for the process).
    size_t recipe_index{0};
    ModelTarget target;
    // Always false in Phase 0: no artifact signature has been validated.
    bool validated{false};
    // True when the match was only possible through the unsafe override.
    bool unsafe_override_used{false};
    luisa::string message;// why the match is/is not validated
};

class ModelRegistry {
public:
    // Process-wide planned recipe table (one recipe per selectable target).
    // Every recipe is `planned` and unpinned; no signature can validate.
    ModelRegistry();

    // Explicit custom table (tests / later-phase loaders). The caller owns the
    // recipes; the registry does NOT relax any matching rule. A recipe only
    // validates when it is `validated`, pinned, and the artifact signature
    // matches exactly (see match_signature). This never weakens the default
    // planned registry.
    explicit ModelRegistry(luisa::vector<ModelRecipe> recipes);

    [[nodiscard]] luisa::span<const ModelRecipe> recipes() const noexcept {
        return luisa::span{_recipes};
    }
    // All selectable targets in stable order (17 entries in Phase 0).
    [[nodiscard]] luisa::vector<ModelTarget> targets() const;

    // Exact-target lookup of the planned recipe declaration. This is a
    // contract lookup, NOT artifact validation; the recipe's support state
    // remains `planned`.
    [[nodiscard]] Result<ModelRecipe> find_recipe(const ModelTarget &target) const;

    // Signature-based selection, fail-closed by default. A match is
    // `validated` ONLY when ALL of the following hold:
    //   * the recipe's support state is `validated`;
    //   * the recipe pins a non-empty checkpoint revision AND config
    //     fingerprint AND package format version;
    //   * the artifact's revision, fingerprint and package format version
    //     equal the recipe's exactly;
    //   * the artifact's canonical tensor signature equals the recipe's
    //     tensor-map signature exactly (all required/known names present, no
    //     duplicates, no unexpected names).
    // Anything else fails closed with `unvalidated`. With
    // policy.allow_unsafe_experimental the *declared* target may instead be
    // returned as an explicitly UNVALIDATED experimental candidate; the outer
    // model_type alone is never enough, even with the override, and the
    // override never marks anything validated.
    [[nodiscard]] Result<RecipeMatch> match_signature(
        const ModelTarget &declared_target,
        const ArtifactSignature &signature,
        const RegistryPolicy &policy = {}) const;

private:
    luisa::vector<ModelRecipe> _recipes;
};

}// namespace luisa::ocr
