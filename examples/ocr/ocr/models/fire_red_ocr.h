#pragma once
// OCR engine -- FireRed-OCR recipe contract (Phase 4). Host-only.
//
// FireRed-OCR reuses the Qwen3-VL graph (runtime/qwen3_vl_executor.*) ONLY
// through its OWN exact tensor map, prompt template, and output-parser
// configuration. It is NOT a second graph implementation: the executor,
// operators, and decode path are shared; what differs is the recipe data.
//
// No GRPO / training operators are imported or referenced anywhere: this is
// an inference-only contract (plan §3: "do not import training/GRPO nodes").
//
// HONESTY (plan §1/§6): no official FireRed-OCR artifact has been pinned or
// inspected. The recipe here is a SYNTHETIC recipe-driven path exercised by
// the synthetic vertical slice; real artifacts fail closed. The official
// gates are recorded in MODELS.md. Nothing in this header validates a real
// checkpoint, and no checkpoint hash / tensor name / prompt is invented as
// an official fact -- every name below is a declared synthetic contract.
#include <cstdint>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "ocr/models/qwen3_vl.h"
#include "ocr/status.h"

namespace luisa::ocr {

// FireRed-OCR output contract: structurally valid Markdown with HTML/OTSL
// tables and LaTeX formulas, validated by output/markdown + output/table.
// The parser id is a declared contract tag, not a claim of official parity.
inline constexpr luisa::string_view kFireRedOutputParserId = "fire_red_markdown";
// FireRed prompt template id (registered in prompt.cpp; synthetic layout).
inline constexpr luisa::string_view kFireRedPromptTemplateId = "fire_red_ocr";

// FireRed-OCR recipe: the Qwen3-VL graph plus FireRed-specific identity.
// The tensor map uses a DIFFERENT prefix layout than the Qwen3-VL synthetic
// contract (text under "language_model.", vision under "model.visual.") so a
// FireRed package can never be loaded through the Qwen3-VL name map by
// accident -- the executor resolves names from THIS map.
struct FireRedOcrConfig {
    Qwen3VlConfig graph;
    Qwen3VlTensorNames names;
    luisa::string prompt_template_id{luisa::string{kFireRedPromptTemplateId}};
    luisa::string output_parser_id{luisa::string{kFireRedOutputParserId}};

    [[nodiscard]] Status validate() const;
};

// The declared SMALL SYNTHETIC FireRed-OCR recipe: same geometry as the
// Qwen3-VL synthetic slice, FireRed tensor prefixes / prompt / parser.
// NOT a real checkpoint.
[[nodiscard]] FireRedOcrConfig fire_red_ocr_synthetic_config();

// Real-artifact gate: ALWAYS fail-closed in Phase 4. No official FireRed-OCR
// checkpoint/config/reference has been qualified, so any real artifact is
// refused with `unvalidated` regardless of its contents. The synthetic path
// is exercised only through the explicit synthetic config above (tests).
[[nodiscard]] Status fire_red_ocr_validate_real_artifact(luisa::string_view checkpoint_revision,
                                                         luisa::string_view config_fingerprint);

// Training-node guard: the FireRed contract must never carry GRPO/training
// components. This checks a recipe's component list for forbidden roles and
// is used by the recipe validation + tests. (The engine has no training ops
// at all; this is a contract-level assertion, not a runtime filter.)
[[nodiscard]] Status fire_red_ocr_check_no_training_nodes(luisa::span<const luisa::string> component_roles);

}// namespace luisa::ocr
