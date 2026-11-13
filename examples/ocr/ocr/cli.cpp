// OCR engine -- reusable CLI parser and structured status report
// (Phase 0 contract; Phase 12 adds --list-models / model_support_report_json).
// Consumes UTF-8 argument bytes; see cli.h for the encoding contract.
#include "ocr/cli.h"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <string_view>
#include <type_traits>
#include <utility>

#include "ocr/model_registry.h"

namespace luisa::ocr {
namespace {

[[nodiscard]] bool parse_u64(std::string_view text, uint64_t &out) noexcept {
    if (text.empty()) { return false; }
    uint64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) { return false; }
    out = value;
    return true;
}

[[nodiscard]] bool parse_u32(std::string_view text, uint32_t &out) noexcept {
    uint64_t value = 0;
    if (!parse_u64(text, value) || value > 0xffffffffull) { return false; }
    out = static_cast<uint32_t>(value);
    return true;
}

[[nodiscard]] bool parse_double(std::string_view text, double &out) noexcept {
    if (text.empty()) { return false; }
    double value = 0.0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) { return false; }
    if (!std::isfinite(value)) { return false; }
    out = value;
    return true;
}

}// namespace

[[nodiscard]] luisa::string cli_usage_text() {
    return luisa::string{
        "example_ocr -- LuisaCompute OCR engine (Phases 0-11: synthetic host/device\n"
        "infrastructure only; no REAL model is validated and Session fails closed)\n"
        "\n"
        "usage: example_ocr --backend <name> (--package <file.lcocr> | --manifest <file.json>)\n"
        " --input <image> [--input <image> ...] [options]\n"
        " example_ocr --help\n"
        " example_ocr --list-models\n"
        "\n"
        "offline (no backend/package/input required):\n"
        "  --help show this usage text (exit 0)\n"
        "  --list-models print the registry-driven model support report JSON\n"
        " (exit 0); every target is planned/unvalidated, session_status\n"
        " is \"unvalidated\", and no real model is claimed to work\n"
        "\n"
        "required (except --help/--list-models):\n"
        "  --backend <name> backend plugin: cuda | vk | dx | fallback | ...\n"
        "  --package <path> packed .lcocr model package\n"
        "  --manifest <path> package/component manifest (alternative or addition)\n"
        "  --input <path> input image/page (repeatable, in reading order)\n"
        "\n"
        "model selection:\n"
        "  --model <target>            pp_ocrv6:tiny|small|medium, paddle_ocr_vl_1_6, ovis_ocr2,\n"
        "                              tele_ocr, glm_ocr, mineru_2_5_pro_2605_1_2b, hunyuan_ocr_1_5,\n"
        "                              surya_2_5, open_doc_0_1b, fire_red_ocr, qwen3_vl_2b_instruct,\n"
        "                              deepseek_ocr_2, dots_mocr, got_ocr2_0, olm_ocr_2\n"
        "  --unsafe-experimental       explicit override: unknown signatures stay UNVALIDATED\n"
        "\n"
        "mode and output:\n"
        "  --mode <mode>               plain (default) | spotting | markdown | extract\n"
        "  --output <path>             write the result JSON here (default: stdout)\n"
        "  --prompt <text>             prompt for extract/prompt-directed modes\n"
        "  --schema <text>             JSON schema text for extract\n"
        "  --language <tag>            language hint, exact match (e.g. zh, en)\n"
        "  --region-guidance <text>    region guidance (models that declare support)\n"
        "  --color-guidance <text>     color guidance (models that declare support)\n"
        "\n"
        "pixel/token/region limits:\n"
        "  --max-overview-pixels <n>   overview/layout pass pixel ceiling (0 = engine default)\n"
        "  --max-crop-pixels <n>       region crop pixel ceiling (0 = engine default)\n"
        "  --max-pixels <n>            compatibility ceiling applied to both bounds\n"
        "  --max-patches-per-vision-batch <n>\n"
        "  --max-regions-per-batch <n>\n"
        "  --max-decode-batch <n>\n"
        "  --max-output-tokens-per-region <n>\n"
        "  --max-output-tokens-per-page <n>\n"
        "\n"
        "sampling (greedy by default):\n"
        "  --temperature <x>           >0 switches to non-greedy sampling\n"
        "  --top-p <x>                 (0,1]\n"
        "  --top-k <n>                 0 = disabled\n"
        "  --seed <n>                  deterministic sampling seed\n"
        "\n"
        "policies and budgets:\n"
        "  --weight-precision <p>      automatic (default) | fp32 | fp16 | bf16 | int8 | int4\n"
        "  --activation-precision <p>  automatic resolves to the recipe default, else fp32\n"
        "  --cache-precision <p>\n"
        "  --cache-policy <p>          automatic (default) | contiguous | paged\n"
        "  --offload <p>               disabled (default) | explicit-host\n"
        "  --graph <p>                 disabled (default) | automatic | enabled\n"
        "  --fast-path <p>             disabled (default) | automatic | enabled\n"
        "  --max-device-bytes <n>      device memory budget (0 = engine default)\n"
        "  --max-host-bytes <n>        host memory budget (0 = engine default)\n"
        "  --max-wall-time-ms <x>      wall-time budget (0 = none)\n"
        "\n"
        "No model is validated yet, so inference is always refused; the refusal\n"
        "is reported as structured JSON, never as fabricated OCR output. Synthetic\n"
        "test parity is not real-model support; see examples/ocr/QUALIFICATION.md.\n"};
}

[[nodiscard]] luisa::string json_escape(luisa::string_view text) {
    luisa::string out;
    out.reserve(text.size() + 8);
    for (auto c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20u) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(static_cast<unsigned char>(c)));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

[[nodiscard]] luisa::string status_report_json(const Status &status) {
    luisa::string json;
    json += "{\"ok\":false,\"code\":\"";
    json += status_code_name(status.code);
    json += "\",\"message\":\"";
    json += json_escape(status.message);
    json += "\",\"detail\":\"";
    json += json_escape(status.detail);
    json += "\"}\n";
    return json;
}

[[nodiscard]] luisa::string model_support_report_json() {
    // Derived from the SAME default planned table Session will consult once
    // real loading is enabled: no hard-coded target list, no invented support
    // state. Synthetic parity never upgrades a recipe; `validated` is exactly
    // "the registry recipe is in the validated support state" (false for all
    // 17 targets today).
    const ModelRegistry registry;
    const auto recipes = registry.recipes();
    luisa::string json;
    json += "{\"target_count\":";
    json += std::to_string(recipes.size());
    json += ",\"real_model_validated\":false";
    json += ",\"session_status\":\"unvalidated\"";
    json += ",\"models\":[";
    bool first = true;
    for (const auto &recipe : recipes) {
        if (!first) { json += ","; }
        first = false;
        json += "{\"target\":\"";
        json += json_escape(recipe.target_name());
        json += "\",\"model\":\"";
        json += json_escape(model_id_name(recipe.id));
        json += "\",\"family\":\"";
        json += json_escape(architecture_family_name(recipe.family));
        json += "\",\"support_state\":\"";
        json += json_escape(support_state_name(recipe.support));
        json += "\",\"validated\":";
        json += recipe.support == SupportState::validated ? "true" : "false";
        json += "}";
    }
    json += "]}\n";
    return json;
}

[[nodiscard]] Status write_status_report(const Status &status,
                                         const luisa::filesystem::path &output) {
    const auto json = status_report_json(status);
    if (output.empty()) {
        if (std::fputs(json.c_str(), stderr) < 0) {
            return Status::error(StatusCode::io_error, "failed to write status report to stderr", "output");
        }
        return Status::success();
    }
    std::FILE *file = std::fopen(luisa::to_string(output).c_str(), "wb");
    if (file == nullptr) {
        return Status::error(StatusCode::io_error,
                             luisa::string{"cannot open output file '"} + luisa::to_string(output) + "'", "output");
    }
    const bool wrote = std::fputs(json.c_str(), file) >= 0;
    const bool closed = std::fclose(file) == 0;
    if (!wrote || !closed) {
        return Status::error(StatusCode::io_error,
                             luisa::string{"failed to write status report to '"} + luisa::to_string(output) + "'", "output");
    }
    return Status::success();
}

[[nodiscard]] Result<CliArgs> parse_cli_args(int argc, const char *const *argv) {
    CliArgs args;
    bool temperature_set = false;
    for (int i = 1; i < argc; i++) {
        const std::string_view arg{argv[i]};
        const auto need_value = [&](luisa::string_view &value) -> bool {
            if (i + 1 >= argc) { return false; }
            value = std::string_view{argv[++i]};
            return true;
        };
        luisa::string_view value;
        if (arg == "--help" || arg == "-h") {
            args.help = true;
            return Result<CliArgs>::success(std::move(args));
        } else if (arg == "--list-models") {
            // Offline qualification report: like --help, needs no backend,
            // package, input, device, or model artifact.
            args.list_models = true;
            return Result<CliArgs>::success(std::move(args));
        } else if (arg == "--backend") {
            if (!need_value(value)) { return Result<CliArgs>::failure(StatusCode::invalid_argument, "--backend requires a value", "backend"); }
            args.session.backend = luisa::string{value};
        } else if (arg == "--package") {
            if (!need_value(value)) { return Result<CliArgs>::failure(StatusCode::invalid_argument, "--package requires a value", "package"); }
            if (!luisa::path_from_utf8(value, args.session.package)) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, "--package is not valid UTF-8", "package");
            }
        } else if (arg == "--manifest") {
            if (!need_value(value)) { return Result<CliArgs>::failure(StatusCode::invalid_argument, "--manifest requires a value", "manifest"); }
            if (!luisa::path_from_utf8(value, args.session.manifest)) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, "--manifest is not valid UTF-8", "manifest");
            }
        } else if (arg == "--model") {
            if (!need_value(value)) { return Result<CliArgs>::failure(StatusCode::invalid_argument, "--model requires a value", "model"); }
            const auto target = parse_model_target(value);
            if (!target.ok()) { return Result<CliArgs>::failure(target.status); }
            args.session.model = luisa::string{value};
        } else if (arg == "--unsafe-experimental") {
            args.session.registry_policy.allow_unsafe_experimental = true;
        } else if (arg == "--input") {
            if (!need_value(value)) { return Result<CliArgs>::failure(StatusCode::invalid_argument, "--input requires a value", "input"); }
            PageInput page;
            if (!luisa::path_from_utf8(value, page.path)) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, "--input is not valid UTF-8", "input");
            }
            page.source_page_index = static_cast<uint32_t>(args.pages.size());
            args.pages.emplace_back(std::move(page));
        } else if (arg == "--mode") {
            if (!need_value(value)) { return Result<CliArgs>::failure(StatusCode::invalid_argument, "--mode requires a value", "mode"); }
            const auto mode = recognize_mode_from_name(value);
            if (!mode.has_value()) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument,
                                                luisa::string{"unknown mode '"} + luisa::string{value} + "' (plain|spotting|markdown|extract)", "mode");
            }
            args.options.mode = *mode;
        } else if (arg == "--output") {
            if (!need_value(value)) { return Result<CliArgs>::failure(StatusCode::invalid_argument, "--output requires a value", "output"); }
            if (!luisa::path_from_utf8(value, args.output)) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, "--output is not valid UTF-8", "output");
            }
        } else if (arg == "--prompt") {
            if (!need_value(value)) { return Result<CliArgs>::failure(StatusCode::invalid_argument, "--prompt requires a value", "prompt"); }
            args.options.prompt = luisa::string{value};
        } else if (arg == "--schema") {
            if (!need_value(value)) { return Result<CliArgs>::failure(StatusCode::invalid_argument, "--schema requires a value", "schema"); }
            args.options.schema = luisa::string{value};
        } else if (arg == "--language") {
            if (!need_value(value)) { return Result<CliArgs>::failure(StatusCode::invalid_argument, "--language requires a value", "language"); }
            args.options.language = luisa::string{value};
        } else if (arg == "--region-guidance") {
            if (!need_value(value)) { return Result<CliArgs>::failure(StatusCode::invalid_argument, "--region-guidance requires a value", "region_guidance"); }
            args.options.region_guidance = luisa::string{value};
        } else if (arg == "--color-guidance") {
            if (!need_value(value)) { return Result<CliArgs>::failure(StatusCode::invalid_argument, "--color-guidance requires a value", "color_guidance"); }
            args.options.color_guidance = luisa::string{value};
        } else if (arg == "--max-overview-pixels") {
            if (!need_value(value) || !parse_u64(value, args.options.max_overview_pixels)) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, "--max-overview-pixels requires a non-negative integer", "max_overview_pixels");
            }
        } else if (arg == "--max-crop-pixels") {
            if (!need_value(value) || !parse_u64(value, args.options.max_crop_pixels)) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, "--max-crop-pixels requires a non-negative integer", "max_crop_pixels");
            }
        } else if (arg == "--max-pixels") {
            if (!need_value(value) || !parse_u64(value, args.options.max_pixels)) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, "--max-pixels requires a non-negative integer", "max_pixels");
            }
        } else if (arg == "--max-patches-per-vision-batch") {
            if (!need_value(value) || !parse_u32(value, args.options.max_patches_per_vision_batch)) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, "--max-patches-per-vision-batch requires a uint32", "max_patches_per_vision_batch");
            }
        } else if (arg == "--max-regions-per-batch") {
            if (!need_value(value) || !parse_u32(value, args.options.max_regions_per_batch)) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, "--max-regions-per-batch requires a uint32", "max_regions_per_batch");
            }
        } else if (arg == "--max-decode-batch") {
            if (!need_value(value) || !parse_u32(value, args.options.max_decode_batch)) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, "--max-decode-batch requires a uint32", "max_decode_batch");
            }
        } else if (arg == "--max-output-tokens-per-region") {
            if (!need_value(value) || !parse_u32(value, args.options.max_output_tokens_per_region)) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, "--max-output-tokens-per-region requires a uint32", "max_output_tokens_per_region");
            }
        } else if (arg == "--max-output-tokens-per-page") {
            if (!need_value(value) || !parse_u32(value, args.options.max_output_tokens_per_page)) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, "--max-output-tokens-per-page requires a uint32", "max_output_tokens_per_page");
            }
        } else if (arg == "--temperature") {
            double temperature = 0.0;
            if (!need_value(value) || !parse_double(value, temperature) || temperature < 0.0) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, "--temperature requires a finite number >= 0", "temperature");
            }
            args.options.sampling.temperature = static_cast<float>(temperature);
            temperature_set = true;
        } else if (arg == "--top-p") {
            double top_p = 0.0;
            if (!need_value(value) || !parse_double(value, top_p)) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, "--top-p requires a finite number", "top_p");
            }
            args.options.sampling.top_p = static_cast<float>(top_p);
        } else if (arg == "--top-k") {
            if (!need_value(value) || !parse_u32(value, args.options.sampling.top_k)) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, "--top-k requires a uint32", "top_k");
            }
        } else if (arg == "--seed") {
            if (!need_value(value) || !parse_u64(value, args.options.sampling.seed)) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, "--seed requires a uint64", "seed");
            }
        } else if (arg == "--weight-precision" || arg == "--activation-precision" || arg == "--cache-precision") {
            if (!need_value(value)) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, luisa::string{arg} + " requires a value", "precision");
            }
            const auto precision = precision_from_name(value);
            if (!precision.has_value()) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument,
                                                luisa::string{"unknown precision '"} + luisa::string{value} + "' (automatic|fp32|fp16|bf16|int8|int4)", "precision");
            }
            if (arg == "--weight-precision") {
                args.session.precision.weights = *precision;
                args.options.precision.weights = *precision;
            } else if (arg == "--activation-precision") {
                args.session.precision.activations = *precision;
                args.options.precision.activations = *precision;
            } else {
                args.session.precision.cache = *precision;
                args.options.precision.cache = *precision;
            }
        } else if (arg == "--cache-policy") {
            if (!need_value(value)) { return Result<CliArgs>::failure(StatusCode::invalid_argument, "--cache-policy requires a value", "cache_policy"); }
            if (value == "automatic") {
                args.options.cache_policy = CachePolicy::automatic;
            } else if (value == "contiguous") {
                args.options.cache_policy = CachePolicy::contiguous;
            } else if (value == "paged") {
                args.options.cache_policy = CachePolicy::paged;
            } else {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, "unknown cache policy (automatic|contiguous|paged)", "cache_policy");
            }
        } else if (arg == "--offload") {
            if (!need_value(value)) { return Result<CliArgs>::failure(StatusCode::invalid_argument, "--offload requires a value", "offload"); }
            if (value == "disabled") {
                args.options.offload_policy = OffloadPolicy::disabled;
            } else if (value == "explicit-host") {
                args.options.offload_policy = OffloadPolicy::explicit_host;
            } else {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, "unknown offload policy (disabled|explicit-host)", "offload");
            }
        } else if (arg == "--graph" || arg == "--fast-path") {
            if (!need_value(value)) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, luisa::string{arg} + " requires a value", arg == "--graph" ? "graph" : "fast_path");
            }
            const auto set = [&value](auto &policy) -> bool {
                using P = std::remove_reference_t<decltype(policy)>;
                if (value == "disabled") {
                    policy = P::disabled;
                } else if (value == "automatic") {
                    policy = P::automatic;
                } else if (value == "enabled") {
                    policy = P::enabled;
                } else {
                    return false;
                }
                return true;
            };
            const bool ok = arg == "--graph" ? set(args.options.graph_policy) : set(args.options.fast_path_policy);
            if (!ok) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument,
                                                luisa::string{arg} + " expects disabled|automatic|enabled",
                                                arg == "--graph" ? "graph" : "fast_path");
            }
        } else if (arg == "--max-device-bytes") {
            if (!need_value(value) || !parse_u64(value, args.options.memory.max_device_bytes)) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, "--max-device-bytes requires a uint64", "max_device_bytes");
            }
        } else if (arg == "--max-host-bytes") {
            if (!need_value(value) || !parse_u64(value, args.options.memory.max_host_bytes)) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, "--max-host-bytes requires a uint64", "max_host_bytes");
            }
        } else if (arg == "--max-wall-time-ms") {
            if (!need_value(value) || !parse_double(value, args.options.max_wall_time_ms) || args.options.max_wall_time_ms < 0.0) {
                return Result<CliArgs>::failure(StatusCode::invalid_argument, "--max-wall-time-ms requires a finite number >= 0", "max_wall_time_ms");
            }
        } else {
            return Result<CliArgs>::failure(StatusCode::invalid_argument,
                                            luisa::string{"unknown argument '"} + luisa::string{arg} + "' (see --help)", "argv");
        }
    }
    // Greedy is the default; an explicit positive temperature opts into
    // non-greedy sampling (validated again by validate_recognize_options).
    if (temperature_set && args.options.sampling.temperature > 0.0f) {
        args.options.sampling.greedy = false;
    }
    return Result<CliArgs>::success(std::move(args));
}

}// namespace luisa::ocr
