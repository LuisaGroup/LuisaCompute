#pragma once

#include <luisa/tile/collective_plan.h>
#include <luisa/tile/dsl.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <locale>
#include <string_view>

namespace row_regroup_diagnostic {
namespace work_detail {

inline void quoted(std::ostream &out, std::string_view text) {
    constexpr char digits[]{"0123456789abcdef"};
    out << '"';
    for (auto value : text) {
        auto c = static_cast<unsigned char>(value);
        if (c == '"' || c == '\\') {
            out << '\\' << static_cast<char>(c);
        } else if (c < 0x20u) {
            out << "\\u00" << digits[c >> 4u] << digits[c & 15u];
        } else {
            out << static_cast<char>(c);
        }
    }
    out << '"';
}

inline const char *kind_name(luisa::compute::tile::CollectiveKind kind) noexcept {
    using K = luisa::compute::tile::CollectiveKind;
    switch (kind) {
        case K::SUM: return "sum";
        case K::MINIMUM: return "minimum";
        case K::MAXIMUM: return "maximum";
        case K::INCLUSIVE_SUM: return "inclusive_sum";
    }
    return "unknown";
}

inline const char *element_name(luisa::compute::tile::ScalarType element) noexcept {
    using E = luisa::compute::tile::ScalarType;
    switch (element) {
        case E::INVALID: return "invalid";
        case E::BOOL: return "bool";
        case E::INT8: return "int8";
        case E::UINT8: return "uint8";
        case E::INT16: return "int16";
        case E::UINT16: return "uint16";
        case E::INT32: return "int32";
        case E::UINT32: return "uint32";
        case E::INT64: return "int64";
        case E::UINT64: return "uint64";
        case E::FLOAT8_E4M3FN: return "float8_e4m3fn";
        case E::FLOAT8_E5M2: return "float8_e5m2";
        case E::BFLOAT16: return "bfloat16";
        case E::FLOAT16: return "float16";
        case E::FLOAT32: return "float32";
        case E::FLOAT64: return "float64";
    }
    return "unknown";
}

}// namespace work_detail

// Independent, cold diagnostic of the actual captured function. This does not
// infer facts by scaling the baseline row block and does not affect eligibility.
// A true return value means the complete JSON was written, not analysis success.
inline bool export_work(const luisa::compute::tile::Kernel &kernel,
                        const std::filesystem::path &directory) {
    using Clock = std::chrono::steady_clock;
    auto valid = kernel.valid();
    luisa::compute::tile::CollectiveWorkAnalysis analysis;
    auto start = Clock::now();
    if (valid) {
        analysis = luisa::compute::tile::analyze_collective_work(kernel.function());
    }
    auto milliseconds = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    auto complete = valid && analysis.ok();
    std::ofstream out{directory / "diagnostic-collective-work.json", std::ios::binary};
    if (!out) { return false; }
    out.imbue(std::locale::classic());
    out << std::setprecision(17)
        << "{\n  \"schema_version\": 1,\n"
        << "  \"analysis\": \"analyze_collective_work\",\n"
        << "  \"source\": \"actual_captured_function\",\n"
        << "  \"facts_kind\": \"logical_ir_only\",\n"
        << "  \"analysis_outside_benchmark_timing\": true,\n"
        << "  \"analysis_wall_ms\": " << milliseconds << ",\n"
        << "  \"ok\": " << (complete ? "true" : "false") << ",\n"
        << "  \"facts_complete\": " << (complete ? "true" : "false") << ",\n"
        << "  \"status\": \"" << (complete ? "complete" : valid ? "unsupported" : "capture_invalid") << "\",\n"
        << "  \"error\": ";
    if (complete) {
        out << "null";
    } else if (!valid) {
        work_detail::quoted(out, "invalid captured kernel");
    } else if (analysis.error.empty()) {
        work_detail::quoted(out, "analysis did not admit any collectives");
    } else {
        work_detail::quoted(out, {analysis.error.data(), analysis.error.size()});
    }
    out << ",\n  \"facts\": ";
    if (!complete) {
        // Partial/default counters from a rejected analysis are not known facts.
        out << "null\n";
    } else {
        out << "{\n"
            << "    \"programs\": " << analysis.programs << ",\n"
            << "    \"elementwise_elements_per_program\": " << analysis.elementwise_elements_per_program << ",\n"
            << "    \"global_read_bytes_per_program\": " << analysis.global_read_bytes_per_program << ",\n"
            << "    \"global_write_bytes_per_program\": " << analysis.global_write_bytes_per_program << ",\n"
            << "    \"materialized_tile_total_bytes\": " << analysis.materialized_tile_total_bytes << ",\n"
            << "    \"materialized_tile_peak_bytes\": " << analysis.materialized_tile_peak_bytes << ",\n"
            << "    \"largest_materialized_tile_elements\": " << analysis.largest_materialized_tile_elements << ",\n"
            << "    \"collectives\": [";
        bool first = true;
        for (auto &&work : analysis.collectives) {
            if (!first) { out << ','; }
            first = false;
            out << "\n      {\"operation_id\": " << work.operation_id
                << ", \"kind\": \"" << work_detail::kind_name(work.kind)
                << "\", \"kind_enum\": " << static_cast<unsigned>(work.kind)
                << ", \"element\": \"" << work_detail::element_name(work.element)
                << "\", \"element_enum\": " << static_cast<unsigned>(work.element)
                << ", \"contribution_extent\": " << work.contribution_extent
                << ", \"independent_elements\": " << work.independent_elements
                << ", \"input_elements\": " << work.input_elements << '}';
        }
        out << "\n    ]\n  }\n";
    }
    out << "}\n";
    out.flush();
    return static_cast<bool>(out);
}

}// namespace row_regroup_diagnostic
