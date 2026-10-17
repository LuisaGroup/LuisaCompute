// Debug & diagnostics harness implementation (plan §9.1 D1/D3/D4). See
// debug_utils.h for the contract, the routing table and the dump-file format.
//
// Unity-build note: file-local helpers live in embed_debug_utils_detail so a
// merged batch cannot collide with another embed/*.cpp (scripts/
// check_unity_build_conflicts.py rule).
#include "embedding/embed/debug_utils.h"

#include <luisa/core/logging.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>

#if defined(LUISA_EMBED_HAS_EXECUTOR)
#include "embedding/embed/executor.h"
#endif
#if defined(LUISA_EMBED_HAS_SESSION)
#include "embedding/embed/session.h"
#endif

namespace luisa::embed {
namespace embed_debug_utils_detail {

// D3 routed-line counters (monotonic since the last reset; relaxed order is
// enough - device-log callbacks are low-rate and tests synchronize first).
std::atomic<uint64_t> g_routed_info{0u};
std::atomic<uint64_t> g_routed_warning{0u};
std::atomic<uint64_t> g_routed_error{0u};
std::atomic<uint64_t> g_routed_other{0u};
std::atomic<uint64_t> g_routed_total{0u};

}// namespace embed_debug_utils_detail

using namespace embed_debug_utils_detail;

void install_validation_preset() noexcept {
    // src/runtime/context.cpp enables the validation layer only for a value
    // that is exactly "1"; mirror that check, warn otherwise (nothing magic -
    // we never write the environment).
    const auto env = std::getenv("LUISA_ENABLE_VALIDATION");
    if (env == nullptr) {
        LUISA_WARNING("D1 dev preset (plan §9.1) is OFF: LUISA_ENABLE_VALIDATION is unset. "
                      "Set it to \"1\" before device creation so API misuse becomes a logged "
                      "LUISA_ERROR instead of driver-side corruption.");
    } else if (luisa::string_view{env} != luisa::string_view{"1"}) {
        LUISA_WARNING("D1 dev preset (plan §9.1) is OFF: LUISA_ENABLE_VALIDATION=\"{}\"; only "
                      "the exact value \"1\" wraps the device in the validation layer.",
                      env);
    }
}

void route_stream_logs(compute::Stream &stream) noexcept {
    stream.set_log_callback([](luisa::string_view message) noexcept {
        g_routed_total.fetch_add(1u, std::memory_order_relaxed);
        if (message.empty()) [[unlikely]] {
            g_routed_other.fetch_add(1u, std::memory_order_relaxed);
            luisa::log_verbose("embed device: (empty line)");
            return;
        }
        switch (message.front()) {
            case 'I': {
                g_routed_info.fetch_add(1u, std::memory_order_relaxed);
                luisa::log_info("embed device: {}", message.substr(1u));
            } break;
            case 'W': {
                g_routed_warning.fetch_add(1u, std::memory_order_relaxed);
                luisa::log_warning("embed device: {}", message.substr(1u));
            } break;
            case 'E': {
                // NEVER route to log_error: it is [[noreturn]] + abort (plan §9.1 D3).
                g_routed_error.fetch_add(1u, std::memory_order_relaxed);
                luisa::log_warning("embed device error: {}", message.substr(1u));
            } break;
            default: {
                g_routed_other.fetch_add(1u, std::memory_order_relaxed);
                luisa::log_verbose("embed device: {}", message);
            } break;
        }
    });
}

uint64_t routed_info_count() noexcept { return g_routed_info.load(std::memory_order_relaxed); }
uint64_t routed_warning_count() noexcept { return g_routed_warning.load(std::memory_order_relaxed); }
uint64_t routed_error_count() noexcept { return g_routed_error.load(std::memory_order_relaxed); }
uint64_t routed_other_count() noexcept { return g_routed_other.load(std::memory_order_relaxed); }
uint64_t routed_total_count() noexcept { return g_routed_total.load(std::memory_order_relaxed); }

void reset_routed_counts() noexcept {
    g_routed_info.store(0u, std::memory_order_relaxed);
    g_routed_warning.store(0u, std::memory_order_relaxed);
    g_routed_error.store(0u, std::memory_order_relaxed);
    g_routed_other.store(0u, std::memory_order_relaxed);
    g_routed_total.store(0u, std::memory_order_relaxed);
}

bool dump_requested_from_env(uint32_t &layer) noexcept {
    const auto env = std::getenv("BGEM3_DUMP_LAYER");
    if (env == nullptr || env[0] == '\0') return false;
    const auto text = luisa::string_view{env};
    uint32_t value = 0u;
    for (const auto c : text) {
        if (c < '0' || c > '9') [[unlikely]] {
            LUISA_WARNING("BGEM3_DUMP_LAYER=\"{}\" is not a decimal layer index; ignoring.",
                          text);
            return false;
        }
        const auto digit = static_cast<uint32_t>(c - '0');
        if (value > (0xFFFFFFFFu - digit) / 10u) [[unlikely]] {
            LUISA_WARNING("BGEM3_DUMP_LAYER=\"{}\" overflows uint32_t; ignoring.", text);
            return false;
        }
        value = value * 10u + digit;
    }
    layer = value;
    return true;
}

#if defined(LUISA_EMBED_HAS_EXECUTOR)
namespace embed_debug_utils_detail {

struct LayerDumpEntry {
    uint32_t layer;
    std::filesystem::path out_dir;
};

std::mutex g_dump_mutex;
luisa::vector<LayerDumpEntry> g_dump_entries;
compute::Device *g_dump_device = nullptr;// caller-owned, outlives session usage
std::atomic<uint64_t> g_dump_completed{0u};

// Writes <dir>/layer<layer>.bin (raw fp32) + <dir>/layer<layer>.json. ASCII
// out-dirs only (luisa::to_string is the never-throwing narrow conversion;
// names are assembled as narrow strings so no throwing path conversions are
// involved).
void write_layer_dump_file(const std::filesystem::path &dir, uint32_t layer,
                           uint32_t rows, uint32_t cols,
                           luisa::span<const float> data) noexcept {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        LUISA_WARNING("layer dump: cannot create {}: {}", luisa::to_string(dir), ec.message());
        return;
    }
    const auto dir_name = luisa::to_string(dir);
    const auto sep = (dir_name.empty() || dir_name.back() == '/' || dir_name.back() == '\\') ? "" : "/";
    const auto bin_name = luisa::format("{}{}layer{}.bin", dir_name, sep, layer);
    const auto json_name = luisa::format("{}{}layer{}.json", dir_name, sep, layer);

    if (auto *bin = std::fopen(bin_name.c_str(), "wb"); bin != nullptr) {
        static_cast<void>(std::fwrite(data.data(), sizeof(float), data.size(), bin));
        static_cast<void>(std::fclose(bin));
    } else {
        LUISA_WARNING("layer dump: cannot open {} for writing", bin_name);
        return;
    }

    if (auto *json = std::fopen(json_name.c_str(), "wb"); json != nullptr) {
        static_cast<void>(std::fprintf(json,
                                       "{\"layer\": %u, \"rows\": %u, \"cols\": %u, "
                                       "\"count\": %zu, \"dtype\": \"float32-le\", "
                                       "\"layout\": \"row-major\"}\n",
                                       layer, rows, cols, data.size()));
        static_cast<void>(std::fclose(json));
    } else {
        LUISA_WARNING("layer dump: cannot open {} for writing", json_name);
        return;
    }
    LUISA_INFO("layer dump: wrote {} + {} ({} floats, {}x{})",
               bin_name, json_name, data.size(), rows, cols);
}

// The single trace-hook sink: consults the registry, reads the layer back on
// a private stream (the B1 contract fires this on the caller's host thread
// after the encode completed, over immutable slot buffers), and writes one
// file pair per registered out_dir for this layer.
void layer_trace_dispatch(uint32_t layer, compute::BufferView<float> view,
                          uint32_t rows, uint32_t cols) noexcept {
    luisa::vector<LayerDumpEntry> targets;
    compute::Device *device = nullptr;
    {
        std::scoped_lock lock{g_dump_mutex};
        device = g_dump_device;
        for (const auto &entry : g_dump_entries) {
            if (entry.layer == layer) targets.push_back(entry);
        }
    }
    if (targets.empty() || device == nullptr) return;

    const auto count = view.size();
    if (count == 0u) [[unlikely]] {
        LUISA_WARNING("layer dump: empty BufferView for layer {}", layer);
        return;
    }
    if (static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols) != count) {
        LUISA_WARNING("layer dump: layer {} metadata mismatch (rows {} x cols {} != view size {}); "
                      "dumping the view contents",
                      layer, rows, cols, count);
    }

    luisa::vector<float> host(count);
    auto readback = device->create_stream();
    readback << view.copy_to(luisa::span{host}) << compute::synchronize();

    for (const auto &entry : targets) {
        write_layer_dump_file(entry.out_dir, layer, rows, cols, luisa::span<const float>{host});
    }
    g_dump_completed.fetch_add(targets.size(), std::memory_order_relaxed);
}

}// namespace embed_debug_utils_detail

void install_layer_dump(Executor &executor, compute::Device &device,
                        uint32_t layer, const std::filesystem::path &out_dir) noexcept {
    {
        std::scoped_lock lock{g_dump_mutex};
        if (g_dump_device != nullptr && g_dump_device != &device) {
            LUISA_WARNING("layer dump: re-pointed at a different Device; all dumps now read back "
                          "on the newest device (one device per process is the supported setup).");
        }
        g_dump_device = &device;
        g_dump_entries.erase(std::remove_if(g_dump_entries.begin(), g_dump_entries.end(),
                                            [layer](const auto &e) noexcept { return e.layer == layer; }),
                             g_dump_entries.end());
        g_dump_entries.push_back(LayerDumpEntry{.layer = layer, .out_dir = out_dir});
    }
    // Re-arm the executor hook (B1 contract: fired by fire_trace_callbacks()
    // after the encode completed; install order vs. encode() is "next
    // encode", plan §9.1 D4).
    executor.set_layer_trace(&layer_trace_dispatch);
    LUISA_INFO("layer dump: layer {} -> {}", layer, luisa::to_string(out_dir));
}

bool maybe_install_layer_dump(Executor &executor, compute::Device &device,
                              const std::filesystem::path &out_dir) noexcept {
    uint32_t layer = 0u;
    if (!dump_requested_from_env(layer)) return false;
    install_layer_dump(executor, device, layer, out_dir);
    return true;
}

uint64_t layer_dump_completed_count() noexcept {
    return g_dump_completed.load(std::memory_order_relaxed);
}

void reset_layer_dump_completed_count() noexcept {
    g_dump_completed.store(0u, std::memory_order_relaxed);
}
#endif// LUISA_EMBED_HAS_EXECUTOR

#if defined(LUISA_EMBED_HAS_SESSION) && defined(LUISA_EMBED_HAS_EXECUTOR)
void install_layer_dump(Session &session, compute::Device &device,
                        uint32_t layer, const std::filesystem::path &out_dir) noexcept {
    {
        std::scoped_lock lock{embed_debug_utils_detail::g_dump_mutex};
        if (embed_debug_utils_detail::g_dump_device != nullptr &&
            embed_debug_utils_detail::g_dump_device != &device) {
            LUISA_WARNING("layer dump: re-pointed at a different Device; all dumps now read back "
                          "on the newest device (one device per process is the supported setup).");
        }
        embed_debug_utils_detail::g_dump_device = &device;
        auto &entries = embed_debug_utils_detail::g_dump_entries;
        entries.erase(std::remove_if(entries.begin(), entries.end(),
                                     [layer](const auto &e) noexcept { return e.layer == layer; }),
                      entries.end());
        entries.push_back(embed_debug_utils_detail::LayerDumpEntry{.layer = layer, .out_dir = out_dir});
    }
    // B1: Session::set_layer_trace broadcasts the hook to every worker
    // executor and fires callbacks synchronously after the encode completed.
    session.set_layer_trace(&embed_debug_utils_detail::layer_trace_dispatch);
    LUISA_INFO("layer dump (session): layer {} -> {}", layer, luisa::to_string(out_dir));
}

bool maybe_install_layer_dump(Session &session, compute::Device &device,
                              const std::filesystem::path &out_dir) noexcept {
    uint32_t layer = 0u;
    if (!dump_requested_from_env(layer)) return false;
    install_layer_dump(session, device, layer, out_dir);
    return true;
}
#endif// Session gate

}// namespace luisa::embed
