#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

namespace lc::vk {

struct TimelineCmdInfo {
    uint32_t tag{0u};
    uint64_t handle{0u};
    uint64_t dst_handle{0u};
    uint32_t dx{0u};
    uint32_t dy{0u};
    uint32_t dz{0u};
    uint64_t bytes{0u};
};

struct TimelineLayerRecord {
    uint32_t layer_idx{0u};
    uint32_t q_begin{0xffffffffu};
    uint32_t q_end{0xffffffffu};
    luisa::vector<TimelineCmdInfo> cmds;
};

struct TimelineSubmitRecord {
    struct ResolvedLayer {
        uint32_t layer_idx{0u};
        uint64_t gpu_begin_ticks{0u};
        uint64_t gpu_end_ticks{0u};
        bool valid{false};
        luisa::vector<TimelineCmdInfo> cmds;
    };

    uint64_t submit_seq{0u};
    uint64_t stream_handle{0u};
    uint32_t stream_tag{0u};
    uint64_t fence{0u};
    uint64_t cpu_record_begin_ns{0u};
    uint64_t cpu_submit_begin_ns{0u};
    uint64_t cpu_submit_end_ns{0u};
    uint64_t cpu_fence_complete_ns{0u};
    uint32_t ts_bits{64u};
    uint64_t gpu_cb_begin_ticks{0u};
    uint64_t gpu_cb_end_ticks{0u};
    bool cb_valid{false};
    uint32_t cmd_count{0u};
    uint32_t layer_count{0u};
    uint32_t truncated_spans{0u};
    luisa::vector<ResolvedLayer> layers;
};

class VkGpuTimeline {
private:
    struct TimelineMarker {
        uint64_t submit_seq{0u};
        uint64_t cpu_ns{0u};
        std::string tag;
    };

    struct TimelineLabel {
        std::string kind;
        uint64_t handle{0u};
        std::string name;
    };

    std::string _path;
    std::atomic<uint64_t> _submit_seq{0u};
    std::mutex _mtx;
    bool _has_header{false};
    bool _dirty{false};
    std::string _device_name;
    double _timestamp_period_ns{0.0};
    uint32_t _graphics_ts_bits{0u};
    uint32_t _compute_ts_bits{0u};
    uint32_t _copy_ts_bits{0u};
    luisa::vector<TimelineLabel> _labels;
    luisa::vector<TimelineMarker> _markers;
    luisa::vector<TimelineSubmitRecord> _submits;

    static void _append_json_escaped(std::string &out, std::string_view s) {
        out.push_back('"');
        for (unsigned char c : s) {
            switch (c) {
                case '"': out.append("\\\""); break;
                case '\\': out.append("\\\\"); break;
                case '\b': out.append("\\b"); break;
                case '\f': out.append("\\f"); break;
                case '\n': out.append("\\n"); break;
                case '\r': out.append("\\r"); break;
                case '\t': out.append("\\t"); break;
                default:
                    if (c < 0x20u) {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(c));
                        out.append(buf);
                    } else {
                        out.push_back(static_cast<char>(c));
                    }
                    break;
            }
        }
        out.push_back('"');
    }

public:
    explicit VkGpuTimeline(std::string path) noexcept
        : _path{std::move(path)} {}

    ~VkGpuTimeline() noexcept {
        flush();
    }

    VkGpuTimeline(const VkGpuTimeline &) = delete;
    VkGpuTimeline &operator=(const VkGpuTimeline &) = delete;

    [[nodiscard]] static uint64_t steady_ns() noexcept {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch())
                .count());
    }

    [[nodiscard]] uint64_t alloc_submit_seq() noexcept {
        return _submit_seq.fetch_add(1u, std::memory_order_acq_rel) + 1u;
    }

    [[nodiscard]] uint64_t last_submit_seq() const noexcept {
        return _submit_seq.load(std::memory_order_acquire);
    }

    void record_header_if_needed(
        luisa::string_view device_name,
        float timestamp_period_ns,
        uint32_t graphics_ts_bits,
        uint32_t compute_ts_bits,
        uint32_t copy_ts_bits) noexcept {
        std::lock_guard lock{_mtx};
        if (_has_header) {
            return;
        }
        _has_header = true;
        _device_name.assign(device_name.data(), device_name.size());
        _timestamp_period_ns = static_cast<double>(timestamp_period_ns);
        _graphics_ts_bits = graphics_ts_bits;
        _compute_ts_bits = compute_ts_bits;
        _copy_ts_bits = copy_ts_bits;
        _dirty = true;
    }

    void add_label(
        luisa::string_view kind,
        uint64_t handle,
        luisa::string_view name) noexcept {
        if (handle == 0u || name.empty()) {
            return;
        }
        std::lock_guard lock{_mtx};
        std::string_view kind_sv{kind.data(), kind.size()};
        std::string_view name_sv{name.data(), name.size()};
        for (auto &lbl : _labels) {
            if (lbl.handle == handle && lbl.kind == kind_sv) {
                if (lbl.name != name_sv) {
                    lbl.name.assign(name_sv);
                    _dirty = true;
                }
                return;
            }
        }
        _labels.emplace_back(TimelineLabel{
            .kind = std::string{kind_sv},
            .handle = handle,
            .name = std::string{name_sv}});
        _dirty = true;
    }

    void add_marker(luisa::string_view tag) noexcept {
        std::lock_guard lock{_mtx};
        _markers.emplace_back(TimelineMarker{
            .submit_seq = last_submit_seq(),
            .cpu_ns = steady_ns(),
            .tag = std::string{tag.data(), tag.size()}});
        _dirty = true;
    }

    void record_submit(TimelineSubmitRecord &&rec) noexcept {
        std::lock_guard lock{_mtx};
        _submits.emplace_back(std::move(rec));
        _dirty = true;
    }

    void flush() noexcept {
        std::lock_guard lock{_mtx};
        if (_path.empty() || !_dirty) {
            return;
        }
        std::error_code ec;
        std::filesystem::path p{_path};
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path(), ec);
        }
        std::ofstream ofs{_path, std::ios::binary | std::ios::trunc};
        if (!ofs.is_open()) {
            return;
        }

        std::string buf;
        buf.reserve(4096u);
        char numbuf[1024];

        if (_has_header) {
            buf.clear();
            buf.append("{\"type\":\"header\",\"device\":");
            _append_json_escaped(buf, _device_name);
            std::snprintf(
                numbuf, sizeof(numbuf),
                ",\"timestamp_period_ns\":%.9g,\"graphics_ts_bits\":%u,"
                "\"compute_ts_bits\":%u,\"copy_ts_bits\":%u,"
                "\"cpu_clock\":\"steady_clock_ns\"}\n",
                _timestamp_period_ns,
                _graphics_ts_bits,
                _compute_ts_bits,
                _copy_ts_bits);
            buf.append(numbuf);
            ofs.write(buf.data(), static_cast<std::streamsize>(buf.size()));
        }

        for (const auto &lbl : _labels) {
            buf.clear();
            buf.append("{\"type\":\"label\",\"kind\":");
            _append_json_escaped(buf, lbl.kind);
            std::snprintf(
                numbuf, sizeof(numbuf), ",\"handle\":%llu,\"name\":",
                static_cast<unsigned long long>(lbl.handle));
            buf.append(numbuf);
            _append_json_escaped(buf, lbl.name);
            buf.append("}\n");
            ofs.write(buf.data(), static_cast<std::streamsize>(buf.size()));
        }

        for (const auto &m : _markers) {
            buf.clear();
            std::snprintf(
                numbuf, sizeof(numbuf),
                "{\"type\":\"marker\",\"submit_seq\":%llu,\"cpu_ns\":%llu,\"tag\":",
                static_cast<unsigned long long>(m.submit_seq),
                static_cast<unsigned long long>(m.cpu_ns));
            buf.append(numbuf);
            _append_json_escaped(buf, m.tag);
            buf.append("}\n");
            ofs.write(buf.data(), static_cast<std::streamsize>(buf.size()));
        }

        std::sort(_submits.begin(), _submits.end(),
                  [](const TimelineSubmitRecord &a, const TimelineSubmitRecord &b) noexcept {
                      return a.submit_seq < b.submit_seq;
                  });

        for (const auto &s : _submits) {
            buf.clear();
            std::snprintf(
                numbuf, sizeof(numbuf),
                "{\"type\":\"submit\",\"submit_seq\":%llu,\"stream_handle\":%llu,"
                "\"stream_tag\":%u,\"fence\":%llu,"
                "\"cpu_record_begin_ns\":%llu,\"cpu_submit_begin_ns\":%llu,"
                "\"cpu_submit_end_ns\":%llu,\"cpu_fence_complete_ns\":%llu,"
                "\"ts_bits\":%u,\"cb_begin\":%llu,\"cb_end\":%llu,\"cb_valid\":%s,"
                "\"cmd_count\":%u,\"layer_count\":%u,\"truncated_spans\":%u,\"layers\":[",
                static_cast<unsigned long long>(s.submit_seq),
                static_cast<unsigned long long>(s.stream_handle),
                s.stream_tag,
                static_cast<unsigned long long>(s.fence),
                static_cast<unsigned long long>(s.cpu_record_begin_ns),
                static_cast<unsigned long long>(s.cpu_submit_begin_ns),
                static_cast<unsigned long long>(s.cpu_submit_end_ns),
                static_cast<unsigned long long>(s.cpu_fence_complete_ns),
                s.ts_bits,
                static_cast<unsigned long long>(s.gpu_cb_begin_ticks),
                static_cast<unsigned long long>(s.gpu_cb_end_ticks),
                s.cb_valid ? "true" : "false",
                s.cmd_count,
                s.layer_count,
                s.truncated_spans);
            buf.append(numbuf);
            for (size_t li = 0u; li < s.layers.size(); ++li) {
                const auto &layer = s.layers[li];
                if (li != 0u) {
                    buf.push_back(',');
                }
                std::snprintf(
                    numbuf, sizeof(numbuf),
                    "{\"layer\":%u,\"valid\":%s,\"begin\":%llu,\"end\":%llu,\"cmds\":[",
                    layer.layer_idx,
                    layer.valid ? "true" : "false",
                    static_cast<unsigned long long>(layer.gpu_begin_ticks),
                    static_cast<unsigned long long>(layer.gpu_end_ticks));
                buf.append(numbuf);
                for (size_t ci = 0u; ci < layer.cmds.size(); ++ci) {
                    const auto &c = layer.cmds[ci];
                    if (ci != 0u) {
                        buf.push_back(',');
                    }
                    std::snprintf(
                        numbuf, sizeof(numbuf),
                        "{\"tag\":%u,\"handle\":%llu,\"dst_handle\":%llu,"
                        "\"dx\":%u,\"dy\":%u,\"dz\":%u,\"bytes\":%llu}",
                        c.tag,
                        static_cast<unsigned long long>(c.handle),
                        static_cast<unsigned long long>(c.dst_handle),
                        c.dx,
                        c.dy,
                        c.dz,
                        static_cast<unsigned long long>(c.bytes));
                    buf.append(numbuf);
                }
                buf.append("]}");
            }
            buf.append("]}\n");
            ofs.write(buf.data(), static_cast<std::streamsize>(buf.size()));
        }

        ofs.flush();
        _dirty = false;
    }
};

}// namespace lc::vk
