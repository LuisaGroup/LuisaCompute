#pragma once

#include <string_view>

#include <luisa/coro/coro_frame_storage.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/string.h>
#include <luisa/dsl/coro_func.h>

namespace luisa::example {

enum class CoroSchedulerKind {
    state_machine,
    wavefront,
    persistent,
};

[[nodiscard]] inline auto parse_coro_scheduler_arg(int argc, char *argv[],
                                                   CoroSchedulerKind default_kind = CoroSchedulerKind::state_machine) noexcept {
    auto kind = default_kind;
    for (auto i = 2; i < argc; i++) {
        if (argv[i] == nullptr) { break; }
        std::string_view arg{argv[i]};
        if (arg == "--scheduler") {
            if (i + 1 >= argc || argv[i + 1] == nullptr) {
                LUISA_ERROR("Missing value for --scheduler. Expected state_machine, wavefront, or persistent.");
            }
            std::string_view value{argv[++i]};
            if (value == "state_machine" || value == "statemachine" || value == "state") {
                kind = CoroSchedulerKind::state_machine;
            } else if (value == "wavefront" || value == "wave") {
                kind = CoroSchedulerKind::wavefront;
            } else if (value == "persistent" || value == "persistent_threads") {
                kind = CoroSchedulerKind::persistent;
            } else {
                LUISA_ERROR("Unknown coroutine scheduler '{}'. Expected state_machine, wavefront, or persistent.", value);
            }
        }
    }
    return kind;
}

[[nodiscard]] inline constexpr auto coro_scheduler_name(CoroSchedulerKind kind) noexcept {
    switch (kind) {
        case CoroSchedulerKind::state_machine: return "state_machine";
        case CoroSchedulerKind::wavefront: return "wavefront";
        case CoroSchedulerKind::persistent: return "persistent";
    }
    return "unknown";
}

[[nodiscard]] inline auto coro_frame_field_name(const luisa::compute::CoroFrameDesc &desc, size_t index) noexcept {
    if (index >= desc.frame_field_count()) { return luisa::format("<invalid:{}>", index); }
    auto *type = desc.frame_field_type(index);
    return luisa::format("{}:{}:{}B", desc.frame_field_name(index), type->description(), type->size());
}

[[nodiscard]] inline auto coro_frame_field_list(const luisa::compute::CoroFrameDesc &desc,
                                                luisa::span<const size_t> fields) noexcept {
    luisa::string s;
    s.append("[");
    for (auto i = 0u; i < fields.size(); i++) {
        if (i != 0u) { s.append(", "); }
        s.append(coro_frame_field_name(desc, fields[i]));
    }
    s.append("]");
    return s;
}

template<typename Coro>
inline void dump_coro_frame_rw(const Coro &coro) noexcept {
    auto &&graph = coro.graph();
    auto &&frame = coro.frame();
    for (auto i = 0u; i < coro.subroutine_count(); i++) {
        auto &&node = graph.node(i);
        auto name = node.name.empty() ? luisa::string{"<entry>"} : node.name;
        LUISA_INFO("  subroutine {} '{}' token={} terminal={} R={} W(union)={} targets={}",
                   i, name, node.token, node.is_terminal,
                   coro_frame_field_list(frame, node.input_field_span()),
                   coro_frame_field_list(frame, node.output_field_span()),
                   node.targets);
        for (auto target : node.targets) {
            if (auto *edge = graph.edge(i, target)) {
                LUISA_INFO("    edge {} -> {} load={} store={}",
                           i, target,
                           coro_frame_field_list(frame, luisa::span{edge->load_fields}),
                           coro_frame_field_list(frame, luisa::span{edge->store_fields}));
            }
        }
    }
}

}// namespace luisa::example
