#pragma once

#include <string_view>

#include <luisa/core/logging.h>

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

}// namespace luisa::example
