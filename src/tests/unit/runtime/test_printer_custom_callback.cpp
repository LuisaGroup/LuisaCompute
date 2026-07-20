// Deterministic custom stream-log-callback test.
//
// Device messages carry a one-byte severity prefix. The callback classifies
// them and the test checks exact payloads (plus stable portions of a source
// location), proving that callback delivery and user-side routing both work.

#include "ut/ut.hpp"
#include "test_device.h"

#include <array>
#include <limits>
#include <memory>
#include <mutex>
#include <string_view>

#include <luisa/luisa-compute.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

#define DEVICE_VERBOSE(FMT, ...) device_log(luisa::format("V{} [dispatch{{}}]", FMT), __VA_ARGS__, $dispatch_id)
#define DEVICE_INFO(FMT, ...) device_log(luisa::format("I{} [dispatch{{}}]", FMT), __VA_ARGS__, $dispatch_id)
#define DEVICE_WARNING(FMT, ...) device_log(luisa::format("W{} [dispatch{{}}]", FMT), __VA_ARGS__, $dispatch_id)
#define DEVICE_ERROR(FMT, ...) device_log(luisa::format("E{} [dispatch{{}}]", FMT), __VA_ARGS__, $dispatch_id)
#define DEVICE_VERBOSE_WITH_LOCATION(FMT, ...) device_log(luisa::format("V{} [{}:{}:dispatch{{}}]", FMT, __FILE__, __LINE__), __VA_ARGS__, $dispatch_id)

namespace {

enum Severity : size_t {
    verbose,
    info,
    warning,
    error,
    severity_count
};

struct CapturedMessages {
    std::mutex mutex;
    std::array<luisa::vector<luisa::string>, severity_count> by_severity;
    luisa::vector<luisa::string> unknown;
};

}// namespace

void test_printer_custom_callback(Device &device) {
    auto captured = std::make_shared<CapturedMessages>();
    auto stream = device.create_stream();
    stream.set_log_callback([captured](luisa::string_view message) noexcept {
        std::scoped_lock lock{captured->mutex};
        if (message.empty()) {
            captured->unknown.emplace_back("<empty>");
            return;
        }
        auto payload = message.substr(1u);
        switch (message.front()) {
            case 'V': captured->by_severity[verbose].emplace_back(payload); break;
            case 'I': captured->by_severity[info].emplace_back(payload); break;
            case 'W': captured->by_severity[warning].emplace_back(payload); break;
            case 'E': captured->by_severity[error].emplace_back(payload); break;
            default: captured->unknown.emplace_back(message); break;
        }
    });

    Kernel1D kernel = []() noexcept {
        DEVICE_VERBOSE_WITH_LOCATION("location_value={}", 17u);
        DEVICE_INFO("info_value={}", 42);
        DEVICE_WARNING("u64_max={}", std::numeric_limits<uint64_t>::max());
        DEVICE_ERROR("flag={}", dispatch_x() == 0u);
    };
    auto shader = device.compile(kernel);
    stream << shader().dispatch(1u)
           << synchronize();

    std::array<luisa::vector<luisa::string>, severity_count> messages;
    luisa::vector<luisa::string> unknown;
    {
        std::scoped_lock lock{captured->mutex};
        messages = captured->by_severity;
        unknown = captured->unknown;
    }

    expect(unknown.empty())
        << "the callback must not receive empty or unclassified printer messages";
    for (auto severity = 0u; severity < severity_count; severity++) {
        expect(static_cast<bool>(messages[severity].size() == 1u))
            << "the callback must receive exactly one message of each severity";
    }

    if (messages[info].size() == 1u) {
        expect(static_cast<bool>(
            messages[info][0] == "info_value=42 [dispatch(0, 0, 0)]"))
            << "the info callback payload must preserve its scalar and dispatch ID";
    }
    if (messages[warning].size() == 1u) {
        expect(static_cast<bool>(
            messages[warning][0] ==
            "u64_max=18446744073709551615 [dispatch(0, 0, 0)]"))
            << "the warning callback payload must preserve a 64-bit unsigned value";
    }
    if (messages[error].size() == 1u) {
        expect(static_cast<bool>(
            messages[error][0] == "flag=true [dispatch(0, 0, 0)]"))
            << "the error callback payload must preserve a boolean value";
    }
    if (messages[verbose].size() == 1u) {
        auto payload = std::string_view{messages[verbose][0]};
        expect(payload.starts_with("location_value=17 ["))
            << "the verbose callback payload must preserve its scalar prefix";
        expect(payload.find("test_printer_custom_callback.cpp:") !=
               std::string_view::npos)
            << "the location-aware message must name its source file";
        expect(payload.ends_with(":dispatch(0, 0, 0)]"))
            << "the location-aware message must preserve its dispatch ID";
    }
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) {
        return 0;
    }
    auto &device = dc->device;
    test_printer_custom_callback(device);
}
