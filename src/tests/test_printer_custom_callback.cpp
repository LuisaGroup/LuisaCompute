#include "spdlog/sinks/stdout_color_sinks.h"

#include <memory>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/dsl/syntax.h>
#include <luisa/dsl/sugar.h>

using namespace luisa;
using namespace luisa::compute;

struct MyStruct {
    float2 a;
    uint2 b;
};

LUISA_STRUCT(MyStruct, a, b) {};

#define DEVICE_VERBOSE(FMT, ...) device_log(luisa::format("V{} [dispatch{{}}]", FMT), __VA_ARGS__, $dispatch_id)
#define DEVICE_INFO(FMT, ...) device_log(luisa::format("I{} [dispatch{{}}]", FMT), __VA_ARGS__, $dispatch_id)
#define DEVICE_WARNING(FMT, ...) device_log(luisa::format("W{} [dispatch{{}}]", FMT), __VA_ARGS__, $dispatch_id)
#define DEVICE_ERROR(FMT, ...) device_log(luisa::format("E{} [dispatch{{}}]", FMT), __VA_ARGS__, $dispatch_id)

#define DEVICE_VERBOSE_WITH_LOCATION(FMT, ...) device_log(luisa::format("V{} [{}:{}:dispatch{{}}]", FMT, __FILE__, __LINE__), __VA_ARGS__, $dispatch_id)
#define DEVICE_INFO_WITH_LOCATION(FMT, ...) device_log(luisa::format("I{} [{}:{}:dispatch{{}}]", FMT, __FILE__, __LINE__), __VA_ARGS__, $dispatch_id)
#define DEVICE_WARNING_WITH_LOCATION(FMT, ...) device_log(luisa::format("W{} [{}:{}:dispatch{{}}]", FMT, __FILE__, __LINE__), __VA_ARGS__, $dispatch_id)
#define DEVICE_ERROR_WITH_LOCATION(FMT, ...) device_log(luisa::format("E{} [{}:{}:dispatch{{}}]", FMT, __FILE__, __LINE__), __VA_ARGS__, $dispatch_id)

int main(int argc, char *argv[]) {

    log_level_verbose();

    Context context{argv[0]};
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>. <backend>: cuda, dx, cpu, metal", argv[0]);
        exit(1);
    }
    Device device = context.create_device(argv[1]);

    Kernel2D kernel = [&]() noexcept {
        UInt2 coord = dispatch_id().xy();
        $if (coord.x == 1) {
            DEVICE_VERBOSE_WITH_LOCATION("hello {} {}", coord, make_float3x3());
        };
        $if (coord.x == coord.y) {
            Float2 v = make_float2(coord) / make_float2(dispatch_size().xy());
            Var<MyStruct> s;
            s.a = v;
            s.b = coord;
            $outline {
                DEVICE_INFO("s = {}", s);
            };
        };
        $if (coord.y == 11) {
            DEVICE_WARNING("u64_max = {}", std::numeric_limits<uint64_t>::max());
        };
    };
    auto shader = device.compile(kernel);
    Stream stream = device.create_stream();
    spdlog::logger logger{"device", std::make_shared<spdlog::sinks::stdout_color_sink_mt>()};
    logger.set_level(spdlog::level::debug);
    stream.set_log_callback([logger](luisa::string_view message) mutable noexcept {
        if (!message.empty()) {
            switch (message.front()) {
                case 'V': logger.debug("{}", message.substr(1)); break;
                case 'I': logger.info("{}", message.substr(1)); break;
                case 'W': logger.warn("{}", message.substr(1)); break;
                case 'E': logger.error("{}", message.substr(1)); break;
                default: logger.debug("[unknown] {}", message); break;
            }
        }
    });
    stream << shader().dispatch(128u, 128u)
           << synchronize();
}
