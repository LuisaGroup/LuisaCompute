#include <Metal/Metal.h>
#include <luisa/core/logging.h>

LUISA_EXTERN_C void luisa_compute_metal_stream_print_function_logs(id<MTLLogContainer> logs) {
    if (logs != nullptr) {
        for (id<MTLFunctionLog> log in logs) {
            LUISA_INFO("[MTLFunctionLog] {}", log.debugDescription.UTF8String);
        }
    }
}
