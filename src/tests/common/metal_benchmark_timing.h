#pragma once

// Benchmark-only C ABI shared by native executables and the Python/ctypes
// driver. Call begin AFTER draining the device, run the measured workload,
// synchronize it, then end. No hooks are installed outside that interval.
// This is not a Runtime API and must not be enabled during host-wall samples.
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LuisaMetalTimingResult {
    double compute_ns;
    double compute_span_ns;
    double command_buffer_ns;
    double calibration_cpu_ns;
    double calibration_gpu_ticks;
    uint64_t compute_passes;
    uint64_t command_buffers;
} LuisaMetalTimingResult;

int luisa_metal_timing_version(void);
int luisa_metal_timing_begin(uint32_t max_compute_passes);
// Control: collect completed command-buffer GPUStartTime/GPUEndTime only.
// No encoder hooks or GPU counter attachments; compute/calibration fields
// in the result remain zero. Use the same workload/batch to audit probe cost.
int luisa_metal_timing_begin_control(void);
int luisa_metal_timing_end(LuisaMetalTimingResult *result);
const char *luisa_metal_timing_error(void);

#ifdef __cplusplus
}
#endif
