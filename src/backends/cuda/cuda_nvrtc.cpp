//
// Created by Mike on 11/4/2023.
//

#include <cstdlib>
#include <cstring>
#include <cstdio>

#include <nvrtc.h>

#include "cuda_nvrtc.h"

#ifdef _MSC_VER
#define LUISA_CUDA_NVRTC_EXPORT extern "C" __declspec(dllexport)
#else
#define LUISA_CUDA_NVRTC_EXPORT extern "C" __attribute__((visibility("default")))
#endif

#define LUISA_CHECK_NVRTC(...)                   \
    do {                                         \
        nvrtcResult ec = __VA_ARGS__;            \
        if (ec != NVRTC_SUCCESS) {               \
            fprintf(stderr, "NVRTC error: %s\n", \
                    nvrtcGetErrorString(ec));    \
            abort();                             \
        }                                        \
    } while (0)

LUISA_CUDA_NVRTC_EXPORT
LUISA_NVRTC_StringBuffer luisa_nvrtc_compile(
    const char *filename, const char *src,
    const char *const *options, size_t num_options) {

#ifdef _WIN32 // work around nvrtc bug
    static auto _ = [] { return _putenv_s("LC_ALL", "POSIX"); }();
#endif

    nvrtcProgram prog;
    LUISA_CHECK_NVRTC(nvrtcCreateProgram(&prog, src, filename, 0, nullptr, nullptr));
    nvrtcResult err = nvrtcCompileProgram(prog, (int)num_options, options);
    size_t log_size = 0;
    LUISA_CHECK_NVRTC(nvrtcGetProgramLogSize(prog, &log_size));
    if (log_size > 1u) {
        char *log = (char *)malloc(log_size);
        LUISA_CHECK_NVRTC(nvrtcGetProgramLog(prog, log));
        fprintf(stderr, "Compile log:\n%s\n", log);
        free(log);
    }
    LUISA_CHECK_NVRTC(err);
    auto is_optix_ir = false;
    for (size_t i = 0u; i < num_options; i++) {
        if (strcmp(options[i], "--optix-ir") == 0 ||
            strcmp(options[i], "-optix-ir") == 0) {
            is_optix_ir = true;
            break;
        }
    }
    LUISA_NVRTC_StringBuffer buffer{};
    // if (is_optix_ir) {
    //     LUISA_CHECK_NVRTC(nvrtcGetOptiXIRSize(prog, &buffer.size));
    //     buffer.data = (char *)malloc(buffer.size);
    //     LUISA_CHECK_NVRTC(nvrtcGetOptiXIR(prog, buffer.data));
    //     LUISA_CHECK_NVRTC(nvrtcDestroyProgram(&prog));
    //     return buffer;
    // }
    LUISA_CHECK_NVRTC(nvrtcGetPTXSize(prog, &buffer.size));
    buffer.data = (char *)malloc(buffer.size);
    LUISA_CHECK_NVRTC(nvrtcGetPTX(prog, buffer.data));
    LUISA_CHECK_NVRTC(nvrtcDestroyProgram(&prog));
    return buffer;
}

LUISA_CUDA_NVRTC_EXPORT
void luisa_nvrtc_free(LUISA_NVRTC_StringBuffer buffer) {
    free(buffer.data);
}

LUISA_CUDA_NVRTC_EXPORT
int luisa_nvrtc_version() {
    int major = 0, minor = 0;
    LUISA_CHECK_NVRTC(nvrtcVersion(&major, &minor));
    return major * 10000 + minor * 100;
}
