//
// Created by Mike on 11/4/2023.
//

#include <cstdlib>
#include <cstdio>

#include <nvrtc.h>

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
char *luisa_nvrtc_compile_to_ptx(
    const char *filename, const char *src,
    const char *const *options, size_t num_options) {

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
    size_t ptx_size = 0;
    LUISA_CHECK_NVRTC(nvrtcGetPTXSize(prog, &ptx_size));
    char *ptx = (char *)malloc(ptx_size);
    LUISA_CHECK_NVRTC(nvrtcGetPTX(prog, ptx));
    LUISA_CHECK_NVRTC(nvrtcDestroyProgram(&prog));
    return ptx;
}

LUISA_CUDA_NVRTC_EXPORT
void luisa_nvrtc_free_ptx(char *ptx) {
    free(ptx);
}

LUISA_CUDA_NVRTC_EXPORT
int luisa_nvrtc_version() {
    int major = 0, minor = 0;
    LUISA_CHECK_NVRTC(nvrtcVersion(&major, &minor));
    return major * 10000 + minor * 100;
}
