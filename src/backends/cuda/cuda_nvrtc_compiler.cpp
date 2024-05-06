//
// Created by mike on 24-4-2.
//

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

// ReSharper disable CppEnforceTypeAliasCodeStyle
// ReSharper disable CppInconsistentNaming
typedef unsigned int nvrtcResult;
typedef struct _nvrtcProgram *nvrtcProgram;
extern nvrtcResult nvrtcVersion(int *major, int *minor);
extern const char *nvrtcGetErrorString(nvrtcResult result);
extern nvrtcResult nvrtcCreateProgram(nvrtcProgram *prog,
                                      const char *src,
                                      const char *name,
                                      int numHeaders,
                                      const char *const *headers,
                                      const char *const *includeNames);
extern nvrtcResult nvrtcDestroyProgram(nvrtcProgram *prog);
extern nvrtcResult nvrtcCompileProgram(nvrtcProgram prog,
                                       int numOptions, const char *const *options);
extern nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t *logSizeRet);
extern nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char *log);
extern nvrtcResult nvrtcGetOptiXIRSize(nvrtcProgram prog, size_t *optixirSizeRet);
extern nvrtcResult nvrtcGetOptiXIR(nvrtcProgram prog, char *optixir);
extern nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t *ptxSizeRet);
extern nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char *ptx);
// ReSharper restore CppInconsistentNaming
// ReSharper restore CppEnforceTypeAliasCodeStyle

static void report_error(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fflush(stderr);
    abort();
}

#define LUISA_CHECK_NVRTC(...)                     \
    do {                                           \
        nvrtcResult ec = __VA_ARGS__;              \
        if (ec != 0) {                             \
            report_error("NVRTC error: %s\n",      \
                         nvrtcGetErrorString(ec)); \
        }                                          \
    } while (0)

typedef struct str_buffer {
    size_t size;
    char *data;
} str_buffer;

static str_buffer read(const char *name) {
    char size_str[16] = {};
    if (fread(size_str, 1, 16, stdin) != 16) {
        report_error("Failed to read %s size from stdin.\n", name);
    }
    str_buffer s = {};
    // parse size
    for (size_t i = 0u; i < 16u; i++) {
        s.size <<= 4u;
        if (size_str[i] >= '0' && size_str[i] <= '9') {
            s.size += size_str[i] - '0';
        } else if (size_str[i] >= 'a' && size_str[i] <= 'f') {
            s.size += size_str[i] - 'a' + 10;
        } else if (size_str[i] >= 'A' && size_str[i] <= 'F') {
            s.size += size_str[i] - 'A' + 10;
        } else {
            report_error("Invalid %s size format.\n", name);
        }
    }
    s.data = (char *)malloc(s.size);
    if (s.data == NULL) {
        report_error("Failed to allocate %s memory (%" PRIu64 "B).\n", name, s.size);
    }
    if (!fread(s.data, s.size, 1, stdin)) {
        free(s.data);
        report_error("Failed to read %s data from stdin.\n", name);
    }
    if (s.data[s.size - 1] != 0) {
        free(s.data);
        report_error("%s data is not null-terminated.\n", name);
    }
    return s;
}

int main(int argc, char *argv[]) {

    if (argc == 1 ||
        (argc == 2 && (strcmp(argv[1], "--version") == 0 || strcmp(argv[1], "-v") == 0))) {
        int major = 0, minor = 0;
        LUISA_CHECK_NVRTC(nvrtcVersion(&major, &minor));
        printf("%u\n", major * 10000u + minor * 100u);
        return 0;
    }

    // read filename and source code
    str_buffer filename = read("filename");
    str_buffer src = read("source code");

    // compile
    nvrtcProgram prog;
    LUISA_CHECK_NVRTC(nvrtcCreateProgram(&prog, src.data, filename.data, 0, NULL, NULL));
    int nvrtc_argc = argc - 1;
    const char *const *nvrtc_argv = argv + 1;
    nvrtcResult err = nvrtcCompileProgram(prog, nvrtc_argc, nvrtc_argv);

    // free memory
    free(filename.data);
    free(src.data);

    // print compile log
    size_t log_size = 0;
    LUISA_CHECK_NVRTC(nvrtcGetProgramLogSize(prog, &log_size));
    if (log_size > 1u) {
        char *log = (char *)malloc(log_size);
        LUISA_CHECK_NVRTC(nvrtcGetProgramLog(prog, log));
        fprintf(stderr, "Compile log:\n%s\n", log);
        free(log);
    }
    LUISA_CHECK_NVRTC(err);

    // get PTX
    str_buffer buffer = {};
    int is_optix_ir = 0;
    for (size_t i = 0u; i < nvrtc_argc; i++) {
        if (strcmp(nvrtc_argv[i], "--optix-ir") == 0 ||
            strcmp(nvrtc_argv[i], "-optix-ir") == 0) {
            is_optix_ir = 1;
            break;
        }
    }
    if (is_optix_ir) {
        LUISA_CHECK_NVRTC(nvrtcGetOptiXIRSize(prog, &buffer.size));
        buffer.data = (char *)malloc(buffer.size);
        LUISA_CHECK_NVRTC(nvrtcGetOptiXIR(prog, buffer.data));
        LUISA_CHECK_NVRTC(nvrtcDestroyProgram(&prog));
    } else {
        LUISA_CHECK_NVRTC(nvrtcGetPTXSize(prog, &buffer.size));
        buffer.data = (char *)malloc(buffer.size);
        LUISA_CHECK_NVRTC(nvrtcGetPTX(prog, buffer.data));
        LUISA_CHECK_NVRTC(nvrtcDestroyProgram(&prog));
    }

    // write PTX to stdout
    size_t written_size = fwrite(buffer.data, 1, buffer.size, stdout);
    if (written_size != buffer.size) {
#ifndef NDEBUG
        FILE *file = fopen("dump.ptx", "wb");
        fwrite(buffer.data, 1, buffer.size, file);
        fclose(file);
#endif
        free(buffer.data);
        report_error("Failed to write PTX data to stdout "
                     "(%" PRIu64 " of total %" PRIu64 "B written).\n",
                     written_size, buffer.size);
    }
    fflush(stdout);// ensure the data is written before exit
    free(buffer.data);
    return 0;
}

#ifdef __cplusplus
}
#endif
