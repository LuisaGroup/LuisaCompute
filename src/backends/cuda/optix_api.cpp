#include <backends/cuda/optix_api.h>
#include <core/logging.h>
#include <core/platform.h>

// The following code is from
//   [Dr.JIT-Core](https://github.com/mitsuba-renderer/drjit-core) by Wenzel Jakob
//
// Originally licensed under BSD-3-Clause:
//
// Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>, All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifdef LUISA_PLATFORM_WINDOWS
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <windows.h>
// The cfgmgr32 header is necessary for interrogating driver information in the registry.
// For convenience the library is also linked in automatically using the #pragma command.
#include <cfgmgr32.h>
#else
#include <dlfcn.h>
#endif

namespace luisa::compute::optix {

static constexpr auto ABI_VERSION = 47u;

[[nodiscard]] void *find_optix_library() noexcept {

#ifdef LUISA_PLATFORM_WINDOWS
    void *handle = (void *)LoadLibraryA("nvoptix.dll");
    if (handle == nullptr) {
        const char *guid = "{4d36e968-e325-11ce-bfc1-08002be10318}",
                   *suffix = "nvoptix.dll",
                   *driver_name = "OpenGLDriverName";
        HKEY reg_key = nullptr;
        unsigned long size = 0,
                      flags = CM_GETIDLIST_FILTER_CLASS | CM_GETIDLIST_FILTER_PRESENT,
                      suffix_len = (unsigned long)strlen(suffix);

        if (CM_Get_Device_ID_List_SizeA(&size, guid, flags))
            return nullptr;

        std::unique_ptr<char[]> dev_names(new char[size]);
        if (CM_Get_Device_ID_ListA(guid, dev_names.get(), size, flags))
            return nullptr;

        for (char *p = dev_names.get(); *p != '\0'; p += strlen(p) + 1) {
            unsigned long node_handle = 0;
            if (CM_Locate_DevNodeA(&node_handle, p, CM_LOCATE_DEVNODE_NORMAL))
                continue;

            if (CM_Open_DevNode_Key(node_handle, KEY_QUERY_VALUE, 0,
                                    RegDisposition_OpenExisting, &reg_key,
                                    CM_REGISTRY_SOFTWARE))
                continue;

            if (RegQueryValueExA(reg_key, driver_name, 0, 0, 0, &size))
                continue;

            std::unique_ptr<char[]> path(new char[size + suffix_len]);
            if (RegQueryValueExA(reg_key, driver_name, 0, 0, (LPBYTE)path.get(), &size))
                continue;

            for (int i = (int)size - 1; i >= 0 && path[i] != '\\'; --i)
                path[i] = '\0';

            strncat(path.get(), suffix, suffix_len);
            handle = (void *)LoadLibraryA(path.get());

            if (handle) { break; }
        }
        if (reg_key != 0) { RegCloseKey(reg_key); }
    }
#else
    void *handle = dlopen("libnvoptix.so.1", RTLD_LAZY);

    if (!handle) {
        glob_t g;
        if (glob(fname, GLOB_BRACE, nullptr, &g) == 0) {
            const char *chosen = nullptr;
            if (g.gl_pathc > 1) {
                LUISA_INFO("find_optix_library(): Multiple versions of "
                           "{} were found on your system!\n",
                           fname);
                std::sort(g.gl_pathv, g.gl_pathv + g.gl_pathc,
                          [](const char *a, const char *b) {
                              while (a != nullptr && b != nullptr) {
                                  while (*a == *b && *a != '\0' && !isdigit(*a)) {
                                      ++a;
                                      ++b;
                                  }
                                  if (isdigit(*a) && isdigit(*b)) {
                                      char *ap, *bp;
                                      int ai = strtol(a, &ap, 10);
                                      int bi = strtol(b, &bp, 10);
                                      if (ai != bi)
                                          return ai < bi;
                                      a = ap;
                                      b = bp;
                                  } else {
                                      return strcmp(a, b) < 0;
                                  }
                              }
                              return false;
                          });
                uint32_t counter = 1;
                for (int j = 0; j < 2; ++j) {
                    for (size_t i = 0; i < g.gl_pathc; ++i) {
                        struct stat buf;
                        // Skip symbolic links at first
                        if (j == 0 && (lstat(g.gl_pathv[i], &buf) || S_ISLNK(buf.st_mode)))
                            continue;
                        LUISA_INFO(" {}. \"{}\"", counter++, g.gl_pathv[i]);
                        chosen = g.gl_pathv[i];
                    }
                    if (chosen)
                        break;
                }
                LUISA_INFO("\nChoosing the last one. Specify a path manually "
                           "using the environment\nvariable '{}' to override this "
                           "behavior.\n",
                           env_var);
            } else if (g.gl_pathc == 1) {
                chosen = g.gl_pathv[0];
            }
            if (chosen)
                handle = dlopen(chosen, RTLD_LAZY);
            globfree(&g);
        }
    }
#endif

    return handle;
}

[[nodiscard]] auto load_optix() noexcept {
    auto handle = find_optix_library();
    LUISA_ASSERT(handle != nullptr, "OptiX library could not be loaded.");
    using QueryFunctionTable_t =
        Result (*)(uint, uint, void *, const void **, void *, size_t);
    auto optixQueryFunctionTable = reinterpret_cast<QueryFunctionTable_t>(
        dynamic_module_find_symbol(handle, "optixQueryFunctionTable"));
    FunctionTable t{};
    if (optixQueryFunctionTable(ABI_VERSION, 0, nullptr, nullptr, &t, sizeof(t)) != 0u) {
        LUISA_ERROR_WITH_LOCATION("Failed to load OptiX function table.");
    }
    return t;
}

[[nodiscard]] const FunctionTable &api() noexcept {
    static auto table = load_optix();
    return table;
}

}// namespace luisa::compute::optix
