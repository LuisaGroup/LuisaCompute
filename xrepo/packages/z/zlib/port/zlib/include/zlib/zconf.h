#pragma once
#ifdef _WIN32
#include "windows/zconf.h"
#elif 0

#else
#if defined(__APPLE__) && defined(__MACH__)
    /* Apple OSX and iOS (Darwin). */
#include <TargetConditionals.h>
    #if TARGET_OS_MAC == 1
    #include "macos/zconf.h"
    #else
    #error ZLib: "Unsupported Apple platform""
    #endif
#endif
#endif