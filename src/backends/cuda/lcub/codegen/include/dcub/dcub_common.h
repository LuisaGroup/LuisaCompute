#pragma once

#ifdef _MSC_VER
#ifdef DCUB_DLL_EXPORTS
#define DCUB_API __declspec(dllexport)
#else
#define DCUB_API __declspec(dllimport)
#endif
#else
#define DCUB_API
#endif

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "dcub_utils.h"

// c++17 workaround
namespace luisa { namespace compute { namespace cuda { namespace dcub {}}}}
