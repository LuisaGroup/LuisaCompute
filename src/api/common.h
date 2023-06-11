#pragma once

#include <core/dll_export.h>

#ifdef __cplusplus
#include <cstdint>
#include <cstddef>
#else
#include <stdint.h>
#include <stddef.h>
#endif

#include <rust/api_types.h>

#ifndef LUISA_COMPUTE_RUST_BINDGEN
#include <rust/ir_common.h>
#endif

#ifdef __cplusplus
#include <rust/api_types.hpp>
#include <rust/ir.hpp>
#endif
