#pragma once
#include <cusparse.h>

#define LUISA_CHECK_CUSPARSE(func)                           \
  do {                                                       \
    cusparseStatus_t status = (func);                        \
    if (status != CUSPARSE_STATUS_SUCCESS) {                 \
      LUISA_ERROR_WITH_LOCATION(                             \
          "{}: {}", status, cusparseGetErrorString(status)); \
    }                                                        \
  } while (0)