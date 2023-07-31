#pragma once
#include <cublas_v2.h>
#include <luisa/core/logging.h>

// cublas API error checking
#define LUISA_CHECK_CUBLAS(err)                                                            \
  do {                                                                                     \
    cublasStatus_t err_ = (err);                                                           \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                                   \
      LUISA_ERROR_WITH_LOCATION(                                                           \
          "cublas error {} : {}", cublasGetStatusName(err_), cublasGetStatusString(err_)); \
    }                                                                                      \
  } while (0)