# Require both successful execution and actual assertions. New fixtures compile
# uncached kernels (or use isolated in-memory caches), so they can additionally
# prove that the CUDA LLVM compiler, rather than NVRTC, generated the PTX.
if (NOT DEFINED TEST_EXECUTABLE)
    message(FATAL_ERROR "TEST_EXECUTABLE is required")
endif ()
execute_process(COMMAND "${TEST_EXECUTABLE}" cuda
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error
        TIMEOUT 280)
message("${output}${error}")
if (NOT result STREQUAL "0")
    message(FATAL_ERROR "CUDA LLVM test failed: ${result}")
endif ()
set(combined_output "${output}${error}")
string(ASCII 27 escape)
string(REGEX REPLACE "${escape}\\[[0-9;]*m" "" combined_output "${combined_output}")
string(TOLOWER "${combined_output}" lower_output)
if (NOT lower_output MATCHES "all tests passed \\([1-9][0-9]* asserts in [0-9]+ tests\\)")
    message(FATAL_ERROR "CUDA LLVM test ran no successful assertions")
endif ()
if (REQUIRE_COMPILE AND NOT combined_output MATCHES "Generated PTX with CUDA LLVM CodeGen in ")
    message(FATAL_ERROR "CUDA LLVM compilation evidence is missing")
endif ()
