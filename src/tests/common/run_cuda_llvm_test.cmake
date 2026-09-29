# Require both successful execution and actual assertions. New fixtures compile
# uncached kernels (or use isolated in-memory caches), so they can additionally
# prove that the CUDA LLVM compiler, rather than NVRTC, generated the shader code.
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
if (REQUIRE_COMPILE AND NOT combined_output MATCHES "Generated (PTX|OptiX IR) with CUDA LLVM CodeGen in ")
    message(FATAL_ERROR "CUDA LLVM compilation evidence is missing")
endif ()

if (REQUIRE_PTX)
    if (REQUIRE_COMPILE AND NOT combined_output MATCHES "Generated PTX with CUDA LLVM CodeGen in ")
        message(FATAL_ERROR "PTX generation evidence is missing")
    endif ()
    if (combined_output MATCHES "Generated OptiX IR with CUDA LLVM CodeGen in " OR
            combined_output MATCHES "OptiX moduleCreate: format=OPTIX_IR,")
        message(FATAL_ERROR "PTX test unexpectedly generated or consumed OptiX IR")
    endif ()
endif ()

if (REQUIRE_OPTIX_IR)
    # Uncached fixtures must prove generation as well as consumption. The
    # older fixtures may hit the cache, but must still consume binary IR.
    if (REQUIRE_COMPILE AND NOT combined_output MATCHES "Generated OptiX IR with CUDA LLVM CodeGen in ")
        message(FATAL_ERROR "OptiX IR generation evidence is missing")
    endif ()
    if (NOT combined_output MATCHES "OptiX moduleCreate: format=OPTIX_IR, bytes=[1-9][0-9]*,")
        message(FATAL_ERROR "OptiX IR module creation evidence is missing")
    endif ()
    # Non-RTX helper shaders may use PTX. An OptiX module using PTX would be a
    # partial fallback and must not satisfy this format-specific regression.
    if (combined_output MATCHES "OptiX moduleCreate: format=PTX,")
        message(FATAL_ERROR "OptiX IR test unexpectedly created a PTX ray-tracing module")
    endif ()
endif ()
