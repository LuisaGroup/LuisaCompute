# CTest WILL_FAIL does not invert signal-based failures. Require the precise
# fatal diagnostic so unrelated crashes or an old C++ exception cannot pass.
if (NOT EXISTS "${TEST_PROGRAM}" OR NOT TEST_CASE MATCHES "^(mutable|const)$")
    message(FATAL_ERROR "Invalid map::at rejection test configuration")
endif ()
execute_process(COMMAND "${TEST_PROGRAM}" "--reject-map-at-${TEST_CASE}"
        RESULT_VARIABLE _result OUTPUT_VARIABLE _stdout ERROR_VARIABLE _stderr
        TIMEOUT 30)
if (_result STREQUAL "0" OR _result MATCHES "[Tt]imeout" OR
        NOT "${_stdout}${_stderr}" MATCHES "luisa::map::at: key not found")
    message(FATAL_ERROR
            "Expected map::at ${TEST_CASE} rejection; result=${_result}\n${_stdout}${_stderr}")
endif ()
message(STATUS "Verified map::at ${TEST_CASE} rejection: ${_result}")
