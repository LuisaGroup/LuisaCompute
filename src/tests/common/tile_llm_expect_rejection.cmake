# CTest WILL_FAIL does not invert signal-based failures. Run the fatal fixture
# check in a child and require its diagnostic, not merely a nonzero exit code.
if (NOT EXISTS "${TEST_PROGRAM}" OR NOT TEST_CASE MATCHES "^(heads|block)$")
    message(FATAL_ERROR "Invalid attention rejection test configuration")
endif ()
execute_process(COMMAND "${TEST_PROGRAM}" "--reject-attention-${TEST_CASE}"
        RESULT_VARIABLE _result OUTPUT_VARIABLE _stdout ERROR_VARIABLE _stderr
        TIMEOUT 30)
if (_result STREQUAL "0" OR _result MATCHES "[Tt]imeout" OR
        NOT "${_stdout}${_stderr}" MATCHES "Invalid attention shape")
    message(FATAL_ERROR
            "Expected attention ${TEST_CASE} rejection; result=${_result}\n${_stdout}${_stderr}")
endif ()
message(STATUS "Verified attention ${TEST_CASE} rejection: ${_result}")
