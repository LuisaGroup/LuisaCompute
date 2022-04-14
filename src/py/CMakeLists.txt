pybind11_add_module(lcapi lcapi.cpp)
target_link_libraries(lcapi PRIVATE luisa::compute)
add_custom_target(lc-py-copy-test ALL
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${CMAKE_CURRENT_SOURCE_DIR}/test.py"
                    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/")

add_custom_target(lc-py-copy-script ALL
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${CMAKE_CURRENT_SOURCE_DIR}/luisa"
                    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/luisa")