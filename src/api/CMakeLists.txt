set(LUISA_COMPUTE_API_SOURCES
        runtime.cpp runtime.h
        type.cpp type.h
        # math.cpp math.h
        logging.cpp logging.h
        language.cpp language.h)
add_library(luisa-compute-api MODULE ${LUISA_COMPUTE_API_SOURCES})
target_link_libraries(luisa-compute-api PRIVATE
        luisa-compute-ast
        luisa-compute-runtime
        luisa-compute-compile
        luisa-compute-backends)
# add_dependencies(luisa-compute-api luisa-compute-serialize)
set_target_properties(luisa-compute-api PROPERTIES
        UNITY_BUILD ${LUISA_COMPUTE_ENABLE_UNITY_BUILD})

# find_package(Python3 COMPONENTS Interpreter)
# if (Python3_Interpreter_FOUND AND Python3_VERSION VERSION_GREATER_EQUAL 3.6)
#     message(STATUS "Found suitable Python interpreter: ${Python3_VERSION}")
#     set(LUISA_COMPUTE_API_PYTHON_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/generate_python_library.py")
#     add_custom_target(luisa-compute-api-python ALL
#             COMMAND ${Python3_EXECUTABLE} ${LUISA_COMPUTE_API_PYTHON_SCRIPT}
#             DEPENDS ${LUISA_COMPUTE_API_PYTHON_SCRIPT}
#             COMMENT "Updating Python bindings")
#     add_dependencies(luisa-compute-api luisa-compute-api-python)
# endif ()
