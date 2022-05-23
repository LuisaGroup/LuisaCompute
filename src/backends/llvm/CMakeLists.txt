if (LUISA_COMPUTE_ENABLE_LLVM)
    find_package(LLVM)
endif ()

if (LLVM_FOUND) # link with llvm for jit
    set(LC_BACKEND_LLVM_SRC
            llvm_device.cpp llvm_device.h
            llvm_codegen.cpp llvm_codegen.h)
    luisa_compute_add_backend(llvm SOURCES ${LC_BACKEND_LLVM_SRC})
    message(STATUS "Build with LLVM backend")
    llvm_map_components_to_libnames(LLVM_LIBS core executionengine support mcjit nativecodegen irreader passes analysis)
    target_include_directories(luisa-compute-backend-llvm PRIVATE ${LLVM_INCLUDE_DIRS})
    target_link_libraries(luisa-compute-backend-llvm PRIVATE ${LLVM_LIBS})
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(luisa-compute-backend-llvm PRIVATE /wd4624 /wd4996) # do not complain about LLVM
    endif ()
else()
    message(WARNING "LLVM not found. LLVM backend will not be built.")
endif ()
