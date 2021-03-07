find_package(LLVM REQUIRED)

if (LLVM_FOUND)
    
    message(STATUS "Found LLVM: ${LLVM_DIR}")
    message(STATUS "Build with LLVM: ${LLVM_VERSION}")
    
    set(LUISA_COMPUTE_BACKEND_LLVM_SOURCES
        dummy.cpp dummy.h)
    luisa_compute_add_backend(llvm SOURCES ${LUISA_COMPUTE_BACKEND_LLVM_SOURCES})
    
    llvm_map_components_to_libnames(
            LLVM_LIBS
            core support mcjit executionengine
            x86asmparser x86codegen x86desc x86disassembler x86info
            irreader passes analysis)
    target_link_libraries(luisa-compute-backend-llvm PRIVATE ${LLVM_LIBS})
    target_compile_definitions(luisa-compute-backend-llvm PRIVATE ${LLVM_DEFINITIONS})
    target_include_directories(luisa-compute-backend-llvm PRIVATE ${LLVM_INCLUDE_DIRS})
endif ()
