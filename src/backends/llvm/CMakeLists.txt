if (LUISA_COMPUTE_ENABLE_LLVM)
    find_package(LLVM)
endif ()

if (LLVM_FOUND) # link with llvm for jit
    set(LC_BACKEND_LLVM_SRC
            llvm_abi.h
            llvm_device.cpp llvm_device.h
            llvm_codegen.cpp llvm_codegen.h
            llvm_codegen_value.cpp
            llvm_codegen_builtin.cpp
            llvm_codegen_type.cpp
            llvm_codegen_expr.cpp
            llvm_codegen_func.cpp
            llvm_codegen_stmt.cpp
            llvm_event.cpp llvm_event.h
            llvm_shader.cpp llvm_shader.h
            llvm_stream.cpp llvm_stream.h
            llvm_texture.cpp llvm_texture.h
            llvm_mesh.cpp llvm_mesh.h
            llvm_accel.cpp llvm_accel.h
            llvm_bindless_array.cpp llvm_bindless_array.h)
    luisa_compute_add_backend(llvm SOURCES ${LC_BACKEND_LLVM_SRC})
    message(STATUS "Build with LLVM backend")
    llvm_map_components_to_libnames(LLVM_LIBS core executionengine support mcjit nativecodegen irreader passes analysis coroutines)
    target_include_directories(luisa-compute-backend-llvm PRIVATE ${LLVM_INCLUDE_DIRS})
    target_link_libraries(luisa-compute-backend-llvm PRIVATE ${LLVM_LIBS})
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(luisa-compute-backend-llvm PRIVATE
                /wd4624 /wd4996 # do not complain about LLVM
                /fp:fast /fp:contract /fp:except-)
    else ()
        target_compile_options(luisa-compute-backend-llvm PRIVATE -ffast-math)
    endif ()
else ()
    message(WARNING "LLVM not found. LLVM backend will not be built.")
endif ()
