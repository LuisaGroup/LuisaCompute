if (LUISA_COMPUTE_PYTHON_PACKAGING)# install LLVM with conan
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
    list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})
    if (NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
        message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
        file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/0.18.1/conan.cmake"
                "${CMAKE_BINARY_DIR}/conan.cmake"
                TLS_VERIFY ON)
    endif ()
    include(${CMAKE_BINARY_DIR}/conan.cmake)
    conan_cmake_configure(REQUIRES llvm-core/13.0.0
            GENERATORS cmake_find_package)
    conan_cmake_autodetect(settings)
    conan_cmake_install(PATH_OR_REFERENCE .
            BUILD missing
            REMOTE conancenter
            SETTINGS ${settings})
endif ()

find_package(LLVM)
if (LLVM_FOUND) # link with llvm for jit
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
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
    llvm_map_components_to_libnames(LLVM_LIBS core executionengine support orcjit nativecodegen irreader passes analysis coroutines)
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
