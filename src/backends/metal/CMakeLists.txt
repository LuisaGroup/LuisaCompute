if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")

    message(STATUS "Build with Metal backend")

    set(METAL_SOURCES

            # common
            ../common/default_binary_io.cpp ../common/default_binary_io.h
            ../common/resource_tracker.h
            ../common/string_scratch.cpp ../common/string_scratch.h

            # metal-cpp
            metal-cpp/SingleHeader/Metal.hpp

            # metal
            metal_api.cpp metal_api.h
            metal_device.cpp metal_device.h
            metal_compiler.cpp metal_compiler.h
            metal_codegen_ast.cpp metal_codegen_ast.h
            metal_texture.cpp metal_texture.h
            metal_stream.cpp metal_stream.h
            metal_event.cpp metal_event.h
            )

    if (LUISA_COMPUTE_ENABLE_RUST)
        list(APPEND METAL_SOURCES metal_codegen_ir.cpp metal_codegen_ir.h)
    endif ()

    luisa_compute_add_backend(metal SOURCES ${METAL_SOURCES})
    target_link_libraries(lc-backend-metal PRIVATE
            "-framework Foundation"
            "-framework Metal"
            "-framework QuartzCore")

elseif (NOT LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES)
    message(FATAL_ERROR "Metal backend requires Apple platforms with clang.")
else ()
    message(WARNING "Metal backend requires Apple platforms with clang. The backend will be disabled.")
endif ()
