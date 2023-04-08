if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Build with Metal backend")
    set(METAL_SOURCES
            metal-cpp/SingleHeader/Metal.hpp
            metal_api.cpp metal_api.h
            metal_device.cpp metal_device.h)
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
