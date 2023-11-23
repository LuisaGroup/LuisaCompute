# TODO
if (WIN32)
    message(STATUS "Build with DirectX backend")
    set(LC_BACKEND_DX_SUPPORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dx_support")
    set(LC_BACKEND_DX_BUILTIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dx_builtin")

    file(GLOB_RECURSE LC_BACKEND_DX_SOURCES CONFIGURE_DEPENDS "*.cpp")
    luisa_compute_add_backend(dx
            SOURCES ${LC_BACKEND_DX_SOURCES}
            SUPPORT_DIR ${LC_BACKEND_DX_SUPPORT_DIR})
    target_precompile_headers(luisa-compute-backend-dx PRIVATE pch.h)
    target_include_directories(luisa-compute-backend-dx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(luisa-compute-backend-dx PRIVATE D3D12 d3dcompiler dxgi luisa-compute-vstl DirectML)
    target_compile_definitions(luisa-compute-backend-dx PRIVATE UNICODE)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_definitions(luisa-compute-backend-dx PRIVATE
                _XM_NO_INTRINSICS_
                m128_f32=vector4_f32
                m128_u32=vector4_u32)
    endif ()

    find_package(CUDAToolkit 11.7 QUIET)
    if (CUDAToolkit_FOUND)
        target_link_libraries(luisa-compute-backend-dx PRIVATE CUDA::cuda_driver)
        target_compile_definitions(luisa-compute-backend-dx PRIVATE LCDX_ENABLE_CUDA)
    endif ()

    if (TARGET luisa-compute-oidn-ext)
        target_link_libraries(luisa-compute-backend-dx PRIVATE luisa-compute-oidn-ext)
    endif()

elseif (NOT LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES)
    message(FATAL_ERROR "The DirectX backend will not be built on non-Windows platforms.")
else ()
    message(WARNING "The DirectX backend will not be built on non-Windows platforms.")
endif ()
