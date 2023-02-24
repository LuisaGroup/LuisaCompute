# TODO
if (WIN32)
    message(STATUS "Build with DirectX backend")
    set(VENGINE_SOURCE
            DXApi/ext.cpp
            DXApi/LCCmdBuffer.cpp
            DXApi/LCCmdBuffer.h
            DXApi/LCDevice.cpp
            DXApi/LCDevice.h
            DXApi/LCEvent.cpp
            DXApi/LCEvent.h
            DXApi/LCSwapChain.cpp
            DXApi/LCSwapChain.h

            HLSL/codegen_stack_data.cpp
            HLSL/codegen_stack_data.h
            HLSL/dx_codegen.cpp
            HLSL/dx_codegen.h
            HLSL/dx_codegen_util.cpp
            HLSL/shader_property.h
            HLSL/struct_generator.cpp
            HLSL/struct_generator.h

            DXRuntime/CommandAllocator.cpp
            DXRuntime/CommandAllocator.h
            DXRuntime/CommandAllocatorBase.h
            DXRuntime/CommandBuffer.cpp
            DXRuntime/CommandBuffer.h
            DXRuntime/CommandQueue.cpp
            DXRuntime/CommandQueue.h
            DXRuntime/Device.cpp
            DXRuntime/Device.h
            DXRuntime/DxPtr.h
            DXRuntime/EnhancedBarrierTracker.cpp.backup
            DXRuntime/EnhancedBarrierTracker.h.backup
            DXRuntime/GlobalSamplers.cpp
            DXRuntime/GlobalSamplers.h
            DXRuntime/ResourceStateTracker.cpp
            DXRuntime/ResourceStateTracker.h
            DXRuntime/ShaderPaths.h

            Resource/D3D12MemoryAllocator/D3D12MemAlloc.cpp
            Resource/D3D12MemoryAllocator/D3D12MemAlloc.h
            Resource/AllocHandle.h
            Resource/BindlessArray.cpp
            Resource/BindlessArray.h
            Resource/BindProperty.h
            Resource/BottomAccel.cpp
            Resource/BottomAccel.h
            Resource/Buffer.cpp
            Resource/Buffer.h
            Resource/BufferView.h
            Resource/DefaultBuffer.cpp
            Resource/DefaultBuffer.h
            Resource/DepthBuffer.cpp
            Resource/DepthBuffer.h
            Resource/DescriptorHeap.cpp
            Resource/DescriptorHeap.h
            Resource/ExternalBuffer.cpp
            Resource/ExternalBuffer.h
            Resource/ExternalDepth.cpp
            Resource/ExternalDepth.h
            Resource/ExternalTexture.cpp
            Resource/ExternalTexture.h
            Resource/GpuAllocator.cpp
            Resource/GpuAllocator.h
            Resource/ReadbackBuffer.cpp
            Resource/ReadbackBuffer.h
            Resource/RenderTexture.cpp
            Resource/RenderTexture.h
            Resource/Resource.cpp
            Resource/Resource.h
            Resource/SwapChain.h
            Resource/TextureBase.cpp
            Resource/TextureBase.h
            Resource/TopAccel.cpp
            Resource/TopAccel.h
            Resource/UploadBuffer.cpp
            Resource/UploadBuffer.h

            Shader/BuiltinKernel.cpp
            Shader/BuiltinKernel.h
            Shader/ComputeShader.cpp
            Shader/ComputeShader.h
            Shader/RasterShader.cpp
            Shader/RasterShader.h
            Shader/RTShader.cpp.backup
            Shader/RTShader.h.backup
            Shader/Shader.cpp
            Shader/Shader.h
            Shader/ShaderCompiler.cpp
            Shader/ShaderCompiler.h
            Shader/ShaderSerializer.cpp
            Shader/ShaderSerializer.h
            Shader/ShaderVariableType.h
            )
    luisa_compute_add_backend(dx SOURCES ${VENGINE_SOURCE})
    target_include_directories(luisa-compute-backend-dx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(luisa-compute-backend-dx PRIVATE D3D12 d3dcompiler dxgi)
    target_compile_definitions(luisa-compute-backend-dx PRIVATE UNICODE)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_definitions(luisa-compute-backend-dx PRIVATE
                _XM_NO_INTRINSICS_
                m128_f32=vector4_f32
                m128_u32=vector4_u32)
    endif ()
    set_target_properties(luisa-compute-backend-dx PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF)
    set(DX_SUPPORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dx_support")
    set(DX_BUILTIN_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dx_builtin")
    mark_as_advanced(DX_SUPPORT_DIR)
    mark_as_advanced(DX_BUILTIN_SHADER_DIR)
    if (CMAKE_BUILD_TYPE)
        add_custom_target(lc-dx-copy-support ALL
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${DX_SUPPORT_DIR}"
                "$<TARGET_FILE_DIR:luisa-compute-core>/")
        add_custom_target(lc-dx-copy-builtin ALL
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${DX_BUILTIN_SHADER_DIR}"
                "$<TARGET_FILE_DIR:luisa-compute-core>/.data/")
        add_dependencies(luisa-compute-backend-dx
                lc-dx-copy-support
                lc-dx-copy-builtin)
    endif ()
    install(DIRECTORY ${DX_SUPPORT_DIR}
            DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY ${DX_BUILTIN_SHADER_DIR}
            DESTINATION ${CMAKE_INSTALL_BINDIR}/.data)
elseif (NOT LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES)
    message(FATAL_ERROR "The DirectX backend will not be built on non-Windows platforms.")
else ()
    message(WARNING "The DirectX backend will not be built on non-Windows platforms.")
endif ()
