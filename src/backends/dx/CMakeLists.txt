# TODO
if (WIN32)
    message(STATUS "Build with DirectX backend")
    set(LC_BACKEND_DX_SUPPORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dx_support")
    set(LC_BACKEND_DX_BUILTIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dx_builtin")
    set(LC_BACKEND_DX_SOURCE
            DXApi/ext.cpp
            DXApi/LCCmdBuffer.cpp DXApi/LCCmdBuffer.h
            DXApi/LCDevice.cpp DXApi/LCDevice.h
            DXApi/LCEvent.cpp DXApi/LCEvent.h
            DXApi/LCSwapChain.cpp DXApi/LCSwapChain.h

            HLSL/codegen_stack_data.cpp HLSL/codegen_stack_data.h
            HLSL/common_iterator.h
            HLSL/dx_codegen.cpp HLSL/dx_codegen.h
            HLSL/dx_codegen_util.cpp
            HLSL/shader_property.h
            HLSL/string_builder.cpp HLSL/string_builder.h
            HLSL/struct_generator.cpp HLSL/struct_generator.h
            HLSL/variant_util.h

            DXRuntime/CommandAllocator.cpp DXRuntime/CommandAllocator.h
            DXRuntime/CommandAllocatorBase.h
            DXRuntime/CommandBuffer.cpp DXRuntime/CommandBuffer.h
            DXRuntime/CommandQueue.cpp DXRuntime/CommandQueue.h
            DXRuntime/Device.cpp DXRuntime/Device.h
            DXRuntime/DxPtr.h
            DXRuntime/GlobalSamplers.cpp DXRuntime/GlobalSamplers.h
            DXRuntime/ResourceStateTracker.cpp DXRuntime/ResourceStateTracker.h

            Resource/D3D12MemoryAllocator/D3D12MemAlloc.cpp Resource/D3D12MemoryAllocator/D3D12MemAlloc.h
            Resource/AllocHandle.h
            Resource/BindlessArray.cpp Resource/BindlessArray.h
            Resource/BindProperty.h
            Resource/BottomAccel.cpp Resource/BottomAccel.h
            Resource/Buffer.cpp Resource/Buffer.h
            Resource/BufferView.h
            Resource/DefaultBuffer.cpp Resource/DefaultBuffer.h
            Resource/DepthBuffer.cpp Resource/DepthBuffer.h
            Resource/DescriptorHeap.cpp Resource/DescriptorHeap.h
            Resource/ExternalBuffer.cpp Resource/ExternalBuffer.h
            Resource/ExternalDepth.cpp Resource/ExternalDepth.h
            Resource/ExternalTexture.cpp Resource/ExternalTexture.h
            Resource/GpuAllocator.cpp Resource/GpuAllocator.h
            Resource/ReadbackBuffer.cpp Resource/ReadbackBuffer.h
            Resource/RenderTexture.cpp Resource/RenderTexture.h
            Resource/Resource.cpp Resource/Resource.h
            Resource/SwapChain.h
            Resource/TextureBase.cpp Resource/TextureBase.h
            Resource/TopAccel.cpp Resource/TopAccel.h
            Resource/UploadBuffer.cpp Resource/UploadBuffer.h

            Shader/BuiltinKernel.cpp Shader/BuiltinKernel.h
            Shader/ComputeShader.cpp Shader/ComputeShader.h
            Shader/RasterShader.cpp Shader/RasterShader.h
            Shader/Shader.cpp Shader/Shader.h
            Shader/ShaderCompiler.cpp Shader/ShaderCompiler.h
            Shader/ShaderSerializer.cpp Shader/ShaderSerializer.h
            Shader/ShaderVariableType.h)
    luisa_compute_add_backend(dx
            SOURCES ${LC_BACKEND_DX_SOURCE}
            SUPPORT_DIR ${LC_BACKEND_DX_SUPPORT_DIR}
            BUILTIN_DIR ${LC_BACKEND_DX_BUILTIN_DIR})
    target_include_directories(lc-backend-dx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(lc-backend-dx PRIVATE D3D12 d3dcompiler dxgi)
    target_compile_definitions(lc-backend-dx PRIVATE UNICODE)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_definitions(lc-backend-dx PRIVATE
                _XM_NO_INTRINSICS_
                m128_f32=vector4_f32
                m128_u32=vector4_u32)
    endif ()
elseif (NOT LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES)
    message(FATAL_ERROR "The DirectX backend will not be built on non-Windows platforms.")
else ()
    message(WARNING "The DirectX backend will not be built on non-Windows platforms.")
endif ()
