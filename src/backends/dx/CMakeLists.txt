if (WIN32)
    message(STATUS "Build with DirectX")
    Set(VENGINE_SOURCE
Api/LCCmdBuffer.cpp
Api/LCDevice.cpp
Api/LCEvent.cpp
Codegen/ArgumentPacker.cpp
Codegen/CodegenStackData.cpp
Codegen/DxCodegen.cpp
Codegen/DxCodegenUtil.cpp
Codegen/ShaderHeader.cpp
Codegen/StructGenerator.cpp
DXRuntime/AccelStructure.cpp
DXRuntime/CommandAllocator.cpp
DXRuntime/CommandBuffer.cpp
DXRuntime/CommandQueue.cpp
DXRuntime/Device.cpp
DXRuntime/GlobalSamplers.cpp
DXRuntime/ResourceStateTracker.cpp
Resource/BindlessArray.cpp
Resource/BottomAccel.cpp
Resource/Buffer.cpp
Resource/D3D12MemoryAllocator/D3D12MemAlloc.cpp
Resource/DefaultBuffer.cpp
Resource/DescriptorHeap.cpp
Resource/IGpuAllocator.cpp
Resource/Mesh.cpp
Resource/ReadbackBuffer.cpp
Resource/RenderTexture.cpp
Resource/Resource.cpp
Resource/TextureBase.cpp
Resource/TopAccel.cpp
Resource/UploadBuffer.cpp
Shader/BuiltinKernel.cpp
Shader/ComputeShader.cpp
Shader/PipelineLibrary.cpp
Shader/RTShader.cpp
Shader/Shader.cpp
Shader/ShaderCompiler.cpp
Shader/ShaderSerializer.cpp
Api/LCCmdBuffer.h
Api/LCDevice.h
Api/LCEvent.h
Codegen/ArgumentPacker.h
Codegen/CodegenStackData.h
Codegen/DxCodegen.h
Codegen/ShaderHeader.h
Codegen/StructGenerator.h
d3dx12.h
dxc/dxcapi.h
DXRuntime/AccelStructure.h
DXRuntime/CommandAllocator.h
DXRuntime/CommandBuffer.h
DXRuntime/CommandQueue.h
DXRuntime/Device.h
DXRuntime/GlobalSamplers.h
DXRuntime/ResourceStateTracker.h
Resource/AllocHandle.h
Resource/BindlessArray.h
Resource/BindProperty.h
Resource/BottomAccel.h
Resource/Buffer.h
Resource/BufferView.h
Resource/D3D12MemoryAllocator/D3D12MemAlloc.h
Resource/DefaultBuffer.h
Resource/DescriptorHeap.h
Resource/IGpuAllocator.h
Resource/Mesh.h
Resource/ReadbackBuffer.h
Resource/RenderTexture.h
Resource/Resource.h
Resource/TextureBase.h
Resource/TopAccel.h
Resource/UploadBuffer.h
Shader/BuiltinKernel.h
Shader/ComputeShader.h
Shader/PipelineLibrary.h
Shader/RTShader.h
Shader/Shader.h
Shader/ShaderCompiler.h
Shader/ShaderSerializer.h
Shader/ShaderVariableType.h
test_compute.h
test_rtx.h
)
    luisa_compute_add_backend(dx SOURCES ${VENGINE_SOURCE})
    target_include_directories(luisa-compute-backend-dx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    target_compile_options(luisa-compute-backend-dx PRIVATE
    /JMC
    /permissive-
    /GS /TP /Qpar /GL /W0 /Gy
    /Zi /Gm-
    /sdl- /GT
    /QIntel-jcc-erratum
    /WX- /Zc:forScope
    /Zc:inline /fp:precise
    /arch:AVX2 /Gr /Oy /Oi
    /FC
    /nologo /GF
    /Ot)
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_link_options(luisa-compute-backend-dx PRIVATE
    /MANIFEST
    $<$<NOT:$<CXX_COMPILER_ID:Clang>>:/LTCG:incremental>
    /NXCOMPAT
    /OPT:REF /INCREMENTAL:NO
    /NOLOGO
    /TLBID:1
    /OPT:ICF /ERRORREPORT:PROMPT)  
endif ()
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
mark_as_advanced(DX_SUPPORT_DIR)
if (CMAKE_BUILD_TYPE)
    add_custom_target(lc-dx-copy-support ALL
            COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${DX_SUPPORT_DIR}"
            "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    add_dependencies(luisa-compute-backend-dx lc-dx-copy-support)
endif ()

endif ()
