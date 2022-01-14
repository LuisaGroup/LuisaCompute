if (WIN32)
    message(STATUS "Build with DirectX")
    Set(VENGINE_SOURCE
Api/LCDevice.cpp
Codegen/ArgumentPacker.cpp
Codegen/DxCodegen.cpp
Codegen/DxCodegenUtil.cpp
Codegen/ShaderHeader.cpp
Codegen/StructGenerator.cpp
Codegen/StructVariableTracker.cpp
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
Shader/ComputeShader.cpp
Shader/RTShader.cpp
Shader/Shader.cpp
Shader/ShaderCompiler.cpp
Api/LCDevice.h
Codegen/ArgumentPacker.h
Codegen/DxCodegen.h
Codegen/ShaderHeader.h
Codegen/StructGenerator.h
Codegen/StructVariableTracker.h
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
Resource/MeshInstance.h
Resource/ReadbackBuffer.h
Resource/RenderTexture.h
Resource/Resource.h
Resource/TextureBase.h
Resource/TopAccel.h
Resource/UploadBuffer.h
Shader/ComputeShader.h
Shader/RTShader.h
Shader/Shader.h
Shader/ShaderCompiler.h
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

    target_link_options(luisa-compute-backend-dx PRIVATE
    /MANIFEST
    $<$<NOT:$<CXX_COMPILER_ID:Clang>>:/LTCG:incremental>
    /NXCOMPAT
    /OPT:REF /INCREMENTAL:NO
    /NOLOGO
    /TLBID:1
    /OPT:ICF /ERRORREPORT:PROMPT)  
    target_link_libraries(luisa-compute-backend-dx PRIVATE D3D12 d3dcompiler dxgi)
    target_compile_definitions(luisa-compute-backend-dx PRIVATE UNICODE)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_definitions(luisa-compute-backend-dx PRIVATE
        _XM_NO_INTRINSICS_
        m128_f32=vector4_f32
        m128_u32=vector4_u32)
    endif ()
    set_target_properties(luisa-compute-backend-dx PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF)
endif ()
