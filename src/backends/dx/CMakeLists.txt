if (WIN32)
    message(STATUS "Build with DirectX")
    Set(VENGINE_SOURCE
Api/LCDevice.cpp
Codegen/ArgumentPacker.cpp
Codegen/DxCodegen.cpp
Codegen/DxCodegenUtil.cpp
Codegen/ShaderCompiler.cpp
Codegen/StructGenerator.cpp
DXRuntime/AccelStructure.cpp
DXRuntime/CommandAllocator.cpp
DXRuntime/CommandBuffer.cpp
DXRuntime/CommandQueue.cpp
DXRuntime/Device.cpp
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
)
    luisa_compute_add_backend(dx SOURCES ${VENGINE_SOURCE})
    target_include_directories(luisa-compute-backend-dx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/Runtime)

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
