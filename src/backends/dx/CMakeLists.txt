if (WIN32)
    
    message(STATUS "Build with DirectX")
    ##[[[]]]
    set(DX_BACKEND_SOURCES
        CJsonObject/BinaryJson.cpp
        CJsonObject/cJSON.cpp
        CJsonObject/CJsonObject.cpp
        CJsonObject/JsonCompiler.cpp
        Common/Common.cpp
        Common/DynamicDLL.cpp
        Common/DynamicLink.cpp
        Common/GameTimer.cpp
        Common/GFXUtil.cpp
        Common/Log.cpp
        Common/MathHelper.cpp
        Common/spin_mutex.cpp
        File/CopyDir.cpp
        File/FilePacker.cpp
        File/FileUtility.cpp
        File/MGUID.cpp
        File/Path.cpp
        JobSystem/JobBucket.cpp
        JobSystem/JobNode.cpp
        JobSystem/JobSystem.cpp
        JobSystem/ThreadPool.cpp
        JobSystem/ThreadTaskHandle.cpp
        PipelineComponent/CommandAllocator.cpp
        PipelineComponent/CommandBuffer.cpp
        PipelineComponent/FrameResource.cpp
        PipelineComponent/RayTracing/RayTracingManager.cpp
        PipelineComponent/ThreadCommand.cpp
        RenderComponent/CBufferAllocator.cpp
        RenderComponent/CBufferPool.cpp
        RenderComponent/ComputeShader.cpp
        RenderComponent/DescHeapPool.cpp
        RenderComponent/DescriptorHeap.cpp
        RenderComponent/GPUResourceBase.cpp
        RenderComponent/IShader.cpp
        RenderComponent/LCMesh.cpp
        RenderComponent/RayShader.cpp
        RenderComponent/ReadbackBuffer.cpp
        RenderComponent/RenderCommand.cpp
        RenderComponent/RenderTexture.cpp
        RenderComponent/StructuredBuffer.cpp
        RenderComponent/Texture.cpp
        RenderComponent/TextureBase.cpp
        RenderComponent/TextureHeap.cpp
        RenderComponent/UploadBuffer.cpp
        RenderComponent/Utility/CommandSignature.cpp
        RenderComponent/Utility/D3D12MemoryAllocator/D3D12MemAlloc.cpp
        RenderComponent/Utility/DefaultBufferAllocator.cpp
        RenderComponent/Utility/DefaultTextureAllocator.cpp
        RenderComponent/Utility/SeparableRendererManager.cpp
        RenderComponent/Utility/ShaderIO.cpp
        render_graph/rg_node.cpp
        render_graph/rg_system.cpp
        RHI/DXCommandVisitor.cpp
        RHI/DXDevice.cpp
        RHI/DXEvent.cpp
        RHI/LCTextureHeap.cpp
        RHI/ShaderCompiler.cpp
        ShaderCompile/HLSLCompiler.cpp
        ShaderCompile/LuisaASTTranslator.cpp
        Singleton/ColorUtility.cpp
        Singleton/Graphics.cpp
        Singleton/MathLib.cpp
        Singleton/MeshLayout.cpp
        Singleton/ShaderID.cpp
        Struct/ConstBuffer.cpp
        TestMain.cpp
        Utility/Actor.cpp
        Utility/BinaryReader.cpp
        Utility/BuddyAllocator.cpp
        Utility/ElementAllocator.cpp
        Utility/IEnumerator.cpp
        Utility/MD5.cpp
        Utility/Random.cpp
        Utility/StringUtility.cpp
        Utility/TaskThread.cpp
        Utility/ThreadResetEvent.cpp
    )
##[[[]]]
    luisa_compute_add_backend(dx SOURCES ${DX_BACKEND_SOURCES})
    target_include_directories(luisa-compute-backend-dx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    
    if (MSVC)
        target_compile_options(luisa-compute-backend-dx PRIVATE
                               /JMC
                               /permissive-
                               /MP
                               /GS /TP /Qpar /GL /W0 /Gy
                               /Zi /Gm-
                               /sdl- /GT
                               $<$<NOT:$<CXX_COMPILER_ID:Clang>>:/QIntel-jcc-erratum>
                               /WX- /Zc:forScope
                               /Zc:inline /fp:precise
                               /arch:AVX2 /Gr /Oy /Oi
                               /std:c++latest /FC
                               /nologo
                               /Ot)
        
        target_link_options(luisa-compute-backend-dx PRIVATE
                            /MANIFEST
                            $<$<NOT:$<CXX_COMPILER_ID:Clang>>:/LTCG:incremental>
                            /NXCOMPAT
                            /OPT:REF /INCREMENTAL:NO
                            /NOLOGO
                            /TLBID:1
                            /OPT:ICF /ERRORREPORT:PROMPT)
        target_compile_definitions(luisa-compute-backend-dx PRIVATE
                                   UNICODE)
    endif ()
    
    target_link_libraries(luisa-compute-backend-dx PRIVATE mimalloc-static D3D12 d3dcompiler dxgi)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_definitions(luisa-compute-backend-dx PRIVATE
                                   _XM_NO_INTRINSICS_
        
                                   m128_f32=vector4_f32
                                   m128_u32=vector4_u32)
    endif ()
    set_target_properties(luisa-compute-backend-dx PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF)
    
    # Copy compiler toolkits
    add_custom_target(luisa-compute-backend-dx-copy-compiler ALL
                      COMMAND ${CMAKE_COMMAND} -E copy_directory
                      "${CMAKE_CURRENT_SOURCE_DIR}/VEngineCompiler"
                      "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/VEngineCompiler")
    add_dependencies(luisa-compute-backend-dx luisa-compute-backend-dx-copy-compiler)
    foreach (CONFIG ${CMAKE_CONFIGURATION_TYPES})
        string(TOLOWER ${CONFIG} config_lower)
        string(TOUPPER ${CONFIG} CONFIG_UPPER)
        add_custom_target(luisa-compute-backend-dx-copy-compiler-${config_lower} ALL
                          COMMAND ${CMAKE_COMMAND} -E copy_directory
                          "${CMAKE_CURRENT_SOURCE_DIR}/VEngineCompiler"
                          "${CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER}}/backends/VEngineCompiler")
        add_dependencies(luisa-compute-backend-dx luisa-compute-backend-dx-copy-compiler-${config_lower})
    endforeach ()
endif ()

