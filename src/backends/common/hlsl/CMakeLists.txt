if (LUISA_COMPUTE_ENABLE_DX OR LUISA_COMPUTE_ENABLE_VULKAN)

    set(LUISA_COMPUTE_HLSL_DEVICE_LIB_SOURCES
            hlsl_header
            spv_alias
            dx_linalg
            hlsl_header_fallback
            raytracing_header
            tex2d_bindless
            tex3d_bindless
            compute_quad
            determinant
            inverse
            indirect
            resource_size
            accel_header
            copy_sign
            bindless_common
            auto_diff
            reduce
            accel_process_vk.dxil
            load_bdls.dxil
            load_bdls_vk.dxil
            set_accel4.dxil
            bc6_encodeblock.dxil
            bc6_trymodeg10.dxil
            bc6_trymodele10.dxil
            bc7_encodeblock.dxil
            bc7_trymode02.dxil
            bc7_trymode137.dxil
            bc7_trymode456.dxil
    )
    list(TRANSFORM LUISA_COMPUTE_HLSL_DEVICE_LIB_SOURCES PREPEND ${CMAKE_CURRENT_LIST_DIR}/builtin/)
    add_custom_target(luisa-compute-backend-hlsl-embed
            COMMAND $<TARGET_FILE:luisa_embed_device_lib> ${LUISA_COMPUTE_HLSL_DEVICE_LIB_SOURCES} --unsigned -o builtin/hlsl_builtin_embed.cpp --preserve-ext
            BYPRODUCTS ${CMAKE_CURRENT_LIST_DIR}/builtin/hlsl_builtin_embed.cpp
            DEPENDS luisa_embed_device_lib ${CMAKE_CURRENT_LIST_FILE}
            SOURCES ${LUISA_COMPUTE_HLSL_DEVICE_LIB_SOURCES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            COMMENT "Embedding HLSL built-in sources..."
            VERBATIM)

    set(LUISA_COMPUTE_HLSL_SOURCES
            builtin/hlsl_builtin_embed.cpp
            access_chain.cpp
            access_chain.h
            binding_to_arg.cpp
            binding_to_arg.h
            codegen_stack_data.cpp
            codegen_stack_data.h
            dxcapi.h
            hlsl_codegen.cpp
            hlsl_codegen.h
            hlsl_codegen_util.cpp
            lc_hlsl_pch.h
            shader_compiler.cpp
            shader_compiler.h
            shader_property.h
            string_builder.cpp
            string_builder.h
            struct_generator.cpp
            struct_generator.h
            variant_util.h
            WinAdapter.h
    )

    add_library(luisa-compute-hlsl-codegen STATIC ${LUISA_COMPUTE_HLSL_SOURCES})
    target_precompile_headers(luisa-compute-hlsl-codegen PRIVATE lc_hlsl_pch.h)
    target_link_libraries(luisa-compute-hlsl-codegen PUBLIC luisa-compute-runtime luisa-compute-core luisa-compute-vstl)

endif ()
