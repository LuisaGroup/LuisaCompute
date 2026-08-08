if (LUISA_COMPUTE_ENABLE_DX OR LUISA_COMPUTE_ENABLE_VULKAN)

    set(LUISA_COMPUTE_HLSL_DEVICE_LIB_SOURCES
            hlsl_header.bytes
            spv_alias.bytes
            dx_linalg.bytes
            vk_linalg.bytes
            hlsl_header_fallback.bytes
            raytracing_header.bytes
            raytracing_motion_header.bytes
            indirect_prepare_vk.spv
            tex2d_bindless.bytes
            tex3d_bindless.bytes
            compute_quad.bytes
            determinant.bytes
            inverse.bytes
            indirect.bytes
            resource_size.bytes
            accel_header.bytes
            copy_sign.bytes
            bindless_common.bytes
            auto_diff.bytes
            reduce.bytes
            bindless_upload_vk.spv
            bindless_upload.bytes
            bindless_upload_vk.bytes
            accel_process_vk.spv
            accel_process.bytes
            accel_process_vk.bytes
            accel_process_vk_motion.bytes
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
    set(LUISA_COMPUTE_HLSL_EMBEDDED_SOURCE
            ${CMAKE_CURRENT_LIST_DIR}/builtin/hlsl_builtin_embed.cpp)
    set_source_files_properties(${LUISA_COMPUTE_HLSL_EMBEDDED_SOURCE}
            PROPERTIES GENERATED TRUE)
    add_custom_target(luisa-compute-backend-hlsl-embed
            COMMAND $<TARGET_FILE:luisa_embed_device_lib> ${LUISA_COMPUTE_HLSL_DEVICE_LIB_SOURCES} --unsigned -o ${LUISA_COMPUTE_HLSL_EMBEDDED_SOURCE} --preserve-ext
            COMMAND $<TARGET_FILE:luisa_embed_device_lib> --check ${LUISA_COMPUTE_HLSL_DEVICE_LIB_SOURCES} --unsigned -o ${LUISA_COMPUTE_HLSL_EMBEDDED_SOURCE} --preserve-ext
            BYPRODUCTS ${LUISA_COMPUTE_HLSL_EMBEDDED_SOURCE}
            DEPENDS luisa_embed_device_lib ${CMAKE_CURRENT_LIST_FILE} ${LUISA_COMPUTE_HLSL_DEVICE_LIB_SOURCES}
            SOURCES ${LUISA_COMPUTE_HLSL_DEVICE_LIB_SOURCES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            COMMENT "Embedding HLSL built-in sources..."
            VERBATIM)
    add_custom_target(luisa-compute-backend-hlsl-embed-check
            COMMAND $<TARGET_FILE:luisa_embed_device_lib> --check ${LUISA_COMPUTE_HLSL_DEVICE_LIB_SOURCES} --unsigned -o ${LUISA_COMPUTE_HLSL_EMBEDDED_SOURCE} --preserve-ext
            DEPENDS luisa_embed_device_lib ${CMAKE_CURRENT_LIST_FILE} ${LUISA_COMPUTE_HLSL_DEVICE_LIB_SOURCES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
            COMMENT "Checking generated HLSL built-in embedding..."
            VERBATIM)

    set(LUISA_COMPUTE_HLSL_SOURCES
            ${LUISA_COMPUTE_HLSL_EMBEDDED_SOURCE}
            access_chain.cpp
            access_chain.h
            binding_to_arg.cpp
            binding_to_arg.h
            codegen_stack_data.cpp
            codegen_stack_data.h
            dxcapi.h
            hlsl_codegen.cpp
            hlsl_codegen.h
            codegen_utils/cbuffer.cpp
            codegen_utils/constant.cpp
            codegen_utils/entry_points.cpp
            codegen_utils/function_codegen.cpp
            codegen_utils/property.cpp
            codegen_utils/resource.cpp
            codegen_utils/type_system.cpp
            codegen_utils/variable.cpp
            lc_hlsl_pch.h
            shader_compiler.cpp
            shader_compiler.h
            shader_property.h
            shared_var.h
            struct_generator.cpp
            struct_generator.h
            variant_util.h
            WinAdapter.h
    )

    add_library(luisa-compute-hlsl-codegen OBJECT ${LUISA_COMPUTE_HLSL_SOURCES})
    # The embed target intentionally runs whenever the codegen target is built.
    # The generator writes only on content changes, so edits to any canonical
    # .bytes/.dxil input cannot leave a stale aggregate or trigger
    # needless recompilation when the aggregate is already current.
    add_dependencies(luisa-compute-hlsl-codegen luisa-compute-backend-hlsl-embed)
    target_link_libraries(luisa-compute-hlsl-codegen PUBLIC luisa-compute-runtime luisa-compute-core luisa-compute-vstl)
    set_target_properties(luisa-compute-hlsl-codegen PROPERTIES POSITION_INDEPENDENT_CODE ON)

    if (UNIX AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(luisa-compute-hlsl-codegen PUBLIC -fms-extensions)
    endif ()

endif ()
