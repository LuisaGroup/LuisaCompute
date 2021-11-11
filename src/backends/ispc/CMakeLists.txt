##[[[]]]
    set(ISPC_BACKEND_SOURCES
        runtime/ispc_codegen.cpp
        runtime/ispc_codegenutil.cpp
        runtime/ispc_compiler.cpp
        runtime/ispc_device.cpp
        runtime/ispc_shader.cpp
    )
##[[[]]]
message(STATUS "Build with ISPC")
luisa_compute_add_backend(ispc SOURCES ${ISPC_BACKEND_SOURCES})

# TODO: will conflict with src/runtime
# target_include_directories(luisa-compute-backend-ispc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
# target_link_libraries(luisa-compute-backend-ispc PRIVATE luisa-compute-vstl)

# TODO: I don't believe we need these
# if (MSVC)
#     target_compile_options(luisa-compute-backend-ispc PRIVATE
#             /JMC
#             /permissive-
#             /GS /TP /Qpar /GL /W0 /Gy
#             /Zi /Gm-
#             /sdl- /GT
#             $<$<NOT:$<CXX_COMPILER_ID:Clang>>:/QIntel-jcc-erratum>
#             /WX- /Zc:forScope
#             /Zc:inline /fp:precise
#             /arch:AVX2 /Gr /Oy /Oi
#             /std:c++latest /FC
#             /nologo
#             /Ot)
#
#     target_link_options(luisa-compute-backend-ispc PRIVATE
#             /MANIFEST
#             $<$<NOT:$<CXX_COMPILER_ID:Clang>>:/LTCG:incremental>
#             /NXCOMPAT
#             /OPT:REF /INCREMENTAL:NO
#             /NOLOGO
#             /TLBID:1
#             /OPT:ICF /ERRORREPORT:PROMPT)
# endif ()
#
# if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#     target_compile_definitions(luisa-compute-backend-ispc PRIVATE
#             UNICODE
#             _XM_NO_INTRINSICS_
#             m128_f32=vector4_f32
#             m128_u32=vector4_u32)
# endif ()

set_target_properties(luisa-compute-backend-ispc PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS OFF
        UNITY_BUILD ON)