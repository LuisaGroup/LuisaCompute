# bge-m3 embedding engine (embedding_model_plan.md).
# Engine static lib + offline packer tool + demo/dump tools; tests live in
# src/tests. Host-only engine TUs (tokenizer, package, scheduler glue) must
# not include lc-dsl device headers (CONVENTIONS.md).

add_library(lc-example-embedding STATIC
        embed/tokenizer.cpp
        embed/package.cpp
        embed/session.cpp
        embed/executor.cpp
        embed/scheduler.cpp
        embed/retrieval_index.cpp
        embed/retrieval_service.cpp
        embed/retrieval_rpc.cpp
        embed/debug_utils.cpp
        embed/kernels/kernels_embed.cpp
        embed/kernels/kernels_embed_vec.cpp
        embed/kernels/kernels_gemm.cpp
        embed/kernels/kernels_norm.cpp
        embed/kernels/kernels_norm_vec.cpp
        embed/kernels/kernels_heads.cpp
        embed/kernels/kernels_heads_vec.cpp
        embed/kernels/kernels_attention.cpp
        embed/kernels/kernels_coop.cpp
        embed/tensor_path.cpp
)
target_link_libraries(lc-example-embedding PUBLIC luisa::compute)
target_link_libraries(lc-example-embedding PUBLIC luisa-compute-ext-yyjson-interface)
target_include_directories(lc-example-embedding PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/examples)

# C-FFI front-end: exports the retrieval steps (initialize / load-dir / query /
# update / unload / encode) as a documented C ABI so any host language can drive
# the retriever through its own FFI. See api/example_embed_retrieval_lib.h.
if (UNIX)
    # The engine is a static lib; on ELF/Mach-O platforms its objects must be
    # position independent before they can be linked into our shared library.
    set_target_properties(lc-example-embedding PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif ()
add_library(example_embed_retrieval_lib SHARED api/example_embed_retrieval_lib.cpp)
target_compile_definitions(example_embed_retrieval_lib PRIVATE EXAMPLE_EMBED_RETRIEVAL_LIB_BUILD)
target_link_libraries(example_embed_retrieval_lib PRIVATE lc-example-embedding)
target_link_libraries(example_embed_retrieval_lib PUBLIC luisa::compute)
target_include_directories(example_embed_retrieval_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/api
        ${CMAKE_SOURCE_DIR}/examples)

luisa_compute_add_executable(example_embed_pack_model tools/pack_model.cpp)
target_include_directories(example_embed_pack_model PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/examples)

luisa_compute_add_executable(example_embed_text example_embed_text.cpp)
target_link_libraries(example_embed_text PRIVATE lc-example-embedding)
target_include_directories(example_embed_text PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/examples)

luisa_compute_add_executable(example_embed_retrieval example_embed_retrieval.cpp)
target_link_libraries(example_embed_retrieval PRIVATE lc-example-embedding)
target_include_directories(example_embed_retrieval PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/examples)
if (WIN32)
    target_link_libraries(example_embed_retrieval PRIVATE ws2_32)
endif ()

luisa_compute_add_executable(example_embed_dump_golden tools/dump_golden.cpp)
target_link_libraries(example_embed_dump_golden PRIVATE lc-example-embedding)
target_include_directories(example_embed_dump_golden PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/examples)
