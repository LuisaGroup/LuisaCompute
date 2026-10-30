# EmbeddingGemma-2 multimodal embedding engine (luisa::mm_embed).

add_library(lc-example-mm-embedding STATIC
    embed/tokenizer.cpp
    embed/package.cpp
    embed/session.cpp
    embed/executor.cpp
    embed/kernels/kernels_norm.cpp
    embed/kernels/kernels_rope.cpp
    embed/kernels/kernels_mlp.cpp
    embed/kernels/kernels_attention.cpp
    embed/kernels/kernels_pool.cpp
    embed/kernels/kernels_gemm.cpp
)
target_link_libraries(lc-example-mm-embedding PUBLIC luisa::compute)
target_link_libraries(lc-example-mm-embedding PUBLIC luisa-compute-ext-yyjson-interface)
target_include_directories(lc-example-mm-embedding PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/examples)

luisa_compute_add_executable(example_mm_embed_pack_model tools/pack_model.cpp)
target_link_libraries(example_mm_embed_pack_model PRIVATE lc-example-mm-embedding)
target_include_directories(example_mm_embed_pack_model PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/examples)

luisa_compute_add_executable(example_mm_embed_dump_golden tools/dump_golden.cpp)
target_link_libraries(example_mm_embed_dump_golden PRIVATE lc-example-mm-embedding)
target_include_directories(example_mm_embed_dump_golden PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/examples)

luisa_compute_add_executable(example_mm_embed_text example_mm_embed_text.cpp)
target_link_libraries(example_mm_embed_text PRIVATE lc-example-mm-embedding)
target_include_directories(example_mm_embed_text PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/examples)

luisa_compute_add_executable(example_mm_embed_test example_mm_embed_test.cpp)
target_link_libraries(example_mm_embed_test PRIVATE lc-example-mm-embedding)
target_include_directories(example_mm_embed_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/examples)
