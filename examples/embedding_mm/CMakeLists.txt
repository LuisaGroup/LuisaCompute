# EmbeddingGemma-2 multimodal embedding engine (luisa::embed_mm).

add_library(lc-example-embedding-mm STATIC
    embed/tokenizer.cpp
    embed/package.cpp
    embed/image_processor.cpp
    embed/video_processor.cpp
    embed/audio_processor.cpp
    embed/session.cpp
    embed/executor.cpp
    embed/kernels/kernels_norm.cpp
    embed/kernels/kernels_rope.cpp
    embed/kernels/kernels_mlp.cpp
    embed/kernels/kernels_attention.cpp
    embed/kernels/kernels_pool.cpp
    embed/kernels/kernels_gemm.cpp
    embed/kernels/kernels_vision.cpp
    embed/kernels/kernels_audio.cpp
)
target_link_libraries(lc-example-embedding-mm PUBLIC luisa::compute)
target_link_libraries(lc-example-embedding-mm PUBLIC luisa-compute-ext-yyjson-interface)
target_link_libraries(lc-example-embedding-mm PUBLIC luisa-compute-ext-stb-interface)
if (WIN32)
    target_link_libraries(lc-example-embedding-mm PUBLIC mfreadwrite mfplat mfuuid ole32)
endif ()
target_include_directories(lc-example-embedding-mm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/examples)

luisa_compute_add_executable(example_embed_mm_pack_model tools/pack_model.cpp)
target_link_libraries(example_embed_mm_pack_model PRIVATE lc-example-embedding-mm)
target_include_directories(example_embed_mm_pack_model PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/examples)

luisa_compute_add_executable(example_embed_mm_dump_golden tools/dump_golden.cpp)
target_link_libraries(example_embed_mm_dump_golden PRIVATE lc-example-embedding-mm)
target_include_directories(example_embed_mm_dump_golden PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/examples)

luisa_compute_add_executable(example_embed_mm_text example_embed_mm_text.cpp)
target_link_libraries(example_embed_mm_text PRIVATE lc-example-embedding-mm)
target_include_directories(example_embed_mm_text PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/examples)

luisa_compute_add_executable(example_embed_mm_test example_embed_mm_test.cpp)
target_link_libraries(example_embed_mm_test PRIVATE lc-example-embedding-mm)
target_include_directories(example_embed_mm_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/examples)

luisa_compute_add_executable(example_embed_mm_vision example_embed_mm_vision.cpp)
target_link_libraries(example_embed_mm_vision PRIVATE lc-example-embedding-mm)
target_include_directories(example_embed_mm_vision PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/examples)

luisa_compute_add_executable(example_embed_mm_audio example_embed_mm_audio.cpp)
target_link_libraries(example_embed_mm_audio PRIVATE lc-example-embedding-mm)
target_include_directories(example_embed_mm_audio PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/examples)

luisa_compute_add_executable(example_embed_mm_video example_embed_mm_video.cpp)
target_link_libraries(example_embed_mm_video PRIVATE lc-example-embedding-mm)
target_include_directories(example_embed_mm_video PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/examples)

luisa_compute_add_executable(example_embed_mm_bench example_embed_mm_bench.cpp)
target_link_libraries(example_embed_mm_bench PRIVATE lc-example-embedding-mm)
target_include_directories(example_embed_mm_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/examples)
