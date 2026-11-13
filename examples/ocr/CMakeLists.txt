# OCR engine (luisa::ocr) -- Phase 2: portable tensor runtime and
# correctness-first operators on top of the Phase 1 host artifacts.
# No model is validated yet; see MODELS.md.
# Host-only engine TUs must not include DSL device headers; the runtime/ops
# device TUs are the only place DSL factories live (device-free factories).

add_library(lc-example-ocr STATIC
    ocr/status.cpp
    ocr/model_recipe.cpp
    ocr/model_registry.cpp
    ocr/session.cpp
    ocr/cli.cpp
    ocr/quantization.cpp
    ocr/package.cpp
    ocr/safetensors_reader.cpp
    ocr/tokenizer.cpp
    ocr/prompt.cpp
    ocr/detokenizer.cpp
    ocr/image.cpp
    ocr/document.cpp
    ocr/preprocess.cpp
    ocr/feature_bundle.cpp
    ocr/output/json_output.cpp
    # Host-safe benchmark options/statistics/reporting (no device/fixture dependency)
    ocr/benchmark.cpp
    # Phase 4 structured outputs (Markdown/table/LaTeX/JSON extraction)
    ocr/output/markdown.cpp
    ocr/output/table.cpp
    ocr/output/extract.cpp
    # Phase 2 runtime (tensor/weights/workspace/shader_cache/dispatch/budget)
    ocr/runtime/tensor.cpp
    ocr/runtime/weights.cpp
    ocr/runtime/workspace.cpp
    ocr/runtime/shader_cache.cpp
    ocr/runtime/dispatch.cpp
    ocr/runtime/memory_budget.cpp
    # Phase 2 operators (device-free DSL factories + independent host oracles)
    ocr/ops/cpu_reference.cpp
    ocr/ops/ops_image.cpp
    ocr/ops/ops_gemm.cpp
    ocr/ops/ops_norm.cpp
    ocr/ops/ops_rope.cpp
    ocr/ops/ops_attention.cpp
    ocr/ops/ops_conv.cpp
    ocr/ops/ops_vision.cpp
    ocr/ops/ops_decode.cpp
    # Phase 9 GDN operators (device-free DSL factories + independent host oracles)
    ocr/ops/ops_gdn.cpp
    # Phase 10 MoE operators (device-free DSL factories + independent host oracles)
    ocr/ops/ops_moe.cpp
    # Phase 3 dense decoder core (shared components, KV cache, executor)
    ocr/models/qwen2_5_vl.cpp
    ocr/runtime/decode_cache.cpp
    ocr/runtime/executor.cpp
    ocr/runtime/dense_vlm_executor.cpp
    # Phase 9 hybrid GDN + periodic full-attention decoder executor + cache
    ocr/runtime/hybrid_gdn_config.cpp
    ocr/runtime/hybrid_gdn_executor.cpp
    # Phase 9 model vertical slices (synthetic; real artifacts fail closed).
    ocr/models/ovis_ocr2.cpp
    ocr/models/surya.cpp
    ocr/runtime/surya_executor.cpp
    # Phase 10 DeepSeek-OCR-2 MoE vertical slice (synthetic; real artifacts fail closed).
    ocr/models/deepseek_ocr2.cpp
    ocr/runtime/moe_vlm_executor.cpp
    # Phase 4 Qwen3-VL vertical slice + FireRed-OCR recipe
    ocr/models/qwen3_vl.cpp
    ocr/models/fire_red_ocr.cpp
    ocr/runtime/qwen3_vl_executor.cpp
    ocr/runtime/qwen3_vl_preprocess.cpp
    # Phase 5 CNN OCR: PP-OCRv6 contract, CTC ops, DB/CTC/reading-order output,
    # and the conventional detector->crop->recognizer->CTC executor.
    ocr/models/pp_ocrv6.cpp
    ocr/ops/ops_ctc.cpp
    ocr/output/db_postprocess.cpp
    ocr/output/ctc_decode.cpp
    ocr/output/reading_order.cpp
    ocr/runtime/cnn_executor.cpp
    ocr/runtime/cnn_executor_result.cpp
    # Phase 6 layout/region infrastructure: PP-DocLayoutV2/V3, token-budgeted
    # region scheduling, PaddleOCR-VL (NaViT + reduction + connector + ERNIE),
    # and OpenDoc (PP-DocLayoutV2 + UniRec two-stage). Host composition bridges.
    ocr/runtime/scheduler.cpp
    ocr/models/pp_doc_layout.cpp
    ocr/models/paddle_ocr_vl.cpp
    ocr/models/open_doc.cpp
    ocr/runtime/layout_vlm_executor.cpp
    ocr/runtime/layout_result.cpp
    ocr/models/paddle_ocr_vl_result.cpp
    ocr/models/open_doc_result.cpp
    # Phase 7 custom dense/layout variants: shared configurable slice executor
    # (vision window/global + connectors + Qwen2/Qwen3/GLM/Hunyuan decoders).
    ocr/runtime/p7_slice_config.cpp
    ocr/runtime/p7_vlm_executor.cpp
    ocr/runtime/p7_vlm_preprocess.cpp
    # Phase 7 model contracts (synthetic; real artifacts fail closed).
    ocr/models/tele_ocr.cpp
    ocr/models/glm_ocr.cpp
    ocr/models/mineru.cpp
    ocr/models/hunyuan_ocr.cpp
    # Phase 8 remaining dense ViT models (synthetic; real artifacts fail closed).
    ocr/models/dots_mocr.cpp
    ocr/models/got_ocr2.cpp
    ocr/models/olm_ocr2.cpp
)
target_link_libraries(lc-example-ocr PUBLIC luisa::compute)
# yyjson interface, aligned with the embedding examples: link the bundled
# interface target when it exists. With a system yyjson there is no such
# target, but luisa::compute already propagates yyjson::yyjson through
# luisa-compute-ext, so the dependency is still satisfied (not optional).
if (TARGET luisa-compute-ext-yyjson-interface)
    target_link_libraries(lc-example-ocr PUBLIC luisa-compute-ext-yyjson-interface)
endif ()
# stb_image (checked JPEG/PNG decode). The compiled stb object lives in
# luisa-compute-ext-stb; link it so the example/tests resolve stbi_* symbols.
if (TARGET luisa-compute-ext-stb)
    target_link_libraries(lc-example-ocr PUBLIC luisa-compute-ext-stb)
endif ()
target_include_directories(lc-example-ocr PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/examples)

luisa_compute_add_executable(example_ocr example_ocr.cpp)
target_link_libraries(example_ocr PRIVATE lc-example-ocr)
target_include_directories(example_ocr PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/examples)

# Benchmark-only device support. This target (not lc-example-ocr) privately
# includes the synthetic test fixtures, keeping production/library dependencies
# acyclic and ensuring benchmark-only code cannot enter the public OCR CLI.
add_library(lc-example-ocr-bench STATIC
    bench/benchmark_device.cpp
    bench/benchmark_ops.cpp
    bench/benchmark_decode.cpp
    bench/benchmark_cnn_layout.cpp
    bench/benchmark_gdn_moe.cpp
    bench/benchmark_qwen.cpp
    bench/benchmark_layout_models.cpp
    bench/benchmark_p7_models.cpp
    bench/benchmark_surya.cpp
)
target_link_libraries(lc-example-ocr-bench PUBLIC lc-example-ocr)
target_include_directories(lc-example-ocr-bench
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/tests
        ${CMAKE_SOURCE_DIR}/src/tests/integration/ocr)
set(LUISA_OCR_BENCHMARK_SOURCE_REVISION "unknown")
find_package(Git QUIET)
if (GIT_EXECUTABLE)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE LUISA_OCR_BENCHMARK_SOURCE_REVISION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
endif ()
target_compile_definitions(lc-example-ocr-bench PRIVATE
    LUISA_OCR_BENCHMARK_SOURCE_REVISION=${LUISA_OCR_BENCHMARK_SOURCE_REVISION}
    LUISA_OCR_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/src/tests/data/ocr"
    LUISA_OCR_EXAMPLES_DIR="${CMAKE_SOURCE_DIR}/examples")

luisa_compute_add_executable(example_ocr_bench example_ocr_bench.cpp)
target_link_libraries(example_ocr_bench PRIVATE lc-example-ocr-bench)
target_include_directories(example_ocr_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
