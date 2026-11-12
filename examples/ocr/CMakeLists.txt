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
    # Phase 3 dense decoder core (shared components, KV cache, executor)
    ocr/models/qwen2_5_vl.cpp
    ocr/runtime/decode_cache.cpp
    ocr/runtime/executor.cpp
    ocr/runtime/dense_vlm_executor.cpp
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

# example_ocr_bench is reserved for the performance/budget harness phase
# (plan Phase 11); its source does not exist yet and is intentionally not
# referenced here.
