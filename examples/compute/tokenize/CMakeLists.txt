luisa_compute_add_executable(tokenizer
    main.cpp
    ngram_tokenizer.cpp
    ngram_library.cpp
    ngram_retriever.cpp
    ngram_kernels.cpp
    ngram_train_kernels.cpp
    ngram_trainer.cpp
    train_demo.cpp
    test_tokenize.cpp
    benchmark_tokenize.cpp
)

target_include_directories(tokenizer PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tests/
    ${CMAKE_SOURCE_DIR}/src/tests/common/
    ${CMAKE_SOURCE_DIR}/examples/
)

# NOTE: no extra target_link_libraries here. luisa_compute_add_executable
# already links the luisa::compute umbrella; linking luisa-compute-vstl (an
# OBJECT library) directly would additionally bake the vstd objects into the
# executable, leaving the bundled LMDB objects (private to luisa-runtime.dll)
# unresolved.
