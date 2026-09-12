luisa_compute_add_executable(tokenizer
    main.cpp
    ngram_tokenizer.cpp
    ngram_library.cpp
    ngram_retriever.cpp
    ngram_kernels.cpp
    test_tokenize.cpp
    benchmark_tokenize.cpp
)

target_include_directories(tokenizer PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tests/
    ${CMAKE_SOURCE_DIR}/src/tests/common/
    ${CMAKE_SOURCE_DIR}/examples/
)

target_link_libraries(tokenizer PRIVATE luisa-compute-runtime luisa-compute-dsl luisa-compute-vstl)
