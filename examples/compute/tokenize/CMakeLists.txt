luisa_compute_add_executable(tokenizer
    main.cpp
    cli.cpp
    tests.cpp
    bm25_scorer.cpp
    file_builder.cpp
    inverted_index.cpp
    lambda_mart.cpp
    levenshtein_automaton.cpp
    mmr_rerank.cpp
    ngram_tokenizer.cpp
    phonetic.cpp
    porter_stemmer.cpp
    query_performance_predictor.cpp
    rm3_expander.cpp
    searcher.cpp
    simhash.cpp
    string_similarity.cpp
)

target_include_directories(tokenizer PRIVATE
    ${CMAKE_SOURCE_DIR}/src/tests/
    ${CMAKE_SOURCE_DIR}/examples/
)
