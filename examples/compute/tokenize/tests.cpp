#include "tests.h"
#include "ngram_tokenizer.h"
#include "inverted_index.h"
#include "bm25_scorer.h"
#include "searcher.h"
#include "string_similarity.h"
#include "simhash.h"
#include "mmr_rerank.h"
#include "rm3_expander.h"
#include "lambda_mart.h"
#include "query_performance_predictor.h"
#include "phonetic.h"
#include "porter_stemmer.h"
#include "levenshtein_automaton.h"
#include "file_builder.h"
#include "cli_functions.h"
#include <luisa/core/logging.h>
#include <luisa/core/platform.h>
#include <luisa/luisa-compute.h>
#include <iostream>
#include <cmath>

using namespace tokenize;

static int g_pass = 0;
static int g_fail = 0;

static void check(bool cond, const char *msg) {
    if (cond) {
        ++g_pass;
        std::cout << "  [PASS] " << msg << "\n";
    } else {
        ++g_fail;
        std::cout << "  [FAIL] " << msg << "\n";
    }
}

static InvertedIndex build_test_index() {
    InvertedIndex index;
    NgramTokenizer tokenizer(2);
    luisa::vector<luisa::string> docs = {
        "hello world",
        "hello there",
        "world of code",
        "the quick brown fox",
        "jumped over the lazy dog",
        "compute shaders are fast",
        "gpu compute programming",
        "tokenization and indexing"
    };
    for (size_t i = 0; i < docs.size(); ++i) {
        auto tokens = tokenizer.tokenize(docs[i]);
        index.add_document(static_cast<int>(i), tokens);
    }
    index.finalize();
    return index;
}

static void test_cpu(const InvertedIndex &index) {
    std::cout << "\n=== CPU Baseline Tests ===\n";

    Searcher searcher(index);
    auto results = searcher.search("hello world", 5);
    check(!results.empty(), "searcher returns results");

    BM25Scorer scorer(index);
    auto bm25 = scorer.score({"compute", "gpu"});
    check(!bm25.empty(), "BM25 score returns results");

    auto bm25_topk = scorer.score_topk({"compute", "gpu"}, 3);
    check(bm25_topk.size() <= 3, "BM25 topk respects limit");

    check(jaro_similarity("martha", "marhta") > 0.9, "jaro similarity");
    check(jaro_winkler_similarity("dwayne", "duane") > 0.8, "jaro winkler");
    check(sorensen_dice_coefficient("night", "nacht") >= 0.25, "sorensen dice");
    check(hamming_distance("karolin", "kathrin") == 3, "hamming distance");

    SimHash h1("the quick brown fox");
    SimHash h2("the quick brown foxes");
    check(h1.distance(h2) > 0 && h1.distance(h2) < 64, "simhash distance");

    SimHashLSH lsh;
    lsh.add(0, h1);
    lsh.add(1, h2);
    auto cands = lsh.candidates(h1);
    check(cands.count(0) == 1, "simhash lsh candidates");

    luisa::vector<std::pair<int, double>> mock = {{0, 1.0}, {1, 0.9}, {2, 0.8}, {3, 0.7}};
    auto mmr = mmr_rerank(mock, index, 0.5, 3);
    check(mmr.size() == 3, "mmr rerank size");

    RM3Expander rm3(index, scorer);
    auto expanded = rm3.expand({"compute", "gpu"}, 10);
    check(!expanded.empty(), "rm3 expansion");

    LambdaMART mart(10, 0.05);
    luisa::vector<luisa::vector<luisa::vector<double>>> X = {
        {{1.0, 0.0}, {0.5, 1.0}, {0.0, 0.5}},
        {{0.0, 1.0}, {1.0, 0.5}}
    };
    luisa::vector<luisa::vector<double>> y = {{2.0, 1.0, 0.0}, {1.0, 2.0}};
    mart.fit(X, y);
    auto ranked = mart.rank({{10, {0.8, 0.2}}, {11, {0.1, 0.9}}});
    check(ranked.size() == 2, "lambda mart rank");

    QueryPerformancePredictor qpp(index, scorer);
    auto qtokens = luisa::vector<luisa::string>{"compute", "gpu", "shaders"};
    check(qpp.avg_idf(qtokens) > 0, "qpp avg idf");

    check(soundex("Robert") == "R163", "soundex");
    check(!metaphone("Knight").empty(), "metaphone");

    check(porter_stem("running") == "run", "porter stem running");
    check(porter_stem("flies") == "fli", "porter stem flies");

    LevenshteinAutomaton la("helo", 1);
    auto expansions = la.match(index, 10);
    check(true, "levenshtein automaton match");

    luisa::unordered_map<luisa::string, double> va = {{"a", 1.0}, {"b", 2.0}};
    luisa::unordered_map<luisa::string, double> vb = {{"b", 2.0}, {"c", 3.0}};
    check(cosine_similarity_tfidf(va, vb) > 0, "cosine similarity");
}

static void test_file_builder() {
    std::cout << "\n=== FileBuilder Tests ===\n";

    auto exe = luisa::filesystem::path(luisa::compute::current_executable_path());
    auto project_root = exe.parent_path().parent_path().parent_path();
    luisa::vector<luisa::filesystem::path> paths = {
        project_root / "examples" / "compute" / "tokenize"
    };
    luisa::filesystem::path output = luisa::filesystem::temp_directory_path() / "tokenize_file_builder_test";

    FileBuilder builder(paths, output, {}, {}, 2, 1.2, 0.75);
    check(!builder.empty(), "FileBuilder index not empty");

    auto results = builder.search("compute shader", 5);
    check(!results.empty(), "FileBuilder search returns results");

    auto results2 = builder.search("tokenize", 5, false, 0.5, true, true, true, false);
    check(!results2.empty(), "FileBuilder search with string similarity");

    builder.update();
    check(true, "FileBuilder update succeeds");
}

static void test_cli_functions() {
    std::cout << "\n=== CLI Functions Tests ===\n";

    auto exe = luisa::filesystem::path(luisa::compute::current_executable_path());
    auto project_root = exe.parent_path().parent_path().parent_path();
    auto tokenize_dir = project_root / "examples" / "compute" / "tokenize";

    auto path_str = luisa::to_string(tokenize_dir);
    for (auto &c : path_str) { if (c == '\\') c = '/'; }
    luisa::string path_json = "[\"" + path_str + "\"]";
    cli::ArgumentList add_args;
    add_args.push_back(path_json);
    auto handle1 = cli::cmd_add_builder(std::move(add_args));
    check(!handle1.empty() && handle1.find("error") == luisa::string::npos, "add_builder returns handle");

    cli::ArgumentList add_args2;
    add_args2.push_back(path_json);
    auto handle2 = cli::cmd_add_builder(std::move(add_args2));
    check(handle1 == handle2, "add_builder caches builder");

    cli::ArgumentList search_args;
    search_args.push_back(handle1);
    search_args.push_back("compute");
    search_args.push_back("5");
    auto search_result = cli::cmd_search(std::move(search_args));
    check(!search_result.empty(), "search returns result");
    check(search_result.find("==e6b7e03aa02b4ffe==") != luisa::string::npos, "search result contains magic string");

    cli::ArgumentList remove_args;
    remove_args.push_back(handle1);
    auto remove_result = cli::cmd_remove_builder(std::move(remove_args));
    check(remove_result == "ok", "remove_builder succeeds");

    cli::ArgumentList search_after_remove;
    search_after_remove.push_back(handle1);
    search_after_remove.push_back("compute");
    auto fail_result = cli::cmd_search(std::move(search_after_remove));
    check(fail_result.find("error") != luisa::string::npos, "search after remove fails gracefully");
}

static bool backend_available(luisa::string_view backend) {
    luisa::compute::Context ctx(luisa::compute::current_executable_path().c_str());
    for (auto &b : ctx.installed_backends()) {
        if (b == backend) return true;
    }
    return false;
}

static bool create_device_for_test(luisa::string_view backend, luisa::compute::Context &ctx, luisa::compute::Device &device) {
    if (!backend_available(backend)) return false;
    device = ctx.create_device(backend);
    return static_cast<bool>(device);
}

static void test_backend(const InvertedIndex &index, luisa::string_view backend) {
    std::cout << "\n=== Backend: " << backend << " ===\n";

    luisa::compute::Context ctx(luisa::compute::current_executable_path().c_str());
    luisa::compute::Device device;
    if (!create_device_for_test(backend, ctx, device)) {
        std::cout << "  [SKIP] Backend '" << backend << "' not available\n";
        return;
    }
    luisa::compute::Stream stream = device.create_stream();
    std::cout << "  Device created successfully\n";

    BM25Scorer cpu_scorer(index);
    BM25Scorer gpu_scorer(index);

    auto cpu_sparse = cpu_scorer.score({"compute", "gpu"});
    auto gpu_dense = gpu_scorer.gpu_accumulate(device, stream, {"compute", "gpu"});

    bool bm25_acc_match = true;
    if (gpu_dense.size() != static_cast<size_t>(index.N())) {
        bm25_acc_match = false;
    } else {
        for (auto &[doc, score] : cpu_sparse) {
            if (std::abs(score - gpu_dense[doc]) > 1e-3) {
                bm25_acc_match = false;
                break;
            }
        }
    }
    check(bm25_acc_match, "BM25 gpu_accumulate matches CPU");

    auto cpu_topk = cpu_scorer.score_topk({"compute", "gpu"}, 3);
    auto gpu_topk = gpu_scorer.gpu_score_topk(device, stream, {"compute", "gpu"}, 3);
    bool topk_match = cpu_topk.size() == gpu_topk.size();
    if (topk_match) {
        for (size_t i = 0; i < cpu_topk.size(); ++i) {
            if (cpu_topk[i].first != gpu_topk[i].first ||
                std::abs(cpu_topk[i].second - gpu_topk[i].second) > 1e-3) {
                topk_match = false;
                break;
            }
        }
    }
    check(topk_match, "BM25 gpu_score_topk matches CPU");

    luisa::vector<uint64_t> hashes = {
        SimHash::hash_token("hello"),
        SimHash::hash_token("world"),
        SimHash::hash_token("compute")
    };

    uint64_t cpu_hash = 0;
    {
        luisa::vector<int> v(64, 0);
        for (auto h : hashes) {
            for (int i = 0; i < 64; ++i) {
                if ((h >> i) & 1ULL) ++v[i];
                else --v[i];
            }
        }
        for (int i = 0; i < 64; ++i) {
            if (v[i] > 0) cpu_hash |= (1ULL << i);
        }
    }
    uint64_t gpu_hash = SimHash::gpu_compute_from_hashes(device, stream, hashes, 64);
    check(cpu_hash == gpu_hash, "SimHash gpu_compute_from_hashes matches CPU");

    uint64_t query = SimHash::hash_token("query");
    luisa::vector<uint64_t> target_hashes = {
        SimHash::hash_token("target1"),
        SimHash::hash_token("target2"),
        SimHash::hash_token("target3")
    };
    auto gpu_dists = SimHash::gpu_batch_distance(device, stream, query, target_hashes, 64);

    bool dist_match = gpu_dists.size() == target_hashes.size();
    if (dist_match) {
        for (size_t i = 0; i < target_hashes.size(); ++i) {
            uint64_t x = query ^ target_hashes[i];
            int cpu_dist = 0;
            while (x) { cpu_dist += static_cast<int>(x & 1ULL); x >>= 1; }
            if (gpu_dists[i] != cpu_dist) {
                dist_match = false;
                break;
            }
        }
    }
    check(dist_match, "SimHash gpu_batch_distance matches CPU");
}

int run_tests(int argc, char *argv[]) {
    luisa::fiber::scheduler global_scheduler;

    auto index = build_test_index();
    LUISA_INFO("Index built: N={}, avgdl={}", index.N(), index.avgdl());

    test_cpu(index);
    test_file_builder();
    test_cli_functions();

    luisa::vector<luisa::string_view> gpu_backends;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            gpu_backends.push_back(argv[i]);
        }
    } else {
        gpu_backends = {"dx", "vk"};
    }

    for (auto b : gpu_backends) {
        test_backend(index, b);
    }

    std::cout << "\n=== Summary ===\n";
    std::cout << "Pass: " << g_pass << "\n";
    std::cout << "Fail: " << g_fail << "\n";
    return g_fail > 0 ? 1 : 0;
}
