// N-gram TRAINING demo: "training = counting" on device, aligned with the
// retrieval inference logic.
//
// Part 1 (retrieval-aligned): trains an n-gram count table over the demo
// corpus with the retriever's own [min_n, max_n] window rule, then shows
//   - the trained index reproduces the current inference exactly (K1/K2/K3
//     drafts unchanged, K3 running directly on the trained table);
//   - per-query draft confidence scores (Add-k log-probs from the counts);
//   - the parallel_mle variant proposing the MOST FREQUENT continuation
//     (MLE argmax, earliest position on ties) among the matches of the
//     winning length.
//
// Part 2 (LM mode, Python BigramModel mirror): BOS/EOS/UNK preprocessing,
// Add-k smoothing, sentence_prob and perplexity on the corpus
//   我 爱 北京 / 我 爱 学习 / 你 爱 北京
// cross-checked against the CPU oracle ReferenceNgramModel, plus a small
// English backoff-mode demo on the retrieval demo corpus.
//
// Usage: tokenizer <backend> --train

#include "ngram_library.h"
#include "ngram_retriever.h"
#include "ngram_trainer.h"
#include "reference.h"
#include "reference_model.h"

#include <luisa/core/logging.h>
#include <luisa/luisa-compute.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

using namespace tokenize;

uint32_t g_mismatches = 0;

void check_close(const char *name, double got, double expected,
                 double tol = 1e-5) {
    const bool ok = std::abs(got - expected) <= tol * std::max(1.0, std::abs(expected));
    if (!ok) ++g_mismatches;
    LUISA_INFO("  {} = {:.6f} (oracle {:.6f}) {}", name, got, expected,
               ok ? "ok" : "MISMATCH");
}

void check_ids(const char *name, luisa::span<const uint32_t> got,
               luisa::span<const uint32_t> expected) {
    const bool ok = got.size() == expected.size() &&
                    std::equal(got.begin(), got.end(), expected.begin());
    if (!ok) ++g_mismatches;
    auto fmt = [](luisa::span<const uint32_t> v) {
        luisa::string s;
        for (auto t : v) s.append(luisa::format("{} ", t));
        return s;
    };
    LUISA_INFO("  {}: [{}] vs oracle [{}] {}", name, fmt(got), fmt(expected),
               ok ? "ok" : "MISMATCH");
}

// ---------- Part 1: retrieval-aligned training ----------

void demo_retrieval_aligned(luisa::compute::Device &device,
                            luisa::compute::Stream &stream) {
    constexpr uint32_t min_n = 2, max_n = 3, k = 4;
    constexpr float add_k = 0.1f;

    NgramLibrary library;
    library.add_document("the quick brown fox jumps over the lazy dog");
    library.add_document("the quick brown fox runs and the quick brown fox sleeps");
    library.add_document("gpu compute shaders are fast and gpu compute shaders are fun");
    library.finalize();
    LUISA_INFO("[part 1] library: {} tokens, {} docs, vocab {}",
               library.size(), library.num_docs(), library.vocab_size);

    // train: build the count index (host) + count on device
    NgramTrainOptions opts;
    opts.min_n = min_n;
    opts.max_n = max_n;
    opts.add_k = add_k;
    NgramTrainer trainer{device, stream, library, opts};

    // host ground-truth counts + a peek at the trained statistics
    const auto counts = reference_count_ngrams(luisa::span{library.tokens}, min_n, max_n);
    const uint32_t t_the = library.find_token("the");
    const uint32_t t_quick = library.find_token("quick");
    const uint32_t t_brown = library.find_token("brown");
    const uint32_t t_fox = library.find_token("fox");
    const uint32_t t_runs = library.find_token("runs");
    const uint32_t t_jumps = library.find_token("jumps");
    const luisa::vector<uint32_t> ctx = {t_quick, t_brown, t_fox};
    const luisa::vector<uint32_t> cont_runs = {t_quick, t_brown, t_fox, t_runs};
    const luisa::vector<uint32_t> cont_jumps = {t_quick, t_brown, t_fox, t_jumps};
    LUISA_INFO("  trained counts: count(quick brown fox) = {}, "
               "count(... -> runs) = {}, count(... -> jumps) = {}",
               reference_count(counts, luisa::span{ctx}),
               reference_count(counts, luisa::span{cont_runs}),
               reference_count(counts, luisa::span{cont_jumps}));

    // queries: trailing token sequences taken from the corpus
    const auto &tokens = library.tokens;
    luisa::vector<luisa::vector<uint32_t>> queries;
    queries.emplace_back(tokens.end() - 4, tokens.end());
    queries.emplace_back(tokens.end() - 2, tokens.end());
    queries.emplace_back(tokens.begin() + 3, tokens.begin() + 9);
    uint32_t max_query_len = 0;
    for (auto &q : queries) max_query_len = std::max(max_query_len, (uint32_t)q.size());

    // the existing variants must be byte-identical with or without training;
    // K3 also runs directly on the trained count index
    struct VariantRun {
        const char *name;
        NgramKernelVariant variant;
        const NgramCountIndex *index;
        bool mle;
    };
    const VariantRun runs[] = {
        {"naive               ", NgramKernelVariant::naive, nullptr, false},
        {"parallel            ", NgramKernelVariant::parallel, nullptr, false},
        {"hash (own index)    ", NgramKernelVariant::hash, nullptr, false},
        {"hash (trained index)", NgramKernelVariant::hash, &trainer.index(), false},
        {"parallel_mle        ", NgramKernelVariant::parallel_mle, &trainer.index(), true},
    };
    const uint32_t stride = std::max(max_n, max_query_len);
    for (auto &run : runs) {
        NgramRetriever retriever{device, stream, library, min_n, max_n, k,
                                 max_query_len, queries.size(), run.variant,
                                 512u, run.index};
        luisa::vector<uint32_t> drafts, draft_lens;
        retriever.retrieve(luisa::span{queries}, drafts, draft_lens);
        for (size_t i = 0; i < queries.size(); ++i) {
            auto expected = run.mle
                                ? reference_retrieve_mle(luisa::span{tokens}, counts,
                                                         luisa::span{queries[i]}, min_n, max_n, k)
                                : reference_retrieve(luisa::span{tokens},
                                                     luisa::span{queries[i]}, min_n, max_n, k);
            luisa::vector<uint32_t> got(drafts.begin() + i * k,
                                        drafts.begin() + i * k + draft_lens[i]);
            check_ids(luisa::format("  [{}] query {}", run.name, i).c_str(),
                      luisa::span{got}, luisa::span{expected});
        }
        // draft confidence scores from the trained counts (device), checked
        // against the host oracle
        luisa::vector<uint32_t> flat(queries.size() * stride, 0u);
        luisa::vector<uint32_t> lens(queries.size());
        for (size_t i = 0; i < queries.size(); ++i) {
            lens[i] = static_cast<uint32_t>(queries[i].size());
            std::copy(queries[i].begin(), queries[i].end(), flat.begin() + i * stride);
        }
        luisa::vector<float> log2_probs;
        trainer.score_drafts(luisa::span{flat}, luisa::span{lens}, stride,
                             luisa::span{drafts}, luisa::span{draft_lens}, k,
                             log2_probs);
        for (size_t i = 0; i < queries.size(); ++i) {
            luisa::vector<uint32_t> draft(drafts.begin() + i * k,
                                          drafts.begin() + i * k + draft_lens[i]);
            const float expected_lp = reference_draft_log2prob(
                counts, library.vocab_size, luisa::span{queries[i]},
                luisa::span{draft}, max_n, add_k);
            check_close(luisa::format("  [{}] query {} draft log2prob",
                                      run.name, i)
                            .c_str(),
                        log2_probs[i], expected_lp, 1e-4);
        }
    }

    // MLE continuation showcase: add one more "the quick brown fox runs" so
    // the continuation "runs" becomes the most frequent (count 2 vs 1); the
    // classic kernels still propose the EARLIEST match ("jumps"), while
    // parallel_mle proposes the most frequent one ("runs").
    NgramLibrary library2;
    library2.add_document("the quick brown fox jumps over the lazy dog");
    library2.add_document("the quick brown fox runs and the quick brown fox sleeps");
    library2.add_document("gpu compute shaders are fast and gpu compute shaders are fun");
    library2.add_document("the quick brown fox runs");
    library2.finalize();
    NgramTrainer trainer2{device, stream, library2, opts};
    const auto counts2 = reference_count_ngrams(luisa::span{library2.tokens}, min_n, max_n);
    luisa::vector<luisa::vector<uint32_t>> q2;
    // trailing 3-gram = (quick, brown, fox)
    q2.emplace_back(library2.tokens.begin(), library2.tokens.begin() + 4);
    luisa::vector<uint32_t> d_par, l_par, d_mle, l_mle;
    {
        NgramRetriever r_par{device, stream, library2, min_n, max_n, k,
                             4u, 1u, NgramKernelVariant::parallel};
        r_par.retrieve(luisa::span{q2}, d_par, l_par);
        NgramRetriever r_mle{device, stream, library2, min_n, max_n, k,
                             4u, 1u, NgramKernelVariant::parallel_mle,
                             512u, &trainer2.index()};
        r_mle.retrieve(luisa::span{q2}, d_mle, l_mle);
    }
    auto expected_mle = reference_retrieve_mle(luisa::span{library2.tokens}, counts2,
                                               luisa::span{q2[0]}, min_n, max_n, k);
    auto expected_par = reference_retrieve(luisa::span{library2.tokens},
                                           luisa::span{q2[0]}, min_n, max_n, k);
    check_ids("  parallel    draft (earliest match)",
              luisa::span{d_par.data(), l_par[0]}, luisa::span{expected_par});
    check_ids("  parallel_mle draft (most frequent continuation)",
              luisa::span{d_mle.data(), l_mle[0]}, luisa::span{expected_mle});
    LUISA_INFO("  -> MLE picks \"runs\" (count 2) over \"jumps\" (count 1): {}",
               !d_mle.empty() && d_mle[0] == t_runs ? "ok" : "note: ids differ");
    if (!d_mle.empty() && d_mle[0] != t_runs) ++g_mismatches;
}

// ---------- Part 2: LM mode (Python BigramModel mirror) ----------

void demo_lm_mode(luisa::compute::Device &device,
                  luisa::compute::Stream &stream) {
    NgramLibrary zh;
    zh.add_document("我 爱 北京");
    zh.add_document("我 爱 学习");
    zh.add_document("你 爱 北京");
    zh.finalize();
    LUISA_INFO("[part 2] corpus: {} tokens, {} sentences, vocab {}",
               zh.size(), zh.num_docs(), zh.vocab_size);

    NgramTrainOptions opts;
    opts.lm_mode = true;
    opts.order = 2;
    opts.add_k = 0.1f;
    opts.smoothing = NgramSmoothing::add_k;
    NgramTrainer lm{device, stream, zh, opts};
    LUISA_INFO("  special ids: <unk>={} <s>={} </s>={} (V={}), padded corpus {} tokens",
               lm.unk_id(), lm.bos_id(), lm.eos_id(), lm.vocab_effective(),
               lm.total_tokens());

    // CPU oracle over the same padded corpus
    ReferenceNgramModel ref;
    ref.train(lm.padded_corpus(), lm.order(), lm.bos_id());

    const uint32_t wo = lm.lm_remap(zh.find_token("我"));
    const uint32_t ai = lm.lm_remap(zh.find_token("爱"));
    const uint32_t bei = lm.lm_remap(zh.find_token("北"));
    const uint32_t jing = lm.lm_remap(zh.find_token("京"));
    const uint32_t xue = lm.lm_remap(zh.find_token("学"));
    const uint32_t ni = lm.lm_remap(zh.find_token("你"));
    const float V = static_cast<float>(lm.vocab_effective());
    const luisa::vector<uint32_t> row_wo_ai = {wo, ai};
    const luisa::vector<uint32_t> row_ni_xue = {ni, xue};

    // MLE sanity from the oracle counts: count(我)=2, count(我,爱)=2 -> P=1
    LUISA_INFO("  oracle counts: count(我)={}, count(我,爱)={} -> MLE P(爱|我)=1.0",
               ref.count(luisa::span{&wo, 1u}),
               ref.count(luisa::span{row_wo_ai}));

    // P(爱|我) with Add-k smoothing
    check_close("P(爱|我) [Add-k k=0.1]",
                lm.conditional_prob(luisa::span{&wo, 1u}, ai),
                std::exp2(ref.sentence_log2prob(luisa::span{row_wo_ai},
                                                NgramSmoothing::add_k, opts.add_k, V)));

    // smoothed non-zero P(学习|你) for the unseen bigram (你,学习)
    check_close("P(学习|你) [Add-k, unseen bigram]",
                lm.conditional_prob(luisa::span{&ni, 1u}, xue),
                std::exp2(ref.sentence_log2prob(luisa::span{row_ni_xue},
                                                NgramSmoothing::add_k, opts.add_k, V)));

    // sentence_prob([<s>,我,爱,北,京,</s>]) — 北京 splits into two CJK tokens
    const luisa::vector<uint32_t> sent = {lm.bos_id(), wo, ai, bei, jing, lm.eos_id()};
    const luisa::vector<uint32_t> sent_len = {static_cast<uint32_t>(sent.size())};
    luisa::vector<float> lp;
    lm.score_sentences(luisa::span{sent}, luisa::span{sent_len}, lp);
    check_close("sentence_prob(<s> 我 爱 北 京 </s>)",
                std::exp2(lp[0]),
                std::exp2(ref.sentence_log2prob(luisa::span{sent},
                                                NgramSmoothing::add_k, opts.add_k, V)));

    // perplexity of the 3-sentence corpus
    const luisa::vector<luisa::vector<uint32_t>> sents = {
        {lm.bos_id(), wo, ai, bei, jing, lm.eos_id()},
        {lm.bos_id(), wo, ai, xue, lm.lm_remap(zh.find_token("习")), lm.eos_id()},
        {lm.bos_id(), ni, ai, bei, jing, lm.eos_id()},
    };
    luisa::vector<uint32_t> flat, lens;
    for (auto &s : sents) {
        flat.insert(flat.end(), s.begin(), s.end());
        lens.push_back(static_cast<uint32_t>(s.size()));
    }
    check_close("corpus perplexity",
                lm.perplexity(luisa::span{flat}, luisa::span{lens}),
                ref.perplexity(luisa::span{sents}, NgramSmoothing::add_k,
                               opts.add_k, V));

    // small English backoff-mode demo on the retrieval demo corpus
    NgramLibrary en;
    en.add_document("the quick brown fox jumps over the lazy dog");
    en.add_document("the quick brown fox runs and the quick brown fox sleeps");
    en.add_document("gpu compute shaders are fast and gpu compute shaders are fun");
    en.finalize();
    NgramTrainOptions en_opts;
    en_opts.lm_mode = true;
    en_opts.order = 2;
    en_opts.smoothing = NgramSmoothing::backoff;
    NgramTrainer lm_en{device, stream, en, en_opts};
    ReferenceNgramModel ref_en;
    ref_en.train(lm_en.padded_corpus(), lm_en.order(), lm_en.bos_id());
    const uint32_t t_brown = lm_en.lm_remap(en.find_token("brown"));
    const uint32_t t_fox = lm_en.lm_remap(en.find_token("fox"));
    const uint32_t t_dog = lm_en.lm_remap(en.find_token("dog"));
    const luisa::vector<uint32_t> row_brown_fox = {t_brown, t_fox};
    const luisa::vector<uint32_t> row_fox_dog = {t_fox, t_dog};
    check_close("P(fox|brown) [backoff, seen]",
                lm_en.conditional_prob(luisa::span{&t_brown, 1u}, t_fox),
                std::exp2(ref_en.sentence_log2prob(
                    luisa::span{row_brown_fox},
                    NgramSmoothing::backoff, en_opts.add_k,
                    static_cast<float>(lm_en.vocab_effective()))));
    check_close("P(dog|fox) [backoff to unigram]",
                lm_en.conditional_prob(luisa::span{&t_fox, 1u}, t_dog),
                std::exp2(ref_en.sentence_log2prob(
                    luisa::span{row_fox_dog},
                    NgramSmoothing::backoff, en_opts.add_k,
                    static_cast<float>(lm_en.vocab_effective()))));
}

}// namespace

int run_train_demo(int argc, char *argv[]) {
    if (argc < 2 || argv[1] == nullptr || argv[1][0] == '\0') {
        LUISA_ERROR("train demo requires a backend argument");
    }
    luisa::compute::Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream();

    demo_retrieval_aligned(device, stream);
    demo_lm_mode(device, stream);

    if (g_mismatches != 0u) {
        LUISA_WARNING("train demo: {} mismatches against the oracle", g_mismatches);
        return 1;
    }
    LUISA_INFO("train demo: all checks match the reference oracles");
    return 0;
}
