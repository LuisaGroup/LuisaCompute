// Test for ocr/runtime/decode_cache + dense decode-step KV parity (Phase 3).
// Covers: contiguous KV capacity math from actual geometry, overflow/OOM
// refusal, append/read, EVERY decode-step KV content + logits vs the oracle,
// cache growth, immediate sequence release + reuse, device metadata, paged
// fail-closed, workspace bounds. Synthetic only; no model/download.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_decode_test_utils.h"

#include "ocr/runtime/decode_cache.h"
#include "ocr/runtime/dense_vlm_executor.h"
#include "ocr/runtime/memory_budget.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

constexpr float kAtol = 3e-4f;
constexpr float kRtol = 3e-4f;

KvCacheConfig syn_kv_config(uint32_t max_seq) {
    KvCacheConfig kc;
    kc.num_layers = ocr::test::kSynLayers;
    kc.num_kv_heads = ocr::test::kSynKvHeads;
    kc.head_dim = ocr::test::kSynHeadDim;
    kc.max_seq_len = max_seq;
    kc.batch = 1;
    kc.dtype_bytes = 4;
    return kc;
}

void test_capacity_math() {
    auto kc = syn_kv_config(64);
    expect(kc.validate().ok());
    auto bytes = kc.total_bytes();
    expect(bytes.ok());
    // layers * batch * tokens * kv_heads * head_dim * 2 * 4
    const uint64_t expect_bytes = 2ull * 1 * 64 * 2 * 4 * 2 * 4;
    expect(*bytes.value == expect_bytes);
    auto elems = kc.layer_elements();
    expect(elems.ok());
    expect(*elems.value == 64ull * 2 * 4);
    // Invalid geometry refused.
    auto bad = kc;
    bad.num_layers = 0;
    expect(!bad.validate().ok());
    bad = kc;
    bad.dtype_bytes = 2;
    expect(!bad.validate().ok());// FP32-only baseline
    // Overflow refused (huge token count).
    auto huge = kc;
    huge.max_seq_len = UINT32_MAX;
    huge.num_layers = UINT32_MAX;
    expect(!huge.total_bytes().ok());
}

void test_budget_preflight(Device &device) {
    auto stream = device.create_stream();
    MemoryBudget budget;
    MemoryLimits limits;
    limits.max_device_bytes = 1024;// far below the real requirement
    budget.set_limits(limits);
    ContiguousKvCache cache;
    auto s = cache.create(device, stream, syn_kv_config(64), &budget);
    expect(!s.ok());
    expect(s.code == StatusCode::out_of_memory);
    // A generous budget succeeds and records the resident bytes.
    MemoryBudget ok_budget;
    ContiguousKvCache ok_cache;
    expect(ok_cache.create(device, stream, syn_kv_config(64), &ok_budget).ok());
    expect(ok_budget.category_bytes(MemoryCategory::kv_cache) == ok_cache.device_bytes());
    expect(ok_cache.device_bytes() > 0u);
}

void test_append_release_reuse(Device &device) {
    auto stream = device.create_stream();
    ContiguousKvCache cache;
    expect(cache.create(device, stream, syn_kv_config(16)).ok());
    // Bind a sequence, append tokens, check growth.
    expect(cache.bind_sequence(0, 4).ok());
    expect(cache.length(0) == 4u);
    expect(cache.append_tokens(0, 3).ok());
    expect(cache.length(0) == 7u);
    // Overrun refused.
    expect(!cache.append_tokens(0, 100).ok());
    expect(cache.length(0) == 7u);// unchanged on failure
    // Occupied slot cannot be re-bound.
    expect(!cache.bind_sequence(0, 2).ok());
    // Release frees the slot immediately; re-bind works.
    cache.release(0);
    expect(cache.length(0) == 0u);
    expect(cache.bind_sequence(0, 2).ok());
    expect(cache.length(0) == 2u);
    // Prompt longer than capacity refused.
    cache.release(0);
    expect(!cache.bind_sequence(0, 17).ok());
    // Device metadata download reflects a recorded snapshot.
    {
        auto list = CommandList::create();
        luisa::vector<uint32_t> kv_len = {5}, step = {2}, state = {static_cast<uint32_t>(DecodeSeqState::decoding)};
        luisa::vector<int32_t> pos = {4};
        expect(cache.state().record(list, kv_len, pos, step, state).ok());
        stream << list.commit() << synchronize();
    }
    luisa::vector<uint32_t> kv_len, step, state;
    luisa::vector<int32_t> pos;
    expect(cache.state().download(stream, kv_len, pos, step, state).ok());
    stream << synchronize();
    expect(kv_len[0] == 5u);
    expect(pos[0] == 4);
    expect(step[0] == 2u);
    expect(state[0] == static_cast<uint32_t>(DecodeSeqState::decoding));
}

void test_paged_fails_closed(Device &device) {
    PagedKvCache paged;
    auto s = paged.create(device, syn_kv_config(64), 16);
    expect(!s.ok());
    expect(s.code == StatusCode::unsupported);
}

// Every decode-step KV content + logits vs the independent oracle.
void test_decode_step_parity(Device &device) {
    ocr::test::TempDir tmp;
    auto weights = ocr::test::generate_synthetic_weights();
    const auto path = tmp.path / "dec.lcocr";
    expect(ocr::test::write_synthetic_package(path, weights).ok());
    auto stream = device.create_stream();
    WeightStore store;
    expect(ocr::test::upload_synthetic_weights(device, stream, path, store).ok());
    DenseExecutorConfig config;
    config.decoder = ocr::test::synthetic_config();
    config.tie_word_embeddings = true;
    DenseVlmExecutor exec;
    expect(exec.create(device, stream, config, store).ok());

    ContiguousKvCache cache;
    expect(cache.create(device, stream, syn_kv_config(ocr::test::kSynMaxPos)).ok());

    luisa::vector<uint32_t> prompt = {2, 5, 8, 11};
    DecodeRequest req;
    req.prompt_tokens = prompt;

    // Oracle prefill.
    ocr::test::OracleResult ref;
    luisa::vector<luisa::vector<float>> kc, vc;
    expect(ocr::test::oracle_forward(weights, prompt, {}, 0, 0, 0, kc, vc, ref).ok());

    luisa::vector<float> logits(ocr::test::kSynVocab);
    {
        auto list = CommandList::create();
        expect(exec.record_prefill(list, req, cache, logits).ok());
        stream << list.commit() << synchronize();
    }
    luisa::vector<float> ref_last(ref.logits.end() - ocr::test::kSynVocab, ref.logits.end());
    ocr::test::expect_close(logits, ref_last, kAtol, kRtol, "prefill last-row logits");

    // Decode several steps; after EACH step compare every layer's full KV
    // content and the step logits against the oracle.
    uint32_t kv_prefix = static_cast<uint32_t>(prompt.size());
    uint32_t token = ocr::test::oracle_argmax(ref_last);
    for (uint32_t step = 0; step < 6; step++) {
        auto list = CommandList::create();
        expect(exec.record_decode_step(list, token, cache, logits).ok());
        stream << list.commit() << synchronize();
        // Oracle: one-token forward over the existing prefix.
        luisa::vector<uint32_t> one = {token};
        ocr::test::OracleResult step_ref;
        expect(ocr::test::oracle_forward(weights, one, {}, 0, 0, kv_prefix, kc, vc, step_ref).ok());
        ocr::test::expect_close(logits, step_ref.logits, kAtol, kRtol, "decode-step logits");
        // KV content of EVERY layer, full committed prefix.
        for (uint32_t l = 0; l < ocr::test::kSynLayers; l++) {
            luisa::vector<float> k_got, v_got;
            expect(cache.download_layer(stream, l, 0, k_got, v_got).ok());
            stream << synchronize();
            ocr::test::expect_close(k_got, kc[l], kAtol, kRtol, "decode-step K cache");
            ocr::test::expect_close(v_got, vc[l], kAtol, kRtol, "decode-step V cache");
        }
        expect(cache.length(0) == kv_prefix + 1u);
        kv_prefix += 1u;
        token = ocr::test::oracle_argmax(logits);
    }
}

// Cache capacity bounds the decode loop; release on completion allows reuse.
void test_capacity_bound_and_reuse(Device &device) {
    ocr::test::TempDir tmp;
    auto weights = ocr::test::generate_synthetic_weights();
    const auto path = tmp.path / "dec2.lcocr";
    expect(ocr::test::write_synthetic_package(path, weights).ok());
    auto stream = device.create_stream();
    WeightStore store;
    expect(ocr::test::upload_synthetic_weights(device, stream, path, store).ok());
    DenseExecutorConfig config;
    config.decoder = ocr::test::synthetic_config();
    config.tie_word_embeddings = true;
    DenseVlmExecutor exec;
    expect(exec.create(device, stream, config, store).ok());

    // Tiny capacity: prompt 3 + at most 2 decode rows.
    ContiguousKvCache cache;
    expect(cache.create(device, stream, syn_kv_config(5)).ok());
    luisa::vector<uint32_t> prompt = {1, 2, 3};
    DecodeRequest req;
    req.prompt_tokens = prompt;
    luisa::vector<float> logits(ocr::test::kSynVocab);
    {
        auto list = CommandList::create();
        expect(exec.record_prefill(list, req, cache, logits).ok());
        stream << list.commit() << synchronize();
    }
    expect(cache.length(0) == 3u);
    // Two decode steps fill the cache to capacity.
    uint32_t token = ocr::test::oracle_argmax(logits);
    for (int i = 0; i < 2; i++) {
        auto list = CommandList::create();
        expect(exec.record_decode_step(list, token, cache, logits).ok());
        stream << list.commit() << synchronize();
        token = ocr::test::oracle_argmax(logits);
    }
    expect(cache.length(0) == 5u);
    // A third step would overrun: refused clearly.
    {
        auto list = CommandList::create();
        auto s = exec.record_decode_step(list, token, cache, logits);
        expect(!s.ok());
        expect(s.code == StatusCode::out_of_memory);
    }
    // Release + a fresh prefill reuses the same cache (no reallocation).
    cache.release(0);
    expect(cache.length(0) == 0u);
    {
        auto list = CommandList::create();
        expect(exec.record_prefill(list, req, cache, logits).ok());
        stream << list.commit() << synchronize();
    }
    expect(cache.length(0) == 3u);
}

void test_workspace_bounds(Device &device) {
    ocr::test::TempDir tmp;
    auto weights = ocr::test::generate_synthetic_weights();
    const auto path = tmp.path / "dec3.lcocr";
    expect(ocr::test::write_synthetic_package(path, weights).ok());
    auto stream = device.create_stream();
    WeightStore store;
    expect(ocr::test::upload_synthetic_weights(device, stream, path, store).ok());
    DenseExecutorConfig config;
    config.decoder = ocr::test::synthetic_config();
    config.tie_word_embeddings = true;
    // A tiny device budget must refuse the workspace preflight.
    config.memory.max_device_bytes = 256;
    DenseVlmExecutor exec;
    auto s = exec.create(device, stream, config, store);
    expect(!s.ok());
    expect(s.code == StatusCode::out_of_memory);
    // With no limit the workspace is bounded and reported.
    config.memory.max_device_bytes = 0;
    DenseVlmExecutor ok_exec;
    expect(ok_exec.create(device, stream, config, store).ok());
    expect(ok_exec.workspace_bytes() > 0u);
    expect(ok_exec.weight_bytes() > 0u);
}

// M3: reset() must not dangle stack spans. Reset, commit, churn the stack,
// delay, then download: the device metadata must read back zero. The old path
// borrowed local vectors into async copy_from and returned (use-after-free).
void test_reset_stack_churn_zero(Device &device) {
    auto stream = device.create_stream();
    ContiguousKvCache cache;
    expect(cache.create(device, stream, syn_kv_config(16)).ok());
    // Put non-zero metadata in place first.
    {
        auto list = CommandList::create();
        luisa::vector<uint32_t> kv_len = {7}, step = {3}, state = {static_cast<uint32_t>(DecodeSeqState::decoding)};
        luisa::vector<int32_t> pos = {6};
        expect(cache.state().record(list, kv_len, pos, step, state).ok());
        stream << list.commit() << synchronize();
    }
    // Reset inside a helper whose locals go out of scope before the drain.
    const auto reset_in_scope = [&]() {
        auto list = CommandList::create();
        expect(cache.reset(list).ok());
        stream << list.commit();
        // list + the reset staging locals are destroyed here; the upload must
        // still be valid because staging is heap-owned + completion-retained.
    };
    reset_in_scope();
    // Churn the stack with unrelated volatile data, then delay before draining.
    volatile uint32_t churn[256];
    for (uint32_t i = 0; i < 256; i++) { churn[i] = 0xDEADBEEFu ^ i; }
    (void) churn;
    stream << synchronize();
    luisa::vector<uint32_t> kv_len, step, state;
    luisa::vector<int32_t> pos;
    expect(cache.state().download(stream, kv_len, pos, step, state).ok());
    stream << synchronize();
    expect(kv_len[0] == 0u);
    expect(pos[0] == 0);
    expect(step[0] == 0u);
    expect(state[0] == static_cast<uint32_t>(DecodeSeqState::empty));
    // Reset/reuse: a fresh sequence binds and runs after the reset.
    expect(cache.bind_sequence(0, 3).ok());
    expect(cache.length(0) == 3u);
}

// S8: create() synchronously zeroes metadata; an unbound sequence is never
// attended. record_finished() marks a released slot finished on the device.
void test_metadata_zero_and_finished(Device &device) {
    auto stream = device.create_stream();
    ContiguousKvCache cache;
    expect(cache.create(device, stream, syn_kv_config(16)).ok());
    // Immediately after create (no record), metadata is zero/empty.
    luisa::vector<uint32_t> kv_len, step, state;
    luisa::vector<int32_t> pos;
    expect(cache.state().download(stream, kv_len, pos, step, state).ok());
    stream << synchronize();
    expect(kv_len[0] == 0u);
    expect(state[0] == static_cast<uint32_t>(DecodeSeqState::empty));
    // Bind + record finished: device state becomes finished, kv_len 0.
    expect(cache.bind_sequence(0, 4).ok());
    {
        auto list = CommandList::create();
        expect(cache.record_finished(list, 0).ok());
        stream << list.commit() << synchronize();
    }
    expect(cache.length(0) == 0u);
    expect(cache.state().download(stream, kv_len, pos, step, state).ok());
    stream << synchronize();
    expect(kv_len[0] == 0u);
    expect(state[0] == static_cast<uint32_t>(DecodeSeqState::finished));
    // A finished slot can be re-bound (reuse) and is not attended stale.
    expect(cache.bind_sequence(0, 2).ok());
    expect(cache.length(0) == 2u);
}

// S3: two decode steps recorded into ONE CommandList must not overwrite each
// other's borrowed upload staging; both steps' logits match the oracle.
void test_two_steps_one_list(Device &device) {
    ocr::test::TempDir tmp;
    auto weights = ocr::test::generate_synthetic_weights();
    const auto path = tmp.path / "twostep.lcocr";
    expect(ocr::test::write_synthetic_package(path, weights).ok());
    auto stream = device.create_stream();
    WeightStore store;
    expect(ocr::test::upload_synthetic_weights(device, stream, path, store).ok());
    DenseExecutorConfig config;
    config.decoder = ocr::test::synthetic_config();
    config.tie_word_embeddings = true;
    DenseVlmExecutor exec;
    expect(exec.create(device, stream, config, store).ok());
    ContiguousKvCache cache;
    expect(cache.create(device, stream, syn_kv_config(ocr::test::kSynMaxPos)).ok());

    luisa::vector<uint32_t> prompt = {2, 5, 8, 11};
    DecodeRequest req;
    req.prompt_tokens = prompt;
    ocr::test::OracleResult ref;
    luisa::vector<luisa::vector<float>> kc, vc;
    expect(ocr::test::oracle_forward(weights, prompt, {}, 0, 0, 0, kc, vc, ref).ok());
    luisa::vector<float> logits0(ocr::test::kSynVocab);
    {
        auto list = CommandList::create();
        expect(exec.record_prefill(list, req, cache, logits0).ok());
        stream << list.commit() << synchronize();
    }
    uint32_t kv_prefix = static_cast<uint32_t>(prompt.size());
    uint32_t token = ocr::test::oracle_argmax(
        luisa::span<const float>{ref.logits.end() - ocr::test::kSynVocab, ocr::test::kSynVocab});
    // Two decode steps in ONE list, distinct output spans.
    luisa::vector<float> logits1(ocr::test::kSynVocab), logits2(ocr::test::kSynVocab);
    uint32_t token2 = 0;
    {
        auto list = CommandList::create();
        expect(exec.record_decode_step(list, token, cache, logits1).ok());
        // Oracle step 1 to learn the next input token for step 2.
        luisa::vector<uint32_t> one = {token};
        ocr::test::OracleResult s1;
        expect(ocr::test::oracle_forward(weights, one, {}, 0, 0, kv_prefix, kc, vc, s1).ok());
        token2 = ocr::test::oracle_argmax(s1.logits);
        expect(exec.record_decode_step(list, token2, cache, logits2).ok());
        stream << list.commit() << synchronize();
        // Compare step 1 logits.
        ocr::test::expect_close(logits1, s1.logits, kAtol, kRtol, "two-in-one-list step1 logits");
        // Oracle step 2.
        luisa::vector<uint32_t> two = {token2};
        ocr::test::OracleResult s2;
        expect(ocr::test::oracle_forward(weights, two, {}, 0, 0, kv_prefix + 1u, kc, vc, s2).ok());
        ocr::test::expect_close(logits2, s2.logits, kAtol, kRtol, "two-in-one-list step2 logits");
    }
    expect(cache.length(0) == kv_prefix + 2u);
    // Final KV matches the oracle after both steps.
    for (uint32_t l = 0; l < ocr::test::kSynLayers; l++) {
        luisa::vector<float> k_got, v_got;
        expect(cache.download_layer(stream, l, 0, k_got, v_got).ok());
        stream << synchronize();
        ocr::test::expect_close(k_got, kc[l], kAtol, kRtol, "two-in-one-list K cache");
        ocr::test::expect_close(v_got, vc[l], kAtol, kRtol, "two-in-one-list V cache");
    }
}

// S5: the cache budget preflight AND recorded requirement both include the
// DecodeState metadata bytes (4 buffers x batch x 4). A budget that fits the
// K/V buffers but NOT the metadata must be refused.
void test_cache_metadata_accounting(Device &device) {
    auto stream = device.create_stream();
    auto kc = syn_kv_config(64);
    auto kv_bytes = kc.total_bytes();
    expect(kv_bytes.ok());
    const uint64_t metadata_bytes = 4ull * kc.batch * 4ull;
    // Budget exactly the K/V bytes (no room for metadata) -> refused.
    {
        MemoryBudget budget;
        MemoryLimits limits;
        limits.max_device_bytes = *kv_bytes.value;
        budget.set_limits(limits);
        ContiguousKvCache cache;
        auto s = cache.create(device, stream, kc, &budget);
        expect(!s.ok());
        expect(s.code == StatusCode::out_of_memory);
    }
    // Budget K/V + metadata -> succeeds; recorded == device_bytes (incl metadata).
    {
        MemoryBudget budget;
        MemoryLimits limits;
        limits.max_device_bytes = *kv_bytes.value + metadata_bytes;
        budget.set_limits(limits);
        ContiguousKvCache cache;
        expect(cache.create(device, stream, kc, &budget).ok());
        expect(cache.device_bytes() == *kv_bytes.value + metadata_bytes);
        expect(budget.category_bytes(MemoryCategory::kv_cache) == cache.device_bytes());
    }
}

// S5: the executor accounts the tied-LM-head transpose buffer in weight_bytes
// and preflights it (device + host staging) against the memory limit.
void test_tied_head_accounting(Device &device) {
    ocr::test::TempDir tmp;
    auto weights = ocr::test::generate_synthetic_weights();
    const auto path = tmp.path / "tied.lcocr";
    expect(ocr::test::write_synthetic_package(path, weights).ok());
    auto stream = device.create_stream();
    WeightStore store;
    expect(ocr::test::upload_synthetic_weights(device, stream, path, store).ok());
    DenseExecutorConfig config;
    config.decoder = ocr::test::synthetic_config();
    config.tie_word_embeddings = true;
    const uint64_t head_bytes = static_cast<uint64_t>(ocr::test::kSynVocab) * ocr::test::kSynHidden * 4ull;
    // No limit: weight_bytes includes the component weights + the tied head.
    DenseVlmExecutor exec;
    expect(exec.create(device, stream, config, store).ok());
    expect(exec.weight_bytes() == store.component_bytes("decoder") + head_bytes);
    // A device limit below component + head (but above component) is refused.
    {
        DenseExecutorConfig tight = config;
        tight.memory.max_device_bytes = store.component_bytes("decoder") + head_bytes - 1u;
        DenseVlmExecutor e2;
        auto s = e2.create(device, stream, tight, store);
        expect(!s.ok());
        expect(s.code == StatusCode::out_of_memory);
    }
    // A host-staging limit below the tied-head staging (2 x head bytes) refuses.
    {
        DenseExecutorConfig tight = config;
        tight.memory.max_host_bytes = head_bytes;// staging needs 2 x head_bytes
        DenseVlmExecutor e3;
        auto s = e3.create(device, stream, tight, store);
        expect(!s.ok());
        expect(s.code == StatusCode::out_of_memory);
    }
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "decode_cache_capacity_math"_test = [&] { test_capacity_math(); };
    "decode_cache_budget_preflight"_test = [&] { test_budget_preflight(device); };
    "decode_cache_append_release_reuse"_test = [&] { test_append_release_reuse(device); };
    "decode_cache_paged_fails_closed"_test = [&] { test_paged_fails_closed(device); };
    "decode_cache_step_parity"_test = [&] { test_decode_step_parity(device); };
    "decode_cache_capacity_bound_reuse"_test = [&] { test_capacity_bound_and_reuse(device); };
    "decode_cache_workspace_bounds"_test = [&] { test_workspace_bounds(device); };
    "decode_cache_reset_stack_churn_zero"_test = [&] { test_reset_stack_churn_zero(device); };
    "decode_cache_metadata_zero_and_finished"_test = [&] { test_metadata_zero_and_finished(device); };
    "decode_cache_two_steps_one_list"_test = [&] { test_two_steps_one_list(device); };
    "decode_cache_metadata_accounting"_test = [&] { test_cache_metadata_accounting(device); };
    "decode_cache_tied_head_accounting"_test = [&] { test_tied_head_accounting(device); };
}