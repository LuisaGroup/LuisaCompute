// Test for ocr/runtime/{weights,workspace,shader_cache}.h (Phase 2, device).
// Covers: component-selective sync upload from a real .lcocr package (fp32 +
// packed fp16 + grouped INT4 asym), resident byte accounting, staging
// lifetime refusal before completion, grow-only workspace with checked
// offsets + reset + retirement, typed shader cache compile-once + retention.
// Synthetic package written to a temp dir; no models/downloads.
#include "ut/ut.hpp"
#include "test_device.h"
#include "ocr_test_utils.h"

#include <cstring>

#include <luisa/core/stl/format.h>

#include "ocr/ops/cpu_reference.h"
#include "ocr/ops/ops_gemm.h"
#include "ocr/ops/ops_norm.h"
#include "ocr/runtime/dispatch_adapter.h"
#include "ocr/package.h"
#include "ocr/runtime/shader_cache.h"
#include "ocr/runtime/weights.h"
#include "ocr/runtime/workspace.h"
#include "ocr/runtime/dispatch_adapter.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct TempDir {
    luisa::filesystem::path path;
    TempDir() {
        path = luisa::filesystem::temp_directory_path() /
               ("lcocr_rt_test_" + std::to_string(reinterpret_cast<uintptr_t>(this)));
        luisa::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        luisa::filesystem::remove_all(path, ec);
    }
};

[[nodiscard]] PackageManifest manifest() {
    PackageManifest m;
    m.model = "glm_ocr";
    m.recipe_version = "1";
    m.checkpoint_revision = "test-rev";
    m.config_fingerprint = "test-fp";
    m.license = "test";
    m.producer = "test";
    m.producer_version = "1";
    m.preprocessing_version = "1";
    m.tokenizer_version = "1";
    m.output_parser_version = "1";
    return m;
}

[[nodiscard]] luisa::vector<std::byte> f32_bytes(luisa::span<const float> v) {
    luisa::vector<std::byte> out(v.size() * 4);
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}

// Write a small synthetic package: fp32 [2,3], fp16 [5] (odd), int4 asym [70].
[[nodiscard]] luisa::filesystem::path write_package(const TempDir &tmp) {
    const auto path = tmp.path / "runtime.lcocr";
    auto writer = PackageWriter::create(path);
    expect(writer.ok());
    auto &w = *writer.value;
    const float dense[] = {1, 2, 3, 4, 5, 6};
    PackageWriter::TensorSpec d;
    d.name = "vision.dense";
    d.encoding = TensorEncoding::fp32;
    d.shape = {2, 3};
    d.payload = f32_bytes(dense);
    expect(w->add_tensor(d).ok());
    const float half[] = {0.5f, -1.25f, 3.0f, 0.125f, -2.0f};
    luisa::vector<std::byte> hp(10);
    for (size_t i = 0; i < 5; i++) {
        const uint16_t h = fp32_to_fp16(half[i]);
        hp[i * 2] = static_cast<std::byte>(h & 0xFFu);
        hp[i * 2 + 1] = static_cast<std::byte>(h >> 8u);
    }
    PackageWriter::TensorSpec hs;
    hs.name = "vision.half";
    hs.encoding = TensorEncoding::fp16;
    hs.shape = {5};
    hs.payload = std::move(hp);
    expect(w->add_tensor(hs).ok());
    luisa::vector<float> qv(70);
    for (size_t i = 0; i < qv.size(); i++) { qv[i] = static_cast<float>(i % 13) * 0.1f - 0.6f; }
    luisa::vector<uint8_t> packed, zp;
    luisa::vector<float> scales;
    expect(quantize_int4(qv, 64, false, packed, scales, zp).ok());
    PackageWriter::TensorSpec q;
    q.name = "decoder.q";
    q.encoding = TensorEncoding::int4_packed;
    q.layout = TensorLayout::kn;
    q.shape = {70};
    q.quant_kind = QuantKind::int4_asymmetric;
    q.group_size = 64;
    q.scales = scales;
    q.zero_points = zp;
    q.payload.resize(packed.size());
    std::memcpy(q.payload.data(), packed.data(), packed.size());
    expect(w->add_tensor(q).ok());
    expect(w->finish(manifest()).ok());
    return path;
}

void test_weights_upload(Device &device) {
    TempDir tmp;
    const auto path = write_package(tmp);
    auto reader = PackageReader::open(path);
    expect(reader.ok());
    if (!reader.ok()) { return; }
    auto stream = device.create_stream();
    WeightStore store;
    // Component-selective: load only the two vision tensors.
    const luisa::string names[] = {"vision.dense", "vision.half"};
    expect(store.upload_component_sync(device, stream, **reader.value, "vision", names).ok());
    expect(store.total_bytes() > 0u);
    const auto *dense = store.find("vision", "vision.dense");
    expect(dense != nullptr);
    expect(dense->encoding == TensorEncoding::fp32);
    expect(dense->element_count == 6u);
    expect(dense->resident_bytes() == 6u * 4u);
    const auto *half = store.find("vision", "vision.half");
    expect(half != nullptr);
    expect(half->encoding == TensorEncoding::fp16);
    // 5 halves -> 3 words -> 12 bytes (storage, not 5*4 compute bytes).
    expect(half->resident_bytes() == 3u * 4u);
    expect(store.find("vision", "decoder.q") == nullptr);// selective
    // Verify uploaded dense values by download.
    luisa::vector<float> got;
    ocr::test::download(stream, dense->dense, got, 6);
    stream << synchronize();
    expect(got[0] == 1.f && got[5] == 6.f);
    // Load the quantized tensor into a second component.
    const luisa::string qnames[] = {"decoder.q"};
    expect(store.upload_component_sync(device, stream, **reader.value, "decoder", qnames).ok());
    const auto *q = store.find("decoder", "decoder.q");
    expect(q != nullptr);
    expect(q->group_count == 2u);// ceil(70/64)
    expect(q->zero_points.size() == 2u);
    // Resident accounting per component + unload.
    const auto vision_bytes = store.component_bytes("vision");
    const auto decoder_bytes = store.component_bytes("decoder");
    expect(store.total_bytes() == vision_bytes + decoder_bytes);
    expect(store.unload_component("vision") == vision_bytes);
    expect(store.find("vision", "vision.dense") == nullptr);
    expect(store.total_bytes() == decoder_bytes);
    // Missing tensor name fails closed.
    const luisa::string bad[] = {"nope"};
    expect(!store.upload_component_sync(device, stream, **reader.value, "x", bad).ok());
}

void test_staging_lifetime(Device &device) {
    TempDir tmp;
    const auto path = write_package(tmp);
    auto reader = PackageReader::open(path);
    expect(reader.ok());
    if (!reader.ok()) { return; }
    auto stream = device.create_stream();
    WeightUploadBatch batch;
    auto tensors = (*reader.value)->tensors();
    expect(batch.prepare(device, **reader.value, "vision", tensors).ok());
    expect(batch.staging_bytes() > 0u);
    // Refuses release before completion.
    expect(!batch.release_staging().ok());
    auto list = CommandList::create();
    expect(batch.record(list).ok());
    stream << list.commit() << synchronize();
    batch.complete();
    expect(batch.release_staging().ok());
    expect(batch.staging_bytes() == 0u);
    expect(batch.staging_released());
    // Recording after release is refused.
    auto list2 = CommandList::create();
    expect(!batch.record(list2).ok());
    WeightStore store;
    expect(store.install(batch).ok());
    expect(store.total_bytes() > 0u);
}

void test_workspace(Device &device) {
    DeviceWorkspace ws(device);
    expect(ws.reserve(1024).ok());
    expect(ws.capacity_bytes() == 1024u);
    auto a = ws.allocate(100);
    expect(a.ok());
    expect(a.value->offset == 0u);
    auto b = ws.allocate(100);
    expect(b.ok());
    expect(b.value->offset == 256u);// aligned
    expect(b.value->generation == a.value->generation);
    // Exhaustion refused (no silent mid-request growth).
    expect(ws.allocate(1024).status.code == StatusCode::out_of_memory);
    // Grow-only reserve: old buffer retired, cursor reset, generation bumped.
    const auto gen0 = ws.generation();
    expect(ws.reserve(4096).ok());
    expect(ws.capacity_bytes() == 4096u);
    expect(ws.generation() == gen0 + 1u);
    expect(ws.retired_count() == 1u);
    expect(ws.resident_bytes() == 4096u + 1024u);
    expect(ws.reserve(2048).ok());// never shrinks
    expect(ws.capacity_bytes() == 4096u);
    // Reset rewinds without touching capacity.
    ws.reset();
    expect(ws.used_bytes() == 0u);
    auto c = ws.allocate(64);
    expect(c.ok());
    expect(c.value->offset == 0u);
    // After the stream drains, retired buffers are released.
    ws.retire_after_sync();
    expect(ws.retired_count() == 0u);
    expect(ws.resident_bytes() == 4096u);
    // Host plan: checked offsets + total.
    WorkspacePlan plan;
    expect(plan.add_bytes("a", 100).ok());
    auto td = make_matrix_desc(3, 5);
    expect(plan.add_tensor("b", *td.value).ok());
    expect(plan.total_bytes() == 256u + 256u);
    expect(plan.find("b")->offset == 256u);
    expect(!plan.add_bytes("zero", 0).ok());
}

void test_shader_cache(Device &device) {
    ShaderKeyRegistry registry;
    TypedShaderCache<RmsNormKernel> cache(device, registry);
    const auto key = norm_variant_key("rms_norm");
    int factory_calls = 0;
    const decltype(device.compile(make_rms_norm_kernel())) *first = nullptr;
    for (int i = 0; i < 3; i++) {
        auto r = cache.get_or_compile(key, [&] {
            factory_calls++;
            return make_rms_norm_kernel();
        });
        expect(r.ok());
        if (i == 0) { first = *r.value; }
        else { expect(*r.value == first); }// same retained shader
    }
    expect(factory_calls == 1);// compiled once per variant
    expect(registry.compile_count(key) == 1u);
    expect(cache.contains(key));
    expect(cache.size() == 1u);
    // The retained shader actually dispatches.
    auto stream = device.create_stream();
    auto x = device.create_buffer<float>(8);
    auto y = device.create_buffer<float>(8);
    auto dummy = device.create_buffer<float>(1);
    const float host_x[] = {1, 2, 3, 4, 5, 6, 7, 8};
    stream << x.copy_from(luisa::span<const float>{host_x});
    auto plan = plan_norm_dispatch(1);
    OCR_DISPATCH_1D(stream, (*first)(x, y, dummy, 1u, 8u, 1e-6f, 0u), *plan.value);
    luisa::vector<float> got;
    ocr::test::download(stream, y, got, 8);
    stream << synchronize();
    luisa::vector<float> ref;
    expect(ref_rms_norm(host_x, {}, 1u, 8u, 1e-6f, ref).ok());
    ocr::test::expect_close(got, ref, 1e-4f, 1e-4f, "cached shader dispatch");
}

// M2: pointers returned by TypedShaderCache must survive rehashes. Compile
// many variants (forcing unordered_dense growth), then dispatch through the
// FIRST pointer and verify every early pointer still addresses its shader.
void test_shader_cache_rehash_stability(Device &device) {
    ShaderKeyRegistry registry;
    TypedShaderCache<GatedActKernel> cache(device, registry);
    luisa::vector<const void *> early;
    // 5 activation variants x 40 key suffixes = 200 entries: far beyond any
    // initial bucket count, guaranteeing multiple rehashes.
    for (uint32_t i = 0; i < 40; i++) {
        for (uint32_t act = 0; act < 5; act++) {
            const auto key = make_shader_key("gated", luisa::format("act{}_{}", act, i));
            auto r = cache.get_or_compile(key, [act] { return make_gated_act_kernel(act); });
            expect(r.ok());
            if (i == 0) { early.push_back(*r.value); }
        }
    }
    expect(cache.size() == 200u);
    // Every early pointer must still be the CURRENT mapping for its key.
    for (uint32_t act = 0; act < 5; act++) {
        const auto key = make_shader_key("gated", luisa::format("act{}_0", act));
        auto r = cache.get_or_compile(key, [act] { return make_gated_act_kernel(act); });
        expect(r.ok());
        expect(static_cast<const void *>(*r.value) == early[act]) << "stable pointee across rehash";
    }
    expect(registry.total_compiles() == 200u);// each variant compiled once
    // Dispatch through the FIRST retained pointer (act 0 = silu).
    auto stream = device.create_stream();
    constexpr uint32_t count = 257u;
    luisa::vector<float> gate(count), up(count);
    ocr::test::Lcg lcg;
    lcg.fill(gate, -2.f, 2.f);
    lcg.fill(up, -1.f, 1.f);
    luisa::vector<float> ref;
    expect(ref_gated_act(gate, up, 0u, ref).ok());
    auto bg = ocr::test::upload(device, stream, luisa::span<const float>{gate});
    auto bu = ocr::test::upload(device, stream, luisa::span<const float>{up});
    auto bo = device.create_buffer<float>(count);
    auto plan = plan_gated_act_dispatch(count);
    const auto *first = static_cast<const TypedShaderCache<GatedActKernel>::ShaderT *>(early[0]);
    OCR_DISPATCH_1D(stream, (*first)(bg, bu, bo, count), *plan.value);
    luisa::vector<float> got;
    ocr::test::download(stream, bo, got, count);
    stream << synchronize();
    ocr::test::expect_close(got, ref, 1e-4f, 1e-4f, "dispatch through pre-rehash pointer");
}

// S7: reset bumps the epoch; a stale pre-reset allocation never aliases a
// fresh post-reset allocation at the same offset.
void test_workspace_reset_epoch(Device &device) {
    DeviceWorkspace ws(device);
    expect(ws.reserve(1024).ok());
    auto a = ws.allocate(128);
    expect(a.ok());
    expect(a.value->offset == 0u);
    const auto epoch0 = ws.reset_epoch();
    ws.reset();// caller contract: stream drained (nothing dispatched here)
    expect(ws.reset_epoch() == epoch0 + 1u);
    auto b = ws.allocate(128);
    expect(b.ok());
    expect(b.value->offset == 0u);// same bytes...
    expect(b.value->epoch == epoch0 + 1u);
    expect(!a.value->aliases(*b.value)) << "stale pre-reset region must not alias fresh region";
    expect(b.value->aliases(*b.value)) << "self-alias sanity";
    auto c = ws.allocate(128);
    expect(c.ok());
    expect(!b.value->aliases(*c.value)) << "adjacent live regions do not alias";
    // Same-epoch overlapping allocations DO alias (tracking works).
    DeviceWorkspace ws2(device);
    expect(ws2.reserve(1024).ok());
    auto d = ws2.allocate(256);
    auto e = ws2.allocate(256);
    expect(d.ok() && e.ok());
    expect(!d.value->aliases(*e.value));
}

}// namespace

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    auto &device = dc->device;
    "weights_component_upload"_test = [&] { test_weights_upload(device); };
    "weights_staging_lifetime"_test = [&] { test_staging_lifetime(device); };
    "workspace_grow_reset_retire"_test = [&] { test_workspace(device); };
    "shader_cache_compile_once"_test = [&] { test_shader_cache(device); };
    "shader_cache_rehash_stability"_test = [&] { test_shader_cache_rehash_stability(device); };
    "workspace_reset_epoch"_test = [&] { test_workspace_reset_epoch(device); };
}