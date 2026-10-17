// Test for embedding/embed/package.h — model.bgm package loader round-trip
// (embedding_model_plan.md §3.3 / milestone M2 exit criterion "loader round-trip").
// Host-only test (LUISA_EMBED_HOST_ONLY=1): creates no device, takes no backend.
//
// This test runs against the REAL packed model (2.27 GB, 296 tensors), path via
// argv[1], default D:/compute/bge-m3/bge-m3/model.bgm. It covers:
// - header/table validation of a genuine BGM3 v1 file (load succeeds; the
//   loader's fail-closed path makes any corruption a loud LUISA_ERROR — that
//   negative path is a [[noreturn]] abort observable as a nonzero process exit)
// - hyper-parameter accessors: 24 layers, 1024 hidden, 16 heads, 4096
//   intermediate, 250002 vocab, 8194 positions
// - tensor_count == 296 and every canonical name resolvable through find():
//   emb.* (4), all 24 layers x 12 tensors, head.* (4) — names regenerated from
//   the packer spec (the header comment of tools/pack_model.cpp), NOT from
//   package.cpp, so this is an independent transcription
// - dims spot-checks: emb.word {250002,1024}, layer7.qkv {1024,3072},
//   layer23.ffn.down {4096,1024}, head.sparse.w {1024,1}; size_bytes ==
//   prod(dims)*4; element_count()/dim() semantics
// - 256-byte alignment: the format guarantees 256-byte aligned blob OFFSETS;
//   the frozen Tensor API exposes no offset, so the test verifies the
//   observable equivalent — every Tensor::data pointer is 256-byte aligned
//   (package.cpp reads the file into a 256-byte aligned base, preserving the
//   on-disk alignment for direct device upload)
// - emission order: find(name) returns the same tensor as tensors()[i] for the
//   canonical emission order
// - negative lookups: unknown names resolve to nullptr
// - VALUE checks (the round-trip against the packer): emb.ln.beta[0..3],
//   layer0.ln1.gamma[0..3] and head.sparse.b[0] compared against constants
//   extracted INDEPENDENTLY of this C++ loader, via
//     python examples/embedding/tools/pack_model_dev.py dump D:/compute/bge-m3/bge-m3/model.bgm <name> <out.bin>
//   + a python struct.unpack("<...f") read of the first elements (float32 LE,
//   exact shortest decimal round-trip repr; extracted 2026-10-17).
//
// Test bodies are plain functions called from main (the explicit-main pattern of
// test_gemm.cpp), NOT "name"_test static registrations: the vendored Boost.UT
// runs deferred/static registrations eagerly during static initialization —
// before main — which would leave no way to consume argv[1] as the package path.
//
// Run: xmake run test_embed_package [D:/compute/bge-m3/bge-m3/model.bgm]
#include "ut/ut.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "embedding/embed/package.h"

using namespace luisa;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

    // independent transcription of the canonical emission order + dims from the
    // format spec in tools/pack_model.cpp (spec literals, not derived from
    // package.cpp's constants, so a drift in either file trips this test)
    struct ExpectedEntry {
        luisa::string name;
        uint32_t rank;
        uint32_t dims[4];
    };
    luisa::vector<ExpectedEntry> expected_entries() {
        luisa::vector<ExpectedEntry> out;
        out.reserve(296u);
        const auto add = [&out](luisa::string name, uint32_t rank, const uint32_t *dims) {
            ExpectedEntry e;
            e.name = std::move(name);
            e.rank = rank;
            for(uint32_t i = 0u; i < 4u; i++) {
                e.dims[i] = i < rank ? dims[i] : 0u;
            }
            out.push_back(std::move(e));
        };
        const uint32_t w[2] = {250002u, 1024u};
        const uint32_t p[2] = {8194u, 1024u};
        const uint32_t h[1] = {1024u};
        const uint32_t qkv[2] = {1024u, 3072u};
        const uint32_t qkvb[1] = {3072u};
        const uint32_t hh[2] = {1024u, 1024u};
        const uint32_t up[2] = {1024u, 4096u};
        const uint32_t upb[1] = {4096u};
        const uint32_t down[2] = {4096u, 1024u};
        const uint32_t one[1] = {1u};
        const uint32_t h1[2] = {1024u, 1u};
        add("emb.word", 2u, w);
        add("emb.pos", 2u, p);
        add("emb.ln.gamma", 1u, h);
        add("emb.ln.beta", 1u, h);
        for(uint32_t i = 0u; i < 24u; i++) {
            const auto pre = luisa::format("layer{}.", i);
            add(pre + "qkv", 2u, qkv);
            add(pre + "qkv.bias", 1u, qkvb);
            add(pre + "attn.out", 2u, hh);
            add(pre + "attn.out.bias", 1u, h);
            add(pre + "ln1.gamma", 1u, h);
            add(pre + "ln1.beta", 1u, h);
            add(pre + "ffn.up", 2u, up);
            add(pre + "ffn.up.bias", 1u, upb);
            add(pre + "ffn.down", 2u, down);
            add(pre + "ffn.down.bias", 1u, h);
            add(pre + "ln2.gamma", 1u, h);
            add(pre + "ln2.beta", 1u, h);
        }
        add("head.sparse.w", 2u, h1);
        add("head.sparse.b", 1u, one);
        add("head.colbert.w", 2u, hh);
        add("head.colbert.b", 1u, h);
        return out;
    }

    uint64_t prod_dims(const ExpectedEntry &e) {
        uint64_t n = 1ull;
        for(uint32_t i = 0u; i < e.rank; i++) {
            n *= e.dims[i];
        }
        return n;
    }

    constexpr float k_tolerate = 1e-7f;

    void test_package_header_and_hyper_params(const luisa::embed::Package &pkg, const luisa::string &path) {
        expect(pkg.tensors().size() == 296u) << "tensor_count must be 296 (4 emb + 24*12 layer + 4 head)";
        expect(pkg.num_layers() == 24u);
        expect(pkg.hidden_size() == 1024u);
        expect(pkg.num_heads() == 16u);
        expect(pkg.intermediate_size() == 4096u);
        expect(pkg.vocab_size() == 250002u);
        expect(pkg.max_positions() == 8194u);
        LUISA_INFO("test_embed_package: loaded \"{}\" ({} tensors)", path, pkg.tensors().size());
    }

    void test_package_names_dims_and_emission_order(const luisa::embed::Package &pkg) {
        const auto expected = expected_entries();
        expect(expected.size() == 296u) << "guard the transcription itself";
        uint32_t mismatch_count = 0u;
        const auto table = pkg.tensors();
        for(size_t i = 0u; i < expected.size(); i++) {
            const auto &e = expected[i];
            const auto *t = pkg.find(e.name);
            if(t == nullptr) {
                LUISA_WARNING("tensor '{}' is not resolvable by name", e.name);
                mismatch_count++;
                continue;
            }
            if(t->rank != e.rank) {
                LUISA_WARNING("tensor '{}': rank {} != expected {}", e.name, t->rank, e.rank);
                mismatch_count++;
            }
            for(uint32_t d = 0u; d < 4u; d++) {
                if(t->dims[d] != e.dims[d]) {
                    LUISA_WARNING("tensor '{}': dims[{}] {} != expected {}", e.name, d, t->dims[d], e.dims[d]);
                    mismatch_count++;
                }
            }
            if(t->size_bytes != prod_dims(e) * 4ull) {
                LUISA_WARNING("tensor '{}': size_bytes {} != prod(dims)*4 {}", e.name, t->size_bytes, prod_dims(e) * 4ull);
                mismatch_count++;
            }
            if(t->element_count() != prod_dims(e)) {
                LUISA_WARNING("tensor '{}': element_count {} != {}", e.name, t->element_count(), prod_dims(e));
                mismatch_count++;
            }
            // dim() semantics: dims beyond rank report 0
            if(t->dim(e.rank) != 0ull || t->dim(0u) != e.dims[0]) {
                LUISA_WARNING("tensor '{}': dim() accessors inconsistent", e.name);
                mismatch_count++;
            }
            // emission order == table order: find() and positional access agree
            if(i >= table.size() || t != &table[i]) {
                LUISA_WARNING("tensor '{}': find() is not the positional table entry {}", e.name, i);
                mismatch_count++;
            }
        }
        expect(mismatch_count == 0u) << "every canonical tensor must resolve with exact rank/dims/size in emission order";
    }

    void test_package_dims_spot_checks(const luisa::embed::Package &pkg) {
        uint32_t mismatch_count = 0u;
        const auto check = [&mismatch_count, &pkg](luisa::string_view name, luisa::vector<uint32_t> dims) {
            const auto *t = pkg.find(name);
            if(t == nullptr) {
                LUISA_WARNING("spot-check tensor '{}' missing", name);
                mismatch_count++;
                return;
            }
            if(t->rank != dims.size()) {
                LUISA_WARNING("spot-check '{}': rank {} != {}", name, t->rank, dims.size());
                mismatch_count++;
                return;
            }
            for(size_t i = 0u; i < dims.size(); i++) {
                if(t->dim(static_cast<uint32_t>(i)) != dims[i]) {
                    LUISA_WARNING("spot-check '{}': dim({}) {} != {}", name, i, t->dim(static_cast<uint32_t>(i)), dims[i]);
                    mismatch_count++;
                }
            }
        };
        check("emb.word", {250002u, 1024u});
        check("layer7.qkv", {1024u, 3072u});
        check("layer23.ffn.down", {4096u, 1024u});
        check("head.sparse.w", {1024u, 1u});
        expect(mismatch_count == 0u) << "spot-checked dims must match the packer spec";
    }

    void test_package_blob_alignment_and_negative_lookup(const luisa::embed::Package &pkg) {
        uint32_t mismatch_count = 0u;
        // 256-byte alignment of every blob offset, observable through the data
        // pointer (see the file header): every Tensor::data must be 256-aligned
        for(const auto &t : pkg.tensors()) {
            if(reinterpret_cast<uintptr_t>(t.data) % 256u != 0u) {
                LUISA_WARNING("tensor data pointer {} is not 256-byte aligned",
                              static_cast<const void *>(t.data));
                mismatch_count++;
            }
        }
        expect(mismatch_count == 0u) << "every blob pointer must be 256-byte aligned";
        // unknown names resolve to nullptr (never a wrong tensor)
        expect(pkg.find("layer24.qkv") == nullptr) << "layer24.qkv must not resolve (24 layers only)";
        expect(pkg.find("emb.wordd") == nullptr) << "a typo'd name must not resolve";
        expect(pkg.find("head.sparse.weight") == nullptr);
        expect(pkg.find("") == nullptr) << "empty name must not resolve";
    }

    void test_package_values_vs_packer_dev_extraction(const luisa::embed::Package &pkg) {
        uint32_t mismatch_count = 0u;
        // values extracted independently with pack_model_dev.py dump + python
        // struct.unpack("<f") — see the file header; float32-exact decimals
        const float emb_ln_beta[4] = {
            -0.15087890625f, -0.241455078125f, 0.25830078125f, 0.2802734375f};
        const float layer0_ln1_gamma[4] = {
            0.94091796875f, 0.91015625f, 0.82958984375f, 0.93798828125f};
        constexpr float head_sparse_b0 = 0.045196533203125f;

        const auto check = [&mismatch_count, &pkg](luisa::string_view name, uint32_t idx, float want) {
            const auto *t = pkg.find(name);
            if(t == nullptr || idx >= t->element_count()) {
                LUISA_WARNING("value check: '{}' element {} unavailable", name, idx);
                mismatch_count++;
                return;
            }
            const float got = t->data[idx];
            if(std::abs(got - want) > k_tolerate) {
                LUISA_WARNING("value check: '{}'[{}] = {}, packer-dev extraction says {}", name, idx, got, want);
                mismatch_count++;
            }
        };
        for(uint32_t i = 0u; i < 4u; i++) {
            check("emb.ln.beta", i, emb_ln_beta[i]);
            check("layer0.ln1.gamma", i, layer0_ln1_gamma[i]);
        }
        check("head.sparse.b", 0u, head_sparse_b0);
        expect(mismatch_count == 0u) << "loader-served values must match the values extracted with pack_model_dev.py";
    }

}// namespace

int main(int argc, char *argv[]) {
    // argv[1] (optional) is the packed model path; the default pins the real
    // bge-m3 package the packer produced (plan §6 M2, dev environment).
    luisa::string bgm_path{"D:/compute/bge-m3/bge-m3/model.bgm"};
    if(argc > 1) {
        if(argv[1][0] == '\0') {
            std::fprintf(stderr, "test_embed_package: empty package path\n");
            return 1;
        }
        bgm_path = luisa::string{argv[1]};
        argc--;
        argv++;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));

    // fatal on any deviation from the BGM3 v1 spec (LUISA_ERROR abort)
    auto pkg = luisa::embed::Package::load(std::filesystem::path{bgm_path});
    if(pkg == nullptr) {
        LUISA_ERROR("Package::load returned null for \"{}\"", bgm_path);
    }

    test_package_header_and_hyper_params(*pkg, bgm_path);
    test_package_names_dims_and_emission_order(*pkg);
    test_package_dims_spot_checks(*pkg);
    test_package_blob_alignment_and_negative_lookup(*pkg);
    test_package_values_vs_packer_dev_extraction(*pkg);
    return 0;
}
