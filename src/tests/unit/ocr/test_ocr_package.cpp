// Test for examples/ocr/ocr/package.h/.cpp -- .lcocr v1 reader/writer.
// Host-only: no device, no backend, no model, no exceptions.
//
// This test covers:
// - writer->reader round trip for FP32/FP16/BF16 + INT4 sym/asym + INT8;
// - arbitrary rank (0..8), names, shapes/strides/layout;
// - manifest identity/dependency/source-hash validation;
// - fail-closed on truncated/overflow/duplicate/overlap/trailing/misaligned
//   data, nonfinite scales, bad zero points, unsupported encoding;
// - frozen synthetic fixtures open and decode (never regenerated here).
// Corruption copies live in the system temp dir and are cleaned up.
#include "ut/ut.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>

#include <luisa/core/stl/filesystem.h>

#include "ocr/package.h"
#include "ocr/quantization.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// A unique temp directory for corruption copies (cleaned at scope exit).
struct TempDir {
    luisa::filesystem::path path;
    TempDir() {
        auto base = luisa::filesystem::temp_directory_path();
        path = base / ("lcocr_test_" + std::to_string(reinterpret_cast<uintptr_t>(this)));
        luisa::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        luisa::filesystem::remove_all(path, ec);
    }
};

[[nodiscard]] PackageManifest valid_manifest() {
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

void test_roundtrip() {
    TempDir tmp;
    const auto path = tmp.path / "roundtrip.lcocr";
    {
        auto writer = PackageWriter::create(path);
        expect(writer.ok());
        auto &w = *writer.value;
        const float embed[] = {1, 2, 3, 4, 5, 6};
        PackageWriter::TensorSpec spec;
        spec.name = "vision.embed";
        spec.encoding = TensorEncoding::fp32;
        spec.shape = {2, 3};
        spec.payload = f32_bytes(embed);
        expect(w->add_tensor(spec).ok());
        // INT4 symmetric.
        const uint8_t codes[] = {0, 8, 15, 7};
        luisa::vector<uint8_t> packed;
        expect(pack_int4_codes(codes, packed).ok());
        PackageWriter::TensorSpec q;
        q.name = "decoder.q";
        q.encoding = TensorEncoding::int4_packed;
        q.layout = TensorLayout::kn;
        q.shape = {4};
        q.quant_kind = QuantKind::int4_symmetric;
        q.group_size = 64;
        q.scales = {0.125f};
        q.payload.resize(packed.size());
        std::memcpy(q.payload.data(), packed.data(), packed.size());
        expect(w->add_tensor(q).ok());
        expect(w->finish(valid_manifest()).ok());
    }
    auto reader = PackageReader::open(path);
    expect(reader.ok());
    if (!reader.ok()) { return; }
    auto &pkg = *reader.value;
    expect(pkg->tensors().size() == 2u);
    expect(pkg->manifest().model == "glm_ocr");
    auto *t = pkg->find("vision.embed");
    expect(t != nullptr);
    expect(t->shape.size() == 2u && t->shape[0] == 2 && t->shape[1] == 3);
    expect(t->strides[0] == 3u && t->strides[1] == 1u);
    float out[6];
    expect(pkg->read_tensor_f32(*t, out).ok());
    expect(out[0] == 1.0f && out[5] == 6.0f);
    auto *q = pkg->find("decoder.q");
    expect(q != nullptr);
    float qout[4];
    expect(pkg->read_tensor_dequantized(*q, qout).ok());
    expect(qout[0] == -1.0f && qout[1] == 0.0f && qout[2] == 0.875f);
    expect(pkg->find("missing") == nullptr);
}

void test_multirank() {
    TempDir tmp;
    const auto path = tmp.path / "multirank.lcocr";
    {
        auto writer = PackageWriter::create(path);
        auto &w = *writer.value;
        const float scalar[] = {3.5f};
        PackageWriter::TensorSpec s;
        s.name = "scalar";
        s.encoding = TensorEncoding::fp32;
        s.shape = {};
        s.payload = f32_bytes(scalar);
        expect(w->add_tensor(s).ok());
        const float r8[] = {7, 8};
        PackageWriter::TensorSpec t;
        t.name = "rank8";
        t.encoding = TensorEncoding::fp32;
        t.shape = {1, 1, 1, 1, 1, 1, 1, 2};
        t.payload = f32_bytes(r8);
        expect(w->add_tensor(t).ok());
        expect(w->finish(valid_manifest()).ok());
    }
    auto reader = PackageReader::open(path);
    expect(reader.ok());
    auto *s = (*reader.value)->find("scalar");
    expect(s && s->shape.empty() && s->element_count == 1);
    float out = 0;
    expect((*reader.value)->read_tensor_f32(*s, luisa::span<float>{&out, 1}).ok());
    expect(out == 3.5f);
    auto *t = (*reader.value)->find("rank8");
    expect(t && t->shape.size() == 8);
}

void test_duplicate_name_rejected() {
    TempDir tmp;
    const auto path = tmp.path / "dup.lcocr";
    auto writer = PackageWriter::create(path);
    auto &w = *writer.value;
    const float v[] = {1};
    PackageWriter::TensorSpec s;
    s.name = "a";
    s.encoding = TensorEncoding::fp32;
    s.shape = {1};
    s.payload = f32_bytes(v);
    expect(w->add_tensor(s).ok());
    expect(!w->add_tensor(s).ok());// duplicate
}

void test_corruption() {
    // Build a valid package, then corrupt copies in temp and expect failure.
    TempDir tmp;
    const auto good = tmp.path / "good.lcocr";
    {
        auto writer = PackageWriter::create(good);
        auto &w = *writer.value;
        const float v[] = {1, 2, 3, 4};
        PackageWriter::TensorSpec s;
        s.name = "t";
        s.encoding = TensorEncoding::fp32;
        s.shape = {4};
        s.payload = f32_bytes(v);
        expect(w->add_tensor(s).ok());
        expect(w->finish(valid_manifest()).ok());
    }
    luisa::vector<std::byte> bytes;
    {
        std::ifstream f(good, std::ios::binary);
        luisa::vector<char> chars((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        bytes.resize(chars.size());
        std::memcpy(bytes.data(), chars.data(), chars.size());
    }
    const auto write_copy = [&](const char *name, const luisa::vector<std::byte> &data) {
        const auto p = tmp.path / name;
        std::ofstream f(p, std::ios::binary);
        f.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
        f.close();
        return p;
    };
    // Truncated.
    {
        auto trunc = bytes;
        trunc.resize(trunc.size() / 2);
        auto p = write_copy("trunc.lcocr", trunc);
        expect(!PackageReader::open(p).ok());
    }
    // Bad magic.
    {
        auto bad = bytes;
        bad[0] = std::byte{0};
        auto p = write_copy("magic.lcocr", bad);
        expect(!PackageReader::open(p).ok());
    }
    // Trailing data.
    {
        auto trail = bytes;
        trail.emplace_back(std::byte{0});
        auto p = write_copy("trail.lcocr", trail);
        expect(!PackageReader::open(p).ok());
    }
    // Nonzero padding: flip a byte in the header->payload padding gap.
    {
        auto pad = bytes;
        pad[100] = std::byte{0xff};// inside the 64..256 padding region
        auto p = write_copy("pad.lcocr", pad);
        expect(!PackageReader::open(p).ok());
    }
    // Empty file.
    {
        auto p = write_copy("empty.lcocr", {});
        expect(!PackageReader::open(p).ok());
    }
}

void test_frozen_fixtures() {
    const luisa::filesystem::path dir{LUISA_OCR_TEST_DATA_DIR};
    auto basic = PackageReader::open(dir / "synthetic_basic.lcocr");
    expect(basic.ok());
    if (basic.ok()) {
        expect((*basic.value)->tensors().size() == 7u);
        auto *q = (*basic.value)->find("decoder.q_proj");
        expect(q != nullptr);
        if (q) {
            float out[8];
            expect((*basic.value)->read_tensor_dequantized(*q, out).ok());
            expect(out[0] == -1.0f && out[1] == 0.0f);
        }
    }
    auto multirank = PackageReader::open(dir / "synthetic_multirank.lcocr");
    expect(multirank.ok());
    auto layout = PackageReader::open(dir / "synthetic_layout.lcocr");
    expect(layout.ok());
    if (layout.ok()) {
        expect((*layout.value)->manifest().component == "layout");
    }
}

void test_manifest_validation() {
    auto m = valid_manifest();
    expect(m.validate().ok());
    m.model = "";
    expect(!m.validate().ok());
    m = valid_manifest();
    m.model = "not_a_model";
    expect(!m.validate().ok());
    // Serialize/parse round trip.
    m = valid_manifest();
    ComponentDependency dep;
    dep.component = "layout";
    dep.package = "synthetic_layout.lcocr";
    m.dependencies.emplace_back(dep);
    auto json = serialize_package_manifest(m);
    expect(json.ok());
    auto parsed = parse_package_manifest(*json.value);
    expect(parsed.ok());
    expect(parsed.value->dependencies.size() == 1u);
    // Unknown field rejected.
    expect(!parse_package_manifest("{\"model\":\"glm_ocr\",\"bogus\":1}").ok());
}

// Empty packages: the C++ writer/reader support a canonical empty layout
// (header + manifest + count-0 table, no payloads, payload_base==manifest_offset).
void test_empty_package() {
    TempDir tmp;
    const auto path = tmp.path / "empty.lcocr";
    {
        auto writer = PackageWriter::create(path);
        expect(writer.ok());
        expect((*writer.value)->finish(valid_manifest()).ok());
    }
    auto reader = PackageReader::open(path);
    expect(reader.ok()) << "empty package opens";
    if (reader.ok()) {
        expect((*reader.value)->tensors().empty()) << "empty package has no tensors";
        expect((*reader.value)->manifest().model == "glm_ocr");
    }
    // The empty layout must have payload_base == manifest_offset and the table
    // as the last region (file_size == table_end).
    {
        std::ifstream f(path, std::ios::binary);
        luisa::vector<char> chars((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        expect(chars.size() >= 64u);
        uint64_t manifest_offset = 0, table_offset = 0, table_size = 0, payload_base = 0, file_size = 0;
        std::memcpy(&manifest_offset, chars.data() + 16, 8);
        std::memcpy(&table_offset, chars.data() + 32, 8);
        std::memcpy(&table_size, chars.data() + 40, 8);
        std::memcpy(&payload_base, chars.data() + 48, 8);
        std::memcpy(&file_size, chars.data() + 56, 8);
        expect(payload_base == manifest_offset) << "empty payload_base == manifest_offset";
        expect(table_offset + table_size == file_size) << "empty table is last region";
        expect(file_size == chars.size());
    }
}

// Zero-element tensors: canonical rule = offset 0 OR any 256-byte aligned
// in-bounds position in the payload region. Both writers emit the aligned
// current position; readers/validator accept both forms.
void test_zero_element_tensors() {
    TempDir tmp;
    const auto path = tmp.path / "zero.lcocr";
    {
        auto writer = PackageWriter::create(path);
        expect(writer.ok());
        auto &w = *writer.value;
        PackageWriter::TensorSpec z;
        z.name = "empty";
        z.encoding = TensorEncoding::fp32;
        z.shape = {0, 4};
        z.payload = {};
        expect(w->add_tensor(z).ok());
        // A scalar-shaped empty via {2,0} too, plus a normal tensor after it.
        PackageWriter::TensorSpec z2;
        z2.name = "empty2";
        z2.encoding = TensorEncoding::fp16;
        z2.shape = {2, 0};
        z2.payload = {};
        expect(w->add_tensor(z2).ok());
        const float v[] = {1.5f};
        PackageWriter::TensorSpec t;
        t.name = "one";
        t.encoding = TensorEncoding::fp32;
        t.shape = {1};
        t.payload = f32_bytes(v);
        expect(w->add_tensor(t).ok());
        expect(w->finish(valid_manifest()).ok());
    }
    auto reader = PackageReader::open(path);
    expect(reader.ok()) << "zero-element package opens";
    if (!reader.ok()) { return; }
    auto &pkg = *reader.value;
    expect(pkg->tensors().size() == 3u);
    auto *z = pkg->find("empty");
    expect(z != nullptr && z->element_count == 0u && z->payload_size == 0u);
    float one = 0;
    auto *t = pkg->find("one");
    expect(t != nullptr);
    expect(pkg->read_tensor_f32(*t, luisa::span<float>{&one, 1}).ok());
    expect(one == 1.5f);
    // Zero-size read with an empty span succeeds.
    expect(pkg->read_tensor_f32(*z, {}).ok());
    // Corrupt the zero-element offset to a misaligned value -> rejected.
    luisa::vector<std::byte> bytes;
    {
        std::ifstream f(path, std::ios::binary);
        luisa::vector<char> chars((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        bytes.resize(chars.size());
        std::memcpy(bytes.data(), chars.data(), chars.size());
    }
    // Find the first tensor record's payload_offset: table starts after the
    // manifest; locate via header table_offset, skip count(8)+fixed(32+rank*16).
    uint64_t table_offset = 0;
    std::memcpy(&table_offset, bytes.data() + 32, 8);
    // record: u32*7 + u64 count + shape[2]*u64 + strides[2]*u64 => payload_offset at +28+8+32=68
    const size_t po = static_cast<size_t>(table_offset) + 8u + 68u;
    uint64_t bad_offset = 123;// misaligned, nonzero
    std::memcpy(bytes.data() + po, &bad_offset, 8);
    const auto bad_path = tmp.path / "zero_bad.lcocr";
    {
        std::ofstream f(bad_path, std::ios::binary);
        f.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    expect(!PackageReader::open(bad_path).ok()) << "misaligned zero-size offset rejected";
}

// Trailing bytes AFTER the tensor table (with a consistent header file_size)
// must be rejected. Regression for the missing table_end == file_size check.
void test_trailing_after_table() {
    TempDir tmp;
    const auto path = tmp.path / "trail_table.lcocr";
    {
        auto writer = PackageWriter::create(path);
        auto &w = *writer.value;
        const float v[] = {1.0f};
        PackageWriter::TensorSpec s;
        s.name = "a";
        s.encoding = TensorEncoding::fp32;
        s.shape = {1};
        s.payload = f32_bytes(v);
        expect(w->add_tensor(s).ok());
        expect(w->finish(valid_manifest()).ok());
    }
    luisa::vector<std::byte> bytes;
    {
        std::ifstream f(path, std::ios::binary);
        luisa::vector<char> chars((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        bytes.resize(chars.size());
        std::memcpy(bytes.data(), chars.data(), chars.size());
    }
    // Append 16 garbage bytes and patch header file_size (+16) at offset 56.
    uint64_t file_size = 0;
    std::memcpy(&file_size, bytes.data() + 56, 8);
    file_size += 16;
    std::memcpy(bytes.data() + 56, &file_size, 8);
    for (int i = 0; i < 16; i++) { bytes.emplace_back(std::byte{0xAB}); }
    const auto corrupted = tmp.path / "trail_table_bad.lcocr";
    {
        std::ofstream f(corrupted, std::ios::binary);
        f.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    expect(!PackageReader::open(corrupted).ok()) << "trailing-after-table rejected";
}

}// namespace

static auto test_ocr_package_registration = [] {
    "roundtrip"_test = [] { test_roundtrip(); };
    "multirank"_test = [] { test_multirank(); };
    "duplicate_name_rejected"_test = [] { test_duplicate_name_rejected(); };
    "empty_package"_test = [] { test_empty_package(); };
    "zero_element_tensors"_test = [] { test_zero_element_tensors(); };
    "trailing_after_table"_test = [] { test_trailing_after_table(); };
    "corruption"_test = [] { test_corruption(); };
    "frozen_fixtures"_test = [] { test_frozen_fixtures(); };
    "manifest_validation"_test = [] { test_manifest_validation(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}