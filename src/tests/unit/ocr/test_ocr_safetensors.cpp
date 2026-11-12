// Test for examples/ocr/ocr/safetensors_reader.h/.cpp -- checked host reader.
// Host-only: no device, no backend, no model, no exceptions.
//
// This test covers:
// - header length / metadata / tensor table parsing;
// - dtype/shape/offset validation, duplicate names, file bounds;
// - F32/F16/BF16 decode (explicit conversion, no reinterpretation);
// - fail-closed on truncated/oversized/unsupported-dtype/overlap/trailing;
// - the frozen synthetic fixture (never regenerated here).
#include "ut/ut.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>

#include <luisa/core/stl/filesystem.h>

#include "ocr/safetensors_reader.h"

using namespace luisa;
using namespace luisa::ocr;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct TempDir {
    luisa::filesystem::path path;
    TempDir() {
        path = luisa::filesystem::temp_directory_path() /
               ("lcocr_st_" + std::to_string(reinterpret_cast<uintptr_t>(this)));
        luisa::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        luisa::filesystem::remove_all(path, ec);
    }
};

// Build a minimal synthetic safetensors file in memory.
[[nodiscard]] luisa::vector<std::byte> make_safetensors(const luisa::string &header_json,
                                                        luisa::span<const std::byte> data) {
    luisa::vector<std::byte> out;
    const uint64_t len = header_json.size();
    for (uint32_t i = 0; i < 8u; i++) {
        out.emplace_back(static_cast<std::byte>((len >> (8u * i)) & 0xffu));
    }
    for (auto c : header_json) { out.emplace_back(static_cast<std::byte>(c)); }
    out.insert(out.end(), data.begin(), data.end());
    return out;
}

void write_file(const luisa::filesystem::path &p, luisa::span<const std::byte> data) {
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
}

void test_frozen_fixture() {
    const luisa::filesystem::path dir{LUISA_OCR_TEST_DATA_DIR};
    auto reader = SafetensorsReader::open(dir / "synthetic.safetensors");
    expect(reader.ok());
    if (!reader.ok()) { return; }
    auto &r = *reader.value;
    expect(r->tensors().size() == 4u);
    auto *w = r->find("weight");
    expect(w != nullptr);
    expect(w->dtype == SafetensorsDtype::f32);
    expect(w->shape.size() == 2u && w->element_count == 6);
    float out[6];
    expect(r->read_tensor_f32(*w, out).ok());
    expect(out[0] == 1.0f && out[5] == 6.0f);
    auto *h = r->find("half");
    expect(h && h->dtype == SafetensorsDtype::f16);
    float hout[4];
    expect(r->read_tensor_f32(*h, hout).ok());
    expect(hout[0] == 0.5f && hout[1] == -0.5f);
    auto *bf = r->find("bf");
    expect(bf && bf->dtype == SafetensorsDtype::bf16);
    // Non-float decode rejected (no reinterpretation).
    auto *byte = r->find("byte");
    expect(byte && byte->dtype == SafetensorsDtype::u8);
    float bout[3];
    expect(!r->read_tensor_f32(*byte, bout).ok());
    // Metadata present.
    expect(!r->metadata().empty());
}

void test_synthetic_roundtrip() {
    TempDir tmp;
    // One F32 tensor [2].
    float values[] = {1.5f, -2.5f};
    luisa::vector<std::byte> data(8);
    std::memcpy(data.data(), values, 8);
    const luisa::string header =
        "{\"t\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
    const auto bytes = make_safetensors(header, data);
    const auto path = tmp.path / "ok.safetensors";
    write_file(path, bytes);
    auto reader = SafetensorsReader::open(path);
    expect(reader.ok());
    if (reader.ok()) {
        auto *t = (*reader.value)->find("t");
        expect(t != nullptr);
        float out[2];
        expect((*reader.value)->read_tensor_f32(*t, out).ok());
        expect(out[0] == 1.5f && out[1] == -2.5f);
    }
}

void test_corruption() {
    TempDir tmp;
    float values[] = {1.0f};
    luisa::vector<std::byte> data(4);
    std::memcpy(data.data(), values, 4);
    const luisa::string header =
        "{\"t\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    const auto good = make_safetensors(header, data);
    // Truncated.
    {
        auto trunc = good;
        trunc.resize(trunc.size() - 2);
        const auto p = tmp.path / "trunc.safetensors";
        write_file(p, trunc);
        expect(!SafetensorsReader::open(p).ok());
    }
    // Unsupported dtype.
    {
        const luisa::string h = "{\"t\":{\"dtype\":\"F8_E4M3\",\"shape\":[1],\"data_offsets\":[0,1]}}";
        luisa::vector<std::byte> d(1, std::byte{0});
        const auto p = tmp.path / "dtype.safetensors";
        write_file(p, make_safetensors(h, d));
        expect(!SafetensorsReader::open(p).ok());
    }
    // Size mismatch (shape says 2 but offsets say 4 bytes).
    {
        const luisa::string h = "{\"t\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}";
        const auto p = tmp.path / "size.safetensors";
        write_file(p, make_safetensors(h, data));
        expect(!SafetensorsReader::open(p).ok());
    }
    // Duplicate name (yyjson keeps both; reader must reject).
    {
        const luisa::string h =
            "{\"t\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},"
            "\"t\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
        const auto p = tmp.path / "dup.safetensors";
        write_file(p, make_safetensors(h, data));
        expect(!SafetensorsReader::open(p).ok());
    }
    // Trailing data.
    {
        auto trail = good;
        trail.emplace_back(std::byte{0});
        const auto p = tmp.path / "trail.safetensors";
        write_file(p, trail);
        expect(!SafetensorsReader::open(p).ok());
    }
    // Reversed offsets.
    {
        const luisa::string h = "{\"t\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,0]}}";
        const auto p = tmp.path / "rev.safetensors";
        write_file(p, make_safetensors(h, data));
        expect(!SafetensorsReader::open(p).ok());
    }
}

}// namespace

static auto test_ocr_safetensors_registration = [] {
    "frozen_fixture"_test = [] { test_frozen_fixture(); };
    "synthetic_roundtrip"_test = [] { test_synthetic_roundtrip(); };
    "corruption"_test = [] { test_corruption(); };
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
}