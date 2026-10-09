// Direct-storage integration test.
//
// Covers the public DStorageExt surface on the backends that implement it
// (real DirectStorage on DX, the host-staged fallback on VK, mapped files on
// CUDA / MTLIO on Metal):
//
//   1. extension discovery + compression capability probe
//   2. file open/close, size round-trip, invalid path, zero-length subviews
//   3. file -> buffer (exact bytes, sub-view, clamping, > staging-size split)
//   4. file -> image 2D (full region, split across the staging buffer,
//      pitch-aligned file layout, sub-region + mip level)
//   5. file -> volume 3D
//   6. pinned host memory -> buffer / image / raw memory
//   7. AnySource streams
//   8. GDeflate compression round-trip (backends that advertise it)
//   9. ordering with a compute stream through a timeline event
//  10. repeatability / staging reuse

#include "ut/ut.hpp"
#include "test_device.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/volume.h>
#include <luisa/runtime/event.h>
#include <luisa/core/logging.h>
#include <luisa/backends/ext/dstorage_ext.hpp>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

// A 1 MiB staging buffer keeps the DX staging-buffer global small enough to
// force request splitting while every individual sub-request still fits.
constexpr size_t kStagingBufferSize = 1ull << 20;
constexpr uint32_t kPatternSeed = 0x9e3779b9u;

// --- fail-closed boundary (must run in a child process) ---------------------
// `LUISA_ERROR` terminates the process, so the documented rejection of a
// compressed read on a backend without a compressed path can only be observed
// from outside.  The child re-runs this binary with `--negative`.
constexpr auto kNegativeFlag = "--negative";
constexpr auto kNegativeDiagnostic = "cannot decompress";


[[nodiscard]] uint8_t pattern_byte(size_t i) noexcept {
    auto v = static_cast<uint32_t>(i) * 2654435761u + kPatternSeed;
    return static_cast<uint8_t>((v >> 13u) ^ v);
}

[[nodiscard]] luisa::vector<uint8_t> make_pattern(size_t size) noexcept {
    luisa::vector<uint8_t> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = pattern_byte(i);
    }
    return data;
}

[[nodiscard]] std::filesystem::path temp_root() noexcept {
    auto dir = std::filesystem::temp_directory_path() / "luisa_dstorage_test";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir;
}

[[nodiscard]] std::filesystem::path write_file(luisa::string_view name,
                                               luisa::span<const uint8_t> data) noexcept {
    auto path = temp_root() / std::string{name};
    std::ofstream out{path, std::ios::binary | std::ios::trunc};
    out.write(reinterpret_cast<const char *>(data.data()),
              static_cast<std::streamsize>(data.size()));
    out.close();
    return path;
}

[[nodiscard]] luisa::vector<uint8_t> read_file(const std::filesystem::path &path) noexcept {
    std::ifstream in{path, std::ios::binary | std::ios::ate};
    auto size = static_cast<size_t>(in.tellg());
    in.seekg(0);
    luisa::vector<uint8_t> data(size);
    in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(size));
    return data;
}

/// Byte-wise comparison with a diagnostic that survives the token filter.
[[nodiscard]] bool bytes_equal(luisa::span<const uint8_t> a,
                               luisa::span<const uint8_t> b,
                               luisa::string_view what) noexcept {
    if (a.size() != b.size()) {
        LUISA_ERROR("[{}] size mismatch: got {} byte(s), expected {} byte(s).",
                    what, a.size(), b.size());
        return false;
    }
    size_t bad = 0u;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            ++bad;
            if (bad == 1u) {
                LUISA_ERROR("[{}] first mismatch at byte {}: got 0x{:02x}, expected 0x{:02x}.",
                            what, i, a[i], b[i]);
            }
        }
    }
    if (bad != 0u) {
        LUISA_ERROR("[{}] {} of {} byte(s) differ.", what, bad, a.size());
        return false;
    }
    return true;
}

/// A pitch-aligned direct-storage texture source (file layout) plus the tightly
/// packed reference the destination texture is expected to hold.
struct TextureSource {
    luisa::vector<uint8_t> padded;// rows padded to the 256-byte pitch (file bytes)
    luisa::vector<uint8_t> tight; // tightly packed rows (expected texture content)
};

[[nodiscard]] TextureSource make_texture_source(PixelStorage storage,
                                                uint3 size) noexcept {
    auto pitch = dstorage_texture_row_pitch(storage, size.x);
    auto tight_row = pixel_storage_size(storage, make_uint3(size.x, 1u, 1u));
    auto rows = static_cast<size_t>(size.y) * size.z;
    TextureSource source;
    source.tight.resize(tight_row * rows);
    source.padded.resize(pitch * rows, 0x5au);// padding bytes are never visible
    for (size_t row = 0; row < rows; ++row) {
        for (size_t i = 0; i < tight_row; ++i) {
            auto v = pattern_byte(row * tight_row + i);
            source.tight[row * tight_row + i] = v;
            source.padded[row * pitch + i] = v;
        }
    }
    return source;
}

[[nodiscard]] std::string read_text_file(const std::filesystem::path &path) noexcept {
    std::ifstream in{path, std::ios::binary};
    return {std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}};
}

/// Child mode: a compressed read must be rejected (fail-closed), so this must
/// never return normally.
void run_negative_case(Device &device, DStorageExt &ext) noexcept {
    auto path = write_file("negative_compressed.bin", make_pattern(4096u));
    auto file = ext.open_file(path.string());
    if (!file) { LUISA_ERROR("negative case: could not open its input file."); }
    auto image = device.create_image<float>(PixelStorage::BYTE4, make_uint2(8u, 8u));
    auto stream = ext.create_stream();
    LUISA_INFO("negative case: dispatching a compressed read that must be rejected.");
    stream << file.copy_to(image, DStorageCompression::GDeflate) << synchronize();
    LUISA_ERROR("negative case: the compressed read was accepted, but it must be rejected.");
}

/// Runs `this <backend> --negative` in a child process and reports whether it
/// died with the documented diagnostic (and a non-zero status).
///
/// The command is spelled `cmd /c ...` on purpose: MSVC's `system()` wraps its
/// argument in quotes, and a command that both starts and ends with a quote
/// gets mangled by `cmd.exe`'s outer-quote handling (the redirect is lost and
/// the status becomes 1).  Routing through an explicit `cmd /c` avoids that.
[[nodiscard]] bool negative_case_rejected(luisa::string_view backend,
                                         luisa::string_view expected_diagnostic) noexcept {
    auto log_path = temp_root() / "negative_case.log";
    std::error_code ec;
    std::filesystem::remove(log_path, ec);
    char command[4096];
    std::snprintf(command, sizeof(command), "cmd /c \"\"%s\" %.*s %.*s\" > \"%s\" 2>&1",
                  luisa::test::safe_argv0(),
                  static_cast<int>(backend.size()), backend.data(),
                  static_cast<int>(std::strlen(kNegativeFlag)), kNegativeFlag,
                  log_path.string().c_str());
    auto status = std::system(command);
    auto log = read_text_file(log_path);
    std::filesystem::remove(log_path, ec);
    auto saw_diagnostic =
        log.find(std::string{expected_diagnostic.data(), expected_diagnostic.size()}) !=
        std::string::npos;
    LUISA_INFO("negative case '{}': child status = {}, documented diagnostic "
               "captured = {}, log tail = {} byte(s).",
               backend, status, saw_diagnostic, log.size());
    if (!saw_diagnostic) {
        LUISA_WARNING("negative case: the child did not report the documented "
                      "diagnostic; the log held {} byte(s).",
                      log.size());
    }
    return status != 0 && saw_diagnostic;
}

[[nodiscard]] luisa::vector<uint8_t> readback_buffer(Stream &stream,
                                                     const Buffer<uint8_t> &buffer) noexcept {
    luisa::vector<uint8_t> data(buffer.size());
    stream << buffer.copy_to(luisa::span{data}) << synchronize();
    return data;
}

[[nodiscard]] luisa::vector<uint8_t> readback_image_level(Stream &stream,
                                                          Image<float> &image,
                                                          uint32_t level) noexcept {
    auto view = image.view(level);
    luisa::vector<uint8_t> data(view.size_bytes());
    stream << view.copy_to(luisa::span{data}) << synchronize();
    return data;
}

[[nodiscard]] luisa::vector<uint8_t> readback_image(Stream &stream,
                                                    Image<float> &image) noexcept {
    return readback_image_level(stream, image, 0u);
}

[[nodiscard]] luisa::vector<uint8_t> readback_volume(Stream &stream,
                                                     Volume<float> &volume) noexcept {
    luisa::vector<uint8_t> data(volume.view().size_bytes());
    stream << volume.copy_to(luisa::span{data}) << synchronize();
    return data;
}

void test_dstorage(Device &device, luisa::string_view requested_backend) {

    auto *ext = device.extension<DStorageExt>();
    if (ext == nullptr) {
        LUISA_INFO("Skipping direct-storage test: backend '{}' does not provide DStorageExt.",
                   device.backend_name());
        return;
    }

    // ---- 1. discovery ------------------------------------------------------
    auto supports_none = ext->supports_compression(DStorageCompression::None);
    auto supports_gdeflate = ext->supports_compression(DStorageCompression::GDeflate);
    LUISA_INFO("DStorage on '{}': supports_compression(None) = {}, supports_compression(GDeflate) = {}.",
               device.backend_name(), supports_none, supports_gdeflate);
    expect(supports_none) << "DStorageCompression::None must always be supported";
    if (requested_backend == "vk") {
        // The Vulkan fallback is host-staged and ships no compressor: if it ever
        // started advertising one, a compressed read would silently be wrong.
        expect(!supports_gdeflate) << "the Vulkan DirectStorage fallback must not advertise GDeflate";
        // ...and a compressed read must fail closed rather than produce garbage.
        expect(negative_case_rejected(requested_backend, kNegativeDiagnostic))
            << "a compressed read on the Vulkan fallback must be rejected";
    }

    auto compute_stream = device.create_stream();

    // ---- 2. stream + file basics ------------------------------------------
    DStorageStreamOption option;
    option.source = DStorageStreamSource::AnySource;
    option.staging_buffer_size = kStagingBufferSize;
    auto dstream = ext->create_stream(option);
    if (!dstream) {
        LUISA_INFO("Skipping direct-storage test: backend '{}' could not create a "
                   "DStorage stream (the DirectStorage runtime may be missing).",
                   device.backend_name());
        return;
    }

    {
        auto missing = ext->open_file((temp_root() / "does_not_exist.bin").string());
        expect(!missing) << "opening a missing path must produce an invalid DStorageFile";
    }

    auto basic_path = write_file("basic.bin", make_pattern(4096u));
    auto basic_file = ext->open_file(basic_path.string());
    expect(static_cast<bool>(basic_file)) << "opening an existing file must succeed";
    expect(basic_file.size_bytes() == 4096u) << "size_bytes() must round-trip";
    // Zero-length views at (and fractions of) the end of the file are legal.
    expect(basic_file.view(basic_file.size_bytes(), 0u).size_bytes() == 0u);
    expect(basic_file.view().subview(0u, 0u).size_bytes() == 0u);
    expect(basic_file.view(1024u, 2048u).size_bytes() == 2048u);
    {
        auto reopened = ext->open_file(basic_path.string());
        expect(static_cast<bool>(reopened)) << "opening the same file twice must succeed";
    }

    // ---- 3. file -> buffer -------------------------------------------------
    {
        auto buffer = device.create_buffer<uint8_t>(4096u);
        dstream << basic_file.copy_to(buffer) << synchronize();
        auto got = readback_buffer(compute_stream, buffer);
        expect(bytes_equal(got, make_pattern(4096u), "file->buffer exact")) << "file -> buffer must be byte exact";
    }
    {
        // Sub-view: only the addressed window may be written.
        auto buffer = device.create_buffer<uint8_t>(4096u);
        auto fill = luisa::vector<uint8_t>(4096u, 0xabu);
        compute_stream << buffer.copy_from(luisa::span{fill}) << synchronize();
        dstream << basic_file.view(1024u, 512u).copy_to(buffer.view(2048u, 512u)) << synchronize();
        auto got = readback_buffer(compute_stream, buffer);
        auto expected = fill;
        auto src = make_pattern(4096u);
        std::memcpy(expected.data() + 2048u, src.data() + 1024u, 512u);
        expect(bytes_equal(got, expected, "file(view)->buffer(view)")) << "sub-view copy must touch only the window";
    }
    {
        // Clamping: a short source view must never read past the view, and the
        // tail of the destination must stay untouched (the regression test for
        // the EOF over-read bug).
        auto buffer = device.create_buffer<uint8_t>(4096u);
        auto fill = luisa::vector<uint8_t>(4096u, 0xcdu);
        compute_stream << buffer.copy_from(luisa::span{fill}) << synchronize();
        dstream << basic_file.view(0u, 128u).copy_to(buffer) << synchronize();
        auto got = readback_buffer(compute_stream, buffer);
        auto expected = fill;
        auto src = make_pattern(4096u);
        std::memcpy(expected.data(), src.data(), 128u);
        expect(bytes_equal(got, expected, "clamped file->buffer")) << "the request size must be clamped to the source view";
    }
    {
        // Above the staging buffer size: the copy must be split transparently.
        auto big_size = kStagingBufferSize * 3u + 12345u;
        auto big_path = write_file("big.bin", make_pattern(big_size));
        auto big_file = ext->open_file(big_path.string());
        expect(static_cast<bool>(big_file));
        expect(big_file.size_bytes() == big_size);
        auto buffer = device.create_buffer<uint8_t>(big_size);
        dstream << big_file.copy_to(buffer) << synchronize();
        auto got = readback_buffer(compute_stream, buffer);
        expect(bytes_equal(got, make_pattern(big_size), "large file->buffer")) << "splitting across the staging buffer must be transparent";
    }

    // ---- 4. file -> image (2D) --------------------------------------------
    {
        // 1024x1024 BYTE4 -> 4 MiB padded, i.e. 4 staging-sized requests on DX.
        constexpr uint32_t w = 1024u, h = 1024u;
        auto source = make_texture_source(PixelStorage::BYTE4, uint3{w, h, 1u});
        auto path = write_file("image2d.bin", source.padded);
        auto file = ext->open_file(path.string());
        expect(static_cast<bool>(file));
        auto image = device.create_image<float>(PixelStorage::BYTE4, make_uint2(w, h));
        dstream << file.copy_to(image) << synchronize();
        auto got = readback_image(compute_stream, image);
        expect(bytes_equal(got, source.tight, "file->image2d")) << "pitch-aligned file -> tightly packed image must match";
    }
    {
        // Non-pitch-aligned width: 100 texels of BYTE4 is 400 bytes per row,
        // padded to 512.  The VK fallback must repack and DX/CUDA must read the
        // padded source with the 256-byte pitch.
        constexpr uint32_t w = 100u, h = 32u;
        expect(dstorage_texture_row_pitch(PixelStorage::BYTE4, w) == 512u);
        auto source = make_texture_source(PixelStorage::BYTE4, uint3{w, h, 1u});
        expect(source.padded.size() == 512u * h) << "the padded layout must pad every row to 256 bytes";
        auto path = write_file("image2d_padded.bin", source.padded);
        auto file = ext->open_file(path.string());
        auto image = device.create_image<float>(PixelStorage::BYTE4, make_uint2(w, h));
        dstream << file.copy_to(image) << synchronize();
        auto got = readback_image(compute_stream, image);
        expect(bytes_equal(got, source.tight, "file->image2d(padded rows)"))
            << "a non-256-aligned row width must still be handled by every backend";
    }
    {
        // Sub-region at a non-zero coordinate + mip 0, plus a mip level.
        constexpr uint32_t w = 256u, h = 256u;
        auto image = device.create_image<float>(PixelStorage::BYTE4, make_uint2(w, h), 2u);
        // Device memory is not guaranteed to be zero-initialized, so clear the
        // level explicitly before checking that only the region is written.
        auto zeros = luisa::vector<uint8_t>(w * h * 4u, 0u);
        compute_stream << image.copy_from(luisa::span{zeros}) << synchronize();
        auto region = make_texture_source(PixelStorage::BYTE4, uint3{64u, 48u, 1u});
        auto path = write_file("image2d_sub.bin", region.padded);
        auto file = ext->open_file(path.string());
        dstream << file.view().copy_to(image, make_uint2(32u, 16u), make_uint2(64u, 48u), 0u) << synchronize();
        auto got = readback_image(compute_stream, image);
        auto expected = luisa::vector<uint8_t>(w * h * 4u, 0u);
        for (uint32_t y = 0; y < 48u; ++y) {
            std::memcpy(expected.data() + ((16u + y) * w + 32u) * 4u,
                        region.tight.data() + y * 64u * 4u, 64u * 4u);
        }
        expect(bytes_equal(got, expected, "file->image2d(sub-region)")) << "sub-region copy must land at the right texels";

        // Mip level 1 (128x128 for a 256x256 image): the copy must target that
        // level only.
        auto mip_source = make_texture_source(PixelStorage::BYTE4, uint3{128u, 128u, 1u});
        auto mip_path = write_file("image2d_mip1.bin", mip_source.padded);
        auto mip_file = ext->open_file(mip_path.string());
        auto mip_image = device.create_image<float>(PixelStorage::BYTE4, make_uint2(w, h), 2u);
        dstream << mip_file.copy_to(mip_image.view(1u)) << synchronize();
        auto mip_got = readback_image_level(compute_stream, mip_image, 1u);
        expect(bytes_equal(mip_got, mip_source.tight, "file->image2d(mip1)")) << "copying into a mip level must target that level only";
    }

    // ---- 5. file -> volume (3D) -------------------------------------------
    {
        constexpr uint32_t w = 128u, h = 64u, d = 40u;
        auto source = make_texture_source(PixelStorage::BYTE4, uint3{w, h, d});
        // 512 * 64 * 40 = 1.25 MiB -> split across a 1 MiB staging buffer.
        expect(source.padded.size() > kStagingBufferSize) << "the volume case must exercise splitting";
        auto path = write_file("volume3d.bin", source.padded);
        auto file = ext->open_file(path.string());
        expect(static_cast<bool>(file));
        auto volume = device.create_volume<float>(PixelStorage::BYTE4, make_uint3(w, h, d));
        dstream << file.copy_to(volume) << synchronize();
        auto got = readback_volume(compute_stream, volume);
        expect(bytes_equal(got, source.tight, "file->volume3d")) << "volume copy must match, including across the staging split";
    }

    // ---- 5b. lifetimes: non-owning views, file outliving its reads -------
    {
        // A `DStorageFileView` is non-owning, so it may die before the transfer
        // completes; the file itself must outlive its pending reads (see the
        // lifetime contract in dstorage_ext_interface.h).
        auto file = ext->open_file(basic_path.string());
        expect(static_cast<bool>(file));
        auto buffer = device.create_buffer<uint8_t>(1024u);
        {
            auto view = file.view(512u, 1024u);// destroyed at the end of this scope
            dstream << view.copy_to(buffer) << synchronize();
        }
        auto expected = make_pattern(4096u);
        auto slice = luisa::vector<uint8_t>(expected.begin() + 512, expected.begin() + 512 + 1024);
        expect(bytes_equal(readback_buffer(compute_stream, buffer), slice, "dead-view read"))
            << "a read must stay valid after its view has been destroyed";
        // The same file handle is still usable for a second read.
        auto small = device.create_buffer<uint8_t>(128u);
        dstream << file.view(0u, 128u).copy_to(small) << synchronize();
        auto head = luisa::vector<uint8_t>(expected.begin(), expected.begin() + 128);
        expect(bytes_equal(readback_buffer(compute_stream, small), head, "reused file handle"))
            << "a file handle must remain usable after a synchronized read";
    }

    // ---- 6. pinned host memory --------------------------------------------
    {
        auto host = make_pattern(kStagingBufferSize + 4096u);
        auto pinned = ext->pin_memory(host.data(), host.size());
        expect(static_cast<bool>(pinned)) << "pin_memory must succeed for a valid range";
        {
            auto buffer = device.create_buffer<uint8_t>(host.size());
            dstream << pinned.copy_to(buffer) << synchronize();
            auto got = readback_buffer(compute_stream, buffer);
            expect(bytes_equal(got, host, "pinned->buffer")) << "memory source -> buffer must be byte exact";
        }
        {
            auto source = make_texture_source(PixelStorage::BYTE4, uint3{128u, 64u, 1u});
            std::memcpy(host.data(), source.padded.data(), source.padded.size());
            auto image = device.create_image<float>(PixelStorage::BYTE4, make_uint2(128u, 64u));
            dstream << pinned.view(0u, source.padded.size()).copy_to(image) << synchronize();
            auto got = readback_image(compute_stream, image);
            expect(bytes_equal(got, source.tight, "pinned->image2d")) << "memory source -> texture must be pitch aligned";
        }
        {
            luisa::vector<uint8_t> out(host.size(), 0u);
            dstream << pinned.copy_to(out.data(), out.size()) << synchronize();
            expect(bytes_equal(out, host, "pinned->memory")) << "memory source -> raw memory must be byte exact";
        }
        expect(basic_file.size_bytes() == 4096u) << "pinning/unpinning must not disturb other resources";
    }
    {
        // Source-specific streams still accept their own source type.
        auto memory_stream = ext->create_stream(
            DStorageStreamOption{DStorageStreamSource::MemorySource, kStagingBufferSize});
        expect(static_cast<bool>(memory_stream));
        auto host = make_pattern(4096u);
        auto pinned = ext->pin_memory(host.data(), host.size());
        auto buffer = device.create_buffer<uint8_t>(4096u);
        memory_stream << pinned.copy_to(buffer) << synchronize();
        auto got = readback_buffer(compute_stream, buffer);
        expect(bytes_equal(got, host, "memory-only stream")) << "a MemorySource stream must serve memory reads";
    }

    // ---- 7. AnySource stream: both source kinds in one queue ---------------
    {
        auto host = make_pattern(2048u);
        auto pinned = ext->pin_memory(host.data(), host.size());
        auto from_file = device.create_buffer<uint8_t>(4096u);
        auto from_memory = device.create_buffer<uint8_t>(2048u);
        dstream << basic_file.copy_to(from_file)
                << pinned.copy_to(from_memory)
                << synchronize();
        expect(bytes_equal(readback_buffer(compute_stream, from_file), make_pattern(4096u), "AnySource(file)"))
            << "an AnySource stream must serve file-sourced reads";
        expect(bytes_equal(readback_buffer(compute_stream, from_memory), host, "AnySource(memory)"))
            << "an AnySource stream must serve memory-sourced reads";
    }

    // ---- 8. compression ----------------------------------------------------
    if (supports_gdeflate) {
        // 200 texels of BYTE4 is 800 bytes per row: not 256-byte aligned, so
        // the padded source layout (1024 B/row) differs from the tightly packed
        // destination footprint.  This is what distinguishes a correct
        // `UncompressedSize` (copyable footprint) from the tightly packed one.
        constexpr uint32_t w = 200u, h = 64u;
        expect(dstorage_texture_row_pitch(PixelStorage::BYTE4, w) != w * 4u)
            << "the compression case must use a non-pitch-aligned width";
        auto source = make_texture_source(PixelStorage::BYTE4, uint3{w, h, 1u});
        luisa::vector<std::byte> compressed;
        ext->compress(source.padded.data(), source.padded.size(),
                      DStorageCompression::GDeflate,
                      DStorageCompressionQuality::Default, compressed);
        expect(!compressed.empty()) << "compress() must produce output";
        LUISA_INFO("GDeflate: {} byte(s) -> {} byte(s).", source.padded.size(), compressed.size());
        auto path = write_file("compressed.gdeflate",
                               luisa::span{reinterpret_cast<const uint8_t *>(compressed.data()),
                                           compressed.size()});
        auto file = ext->open_file(path.string());
        expect(static_cast<bool>(file));
        auto image = device.create_image<float>(PixelStorage::BYTE4, make_uint2(w, h));
        dstream << file.copy_to(image, DStorageCompression::GDeflate) << synchronize();
        auto got = readback_image(compute_stream, image);
        expect(bytes_equal(got, source.tight, "gdeflate->image2d")) << "GDeflate decompression must reproduce the source";
    } else {
        LUISA_INFO("Backend '{}' does not support GDeflate; skipping the compression round-trip.",
                   device.backend_name());
    }

    // ---- 9. ordering with a compute stream ---------------------------------
    {
        auto event = device.create_timeline_event();
        auto buffer = device.create_buffer<uint8_t>(4096u);
        auto out = device.create_buffer<uint8_t>(4096u);
        dstream << basic_file.copy_to(buffer) << event.signal(1u);
        compute_stream << event.wait(1u) << buffer.view().copy_to(out.view()) << synchronize();
        auto got = readback_buffer(compute_stream, out);
        expect(bytes_equal(got, make_pattern(4096u), "event-ordered file->buffer"))
            << "a compute stream gated on the DStorage timeline event must observe the copied data";
    }

    // ---- 10. repeatability (staging reuse) ---------------------------------
    for (size_t iteration = 0u; iteration < 3u; ++iteration) {
        static_cast<void>(iteration);
        auto source = make_texture_source(PixelStorage::BYTE4, uint3{256u, 128u, 1u});
        auto path = write_file("repeat.bin", source.padded);
        auto file = ext->open_file(path.string());
        auto buffer = device.create_buffer<uint8_t>(source.padded.size());
        auto image = device.create_image<float>(PixelStorage::BYTE4, make_uint2(256u, 128u));
        dstream << file.copy_to(buffer) << synchronize();
        dstream << file.copy_to(image) << synchronize();
        expect(bytes_equal(readback_buffer(compute_stream, buffer), source.padded, "repeat->buffer"))
            << "repeated copies must reuse the staging ring correctly";
        expect(bytes_equal(readback_image(compute_stream, image), source.tight, "repeat->image"))
            << "repeated texture copies must reuse the staging ring correctly";
    }

    LUISA_INFO("Direct-storage test finished on backend '{}'.", device.backend_name());
}

}// namespace

int main(int argc, char *argv[]) {
    // `--validation` runs everything through the validation-layer device (see
    // the `test` skill); the default is the raw backend.
    auto enable_validation = false;
    auto negative = false;
    for (auto i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--validation") == 0) {
            enable_validation = true;
        } else if (std::strcmp(argv[i], kNegativeFlag) == 0) {
            negative = true;
        }
    }
    auto dc = luisa::test::create_device_from_ut(argc, argv, nullptr, enable_validation);
    if (!dc) {
        return 0;
    }
    auto &device = dc->device;
    auto *ext = device.extension<DStorageExt>();
    if (negative) {
        // Child mode: perform only the read that must be rejected.
        if (ext == nullptr) {
            LUISA_ERROR("negative case: backend '{}' does not provide DStorageExt.", argv[1]);
        }
        run_negative_case(device, *ext);
        return 0;
    }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    if (enable_validation) {
        LUISA_INFO("Direct-storage test running with the validation layer enabled.");
    }
    test_dstorage(device, argc > 1 && argv[1] != nullptr ? luisa::string_view{argv[1]} : luisa::string_view{});
}
