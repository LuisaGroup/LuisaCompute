// GDeflate GPU compression/decompression round-trip test.
//
// Generates a random byte buffer, compresses it on the GPU using a
// GDeflate-compatible tile stream, then decompresses it on the GPU and
// verifies the result matches the original data exactly.
//
// Additionally builds real fixed-Huffman and dynamic-Huffman GDeflate tiles on
// the host (mirroring libdeflate's 32-stream bit assignment) and verifies that
// the GPU decompressor decodes them correctly.

#include "gdeflate.h"

#include <luisa/luisa-compute.h>
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::example::gdeflate;

namespace {

// ============================================================================
// Host-side GDeflate tile writer
//
// Mirrors libdeflate's GDeflate output bitstream (deflate_compress.c in
// D:/DirectStorage/GDeflate/3rdparty/libdeflate):
//   - there is a current stream index (0..31) that advances after every
//     literal / length symbol and after every flushed deferred distance;
//   - block headers and the dynamic-Huffman header fields go entirely to
//     stream 0;
//   - the precode lengths and code-length items advance one stream per item;
//   - a match's distance code is deferred and written into the same stream as
//     its length code at the start of the next round (or in the tail);
//   - each stream packs bits LSB-first into 32-bit packets; packet p of
//     stream s is stored at word p*32 + s.
// ============================================================================

constexpr uint32_t kNumStreams = 32u;

inline uint32_t reverse_low_bits(uint32_t value, uint32_t n) noexcept {
    uint32_t r = 0u;
    for (uint32_t i = 0u; i < n; ++i) {
        r |= ((value >> i) & 1u) << (n - 1u - i);
    }
    return r;
}

class GDeflateTileWriter {
public:
    GDeflateTileWriter() noexcept {
        // Initial 32 packets (one per stream) occupy words 0..31.
        _words.resize(kNumStreams, 0u);
        for (uint32_t s = 0u; s < kNumStreams; ++s) {
            _write_ptr[s] = s;
            _input_bitcount[s] = 32u;
            _next_ptr[s] = -1;
        }
    }

    void reset() noexcept { _idx = 0u; }

    // Append `n` bits LSB-first to the current stream, replicating libdeflate's
    // GDeflate packet scheduling: flush a 32-bit packet when the stream bit
    // buffer fills, and reserve the next packet whenever the emulated decoder
    // bit count (input_bitcount) drops below 32.
    void add_bits(uint32_t bits, uint32_t n) noexcept {
        const uint32_t s = _idx;
        _bitbuf[s] |= static_cast<uint64_t>(bits & mask32(n)) << _bitcount[s];
        _bitcount[s] += n;
        _input_bitcount[s] -= n;
        if (_bitcount[s] >= 32u) {
            _words[_write_ptr[s]] = static_cast<uint32_t>(_bitbuf[s]);
            _bitbuf[s] >>= 32u;
            _bitcount[s] -= 32u;
            _write_ptr[s] = static_cast<uint32_t>(_next_ptr[s]);
            _next_ptr[s] = -1;
        }
        if (_input_bitcount[s] < 32u && _next_ptr[s] < 0) {
            _next_ptr[s] = static_cast<int32_t>(_next_word);
            _words.push_back(0u);
            ++_next_word;
            _input_bitcount[s] += 32u;
        }
    }

    void advance() noexcept {
        _idx = (_idx + 1u) % kNumStreams;
        if (_idx == 0u) { ++_round; }
    }

    // Flush deferred copies whose stream is about to receive the next symbol.
    void write_prev_round_copies() noexcept {
        while (_round > 1u && _copies[_idx].active && _copies[_idx].round == _round - 1u) {
            _copies[_idx].active = false;
            add_bits(_copies[_idx].bits, _copies[_idx].bitcount);
            advance();
        }
    }

    // Emit a DEFLATE Huffman code (MSB-first) to the current stream.
    void emit_huffman(uint32_t code, uint32_t len) noexcept {
        add_bits(reverse_low_bits(code, len), len);
    }

    // Emit one literal (or any litlen symbol without deferred distance).
    void emit_literal(uint32_t code, uint32_t len) noexcept {
        write_prev_round_copies();
        emit_huffman(code, len);
        advance();
    }

    // Emit a match: length code (+ extra bits) now, distance code (+ extra
    // bits) deferred into the same stream.
    void emit_match(uint32_t len_code, uint32_t len_code_len,
                    uint32_t len_extra, uint32_t len_extra_n,
                    uint32_t dist_code, uint32_t dist_code_len,
                    uint32_t dist_extra, uint32_t dist_extra_n) noexcept {
        write_prev_round_copies();
        add_bits(reverse_low_bits(len_code, len_code_len) | (len_extra << len_code_len),
                 len_code_len + len_extra_n);
        auto &c = _copies[_idx];
        c.active = true;
        c.round = _round;
        c.bits = reverse_low_bits(dist_code, dist_code_len) | (dist_extra << dist_code_len);
        c.bitcount = dist_code_len + dist_extra_n;
        advance();
    }

    // Emit the end-of-block symbol to the current stream (no advance).
    void emit_eob(uint32_t code, uint32_t len) noexcept {
        emit_huffman(code, len);
    }


    // Flush remaining deferred copies and all partial packets; return the
    // swizzled word array (packet p of stream s is not necessarily at p*32+s;
    // the placement follows libdeflate's reservation order).
    std::vector<uint32_t> finish() noexcept {
        const uint32_t split = _idx % kNumStreams;
        for (uint32_t n = split; n < kNumStreams; ++n) {
            if (_copies[n].active && _round > 1u && _copies[n].round == _round - 1u) {
                _idx = n;
                add_bits(_copies[n].bits, _copies[n].bitcount);
                _copies[n].active = false;
            }
        }
        for (uint32_t n = 0u; n < split; ++n) {
            if (_copies[n].active && _copies[n].round == _round) {
                _idx = n;
                add_bits(_copies[n].bits, _copies[n].bitcount);
                _copies[n].active = false;
            }
        }
        // Final flush: write every stream's partial packet in stream order.
        for (uint32_t s = 0u; s < kNumStreams; ++s) {
            _words[_write_ptr[s]] = static_cast<uint32_t>(_bitbuf[s]);
            _bitbuf[s] >>= 32u;
            _bitcount[s] -= 32u;
            _write_ptr[s] = static_cast<uint32_t>(_next_ptr[s]);
            _next_ptr[s] = -1;
        }
        return std::move(_words);
    }

private:
    static uint32_t mask32(uint32_t n) noexcept {
        return n >= 32u ? 0xFFFFFFFFu : ((1u << n) - 1u);
    }

    struct DeferredCopy {
        bool active = false;
        uint32_t round = 0u;
        uint32_t bits = 0u;
        uint32_t bitcount = 0u;
    };

    std::array<uint64_t, kNumStreams> _bitbuf{};
    std::array<uint32_t, kNumStreams> _bitcount{};
    std::array<uint32_t, kNumStreams> _input_bitcount{}; // starts at 32 per stream
    std::array<uint32_t, kNumStreams> _write_ptr{};      // word index of current packet
    std::array<int32_t, kNumStreams> _next_ptr{};        // reserved next packet, -1 = none
    std::array<DeferredCopy, kNumStreams> _copies{};
    std::vector<uint32_t> _words;                        // word array (grows with reservations)
    uint32_t _next_word = kNumStreams;
    uint32_t _idx = 0u;
    uint32_t _round = 1u;
};

// ============================================================================
// Fixed Huffman code tables (standard DEFLATE fixed codes).
// ============================================================================

inline void fixed_litlen_tables(std::array<uint32_t, 288> &codes, std::array<uint32_t, 288> &lens) noexcept {
    for (uint32_t b = 0u; b < 144u; ++b) { codes[b] = b + 0x30u; lens[b] = 8u; }
    for (uint32_t b = 144u; b < 256u; ++b) { codes[b] = b - 144u + 0x190u; lens[b] = 9u; }
    // EOB = 256 -> 7-bit code 0; length codes 257..287.
    codes[256] = 0u; lens[256] = 7u;
    for (uint32_t s = 257u; s < 280u; ++s) { codes[s] = s - 256u; lens[s] = 7u; }
    for (uint32_t s = 280u; s < 288u; ++s) { codes[s] = s - 280u + 0xC0u; lens[s] = 8u; }
}

// ============================================================================
// Tile stream assembly (matches write_tilestream_header in gdeflate_kernels.h).
// ============================================================================

inline std::vector<uint8_t> build_tile_stream(const std::vector<uint32_t> &tile_words,
                                              uint32_t output_size) noexcept {
    constexpr uint32_t num_tiles = 1u;
    uint32_t tile_bytes = static_cast<uint32_t>(tile_words.size() * sizeof(uint32_t));
    uint32_t data_start = kStreamHeaderSize + num_tiles * sizeof(uint32_t); // 12
    std::vector<uint8_t> stream(data_start + tile_bytes, 0u);
    uint32_t word1 = 4u | (0xFBu << 8u) | (num_tiles << 16u);
    uint32_t word2 = output_size << 2u;
    std::memcpy(stream.data(), &word1, sizeof(word1));
    std::memcpy(stream.data() + 4u, &word2, sizeof(word2));
    // Pointer table: tile 0's compressed size (offset is relative to data start).
    std::memcpy(stream.data() + 8u, &tile_bytes, sizeof(tile_bytes));
    std::memcpy(stream.data() + data_start, tile_words.data(), tile_bytes);
    return stream;
}

// Run one crafted tile through the GPU decompressor and compare the output.
bool run_tile_test(GDeflateCodec &codec, Stream &stream, Device &device,
                   const char *name, const std::vector<uint32_t> &tile_words,
                   const std::vector<uint8_t> &expected) noexcept {
    auto tile_stream = build_tile_stream(tile_words, static_cast<uint32_t>(expected.size()));
    ByteBuffer input = device.create_byte_buffer(tile_stream.size());
    stream << input.copy_from(tile_stream.data()) << synchronize();

    ByteBuffer output = codec.allocate_uncompressed(static_cast<uint32_t>(expected.size()));
    codec.decompress(input, output, static_cast<uint32_t>(expected.size()));

    std::vector<uint8_t> host_out(expected.size());
    stream << output.copy_to(host_out.data()) << synchronize();

    if (host_out != expected) {
        LUISA_WARNING("GDeflate {} tile test FAILED ({} bytes)", name, expected.size());
        for (size_t i = 0u; i < expected.size(); ++i) {
            if (host_out[i] != expected[i]) {
                LUISA_WARNING("  first mismatch at byte {}: expected {}, got {}",
                              i, static_cast<int>(expected[i]), static_cast<int>(host_out[i]));
                break;
            }
        }
        return false;
    }
    LUISA_INFO("GDeflate {} tile test: PASSED ({} bytes)", name, expected.size());
    return true;
}

// Fixed-Huffman tile containing every byte value 0..255 as literals.
std::vector<uint32_t> build_fixed_literal_tile() noexcept {
    std::array<uint32_t, 288> codes{};
    std::array<uint32_t, 288> lens{};
    fixed_litlen_tables(codes, lens);
    GDeflateTileWriter w;
    w.reset();
    w.add_bits(1u, 1u); // BFINAL = 1
    w.add_bits(1u, 2u); // BTYPE = 01 (fixed Huffman)
    for (uint32_t b = 0u; b < 256u; ++b) {
        w.emit_literal(codes[b], lens[b]);
    }
    w.emit_eob(codes[256], lens[256]);
    w.advance();
    return w.finish();
}

// Fixed-Huffman tile with a real length/distance match:
// "ABCD" + (length 4, distance 4) + EOB -> "ABCDABCD" (8 bytes).
// DEFLATE64: length 4 = symbol 258 (baseLength[2], 0 extra),
// distance 4 = symbol 3 (baseDist[3], 0 extra).
std::vector<uint32_t> build_fixed_match_tile() noexcept {
    std::array<uint32_t, 288> codes{};
    std::array<uint32_t, 288> lens{};
    fixed_litlen_tables(codes, lens);
    GDeflateTileWriter w;
    w.reset();
    w.add_bits(1u, 1u); // BFINAL = 1
    w.add_bits(1u, 2u); // BTYPE = 01 (fixed Huffman)
    const char data[] = "ABCD";
    for (uint32_t i = 0u; i < 4u; ++i) {
        auto c = static_cast<uint8_t>(data[i]);
        w.emit_literal(codes[c], lens[c]);
    }
    w.emit_match(codes[258], lens[258], 0u, 0u, // length 4, 0 extra bits
                 3u, 5u, 0u, 0u);               // distance 4 = symbol 3, 0 extra bits
    w.emit_eob(codes[256], lens[256]);
    w.advance();
    return w.finish();
}

// Dynamic-Huffman tile per the task recipe:
//   litlen tree: symbols 0..253 length 8 (codes 0..253),
//                symbols 254,255,256(EOB),257(length 3) length 9
//                (codes 508,509,510,511); Kraft = 1.0.
//   distance tree: symbol 0 length 1 (code 0, distance 1).
//   HLIT=258, HDIST=1, HCLEN=14.
//   precode: sym1 length 1 (code 0), sym8 length 2 (code 10), sym9 length 2
//            (code 11).
//   data: 'A' (code 65) + length 257 (length 3, distance 1) + EOB
//         -> "AAAA" (4 bytes).
std::vector<uint32_t> build_dynamic_tile() noexcept {
    GDeflateTileWriter w;
    // Block header: BFINAL=1, BTYPE=10 (dynamic).
    w.reset();
    w.add_bits(1u, 1u);
    w.add_bits(2u, 2u);
    // HLIT/HDIST/HCLEN fields (LSB-first), all to stream 0.
    w.reset();
    w.add_bits(258u - 257u, 5u); // HLIT = 258 -> field 1
    w.add_bits(1u - 1u, 5u);     // HDIST = 1 -> field 0
    w.add_bits(18u - 4u, 4u);    // HCLEN = 14 -> field 14
    // 18 precode lengths, in permutation order, 3 bits each (LSB-first),
    // one stream per entry.
    const uint32_t precode_lens[18] = {0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    for (uint32_t i = 0u; i < 18u; ++i) {
        w.add_bits(precode_lens[i], 3u);
        w.advance();
    }
    // Code-length items: 254 x sym8, 4 x sym9, 1 x sym1; one stream per item.
    w.reset();
    for (uint32_t i = 0u; i < 254u; ++i) {
        w.emit_huffman(2u, 2u); // precode code of sym 8 = 10
        w.advance();
    }
    for (uint32_t i = 0u; i < 4u; ++i) {
        w.emit_huffman(3u, 2u); // precode code of sym 9 = 11
        w.advance();
    }
    w.emit_huffman(0u, 1u); // precode code of sym 1 = 0
    w.advance();
    // Data symbols.
    w.reset();
    w.emit_literal(65u, 8u);        // 'A' (8-bit code 65)
    w.emit_match(511u, 9u, 0u, 0u,  // length symbol 257 (length 3, 0 extra)
                 0u, 1u, 0u, 0u);   // distance symbol 0 (distance 1, 0 extra)
    w.emit_eob(510u, 9u);           // EOB (9-bit code 510)
    w.advance();
    return w.finish();
}

}// namespace

bool run_huffman_tile_tests(GDeflateCodec &codec, Stream &stream, Device &device) noexcept {
    bool ok = true;

    // Fixed Huffman, literal-only tile covering all 256 byte values.
    {
        auto tile = build_fixed_literal_tile();
        std::vector<uint8_t> expected(256u);
        for (uint32_t i = 0u; i < 256u; ++i) { expected[i] = static_cast<uint8_t>(i); }
        ok &= run_tile_test(codec, stream, device, "fixed-Huffman literal", tile, expected);
    }

    // Fixed Huffman, tile with a length/distance match.
    {
        auto tile = build_fixed_match_tile();
        std::vector<uint8_t> expected{'A', 'B', 'C', 'D', 'A', 'B', 'C', 'D'};
        ok &= run_tile_test(codec, stream, device, "fixed-Huffman match", tile, expected);
    }

    // Dynamic Huffman tile.
    {
        auto tile = build_dynamic_tile();
        std::vector<uint8_t> expected{'A', 'A', 'A', 'A'};
        ok &= run_tile_test(codec, stream, device, "dynamic-Huffman", tile, expected);
    }

    return ok;
}

int main(int argc, char *argv[]) {
    auto executable = argc > 0 && argv != nullptr && argv[0] != nullptr ? argv[0] : "";
    if (argc <= 1 || argv == nullptr || argv[1] == nullptr || argv[1][0] == '\0') {
        LUISA_INFO("Usage: {} <backend> [size] [repeat]", executable);
        return 1;
    }

    uint32_t size = 1024u * 1024u; // 1 MiB default
    if (argc > 2 && argv[2] != nullptr) {
        size = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
        if (size == 0u) { size = 1024u * 1024u; }
    }

    uint32_t repeat = 1u; // benchmark iterations (averaged)
    if (argc > 3 && argv[3] != nullptr) {
        repeat = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
        if (repeat == 0u) { repeat = 1u; }
    }

    Context ctx{argv[0]};
    Device device = ctx.create_device(argv[1]);
    Stream stream = device.create_stream();

    // Generate random input data.
    std::mt19937 rng{12345u};
    std::uniform_int_distribution<int> dist{0, 255};
    std::vector<std::byte> host_input(size);
    for (auto &b : host_input) {
        b = static_cast<std::byte>(dist(rng));
    }

    ByteBuffer input = device.create_byte_buffer(size);
    stream << input.copy_from(host_input.data()) << synchronize();

    GDeflateCodec codec{device, stream};

    ByteBuffer compressed = codec.allocate_compressed(size);
    ByteBuffer decompressed = codec.allocate_uncompressed(size);

    // Warm up (also validates correctness on the first pass).
    uint32_t compressed_size = codec.compress(input, compressed, size);
    codec.decompress(compressed, decompressed, size);

    Clock clock;
    for (uint32_t i = 0u; i < repeat; ++i) {
        codec.compress(input, compressed, size);
    }
    auto compress_ms = clock.toc() / static_cast<double>(repeat);

    clock.tic();
    for (uint32_t i = 0u; i < repeat; ++i) {
        codec.decompress(compressed, decompressed, size);
    }
    auto decompress_ms = clock.toc() / static_cast<double>(repeat);

    std::vector<std::byte> host_output(size);
    stream << decompressed.copy_to(host_output.data()) << synchronize();

    bool ok = true;
    for (uint32_t i = 0u; i < size; ++i) {
        if (host_input[i] != host_output[i]) {
            LUISA_ERROR("Mismatch at byte {}: expected {}, got {}",
                        i, static_cast<int>(host_input[i]), static_cast<int>(host_output[i]));
            ok = false;
            break;
        }
    }

    if (ok) {
        double ratio = static_cast<double>(compressed_size) / static_cast<double>(size);
        LUISA_INFO("GDeflate round-trip: PASSED");
        LUISA_INFO("  input size:      {} bytes", size);
        LUISA_INFO("  compressed size: {} bytes ({:.2f}x)", compressed_size, ratio);
        LUISA_INFO("  compress time:   {} ms", compress_ms);
        LUISA_INFO("  decompress time: {} ms", decompress_ms);

        if (!run_huffman_tile_tests(codec, stream, device)) {
            LUISA_WARNING("GDeflate fixed/dynamic Huffman tile tests: FAILED");
            return 1;
        }
        LUISA_INFO("GDeflate fixed/dynamic Huffman tile tests: PASSED");
        return 0;
    }

    LUISA_WARNING("GDeflate round-trip: FAILED");
    return 1;
}
