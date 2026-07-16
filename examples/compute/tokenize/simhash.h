#pragma once

#include <luisa/core/stl.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <cstdint>

namespace tokenize {

class SimHash {
public:
    explicit SimHash(luisa::string_view text = "", int hashbits = 64);

    [[nodiscard]] static uint64_t hash_token(luisa::string_view token);
    [[nodiscard]] int distance(const SimHash &other) const noexcept;
    [[nodiscard]] bool is_near_duplicate(const SimHash &other, int threshold = 3) const noexcept;
    [[nodiscard]] uint64_t value() const noexcept { return _value; }

    // GPU accelerated compute from pre-hashed token hashes (bit-voting kernel)
    [[nodiscard]] static uint64_t gpu_compute_from_hashes(luisa::compute::Device &device,
                                                          luisa::compute::Stream &stream,
                                                          const luisa::vector<uint64_t> &token_hashes,
                                                          int hashbits = 64);

    // GPU batch Hamming distance: one thread per hash, popcount XOR
    [[nodiscard]] static luisa::vector<int> gpu_batch_distance(luisa::compute::Device &device,
                                                               luisa::compute::Stream &stream,
                                                               uint64_t query,
                                                               const luisa::vector<uint64_t> &hashes,
                                                               int hashbits = 64);

private:
    [[nodiscard]] uint64_t compute(luisa::string_view text) const;

    int _hashbits;
    uint64_t _value;
};

class SimHashLSH {
public:
    explicit SimHashLSH(int hashbits = 64, int band_bits = 4);

    void add(int doc_id, const SimHash &simhash);
    [[nodiscard]] luisa::unordered_set<int> candidates(const SimHash &simhash) const;
    void remove(int doc_id);

private:
    int _hashbits;
    int _band_bits;
    int _num_bands;
    luisa::unordered_map<uint64_t, luisa::unordered_set<int>> _buckets;
    luisa::unordered_map<int, SimHash> _hashes;
};

}// namespace tokenize
