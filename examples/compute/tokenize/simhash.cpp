#include "simhash.h"
#include <luisa/vstl/md5.h>
#include <luisa/vstl/vstring.h>
#include <luisa/core/logging.h>
#include <luisa/core/platform.h>
#include <luisa/core/fiber.h>
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/command_list.h>

using namespace luisa;
using namespace luisa::compute;

namespace tokenize {

uint64_t SimHash::hash_token(luisa::string_view token) {
    vstd::MD5 md5{luisa::string{token}};
    auto str = md5.to_string(false);
    uint64_t val = 0;
    for (size_t i = 0; i < 16 && i < str.size(); ++i) {
        char c = str[i];
        int digit = 0;
        if (c >= '0' && c <= '9') digit = c - '0';
        else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
        val = (val << 4) | static_cast<uint64_t>(digit);
    }
    return val;
}

SimHash::SimHash(luisa::string_view text, int hashbits)
    : _hashbits(hashbits), _value(compute(text)) {}

uint64_t SimHash::compute(luisa::string_view text) const {
    luisa::vector<int> v(static_cast<size_t>(_hashbits), 0);
    luisa::unordered_set<luisa::string> tokens;
    size_t start = 0;
    for (size_t i = 0; i <= text.size(); ++i) {
        if (i == text.size() || text[i] == ' ') {
            if (i > start) tokens.emplace(text.substr(start, i - start));
            start = i + 1;
        }
    }
    if (!tokens.empty()) {
        luisa::vector<luisa::string> token_vec(tokens.begin(), tokens.end());
        luisa::vector<luisa::vector<int>> locals;
        luisa::fiber::mutex mtx;
        luisa::fiber::parallel(static_cast<uint32_t>(token_vec.size()), [&](uint32_t begin, uint32_t end) noexcept {
            luisa::vector<int> local_v(static_cast<size_t>(_hashbits), 0);
            for (uint32_t idx = begin; idx < end; ++idx) {
                uint64_t h = hash_token(token_vec[idx]);
                for (int i = 0; i < _hashbits; ++i) {
                    if ((h >> i) & 1ULL) ++local_v[i];
                    else --local_v[i];
                }
            }
            {
                luisa::fiber::lock lck(mtx);
                locals.push_back(std::move(local_v));
            }
        });
        for (const auto &local_v : locals) {
            for (size_t i = 0; i < v.size(); ++i) {
                v[i] += local_v[i];
            }
        }
    }
    uint64_t result = 0;
    for (int i = 0; i < _hashbits; ++i) {
        if (v[i] > 0) result |= (1ULL << i);
    }
    return result;
}

uint64_t SimHash::gpu_compute_from_hashes(Device &device, Stream &stream, const luisa::vector<uint64_t> &token_hashes, int hashbits) {
    if (!device || token_hashes.empty()) {
        // CPU fallback
        luisa::vector<int> v(static_cast<size_t>(hashbits), 0);
        luisa::vector<luisa::vector<int>> locals;
        luisa::fiber::mutex mtx;
        luisa::fiber::parallel(static_cast<uint32_t>(token_hashes.size()), [&](uint32_t begin, uint32_t end) noexcept {
            luisa::vector<int> local_v(static_cast<size_t>(hashbits), 0);
            for (uint32_t i = begin; i < end; ++i) {
                uint64_t h = token_hashes[i];
                for (int j = 0; j < hashbits; ++j) {
                    if ((h >> j) & 1ULL) ++local_v[j];
                    else --local_v[j];
                }
            }
            {
                luisa::fiber::lock lck(mtx);
                locals.push_back(std::move(local_v));
            }
        });
        for (const auto &local_v : locals) {
            for (size_t i = 0; i < v.size(); ++i) {
                v[i] += local_v[i];
            }
        }
        uint64_t result = 0;
        for (int i = 0; i < hashbits; ++i) {
            if (v[i] > 0) result |= (1ULL << i);
        }
        return result;
    }

    Kernel1D simhash_compute_kernel = [](BufferVar<ulong> token_hashes,
                                          BufferVar<int> v_out,
                                          Var<int> num_tokens,
                                          Var<int> hashbits) noexcept {
        $ idx = dispatch_x();
        $if (idx >= num_tokens) {
            return;
        };
        Var<ulong> h = token_hashes.read(idx);
        for (auto i : dynamic_range(hashbits)) {
            $if (((h >> i) & 1ull) != 0ull) {
                v_out.atomic(i).fetch_add(1);
            } $else {
                v_out.atomic(i).fetch_sub(1);
            };
        }
    };

    auto compute_shader = device.compile(simhash_compute_kernel);

    int n = static_cast<int>(token_hashes.size());
    auto gpu_hashes = device.create_buffer<ulong>(token_hashes.size());
    auto gpu_v = device.create_buffer<int>(hashbits);
    luisa::vector<int> zero_v(static_cast<size_t>(hashbits), 0);

    CommandList cmdlist = CommandList::create();
    cmdlist << gpu_hashes.copy_from(luisa::span{token_hashes.data(), token_hashes.size()})
            << gpu_v.copy_from(luisa::span{zero_v.data(), zero_v.size()})
            << compute_shader(gpu_hashes, gpu_v, n, hashbits).dispatch(n);
    stream << cmdlist.commit();

    luisa::vector<int> v(static_cast<size_t>(hashbits));
    stream << gpu_v.copy_to(luisa::span{v.data(), v.size()}) << synchronize();

    uint64_t result = 0;
    for (int i = 0; i < hashbits; ++i) {
        if (v[i] > 0) result |= (1ULL << i);
    }
    return result;
}

luisa::vector<int> SimHash::gpu_batch_distance(Device &device, Stream &stream, uint64_t query, const luisa::vector<uint64_t> &hashes, int hashbits) {
    if (!device || hashes.empty()) {
        luisa::vector<int> dists;
        dists.resize(hashes.size());
        luisa::fiber::parallel(static_cast<uint32_t>(hashes.size()), [&](uint32_t i) noexcept {
            uint64_t x = query ^ hashes[i];
            int d = 0;
            while (x) {
                d += static_cast<int>(x & 1ULL);
                x >>= 1;
            }
            dists[i] = d;
        });
        return dists;
    }

    Kernel1D simhash_distance_kernel = [](BufferVar<ulong> hashes,
                                          BufferVar<int> out_distances,
                                          Var<ulong> query,
                                          Var<int> num_hashes,
                                          Var<int> hashbits) noexcept {
        $ idx = dispatch_x();
        $if (idx >= num_hashes) {
            return;
        };
        Var<ulong> h = hashes.read(idx);
        Var<ulong> x = query ^ h;
        Var<uint> dist = popcount(x);
        out_distances.write(idx, cast<int>(dist));
    };

    auto distance_shader = device.compile(simhash_distance_kernel);

    int n = static_cast<int>(hashes.size());
    auto gpu_hashes = device.create_buffer<ulong>(hashes.size());
    auto gpu_dists = device.create_buffer<int>(hashes.size());

    CommandList cmdlist = CommandList::create();
    cmdlist << gpu_hashes.copy_from(luisa::span{hashes.data(), hashes.size()})
            << distance_shader(gpu_hashes, gpu_dists, query, n, hashbits).dispatch(n);
    stream << cmdlist.commit();

    luisa::vector<int> dists(static_cast<size_t>(n));
    stream << gpu_dists.copy_to(luisa::span{dists.data(), dists.size()}) << synchronize();
    return dists;
}

int SimHash::distance(const SimHash &other) const noexcept {
    uint64_t x = _value ^ other._value;
    int dist = 0;
    while (x) {
        dist += static_cast<int>(x & 1ULL);
        x >>= 1;
    }
    return dist;
}

bool SimHash::is_near_duplicate(const SimHash &other, int threshold) const noexcept {
    return distance(other) <= threshold;
}

SimHashLSH::SimHashLSH(int hashbits, int band_bits)
    : _hashbits(hashbits), _band_bits(band_bits), _num_bands(hashbits / band_bits) {}

void SimHashLSH::add(int doc_id, const SimHash &simhash) {
    _hashes[doc_id] = simhash;
    uint64_t mask = (1ULL << _band_bits) - 1;
    for (int band = 0; band < _num_bands; ++band) {
        uint64_t val = (simhash.value() >> (band * _band_bits)) & mask;
        uint64_t key = (static_cast<uint64_t>(band) << 32) | val;
        _buckets[key].insert(doc_id);
    }
}

luisa::unordered_set<int> SimHashLSH::candidates(const SimHash &simhash) const {
    luisa::unordered_set<int> result;
    uint64_t mask = (1ULL << _band_bits) - 1;
    for (int band = 0; band < _num_bands; ++band) {
        uint64_t val = (simhash.value() >> (band * _band_bits)) & mask;
        uint64_t key = (static_cast<uint64_t>(band) << 32) | val;
        auto it = _buckets.find(key);
        if (it != _buckets.end()) {
            for (int d : it->second) result.insert(d);
        }
    }
    return result;
}

void SimHashLSH::remove(int doc_id) {
    auto it = _hashes.find(doc_id);
    if (it == _hashes.end()) return;
    auto h = it->second;
    _hashes.erase(it);
    uint64_t mask = (1ULL << _band_bits) - 1;
    for (int band = 0; band < _num_bands; ++band) {
        uint64_t val = (h.value() >> (band * _band_bits)) & mask;
        uint64_t key = (static_cast<uint64_t>(band) << 32) | val;
        auto bit = _buckets.find(key);
        if (bit != _buckets.end()) {
            bit->second.erase(doc_id);
            if (bit->second.empty()) _buckets.erase(bit);
        }
    }
}

}// namespace tokenize
