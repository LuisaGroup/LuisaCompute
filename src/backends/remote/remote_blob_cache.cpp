#include "remote_blob_cache.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include <luisa/core/stl/hash.h>

namespace luisa::compute::remote {

namespace {

[[nodiscard]] constexpr uint32_t rotate_right(
    uint32_t value, uint32_t amount) noexcept {
    return (value >> amount) | (value << (32u - amount));
}

[[nodiscard]] std::array<std::byte, blob_digest_size> sha256(
    luisa::span<const std::byte> data) noexcept {

    static constexpr std::array<uint32_t, 64u> constants{
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

    std::array<uint32_t, 8u> state{
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

    auto process_block = [&](const std::byte *block) noexcept {
        std::array<uint32_t, 64u> words{};
        for (auto i = 0u; i < 16u; i++) {
            auto offset = i * 4u;
            words[i] =
                static_cast<uint32_t>(
                    std::to_integer<uint8_t>(block[offset]))
                    << 24u |
                static_cast<uint32_t>(
                    std::to_integer<uint8_t>(block[offset + 1u]))
                    << 16u |
                static_cast<uint32_t>(
                    std::to_integer<uint8_t>(block[offset + 2u]))
                    << 8u |
                static_cast<uint32_t>(
                    std::to_integer<uint8_t>(block[offset + 3u]));
        }
        for (auto i = 16u; i < 64u; i++) {
            auto x = words[i - 15u];
            auto y = words[i - 2u];
            auto s0 = rotate_right(x, 7u) ^
                      rotate_right(x, 18u) ^ (x >> 3u);
            auto s1 = rotate_right(y, 17u) ^
                      rotate_right(y, 19u) ^ (y >> 10u);
            words[i] = words[i - 16u] + s0 +
                       words[i - 7u] + s1;
        }

        auto a = state[0u];
        auto b = state[1u];
        auto c = state[2u];
        auto d = state[3u];
        auto e = state[4u];
        auto f = state[5u];
        auto g = state[6u];
        auto h = state[7u];
        for (auto i = 0u; i < 64u; i++) {
            auto s1 = rotate_right(e, 6u) ^
                      rotate_right(e, 11u) ^
                      rotate_right(e, 25u);
            auto choice = (e & f) ^ (~e & g);
            auto temp1 = h + s1 + choice +
                         constants[i] + words[i];
            auto s0 = rotate_right(a, 2u) ^
                      rotate_right(a, 13u) ^
                      rotate_right(a, 22u);
            auto majority = (a & b) ^ (a & c) ^ (b & c);
            auto temp2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state[0u] += a;
        state[1u] += b;
        state[2u] += c;
        state[3u] += d;
        state[4u] += e;
        state[5u] += f;
        state[6u] += g;
        state[7u] += h;
    };

    auto full_block_count = data.size() / 64u;
    for (auto i = 0u; i < full_block_count; i++) {
        process_block(data.data() + i * 64u);
    }

    std::array<std::byte, 128u> tail{};
    auto remaining = data.size() % 64u;
    if (remaining != 0u) {
        std::copy_n(
            data.data() + full_block_count * 64u,
            remaining, tail.data());
    }
    tail[remaining] = std::byte{0x80u};
    auto tail_size = remaining < 56u ? 64u : 128u;
    auto bit_count = static_cast<uint64_t>(data.size()) * 8u;
    for (auto i = 0u; i < 8u; i++) {
        tail[tail_size - 1u - i] =
            static_cast<std::byte>(bit_count >> (i * 8u));
    }
    process_block(tail.data());
    if (tail_size == 128u) {
        process_block(tail.data() + 64u);
    }

    std::array<std::byte, blob_digest_size> digest{};
    for (auto i = 0u; i < state.size(); i++) {
        digest[i * 4u] = static_cast<std::byte>(state[i] >> 24u);
        digest[i * 4u + 1u] =
            static_cast<std::byte>(state[i] >> 16u);
        digest[i * 4u + 2u] =
            static_cast<std::byte>(state[i] >> 8u);
        digest[i * 4u + 3u] = static_cast<std::byte>(state[i]);
    }
    return digest;
}

[[nodiscard]] bool equal_bytes(
    luisa::span<const std::byte> lhs,
    luisa::span<const std::byte> rhs) noexcept {
    return lhs.size() == rhs.size() &&
           (lhs.empty() ||
            std::memcmp(lhs.data(), rhs.data(), lhs.size()) == 0);
}

}// namespace

size_t BlobKeyHash::operator()(const BlobKey &key) const noexcept {
    auto digest_hash = luisa::hash64(
        key.digest.data(), key.digest.size(),
        0x4c4352505f424c4full);
    return static_cast<size_t>(luisa::hash64(
        &key.size, sizeof(key.size), digest_hash));
}

BlobKey compute_blob_key(luisa::span<const std::byte> bytes) noexcept {
    return BlobKey{
        .digest = sha256(bytes),
        .size = static_cast<uint64_t>(bytes.size())};
}

void write_blob_key(Writer &writer, const BlobKey &key) noexcept {
    writer.write_u64(key.size);
    writer.write_bytes(key.digest);
}

bool read_blob_key(Reader &reader, BlobKey &key) noexcept {
    key.size = reader.read_u64();
    auto digest = reader.read_bytes(blob_digest_size);
    if (!reader.ok()) { return false; }
    std::copy(digest.begin(), digest.end(), key.digest.begin());
    return true;
}

BlobCache::BlobCache(uint64_t capacity_bytes) noexcept
    : _capacity_bytes{capacity_bytes} {}

void BlobCache::_touch(Entry &entry) noexcept {
    _lru.splice(_lru.begin(), _lru, entry.lru_iterator);
    entry.lru_iterator = _lru.begin();
}

bool BlobCache::_make_space(uint64_t size) noexcept {
    if (size > _capacity_bytes) { return false; }
    while (_stats.resident_bytes > _capacity_bytes - size) {
        auto removed = false;
        for (auto iterator = _lru.end(); iterator != _lru.begin();) {
            --iterator;
            auto entry = _entries.find(*iterator);
            if (entry != _entries.end() &&
                entry->second.blob.use_count() == 1u) {
                _stats.resident_bytes -= entry->first.size;
                _stats.evictions++;
                _entries.erase(entry);
                _lru.erase(iterator);
                removed = true;
                break;
            }
        }
        if (!removed) { return false; }
    }
    return true;
}

BlobCache::BlobPtr BlobCache::find(const BlobKey &key) noexcept {
    std::scoped_lock lock{_mutex};
    if (auto iterator = _entries.find(key);
        iterator != _entries.end()) {
        _touch(iterator->second);
        _stats.hits++;
        return iterator->second.blob;
    }
    _stats.misses++;
    return {};
}

BlobCache::BlobPtr BlobCache::publish(
    const BlobKey &key,
    luisa::span<const std::byte> bytes,
    BlobCacheError &cache_error,
    luisa::string &error) noexcept {
    cache_error = BlobCacheError::NONE;
    if (key.size != bytes.size() || compute_blob_key(bytes) != key) {
        cache_error = BlobCacheError::DIGEST_MISMATCH;
        error = "Remote blob body does not match its declared digest and size.";
        return {};
    }
    auto storage = luisa::make_shared<Blob>();
    storage->assign(bytes.begin(), bytes.end());
    std::scoped_lock lock{_mutex};
    _stats.uploaded_bytes += key.size;
    if (auto iterator = _entries.find(key);
        iterator != _entries.end()) {
        if (!equal_bytes(*iterator->second.blob, bytes)) {
            cache_error = BlobCacheError::COLLISION;
            error = "Remote blob-cache digest collision detected.";
            return {};
        }
        _touch(iterator->second);
        return iterator->second.blob;
    }
    if (!_make_space(key.size)) {
        cache_error = BlobCacheError::CAPACITY;
        error = "Remote blob cache has no evictable capacity for this upload.";
        return {};
    }
    _lru.emplace_front(key);
    BlobPtr blob = storage;
    _entries.emplace(
        key, Entry{.blob = blob, .lru_iterator = _lru.begin()});
    _stats.stores++;
    _stats.resident_bytes += key.size;
    return blob;
}

BlobCacheStats BlobCache::stats() const noexcept {
    std::scoped_lock lock{_mutex};
    auto stats = _stats;
    stats.resident_entries = _entries.size();
    return stats;
}

}// namespace luisa::compute::remote
