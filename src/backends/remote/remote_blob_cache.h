#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <unordered_map>

#include <luisa/core/stl/memory.h>
#include <luisa/core/stl/string.h>
#include <luisa/core/stl/vector.h>

#include "remote_protocol.h"

namespace luisa::compute::remote {

constexpr size_t blob_digest_size = 32u;

struct BlobKey {
    std::array<std::byte, blob_digest_size> digest{};
    uint64_t size{};

    [[nodiscard]] bool operator==(const BlobKey &) const noexcept = default;
};

struct BlobKeyHash {
    [[nodiscard]] size_t operator()(const BlobKey &key) const noexcept;
};

[[nodiscard]] BlobKey compute_blob_key(
    luisa::span<const std::byte> bytes) noexcept;

void write_blob_key(Writer &writer, const BlobKey &key) noexcept;
[[nodiscard]] bool read_blob_key(Reader &reader, BlobKey &key) noexcept;

struct BlobCacheStats {
    uint64_t hits{};
    uint64_t misses{};
    uint64_t stores{};
    uint64_t evictions{};
    uint64_t uploaded_bytes{};
    uint64_t resident_bytes{};
    uint64_t resident_entries{};
};

enum class BlobCacheError {
    NONE,
    DIGEST_MISMATCH,
    COLLISION,
    CAPACITY,
};

class BlobCache {

public:
    using Blob = luisa::vector<std::byte>;
    using BlobPtr = luisa::shared_ptr<const Blob>;

private:
    struct Entry {
        BlobPtr blob;
        std::list<BlobKey>::iterator lru_iterator;
    };

    uint64_t _capacity_bytes{};
    mutable std::mutex _mutex;
    std::list<BlobKey> _lru;
    std::unordered_map<BlobKey, Entry, BlobKeyHash> _entries;
    BlobCacheStats _stats;

private:
    void _touch(Entry &entry) noexcept;
    [[nodiscard]] bool _make_space(uint64_t size) noexcept;

public:
    explicit BlobCache(uint64_t capacity_bytes) noexcept;

    [[nodiscard]] uint64_t capacity_bytes() const noexcept {
        return _capacity_bytes;
    }

    [[nodiscard]] BlobPtr find(const BlobKey &key) noexcept;

    [[nodiscard]] BlobPtr publish(
        const BlobKey &key,
        luisa::span<const std::byte> bytes,
        BlobCacheError &cache_error,
        luisa::string &error) noexcept;

    [[nodiscard]] BlobCacheStats stats() const noexcept;
};

}// namespace luisa::compute::remote
