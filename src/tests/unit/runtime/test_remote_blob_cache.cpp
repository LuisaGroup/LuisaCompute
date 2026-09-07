// Tests for the remote backend content-addressed upload cache.

#include "ut/ut.hpp"

#include <cstring>

#include "remote_blob_cache.h"
#include "remote_command_codec.h"

using namespace luisa;
using namespace luisa::compute::remote;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

[[nodiscard]] vector<std::byte> bytes_of(string_view text) {
    vector<std::byte> bytes(text.size());
    if (!bytes.empty()) {
        std::memcpy(bytes.data(), text.data(), text.size());
    }
    return bytes;
}

[[nodiscard]] string digest_string(const BlobKey &key) {
    constexpr char hex[] = "0123456789abcdef";
    string result;
    result.resize(key.digest.size() * 2u);
    for (size_t i = 0u; i < key.digest.size(); i++) {
        auto value = std::to_integer<uint8_t>(key.digest[i]);
        result[i * 2u] = hex[value >> 4u];
        result[i * 2u + 1u] = hex[value & 0x0fu];
    }
    return result;
}

void test_sha256_and_wire_key() {
    auto empty = bytes_of("");
    auto abc = bytes_of("abc");
    auto empty_key = compute_blob_key(empty);
    auto abc_key = compute_blob_key(abc);
    expect(empty_key.size == 0u);
    expect(digest_string(empty_key) ==
           "e3b0c44298fc1c149afbf4c8996fb924"
           "27ae41e4649b934ca495991b7852b855");
    expect(abc_key.size == 3u);
    expect(digest_string(abc_key) ==
           "ba7816bf8f01cfea414140de5dae2223"
           "b00361a396177a9cb410ff61f20015ad");

    Writer writer;
    write_blob_key(writer, abc_key);
    Reader reader{writer.bytes()};
    BlobKey decoded;
    expect(read_blob_key(reader, decoded));
    expect(reader.finish());
    expect(decoded == abc_key);
}

void test_lru_and_pinning() {
    BlobCache cache{8u};
    auto first = bytes_of("aaaa");
    auto second = bytes_of("bbbbb");
    auto first_key = compute_blob_key(first);
    auto second_key = compute_blob_key(second);
    string error;
    BlobCacheError cache_error{};
    auto first_blob = cache.publish(
        first_key, first, cache_error, error);
    expect(first_blob != nullptr);
    expect(cache_error == BlobCacheError::NONE);

    auto pin = cache.find(first_key);
    expect(pin != nullptr);
    auto blocked = cache.publish(
        second_key, second, cache_error, error);
    expect(blocked == nullptr);
    expect(cache_error == BlobCacheError::CAPACITY);

    pin.reset();
    first_blob.reset();
    error.clear();
    auto second_blob = cache.publish(
        second_key, second, cache_error, error);
    expect(second_blob != nullptr);
    expect(cache.find(first_key) == nullptr);
    expect(cache.find(second_key) != nullptr);
    auto stats = cache.stats();
    expect(stats.evictions == 1u);
    expect(stats.resident_entries == 1u);
    expect(stats.resident_bytes == second.size());
}

void test_digest_mismatch_rejected() {
    BlobCache cache{64u};
    auto expected = bytes_of("expected");
    auto actual = bytes_of("tampered");
    auto key = compute_blob_key(expected);
    string error;
    BlobCacheError cache_error{};
    expect(cache.publish(key, actual, cache_error, error) == nullptr);
    expect(cache_error == BlobCacheError::DIGEST_MISMATCH);
    expect(!error.empty());
    expect(cache.stats().resident_entries == 0u);
}

void test_upload_plan_deduplicates_and_encodes_references() {
    auto first = bytes_of("same upload bytes");
    auto second = bytes_of("same upload bytes");
    vector<unique_ptr<compute::Command>> commands;
    commands.emplace_back(make_unique<compute::BufferUploadCommand>(
        make_resource_id(ResourceKind::BUFFER, 1u),
        0u, first.size(), first.data()));
    commands.emplace_back(make_unique<compute::BufferUploadCommand>(
        make_resource_id(ResourceKind::BUFFER, 2u),
        0u, second.size(), second.data()));
    auto plan = plan_upload_blobs(commands, 1u, 1024u);
    expect(static_cast<bool>(plan));
    expect(plan.blobs.size() == 1u);
    expect(plan.references.size() == 2u);
    expect(plan.find(0u) == 0u);
    expect(plan.find(1u) == 0u);

    auto encoded = encode_submission(
        make_resource_id(ResourceKind::STREAM, 3u),
        9u, commands, {}, &plan);
    expect(static_cast<bool>(encoded));
    Reader reader{encoded.payload};
    expect(reader.read_u64() ==
           make_resource_id(ResourceKind::STREAM, 3u));
    expect(reader.read_u64() == 9u);
    expect(reader.read_u64() == 2u);
    for (auto i = 0u; i < 2u; i++) {
        expect(static_cast<WireCommand>(reader.read_u16()) ==
               WireCommand::BUFFER_UPLOAD_CACHED);
        expect(reader.read_u64() ==
               make_resource_id(ResourceKind::BUFFER, i + 1u));
        expect(reader.read_u64() == 0u);
        expect(reader.read_u64() == first.size());
        expect(reader.read_u32() == 0u);
    }
    expect(reader.finish());
}

}// namespace

static auto test_remote_blob_cache_registration = [] {
    "remote_blob_cache_sha256_and_wire_key"_test =
        test_sha256_and_wire_key;
    "remote_blob_cache_lru_and_pinning"_test =
        test_lru_and_pinning;
    "remote_blob_cache_digest_mismatch"_test =
        test_digest_mismatch_rejected;
    "remote_blob_cache_upload_plan"_test =
        test_upload_plan_deduplicates_and_encodes_references;
    return 0;
}();

int main(int argc, char *argv[]) {
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
}
