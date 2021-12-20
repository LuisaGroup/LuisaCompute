#pragma once

#include <vstl/config.h>
#include <cstdint>
#include <utility>
#include <tuple>

#include <core/hash.h>

[[nodiscard]] inline uint64_t vstd_xxhash_gethash(const void *data, size_t size, uint64_t seed = 19980810u) noexcept {
    return luisa::detail::xxh3_hash64(data, size, seed);
}

[[nodiscard]] inline uint64_t vstd_xxhash_gethash_small(const void *data, size_t size, uint64_t seed = 19980810u) noexcept {
    return vstd_xxhash_gethash(data, size, seed);
}


//VENGINE_C_FUNC_COMMON size_t vstd_xxhash_gethash(void const* ptr, size_t sz);
//VENGINE_C_FUNC_COMMON size_t vstd_xxhash_gethash_seed(void const* ptr, size_t sz, size_t seed);
//Size must less than 32 in x64
//VENGINE_C_FUNC_COMMON size_t vstd_xxhash_gethash_small(void const* ptr, size_t sz);
//VENGINE_C_FUNC_COMMON size_t vstd_xxhash_gethash_small_seed(void const* ptr, size_t sz, size_t seed);

namespace vstd {
template<typename K>
struct hash;
}

class Hash {
public:
	static constexpr size_t FNV_offset_basis = 14695981039346656037ULL;
	static constexpr size_t FNV_prime = 1099511628211ULL;
	static size_t GetNextHash(
		size_t curHash,
		size_t lastHash = FNV_offset_basis) {
		return (lastHash ^ curHash) * FNV_prime;
	}
	static size_t CharArrayHash(
		const void* First,
		const size_t Count) noexcept {// accumulate range [_First, First + Count) into partial FNV-1a hash Val
		return vstd_xxhash_gethash(First, Count);
	}
	template<typename... T>
	static size_t MultipleHash(T const&... values) {
		auto func = [](auto&& value) -> size_t {
			vstd::hash<std::remove_cvref_t<decltype(value)>> hs;
			return hs(value);
		};
		auto results = {func(values)...};
		size_t initHash = FNV_offset_basis;
		for (auto&& i : results) {
			initHash = GetNextHash(i, initHash);
		}
		return initHash;
	}
};
namespace vstd {
template<typename K>
struct hash {
	size_t operator()(K const& value) const noexcept {
		if constexpr (sizeof(K) < (sizeof(size_t) / 2)) {
			return vstd_xxhash_gethash_small(&value, sizeof(K));
		} else {
			return Hash::CharArrayHash(&value, sizeof(K));
		}
	}
};
template<>
struct hash<int8_t> {
	size_t operator()(int8_t const& value) const noexcept {
		hash<uint64_t> hs;
		return hs(value);
	};
};
template<>
struct hash<uint8_t> {
	size_t operator()(uint8_t const& value) const noexcept {
		hash<uint64_t> hs;
		return hs(value);
	};
};
template<>
struct hash<int16_t> {
	size_t operator()(int16_t const& value) const noexcept {
		hash<uint64_t> hs;
		return hs(value);
	};
};
template<>
struct hash<uint16_t> {
	size_t operator()(uint16_t const& value) const noexcept {
		hash<uint64_t> hs;
		return hs(value);
	};
};
template<>
struct hash<int32_t> {
	size_t operator()(int32_t const& value) const noexcept {
		hash<uint64_t> hs;
		return hs(value);
	};
};
template<>
struct hash<uint32_t> {
	size_t operator()(uint32_t const& value) const noexcept {
		hash<uint64_t> hs;
		return hs(value);
	};
};
template<typename A, typename B>
struct hash<std::pair<A, B>> {
	size_t operator()(std::pair<A, B> const& v) const noexcept {
		hash<A> hs;
		hash<B> hs1;
		return Hash::GetNextHash(hs(v.first), hs1(v.second));
	}
};
template<typename... T>
struct hash<std::tuple<T...>> {
	size_t operator()(std::tuple<T...> const& tp) const noexcept {
        return []<size_t... i>(const auto &tp, std::index_sequence<i...>) noexcept {
            return Hash::MultipleHash(std::get<i>(tp)...);
        }(tp, std::index_sequence_for<T...>{});
	}
};

}// namespace vstd
