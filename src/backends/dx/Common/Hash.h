#pragma once
#include <VEngineConfig.h>
#include <stdint.h>
class Hash
{
public:
	static constexpr size_t _FNV_offset_basis = 14695981039346656037ULL;
	static constexpr size_t _FNV_prime = 1099511628211ULL;
	static size_t Int32ArrayHash(
		const uint32_t* const _First,
		const uint32_t* const _End) noexcept { // accumulate range [_First, _First + _Count) into partial FNV-1a hash _Val
		size_t _Val = _FNV_offset_basis;
		for (const uint32_t* i = _First; i != _End; ++i) {
			_Val ^= static_cast<size_t>(*i);
			_Val *= _FNV_prime;
		}
		return _Val;
	}
	static size_t CharArrayHash(
		const char* _First,
		const size_t _Count) noexcept { // accumulate range [_First, _First + _Count) into partial FNV-1a hash _Val
		size_t _Val = _FNV_offset_basis;
		const uint32_t* _IntPtrEnd;
		{
			const uint32_t* _IntPtr = (const uint32_t*)_First;
			_IntPtrEnd = _IntPtr + (_Count / sizeof(uint32_t));
			for (; _IntPtr != _IntPtrEnd; ++_IntPtr) {
				_Val ^= static_cast<size_t>(*_IntPtr);
				_Val *= _FNV_prime;
			}
		}
		const char* _End = _First + _Count;
		for (const char* start = (const char*)_IntPtrEnd; start != _End; ++start)
		{
			_Val ^= static_cast<size_t>(*start);
			_Val *= _FNV_prime;
		}
		return _Val;
	}
};
namespace vengine
{
	template <typename K>
	struct hash
	{
		inline size_t operator()(K const& value) const noexcept
		{
			return Hash::CharArrayHash((char const*)&value, sizeof(K));
		}
	};
	inline static size_t GetIntegerHash(size_t a)
	{
		a = (a + 0xfd7046c5) + (a << 3);
		a = (a + 0xfd7046c5) + (a >> 3);
		a = (a ^ 0xb55a4f09) ^ (a << 16);
		a = (a ^ 0xb55a4f09) ^ (a >> 16);
		return a;
	}
	template <>
	struct hash<uint32_t>
	{
		inline size_t operator()(uint32_t value) const noexcept
		{
			return GetIntegerHash(value);
		}
	};

	template <>
	struct hash<int32_t>
	{
		inline size_t operator()(int32_t value) const noexcept
		{
			return GetIntegerHash(value);
		}
	};
	template <>
	struct hash<size_t>
	{
		inline size_t operator()(size_t value) const noexcept
		{
			return GetIntegerHash(value);
		}
	};

	template <>
	struct hash<int64_t>
	{
		inline size_t operator()(int64_t value) const noexcept
		{
			return GetIntegerHash(value);
		}
	};
}