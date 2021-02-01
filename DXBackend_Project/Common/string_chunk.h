#pragma once
#include <stdint.h>
#include "vstring.h"
#include "MetaLib.h"
#include <iostream>
class string_chunk
{
	friend std::ostream& operator << (std::ostream& out, const string_chunk& obj) noexcept;
	char const* data;
	size_t mSize;
public:
	constexpr char const* c_str() const
	{
		return data;
	}
	constexpr size_t size() const
	{
		return mSize;
	}
	constexpr char const* begin() const
	{
		return data;
	}
	constexpr char const* end() const
	{
		return data + mSize;
	}
	constexpr void operator+(int64_t i)
	{
		mSize += i;
	}
	constexpr void operator-(int64_t i)
	{
		mSize -= i;
	}
	constexpr void operator++()
	{
		mSize++;
	}
	constexpr void operator--()
	{
		mSize--;
	}
	constexpr string_chunk() : data(nullptr), mSize(0)
	{}
	constexpr string_chunk(std::nullptr_t) : data(nullptr), mSize(0)
	{}
	string_chunk(char const* data) : data(data), mSize(0)
	{
		mSize = strlen(data);
	}

	constexpr string_chunk(char const* data, size_t mSize) : data(data), mSize(mSize)
	{}

	constexpr string_chunk(char const* data, char const* end) : data(data), mSize(end - data)
	{}
	string_chunk(vengine::string const& str) : data(str.data()), mSize(str.size())
	{}

	constexpr bool operator==(const string_chunk& chunk) const
	{
		if (mSize != chunk.mSize) return false;
		return BinaryEqualTo_Size(data, chunk.data, mSize);
	}
	bool operator==(char const* chunk) const
	{
		size_t s = strlen(chunk);
		if (mSize != s) return false;
		return BinaryEqualTo_Size(data, chunk, mSize);
	}
	constexpr bool operator==(char c) const
	{
		if (mSize != 1) return false;
		return *data == c;
	}
	constexpr bool operator!=(char c) const
	{
		return !operator==(c);
	}
	bool operator!=(char const* chunk) const
	{
		return !operator==(chunk);
	}
	constexpr bool operator!=(const string_chunk& chunk) const
	{
		return !operator==(chunk);
	}
};

namespace vengine
{
	template <>
	struct hash<string_chunk>
	{
		inline size_t operator()(const string_chunk& str) const noexcept
		{
			return Hash::CharArrayHash(str.c_str(), str.size());
		}
	};
}
inline std::ostream& operator << (std::ostream& out, const string_chunk& obj) noexcept
{
	if (!obj.c_str()) return out;
	auto end = obj.c_str() + obj.size();
	for (auto i = obj.c_str(); i < end; ++i)
	{
		out << *i;
	}
	return out;
}