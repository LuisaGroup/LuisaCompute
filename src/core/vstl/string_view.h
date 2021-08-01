#pragma once
#include <core/vstl/vstlconfig.h>
#include <stdint.h>
#include <core/vstl/MetaLib.h>
#include <iostream>
#include <string.h>
namespace vstd {
class string;
class wstring;
class VENGINE_DLL_COMMON string_view {
	friend std::ostream& operator<<(std::ostream& out, const string_view& obj) noexcept;
	char const* data;
	size_t mSize;

public:
	char const* c_str() const {
		return data;
	}
	size_t size() const {
		return mSize;
	}
	char const* begin() const {
		return data;
	}
	operator char const*() const{
		return data;
	}
	char const* end() const {
		return data + mSize;
	}
	void operator+(int64_t i) {
		mSize += i;
	}
	void operator-(int64_t i) {
		mSize -= i;
	}
	void operator++() {
		mSize++;
	}
	void operator--() {
		mSize--;
	}
	string_view() : data(nullptr), mSize(0) {}
	string_view(std::nullptr_t) : data(nullptr), mSize(0) {}
	string_view(char const* data) : data(data), mSize(0) {
		mSize = strlen(data);
	}

	string_view(char const* data, size_t mSize) : data(data), mSize(mSize) {}

	string_view(char const* data, char const* end) : data(data), mSize(end - data) {}
	string_view(vstd::string const& str);

	bool operator==(const string_view& chunk) const {
		if (mSize != chunk.mSize) return false;
		return memcmp(data, chunk.data, mSize) == 0;
	}
	bool operator==(char const* chunk) const {
		size_t s = strlen(chunk);
		if (mSize != s) return false;
		return memcmp(data, chunk, mSize) == 0;
	}
	bool operator==(char c) const {
		if (mSize != 1) return false;
		return *data == c;
	}
	bool operator!=(char c) const {
		return !operator==(c);
	}
	bool operator!=(char const* chunk) const {
		return !operator==(chunk);
	}
	bool operator!=(const string_view& chunk) const {
		return !operator==(chunk);
	}
};
class VENGINE_DLL_COMMON wstring_view {
	wchar_t const* data;
	size_t mSize;
	static size_t wstrlen(wchar_t const* ptr) {
		size_t s = 0;
		while (*ptr != 0) {
			ptr++;
			s++;
		}
		return s;
	}

public:
	wchar_t const* c_str() const {
		return data;
	}
	operator wchar_t const *() const {
		return data;
	}
	size_t size() const {
		return mSize;
	}
	wchar_t const* begin() const {
		return data;
	}
	wchar_t const* end() const {
		return data + mSize;
	}
	void operator+(int64_t i) {
		mSize += i;
	}
	void operator-(int64_t i) {
		mSize -= i;
	}
	void operator++() {
		mSize++;
	}
	void operator--() {
		mSize--;
	}
	wstring_view() : data(nullptr), mSize(0) {}
	wstring_view(std::nullptr_t) : data(nullptr), mSize(0) {}
	wstring_view(wchar_t const* data) : data(data), mSize(0) {
		mSize = wstrlen(data);
	}

	wstring_view(wchar_t const* data, size_t mSize) : data(data), mSize(mSize) {}

	wstring_view(wchar_t const* data, wchar_t const* end) : data(data), mSize(end - data) {}
	wstring_view(vstd::wstring const& str);

	bool operator==(const wstring_view& chunk) const {
		if (mSize != chunk.mSize) return false;
		return memcmp(data, chunk.data, mSize * sizeof(wchar_t)) == 0;
	}
	bool operator==(wchar_t const* chunk) const {
		size_t s = wstrlen(chunk);
		if (mSize != s) return false;
		return memcmp(data, chunk, mSize * sizeof(wchar_t)) == 0;
	}
	bool operator==(wchar_t c) const {
		if (mSize != 1) return false;
		return *data == c;
	}
	bool operator!=(wchar_t c) const {
		return !operator==(c);
	}
	bool operator!=(wchar_t const* chunk) const {
		return !operator==(chunk);
	}
	bool operator!=(const wstring_view& chunk) const {
		return !operator==(chunk);
	}
};

template<>
struct hash<string_view> {
	inline size_t operator()(const string_view& str) const noexcept {
		return Hash::CharArrayHash(str.c_str(), str.size());
	}
};
template<>
struct hash<wstring_view> {
	inline size_t operator()(const wstring_view& str) const noexcept {
		return Hash::CharArrayHash(reinterpret_cast<char const*>(str.c_str()), str.size() * 2);
	}
};
inline std::ostream& operator<<(std::ostream& out, const string_view& obj) noexcept {
	if (!obj.c_str()) return out;
	auto end = obj.c_str() + obj.size();
	for (auto i = obj.c_str(); i < end; ++i) {
		out << *i;
	}
	return out;
}
}// namespace vstd
