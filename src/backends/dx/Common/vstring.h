#pragma once
#include <VEngineConfig.h>
#include <stdint.h>
#include <xhash>
#include <iostream>
#include <Common/Hash.h>
#include <Common/Memory.h>
#include <Common/string_view.h>
namespace vengine {
class VENGINE_DLL_COMMON string {
	friend VENGINE_DLL_COMMON std::ostream& operator<<(std::ostream& out, const string& obj) noexcept;
	friend VENGINE_DLL_COMMON std::istream& operator>>(std::istream& in, string& obj) noexcept;

private:
	char* ptr = nullptr;
	size_t lenSize = 0;
	size_t capacity = 0;
	static constexpr size_t PLACEHOLDERSIZE = 32;
	std::aligned_storage_t<PLACEHOLDERSIZE, 1> localStorage;
	bool Equal(char const* str, size_t count) const noexcept;
	void* string_malloc(size_t sz);
	void string_free(void* freeMem);
	
public:
	string(const string& a, const string& b) noexcept;
	string(string_view a, const string& b) noexcept;
	string(const string& a, string_view b) noexcept;
	string(const string& a, const char* b) noexcept;
	string(const char* a, const string& b) noexcept;
	string(const string& a, char b) noexcept;
	string(char a, const string& b) noexcept;
	size_t size() const noexcept { return lenSize; }
	size_t length() const noexcept { return lenSize; }
	size_t getCapacity() const noexcept { return capacity; }
	string() noexcept;
	string(const char* cstr) noexcept;
	string(const char* cstrBegin, const char* cstrEnd) noexcept;
	string(string_view tempView);
	string(const string& data) noexcept;
	string(string&& data) noexcept;
	string(size_t size, char c) noexcept;
	void clear() noexcept {
		if (ptr)
			ptr[0] = 0;
		lenSize = 0;
	}
	void push_back(char c) noexcept {
		(*this) += c;
	}
	bool empty() const noexcept {
		return lenSize == 0;
	}
	string& operator=(const string& data) noexcept;
	string& operator=(string&& data) noexcept;
	string& operator=(const char* data) noexcept;
	string& operator=(char data) noexcept;
	string& operator=(string_view view) noexcept;
	string& assign(const string& data) noexcept {
		return operator=(data);
	}
	string& assign(const char* data) noexcept {
		return operator=(data);
	}
	void push_back_all(char const* c, size_t newStrLen) noexcept;
	string& assign(char data) noexcept {
		return operator=(data);
	}
	void reserve(size_t targetCapacity) noexcept;
	char* data() const noexcept { return ptr; }
	char* begin() const noexcept { return ptr; }
	char* end() const noexcept { return ptr + lenSize; }
	void resize(size_t newSize) noexcept;
	char const* c_str() const noexcept { return ptr; }
	string operator+(const string& str) const noexcept {
		return string(*this, str);
	}
	string operator+(const char* str) const noexcept {
		return string(*this, str);
	}
	string operator+(string_view str) const noexcept {
		return string(*this, str);
	}
	string operator+(char str) const noexcept {
		return string(*this, str);
	}
	string& operator+=(const string& str) noexcept;
	string& operator+=(const char* str) noexcept;
	string& operator+=(char str) noexcept;
	string& operator+=(string_view str) noexcept;
	string& operator<<(string_view str) noexcept {
		return operator+=(str);
	}
	string& operator<<(const string& str) noexcept {
		return operator+=(str);
	}
	string& operator<<(const char* str) noexcept {
		return operator+=(str);
	}
	string& operator<<(char str) noexcept {
		return operator+=(str);
	}
	char& operator[](size_t index) noexcept;
	char const& operator[](size_t index) const noexcept;
	bool operator==(const string& str) const noexcept {
		if (str.lenSize != lenSize) return false;
		return Equal(str.data(), str.lenSize);
	}
	bool operator==(const char* str) const noexcept {
		auto sz = strlen(str);
		if (sz != lenSize) return false;
		return Equal(str, sz);
	}
	bool operator==(string_view v) const noexcept {
		if (v.size() != lenSize) return false;
		return Equal(v.c_str(), v.size());
	}
	bool operator!=(const string& str) const noexcept {
		return !operator==(str);
	}
	bool operator!=(const char* str) const noexcept {
		return !operator==(str);
	}
	void erase(size_t index) noexcept;
	~string() noexcept;
};
class VENGINE_DLL_COMMON wstring {
private:
	wchar_t* ptr = nullptr;
	size_t lenSize = 0;
	size_t capacity = 0;
	bool Equal(wchar_t const* str, size_t count) const noexcept;
	static size_t wstrLen(wchar_t const* ptr) {
		size_t sz = 0;
		while (ptr[sz] != 0) {
			sz++;
		}
		return sz;
	}
	static constexpr size_t PLACEHOLDERSIZE = 32;
	std::aligned_storage_t<PLACEHOLDERSIZE, 1> localStorage;
	void* wstring_malloc(size_t sz);
	void wstring_free(void* freeMem);

public:
	wstring(const wstring& a, const wstring& b) noexcept;
	wstring(const wstring& a, const wchar_t* b) noexcept;
	wstring(wstring_view a, const wstring& b) noexcept;
	wstring(const wstring& a, wstring_view b) noexcept;
	wstring(const wchar_t* a, const wstring& b) noexcept;
	wstring(const wstring& a, wchar_t b) noexcept;
	wstring(wchar_t a, const wstring& b) noexcept;
	size_t size() const noexcept { return lenSize; }
	size_t length() const noexcept { return lenSize; }
	size_t getCapacity() const noexcept { return capacity; }
	wstring() noexcept;
	wstring(const wchar_t* wchr) noexcept;
	wstring(const wchar_t* wchr, const wchar_t* wchrEnd) noexcept;
	wstring(const char* wchr) noexcept;
	wstring(const char* wchr, const char* wchrEnd) noexcept;
	wstring(wstring_view chunk);
	wstring(string_view chunk);
	wstring(const wstring& data) noexcept;
	wstring(wstring&& data) noexcept;
	wstring(size_t size, wchar_t c) noexcept;
	wstring(string const& str) noexcept;
	void clear() noexcept {
		if (ptr)
			ptr[0] = 0;
		lenSize = 0;
	}
	void push_back(wchar_t c) noexcept {
		(*this) += c;
	}
	bool empty() const noexcept {
		return lenSize == 0;
	}
	wstring& operator=(const wstring& data) noexcept;
	wstring& operator=(wstring&& data) noexcept;
	wstring& operator=(const wchar_t* data) noexcept;
	wstring& operator=(wstring_view data) noexcept;
	wstring& operator=(wchar_t data) noexcept;
	wstring& assign(const wstring& data) noexcept {
		return operator=(data);
	}
	wstring& assign(const wchar_t* data) noexcept {
		return operator=(data);
	}
	wstring& assign(wchar_t data) noexcept {
		return operator=(data);
	}
	wchar_t const* begin() const noexcept { return ptr; }
	wchar_t const* end() const noexcept { return ptr + lenSize; }
	void reserve(size_t targetCapacity) noexcept;
	wchar_t* data() const noexcept { return ptr; }
	void resize(size_t newSize) noexcept;
	wchar_t const* c_str() const noexcept { return ptr; }
	wstring operator+(const wstring& str) const noexcept {
		return wstring(*this, str);
	}
	wstring operator+(const wchar_t* str) const noexcept {
		return wstring(*this, str);
	}
	wstring operator+(wstring_view str) const noexcept {
		return wstring(*this, str);
	}
	wstring operator+(wchar_t str) const noexcept {
		return wstring(*this, str);
	}
	wstring& operator+=(const wstring& str) noexcept;
	wstring& operator+=(const wchar_t* str) noexcept;
	wstring& operator+=(wstring_view str) noexcept;
	wstring& operator+=(wchar_t str) noexcept;
	wstring& operator<<(const wstring& str) noexcept {
		return operator+=(str);
	}
	wstring& operator<<(const wchar_t* str) noexcept {
		return operator+=(str);
	}
	wstring& operator<<(wstring_view str) noexcept {
		return operator+=(str);
	}
	wstring& operator<<(wchar_t str) noexcept {
		return operator+=(str);
	}
	wchar_t& operator[](size_t index) noexcept;
	wchar_t const& operator[](size_t index) const noexcept;
	bool operator==(const wstring& str) const noexcept {
		if (str.lenSize != lenSize) return false;
		return Equal(str.data(), str.lenSize);
	}
	bool operator==(const wchar_t* str) const noexcept {
		auto sz = wstrLen(str);
		if (sz != lenSize) return false;
		return Equal(str, sz);
	}
	bool operator==(wstring_view v) const noexcept {
		if (v.size() != lenSize) return false;
		return Equal(v.c_str(), v.size());
	}
	bool operator!=(const string& str) const noexcept {
		return !operator==(str);
	}
	bool operator!=(const char* str) const noexcept {
		return !operator==(str);
	}
	void erase(size_t index) noexcept;
	~wstring() noexcept;
};

template<class _Elem, class _UTy>
_Elem* UIntegral_to_buff(_Elem* _RNext, _UTy _UVal) noexcept {// format _UVal into buffer *ending at* _RNext
	static_assert(std::is_unsigned_v<_UTy>, "_UTy must be unsigned");

#ifdef _WIN64
	auto _UVal_trunc = _UVal;
#else// ^^^ _WIN64 ^^^ // vvv !_WIN64 vvv

	constexpr bool _Big_uty = sizeof(_UTy) > 4;
	if _CONSTEXPR_IF (_Big_uty) {// For 64-bit numbers, work in chunks to avoid 64-bit divisions.
		while (_UVal > 0xFFFFFFFFU) {
			auto _UVal_chunk = static_cast<unsigned long>(_UVal % 1000000000);
			_UVal /= 1000000000;

			for (int32_t _Idx = 0; _Idx != 9; ++_Idx) {
				*--_RNext = static_cast<_Elem>('0' + _UVal_chunk % 10);
				_UVal_chunk /= 10;
			}
		}
	}

	auto _UVal_trunc = static_cast<unsigned long>(_UVal);
#endif// _WIN64

	do {
		*--_RNext = static_cast<_Elem>('0' + _UVal_trunc % 10);
		_UVal_trunc /= 10;
	} while (_UVal_trunc != 0);
	return _RNext;
}
template<class _Ty>
inline string IntegerToString(const _Ty _Val) noexcept {// convert _Val to string
	static_assert(std::is_integral_v<_Ty>, "_Ty must be integral");
	using _UTy = std::make_unsigned_t<_Ty>;
	char _Buff[21];// can hold -2^63 and 2^64 - 1, plus NUL
	char* const _Buff_end = std::end(_Buff);
	char* _RNext = _Buff_end;
	const auto _UVal = static_cast<_UTy>(_Val);
	if (_Val < 0) {
		_RNext = UIntegral_to_buff(_RNext, static_cast<_UTy>(0 - _UVal));
		*--_RNext = '-';
	} else {
		_RNext = UIntegral_to_buff(_RNext, _UVal);
	}

	return string(_RNext, _Buff_end);
}
template<class _Ty>
inline void IntegerToString(const _Ty _Val, string& str) noexcept {// convert _Val to string
	static_assert(std::is_integral_v<_Ty>, "_Ty must be integral");
	using _UTy = std::make_unsigned_t<_Ty>;
	char _Buff[21];// can hold -2^63 and 2^64 - 1, plus NUL
	char* const _Buff_end = std::end(_Buff);
	char* _RNext = _Buff_end;
	const auto _UVal = static_cast<_UTy>(_Val);
	if (_Val < 0) {
		_RNext = UIntegral_to_buff(_RNext, static_cast<_UTy>(0 - _UVal));
		*--_RNext = '-';
	} else {
		_RNext = UIntegral_to_buff(_RNext, _UVal);
	}
	str.push_back_all(_RNext, _Buff_end - _RNext);
}
inline string to_string(double _Val) noexcept {
	const auto _Len = static_cast<size_t>(_CSTD _scprintf("%f", _Val));
	string _Str(_Len, '\0');
	_CSTD sprintf_s(&_Str[0], _Len + 1, "%f", _Val);
	return _Str;
}
inline void to_string(double _Val, vengine::string& str) noexcept {
	const auto _Len = static_cast<size_t>(_CSTD _scprintf("%f", _Val));
	size_t oldSize = str.size();
	str.resize(oldSize + _Len);
	_CSTD sprintf_s(&str[oldSize], _Len + 1, "%f", _Val);
}
inline string to_string(float _Val) noexcept {
	return to_string((double)_Val);
}

inline string to_string(int32_t _Val) noexcept {
	return IntegerToString(_Val);
}
inline string to_string(uint32_t _Val) noexcept {
	return IntegerToString(_Val);
}
inline string to_string(int16_t _Val) noexcept {
	return IntegerToString(_Val);
}
inline string to_string(uint16_t _Val) noexcept {
	return IntegerToString(_Val);
}
inline string to_string(int8_t _Val) noexcept {
	return IntegerToString(_Val);
}
inline string to_string(uint8_t _Val) noexcept {
	return IntegerToString(_Val);
}
inline string to_string(int64_t _Val) noexcept {
	return IntegerToString(_Val);
}
inline string to_string(uint64_t _Val) noexcept {
	return IntegerToString(_Val);
}

inline void to_string(float _Val, vengine::string& str) noexcept {
	to_string((double)_Val, str);
}

inline void to_string(int32_t _Val, vengine::string& str) noexcept {
	IntegerToString(_Val, str);
}
inline void to_string(uint32_t _Val, vengine::string& str) noexcept {
	IntegerToString(_Val, str);
}
inline void to_string(int16_t _Val, vengine::string& str) noexcept {
	IntegerToString(_Val, str);
}
inline void to_string(uint16_t _Val, vengine::string& str) noexcept {
	IntegerToString(_Val, str);
}
inline void to_string(int8_t _Val, vengine::string& str) noexcept {
	IntegerToString(_Val, str);
}
inline void to_string(uint8_t _Val, vengine::string& str) noexcept {
	IntegerToString(_Val, str);
}
inline void to_string(int64_t _Val, vengine::string& str) noexcept {
	IntegerToString(_Val, str);
}
inline void to_string(uint64_t _Val, vengine::string& str) noexcept {
	IntegerToString(_Val, str);
}
template<class _Ty>
inline wstring IntegerToWString(const _Ty _Val) noexcept {// convert _Val to string
	static_assert(std::is_integral_v<_Ty>, "_Ty must be integral");
	using _UTy = std::make_unsigned_t<_Ty>;
	wchar_t _Buff[21];// can hold -2^63 and 2^64 - 1, plus NUL
	wchar_t* const _Buff_end = std::end(_Buff);
	wchar_t* _RNext = _Buff_end;
	const auto _UVal = static_cast<_UTy>(_Val);
	if (_Val < 0) {
		_RNext = UIntegral_to_buff(_RNext, static_cast<_UTy>(0 - _UVal));
		*--_RNext = '-';
	} else {
		_RNext = UIntegral_to_buff(_RNext, _UVal);
	}

	return wstring(_RNext, _Buff_end);
}

inline wstring to_wstring(double _Val) noexcept {// convert double to wstring
	const auto _Len = static_cast<size_t>(_CSTD _scwprintf(L"%f", _Val));
	wstring _Str(_Len, L'\0');
	_CSTD swprintf_s(&_Str[0], _Len + 1, L"%f", _Val);
	return _Str;
}
inline wstring to_wstring(float _Val) noexcept {
	return to_wstring((double)_Val);
}

inline wstring to_wstring(int32_t _Val) noexcept {
	return IntegerToWString(_Val);
}
inline wstring to_wstring(uint32_t _Val) noexcept {
	return IntegerToWString(_Val);
}
inline wstring to_wstring(int16_t _Val) noexcept {
	return IntegerToWString(_Val);
}
inline wstring to_wstring(uint16_t _Val) noexcept {
	return IntegerToWString(_Val);
}
inline wstring to_wstring(int8_t _Val) noexcept {
	return IntegerToWString(_Val);
}
inline wstring to_wstring(uint8_t _Val) noexcept {
	return IntegerToWString(_Val);
}
inline wstring to_wstring(int64_t _Val) noexcept {
	return IntegerToWString(_Val);
}
inline wstring to_wstring(uint64_t _Val) noexcept {
	return IntegerToWString(_Val);
}

}// namespace vengine
VENGINE_DLL_COMMON vengine::string_view operator""_sv(char const* str, size_t sz);

VENGINE_DLL_COMMON vengine::wstring_view operator"" _sv(wchar_t const* str, size_t sz);

inline vengine::string operator+(char c, const vengine::string& str) noexcept {
	return vengine::string(c, str);
}

inline vengine::string operator+(const char* c, const vengine::string& str) noexcept {
	return vengine::string(c, str);
}

inline vengine::wstring operator+(wchar_t c, const vengine::wstring& str) noexcept {
	return vengine::wstring(c, str);
}

inline vengine::wstring operator+(const wchar_t* c, const vengine::wstring& str) noexcept {
	return vengine::wstring(c, str);
}

#include <Common/Hash.h>
namespace vengine {
template<>
struct hash<vengine::string> {
	inline size_t operator()(const vengine::string& str) const noexcept {
		return Hash::CharArrayHash(str.c_str(), str.size());
	}
};
template<>
struct hash<vengine::wstring> {
	inline size_t operator()(const vengine::wstring& str) const noexcept {
		return Hash::CharArrayHash((const char*)str.c_str(), str.size() * 2);
	}
};
}// namespace vengine