#include <Common/vstring.h>
#include <Common/Pool.h>
#include <mutex>
#include <Common/Runnable.h>
#include <Common/Memory.h>
#include <Common/vector.h>
#include <Common/DynamicDLL.h>
#include <Common/MetaLib.h>
//#include "BinaryLinkedAllocator.h"
#include <Common/LinkedList.h>
#include <string.h>
#include <Common/VAllocator.h>
#include <mimalloc.h>
namespace v_mimalloc {
funcPtr_t<void*(size_t)> mallocFunc = nullptr;
funcPtr_t<void(void*)> freeFunc = nullptr;
static std::atomic_bool memoryInitialized = false;
static StackObject<DynamicDLL> vengine_malloc_dll;
}// namespace v_mimalloc

void vengine_init_malloc() {
	using namespace v_mimalloc;
	if (memoryInitialized.exchange(true)) return;
	vengine_malloc_dll.New("mimalloc-override.dll"_sv);
	vengine_malloc_dll->GetDLLFunc(mallocFunc, "mi_malloc"_sv);
	vengine_malloc_dll->GetDLLFunc(freeFunc, "mi_free"_sv);
}
void vengine_init_malloc_path(
	char const* path) {
	using namespace v_mimalloc;
	if (memoryInitialized.exchange(true)) return;
	vengine_malloc_dll.New(path);
	vengine_malloc_dll->GetDLLFunc(mallocFunc, "mi_malloc"_sv);
	vengine_malloc_dll->GetDLLFunc(freeFunc, "mi_free"_sv);
}
void vengine_init_malloc_custom(
	funcPtr_t<void*(size_t)> mallocFunc,
	funcPtr_t<void(void*)> freeFunc) {
	using namespace v_mimalloc;
	if (memoryInitialized.exchange(true)) return;
	mallocFunc = mallocFunc;
	freeFunc = freeFunc;
}
void* vengine_default_malloc(size_t sz) {
	return malloc(sz);
}
void vengine_default_free(void* ptr) {
	free(ptr);
}

void* vengine_malloc(size_t size) {
	using namespace v_mimalloc;
	return mi_malloc(size);
}
void vengine_free(void* ptr) {
	using namespace v_mimalloc;
	mi_free(ptr);
}
namespace vengine {
void* string::string_malloc(size_t sz) {
	if (sz <= PLACEHOLDERSIZE) {
		return &localStorage;
	}
	return vengine_malloc(sz);
}
void string::string_free(void* freeMem) {
	if (freeMem != reinterpret_cast<void*>(&localStorage))
		vengine_free(freeMem);
}
void* wstring::wstring_malloc(size_t sz) {
	if (sz <= PLACEHOLDERSIZE) {
		return &localStorage;
	}
	return vengine_malloc(sz);
}
void wstring::wstring_free(void* freeMem) {
	if (freeMem != reinterpret_cast<void*>(&localStorage))
		vengine_free(freeMem);
}

string::string() noexcept {
	ptr = nullptr;
	capacity = 0;
	lenSize = 0;
}
string::~string() noexcept {
	if (ptr) {
		string_free(ptr);
	}
}
string::string(char const* chr) noexcept {
	size_t size = strlen(chr);
	size_t newLenSize = size;
	size += 1;
	reserve(size);
	lenSize = newLenSize;
	memcpy(ptr, chr, size);
}
string::string(const char* chr, const char* chrEnd) noexcept {
	size_t size = chrEnd - chr;
	size_t newLenSize = size;
	size += 1;
	reserve(size);
	lenSize = newLenSize;
	memcpy(ptr, chr, newLenSize);
	ptr[lenSize] = 0;
}
string::string(string_view tempView) {
	auto size = tempView.size();
	size_t newLenSize = size;
	size += 1;
	reserve(size);
	lenSize = newLenSize;
	memcpy(ptr, tempView.c_str(), newLenSize);
	ptr[lenSize] = 0;
}
void string::push_back_all(char const* c, size_t newStrLen) noexcept {
	size_t newCapacity = lenSize + newStrLen + 1;
	reserve(newCapacity);
	memcpy(ptr + lenSize, c, newStrLen);
	lenSize = newCapacity - 1;
	ptr[lenSize] = 0;
}
void string::reserve(size_t targetCapacity) noexcept {
	if (capacity >= targetCapacity) return;
	char* newPtr = (char*)string_malloc(targetCapacity);
	if (ptr) {
		memcpy(newPtr, ptr, lenSize + 1);
		string_free(ptr);
	}
	ptr = newPtr;
	capacity = targetCapacity;
}
string::string(const string& data) noexcept {
	if (data.ptr) {
		reserve(data.capacity);
		lenSize = data.lenSize;
		memcpy(ptr, data.ptr, lenSize);
		ptr[lenSize] = 0;
	} else {
		ptr = nullptr;
		capacity = 0;
		lenSize = 0;
	}
}
string::string(string&& data) noexcept {
	if (data.ptr == reinterpret_cast<void*>(&data.localStorage)) {
		ptr = reinterpret_cast<char*>(&localStorage);
		localStorage = data.localStorage;
	} else {
		ptr = data.ptr;
	}
	lenSize = data.lenSize;
	capacity = data.capacity;
	data.ptr = nullptr;
	data.lenSize = 0;
	data.capacity = 0;
}
void string::resize(size_t newSize) noexcept {
	reserve(newSize + 1);
	lenSize = newSize;
	ptr[lenSize] = 0;
}
string::string(size_t size, char c) noexcept {
	reserve(size + 1);
	lenSize = size;
	memset(ptr, c, lenSize);
	ptr[lenSize] = 0;
}
string& string::operator=(const string& data) noexcept {
	if (data.ptr) {
		reserve(data.capacity);
		lenSize = data.lenSize;
		memcpy(ptr, data.ptr, lenSize);
		ptr[lenSize] = 0;
	} else {
		if (ptr)
			string_free(ptr);
		ptr = nullptr;
		capacity = 0;
		lenSize = 0;
	}
	return *this;
}
string& string::operator=(string&& data) noexcept {
	this->~string();
	new (this) string(std::move(data));
	return *this;
}
string& string::operator=(const char* c) noexcept {
	size_t cSize = strlen(c);
	reserve(cSize + 1);
	lenSize = cSize;
	memcpy(ptr, c, cSize);
	ptr[lenSize] = 0;
	return *this;
}
string& string::operator=(char data) noexcept {
	lenSize = 1;
	reserve(2);
	ptr[0] = data;
	ptr[1] = 0;
	return *this;
}
string& string::operator=(string_view view) noexcept {
	size_t cSize = view.size();
	reserve(cSize + 1);
	lenSize = cSize;
	memcpy(ptr, view.c_str(), cSize);
	ptr[lenSize] = 0;
	return *this;
}
string& string::operator+=(const string& str) noexcept {
	if (str.ptr) {
		size_t newCapacity = lenSize + str.lenSize + 1;
		reserve(newCapacity);
		memcpy(ptr + lenSize, str.ptr, str.lenSize);
		lenSize = newCapacity - 1;
		ptr[lenSize] = 0;
	}
	return *this;
}
string& string::operator+=(const char* str) noexcept {
	size_t newStrLen = strlen(str);
	size_t newCapacity = lenSize + newStrLen + 1;
	reserve(newCapacity);
	memcpy(ptr + lenSize, str, newStrLen);
	lenSize = newCapacity - 1;
	ptr[lenSize] = 0;
	return *this;
}
string& string::operator+=(char str) noexcept {
	static const size_t newStrLen = 1;
	size_t newCapacity = lenSize + newStrLen + 1;
	reserve(newCapacity);
	ptr[lenSize] = str;
	lenSize = newCapacity - 1;
	ptr[lenSize] = 0;
	return *this;
}
string& string::operator+=(string_view str) noexcept {
	size_t newStrLen = str.size();
	size_t newCapacity = lenSize + newStrLen + 1;
	reserve(newCapacity);
	memcpy(ptr + lenSize, str.c_str(), newStrLen);
	lenSize = newCapacity - 1;
	ptr[lenSize] = 0;
	return *this;
}
string::string(const string& a, const string& b) noexcept {
	if (!a.ptr && !b.ptr) {
		ptr = nullptr;
		capacity = 0;
		lenSize = 0;
	} else {
		size_t newLenSize = a.lenSize + b.lenSize;
		reserve(newLenSize + 1);
		lenSize = newLenSize;
		if (a.ptr)
			memcpy(ptr, a.ptr, a.lenSize);
		if (b.ptr)
			memcpy(ptr + a.lenSize, b.ptr, b.lenSize);
		ptr[lenSize] = 0;
	}
}
string::string(string_view a, const string& b) noexcept {
	size_t newLen = a.size();
	size_t newLenSize = b.lenSize + newLen;
	reserve(newLenSize + 1);
	lenSize = newLenSize;
	memcpy(ptr, a.c_str(), newLen);
	if (b.ptr)
		memcpy(ptr + newLen, b.ptr, b.lenSize);
	ptr[lenSize] = 0;
}
string::string(const string& a, string_view b) noexcept {
	size_t newLen = b.size();
	size_t newLenSize = a.lenSize + newLen;
	reserve(newLenSize + 1);
	lenSize = newLenSize;
	if (a.ptr)
		memcpy(ptr, a.ptr, a.lenSize);
	memcpy(ptr + a.lenSize, b.c_str(), newLen);
	ptr[lenSize] = 0;
}
string::string(const string& a, const char* b) noexcept {
	size_t newLen = strlen(b);
	size_t newLenSize = a.lenSize + newLen;
	reserve(newLenSize + 1);
	lenSize = newLenSize;
	if (a.ptr)
		memcpy(ptr, a.ptr, a.lenSize);
	memcpy(ptr + a.lenSize, b, newLen);
	ptr[lenSize] = 0;
}
string::string(const char* a, const string& b) noexcept {
	size_t newLen = strlen(a);
	size_t newLenSize = b.lenSize + newLen;
	reserve(newLenSize + 1);
	lenSize = newLenSize;
	memcpy(ptr, a, newLen);
	if (b.ptr)
		memcpy(ptr + newLen, b.ptr, b.lenSize);
	ptr[lenSize] = 0;
}
string::string(const string& a, char b) noexcept {
	size_t newLenSize = a.lenSize + 1;
	reserve(newLenSize + 1);
	lenSize = newLenSize;
	if (a.ptr)
		memcpy(ptr, a.ptr, a.lenSize);
	ptr[a.lenSize] = b;
	ptr[newLenSize] = 0;
}
string::string(char a, const string& b) noexcept {
	size_t newLenSize = b.lenSize + 1;
	reserve(newLenSize + 1);
	lenSize = newLenSize;
	if (b.ptr)
		memcpy(ptr + 1, b.ptr, b.lenSize);
	ptr[0] = a;
	ptr[newLenSize] = 0;
}
char& string::operator[](size_t index) noexcept {
#if defined(DEBUG)
	if (index >= lenSize)
		throw "Out of Range Exception!";
#endif
	return ptr[index];
}
void string::erase(size_t index) noexcept {
#if defined(DEBUG)
	if (index >= lenSize)
		throw "Out of Range Exception!";
#endif
	memmove(ptr + index, ptr + index + 1, (lenSize - index));
	lenSize--;
}
char const& string::operator[](size_t index) const noexcept {
#if defined(DEBUG)
	if (index >= lenSize)
		throw "Out of Range Exception!";
#endif
	return ptr[index];
}
bool string::Equal(char const* str, size_t count) const noexcept {
	return memcmp(str, ptr, count) == 0;
}
std::ostream& operator<<(std::ostream& out, const string& obj) noexcept {
	if (!obj.ptr) return out;
	for (size_t i = 0; i < obj.lenSize; ++i) {
		out << obj.ptr[i];
	}
	return out;
}
std::istream& operator>>(std::istream& in, string& obj) noexcept {
	char cArr[1024];
	in.getline(cArr, 1024);
	obj = cArr;
	return in;
}
#pragma endregion
#pragma region wstring
wstring::wstring() noexcept {
	ptr = nullptr;
	capacity = 0;
	lenSize = 0;
}
wstring::~wstring() noexcept {
	if (ptr)
		wstring_free(ptr);
}
wstring::wstring(wchar_t const* chr) noexcept {
	size_t size = wstrLen(chr);
	size_t newLenSize = size;
	size += 1;
	reserve(size);
	lenSize = newLenSize;
	memcpy(ptr, chr, size * 2);
	ptr[lenSize] = 0;
}
wstring::wstring(const wchar_t* wchr, const wchar_t* wchrEnd) noexcept {
	size_t size = wchrEnd - wchr;
	size_t newLenSize = size;
	size += 1;
	reserve(size);
	lenSize = newLenSize;
	memcpy(ptr, wchr, newLenSize * 2);
	ptr[lenSize] = 0;
}
wstring::wstring(const char* chr) noexcept {
	size_t size = strlen(chr);
	size_t newLenSize = size;
	size += 1;
	reserve(size);
	lenSize = newLenSize;
	for (size_t i = 0; i < newLenSize; ++i) {
		ptr[i] = chr[i];
	}
	ptr[lenSize] = 0;
}
wstring::wstring(const char* chr, const char* wchrEnd) noexcept {
	size_t size = wchrEnd - chr;
	size_t newLenSize = size;
	size += 1;
	reserve(size);
	lenSize = newLenSize;
	for (size_t i = 0; i < newLenSize; ++i) {
		ptr[i] = chr[i];
	}
	ptr[lenSize] = 0;
}
wstring::wstring(wstring_view chunk) {
	size_t size = chunk.size();
	size_t newLenSize = size;
	size += 1;
	reserve(size);
	lenSize = newLenSize;
	memcpy(ptr, chunk.c_str(), newLenSize * 2);
	ptr[lenSize] = 0;
}
wstring::wstring(string_view chunk) {
	size_t size = chunk.size();
	size_t newLenSize = size;
	size += 1;
	reserve(size);
	lenSize = newLenSize;
	for (size_t i = 0; i < newLenSize; ++i) {
		ptr[i] = chunk.c_str()[i];
	}
	ptr[lenSize] = 0;
}
void wstring::reserve(size_t targetCapacity) noexcept {
	if (capacity >= targetCapacity) return;
	targetCapacity *= 2;
	wchar_t* newPtr = (wchar_t*)(char*)wstring_malloc(targetCapacity);
	targetCapacity /= 2;
	if (ptr) {
		memcpy(newPtr, ptr, (lenSize + 1) * 2);
		wstring_free(ptr);
	}
	ptr = newPtr;
	capacity = targetCapacity;
}
wstring::wstring(const wstring& data) noexcept {
	if (data.ptr) {
		reserve(data.capacity);
		lenSize = data.lenSize;
		memcpy(ptr, data.ptr, lenSize * 2);
		ptr[lenSize] = 0;
	} else {
		ptr = nullptr;
		capacity = 0;
		lenSize = 0;
	}
}
wstring::wstring(wstring&& data) noexcept {
	if (data.ptr == reinterpret_cast<void*>(&data.localStorage)) {
		ptr = reinterpret_cast<wchar_t*>(&localStorage);
		localStorage = data.localStorage;
	} else {
		ptr = data.ptr;
	}

	lenSize = data.lenSize;
	capacity = data.capacity;
	data.ptr = nullptr;
	data.lenSize = 0;
	data.capacity = 0;
}
void wstring::resize(size_t newSize) noexcept {
	reserve(newSize + 1);
	lenSize = newSize;
	ptr[lenSize] = 0;
}
wstring::wstring(size_t size, wchar_t c) noexcept {
	reserve(size + 1);
	lenSize = size;
	memset(ptr, c, lenSize * 2);
	ptr[lenSize] = 0;
}
wstring::wstring(string const& str) noexcept {
	reserve(str.getCapacity());
	lenSize = str.size();
	for (size_t i = 0; i < lenSize; ++i)
		ptr[i] = str[i];
	ptr[lenSize] = 0;
}
wstring& wstring::operator=(const wstring& data) noexcept {
	if (data.ptr) {
		reserve(data.capacity);
		lenSize = data.lenSize;
		memcpy(ptr, data.ptr, lenSize * 2);
		ptr[lenSize] = 0;
	} else {
		if (ptr)
			wstring_free(ptr);
		ptr = nullptr;
		capacity = 0;
		lenSize = 0;
	}
	return *this;
}
wstring& wstring::operator=(wstring&& data) noexcept {
	this->~wstring();
	new (this) wstring(std::move(data));
	return *this;
}
wstring& wstring::operator=(const wchar_t* c) noexcept {
	size_t cSize = wstrLen(c);
	reserve(cSize + 1);
	lenSize = cSize;
	memcpy(ptr, c, cSize * 2);
	ptr[lenSize] = 0;
	return *this;
}
wstring& wstring::operator=(wstring_view data) noexcept {
	size_t cSize = data.size();
	reserve(cSize + 1);
	lenSize = cSize;
	memcpy(ptr, data.c_str(), cSize * 2);
	ptr[lenSize] = 0;
	return *this;
}
wstring& wstring::operator=(wchar_t data) noexcept {
	lenSize = 1;
	reserve(2);
	ptr[0] = data;
	ptr[1] = 0;
	return *this;
}
wstring& wstring::operator+=(const wstring& str) noexcept {
	if (str.ptr) {
		size_t newCapacity = lenSize + str.lenSize + 1;
		reserve(newCapacity);
		memcpy(ptr + lenSize, str.ptr, str.lenSize * 2);
		lenSize = newCapacity - 1;
		ptr[lenSize] = 0;
	}
	return *this;
}
wstring& wstring::operator+=(const wchar_t* str) noexcept {
	size_t newStrLen = wstrLen(str);
	size_t newCapacity = lenSize + newStrLen + 1;
	reserve(newCapacity);
	memcpy(ptr + lenSize, str, newStrLen * 2);
	lenSize = newCapacity - 1;
	ptr[lenSize] = 0;
	return *this;
}
wstring& wstring::operator+=(wstring_view str) noexcept {
	size_t newStrLen = str.size();
	size_t newCapacity = lenSize + newStrLen + 1;
	reserve(newCapacity);
	memcpy(ptr + lenSize, str.c_str(), newStrLen * 2);
	lenSize = newCapacity - 1;
	ptr[lenSize] = 0;
	return *this;
}
wstring& wstring::operator+=(wchar_t str) noexcept {
	static const size_t newStrLen = 1;
	size_t newCapacity = lenSize + newStrLen + 1;
	reserve(newCapacity);
	ptr[lenSize] = str;
	lenSize = newCapacity - 1;
	ptr[lenSize] = 0;
	return *this;
}
wstring::wstring(const wstring& a, const wstring& b) noexcept {
	if (!a.ptr && !b.ptr) {
		ptr = nullptr;
		capacity = 0;
		lenSize = 0;
	} else {
		size_t newLenSize = a.lenSize + b.lenSize;
		reserve(newLenSize + 1);
		lenSize = newLenSize;
		if (a.ptr)
			memcpy(ptr, a.ptr, a.lenSize * 2);
		if (b.ptr)
			memcpy(ptr + a.lenSize, b.ptr, b.lenSize * 2);
		ptr[lenSize] = 0;
	}
}
wstring::wstring(const wstring& a, const wchar_t* b) noexcept {
	size_t newLen = wstrLen(b);
	size_t newLenSize = a.lenSize + newLen;
	reserve(newLenSize + 1);
	lenSize = newLenSize;
	if (a.ptr)
		memcpy(ptr, a.ptr, a.lenSize * 2);
	memcpy(ptr + a.lenSize, b, newLen * 2);
	ptr[lenSize] = 0;
}
wstring::wstring(const wchar_t* a, const wstring& b) noexcept {
	size_t newLen = wstrLen(a);
	size_t newLenSize = b.lenSize + newLen;
	reserve(newLenSize + 1);
	lenSize = newLenSize;
	memcpy(ptr, a, newLen * 2);
	if (b.ptr)
		memcpy(ptr + newLen, b.ptr, b.lenSize * 2);
	ptr[lenSize] = 0;
}
wstring::wstring(wstring_view a, const wstring& b) noexcept {
	size_t newLen = a.size();
	size_t newLenSize = b.lenSize + newLen;
	reserve(newLenSize + 1);
	lenSize = newLenSize;
	memcpy(ptr, a.c_str(), newLen * 2);
	if (b.ptr)
		memcpy(ptr + newLen, b.ptr, b.lenSize * 2);
	ptr[lenSize] = 0;
}
wstring::wstring(const wstring& a, wstring_view b) noexcept {
	size_t newLen = b.size();
	size_t newLenSize = a.lenSize + newLen;
	reserve(newLenSize + 1);
	lenSize = newLenSize;
	if (a.ptr)
		memcpy(ptr, a.ptr, a.lenSize * 2);
	memcpy(ptr + a.lenSize, b.c_str(), newLen * 2);
	ptr[lenSize] = 0;
}
wstring::wstring(const wstring& a, wchar_t b) noexcept {
	size_t newLenSize = a.lenSize + 1;
	reserve(newLenSize + 1);
	lenSize = newLenSize;
	if (a.ptr)
		memcpy(ptr, a.ptr, a.lenSize * 2);
	ptr[a.lenSize] = b;
	ptr[newLenSize] = 0;
}
wstring::wstring(wchar_t a, const wstring& b) noexcept {
	size_t newLenSize = b.lenSize + 1;
	reserve(newLenSize + 1);
	lenSize = newLenSize;
	if (b.ptr)
		memcpy(ptr + 1, b.ptr, b.lenSize * 2);
	ptr[0] = a;
	ptr[newLenSize] = 0;
}

wchar_t& wstring::operator[](size_t index) noexcept {
#if defined(DEBUG)
	if (index >= lenSize)
		throw "Out of Range Exception!";
#endif
	return ptr[index];
}
void wstring::erase(size_t index) noexcept {
#if defined(DEBUG)
	if (index >= lenSize)
		throw "Out of Range Exception!";
#endif
	memmove(ptr + index, ptr + index + 1, (lenSize - index) * 2);
	lenSize--;
}
wchar_t const& wstring::operator[](size_t index) const noexcept {
#if defined(DEBUG)
	if (index >= lenSize)
		throw "Out of Range Exception!";
#endif
	return ptr[index];
}
bool wstring::Equal(wchar_t const* str, size_t count) const noexcept {

	return memcmp(str, ptr, count * sizeof(wchar_t)) == 0;
}
string_view::string_view(vengine::string const& str) : data(str.data()), mSize(str.size()) {}
wstring_view::wstring_view(vengine::wstring const& str) : data(str.data()), mSize(str.size()) {}

#pragma endregion
}// namespace vengine
#include <Common/Log.h>
#include <Windows.h>
DynamicDLL::DynamicDLL(char const* name) {
	inst = reinterpret_cast<size_t>(LoadLibraryA(name));
	if (inst == 0) {
		VEngine_Log(
			{"Can not find DLL ",
			 name});
		VENGINE_EXIT;
	}
}
DynamicDLL::~DynamicDLL() {
	FreeLibrary(reinterpret_cast<HINSTANCE>(inst));
}

size_t DynamicDLL::GetFuncPtr(char const* name) {
	return reinterpret_cast<size_t>(GetProcAddress(reinterpret_cast<HINSTANCE>(inst), name));
}

vengine::string_view operator"" _sv(char const* str, size_t sz) {
	return vengine::string_view(str, sz);
}

vengine::wstring_view operator"" _sv(wchar_t const* str, size_t sz) {
	return vengine::wstring_view(str, sz);
}