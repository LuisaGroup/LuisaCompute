#include "vstring.h"
#include "Pool.h"
#include <mutex>
#include "Runnable.h"
#include "Memory.h"
#include "vector.h"
#include "DynamicDLL.h"
#include "MetaLib.h"
//#include "BinaryLinkedAllocator.h"
#include "LinkedList.h"
namespace v_mimalloc {
funcPtr_t<void*(size_t)> Alloc::mallocFunc = nullptr;
funcPtr_t<void(void*)> Alloc::freeFunc = nullptr;
static bool memoryInitialized = false;
}// namespace v_mimalloc
namespace vengine {
void vengine_init_malloc() {
	using namespace v_mimalloc;
	static StackObject<DynamicDLL> vengine_malloc_dll;
	if (memoryInitialized) return;
	memoryInitialized = true;
	vengine_malloc_dll.New("mimalloc-override.dll");
	/*vengine_malloc_dll->GetDLLFuncFromExample(Alloc::mallocFunc, "mi_malloc");
	vengine_malloc_dll->GetDLLFuncFromExample(Alloc::freeFunc, "mi_free");*/
	vengine_malloc_dll->GetDLLFunc(Alloc::mallocFunc, "mi_malloc");
	vengine_malloc_dll->GetDLLFunc(Alloc::freeFunc, "mi_free");
}
void vengine_init_malloc(
	funcPtr_t<void*(size_t)> mallocFunc,
	funcPtr_t<void(void*)> freeFunc) {
	using namespace v_mimalloc;
	if (memoryInitialized) return;
	memoryInitialized = true;
	Alloc::mallocFunc = mallocFunc;
	Alloc::freeFunc = freeFunc;
}
string::string() noexcept {
	ptr = nullptr;
	capacity = 0;
	lenSize = 0;
}
string::~string() noexcept {
	if (ptr) {
		vengine_free(ptr);
	}
}
string::string(char const* chr) noexcept {
	size_t size = strlen(chr);
	uint64_t newLenSize = size;
	size += 1;
	reserve(size);
	lenSize = newLenSize;
	memcpy(ptr, chr, size);
}
string::string(const char* chr, const char* chrEnd) noexcept {
	size_t size = chrEnd - chr;
	uint64_t newLenSize = size;
	size += 1;
	reserve(size);
	lenSize = newLenSize;
	memcpy(ptr, chr, newLenSize);
	ptr[lenSize] = 0;
}
string::string(string_view tempView) {
	auto size = tempView.size();
	uint64_t newLenSize = size;
	size += 1;
	reserve(size);
	lenSize = newLenSize;
	memcpy(ptr, tempView.c_str(), newLenSize);
	ptr[lenSize] = 0;
}
void string::push_back_all(char const* c, uint64_t newStrLen) noexcept {
	uint64_t newCapacity = lenSize + newStrLen + 1;
	reserve(newCapacity);
	memcpy(ptr + lenSize, c, newStrLen);
	lenSize = newCapacity - 1;
	ptr[lenSize] = 0;
}
void string::reserve(uint64_t targetCapacity) noexcept {
	if (capacity >= targetCapacity) return;
	char* newPtr = (char*)vengine_malloc(targetCapacity);
	if (ptr) {
		memcpy(newPtr, ptr, lenSize + 1);
		vengine_free(ptr);
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
	ptr = data.ptr;
	lenSize = data.lenSize;
	capacity = data.capacity;
	data.ptr = nullptr;
	data.lenSize = 0;
	data.capacity = 0;
}
void string::resize(uint64_t newSize) noexcept {
	reserve(newSize + 1);
	lenSize = newSize;
	ptr[lenSize] = 0;
}
string::string(uint64_t size, char c) noexcept {
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
			vengine_free(ptr);
		ptr = nullptr;
		capacity = 0;
		lenSize = 0;
	}
	return *this;
}
string& string::operator=(string&& data) noexcept {
	if (ptr)
		vengine_free(ptr);
	ptr = data.ptr;
	lenSize = data.lenSize;
	capacity = data.capacity;
	data.ptr = nullptr;
	data.lenSize = 0;
	data.capacity = 0;
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
string& string::operator+=(const string& str) noexcept {
	if (str.ptr) {
		uint64_t newCapacity = lenSize + str.lenSize + 1;
		reserve(newCapacity);
		memcpy(ptr + lenSize, str.ptr, str.lenSize);
		lenSize = newCapacity - 1;
		ptr[lenSize] = 0;
	}
	return *this;
}
string& string::operator+=(const char* str) noexcept {
	uint64_t newStrLen = strlen(str);
	uint64_t newCapacity = lenSize + newStrLen + 1;
	reserve(newCapacity);
	memcpy(ptr + lenSize, str, newStrLen);
	lenSize = newCapacity - 1;
	ptr[lenSize] = 0;
	return *this;
}
string& string::operator+=(char str) noexcept {
	static const uint64_t newStrLen = 1;
	uint64_t newCapacity = lenSize + newStrLen + 1;
	reserve(newCapacity);
	ptr[lenSize] = str;
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
		uint64_t newLenSize = a.lenSize + b.lenSize;
		reserve(newLenSize + 1);
		lenSize = newLenSize;
		if (a.ptr)
			memcpy(ptr, a.ptr, a.lenSize);
		if (b.ptr)
			memcpy(ptr + a.lenSize, b.ptr, b.lenSize);
		ptr[lenSize] = 0;
	}
}
string::string(const string& a, const char* b) noexcept {
	size_t newLen = strlen(b);
	uint64_t newLenSize = a.lenSize + newLen;
	reserve(newLenSize + 1);
	lenSize = newLenSize;
	if (a.ptr)
		memcpy(ptr, a.ptr, a.lenSize);
	memcpy(ptr + a.lenSize, b, newLen);
	ptr[lenSize] = 0;
}
string::string(const char* a, const string& b) noexcept {
	size_t newLen = strlen(a);
	uint64_t newLenSize = b.lenSize + newLen;
	reserve(newLenSize + 1);
	lenSize = newLenSize;
	memcpy(ptr, a, newLen);
	if (b.ptr)
		memcpy(ptr + newLen, b.ptr, b.lenSize);
	ptr[lenSize] = 0;
}
string::string(const string& a, char b) noexcept {
	uint64_t newLenSize = a.lenSize + 1;
	reserve(newLenSize + 1);
	lenSize = newLenSize;
	if (a.ptr)
		memcpy(ptr, a.ptr, a.lenSize);
	ptr[a.lenSize] = b;
	ptr[newLenSize] = 0;
}
string::string(char a, const string& b) noexcept {
	uint64_t newLenSize = b.lenSize + 1;
	reserve(newLenSize + 1);
	lenSize = newLenSize;
	if (b.ptr)
		memcpy(ptr + 1, b.ptr, b.lenSize);
	ptr[0] = a;
	ptr[newLenSize] = 0;
}
char& string::operator[](uint64_t index) noexcept {
#if defined(DEBUG) || defined(_DEBUG)
	if (index >= lenSize)
		throw "Out of Range Exception!";
#endif
	return ptr[index];
}
void string::erase(uint64_t index) noexcept {
#if defined(DEBUG) || defined(_DEBUG)
	if (index >= lenSize)
		throw "Out of Range Exception!";
#endif
	memmove(ptr + index, ptr + index + 1, (lenSize - index));
	lenSize--;
}
char const& string::operator[](uint64_t index) const noexcept {
#if defined(DEBUG) || defined(_DEBUG)
	if (index >= lenSize)
		throw "Out of Range Exception!";
#endif
	return ptr[index];
}
bool string::Equal(char const* str, uint64_t count) const noexcept {
	uint64_t bit64Count = count / 8;
	uint64_t leftedCount = count - bit64Count * 8;
	uint64_t const* value = (uint64_t const*)str;
	uint64_t const* oriValue = (uint64_t const*)ptr;
	for (uint64_t i = 0; i < bit64Count; ++i) {
		if (value[i] != oriValue[i]) return false;
	}
	char const* c = (char const*)(value + bit64Count);
	char const* oriC = (char const*)(oriValue + bit64Count);
	for (uint64_t i = 0; i < leftedCount; ++i) {
		if (c[i] != oriC[i]) return false;
	}
	return true;
}
std::ostream& operator<<(std::ostream& out, const string& obj) noexcept {
	if (!obj.ptr) return out;
	for (uint64_t i = 0; i < obj.lenSize; ++i) {
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
		vengine_free(ptr);
}
wstring::wstring(wchar_t const* chr) noexcept {
	size_t size = wstrLen(chr);
	uint64_t newLenSize = size;
	size += 1;
	reserve(size);
	lenSize = newLenSize;
	memcpy(ptr, chr, size * 2);
	ptr[lenSize] = 0;
}
wstring::wstring(const wchar_t* wchr, const wchar_t* wchrEnd) noexcept {
	size_t size = wchrEnd - wchr;
	uint64_t newLenSize = size;
	size += 1;
	reserve(size);
	lenSize = newLenSize;
	memcpy(ptr, wchr, newLenSize * 2);
	ptr[lenSize] = 0;
}
wstring::wstring(const char* chr) noexcept {
	size_t size = strlen(chr);
	uint64_t newLenSize = size;
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
	uint64_t newLenSize = size;
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
	uint64_t newLenSize = size;
	size += 1;
	reserve(size);
	lenSize = newLenSize;
	memcpy(ptr, chunk.c_str(), newLenSize * 2);
	ptr[lenSize] = 0;
}
wstring::wstring(string_view chunk) {
	size_t size = chunk.size();
	uint64_t newLenSize = size;
	size += 1;
	reserve(size);
	lenSize = newLenSize;
	for (size_t i = 0; i < newLenSize; ++i) {
		ptr[i] = chunk.c_str()[i];
	}
	ptr[lenSize] = 0;
}
void wstring::reserve(uint64_t targetCapacity) noexcept {
	if (capacity >= targetCapacity) return;
	targetCapacity *= 2;
	wchar_t* newPtr = (wchar_t*)(char*)vengine_malloc(targetCapacity);
	targetCapacity /= 2;
	if (ptr) {
		memcpy(newPtr, ptr, (lenSize + 1) * 2);
		vengine_free(ptr);
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
	ptr = data.ptr;
	lenSize = data.lenSize;
	capacity = data.capacity;
	data.ptr = nullptr;
	data.lenSize = 0;
	data.capacity = 0;
}
void wstring::resize(uint64_t newSize) noexcept {
	reserve(newSize + 1);
	lenSize = newSize;
	ptr[lenSize] = 0;
}
wstring::wstring(uint64_t size, wchar_t c) noexcept {
	reserve(size + 1);
	lenSize = size;
	memset(ptr, c, lenSize * 2);
	ptr[lenSize] = 0;
}
wstring::wstring(string const& str) noexcept {
	reserve(str.getCapacity());
	lenSize = str.size();
	for (uint64_t i = 0; i < lenSize; ++i)
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
			vengine_free(ptr);
		ptr = nullptr;
		capacity = 0;
		lenSize = 0;
	}
	return *this;
}
wstring& wstring::operator=(wstring&& data) noexcept {
	if (ptr)
		vengine_free(ptr);
	ptr = data.ptr;
	lenSize = data.lenSize;
	capacity = data.capacity;
	data.ptr = nullptr;
	data.lenSize = 0;
	data.capacity = 0;
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
wstring& wstring::operator=(wchar_t data) noexcept {
	lenSize = 1;
	reserve(2);
	ptr[0] = data;
	ptr[1] = 0;
	return *this;
}
wstring& wstring::operator+=(const wstring& str) noexcept {
	if (str.ptr) {
		uint64_t newCapacity = lenSize + str.lenSize + 1;
		reserve(newCapacity);
		memcpy(ptr + lenSize, str.ptr, str.lenSize * 2);
		lenSize = newCapacity - 1;
		ptr[lenSize] = 0;
	}
	return *this;
}
wstring& wstring::operator+=(const wchar_t* str) noexcept {
	uint64_t newStrLen = wstrLen(str);
	uint64_t newCapacity = lenSize + newStrLen + 1;
	reserve(newCapacity);
	memcpy(ptr + lenSize, str, newStrLen * 2);
	lenSize = newCapacity - 1;
	ptr[lenSize] = 0;
	return *this;
}
wstring& wstring::operator+=(wchar_t str) noexcept {
	static const uint64_t newStrLen = 1;
	uint64_t newCapacity = lenSize + newStrLen + 1;
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
		uint64_t newLenSize = a.lenSize + b.lenSize;
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
	uint64_t newLenSize = a.lenSize + newLen;
	reserve(newLenSize + 1);
	lenSize = newLenSize;
	if (a.ptr)
		memcpy(ptr, a.ptr, a.lenSize * 2);
	memcpy(ptr + a.lenSize, b, newLen * 2);
	ptr[lenSize] = 0;
}
wstring::wstring(const wchar_t* a, const wstring& b) noexcept {
	size_t newLen = wstrLen(a);
	uint64_t newLenSize = b.lenSize + newLen;
	reserve(newLenSize + 1);
	lenSize = newLenSize;
	memcpy(ptr, a, newLen * 2);
	if (b.ptr)
		memcpy(ptr + newLen, b.ptr, b.lenSize * 2);
	ptr[lenSize] = 0;
}
wstring::wstring(const wstring& a, wchar_t b) noexcept {
	uint64_t newLenSize = a.lenSize + 1;
	reserve(newLenSize + 1);
	lenSize = newLenSize;
	if (a.ptr)
		memcpy(ptr, a.ptr, a.lenSize * 2);
	ptr[a.lenSize] = b;
	ptr[newLenSize] = 0;
}
wstring::wstring(wchar_t a, const wstring& b) noexcept {
	uint64_t newLenSize = b.lenSize + 1;
	reserve(newLenSize + 1);
	lenSize = newLenSize;
	if (b.ptr)
		memcpy(ptr + 1, b.ptr, b.lenSize * 2);
	ptr[0] = a;
	ptr[newLenSize] = 0;
}

wchar_t& wstring::operator[](uint64_t index) noexcept {
#if defined(DEBUG) || defined(_DEBUG)
	if (index >= lenSize)
		throw "Out of Range Exception!";
#endif
	return ptr[index];
}
void wstring::erase(uint64_t index) noexcept {
#if defined(DEBUG) || defined(_DEBUG)
	if (index >= lenSize)
		throw "Out of Range Exception!";
#endif
	memmove(ptr + index, ptr + index + 1, (lenSize - index) * 2);
	lenSize--;
}
wchar_t const& wstring::operator[](uint64_t index) const noexcept {
#if defined(DEBUG) || defined(_DEBUG)
	if (index >= lenSize)
		throw "Out of Range Exception!";
#endif
	return ptr[index];
}
bool wstring::Equal(wchar_t const* str, uint64_t count) const noexcept {
	uint64_t bit64Count = count / 8;
	uint64_t leftedCount = count - bit64Count * 8;
	uint64_t const* value = (uint64_t const*)str;
	uint64_t const* oriValue = (uint64_t const*)ptr;
	for (uint64_t i = 0; i < bit64Count; ++i) {
		if (value[i] != oriValue[i]) return false;
	}
	wchar_t const* c = (wchar_t const*)(value + bit64Count);
	wchar_t const* oriC = (wchar_t const*)(oriValue + bit64Count);
	for (uint64_t i = 0; i < leftedCount; ++i) {
		if (c[i] != oriC[i]) return false;
	}
	return true;
}
string_view::string_view(vengine::string const& str) : data(str.data()), mSize(str.size()) {}
wstring_view::wstring_view(vengine::wstring const& str) : data(str.data()), mSize(str.size()) {}

#pragma endregion
}// namespace vengine
#include "Log.h"
DynamicDLL::DynamicDLL(char const* name) {
	inst = LoadLibraryA(name);
	if (inst == nullptr) {
		VEngine_Log(
			{"Can not find DLL ",
			 name});
		throw 0;
	}
}
DynamicDLL::~DynamicDLL() {
	FreeLibrary(inst);
}
