
#include <vstl/vstring.h>
#include <vstl/Pool.h>
#include <mutex>
#include <vstl/functional.h>
#include <vstl/Memory.h>
#include <vstl/vector.h>
#include <vstl/MetaLib.h>
//#include "BinaryLinkedAllocator.h"
#include <vstl/TreeMap.h>
#include <ext/mimalloc/include/mimalloc.h>
namespace v_mimalloc {
vstd::funcPtr_t<void*(size_t)> mallocFunc = nullptr;
vstd::funcPtr_t<void(void*)> freeFunc = nullptr;
vstd::funcPtr_t<void*(void*, size_t)> reallocFunc = nullptr;
struct MAllocator {
	MAllocator() {
		mallocFunc = malloc;
		freeFunc = free;
		reallocFunc = realloc;
	}

};
static MAllocator vengine_malloc_dll;
}// namespace v_mimalloc

void* vengine_default_malloc(size_t sz) {
	using namespace v_mimalloc;
	return mallocFunc(sz);
}
void vengine_default_free(void* ptr) {
	using namespace v_mimalloc;
	freeFunc(ptr);
}

void* vengine_default_realloc(void* ptr, size_t size) {
	using namespace v_mimalloc;
	return reallocFunc(ptr, size);
}

void* vengine_malloc(size_t size) {
	using namespace v_mimalloc;
	return mallocFunc(size);
}
void vengine_free(void* ptr) {
	using namespace v_mimalloc;
	freeFunc(ptr);
}
void* vengine_realloc(void* ptr, size_t size) {
	using namespace v_mimalloc;
	return reallocFunc(ptr, size);
}
namespace vstd {
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
void string::push_back_all(char c, size_t newStrLen) noexcept {
	size_t newCapacity = lenSize + newStrLen + 1;
	reserve(newCapacity);
	auto originPtr = ptr + lenSize;
	for (auto&& i : ptr_range(originPtr, originPtr + newStrLen)) {
		i = c;
	}
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
	wchar_t* newPtr = (wchar_t*)wstring_malloc(targetCapacity * sizeof(wchar_t));
	if (ptr) {
		memcpy(newPtr, ptr, (lenSize + 1) * sizeof(wchar_t));
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
string_view::string_view(string const& str) : data(str.data()), mSize(str.size()) {}
wstring_view::wstring_view(wstring const& str) : data(str.data()), mSize(str.size()) {}
namespace detail {

struct Node {
	bool color;	 // 1 -> Red, 0 -> Black
	Node* parent;// pointer to the parent
	Node* left;	 // pointer to left child
	Node* right; // pointer to right child
};

void leftRotate(void* vx, void*& root) {
	Node* x = reinterpret_cast<Node*>(vx);
	Node* y = x->right;
	x->right = y->left;
	if (y->left != nullptr) {
		y->left->parent = x;
	}
	y->parent = x->parent;
	if (x->parent == nullptr) {
		root = y;
	} else if (x == x->parent->left) {
		x->parent->left = y;
	} else {
		x->parent->right = y;
	}
	y->left = x;
	x->parent = y;
}

// rotate right at node x
void rightRotate(void* vx, void*& root) {
	Node* x = reinterpret_cast<Node*>(vx);
	Node* y = x->left;
	x->left = y->right;
	if (y->right != nullptr) {
		y->right->parent = x;
	}
	y->parent = x->parent;
	if (x->parent == nullptr) {
		root = y;
	} else if (x == x->parent->right) {
		x->parent->right = y;
	} else {
		x->parent->left = y;
	}
	y->right = x;
	x->parent = y;
}
void fixDelete(void* vptr, void*& vRoot, Node*& tNullParent) {
	Node* x = reinterpret_cast<Node*>(vptr);
	Node* root = reinterpret_cast<Node*>(vRoot);
	Node* s;
	bool xIsNull;
	while (x != root && ((xIsNull = (x == nullptr)) || x->color == 0)) {
		auto&& xParent = xIsNull ? *reinterpret_cast<Node**>(&tNullParent) : x->parent;
		if (x == xParent->left) {
			s = xParent->right;
			if (s->color == 1) {
				// case 3.1
				s->color = 0;
				xParent->color = 1;
				leftRotate(xParent, vRoot);
				s = xParent->right;
			}
			bool leftNull = s->left == nullptr || s->left->color == 0;
			bool rightNull = s->right == nullptr || s->right->color == 0;
			if (leftNull && rightNull){
				// case 3.2
				s->color = 1;
				x = xParent;
			} else {
				if (rightNull) {
					// case 3.3
					s->left->color = 0;
					s->color = 1;
					rightRotate(s, vRoot);
					s = xParent->right;
				}

				// case 3.4
				s->color = xParent->color;
				xParent->color = 0;
				s->right->color = 0;
				leftRotate(xParent, vRoot);
				x = root;
			}
		} else {
			s = xParent->left;
			if (s->color == 1) {
				// case 3.1
				s->color = 0;
				xParent->color = 1;
				rightRotate(xParent, vRoot);
				s = xParent->left;
			}
			bool leftNull = s->left == nullptr || s->left->color == 0;
			bool rightNull = s->right == nullptr || s->right->color == 0;
			if (leftNull && rightNull){
				// case 3.2
				s->color = 1;
				x = xParent;
			} else {
				if (leftNull) {
					// case 3.3
					s->right->color = 0;
					s->color = 1;
					leftRotate(s, vRoot);
					s = xParent->left;
				}

				// case 3.4
				s->color = xParent->color;
				xParent->color = 0;
				s->left->color = 0;
				rightRotate(xParent, vRoot);
				x = root;
			}
		}
	}
	if(x != nullptr)
		x->color = 0;
}

void TreeMapUtility::fixInsert(void* vk, void*& vRoot) {
	Node* k = reinterpret_cast<Node*>(vk);
	Node* u;
	while (k->parent->color == 1) {
		if (k->parent == k->parent->parent->right) {
			u = k->parent->parent->left;// uncle
			if (u != nullptr && u->color == 1) {
				// case 3.1
				u->color = 0;
				k->parent->color = 0;
				k->parent->parent->color = 1;
				k = k->parent->parent;
			} else {
				if (k == k->parent->left) {
					// case 3.2.2
					k = k->parent;
					rightRotate(k, vRoot);
				}
				// case 3.2.1
				k->parent->color = 0;
				k->parent->parent->color = 1;
				leftRotate(k->parent->parent, vRoot);
			}
		} else {
			u = k->parent->parent->right;// uncle

			if (u != nullptr && u->color == 1) {
				// mirror case 3.1
				u->color = 0;
				k->parent->color = 0;
				k->parent->parent->color = 1;
				k = k->parent->parent;
			} else {
				if (k == k->parent->right) {
					// mirror case 3.2.2
					k = k->parent;
					leftRotate(k, vRoot);
				}
				// mirror case 3.2.1
				k->parent->color = 0;
				k->parent->parent->color = 1;
				rightRotate(k->parent->parent, vRoot);
			}
		}
		if (k == vRoot) {
			break;
		}
	}
	reinterpret_cast<Node*>(vRoot)->color = 0;
}
Node* minimum(Node* node) {
	while (node->left != nullptr) {
		node = node->left;
	}
	return node;
}

// find the node with the maximum key
Node* maximum(Node* node) {
	while (node->right != nullptr) {
		node = node->right;
	}
	return node;
}

void rbTransplant(Node* u, Node* v, void*& root, Node*& tNullParent) {
	if (u->parent == nullptr) {
		root = v;
	} else if (u == u->parent->left) {
		u->parent->left = v;
	} else {
		u->parent->right = v;
	}
	if (v == nullptr)
		tNullParent = u->parent;
	else
		v->parent = u->parent;
}

void TreeMapUtility::deleteOneNode(void* vz, void*& root) {
	Node* tNullParent = nullptr;
	Node* z = reinterpret_cast<Node*>(vz);
	Node* x;
	Node* y;
	y = z;
	int y_original_color = y->color;
	if (z->left == nullptr) {
		x = z->right;
		rbTransplant(z, z->right, root, tNullParent);
	} else if (z->right == nullptr) {
		x = z->left;
		rbTransplant(z, z->left, root, tNullParent);
	} else {
		y = minimum(z->right);
		y_original_color = y->color;
		x = y->right;
		if (y->parent == z) {
			if(x)
			x->parent = y;
			else tNullParent = y;
		} else {
			rbTransplant(y, y->right, root, tNullParent);
			y->right = z->right;
			y->right->parent = y;
		}

		rbTransplant(z, y, root, tNullParent);
		y->left = z->left;
		y->left->parent = y;
		y->color = z->color;
	}
	if (y_original_color == 0) {
		fixDelete(x, root, tNullParent);
	}
}

void* TreeMapUtility::getNext(void* vptr) {
	Node* ptr = reinterpret_cast<Node*>(vptr);
	if (ptr->right == nullptr) {
		Node* pNode;
		while (((pNode = ptr->parent) != nullptr) && (ptr == pNode->right)) {
			ptr = pNode;
		}
		ptr = pNode;
	} else {
		ptr = minimum(ptr->right);
	}
	return ptr;
}
void* TreeMapUtility::getLast(void* vptr) {
	Node* ptr = reinterpret_cast<Node*>(vptr);
	if (ptr->left == nullptr) {
		Node* pNode;
		while (((pNode = ptr->parent) != nullptr) && (ptr == pNode->left)) {
			ptr = pNode;
		}
		if (ptr != nullptr) {
			ptr = pNode;
		}
	} else {
		ptr = maximum(ptr->left);
	}
	return ptr;
}

}// namespace detail
#pragma endregion
}// namespace vstd

vstd::string_view operator"" _sv(char const* str, size_t sz) {
	return vstd::string_view(str, sz);
}

vstd::wstring_view operator"" _sv(wchar_t const* str, size_t sz) {
	return vstd::wstring_view(str, sz);
}

#ifdef EXPORT_UNITY_FUNCTION
VENGINE_UNITY_EXTERN void vengine_memcpy(void* dest, void* src, uint64 sz) {
	memcpy(dest, src, sz);
}
VENGINE_UNITY_EXTERN void vengine_memset(void* dest, byte b, uint64 sz) {
	memset(dest, b, sz);
}
VENGINE_UNITY_EXTERN void vengine_memmove(void* dest, void* src, uint64 sz) {
	memmove(dest, src, sz);
}
#endif