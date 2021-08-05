#pragma once
#include <util/vstlconfig.h>
#include <memory>
#include <stdint.h>
#include <util/Memory.h>
class BitVector {
public:
	struct Iterator {
		friend class BitVector;

	private:
		BitVector* arr;
		size_t index;
		Iterator(BitVector* arr, size_t index) noexcept : arr(arr), index(index) {}

	public:
		operator bool() const noexcept {
			return arr->Get(index);
		}
		bool operator!() const {
			return !operator bool();
		}
		void operator=(bool value) noexcept {
			arr->Set(index, value);
		}
		bool operator==(const Iterator& another) const noexcept {
			return arr == another.arr && index == another.index;
		}
		bool operator!=(const Iterator& another) const noexcept {
			return !operator==(another);
		}
		void operator++() noexcept {
			index++;
		}
	};

private:
	static constexpr uint8_t bitOffsetArray[8]{
		0b00000001,
		0b00000010,
		0b00000100,
		0b00001000,
		0b00010000,
		0b00100000,
		0b01000000,
		0b10000000};
	static constexpr uint8_t bitOffsetReversedArray[8]{
		0b11111110,
		0b11111101,
		0b11111011,
		0b11110111,
		0b11101111,
		0b11011111,
		0b10111111,
		0b01111111};
	uint8_t* ptr = nullptr;
	size_t length = 0;
	size_t capacity = 0;
	
	bool Get(size_t index) const noexcept {
#ifndef NDEBUG
		if (index >= length) throw "Index Out of Range!";
#endif
		size_t elementIndex = index / 8;
		size_t factor = index - (elementIndex * 8);
		return ptr[elementIndex] & bitOffsetArray[factor];
	}
	void Set(size_t index, bool value) noexcept {
#ifndef NDEBUG
		if (index >= length) throw "Index Out of Range!";
#endif
		size_t elementIndex = index / 8;
		size_t factor = index - (elementIndex * 8);
		if (value) {
			ptr[elementIndex] |= bitOffsetArray[factor];
		} else {
			ptr[elementIndex] &= bitOffsetReversedArray[factor];
		}
	}
	void Reserve() noexcept {
		if (length <= capacity) return;
		size_t capa = length * 1.5 + 8;
		const size_t charSize = (capa % 8 > 0) ? capa / 8 + 1 : capa / 8;
		uint8_t* newPtr = (uint8_t*)vengine_default_malloc(sizeof(uint8_t) * charSize);
		if (ptr) {
			const size_t oldCharSize = (length % 8 > 0) ? length / 8 + 1 : length / 8;
			memcpy(newPtr, ptr, oldCharSize);
			vengine_default_free(ptr);
		}
		ptr = newPtr;
		capacity = charSize * 8;
	}

public:
	BitVector(BitVector&& o)
		: ptr(o.ptr),
		  length(o.length),
		  capacity(o.capacity) {
		o.ptr = nullptr;
	}
	size_t size() const noexcept {
		return length;
	}
	size_t Capacity() const noexcept {
		return capacity;
	}
	Iterator begin() const noexcept {
		return Iterator((BitVector*)this, 0);
	}
	Iterator end() const noexcept {
		return Iterator((BitVector*)this, length);
	}
	Iterator operator[](size_t index) const noexcept {
		return Iterator((BitVector*)this, index);
	}
	//bool RemoveLast()
	void Reserve(size_t capa) noexcept {
		if (capa <= capacity) return;
		const size_t charSize = (capa % 8 > 0) ? capa / 8 + 1 : capa / 8;
		uint8_t* newPtr = (uint8_t*)vengine_default_malloc(sizeof(uint8_t) * charSize);
		if (ptr) {
			const size_t oldCharSize = (length % 8 > 0) ? length / 8 + 1 : length / 8;
			memcpy(newPtr, ptr, oldCharSize);
			vengine_default_free(ptr);
		}
		ptr = newPtr;
		capacity = charSize * 8;
	}

	BitVector() noexcept {}
	void Resize(size_t size) noexcept {
		Reserve(size);
		length = size;
	}
	bool Empty() const noexcept {
		return length == 0;
	}
	void PushBack(bool value) noexcept {
		size_t last = length++;
		Reserve();
		Set(last, value);
	}
	bool RemoveLast() noexcept {
#if defined(DEBUG)
		if (Empty()) {
			throw "Empty can not move!";
		}
#endif
		bool value = Get(length);
		length--;
		return value;
	}
	void Reset() noexcept {
		if (!ptr) return;
		const size_t oldCharSize = (length % 8 > 0) ? length / 8 + 1 : length / 8;
		memset(ptr, 0, oldCharSize);
	}
	void Clear() noexcept {
		length = 0;
	}
	~BitVector() noexcept {
		if (ptr) {
			vengine_default_free(ptr);
		}
	}
};
