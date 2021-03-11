#pragma once
#include <stdint.h>
#include <memory>
#include "Memory.h"
class BitArray {
public:
	struct Iterator {
		friend class BitArray;

	private:
		BitArray* arr;
		size_t index;
		constexpr Iterator(BitArray* arr, size_t index) : arr(arr), index(index) {}

	public:
		operator bool() const {
			return arr->Get(index);
		}
		void operator=(bool value) {
			arr->Set(index, value);
		}
		constexpr bool operator==(const Iterator& another) const {
			return arr == another.arr && index == another.index;
		}
		constexpr bool operator!=(const Iterator& another) const {
			return !operator==(another);
		}
		constexpr void operator++() {
			index++;
		}
	};

private:
	inline static const uint8_t bitOffsetArray[8]{
		0b00000001,
		0b00000010,
		0b00000100,
		0b00001000,
		0b00010000,
		0b00100000,
		0b01000000,
		0b10000000};
	inline static const uint8_t bitOffsetReversedArray[8]{
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
	bool Get(size_t index) const {
#ifndef NDEBUG
		if (index >= length) throw "Index Out of Range!";
#endif
		size_t elementIndex = index / 8;
		size_t factor = index - (elementIndex * 8);
		return ptr[elementIndex] & bitOffsetArray[factor];
	}
	inline constexpr void Set(size_t index, bool value) {
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

public:
	BitArray(BitArray&& o)
		: ptr(o.ptr),
		  length(o.length) {
		o.ptr = nullptr;
	}
	Iterator begin() const {
		return Iterator((BitArray*)this, 0);
	}
	Iterator end() const {
		return Iterator((BitArray*)this, length);
	}
	Iterator operator[](size_t index) {
		return Iterator((BitArray*)this, index);
	}
	inline BitArray(size_t length) : length(length) {
		const size_t capa = (length % 8 > 0) ? length / 8 + 1 : length / 8;
		ptr = (uint8_t*)vengine_malloc(capa);
		memset(ptr, 0, sizeof(uint8_t) * capa);
	}
	inline void Reset(bool target) {
		size_t capa = (length % 8 > 0) ? length / 8 + 1 : length / 8;
		memset(ptr, target ? 1 : 0, sizeof(uint8_t) * capa);
	}
	inline ~BitArray() {
		if (ptr) {
			vengine_free(ptr);
		}
	}
};