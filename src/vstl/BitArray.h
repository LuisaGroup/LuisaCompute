#pragma once
#include <vstl/config.h>
#include <stdint.h>
#include <memory>
namespace vstd {

class BitArray {
public:
	struct Iterator {
		friend class BitArray;

	private:
		BitArray const* arr;
		size_t index;
		Iterator(BitArray const* arr, size_t index) : arr(arr), index(index) {}

	public:
		bool operator!() const {
			return !operator bool();
		}

		operator bool() const {
			return arr->Get(index);
		}
		void operator=(bool value) {
			arr->Set(index, value);
		}
		bool operator==(const Iterator& another) const {
			return arr == another.arr && index == another.index;
		}
		bool operator!=(const Iterator& another) const {
			return !operator==(another);
		}
		inline void operator++() {
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
#ifdef DEBUG
		if (index >= length) throw "Index Out of Range!";
#endif
		size_t elementIndex = index / 8;
		size_t factor = index - (elementIndex * 8);
		return ptr[elementIndex] & bitOffsetArray[factor];
	}
	inline void Set(size_t index, bool value) const {
#ifdef DEBUG
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
		return Iterator(this, 0);
	}
	Iterator end() const {
		return Iterator(this, length);
	}
	Iterator operator[](size_t index) const {
		return Iterator(this, index);
	}
	inline BitArray(size_t length) : length(length) {
		const size_t capa = (length % 8 > 0) ? length / 8 + 1 : length / 8;
		ptr = new uint8_t[capa];
		memset(ptr, 0, sizeof(uint8_t) * capa);
	}
	inline void Reset(bool target) {
		size_t capa = (length % 8 > 0) ? length / 8 + 1 : length / 8;
		memset(ptr, target ? 255 : 0, sizeof(uint8_t) * capa);
	}
	inline ~BitArray() {
		if (ptr) {
			delete ptr;
		}
	}
};

template<size_t length>
class StaticBitArray {
public:
	struct Iterator {
		friend class StaticBitArray;

	private:
		StaticBitArray const* arr;
		size_t index;
		Iterator(StaticBitArray const* arr, size_t index) : arr(arr), index(index) {}

	public:
		bool operator!() const {
			return !operator bool();
		}

		operator bool() const {
			return arr->Get(index);
		}
		void operator=(bool value) {
			arr->Set(index, value);
		}
		bool operator==(const Iterator& another) const {
			return arr == another.arr && index == another.index;
		}
		bool operator!=(const Iterator& another) const {
			return !operator==(another);
		}
		inline void operator++() {
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
	mutable std::aligned_storage_t<((length % 8 > 0) ? length / 8 + 1 : length / 8), 1> data;
	uint8_t* ptr() const {
		return reinterpret_cast<uint8_t*>(&data);
	}
	bool Get(size_t index) const {
#ifdef DEBUG
		if (index >= length) throw "Index Out of Range!";
#endif
		size_t elementIndex = index / 8;
		size_t factor = index - (elementIndex * 8);
		return ptr()[elementIndex] & bitOffsetArray[factor];
	}
	inline void Set(size_t index, bool value) const {
#ifdef DEBUG
		if (index >= length) throw "Index Out of Range!";
#endif
		size_t elementIndex = index / 8;
		size_t factor = index - (elementIndex * 8);
		if (value) {
			ptr()[elementIndex] |= bitOffsetArray[factor];
		} else {
			ptr()[elementIndex] &= bitOffsetReversedArray[factor];
		}
	}

public:
	Iterator begin() const {
		return Iterator(this, 0);
	}
	Iterator end() const {
		return Iterator(this, length);
	}
	Iterator operator[](size_t index) const {
		return Iterator(this, index);
	}

	StaticBitArray() {
		memset(ptr(), 0, sizeof(data));
	}
	void Reset(bool target) {
		memset(ptr(), target ? 255 : 0, sizeof(data));
	}
};
}// namespace vstd