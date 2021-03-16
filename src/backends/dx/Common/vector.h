#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <memory>
#include <initializer_list>
#include <type_traits>
#include "Memory.h"

namespace vengine {

template<typename T, bool useVEngineAlloc = true, bool forceTrivial = std::is_trivial_v<T>>
class vector {
private:
	struct ValueType {
		T value;
		~ValueType() {}
	};
	T* arr;
	uint64_t mSize;
	uint64_t mCapacity;
	static constexpr uint64_t GetNewVectorSize(uint64_t oldSize) {
		if constexpr (useVEngineAlloc) {
			return oldSize * 1.5 + 8;

		} else {
			if (oldSize == 0)
				oldSize = 8;
			oldSize *= 2;
			return oldSize;
		}
	}
	static T* Allocate(uint64_t& capacity) noexcept {
		if constexpr (useVEngineAlloc) {
			capacity *= sizeof(T);
			auto ptr = (T*)vengine_malloc(capacity);
			capacity /= sizeof(T);
			return ptr;
		} else {
			return (T*)malloc(sizeof(T) * capacity);
		}
	}

	void Free(T* ptr) noexcept {
		if constexpr (useVEngineAlloc) {
			vengine_free(ptr);
		} else {
			free(ptr);
		}
	}

public:
	void reserve(uint64_t newCapacity) noexcept {
		if (newCapacity <= mCapacity) return;
		T* newArr = Allocate(newCapacity);
		if (arr) {
			if constexpr (std::is_trivially_copyable_v<T> || forceTrivial) {
				memcpy(newArr, arr, sizeof(T) * mSize);
				if constexpr (!(std::is_trivially_destructible_v<T> || forceTrivial)) {
					for (uint64_t i = 0; i < mSize; ++i) {
						(reinterpret_cast<ValueType*>(arr + i))->~ValueType();
					}
				}
			} else {
				for (uint64_t i = 0; i < mSize; ++i) {
					new (newArr + i) T(static_cast<T&&>(arr[i]));
					if constexpr (!(std::is_trivially_destructible_v<T> || forceTrivial)) {
						(reinterpret_cast<ValueType*>(arr + i))->~ValueType();
					}
				}
			}
			auto tempArr = arr;
			arr = newArr;
			Free(tempArr);
		} else {
			arr = newArr;
		}
		mCapacity = newCapacity;
	}
	T* data() const noexcept { return arr; }
	uint64_t size() const noexcept { return mSize; }
	uint64_t capacity() const noexcept { return mCapacity; }
	struct Iterator {
		friend class vector<T, useVEngineAlloc, forceTrivial>;

	private:
		const vector<T, useVEngineAlloc, forceTrivial>* lst;
		uint64_t index;
		constexpr Iterator(const vector<T, useVEngineAlloc, forceTrivial>* lst, uint64_t index) noexcept : lst(lst), index(index) {}

	public:
		bool constexpr operator==(const Iterator& ite) const noexcept {
			return index == ite.index;
		}
		bool constexpr operator!=(const Iterator& ite) const noexcept {
			return index != ite.index;
		}
		void operator++() noexcept {
			index++;
		}
		void operator--() noexcept {
			index--;
		}
		uint64_t GetIndex() const noexcept {
			return index;
		}
		void operator++(int32_t) noexcept {
			index++;
		}
		void operator--(int32_t) noexcept {
			index--;
		}
		Iterator constexpr operator+(uint64_t value) const noexcept {
			return Iterator(lst, index + value);
		}
		Iterator constexpr operator-(uint64_t value) const noexcept {
			return Iterator(lst, index - value);
		}
		Iterator& operator+=(uint64_t value) noexcept {
			index += value;
			return *this;
		}
		Iterator& operator-=(uint64_t value) noexcept {
			index -= value;
			return *this;
		}
		T* operator->() const noexcept {
#if defined(DEBUG) || defined(_DEBUG)
			if (index >= lst->mSize) throw "Out of Range!";
#endif
			return &(*lst).arr[index];
		}
		T& operator*() const noexcept {
#if defined(DEBUG) || defined(_DEBUG)
			if (index >= lst->mSize) throw "Out of Range!";
#endif
			return (*lst).arr[index];
		}
	};
	vector(uint64_t mSize) noexcept : mSize(mSize), mCapacity(mSize) {
		arr = Allocate(mCapacity);
		if constexpr (!(std::is_trivially_constructible_v<T> || forceTrivial)) {
			for (uint64_t i = 0; i < mSize; ++i) {
				new (arr + i) T();
			}
		}
	}
	vector(std::initializer_list<T> const& lst) : mSize(lst.size()), mCapacity(lst.size()) {
		arr = Allocate(mCapacity);
		if constexpr (!(std::is_trivially_copy_constructible_v<T> || forceTrivial)) {
			for (uint64_t i = 0; i < mSize; ++i) {
				new (arr + i) T(lst.begin()[i]);
			}
		} else {
			memcpy(arr, lst.begin(), sizeof(T) * mSize);
		}
	}
	vector(const vector<T, useVEngineAlloc, forceTrivial>& another) noexcept : mSize(another.mSize), mCapacity(another.mCapacity) {
		arr = Allocate(mCapacity);
		if constexpr (!(std::is_trivially_copy_constructible_v<T> || forceTrivial)) {
			for (uint64_t i = 0; i < mSize; ++i) {
				new (arr + i) T(another.arr[i]);
			}
		} else
			memcpy(arr, another.arr, sizeof(T) * mSize);
	}
	vector(vector<T, useVEngineAlloc, forceTrivial>&& another) noexcept
		: mSize(another.mSize), mCapacity(another.mCapacity),
		  arr(another.arr) {
		another.arr = nullptr;
		another.mSize = 0;
		another.mCapacity = 0;
	}
	void operator=(const vector<T, useVEngineAlloc, forceTrivial>& another) noexcept {
		if (&another == this) return;
		clear();
		reserve(another.mSize);
		mSize = another.mSize;
		if constexpr (!(std::is_trivially_copy_constructible_v<T> || forceTrivial)) {
			for (uint64_t i = 0; i < mSize; ++i) {
				new (arr + i) T(another.arr[i]);
			}
		} else {
			memcpy(arr, another.arr, sizeof(T) * mSize);
		}
	}
	void operator=(vector<T, useVEngineAlloc, forceTrivial>&& another) noexcept {
		if (&another == this) return;
		clear();
		if (arr)
			Free(arr);
		arr = another.arr;
		mSize = another.mSize;
		mCapacity = another.mCapacity;
		another.arr = nullptr;
		another.mSize = 0;
		another.mCapacity = 0;
	}
	void push_back_all(const T* values, uint64_t count) noexcept {
		if constexpr (!(std::is_trivial_v<T> || forceTrivial)) {
			for (uint64_t i = 0; i < count; ++i) {
				push_back(values[i]);
			}
		} else {
			if (mSize + count > mCapacity) {
				uint64_t newCapacity = GetNewVectorSize(mCapacity);
				uint64_t values[2] = {
					mCapacity + 1, count + mSize};
				newCapacity = newCapacity > values[0] ? newCapacity : values[0];
				newCapacity = newCapacity > values[1] ? newCapacity : values[1];
				reserve(newCapacity);
			}
			memcpy(arr + mSize, values, count * sizeof(T));
			mSize += count;
		}
	}

	void push_back_all(const std::initializer_list<T>& list) noexcept {

		push_back_all(list.begin(), list.size());
	}
	void SetZero() const noexcept {
		if constexpr (!(std::is_trivial_v<T> || forceTrivial)) {
			std::fill_n(arr, mSize, T{0});
//			static_assert(false, "Non-Trivial data cannot be setted");
		} else {
			if (arr) memset(arr, 0, sizeof(T) * mSize);
		}
	}
	void operator=(std::initializer_list<T> const& list) noexcept {
		clear();
		reserve(list.size());
		mSize = list.size();
		if constexpr (!(std::is_trivially_copy_constructible_v<T> || forceTrivial)) {
			for (uint64_t i = 0; i < mSize; ++i) {
				new (arr + i) T(list.begin()[i]);
			}
		} else {
			memcpy(arr, list.begin(), sizeof(T) * mSize);
		}
	}
	vector() noexcept : mCapacity(0), mSize(0), arr(nullptr) {
	}
	~vector() noexcept {
		if (arr) {
			if constexpr (!(std::is_trivially_destructible_v<T> || forceTrivial)) {
				for (uint64_t i = 0; i < mSize; ++i) {
					(reinterpret_cast<ValueType*>(arr + i))->~ValueType();
				}
			}
			Free(arr);
		}
	}
	bool empty() const noexcept {
		return mSize == 0;
	}

	template<typename... Args>
	T& emplace_back(Args&&... args) noexcept {
		if (mSize >= mCapacity) {
			uint64_t newCapacity = GetNewVectorSize(mCapacity);
			reserve(newCapacity);
		}
		T* ptr = arr + mSize;
		new (ptr) T(std::forward<Args>(args)...);
		mSize++;
		return *ptr;
	}

	void push_back(const T& value) noexcept {
		emplace_back(value);
	}
	void push_back(T& value) noexcept {
		emplace_back(value);
	}
	void push_back(T&& value) noexcept {
		emplace_back(static_cast<T&&>(value));
	}

	Iterator begin() const noexcept {
		return Iterator(this, 0);
	}
	Iterator end() const noexcept {
		return Iterator(this, mSize);
	}

	void erase(const Iterator& ite) noexcept {
#if defined(DEBUG) || defined(_DEBUG)
		if (ite.index >= mSize) throw "Out of Range!";
#endif
		if constexpr (!(std::is_trivial_v<T> || forceTrivial)) {
			if constexpr (!(std::is_trivially_copyable_v<T> || forceTrivial)) {
				if (ite.index < mSize - 1) {
					for (uint64_t i = ite.index; i < mSize - 1; ++i) {
						arr[i] = arr[i + 1];
					}
				}
			}
			if constexpr (!(std::is_trivially_destructible_v<T> || forceTrivial))
				(reinterpret_cast<ValueType*>(arr + mSize - 1))->~ValueType();
		} else {
			if (ite.index < mSize - 1) {
				memmove(arr + ite.index, arr + ite.index + 1, (mSize - ite.index - 1) * sizeof(T));
			}
		}
		mSize--;
	}

	decltype(auto) erase_last() noexcept {
#if defined(DEBUG) || defined(_DEBUG)
		if (mSize == 0) throw "Empty List!";
#endif
		mSize--;
		if constexpr (!(std::is_trivial_v<T> || forceTrivial)) {
			T tempValue = arr[mSize];
			(reinterpret_cast<ValueType*>(arr + mSize))->~ValueType();
			return tempValue;
		} else {
			return (T const&)arr[mSize];
		}
	}
	void clear() noexcept {
		if constexpr (!(std::is_trivially_destructible_v<T> || forceTrivial)) {
			for (uint64_t i = 0; i < mSize; ++i) {
				(reinterpret_cast<ValueType*>(arr + i))->~ValueType();
			}
		}
		mSize = 0;
	}
	void dispose() noexcept {
		clear();
		mCapacity = 0;
		if (arr) {
			Free(arr);
			arr = nullptr;
		}
	}
	void resize(uint64_t newSize) noexcept {
		reserve(newSize);
		if constexpr (!(std::is_trivially_constructible_v<T> || forceTrivial)) {
			for (uint64_t i = mSize; i < newSize; ++i) {
				new (arr + i) T();
			}
		}
		mSize = newSize;
	}
	T& operator[](uint64_t index) noexcept {
#if defined(DEBUG) || defined(_DEBUG)
		if (index >= mSize) throw "Out of Range!";
#endif
		return arr[index];
	}
	const T& operator[](uint64_t index) const noexcept {
#if defined(DEBUG) || defined(_DEBUG)
		if (index >= mSize) throw "Out of Range!";
#endif
		return arr[index];
	}
};
}// namespace vengine
template<typename T, bool useVEngineAlloc = true>
using ArrayList = typename vengine::vector<T, useVEngineAlloc, true>;
