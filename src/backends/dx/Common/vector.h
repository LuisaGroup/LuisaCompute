#pragma once
#include <VEngineConfig.h>
#include <stdint.h>
#include <stdlib.h>
#include <memory>
#include <initializer_list>
#include <type_traits>
#include <Common/Memory.h>
#include <Common/VAllocator.h>
#include <span>
namespace vstd {

template<typename T, VEngine_AllocType allocType = VEngine_AllocType::VEngine, bool forceTrivial = false>
class vector {
private:
	using SelfType = vector<T, allocType, forceTrivial>;
	T* arr;
	size_t mSize;
	size_t mCapacity;
	VAllocHandle<allocType> allocHandle;
	static size_t GetNewVectorSize(size_t oldSize) {
		return oldSize * 1.5 + 8;
	}
	T* Allocate(size_t& capacity) noexcept {
		return reinterpret_cast<T*>(allocHandle.Malloc(sizeof(T) * capacity));
	}

	void Free(T* ptr) noexcept {
		allocHandle.Free(ptr);
	}
	void ResizeRange(size_t count) {
		if (mSize + count > mCapacity) {
			size_t newCapacity = GetNewVectorSize(mCapacity);
			size_t values[2] = {
				mCapacity + 1, count + mSize};
			newCapacity = newCapacity > values[0] ? newCapacity : values[0];
			newCapacity = newCapacity > values[1] ? newCapacity : values[1];
			reserve(newCapacity);
		}
	}

public:
	using Iterator = T*;
	void reserve(size_t newCapacity) noexcept {
		if (newCapacity <= mCapacity) return;
		T* newArr = Allocate(newCapacity);
		if (arr) {
			if constexpr (std::is_trivially_move_constructible_v<T> || forceTrivial) {
				memcpy(newArr, arr, sizeof(T) * mSize);
				if constexpr (!(std::is_trivially_destructible_v<T> || forceTrivial)) {
					auto ee = end();
					for (auto i = begin(); i != ee; ++i) {
						i->~T();
					}
				}
			} else {
				for (size_t i = 0; i < mSize; ++i) {
					auto oldT = arr + i;
					new (newArr + i) T(std::move(*oldT));
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
	size_t size() const noexcept { return mSize; }
	size_t capacity() const noexcept { return mCapacity; }
	vector(size_t mSize) noexcept : mSize(mSize), mCapacity(mSize) {
		arr = Allocate(mCapacity);
		if constexpr (!(std::is_trivially_constructible_v<T> || forceTrivial)) {
			auto ee = end();
			for (auto i = begin(); i != ee; ++i) {
				new (i) T();
			}
		}
	}
	vector(std::initializer_list<T> const& lst) : mSize(lst.size()), mCapacity(lst.size()) {
		arr = Allocate(mCapacity);
		auto ptr = &static_cast<T const&>(*lst.begin());
		if constexpr (!(std::is_trivially_copy_constructible_v<T> || forceTrivial)) {
			for (size_t i = 0; i < mSize; ++i) {
				new (arr + i) T(ptr[i]);
			}
		} else {
			memcpy(arr, ptr, sizeof(T) * mSize);
		}
	}
	vector(std::span<T> const& lst) : mSize(lst.size()), mCapacity(lst.size()) {
		arr = Allocate(mCapacity);
		if constexpr (!(std::is_trivially_copy_constructible_v<T> || forceTrivial)) {
			for (size_t i = 0; i < mSize; ++i) {
				new (arr + i) T(lst[i]);
			}
		} else {
			memcpy(arr, &static_cast<T const&>(*lst.begin()), sizeof(T) * mSize);
		}
	}
	vector(const SelfType& another) noexcept
		: mSize(another.mSize),
		  mCapacity(another.mCapacity) {
		arr = Allocate(mCapacity);
		if constexpr (!(std::is_trivially_copy_constructible_v<T> || forceTrivial)) {
			for (size_t i = 0; i < mSize; ++i) {
				new (arr + i) T(another.arr[i]);
			}
		} else
			memcpy(arr, another.arr, sizeof(T) * mSize);
	}
	vector(SelfType&& another) noexcept
		: mSize(another.mSize),
		  mCapacity(another.mCapacity),
		  arr(another.arr) {
		another.arr = nullptr;
		another.mSize = 0;
		another.mCapacity = 0;
	}
	void operator=(const SelfType& another) noexcept {
		clear();
		reserve(another.mSize);
		mSize = another.mSize;
		if constexpr (!(std::is_trivially_copy_constructible_v<T> || forceTrivial)) {
			for (size_t i = 0; i < mSize; ++i) {
				new (arr + i) T(another.arr[i]);
			}
		} else {
			memcpy(arr, another.arr, sizeof(T) * mSize);
		}
	}
	void operator=(SelfType&& another) noexcept {
		this->~SelfType();
		new (this) SelfType(std::move(another));
	}
	
	void push_back_all(const T* values, size_t count) noexcept {
		ResizeRange(count);
		if constexpr (!(std::is_trivially_copy_constructible_v<T> || forceTrivial)) {
			auto endPtr = arr + mSize;
			for (size_t i = 0; i < count; ++i) {
				T* ptr = endPtr + i;
				new (ptr) T(values[i]);
			}
		} else {
			memcpy(arr + mSize, values, count * sizeof(T));
		}
		mSize += count;
	}
	void push_back_all(std::span<T> sp) noexcept {
		push_back_all(sp.data(), sp.size());
	}
	template<typename Func>
	void push_back_func(Func&& f, size_t count) {
		ResizeRange(count);
		auto endPtr = arr + mSize;
		for (size_t i = 0; i < count; ++i) {
			T* ptr = endPtr + i;
			new (ptr) T(std::move(f(i)));
		}
	}
	operator std::span<T>() const {
		return std::span<T>(begin(), end());
	}
	void push_back_all(const std::initializer_list<T>& list) noexcept {
		push_back_all(&static_cast<T const&>(*list.begin()), list.size());
	}
	void SetZero() const noexcept {
		if constexpr (!(std::is_trivial_v<T> || forceTrivial)) {
			static_assert(std::_Always_false<T>, "Non-Trivial data cannot be setted");
		} else {
			if (arr) memset(arr, 0, sizeof(T) * mSize);
		}
	}
	void operator=(std::initializer_list<T> const& list) noexcept {
		clear();
		reserve(list.size());
		mSize = list.size();
		auto ptr = &static_cast<T const&>(*list.begin());
		if constexpr (!(std::is_trivially_copy_constructible_v<T> || forceTrivial)) {
			for (size_t i = 0; i < mSize; ++i) {
				new (arr + i) T(ptr[i]);
			}
		} else {
			memcpy(arr, ptr, sizeof(T) * mSize);
		}
	}
	vector() noexcept : mCapacity(0), mSize(0), arr(nullptr) {
	}
	~vector() noexcept {
		if (arr) {
			if constexpr (!(std::is_trivially_destructible_v<T> || forceTrivial)) {
				auto ee = end();
				for (auto i = begin(); i != ee; ++i) {
					i->~T();
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
			size_t newCapacity = GetNewVectorSize(mCapacity);
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

	T* begin() const noexcept {
		return arr;
	}
	T* end() const noexcept {
		return arr + mSize;
	}

	void erase(T* ite) noexcept {
		size_t index = reinterpret_cast<size_t>(ite)
					   - reinterpret_cast<size_t>(arr);
		index /= sizeof(T);
#if defined(DEBUG)
		if (index >= mSize) throw "Out of Range!";
#endif
		if constexpr (!(std::is_trivially_move_constructible_v<T> || forceTrivial)) {
			if (index < mSize - 1) {
				auto ee = end() - 1;
				for (auto i = begin(); i != ee; ++i) {
					*i = std::move(i[1]);
				}
			}
			if constexpr (!(std::is_trivially_destructible_v<T> || forceTrivial))
				(arr + mSize - 1)->~T();
		} else {
			if (index < mSize - 1) {
				memmove(ite, ite + 1, (mSize - index - 1) * sizeof(T));
			}
		}
		mSize--;
	}

	decltype(auto) erase_last() noexcept {
		mSize--;
		if constexpr (!(std::is_trivially_destructible_v<T> || forceTrivial)) {
			(arr + mSize)->~T();
		} else {
			return (T const&)arr[mSize];
		}
	}
	void clear() noexcept {
		if constexpr (!(std::is_trivially_destructible_v<T> || forceTrivial)) {
			auto ee = end();
			for (auto i = begin(); i != ee; ++i) {
				i->~T();
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
	void resize(size_t newSize) noexcept {
		reserve(newSize);
		if constexpr (!(std::is_trivially_constructible_v<T> || forceTrivial)) {
			auto bb = begin() + mSize;
			auto ee = begin() + newSize;
			for (auto i = begin(); i != ee; ++i) {
				new (i) T();
			}
		}
		mSize = newSize;
	}
	T& operator[](size_t index) noexcept {
#if defined(DEBUG)
		if (index >= mSize) throw "Out of Range!";
#endif
		return arr[index];
	}
	const T& operator[](size_t index) const noexcept {
#if defined(DEBUG)
		if (index >= mSize) throw "Out of Range!";
#endif
		return arr[index];
	}
};
}// namespace vstd
template<typename T, VEngine_AllocType allocType = VEngine_AllocType::VEngine>
using ArrayList = typename vstd::vector<T, allocType, true>;