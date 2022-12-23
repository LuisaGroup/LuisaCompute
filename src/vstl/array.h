#pragma once
#include <type_traits>
#include "v_allocator.h"
namespace vstd {
template<typename T, size_t mSize>
	requires(mSize > 0)
class array {
	std::aligned_storage_t<sizeof(T) * mSize, alignof(T)> storage;

public:
	static constexpr size_t size = mSize;
	static constexpr size_t byte_size = mSize * sizeof(T);
	using value_type = T;
	template<typename... Args>
		requires(std::is_constructible_v<T, Args&&...>)
	array(Args&&... args) {
		if constexpr (!std::is_trivially_constructible_v<T>) {
			T* ptr = reinterpret_cast<T*>(&storage);
			T* endPtr = ptr + mSize;
			while (ptr != endPtr) {
				new (ptr) T(std::forward<Args>(args)...);
				++ptr;
			}
		}
	}
	array(array&& v) {
		if constexpr (std::is_trivially_move_constructible_v<T>) {
			memcpy(&storage, &v.storage, byte_size);
		} else {
			T* ptr = reinterpret_cast<T*>(&storage);
			T* otherPtr = reinterpret_cast<T*>(&v.storage);
			for (size_t i = 0; i < mSize; ++i) {
				new (ptr + i) T(std::move(otherPtr[i]));
			}
		}
	}
	array(std::initializer_list<T> list) {
		assert(list.size() == mSize);
		if constexpr (std::is_trivially_copy_constructible_v<T>) {
			memcpy(&storage, list.begin(), byte_size);
		} else {
			T* ptr = reinterpret_cast<T*>(&storage);
			T* otherPtr = list.begin();
			for (size_t i = 0; i < mSize; ++i) {
				new (ptr + i) T(otherPtr[i]);
			}
		}
	}
	array(array const& v) {
		if constexpr (std::is_trivially_copy_constructible_v<T>) {
			memcpy(&storage, &v.storage, byte_size);
		} else {
			T* ptr = reinterpret_cast<T*>(&storage);
			T const* otherPtr = reinterpret_cast<T const*>(&v.storage);
			for (size_t i = 0; i < mSize; ++i) {
				new (ptr + i) T(otherPtr[i]);
			}
		}
	}
	array& operator=(array const& v) {
		if constexpr (std::is_trivially_copy_assignable_v<T>) {
			memcpy(&storage, &v.storage, byte_size);
		} else {
			T* ptr = reinterpret_cast<T*>(&storage);
			T const* otherPtr = reinterpret_cast<T const*>(&v.storage);
			for (size_t i = 0; i < mSize; ++i) {
				ptr[i] = otherPtr[i];
			}
		}
		return *this;
	}
	array& operator=(array&& v) {
		if constexpr (std::is_trivially_move_assignable_v<T>) {
			memcpy(&storage, &v.storage, byte_size);
		} else {
			T* ptr = reinterpret_cast<T*>(&storage);
			T* otherPtr = reinterpret_cast<T*>(&v.storage);
			for (size_t i = 0; i < mSize; ++i) {
				ptr[i] = std::move(otherPtr[i]);
			}
		}
		return *this;
	}
	array& operator=(std::initializer_list<T> v){
		this->~array();
		new(this)array(v);
		return *this;
	}
	~array() {
		if constexpr (!std::is_trivially_destructible_v<T>) {
			T* ptr = reinterpret_cast<T*>(&storage);
			T* endPtr = ptr + mSize;
			while (ptr != endPtr) {
				ptr->~T();
				++ptr;
			}
		}
	}
	T const& operator[](size_t index) const& {
		T const* ptr = reinterpret_cast<T const*>(&storage);
		return ptr[index];
	}
	T& operator[](size_t index) & {
		T* ptr = reinterpret_cast<T*>(&storage);
		return ptr[index];
	}
	T&& operator[](size_t index) && {
		T* ptr = reinterpret_cast<T*>(&storage);
		return std::move(ptr[index]);
	}
	T* data() {
		return reinterpret_cast<T*>(&storage);
	}
	T const* data() const {
		return reinterpret_cast<T const*>(&storage);
	}
	T* begin() {
		return reinterpret_cast<T*>(&storage);
	}
	T* end() {
		return reinterpret_cast<T*>(&storage) + mSize;
	}
	T const* begin() const {
		return reinterpret_cast<T const*>(&storage);
	}
	T const* end() const {
		return reinterpret_cast<T const*>(&storage) + mSize;
	}
};
}// namespace vstd