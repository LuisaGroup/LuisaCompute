#pragma once
#include <luisa/std.hpp>
#include <luisa/resources.hpp>
#include "base.hpp"
template<typename T, uint32 memory_flags = MemoryType::Persist>
class vector;

template<typename T>
class vector<T, MemoryType::Persist> {
private:
	[[clang::annotate("luisa-shader", "vec_ptr")]] uint64 _ptr;
	uint64 _size;
	uint64 _stride;
	uint64 _capacity;
	[[clang::annotate("luisa-shader", "vec_dtor")]] uint64 _dtor;

public:
	[[access]] T& operator[](uint64 index);
	auto ptr() const { return _ptr; }
	auto size() const { return _size; }
	auto capacity() const { return _capacity; }
	vector() {
		_ptr = 0;
		_size = 0;
		_capacity = 0;
		luisa::shader::Ref<T> ref(_ptr);
		if (!luisa::shader::is_trivial(ref.get())) {
			_dtor = (uint64)(&luisa::shader::dispose<T>);
		} else {
			_dtor = 0;
		}
		_stride = sizeof(T);
	}
	void reserve(uint64 dst) {
		if (dst <= _capacity) {
			return;
		}
		auto new_ptr = luisa::shader::persist_malloc(dst * sizeof(T));
		if (_ptr != 0) {
			luisa::shader::memcpy(new_ptr, _ptr, dst * sizeof(T));
			luisa::shader::persist_free(_ptr);
		}
		_ptr = new_ptr;
		_capacity = dst;
	}
	void emplace_back(T& t) {
		auto idx = _size;
		_size += 1;
		if (_size > _capacity) {
			reserve((uint64)(double(_capacity) * 1.5) + 8);
		}
		operator[](idx) = t;
	}
	void emplace_back(T t) {
		auto idx = _size;
		_size += 1;
		if (_size > _capacity) {
			reserve((uint64)(double(_capacity) * 1.5) + 8);
		}
		operator[](idx) = t;
	}
	void pop_back() {
		if (_size > 0) {
			if (_dtor) {
				luisa::shader::FunctionRef<void(T&)> disposer(_dtor);
				disposer(operator[](_size - 1));
			}
			_size--;
		}
	}
	void clear() {
		if (_size == 0) return;
		if (_dtor) {
			luisa::shader::FunctionRef<void(T&)> disposer(_dtor);
			for (uint64 i = 0; i < _size; ++i) {
				disposer(operator[](i));
			}
		}
		_size = 0;
	}
};
template<typename T>
class vector<T, MemoryType::Temp> {
private:
	uint64 _ptr;
	uint64 _size;
	uint64 _stride;
	uint64 _capacity;
	[[clang::annotate("luisa-shader", "vec_dtor")]] uint64 _dtor;

public:
	[[access]] T& operator[](uint64 index);
	auto ptr() const { return _ptr; }
	auto size() const { return _size; }
	auto capacity() const { return _capacity; }
	vector() {
		_ptr = 0;
		_size = 0;
		_capacity = 0;
		luisa::shader::Ref<T> ref(_ptr);
		if (!luisa::shader::is_trivial(ref.get())) {
			_dtor = (uint64)(&luisa::shader::dispose<T>);
		} else {
			_dtor = 0;
		}
		_stride = sizeof(T);
	}
	void reserve(uint64 dst) {
		if (dst <= _capacity) {
			return;
		}
		auto new_ptr = luisa::shader::temp_malloc(dst * sizeof(T));
		if (_ptr != 0) {
			luisa::shader::memcpy(new_ptr, _ptr, dst * sizeof(T));
		}
		_ptr = new_ptr;
		_capacity = dst;
	}
	void emplace_back(T& t) {
		auto idx = _size;
		_size += 1;
		if (_size > _capacity) {
			reserve((uint64)(double(_capacity) * 1.5) + 8);
		}
		operator[](idx) = t;
	}
	void emplace_back(T t) {
		auto idx = _size;
		_size += 1;
		if (_size > _capacity) {
			reserve((uint64)(double(_capacity) * 1.5) + 8);
		}
		operator[](idx) = t;
	}
	void pop_back() {
		if (_size > 0) {
			if (_dtor) {
				luisa::shader::FunctionRef<void(T&)> disposer(_dtor);
				disposer(operator[](_size - 1));
			}
			_size--;
		}
	}
	void clear() {
		if (_size == 0) return;
		if (_dtor) {
			luisa::shader::FunctionRef<void(T&)> disposer(_dtor);
			for (uint64 i = 0; i < _size; ++i) {
				disposer(operator[](i));
			}
		}
		_size = 0;
	}
};