#pragma once
#include <luisa/resources.hpp>
#include "base.hpp"
template<uint32 memory_flags = MemoryType::Persist>
class basic_string;

template<>
class basic_string<MemoryType::Persist> {
private:
	[[clang::annotate("luisa-shader", "str_ptr")]] uint64 _ptr;
	uint64 _size;
	uint64 _capacity;

public:
	basic_string() {
		_ptr = 0;
		_size = 0;
		_capacity = 0;
	}
	basic_string(luisa::shader::StringView string_view) {
		_ptr = luisa::shader::persist_malloc(string_view._len);
		_size = string_view._len;
		_capacity = string_view._len;
		luisa::shader::memcpy(_ptr, string_view._ptr, string_view._len);
	}
	luisa::shader::StringView view() {
		luisa::shader::StringView s;
		s._ptr = _ptr;
		s._len = _size;
		return s;
	}
	// TODO: emplace
	void dispose() {
		if (_ptr != 0) {
			luisa::shader::persist_free(_ptr);
		}
	}
};
using string = basic_string<MemoryType::Persist>;