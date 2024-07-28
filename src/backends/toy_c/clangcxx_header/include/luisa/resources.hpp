#pragma once
#include "types.hpp"
#include "functions/custom.hpp"
// TODO
namespace luisa::shader {
template<typename T>
struct Ref {
	uint64 _ptr;
	Ref() {
		_ptr = 0;
	}
	Ref(uint64 ptr) {
		_ptr = ptr;
	}
	template<typename Dst>
	Ref<Dst> cast_to() {
		return Ref<Dst>(_ptr);
	}
	[[access]] T& get();
	[[access]] T& operator*();
};
template<typename T>
struct BufferView {
	uint64 _ptr;
	uint64 _len;
	BufferView() {
		_ptr = 0;
		_len = 0;
	}
	BufferView(uint64 ptr, uint64 len) {
		_ptr = ptr;
		_len = len;
	}
	template<typename Dst>
	BufferView<Dst> cast_to() {
		return BufferView<Dst>(_ptr, _len * sizeof(T) / sizeof(Dst));
	}
	[[access]] T& operator[](uint64 index);
};
struct StringView {
	[[clang::annotate("luisa-shader", "strview_ptr")]] uint64 _ptr;
	uint64 _len;
	StringView() {
		_ptr = 0;
		_len = 0;
	}
	StringView(uint64 ptr, uint64 len) {
		_ptr = ptr;
		_len = len;
	}
	[[access]] int8& operator[](uint64 index);
};
template<typename FuncType, bool is_invocable = true>
trait FunctionRef;
template<typename FuncType, bool is_invocable = true>
trait FunctorRef;
template<typename Ret, typename... Args>
struct FunctionRef<Ret(Args...), true> {
	uint64 ptr;
	FunctionRef(uint64 value = 0) : ptr(value) {}
	Ret operator()(Args... args) {
		return luisa::shader::detail::__lc_builtin_invoke__<Ret>(ptr, args...);
	}
};
template<typename Ret, typename... Args>
struct FunctorRef<Ret(Args...), true> {
	uint64 usr_data;
	uint64 ptr;
	FunctorRef(uint64 usr_data = 0, uint64 value = 0) : usr_data(usr_data), ptr(value) {}
	Ret operator()(Args... args) {
		return luisa::shader::detail::__lc_builtin_invoke__<Ret>(ptr, usr_data, args...);
	}
};
template <bool value = true>
struct Finalizer_t;
template <>
struct Finalizer_t<true> {
	[[clang::annotate("luisa-shader", "finalizer")]] uint64 ptr;
	Finalizer_t(uint64 value = 0) {
		ptr = value;
	}
	void operator()(uint64 pointer) {
		luisa::shader::detail::__lc_builtin_invoke__<void>(ptr, pointer);
	}
};
using Finalizer = Finalizer_t<>;
template<typename T>
[[ext_call("rtti_call")]] void rtti_call(FunctorRef<void(StringView, uint64)> callback, T& t);

#define BIND_FUNCTION(FUNCTYPE, x) luisa::shader::FunctionRef<FUNCTYPE, luisa::shader::invocable_v<decltype(x), FUNCTYPE>>((uint64)(&(x)))
#define BIND_FINALIZER(T, x) luisa::shader::Finalizer_t<luisa::shader::invocable_v<decltype(x), void(T&)>>((uint64)(&(x)))
#define BIND_FUNCTOR(FUNCTYPE, usr_data, x) luisa::shader::FunctorRef<FUNCTYPE, luisa::shader::functor_invocable_v<decltype(x), FUNCTYPE>>(usr_data, (uint64)(&(x)))
template<typename T>
Ref<T> temp_new() {
	return Ref<T>(temp_malloc(sizeof(T)));
}
template<typename T>
BufferView<T> temp_new_buffer(uint64 size) {
	return BufferView<T>(temp_malloc(sizeof(T) * size), size);
}
template<typename T>
BufferView<T> persist_new_buffer(uint64 size) {
	return BufferView<T>(persist_malloc(sizeof(T) * size), size);
}
template<typename T>
Ref<T> persist_new() {
	return Ref<T>(persist_malloc(sizeof(T)));
}
template<typename T>
void persist_delete(Ref<T> ref) {
	persist_delete<T>(ref._ptr);
}
template<typename T>
void persist_delete(BufferView<T> ref) {
	persist_delete<T>(ref._ptr);
}
template<concepts::string_literal Str>
[[ext_call("to_string")]] StringView to_strview(Str&& fmt);

}// namespace luisa::shader