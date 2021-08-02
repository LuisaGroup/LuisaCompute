#pragma once
#include "MetaLib.h"
namespace vstd {
template<typename T>
class unique_ptr {
	template<typename B>
	friend class unique_ptr;

private:
	T* ptr;
	void* originPtr;
	funcPtr_t<void(void*)> deleter;
	unique_ptr(
		T* ptr,
		void* originPtr,
		funcPtr_t<void(void*)> deleter)
		: ptr(ptr),
		  originPtr(originPtr),
		  deleter(deleter) {
	}

public:
	using SelfT = unique_ptr<T>;
	unique_ptr() noexcept
		: ptr(nullptr),
		  originPtr(nullptr),
		  deleter(nullptr) {}
	unique_ptr(std::nullptr_t) noexcept : unique_ptr() {}
	unique_ptr(T* ptr, funcPtr_t<void(void*)> disposer) : ptr(ptr), originPtr(ptr), deleter(disposer) {}
	unique_ptr(T* ptr) noexcept
		: ptr(ptr), originPtr(ptr),
		  deleter([](void* ptr) {
			  delete reinterpret_cast<T*>(ptr);
		  }) {}
	unique_ptr(SelfT const&) = delete;
	unique_ptr(SelfT&) = delete;
	unique_ptr(SelfT const&&) = delete;
	unique_ptr(SelfT&& o) noexcept
		: ptr(o.ptr),
		  originPtr(o.originPtr),
		  deleter(o.deleter) {
		o.ptr = nullptr;
		o.originPtr = nullptr;
	}
	template<typename... Args>
	SelfT& operator=(Args&&... args) {
		this->~unique_ptr();
		new (this) SelfT(std::forward<Args>(args)...);
		return *this;
	}
	template<typename F>
	unique_ptr<F> cast_to() && noexcept {
		auto disp = vstd::create_disposer([&]() {
			ptr = nullptr;
			originPtr = nullptr;
		});
		return unique_ptr<F>(
			static_cast<F*>(ptr),
			originPtr,
			deleter);
	}
	T* get() const {
		return ptr;
	}
	T* operator->() const {
		return ptr;
	}
	T& operator*() const {
		return *ptr;
	}

	void reset() {
		if (originPtr) {
			deleter(originPtr);
		}
		ptr = nullptr;
		originPtr = nullptr;
	}
	~unique_ptr() {
		if (originPtr) {
			deleter(originPtr);
		}
	}
};
}// namespace vstd