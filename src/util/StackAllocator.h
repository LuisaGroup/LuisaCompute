#pragma once
#include <util/vstlconfig.h>
#include <stdint.h>
#include <type_traits>
class VENGINE_DLL_COMMON StackBuffer {
public:
	void* data() const { return ptr; }
	size_t length() const {
		return mLength;
	}
	~StackBuffer();
	static StackBuffer Allocate(size_t size);
	static StackBuffer Allocate_Align16(size_t size);
	static StackBuffer Allocate_Align32(size_t size);
	static void* stack_malloc(size_t sz);
	static void stack_free(void* ptr);
	static void* GetCurrentPtr();
	StackBuffer(StackBuffer&& stk);

private:
	StackBuffer(
		void* ptr,
		size_t mLength) : ptr(ptr), mLength(mLength) {}
	StackBuffer(StackBuffer const&) = delete;
	void* ptr;
	size_t mLength;
};
