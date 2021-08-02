#pragma once
#include <stdint.h>
#include <core/vstl/Memory.h>
#include <Common/VAllocator.h>

using uint = uint32_t;
template<typename T, bool isPureValueType = false, VEngine_AllocType allocType = VEngine_AllocType::VEngine>
class RandomVector;

template<typename T, VEngine_AllocType allocType>
class RandomVector<T, false, allocType> final {
private:
	std::pair<T, uint*>* arr;
	size_t size;
	size_t capacity;
	VAllocHandle<allocType> allocHandle;
	void* randomvector_malloc(size_t size)
	{
		return allocHandle.Malloc(size);
	}
	void randomvector_free(void* ptr)
	{
		return allocHandle.Free(ptr);
	}

	void Resize(uint newCapacity)
	{
		if (newCapacity <= capacity) return;
		uint maxCapa;
		maxCapa = capacity * 1.5 + 8;

		if (maxCapa > newCapacity)
			newCapacity = maxCapa;
		std::pair<T, uint*>* newArr = (std::pair<T, uint*>*)randomvector_malloc(sizeof(std::pair<T, uint*>) * newCapacity);
		for (uint i = 0; i < size; ++i)
		{
			new (newArr + i)std::pair<T, uint*>(arr[i]);
			arr[i].~pair<T, uint*>();
		}
		if (arr)
			randomvector_free(arr);
		arr = newArr;
		capacity = newCapacity;
	}
public:
	RandomVector(RandomVector<T, false, allocType>&& o)
		: arr(o.arr),
		  size(o.size),
		  capacity(o.capacity) {
		o.arr = nullptr;
	}
	std::pair<T, uint*>* GetData() const
	{
		return arr;
	}
	size_t Length() const {
		return size;
	}
	void Clear()
	{
		for (uint i = 0; i < size; ++i)
		{
			arr[i].~pair<T, uint*>();
		}
		size = 0;
	}
	RandomVector(uint capacity) :
		capacity(capacity),
		size(0)
	{
		arr = (std::pair<T, uint*>*)randomvector_malloc(sizeof(std::pair<T, uint*>) * capacity);
		memset(arr, 0, sizeof(std::pair<T, uint*>) * capacity);
	}

	void Reserve(uint newCapacity)
	{
		if (newCapacity <= capacity) return;
		std::pair<T, uint*>* newArr = (std::pair<T, uint*>*)randomvector_malloc(sizeof(std::pair<T, uint*>) * newCapacity);
		for (uint i = 0; i < size; ++i)
		{
			new (newArr + i)std::pair<T, uint*>(arr[i]);
			arr[i].~pair<T, uint*>();
		}
		if (arr)
			randomvector_free(arr);
		arr = newArr;
		capacity = newCapacity;
	}

	RandomVector() :
		capacity(0),
		size(0),
		arr(nullptr)
	{

	}
	T& operator[](uint index)
	{
#ifdef NDEBUG
		return arr[index].first;
#else
		if (index >= size) throw "Index Out of Range!";
		return arr[index].first;
#endif
	}
	void Add(const T& value, uint* indexFlagPtr)
	{
		Resize(size + 1);
		*indexFlagPtr = size;
		size++;
		auto& a = arr[*indexFlagPtr];
		new (&a)std::pair<T, uint*>(value, indexFlagPtr);
	}
	void Remove(uint targetIndex)
	{
		if (targetIndex >= size) throw "Index Out of Range!";
		size--;
		if (targetIndex != size) {
			arr[targetIndex] = arr[size];
			*arr[targetIndex].second = targetIndex;
		}
		arr[size].~pair<T, uint*>();

	}

	~RandomVector()
	{
		for (uint i = 0; i < size; ++i)
		{
			arr[i].~pair<T, uint*>();
		}
		if (arr)
			randomvector_free(arr);
	}
};

template<typename T, VEngine_AllocType allocType>
class RandomVector<T, true, allocType> final {
private:
	std::pair<T, uint*>* arr;
	size_t size;
	size_t capacity;
	VAllocHandle<allocType> allocHandle;
	void* randomvector_malloc(size_t size) {
		return allocHandle.Malloc(size);
	}
	void randomvector_free(void* ptr) {
		return allocHandle.Free(ptr);
	}

	void Resize(uint newCapacity)
	{
		if (newCapacity <= capacity) return;
		uint maxCapa;
		maxCapa = capacity * 1.5 + 8;

		if (maxCapa > newCapacity)
			newCapacity = maxCapa;
		std::pair<T, uint*>* newArr = (std::pair<T, uint*>*)randomvector_malloc(sizeof(std::pair<T, uint*>) * newCapacity);
		if (arr)
		{
			memcpy(newArr, arr, sizeof(std::pair<T, uint*>) * size);
			randomvector_free(arr);
		}
		arr = newArr;
		capacity = newCapacity;
	}
public:
	RandomVector(RandomVector<T, true, allocType>&& o)
		: arr(o.arr),
		  size(o.size),
		  capacity(o.capacity) {
		o.arr = nullptr;
		o.size = 0;
		o.capacity = 0;
	}
	std::pair<T, uint*>* GetData() const
	{
		return arr;
	}
	size_t Length() const {
		return size;
	}
	void Clear()
	{
		size = 0;
	}
	RandomVector(uint capacity) :
		capacity(capacity),
		size(0)
	{
		arr = (std::pair<T, uint*>*)randomvector_malloc(sizeof(std::pair<T, uint*>) * capacity);
		memset(arr, 0, sizeof(std::pair<T, uint*>) * capacity);
	}

	void Reserve(uint newCapacity)
	{
		if (newCapacity <= capacity) return;
		std::pair<T, uint*>* newArr = (std::pair<T, uint*>*)randomvector_malloc(sizeof(std::pair<T, uint*>) * newCapacity);
		if (arr)
		{
			memcpy(newArr, arr, sizeof(std::pair<T, uint*>) * size);
			randomvector_free(arr);
		}
		arr = newArr;
		capacity = newCapacity;
	}

	RandomVector() :
		capacity(0),
		size(0),
		arr(nullptr)
	{

	}
	T& operator[](uint index)
	{
#ifdef NDEBUG
		return arr[index].first;
#else
		if (index >= size) throw "Index Out of Range!";
		return arr[index].first;
#endif
	}
	void Add(const T& value, uint* indexFlagPtr)
	{
		Resize(size + 1);
		*indexFlagPtr = size;
		size++;
		auto& a = arr[*indexFlagPtr];
		new (&a)std::pair<T, uint*>(value, indexFlagPtr);
	}
	void Remove(uint targetIndex)
	{
		if (targetIndex >= size) throw "Index Out of Range!";
		size--;
		if (targetIndex != size) {
			arr[targetIndex] = arr[size];
			*arr[targetIndex].second = targetIndex;
		}
	}

	~RandomVector()
	{
		if (arr)
			randomvector_free(arr);
	}
};
