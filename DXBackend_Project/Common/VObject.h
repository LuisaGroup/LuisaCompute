#pragma once
#include <atomic>
#include <mutex>
#include "Runnable.h"
#include "MetaLib.h"
#include <assert.h>
#include "vector.h"
#include "Memory.h"
class PtrLink;

class VObject {
	friend class PtrLink;

private:
	vengine::vector<Runnable<void(VObject*)>> disposeFuncs;
	static std::atomic<uint64_t> CurrentID;
	uint64_t instanceID;

protected:
	VObject() {
		instanceID = ++CurrentID;
	}

public:
	Type GetType() const noexcept {
		return typeid(*const_cast<VObject*>(this));
	}
	void AddEventBeforeDispose(Runnable<void(VObject*)>&& func) noexcept;
	uint64_t GetInstanceID() const noexcept { return instanceID; }
	virtual ~VObject() noexcept;
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
	KILL_COPY_CONSTRUCT(VObject)
};

class PtrLink;
class PtrWeakLink;
struct LinkHeap {
	friend class PtrLink;
	friend class PtrWeakLink;

private:
	static void Resize() noexcept;
	funcPtr_t<void(void*)> disposer;
	std::atomic<int32_t> refCount;
	std::atomic<int32_t> looseRefCount;
	static ArrayList<LinkHeap*, false> heapPtrs;
	static spin_mutex mtx;
	static LinkHeap* GetHeap(void* obj, void (*disp)(void*)) noexcept;
	static void ReturnHeap(LinkHeap* value) noexcept;

public:
	void* ptr;
};

class VEngine;
class PtrWeakLink;
class PtrLink {
	friend class VEngine;
	friend class PtrWeakLink;

public:
	LinkHeap* heapPtr;
	size_t offset = 0;
	PtrLink() noexcept : heapPtr(nullptr) {
	}
	void Dispose() noexcept;
	template<typename T>
	PtrLink(T* obj, funcPtr_t<void(void*)> disposer) noexcept {
		heapPtr = LinkHeap::GetHeap(obj, disposer);
	}

	PtrLink(const PtrLink& p) noexcept;
	PtrLink(PtrLink&& p) noexcept;
	void operator=(const PtrLink& p) noexcept;
	void operator=(PtrLink&& p) noexcept;
	PtrLink(const PtrWeakLink& link) noexcept;
	PtrLink(PtrWeakLink&& link) noexcept;

	void Destroy() noexcept;
	~PtrLink() noexcept {
		Dispose();
	}
};
class PtrWeakLink {
public:
	LinkHeap* heapPtr;
	size_t offset = 0;
	PtrWeakLink() noexcept : heapPtr(nullptr) {
	}

	void Dispose() noexcept;
	PtrWeakLink(const PtrLink& p) noexcept;
	PtrWeakLink(const PtrWeakLink& p) noexcept;
	PtrWeakLink(PtrWeakLink&& p) noexcept;//TODO
	void operator=(const PtrLink& p) noexcept;
	void operator=(const PtrWeakLink& p) noexcept;
	void operator=(PtrWeakLink&& p) noexcept;//TODO
	void Destroy() noexcept;

	~PtrWeakLink() noexcept {
		Dispose();
	}
};

template<typename T>
class ObjWeakPtr;
template<typename T>
class ObjectPtr;
template<typename T>
class ObjectPtr {
private:
	friend class ObjWeakPtr<T>;
	PtrLink link;
	inline ObjectPtr(T* ptr, funcPtr_t<void(void*)> disposer) noexcept : link(ptr, disposer) {
	}
	T* GetPtr() const noexcept {
		return reinterpret_cast<T*>(reinterpret_cast<size_t>(link.heapPtr->ptr) + link.offset);
	}

public:
	ObjectPtr(const PtrLink& link, size_t addOffset) noexcept : link(link) {
		this->link.offset += addOffset;
	}
	ObjectPtr(PtrLink&& link, size_t addOffset) noexcept : link(std::move(link)) {
		this->link.offset += addOffset;
	}
	inline ObjectPtr() noexcept : link() {}
	inline ObjectPtr(std::nullptr_t) noexcept : link() {
	}
	inline ObjectPtr(const ObjectPtr<T>& ptr) noexcept : link(ptr.link) {
	}
	inline ObjectPtr(ObjectPtr<T>&& ptr) noexcept : link(std::move(ptr.link)) {
	}
	inline ObjectPtr(const ObjWeakPtr<T>& ptr) noexcept;
	inline ObjectPtr(ObjWeakPtr<T>&& ptr) noexcept;
	static ObjectPtr<T> MakePtr(T* ptr) noexcept {
		return ObjectPtr<T>(ptr, [](void* ptr) -> void {
			delete (reinterpret_cast<T*>(ptr));
		});
	}
	static ObjectPtr<T> MakePtr(T* ptr, funcPtr_t<void(void*)> disposer) noexcept {
		return ObjectPtr<T>(ptr, disposer);
	}
	template<typename... Args>
	static ObjectPtr<T> NewObject(Args&&... args) {
		T* ptr = vengine_new<T>(std::forward<Args>(args)...);
		return ObjectPtr<T>(ptr, [](void* ptr) -> void {
			vengine_delete<T>(reinterpret_cast<T*>(ptr));
		});
	}
	static ObjectPtr<T> MakePtrNoMemoryFree(T* ptr) noexcept {
		return ObjectPtr<T>(ptr, [](void* ptr) -> void {
			if (std::is_trivially_destructible_v<T>)
				(reinterpret_cast<T*>(ptr))->~T();
		});
	}
	static ObjectPtr<T> MakePtr(ObjectPtr<T>) noexcept = delete;

	inline operator bool() const noexcept {
		return link.heapPtr != nullptr && link.heapPtr->ptr != nullptr;
	}

	inline operator T*() const noexcept {
#ifdef _DEBUG
		//Null Check!
		assert(link.heapPtr != nullptr);
#endif
		return GetPtr();
	}

	inline void Destroy() noexcept {
		link.Destroy();
	}

	template<typename F>
	inline ObjectPtr<F> CastTo() const noexcept {
		static T* const ptr = reinterpret_cast<T*>(0x7fffffff);
		static F* const fPtr = static_cast<F*>(ptr);
		static size_t const ptrOffset = (size_t)fPtr - (size_t)ptr;
		return ObjectPtr<F>(link, ptrOffset);
	}
	template<typename F>
	inline ObjectPtr<F> Reinterpret_CastTo() const noexcept {
		return ObjectPtr<F>(link, 0);
	}
	inline void operator=(const ObjWeakPtr<T>& other) noexcept;
	inline void operator=(const ObjectPtr<T>& other) noexcept {
		link = other.link;
	}
	inline void operator=(ObjectPtr<T>&& other) noexcept {
		link = std::move(other.link);
	}
	inline void operator=(T* other) noexcept = delete;
	inline void operator=(void* other) noexcept = delete;
	inline void operator=(std::nullptr_t t) noexcept {
		link.Dispose();
	}

	inline T* operator->() const noexcept {
#ifdef _DEBUG
		//Null Check!
		assert(link.heapPtr != nullptr);
#endif
		return GetPtr();
	}

	inline T& operator*() noexcept {
#ifdef _DEBUG
		//Null Check!
		assert(link.heapPtr != nullptr);
#endif
		return *GetPtr();
	}

	inline T const& operator*() const noexcept {
#ifdef _DEBUG
		//Null Check!
		assert(link.heapPtr != nullptr);
#endif
		return *GetPtr();
	}

	inline bool operator==(const ObjectPtr<T>& ptr) const noexcept {
		return link.heapPtr == ptr.link.heapPtr;
	}
	inline bool operator!=(const ObjectPtr<T>& ptr) const noexcept {
		return link.heapPtr != ptr.link.heapPtr;
	}
};

template<typename T>
class ObjectPtr<T[]> {

private:
	friend class ObjWeakPtr<T[]>;
	PtrLink link;
	inline ObjectPtr(T* ptr, funcPtr_t<void(void*)> disposer) noexcept : link(ptr, disposer) {
	}
	T* GetPtr() const noexcept {
		return reinterpret_cast<T*>(reinterpret_cast<size_t>(link.heapPtr->ptr) + link.offset);
	}

public:
	ObjectPtr(const PtrLink& link, size_t addOffset) noexcept : link(link) {
		this->link.offset += addOffset;
	}
	ObjectPtr(PtrLink&& link, size_t addOffset) noexcept : link(std::move(link)) {
		this->link.offset += addOffset;
	}
	inline ObjectPtr() noexcept : link() {}
	inline ObjectPtr(std::nullptr_t) noexcept : link() {
	}
	inline ObjectPtr(const ObjectPtr<T[]>& ptr) noexcept : link(ptr.link) {
	}
	inline ObjectPtr(ObjectPtr<T[]>&& ptr) noexcept : link(std::move(ptr.link)) {
	}
	inline ObjectPtr(const ObjWeakPtr<T[]>& ptr) noexcept;
	static ObjectPtr<T[]> MakePtr(T* ptr) noexcept {
		return ObjectPtr<T[]>(ptr, [](void* ptr) -> void {
			delete[](reinterpret_cast<T*>(ptr));
		});
	}
	static ObjectPtr<T[]> MakePtr(T* ptr, funcPtr_t<void(void*)> disposer) noexcept {
		return ObjectPtr<T[]>(ptr, disposer);
	}
	static ObjectPtr<T[]> MakePtr(ObjectPtr<T[]>) noexcept = delete;

	inline operator bool() const noexcept {
		return link.heapPtr != nullptr && link.heapPtr->ptr != nullptr;
	}

	inline operator T*() const noexcept {
#ifdef _DEBUG
		//Null Check!
		assert(link.heapPtr != nullptr);
#endif
		return GetPtr();
	}

	inline void Destroy() noexcept {
		link.Destroy();
	}

	template<typename F>
	inline ObjectPtr<F[]> Reinterpret_CastTo() const noexcept {
		return ObjectPtr<F[]>(link, 0);
	}
	inline void operator=(const ObjWeakPtr<T[]>& other) noexcept;
	inline void operator=(const ObjectPtr<T[]>& other) noexcept {
		link = other.link;
	}
	inline void operator=(ObjectPtr<T[]>&& other) noexcept {
		link = std::move(other.link);
	}

	inline void operator=(T* other) noexcept = delete;
	inline void operator=(void* other) noexcept = delete;
	inline void operator=(std::nullptr_t t) noexcept {
		link.Dispose();
	}

	inline T* operator->() const noexcept {
#ifdef _DEBUG
		//Null Check!
		assert(link.heapPtr != nullptr);
#endif
		return GetPtr();
	}

	inline T& operator*() noexcept {
#ifdef _DEBUG
		//Null Check!
		assert(link.heapPtr != nullptr);
#endif
		return *GetPtr();
	}

	inline T const& operator*() const noexcept {
#ifdef _DEBUG
		//Null Check!
		assert(link.heapPtr != nullptr);
#endif
		return *GetPtr();
	}
	inline T& operator[](uint64_t key) noexcept {
		return GetPtr()[key];
	}

	inline T const& operator[](uint64_t key) const noexcept {
		return GetPtr()[key];
	}

	inline bool operator==(const ObjectPtr<T[]>& ptr) const noexcept {
		return link.heapPtr == ptr.link.heapPtr;
	}
	inline bool operator!=(const ObjectPtr<T[]>& ptr) const noexcept {
		return link.heapPtr != ptr.link.heapPtr;
	}
};

template<typename T>
class ObjWeakPtr {
private:
	friend class ObjectPtr<T>;
	PtrWeakLink link;
	T* GetPtr() const noexcept {
		return reinterpret_cast<T*>(reinterpret_cast<size_t>(link.heapPtr->ptr) + link.offset);
	}

public:
	inline ObjWeakPtr() noexcept : link() {}
	inline ObjWeakPtr(std::nullptr_t) noexcept : link() {
	}
	inline ObjWeakPtr(const ObjWeakPtr<T>& ptr) noexcept : link(ptr.link) {
	}
	inline ObjWeakPtr(ObjWeakPtr<T>&& ptr) noexcept : link(std::move(ptr.link)) {
	}
	inline ObjWeakPtr(const ObjectPtr<T>& ptr) noexcept : link(ptr.link) {
	}
	ObjWeakPtr(const PtrWeakLink& link, size_t addOffset) noexcept : link(link) {
		this->link.offset += addOffset;
	}
	ObjWeakPtr(PtrWeakLink&& link, size_t addOffset) noexcept : link(std::move(link)) {
		this->link.offset += addOffset;
	}

	inline operator bool() const noexcept {
		return link.heapPtr != nullptr && link.heapPtr->ptr != nullptr;
	}

	inline operator T*() const noexcept {
#ifdef _DEBUG
		//Null Check!
		assert(link.heapPtr != nullptr);
#endif
		return GetPtr();
	}

	inline void Destroy() noexcept {
		link.Destroy();
	}

	template<typename F>
	inline ObjWeakPtr<F> CastTo() const noexcept {
		static T* const ptr = reinterpret_cast<T*>(0x7fffffff);
		static F* const fPtr = static_cast<F*>(ptr);
		static size_t const ptrOffset = (size_t)fPtr - (size_t)ptr;
		return ObjWeakPtr<F>(link, ptrOffset);
	}
	template<typename F>
	inline ObjWeakPtr<F> Reinterpret_CastTo() const noexcept {
		return ObjWeakPtr<F>(link, 0);
	}
	inline void operator=(const ObjWeakPtr<T>& other) noexcept {
		link = other.link;
	}
	inline void operator=(ObjWeakPtr<T>&& other) noexcept {
		link = std::move(other.link);
	}

	inline void operator=(const ObjectPtr<T>& other) noexcept {
		link = other.link;
	}

	inline void operator=(T* other) noexcept = delete;
	inline void operator=(void* other) noexcept = delete;
	inline void operator=(std::nullptr_t t) noexcept {
		link.Dispose();
	}

	inline T* operator->() const noexcept {
#ifdef _DEBUG
		//Null Check!
		assert(link.heapPtr != nullptr);
#endif
		return GetPtr();
	}

	inline T& operator*() noexcept {
#ifdef _DEBUG
		//Null Check!
		assert(link.heapPtr != nullptr);
#endif
		return *GetPtr();
	}

	inline T const& operator*() const noexcept {
#ifdef _DEBUG
		//Null Check!
		assert(link.heapPtr != nullptr);
#endif
		return *GetPtr();
	}

	inline bool operator==(const ObjWeakPtr<T>& ptr) const noexcept {
		return link.heapPtr == ptr.link.heapPtr;
	}
	inline bool operator!=(const ObjWeakPtr<T>& ptr) const noexcept {
		return link.heapPtr != ptr.link.heapPtr;
	}
};

template<typename T>
class ObjWeakPtr<T[]> {
private:
	friend class ObjectPtr<T[]>;
	PtrWeakLink link;
	T* GetPtr() const noexcept {
		return reinterpret_cast<T*>(reinterpret_cast<size_t>(link.heapPtr->ptr) + link.offset);
	}

public:
	inline ObjWeakPtr() noexcept : link() {}
	inline ObjWeakPtr(std::nullptr_t) noexcept : link() {
	}
	inline ObjWeakPtr(const ObjWeakPtr<T[]>& ptr) noexcept : link(ptr.link) {
	}
	inline ObjWeakPtr(ObjWeakPtr<T[]>&& ptr) noexcept : link(std::move(ptr.link)) {
	}
	inline ObjWeakPtr(const ObjectPtr<T[]>& ptr) noexcept : link(ptr.link) {
	}
	ObjWeakPtr(const PtrWeakLink& link, size_t addOffset) noexcept : link(link) {
		this->link.offset += addOffset;
	}
	ObjWeakPtr(PtrWeakLink&& link, size_t addOffset) noexcept : link(std::move(link)) {
		this->link.offset += addOffset;
	}

	inline operator bool() const noexcept {
		return link.heapPtr != nullptr && link.heapPtr->ptr != nullptr;
	}

	inline operator T*() const noexcept {
#ifdef _DEBUG
		//Null Check!
		assert(link.heapPtr != nullptr);
#endif
		return GetPtr();
	}

	inline void Destroy() noexcept {
		link.Destroy();
	}

	template<typename F>
	inline ObjWeakPtr<F[]> Reinterpret_CastTo() const noexcept {
		return ObjWeakPtr<F[]>(link, 0);
	}
	inline void operator=(const ObjWeakPtr<T[]>& other) noexcept {
		link = other.link;
	}
	inline void operator=(ObjWeakPtr<T[]>&& other) noexcept {
		link = std::move(other.link);
	}

	inline void operator=(const ObjectPtr<T[]>& other) noexcept {
		link = other.link;
	}

	inline void operator=(T* other) noexcept = delete;
	inline void operator=(void* other) noexcept = delete;
	inline void operator=(std::nullptr_t t) noexcept {
		link.Dispose();
	}

	inline T* operator->() const noexcept {
#ifdef _DEBUG
		//Null Check!
		assert(link.heapPtr != nullptr);
#endif
		return GetPtr();
	}

	inline T& operator*() noexcept {
#ifdef _DEBUG
		//Null Check!
		assert(link.heapPtr != nullptr);
#endif
		return *GetPtr();
	}
	inline T const& operator*() const noexcept {
#ifdef _DEBUG
		//Null Check!
		assert(link.heapPtr != nullptr);
#endif
		return *GetPtr();
	}
	inline T& operator[](uint64_t key) noexcept {
		return GetPtr()[key];
	}

	inline T const& operator[](uint64_t key) const noexcept {
		return GetPtr()[key];
	}
	inline bool operator==(const ObjWeakPtr<T[]>& ptr) const noexcept {
		return link.heapPtr == ptr.link.heapPtr;
	}
	inline bool operator!=(const ObjWeakPtr<T[]>& ptr) const noexcept {
		return link.heapPtr != ptr.link.heapPtr;
	}
};
template<typename T>
inline ObjectPtr<T>::ObjectPtr(const ObjWeakPtr<T>& ptr) noexcept : link(ptr.link) {
}
template<typename T>
inline ObjectPtr<T>::ObjectPtr(ObjWeakPtr<T>&& ptr) noexcept : link(std::move(ptr.link)) {
}
template<typename T>
inline void ObjectPtr<T>::operator=(const ObjWeakPtr<T>& other) noexcept {
	link = other.link;
}

template<typename T>
inline ObjectPtr<T[]>::ObjectPtr(const ObjWeakPtr<T[]>& ptr) noexcept : link(ptr.link) {
}
template<typename T>
inline void ObjectPtr<T[]>::operator=(const ObjWeakPtr<T[]>& other) noexcept {
	link = other.link;
}

template<typename T>
inline static ObjectPtr<T> MakeObjectPtr(T* ptr) noexcept {
	return ObjectPtr<T>::MakePtr(ptr);
}

template<typename T>
inline static ObjectPtr<T> MakeObjectPtr(T* ptr, funcPtr_t<void(void*)> disposer) noexcept {
	return ObjectPtr<T>::MakePtr(ptr, disposer);
}