#pragma once
#include "../Common/MCollection.h"
#include "../Common/DLL.h"
class  Actor final
{
private:
	struct Pointer
	{
		void* ptr;
		void(*disposer)(void*);
		void operator=(Pointer const& p);
		~Pointer();
		Pointer() {};
		Pointer(
			void* ptr,
			void(*disposer)(void*)) :
			ptr(ptr), disposer(disposer) {}
	};
	HashMap<Type, Pointer> hash;
	void* GetComponent(Type t) const;
	void RemoveComponent(Type t);
	void SetComponent(Type t, void* ptr, void(*disposer)(void*));
public:
	Actor();
	Actor(uint32_t initComponentCapacity);
	~Actor();
	template <typename T>
	T* GetComponent() const
	{
		return (T*)GetComponent(typeid(T));
	}
	template <typename T>
	void RemoveComponent()
	{
		RemoveComponent(typeid(T));
	}
	template <typename T>
	void SetComponent(T* ptr)
	{
		SetComponent(
			typeid(T),
			ptr,
			[](void* pp)->void
			{
				delete ((T*)pp);
			}
		);
	}
	KILL_COPY_CONSTRUCT(Actor)
	DECLARE_VENGINE_OVERRIDE_OPERATOR_NEW
};