#pragma once
#include "BuddyAllocator.h"
#include "../Common/LinkedList.h"

class  ElementAllocator
{
public:
	struct AllocatedElement
	{
		BuddyNode* parentNode;
		LinkedNode<AllocatedElement>* brother;
		uint64_t offset;
		uint64_t size;
		bool avaliable;
	};
	struct AllocateHandle
	{
		friend class ElementAllocator;

	private:
		LinkedNode<AllocatedElement>* node;
		AllocateHandle(
			LinkedNode<AllocatedElement>* node
		) : node(node)
		{

		}
	public:
		AllocateHandle(AllocateHandle const& ele)
		{
			node = ele.node;
		}
		void operator=(AllocateHandle const& ele)
		{
			node = ele.node;
		}
		AllocateHandle() : node(nullptr) {}
		template<typename T>
		T* GetBlockResource() const
		{
			return node->obj.parentNode->GetBlockResource<T>();
		}
		uint64_t GetPosition() const
		{
			return node->obj.offset + node->obj.parentNode->GetPosition();
		}
		operator bool() const
		{
			return node;
		}
	};
private:
	StackObject<BuddyAllocator> buddyAlloc;
	Pool<LinkedNode<AllocatedElement>> bdyNodePool;
	vengine::vector<LinkedList<AllocatedElement>> linkLists;
	size_t maxSize;
public:
	ElementAllocator(
		size_t fullSize,
		Runnable<void* (uint64_t)> const& blockConstructor,
		Runnable<void(void*)> const& resourceBlockDestructor);
	AllocateHandle Allocate(size_t size);
	void Release(AllocateHandle handle);
	~ElementAllocator();

	KILL_COPY_CONSTRUCT(ElementAllocator)
};
