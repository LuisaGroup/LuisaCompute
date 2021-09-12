#pragma once
#include <Utility/BuddyAllocator.h>
#include <Common/LinkedList.h>

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
		AllocateHandle(
			LinkedNode<AllocatedElement>* node
		) : node(node)
		{

		}
	public:
		LinkedNode<AllocatedElement>* node;

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
		[[nodiscard]] operator bool() const noexcept { return node != nullptr; }
	};
private:
	StackObject<BuddyAllocator> buddyAlloc;
	Pool<LinkedNode<AllocatedElement>> bdyNodePool;
	vstd::vector<LinkedList<AllocatedElement>> linkLists;
	size_t maxSize;
public:
	ElementAllocator(
		size_t fullSize,
		Runnable<void* (uint64_t)> const& blockConstructor,
		Runnable<void(void*)> const& resourceBlockDestructor);
	AllocateHandle Allocate(size_t size);
	void Release(AllocateHandle handle);
	~ElementAllocator();

	VSTL_DELETE_COPY_CONSTRUCT(ElementAllocator)
};
