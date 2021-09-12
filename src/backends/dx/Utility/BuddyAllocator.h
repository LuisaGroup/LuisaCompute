#pragma once
#include <Common/Common.h>
#include <util/Runnable.h>
using uint = uint32_t;
class BuddyLinkedList;
class BuddyBinaryTree;
class BuddyAllocator;
struct BuddyNode
{
	friend class BuddyLinkedList;
	friend class BuddyBinaryTree;
	friend class BuddyAllocator;
private:
	BuddyNode* left;
	BuddyNode* right;
	BuddyNode* parent;
	uint64_t size;
	uint64_t allocatedMem;
	uint layer;
	bool isUsing;
	void* resourceBlock;
public:
	uint64_t GetSize() const { return size; }
	uint64_t GetPosition() const { return allocatedMem; }
	uint GetLayer() const { return layer; }
	template<typename T>
	T* GetBlockResource() const
	{
		return (T*)resourceBlock;
	}
};

class BuddyLinkedList
{

private:
	BuddyNode* firstNode = nullptr;
	uint64_t mSize = 0;
public:
	BuddyLinkedList() noexcept {}
	uint64_t size() const noexcept { return mSize; }
	void Add(BuddyNode* node)noexcept;
	BuddyNode* RemoveHead() noexcept;
	void Remove(BuddyNode* node)noexcept;
	BuddyNode* GetNode() const noexcept;
};
class BuddyBinaryTree
{

private:
	BuddyNode* rootNode = nullptr;
public:
	BuddyBinaryTree(uint64_t initSize, void* ptr, BuddyLinkedList* links, Pool<BuddyNode>* nodePool)noexcept;
	static void Split(uint layer, BuddyLinkedList* links, Pool<BuddyNode>* nodePool)noexcept;
	static void Combine(BuddyNode* parentNode, BuddyLinkedList* links, Pool<BuddyNode>* nodePool)noexcept;
	static void TryCombine(BuddyNode* currentNode, BuddyLinkedList* links, Pool<BuddyNode>* nodePool)noexcept;
};

class  BuddyAllocator
{
	Pool<BuddyNode> nodePool;
	Pool<BuddyBinaryTree> treePool;
	Runnable<void* (uint64_t)> resourceBlockConstruct;
	Runnable<void(void*)> resourceBlockDestructor;
	ArrayList<BuddyLinkedList> linkList;
	uint64_t initSize;
	uint binaryLayer;
	uint treeCapacity;
	ArrayList<std::pair<BuddyBinaryTree*, void*>> allocatedTree;
	static constexpr uint GetPow2(uint i)
	{
		uint sm = 1;
		for (uint a = 0; a < i; ++a)
			sm *= 2;
		return sm;
	}
	BuddyNode* Allocate_T(uint targetLayer);
public:
	BuddyAllocator(uint binaryLayer, uint treeCapacity, uint64_t initSize, Runnable<void* (uint64_t)> const& blockConstructor, Runnable<void(void*)> const& resourceBlockDestructor);
	void Free(BuddyNode* node);
	BuddyNode* Allocate(uint targetLayer);
	~BuddyAllocator();
	VSTL_OVERRIDE_OPERATOR_NEW
};
