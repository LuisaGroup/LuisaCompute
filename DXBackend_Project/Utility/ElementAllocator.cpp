#include "ElementAllocator.h"
namespace AllocatorMath {
static constexpr void GetBinaryLayer(size_t size, size_t& readCapacity, size_t& layer) {
	size -= 1;
	readCapacity = 1;
	layer = 0;
	while (true) {
		size >>= 1;
		readCapacity <<= 1;
		layer++;
		if (size == 0) {
			return;
		}
	}
}
static constexpr size_t GetLayer(size_t size) {
	size -= 1;
	size_t layer = 0;
	while (true) {
		size >>= 1;
		layer++;
		if (size == 0) {
			return layer;
		}
	}
}
}// namespace AllocatorMath

ElementAllocator::ElementAllocator(
	size_t fullSize,
	Runnable<void*(uint64_t)> const& blockConstructor,
	Runnable<void(void*)> const& resourceBlockDestructor) : bdyNodePool(128) {
	using namespace AllocatorMath;
	size_t layer;
	GetBinaryLayer(
		fullSize, fullSize, layer);
	maxSize = fullSize;
	linkLists.resize(layer);
	buddyAlloc.New(
		layer,
		4,
		fullSize,
		blockConstructor,
		resourceBlockDestructor);
}
ElementAllocator::AllocateHandle ElementAllocator::Allocate(size_t size) {
	using namespace AllocatorMath;
	size_t currentLayer;
	size_t originSize = size;
	GetBinaryLayer(
		size, size, currentLayer);
	if (size == 0 || size > maxSize) {
		throw "Out of Range!";
	}
	currentLayer = linkLists.size() - currentLayer;
	auto setBrother = [originSize](AllocatedElement& ele) -> void {
		if (ele.offset == 0)
			return;
		auto leftedSize = ele.size - originSize;
		ele.brother->obj.size += leftedSize;
		ele.size = originSize;
		ele.offset += leftedSize;
	};
	//First Fit
	LinkedNode<AllocatedElement>* minEle = nullptr;
	size_t smallestSize = -1;
	auto FitElement = [&]() -> LinkedNode<AllocatedElement>* {
		setBrother(minEle->obj);
		minEle->obj.avaliable = false;
		LinkedList<AllocatedElement>::Remove(minEle);
		return minEle;
	};
	{
		LinkedList<AllocatedElement>& linkList = linkLists[currentLayer];
		for (auto ptr = linkList.Begin(); ptr; ptr = ptr->Next()) {
			if (ptr->obj.size >= originSize) {
				if (ptr->obj.size < smallestSize) {
					smallestSize = ptr->obj.size;
					minEle = ptr;
				}
			}
		}
		if (minEle)
			return FitElement();
	}
	if (currentLayer > 0) {
		LinkedList<AllocatedElement>& linkList = linkLists[currentLayer - 1];
		for (auto ptr = linkList.Begin(); ptr; ptr = ptr->Next()) {
			if (ptr->obj.size < smallestSize) {
				smallestSize = ptr->obj.size;
				minEle = ptr;
			}
		}
		if (minEle)
			return FitElement();
	}
	//Buddy Allocate
	BuddyNode* node = buddyAlloc->Allocate(currentLayer);
	LinkedNode<AllocatedElement>* left = bdyNodePool.New();
	size_t lefted = size - originSize;
	LinkedNode<AllocatedElement>* right = nullptr;
	if (lefted > 0) {
		right = bdyNodePool.New();
		right->obj.brother = left;
		right->obj.parentNode = node;
		right->obj.offset = originSize;
		right->obj.size = lefted;
		right->obj.avaliable = true;
		size_t leftLayer = GetLayer(lefted);
		auto&& lst = linkLists[linkLists.size() - leftLayer];
		lst.Add(right);
	}
	left->obj.brother = right;
	left->obj.parentNode = node;
	left->obj.offset = 0;
	left->obj.size = originSize;
	left->obj.avaliable = false;
	return left;
}
void ElementAllocator::Release(ElementAllocator::AllocateHandle handle) {
	using namespace AllocatorMath;
	auto element = handle.node;
	auto brother = element->obj.brother;
	//Shouldn't be here;
	if (handle.node->obj.avaliable) {
		VEngine_Log("Try to dispose a disposed resource!\n");
		throw 0;
	}
	if (brother) {
		if (brother->obj.avaliable) {
			LinkedList<AllocatedElement>::Remove(brother);
			buddyAlloc->Free(element->obj.parentNode);
			bdyNodePool.Delete(brother);
			bdyNodePool.Delete(element);
		} else {
			element->obj.avaliable = true;
			auto layer = GetLayer(element->obj.size);
			auto&& linkList = linkLists[linkLists.size() - layer];
			linkList.Add(element);
		}
	} else {
		buddyAlloc->Free(element->obj.parentNode);
		bdyNodePool.Delete(element);
	}
}
ElementAllocator::~ElementAllocator() {
	buddyAlloc.Delete();
}
