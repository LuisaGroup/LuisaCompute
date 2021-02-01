#include "BuddyAllocator.h"
void BuddyLinkedList::Add(BuddyNode* node) noexcept {
	node->left = nullptr;
	node->right = firstNode;
	if (firstNode)
		firstNode->left = node;
	firstNode = node;
	mSize++;
}
BuddyNode* BuddyLinkedList::RemoveHead() noexcept {
	auto temp = firstNode;
	firstNode = firstNode->right;
	if (firstNode)
		firstNode->left = nullptr;
	mSize--;
	return temp;
}
void BuddyLinkedList::Remove(BuddyNode* node) noexcept {
	if (node->right) {
		node->right->left = node->left;
	}
	if (node->left) {
		node->left->right = node->right;
	}
	if (firstNode == node) {
		firstNode = node->right;
	}
	mSize--;
}
BuddyNode* BuddyLinkedList::GetNode() const noexcept {
	return firstNode;
}
BuddyBinaryTree::BuddyBinaryTree(uint64_t initSize, void* ptr, BuddyLinkedList* links, Pool<BuddyNode>* nodePool) noexcept {
	rootNode = nodePool->New();
	memset(rootNode, 0, sizeof(BuddyNode));
	rootNode->size = initSize;
	rootNode->resourceBlock = ptr;
	links[0].Add(rootNode);
}
void BuddyBinaryTree::Split(uint layer, BuddyLinkedList* links, Pool<BuddyNode>* nodePool) noexcept {
	auto targetNode = links[layer].RemoveHead();
	targetNode->isUsing = true;
	BuddyNode* leftNode = nodePool->New();
	leftNode->size = targetNode->size / 2;
	leftNode->layer = layer + 1;
	leftNode->allocatedMem = targetNode->allocatedMem;
	leftNode->parent = targetNode;
	leftNode->isUsing = false;
	leftNode->resourceBlock = targetNode->resourceBlock;
	BuddyNode* rightNode = nodePool->New();
	memcpy(rightNode, leftNode, sizeof(BuddyNode));
	rightNode->allocatedMem += rightNode->size;
	auto&& lk = links[leftNode->layer];
	lk.Add(leftNode);
	lk.Add(rightNode);
	targetNode->left = leftNode;
	targetNode->right = rightNode;
}
void BuddyBinaryTree::Combine(BuddyNode* parentNode, BuddyLinkedList* links, Pool<BuddyNode>* nodePool) noexcept {
	auto&& sonLink = links[parentNode->left->layer];
	sonLink.Remove(parentNode->left);
	sonLink.Remove(parentNode->right);
	nodePool->Delete(parentNode->left);
	nodePool->Delete(parentNode->right);
	parentNode->isUsing = false;
	links[parentNode->layer].Add(parentNode);
}
void BuddyBinaryTree::TryCombine(BuddyNode* currentNode, BuddyLinkedList* links, Pool<BuddyNode>* nodePool) noexcept {
	while (currentNode->parent != nullptr) {
		currentNode = currentNode->parent;
		if (currentNode->left->isUsing || currentNode->right->isUsing) return;
		Combine(currentNode, links, nodePool);
	}
}
BuddyNode* BuddyAllocator::Allocate_T(uint targetLayer) {
	auto&& curLink = linkList[targetLayer];
	if (curLink.size()) {
		return curLink.RemoveHead();
	} else {
		void* ptr;
		//Split
		uint i = targetLayer;
		while (i > 0) {
			i--;
			if (linkList[i].size() > 0)
				goto ADD_NEW_BINARY;
		}
		ptr = resourceBlockConstruct(initSize);
		allocatedTree.push_back({treePool.New(initSize, ptr, linkList.data(), &nodePool), ptr});
		//Add New Binary
	ADD_NEW_BINARY:
		for (; i < targetLayer; ++i) {
			BuddyBinaryTree::Split(i, linkList.data(), &nodePool);
		}
		return curLink.RemoveHead();
	}
}
BuddyAllocator::BuddyAllocator(uint binaryLayer, uint treeCapacity, uint64_t initSize, Runnable<void*(uint64_t)> const& blockConstructor, Runnable<void(void*)> const& resourceBlockDestructor)
	: nodePool((GetPow2(binaryLayer) - 1) * treeCapacity),
	  initSize(initSize),
	  treePool(treeCapacity),
	  linkList(binaryLayer),
	  binaryLayer(binaryLayer),
	  treeCapacity(treeCapacity),
	  resourceBlockConstruct(blockConstructor),
	  resourceBlockDestructor(resourceBlockDestructor) {
	linkList.SetZero();
}
void BuddyAllocator::Free(BuddyNode* node) {
	node->isUsing = false;
	linkList[node->layer].Add(node);
	BuddyBinaryTree::TryCombine(node, linkList.data(), &nodePool);
}
BuddyNode* BuddyAllocator::Allocate(uint targetLayer) {
	auto n = Allocate_T(targetLayer);
	n->isUsing = true;
	return n;
}
BuddyAllocator::~BuddyAllocator() {
	for (uint i = 0; i < allocatedTree.size(); ++i) {
		resourceBlockDestructor(allocatedTree[i].second);
	}
}
