#pragma once
#include "vstlconfig.h"
#include <type_traits>
#include <stdint.h>
#include <memory>
#include <core/vstl/Pool.h>
#include <core/vstl/vector.h>
#include <core/vstl/Hash.h>
#include <core/vstl/MetaLib.h>
#include <core/vstl/VAllocator.h>

template<typename K, typename V, typename Hash = vstd::hash<K>, typename Equal = std::equal_to<K>, VEngine_AllocType allocType = VEngine_AllocType::VEngine>
class HashMap {
public:
	static_assert(allocType != VEngine_AllocType::Stack, "Hashmap do not support stack!");
	using KeyType = K;
	using ValueType = V;
	using HashType = Hash;
	using EqualType = Equal;
	using SelfType = HashMap<K, V, Hash, Equal, allocType>;
	struct NodePair {
	public:
		K first;
		V second;
		NodePair() {}
		template<typename A, typename... B>
		NodePair(A&& a, B&&... b) : first(std::forward<A>(a)), second(std::forward<B>(b)...) {}
	};

	struct LinkNode : public NodePair {
		LinkNode* last = nullptr;
		LinkNode* next = nullptr;
		size_t arrayIndex;
		LinkNode() noexcept {}
		template<typename A, typename... B>
		LinkNode(size_t arrayIndex, A&& key, B&&... args) noexcept : NodePair(std::forward<A>(key), std::forward<B>(args)...), arrayIndex(arrayIndex) {}

		static void Add(LinkNode*& source, LinkNode* dest) noexcept {
			if (!source) {
				source = dest;
			} else {
				if (source->next) {
					source->next->last = dest;
				}
				dest->next = source->next;
				dest->last = source;
				source->next = dest;
			}
		}
	};

public:
	struct Iterator {
	private:
		LinkNode** ii;

	public:
		Iterator(LinkNode** ii) : ii(ii) {}
		bool operator==(const Iterator& ite) const noexcept {
			return ii == ite.ii;
		}
		bool operator!=(const Iterator& ite) const noexcept {
			return ii != ite.ii;
		}
		void operator++() noexcept {
			++ii;
		}
		void operator--() noexcept {
			--ii;
		}
		void operator++(int32_t) noexcept {
			ii++;
		}
		void operator--(int32_t) noexcept {
			ii--;
		}

		NodePair const* operator->() const noexcept {
			return *ii;
		}
		NodePair const& operator*() const noexcept {
			return *operator->();
		}
	};
	struct Index {
		friend class SelfType;

	private:
		const SelfType* map;
		size_t hashValue;
		SelfType::LinkNode* node;
		Index(const SelfType* map, size_t hashValue, SelfType::LinkNode* node) noexcept : map(map), hashValue(hashValue), node(node) {}

	public:
		Index() : map(nullptr), hashValue(0), node(nullptr) {}
		bool operator==(const Index& a) const noexcept {
			return node == a.node;
		}
		operator bool() const noexcept {
			return node;
		}
		bool operator!() const noexcept {
			return !operator bool();
		}
		bool operator!=(const Index& a) const noexcept {
			return !operator==(a);
		}
		inline K const& Key() const noexcept;
		inline V& Value() const noexcept;
	};

private:
	ArrayList<LinkNode*, allocType> allocatedNodes;
	struct HashArray {
	private:
		LinkNode** nodesPtr = nullptr;
		size_t mSize;
		VAllocHandle<allocType> allocHandle;

	public:
		HashArray(HashArray&& map)
			: nodesPtr(map.nodesPtr),
			  mSize(map.mSize) {
			map.mSize = 0;
			map.nodesPtr = nullptr;
		}
		size_t size() const noexcept { return mSize; }
		HashArray() noexcept : mSize(0) {}
		void ClearAll() {
			memset(nodesPtr, 0, sizeof(LinkNode*) * mSize);
		}

		HashArray(size_t mSize) noexcept : mSize(mSize) {
			nodesPtr = (LinkNode**)allocHandle.Malloc(sizeof(LinkNode*) * mSize);
			memset(nodesPtr, 0, sizeof(LinkNode*) * mSize);
		}
		HashArray(HashArray& arr) noexcept : nodesPtr(arr.nodesPtr) {

			mSize = arr.mSize;
			arr.nodesPtr = nullptr;
		}
		void operator=(HashArray& arr) noexcept {
			nodesPtr = arr.nodesPtr;
			mSize = arr.mSize;
			arr.nodesPtr = nullptr;
		}
		void operator=(HashArray&& arr) noexcept {
			operator=(arr);
		}
		~HashArray() noexcept {
			if (nodesPtr) allocHandle.Free(nodesPtr);
		}
		LinkNode* const& operator[](size_t i) const noexcept {
			return nodesPtr[i];
		}
		LinkNode*& operator[](size_t i) noexcept {
			return nodesPtr[i];
		}
	};

	HashArray nodeVec;
	Pool<LinkNode, allocType, true> pool;
	inline static const Hash hsFunc;
	inline static const Equal eqFunc;
	template<typename A, typename... B>
	LinkNode* GetNewLinkNode(A&& key, B&&... args) {
		LinkNode* newNode = pool.New(allocatedNodes.size(), std::forward<A>(key), std::forward<B>(args)...);
		allocatedNodes.push_back(newNode);
		return newNode;
	}
	template<typename A>
	LinkNode* GetNewLinkNode(A&& key) {
		LinkNode* newNode = pool.New(allocatedNodes.size(), std::forward<A>(key));
		allocatedNodes.push_back(newNode);
		return newNode;
	}
	void DeleteLinkNode(LinkNode* oldNode) {
		auto ite = allocatedNodes.end() - 1;
		if (*ite != oldNode) {
			(*ite)->arrayIndex = oldNode->arrayIndex;
			allocatedNodes[oldNode->arrayIndex] = *ite;
		}
		allocatedNodes.erase(ite);
		pool.Delete(oldNode);
	}
	static size_t GetPow2Size(size_t capacity) noexcept {
		size_t ssize = 1;
		while (ssize < capacity)
			ssize <<= 1;
		return ssize;
	}
	static size_t GetHash(size_t hash, size_t size) noexcept {
		return hash & (size - 1);
	}
	void Resize(size_t newCapacity) noexcept {
		size_t capacity = nodeVec.size();
		if (capacity >= newCapacity) return;
		allocatedNodes.reserve(newCapacity);
		HashArray newNode(newCapacity);
		for (auto node : allocatedNodes) {
			auto next = node->next;
			node->last = nullptr;
			node->next = nullptr;
			size_t hashValue = hsFunc(node->first);
			hashValue = GetHash(hashValue, newCapacity);
			LinkNode*& targetHeaderLink = newNode[hashValue];
			if (!targetHeaderLink) {
				targetHeaderLink = node;
			} else {
				node->next = targetHeaderLink;
				targetHeaderLink->last = node;
				targetHeaderLink = node;
			}
		}
		nodeVec = newNode;
	}
	static Index EmptyIndex() noexcept {
		return Index(nullptr, -1, nullptr);
	}

public:
	size_t Size() const {
		return allocatedNodes.size();
	}
	decltype(auto) begin() const {
		return Iterator(allocatedNodes.begin());
	}
	decltype(auto) end() const {
		return Iterator(allocatedNodes.end());
	}
	//////////////////Construct & Destruct
	HashMap(size_t capacity) noexcept : pool(capacity) {
		if (capacity < 2) capacity = 2;
		capacity = GetPow2Size(capacity);
		nodeVec = HashArray(capacity);
		allocatedNodes.reserve(capacity);
	}
	HashMap(SelfType&& map)
		: allocatedNodes(std::move(map.allocatedNodes)),
		  nodeVec(std::move(map.nodeVec)),
		  pool(std::move(map.pool)) {
	}

	void operator=(SelfType&& map) {
		this->~SelfType();
		new (this) SelfType(std::move(map));
	}
	~HashMap() noexcept {
		for (auto& ite : allocatedNodes) {
			pool.Delete(ite);
		}
	}
	HashMap() noexcept : HashMap(16) {}
	///////////////////////
	Index Insert(const K& key, const V& value) noexcept {
		size_t hashOriginValue = hsFunc(key);
		size_t hashValue;

		auto a = nodeVec.size();
		hashValue = GetHash(hashOriginValue, nodeVec.size());
		for (LinkNode* node = nodeVec[hashValue]; node != nullptr; node = node->next) {
			if (eqFunc(node->first, key)) {
				node->second = value;
				return Index(this, hashOriginValue, node);
			}
		}

		size_t targetCapacity = (size_t)((allocatedNodes.size() + 1) / 0.75);
		if (targetCapacity < 16) targetCapacity = 16;
		if (targetCapacity >= nodeVec.size()) {
			Resize(GetPow2Size(targetCapacity));
			hashValue = GetHash(hashOriginValue, nodeVec.size());
		}
		LinkNode* newNode = GetNewLinkNode(key, value);
		LinkNode::Add(nodeVec[hashValue], newNode);
		return Index(this, hashOriginValue, newNode);
	}

	Index Insert(const K& key, V&& value) noexcept {
		size_t hashOriginValue = hsFunc(key);
		size_t hashValue;

		auto a = nodeVec.size();
		hashValue = GetHash(hashOriginValue, nodeVec.size());
		for (LinkNode* node = nodeVec[hashValue]; node != nullptr; node = node->next) {
			if (eqFunc(node->first, key)) {
				node->second = std::move(value);
				return Index(this, hashOriginValue, node);
			}
		}

		size_t targetCapacity = (size_t)((allocatedNodes.size() + 1) / 0.75);
		if (targetCapacity < 16) targetCapacity = 16;
		if (targetCapacity >= nodeVec.size()) {
			Resize(GetPow2Size(targetCapacity));
			hashValue = GetHash(hashOriginValue, nodeVec.size());
		}
		LinkNode* newNode = GetNewLinkNode(key, std::move(value));
		LinkNode::Add(nodeVec[hashValue], newNode);
		return Index(this, hashOriginValue, newNode);
	}
	template<typename... ARGS>
	Index ForceEmplace(const K& key, ARGS&&... args) {
		size_t hashOriginValue = hsFunc(key);
		size_t hashValue;

		auto a = nodeVec.size();
		hashValue = GetHash(hashOriginValue, nodeVec.size());
		for (LinkNode* node = nodeVec[hashValue]; node != nullptr; node = node->next) {
			if (eqFunc(node->first, key)) {
				//node->second = std::move(value);
				node->second.~V();
				new (&node->second) V(std::forward<ARGS>(args)...);
				return Index(this, hashOriginValue, node);
			}
		}

		size_t targetCapacity = (size_t)((allocatedNodes.size() + 1) / 0.75);
		if (targetCapacity < 16) targetCapacity = 16;
		if (targetCapacity >= nodeVec.size()) {
			Resize(GetPow2Size(targetCapacity));
			hashValue = GetHash(hashOriginValue, nodeVec.size());
		}
		LinkNode* newNode = GetNewLinkNode(key, std::forward<ARGS>(args)...);
		LinkNode::Add(nodeVec[hashValue], newNode);
		return Index(this, hashOriginValue, newNode);
	}
	template<typename... ARGS>
	Index Emplace(const K& key, ARGS&&... args) {
		size_t hashOriginValue = hsFunc(key);
		size_t hashValue;

		auto a = nodeVec.size();
		hashValue = GetHash(hashOriginValue, nodeVec.size());
		for (LinkNode* node = nodeVec[hashValue]; node != nullptr; node = node->next) {
			if (eqFunc(node->first, key)) {
				return Index(this, hashOriginValue, node);
			}
		}

		size_t targetCapacity = (size_t)((allocatedNodes.size() + 1) / 0.75);
		if (targetCapacity < 16) targetCapacity = 16;
		if (targetCapacity >= nodeVec.size()) {
			Resize(GetPow2Size(targetCapacity));
			hashValue = GetHash(hashOriginValue, nodeVec.size());
		}
		LinkNode* newNode = GetNewLinkNode(key, std::forward<ARGS>(args)...);
		LinkNode::Add(nodeVec[hashValue], newNode);
		return Index(this, hashOriginValue, newNode);
	}
	template<typename... ARGS>
	Index TryEmplace(bool& isNewElement, const K& key, ARGS&&... args) {
		size_t hashOriginValue = hsFunc(key);
		size_t hashValue;

		auto a = nodeVec.size();
		hashValue = GetHash(hashOriginValue, nodeVec.size());
		for (LinkNode* node = nodeVec[hashValue]; node != nullptr; node = node->next) {
			if (eqFunc(node->first, key)) {
				isNewElement = false;
				return Index(this, hashOriginValue, node);
			}
		}

		size_t targetCapacity = (size_t)((allocatedNodes.size() + 1) / 0.75);
		if (targetCapacity < 16) targetCapacity = 16;
		if (targetCapacity >= nodeVec.size()) {
			Resize(GetPow2Size(targetCapacity));
			hashValue = GetHash(hashOriginValue, nodeVec.size());
		}
		LinkNode* newNode = GetNewLinkNode(key, std::forward<ARGS>(args)...);
		LinkNode::Add(nodeVec[hashValue], newNode);
		isNewElement = true;
		return Index(this, hashOriginValue, newNode);
	}

	//void(size_t, K const&, V&)
	template<typename Func>
	void IterateAll(const Func& func) const noexcept {
		using FuncType = std::remove_cvref_t<std::remove_pointer_t<Func>>;
		static constexpr size_t ArgSize = FuncArgCount<FuncType>;
		using ReturnType = typename FunctionDataType<FuncType>::RetType;
		static_assert(std::is_same_v<ReturnType, void>, "Iterate functor must return void!");
		if constexpr (ArgSize == 3) {
			for (size_t i = 0; i < allocatedNodes.size(); ++i) {
				auto vv = allocatedNodes[i];
				func(i, (K const&)vv->first, vv->second);
			}
		} else if constexpr (ArgSize == 2) {
			for (auto vv : allocatedNodes) {
				func((K const&)vv->first, vv->second);
			}
		} else if constexpr (ArgSize == 1) {
			for (auto vv : allocatedNodes) {
				func(vv->second);
			}
		} else {
			static_assert(std::_Always_false<Func>, "Invalid Iterate Functions");
		}
	}
	void Reserve(size_t capacity) noexcept {
		size_t newCapacity = GetPow2Size(capacity);
		Resize(newCapacity);
	}
	Index Find(const K& key) const noexcept {
		size_t hashOriginValue = hsFunc(key);
		size_t hashValue = GetHash(hashOriginValue, nodeVec.size());
		for (LinkNode* node = nodeVec[hashValue]; node != nullptr; node = node->next) {
			if (eqFunc(node->first, key)) {
				return Index(this, hashOriginValue, node);
			}
		}

		return EmptyIndex();
	}
	void Remove(const K& key) noexcept {

		size_t hashOriginValue = hsFunc(key);
		size_t hashValue = GetHash(hashOriginValue, nodeVec.size());
		LinkNode*& startNode = nodeVec[hashValue];
		for (LinkNode* node = startNode; node != nullptr; node = node->next) {
			if (eqFunc(node->first, key)) {
				if (startNode == node) {
					startNode = node->next;
				}
				if (node->next)
					node->next->last = node->last;
				if (node->last)
					node->last->next = node->next;
				DeleteLinkNode(node);
				return;
			}
		}
	}

	void Remove(const Index& ite) noexcept {

		size_t hashValue = GetHash(ite.hashValue, nodeVec.size());
		if (nodeVec[hashValue] == ite.node) {
			nodeVec[hashValue] = ite.node->next;
		}
		if (ite.node->last)
			ite.node->last->next = ite.node->next;
		if (ite.node->next)
			ite.node->next->last = ite.node->last;
		DeleteLinkNode(ite.node);
	}
	V& operator[](const K& key) noexcept {

		size_t hashOriginValue = hsFunc(key);
		size_t hashValue = GetHash(hashOriginValue, nodeVec.size());
		for (LinkNode* node = nodeVec[hashValue]; node != nullptr; node = node->next) {
			if (eqFunc(node->first, key)) {
				return node->second;
			}
		}

		return *(V*)nullptr;
	}
	V const& operator[](const K& key) const noexcept {

		size_t hashOriginValue = hsFunc(key);
		size_t hashValue = GetHash(hashOriginValue, nodeVec.size());
		for (LinkNode* node = nodeVec[hashValue]; node != nullptr; node = node->next) {
			if (eqFunc(node->first, key)) {
				return node->second;
			}
		}

		return *(V const*)nullptr;
	}
	bool TryGet(const K& key, V& value) const noexcept {

		size_t hashOriginValue = hsFunc(key);
		size_t hashValue = GetHash(hashOriginValue, nodeVec.size());
		for (LinkNode* node = nodeVec[hashValue]; node != nullptr; node = node->next) {
			if (eqFunc(node->first, key)) {
				value = node->second;
				return true;
			}
		}

		return false;
	}
	bool Contains(const K& key) const noexcept {

		size_t hashOriginValue = hsFunc(key);
		size_t hashValue = GetHash(hashOriginValue, nodeVec.size());
		for (LinkNode* node = nodeVec[hashValue]; node != nullptr; node = node->next) {
			if (eqFunc(node->first, key)) {
				return true;
			}
		}

		return false;
	}
	void Clear() noexcept {
		if (allocatedNodes.empty()) return;
		nodeVec.ClearAll();
		for (auto& ite : allocatedNodes) {
			pool.Delete(ite);
		}
		allocatedNodes.clear();
	}
	size_t size() const noexcept { return allocatedNodes.size(); }

	size_t GetCapacity() const noexcept { return nodeVec.size(); }
};

template<typename K, typename V, typename Hash, typename Equal, VEngine_AllocType allocType>
inline K const& HashMap<K, V, Hash, Equal, allocType>::Index::Key() const noexcept {
	return node->first;
}
template<typename K, typename V, typename Hash, typename Equal, VEngine_AllocType allocType>
inline V& HashMap<K, V, Hash, Equal, allocType>::Index::Value() const noexcept {
	return node->second;
}