// clang-format off
#pragma once
#include <vstl/TreeMapBase.h>
namespace vstd {
template<typename K, typename V, typename Compare = compare<K>>
class TreeMap {
private:
	using Element = TreeElement<K, V>;
	// data structure that represents a node in the tree
	struct Node {
		bool color;	 // 1 -> Red, 0 -> Black
		Node* parent;// pointer to the parent
		Node* left;	 // pointer to left child
		Node* right; // pointer to right child
		Element data;// holds the key
		template<typename A, typename... B>
		Node(A&& a, B&&... b)
			: data(std::forward<A>(a), std::forward<B>(b)...) {
		}
	};
	typedef Node* NodePtr;


	inline static const Compare comp;

	Pool<Node> pool;
	size_t len;

	NodePtr root;

public:
	struct Iterator {
		friend class TreeMap;

	private:
		NodePtr node;
		TreeMap const* selfPtr;
		Iterator(
			NodePtr node,
			TreeMap const* selfPtr)
			: node(node), selfPtr(selfPtr) {}

	public:
		Element& operator*() const {
			return node->data;
		}
		Element* operator->() const {
			return &node->data;
		}
		void operator++() {
			node = selfPtr->getNext(node);
		}
		void operator--() {
			node = selfPtr->getLast(node);
		}
		bool operator==(Iterator const& ptr) const {
			return node == ptr.node;
		}
		bool operator==(std::nullptr_t) const {
			return node == nullptr;
		}
		bool operator!=(std::nullptr_t) const {
			return node != nullptr;
		}
		operator bool() const {
			return node != nullptr;
		}
		bool operator!() const {
			return node == nullptr;
		}
	};

private:
	template<typename Key>
	NodePtr searchTreeHelper(NodePtr node, Key const& key) const {
		if (node == nullptr)
			return nullptr;
		auto compResult = comp(node->data.first, key);
		static_assert(std::is_same_v<decltype(compResult), int32_t>, "compare result must be int32");
		if (compResult == 0) {
			return node;
		}

		if (compResult > 0) {
			return searchTreeHelper(node->left, key);
		}
		return searchTreeHelper(node->right, key);
	}

	template<typename Key>
	std::pair<NodePtr, int32> searchClosestTreeHelper(NodePtr node, Key const& key, NodePtr lastNode, int32_t lastFlag) const {
		if (node == nullptr) {
			return {lastNode, lastFlag};
		}
		auto compResult = comp(node->data.first, key);
		static_assert(std::is_same_v<decltype(compResult), int32_t>, "compare result must be int32");
		if (compResult == 0) {
			return {node, 0};
		}

		if (compResult > 0) {
			return searchClosestTreeHelper(node->left, key, node, 1);
		}
		return searchClosestTreeHelper(node->right, key, node, -1);
	}

	void deleteOneNode(NodePtr z) {
		detail::TreeMapUtility::deleteOneNode(z, *reinterpret_cast<void**>(&root));
		--len;
		pool.Delete(z);
	}
	size_t size() const {
		return len;
	}
	template<typename Key>
	bool deleteNodeHelper(NodePtr node, Key const& key) {
		// find the node containing key
		NodePtr z = nullptr;
		while (node != nullptr) {
			auto compResult = comp(node->data.first, key);
			static_assert(std::is_same_v<decltype(compResult), int32_t>, "compare result must be int32");
			if (compResult == 0) {
				z = node;
				break;
			}
			if (compResult <= 0) {
				node = node->right;
			} else {
				node = node->left;
			}
		}

		if (z == nullptr) {
			return false;
		}
		deleteOneNode(z);
		return true;
	}

	void fixInsert(NodePtr k) {
		detail::TreeMapUtility::fixInsert(k, *reinterpret_cast<void**>(&root));
	}
	// find the node with the minimum key
	NodePtr minimum(NodePtr node) const {
		while (node->left != nullptr) {
			node = node->left;
		}
		return node;
	}

	// find the node with the maximum key
	NodePtr maximum(NodePtr node) const {
		while (node->right != nullptr) {
			node = node->right;
		}
		return node;
	}
	Node* getNext(Node* ptr) const {
		return reinterpret_cast<Node*>(detail::TreeMapUtility::getNext(ptr));
	}

	Node* getLast(Node* ptr) const {
		return reinterpret_cast<Node*>(detail::TreeMapUtility::getLast(ptr));
	}

	template<typename Func>
	void IterateAll(NodePtr node, Func&& func) const {
		if (node == nullptr) return;
		func(node->data);
		IterateAll(node->left, std::forward<Func>(func));
		IterateAll(node->right, std::forward<Func>(func));
	}
	template<typename Func>
	bool IterateAll_Earlybreak(NodePtr node, Func&& func) const {
		if (node == nullptr) return true;
		if (func(node->data)) {
			if (IterateAll_Earlybreak(node->left, std::forward<Func>(func)))
				return IterateAll_Earlybreak(node->right, std::forward<Func>(func));
		}
		return false;
	}

public:
	TreeMap(size_t initCapacity = 32)
		: pool(initCapacity, true) {
		len = 0;
		root = nullptr;
	}

	TreeMap(TreeMap const&) = delete;
	TreeMap(TreeMap&& o)
		: pool(std::move(o.pool)),
		  root(o.root) {
		o.root = nullptr;
		len = o.len;
		o.len = 0;
	}
	~TreeMap() = default;

	// search the tree for the key k
	// and return the corresponding node
	template<typename Key>
	Iterator find(Key const& k) const {
		return {searchTreeHelper(this->root, k), this};
	}
	// find closest node(null if the whole map is empty)
	template<typename Key>
	std::pair<Iterator, int32> find_closest(Key const& k) {
		auto pair = searchClosestTreeHelper(this->root, k, nullptr, 0);
		return {Iterator{pair.first, this}, pair.second};
	}

	// insert the key to the tree in its appropriate position
	// and fix the tree

	template<typename Key, typename... Value>
	std::pair<Iterator, bool> try_insert(Key&& key, Value&&... value) {
		// Ordinary Binary Search Insertion
		NodePtr y = nullptr;
		NodePtr x = this->root;
		int compResult;
		while (x != nullptr) {
			y = x;
			auto mCompResult = comp(x->data.first, key);
			static_assert(std::is_same_v<decltype(compResult), int32_t>, "compare result must be int32");
			compResult = mCompResult;
			if (compResult > 0) {
				x = x->left;
			} else if (compResult == 0) {
				return {Iterator{x, this}, false};

			} else {
				x = x->right;
			}
		}
		NodePtr node = pool.New(std::forward<Key>(key), std::forward<Value>(value)...);
		++len;
		node->parent = nullptr;
		node->left = nullptr;
		node->right = nullptr;
		node->color = true;// new node must be red
		// y is parent of x
		node->parent = y;
		if (y == nullptr) {
			root = node;
		} else if (compResult > 0) {
			y->left = node;
		} else {
			y->right = node;
		}

		// if new node is a root node, simply return
		if (node->parent == nullptr) {
			node->color = false;
			return {Iterator{node, this}, true};
		}

		// if the grandparent is null, simply return
		if (node->parent->parent == nullptr) {
			return {Iterator{node, this}, true};
		}
		// Fix the tree
		fixInsert(node);
		return {Iterator{node, this}, true};
	}

	template<typename Key, typename... Value>
	Iterator insert_or_assign(Key&& key, Value&&... value) {
		// Ordinary Binary Search Insertion
		NodePtr y = nullptr;
		NodePtr x = this->root;
		int compResult;
		while (x != nullptr) {
			y = x;
			auto mCompResult = comp(x->data.first, key);
			static_assert(std::is_same_v<decltype(compResult), int32_t>, "compare result must be int32");
			compResult = mCompResult;
			if (compResult > 0) {
				x = x->left;
			} else if (compResult == 0) {
				if constexpr (!std::is_same_v<V, void>) {
					if constexpr (std::is_move_constructible_v<V>) {
						x->data.second.~V();
						new (&x->data.second) V(std::forward<Value>(value)...);
					} else {
						static_assert(AlwaysFalse<Key>, "map value not move constructible!");
					}
				}
				return {x, this};

			} else {
				x = x->right;
			}
		}
		NodePtr node = pool.New(std::forward<Key>(key), std::forward<Value>(value)...);
		++len;
		node->parent = nullptr;
		node->left = nullptr;
		node->right = nullptr;
		node->color = true;// new node must be red
		// y is parent of x
		node->parent = y;
		if (y == nullptr) {
			root = node;
		} else if (compResult > 0) {
			y->left = node;
		} else {
			y->right = node;
		}

		// if new node is a root node, simply return
		if (node->parent == nullptr) {
			node->color = false;
			return {node, this};
		}

		// if the grandparent is null, simply return
		if (node->parent->parent == nullptr) {
			return {node, this};
		}
		// Fix the tree
		fixInsert(node);
		return {node, this};
	}

	// delete the node from the tree
	template<typename Key>
	bool remove(Key const& data) {
		return deleteNodeHelper(this->root, data);
	}
	void remove(Iterator const& ite) {
		deleteOneNode(ite.node);
	}
	void remove(Iterator&& ite) {
		remove((Iterator const&)ite);
	}
	void remove(Iterator const&& ite) {
		remove((Iterator const&)ite);
	}
	Iterator begin() const {
		if (this->root == nullptr)
			return {nullptr, nullptr};
		return {minimum(this->root), this};
	}

	Iterator last() const {
		return {maximum(this->root), this};
	}

	std::nullptr_t end() const { return nullptr; }
	template<typename Key>
	Iterator operator[](Key&& key) const {
		return find(key);
	}

	template<typename Func>
	void unsort_iterate(Func&& func) const {
		if constexpr (std::is_same_v<void, decltype(func(std::declval<Element>()))>)
			IterateAll(root, std::forward<Func>(func));
		else
			IterateAll_Earlybreak(root, std::forward<Func>(func));
	}
};

}// namespace vstd
// clang-format on