#pragma once
#include <Common/Pool.h>
struct LinkedListLinker {
	LinkedListLinker* last = nullptr;
	LinkedListLinker* next = nullptr;
};
template<typename T>
class LinkedList;
template<typename F>
struct LinkedNode {
	friend class LinkedList<F>;

private:
	LinkedListLinker linker;

public:
	LinkedNode() {}
	F obj;
	LinkedNode(F const& obj) : obj(obj) {
	}
	LinkedNode<F>* Next() const {
		return (LinkedNode<F>*)linker.next;
	}
	LinkedNode<F>* Last() const {
		return (LinkedNode<F>*)linker.last;
	}
	inline ~LinkedNode();
};
template<typename T>
class LinkedList {
private:
	LinkedListLinker linker;

public:
	LinkedList() {
	}
	void Add(LinkedNode<T>* node) {
		node->linker.last = &linker;
		node->linker.next = linker.next;
		if (linker.next) {
			linker.next->last = &node->linker;
		}
		linker.next = &node->linker;
	}
	bool Empty() const {
		return !linker.next;
	}
	static void Remove(LinkedNode<T>* link) {
		if (link->linker.last) {
			link->linker.last->next = link->linker.next;
		}
		if (link->linker.next) {
			link->linker.next->last = link->linker.last;
		}
		link->linker.next = nullptr;
		link->linker.last = nullptr;
	}
	LinkedNode<T>* Begin() const {
		return (LinkedNode<T>*)linker.next;
	}
	void Clear() {
		linker.next = nullptr;
	}
};

template<typename F>
inline LinkedNode<F>::~LinkedNode() {
	if (linker.last || linker.next) {
		LinkedList<F>::Remove(this);
	}
}