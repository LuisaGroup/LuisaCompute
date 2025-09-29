#pragma once
#include <luisa/core/stl/string.h>
#include <luisa/vstl/pool.h>
namespace luisa::utils {
class  ManagedFirstFit {

public:
	class  Node {

	private:
		Node* _next{nullptr};
		size_t _offset{0u};
		size_t _size{0u};

	private:
		friend class ManagedFirstFit;

	public:
		Node() noexcept;
		Node(Node&&) noexcept = delete;
		Node(const Node&) noexcept = delete;
		Node& operator=(Node&&) noexcept = delete;
		Node& operator=(const Node&) noexcept = delete;
		[[nodiscard]] auto offset() const noexcept { return _offset; }
		[[nodiscard]] auto size() const noexcept { return _size; }
	};

private:
	Node _free_list;
	size_t _alignment;
	vstd::Pool<Node, true> _node_pool;

private:
	void _destroy() noexcept;

public:
	explicit ManagedFirstFit(size_t size, size_t alignment) noexcept;
	~ManagedFirstFit() noexcept;
	ManagedFirstFit(ManagedFirstFit&&) noexcept;
	ManagedFirstFit(const ManagedFirstFit&) noexcept = delete;
	[[nodiscard]] Node* allocate(size_t size) noexcept;
	[[nodiscard]] Node* allocate_best_fit(size_t size) noexcept;
	void free(Node* node) noexcept;
	[[nodiscard]] auto size() const noexcept { return _free_list._size; }
	[[nodiscard]] auto alignment() const noexcept { return _alignment; }
	void clean_all() noexcept;
};
}// namespace luisa::utils