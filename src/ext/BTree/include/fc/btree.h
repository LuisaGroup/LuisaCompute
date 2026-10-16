#ifndef __FC_BTREE_H__
#define __FC_BTREE_H__

#ifndef FC_USE_SIMD
#define FC_USE_SIMD 0
#endif // FC_USE_SIMD

#ifndef FC_PREFER_BINARY_SEARCH
#define FC_PREFER_BINARY_SEARCH 0
#endif //FC_PREFER_BINARY_SEARCH

#if FC_USE_SIMD
#include "fc/comp.h"
#ifdef _MSC_VER
#pragma warning(disable : 4324)
#endif // MSC_VER
#endif // FC_USE_SIMD

#include "fc/details.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace frozenca {

template <Containable K, Containable V> struct BTreePair {
  K first;
  V second;

  BTreePair(K &&k, V &&v): first(std::forward<K>(k)), second(std::forward<V>(v)) {}
  
  BTreePair() = default;
  
  BTreePair(K &&k): first(std::forward<K>(k)), second() {}
  
  BTreePair(V &&v) requires(!std::is_same_v<K, V>)
      : first(), second(std::forward<V>(v)) {}

  operator std::pair<const K &, V &>() noexcept { return {first, second}; }

  friend bool operator==(const BTreePair &lhs, const BTreePair &rhs) noexcept {
    return lhs.first == rhs.first && lhs.second == rhs.second;
  }

  friend bool operator!=(const BTreePair &lhs, const BTreePair &rhs) noexcept {
    return !(lhs == rhs);
  }
};

template <typename T> struct TreePairRef { using type = T &; };

template <typename T, typename U> struct TreePairRef<BTreePair<T, U>> {
  using type = std::pair<const T &, U &>;
};

template <typename TreePair> using PairRefType = typename TreePairRef<TreePair>::type;

template <typename T, typename U>
bool operator==(const BTreePair<T, U> &lhs,
                const PairRefType<BTreePair<T, U>> &rhs) noexcept {
  return lhs.first == rhs.first && lhs.second == rhs.second;
}

template <typename T, typename U>
bool operator!=(const BTreePair<T, U> &lhs,
                const PairRefType<BTreePair<T, U>> &rhs) noexcept {
  return !(lhs == rhs);
}

template <typename T, typename U>
bool operator==(const PairRefType<BTreePair<T, U>> &lhs,
                const BTreePair<T, U> &rhs) noexcept {
  return rhs == lhs;
}

template <typename T, typename U>
bool operator!=(const PairRefType<BTreePair<T, U>> &lhs,
                const BTreePair<T, U> &rhs) noexcept {
  return rhs != lhs;
}

template <typename V> struct Projection {
  const auto &operator()(const V &value) const noexcept { return value.first; }
};

template <typename V> struct ProjectionIter {
  auto &operator()(V &iter_ref) noexcept { return iter_ref.first; }

  const auto &operator()(const V &iter_ref) const noexcept {
    return iter_ref.first;
  }
};

template <Containable K, typename V, attr_t Fanout, typename Comp,
          bool AllowDup, template <typename T> class Alloc>
requires(Fanout >= 2) class BTreeBase;

template <Containable K, typename V, attr_t Fanout, typename Comp,
          bool AllowDup, template <typename T> class AllocTemplate>
struct join_helper;

template <Containable K, typename V, attr_t Fanout, typename Comp,
          bool AllowDup, template <typename T> class AllocTemplate, typename T>
struct split_helper;

template <Containable K, typename V, attr_t Fanout, typename Comp,
          bool AllowDup, template <typename T> class AllocTemplate>
requires(Fanout >= 2) class BTreeBase {

  struct Node;
  using Alloc = AllocTemplate<Node>;

  struct Deleter {
    [[no_unique_address]] Alloc alloc_;
    Deleter(const Alloc &alloc) : alloc_{alloc} {}

    template <typename T> void operator()(T *node) noexcept {
      std::destroy_at(node);
      alloc_.deallocate(node, 1);
    }
  };

  // invariant: V is either K or pair<const K, Value> for some Value type.
  static constexpr bool is_set_ = std::is_same_v<K, V>;

  static constexpr bool is_disk_ = DiskAllocable<V>;

  static constexpr auto disk_max_nkeys = static_cast<std::size_t>(2 * Fanout);

  static constexpr bool use_linsearch_ =
#if FC_PREFER_BINARY_SEARCH
      std::is_arithmetic_v<K> && (Fanout <= 32);
#else
      std::is_arithmetic_v<K> && (Fanout <= 128);
#endif // FC_PREFER_BINARY_SEARCH

  static constexpr bool CompIsLess = std::is_same_v<Comp, std::ranges::less> ||
                                     std::is_same_v<Comp, std::less<K>>;
  static constexpr bool CompIsGreater =
      std::is_same_v<Comp, std::ranges::greater> ||
      std::is_same_v<Comp, std::greater<K>>;

  static constexpr bool use_simd_ =
#if FC_USE_SIMD
      is_set_ && CanUseSimd<K> && (Fanout % (sizeof(K) == 4 ? 8 : 4) == 0) &&
      (Fanout <= 128) && (CompIsLess || CompIsGreater);
#else
      false;
#endif // FC_USE_SIMD

#if FC_USE_SIMD
  struct alignas(64) Node {
#else
  struct Node {
#endif // FC_USE_SIND
    using keys_type =
        std::conditional_t<is_disk_, std::array<V, disk_max_nkeys>,
                           std::vector<V>>;

    // invariant: except root, t - 1 <= #(key) <= 2 * t - 1
    // invariant: for root, 0 <= #(key) <= 2 * t - 1
    // invariant: keys are sorted
    // invariant: for internal nodes, t <= #(child) == (#(key) + 1)) <= 2 * t
    // invariant: for root, 0 <= #(child) == (#(key) + 1)) <= 2 * t
    // invariant: for leaves, 0 == #(child)
    // invariant: child_0 <= key_0 <= child_1 <= ... <=  key_(N - 1) <= child_N
    keys_type keys_;
    Node *parent_ = nullptr;
    attr_t size_ = 0; // number of keys in the subtree (not keys in this node)
    attr_t index_ = 0;
    attr_t height_ = 0;
    attr_t num_keys_ =
        0; // number of keys in this node, used only for disk variant
      std::vector<std::unique_ptr<Node, Deleter>> children_;

    Node() { keys_.reserve(disk_max_nkeys); }

    Node() requires(is_disk_) {
      if constexpr (use_simd_) {
        keys_.fill(std::numeric_limits<K>::max());
      }
    }

    Node(const Node &node) = delete;
    Node &operator=(const Node &node) = delete;
    Node(Node &&node) = delete;
    Node &operator=(Node &&node) = delete;

    [[nodiscard]] bool is_leaf() const noexcept { return children_.empty(); }

    [[nodiscard]] bool is_full() const noexcept {
      if constexpr (is_disk_) {
        return num_keys_ == 2 * Fanout - 1;
      } else {
        return std::ssize(keys_) == 2 * Fanout - 1;
      }
    }

    [[nodiscard]] bool can_take_key() const noexcept {
      if constexpr (is_disk_) {
        return num_keys_ > Fanout - 1;
      } else {
        return std::ssize(keys_) > Fanout - 1;
      }
    }

    [[nodiscard]] bool has_minimal_keys() const noexcept {
      if constexpr (is_disk_) {
        return parent_ && num_keys_ == Fanout - 1;
      } else {
        return parent_ && std::ssize(keys_) == Fanout - 1;
      }
    }

    [[nodiscard]] bool empty() const noexcept {
      if constexpr (is_disk_) {
        return num_keys_ == 0;
      } else {
        return keys_.empty();
      }
    }

    void clear_keys() noexcept {
      if constexpr (is_disk_) {
        num_keys_ = 0;
      } else {
        keys_.clear();
      }
    }

    [[nodiscard]] attr_t nkeys() const noexcept {
      if constexpr (is_disk_) {
        return num_keys_;
      } else {
        return static_cast<attr_t>(std::ssize(keys_));
      }
    }
  };

  struct BTreeNonConstIterTraits {
    using difference_type = attr_t;
    using value_type = V;
    using pointer = V *;
    using reference = V &;
    using iterator_category = std::bidirectional_iterator_tag;
    using iterator_concept = iterator_category;

    static reference make_ref(value_type &val) noexcept { return val; }
  };

  struct BTreeConstIterTraits {
    using difference_type = attr_t;
    using value_type = V;
    using pointer = const V *;
    using reference = const V &;
    using iterator_category = std::bidirectional_iterator_tag;
    using iterator_concept = iterator_category;

    static reference make_ref(const value_type &val) noexcept { return val; }
  };

  struct BTreeRefIterTraits {
    using difference_type = attr_t;
    using value_type = V;
    using pointer = V *;
    using reference = PairRefType<V>;
    using iterator_category = std::bidirectional_iterator_tag;
    using iterator_concept = iterator_category;

    static reference make_ref(value_type &val) noexcept {
      return {std::cref(val.first), std::ref(val.second)};
    }
  };

  template <typename IterTraits> struct BTreeIterator {
    using difference_type = typename IterTraits::difference_type;
    using value_type = typename IterTraits::value_type;
    using pointer = typename IterTraits::pointer;
    using reference = typename IterTraits::reference;
    using iterator_category = typename IterTraits::iterator_category;
    using iterator_concept = typename IterTraits::iterator_concept;

    Node *node_ = nullptr;
    attr_t index_;

    BTreeIterator() noexcept = default;

    BTreeIterator(Node *node, attr_t i) noexcept : node_{node}, index_{i} {
      assert(node_ && i >= 0 && i <= node_->nkeys());
    }

    template <typename IterTraitsOther>
    BTreeIterator(const BTreeIterator<IterTraitsOther> &other) noexcept
        : BTreeIterator(other.node_, other.index_) {}

    reference operator*() const noexcept {
      return IterTraits::make_ref(node_->keys_[index_]);
    }

    pointer operator->() const noexcept { return &(node_->keys_[index_]); }

    // useful remark:
    // incrementing/decrementing iterator in an internal node will always
    // produce an iterator in a leaf node,
    // incrementing/decrementing iterator in a leaf node will always produce
    // an iterator in a leaf node for non-boundary keys,
    // an iterator in an internal node for boundary keys

    void climb() noexcept {
      while (node_->parent_ && index_ == node_->nkeys()) {
        index_ = node_->index_;
        node_ = node_->parent_;
      }
    }

    void dig() noexcept {
      while (!node_->children_.empty()) {
        auto id = index_;
        index_ = ssize(node_->children_[id]->keys_);
        node_ = node_->children_[id].get();
      }
    }

    void increment() noexcept {
      // we don't do past to end() check for efficiency
      if (!node_->is_leaf()) {
        node_ = leftmost_leaf(node_->children_[index_ + 1].get());
        index_ = 0;
      } else {
        ++index_;
        while (node_->parent_ && index_ == node_->nkeys()) {
          index_ = node_->index_;
          node_ = node_->parent_;
        }
      }
    }

    void decrement() noexcept {
      if (!node_->is_leaf()) {
        node_ = rightmost_leaf(node_->children_[index_].get());
        index_ = node_->nkeys() - 1;
      } else if (index_ > 0) {
        --index_;
      } else {
        while (node_->parent_ && node_->index_ == 0) {
          node_ = node_->parent_;
        }
        if (node_->index_ > 0) {
          index_ = node_->index_ - 1;
          node_ = node_->parent_;
        }
      }
    }

    bool verify() noexcept {
      // Uncomment this line for testing
      // assert(!node_->parent_ || (index_ < node_->nkeys()));
      return true;
    }

    BTreeIterator &operator++() noexcept {
      increment();
      assert(verify());
      return *this;
    }

    BTreeIterator operator++(int) noexcept {
      BTreeIterator temp = *this;
      increment();
      assert(verify());
      return temp;
    }

    BTreeIterator &operator--() noexcept {
      decrement();
      assert(verify());
      return *this;
    }

    BTreeIterator operator--(int) noexcept {
      BTreeIterator temp = *this;
      decrement();
      assert(verify());
      return temp;
    }

    friend bool operator==(const BTreeIterator &x,
                           const BTreeIterator &y) noexcept {
      return x.node_ == y.node_ && x.index_ == y.index_;
    }

    friend bool operator!=(const BTreeIterator &x,
                           const BTreeIterator &y) noexcept {
      return !(x == y);
    }
  };

public:
  using key_type = K;
  using value_type = V;
  using reference_type = std::conditional_t<is_set_, const V &, PairRefType<V>>;
  using const_reference_type = const V &;
  using node_type = Node;
  using size_type = std::size_t;
  using difference_type = attr_t;
  using allocator_type = Alloc;
  using deleter_type = Deleter;
    using nodeptr_type = std::unique_ptr<Node, deleter_type>;
  using Proj =
      std::conditional_t<is_set_, std::identity, Projection<const V &>>;
  using ProjIter = std::conditional_t<is_set_, std::identity,
                                      ProjectionIter<PairRefType<V>>>;

  static_assert(
      std::indirect_strict_weak_order<
          Comp, std::projected<std::ranges::iterator_t<std::vector<V>>, Proj>>);

  // invariant: K cannot be mutated
  // so if V is K, uses a const iterator.
  // if V is BTreePair<K, V>, uses a non-const iterator (but only value can
  // be mutated)
private:
  using nonconst_iterator_type = BTreeIterator<BTreeNonConstIterTraits>;

public:
  using iterator_type = BTreeIterator<
      std::conditional_t<is_set_, BTreeConstIterTraits, BTreeRefIterTraits>>;
  using const_iterator_type = BTreeIterator<BTreeConstIterTraits>;
  using reverse_iterator_type = std::reverse_iterator<iterator_type>;
  using const_reverse_iterator_type =
      std::reverse_iterator<const_iterator_type>;

private:
  [[no_unique_address]] Alloc alloc_;
  nodeptr_type root_;
  const_iterator_type begin_;

protected:
  nodeptr_type make_node() {
    auto buf = alloc_.allocate(1);
      Node *node = new (buf) Node();
      return nodeptr_type(node, deleter_type(alloc_));
    }

public:
  BTreeBase(const Alloc &alloc = Alloc{})
      : alloc_{alloc}, root_(make_node()), begin_{root_.get(), 0} {}

  BTreeBase(std::initializer_list<value_type> init,
            const Alloc &alloc = Alloc{})
      : BTreeBase(alloc) {
    for (auto val : init) {
      insert(std::move(val));
    }
  }

  BTreeBase(const BTreeBase &other) = delete;
  BTreeBase &operator=(const BTreeBase &other) = delete;
  BTreeBase(BTreeBase &&other) noexcept = default;
  BTreeBase &operator=(BTreeBase &&other) noexcept = default;

  void swap(BTreeBase &other) noexcept {
    std::swap(alloc_, other.alloc_);
    std::swap(root_, other.root_);
    std::swap(begin_, other.begin_);
  }

  bool verify(const Node *node) const {
    // invariant: node never null
    assert(node);

    // invariant: except root, t - 1 <= #(key) <= 2 * t - 1
    assert(!node->parent_ ||
           (node->nkeys() >= Fanout - 1 && node->nkeys() <= 2 * Fanout - 1));

    // invariant: keys are sorted
    assert(std::ranges::is_sorted(node->keys_.begin(),
                                  node->keys_.begin() + node->nkeys(), Comp{},
                                  Proj{}));

    // invariant: for internal nodes, t <= #(child) == (#(key) + 1)) <= 2 * t
    assert(!node->parent_ || node->is_leaf() ||
           (std::ssize(node->children_) >= Fanout &&
            std::ssize(node->children_) == node->nkeys() + 1 &&
            std::ssize(node->children_) <= 2 * Fanout));

    // index check
    assert(!node->parent_ ||
           node == node->parent_->children_[node->index_].get());

    // invariant: child_0 <= key_0 <= child_1 <= ... <=  key_(N - 1) <=
    // child_N
    if (!node->is_leaf()) {
      auto num_keys = node->nkeys();

      for (attr_t i = 0; i < node->nkeys(); ++i) {
        assert(node->children_[i]);
        assert(!Comp{}(
            Proj{}(node->keys_[i]),
            Proj{}(
                node->children_[i]->keys_[node->children_[i]->nkeys() - 1])));
        assert(!Comp{}(Proj{}(node->children_[i + 1]->keys_[0]),
                       Proj{}(node->keys_[i])));
        // parent check
        assert(node->children_[i]->parent_ == node);
        // recursive check
        assert(verify(node->children_[i].get()));
        assert(node->height_ == node->children_[i]->height_ + 1);
        num_keys += node->children_[i]->size_;
      }
      // parent check
      assert(node->children_.back()->parent_ == node);
      assert(verify(node->children_.back().get()));
      assert(node->height_ == node->children_.back()->height_ + 1);
      num_keys += node->children_.back()->size_;
      assert(node->size_ == num_keys);
    } else {
      assert(node->size_ == node->nkeys());
      assert(node->height_ == 0);
    }

    return true;
  }

  [[nodiscard]] bool verify() const {
    // Uncomment these lines for testing
#ifdef _CONTROL_IN_TEST
     assert(begin_ == const_iterator_type(leftmost_leaf(root_.get()), 0));
     assert(verify(root_.get()));
#endif
    return true;
  }

  [[nodiscard]] iterator_type begin() noexcept { return begin_; }

  [[nodiscard]] const_iterator_type begin() const noexcept {
    return const_iterator_type(begin_);
  }

  [[nodiscard]] const_iterator_type cbegin() const noexcept {
    return const_iterator_type(begin_);
  }

  [[nodiscard]] iterator_type end() noexcept {
    return iterator_type(root_.get(), root_->nkeys());
  }

  [[nodiscard]] const_iterator_type end() const noexcept {
    return const_iterator_type(root_.get(), root_->nkeys());
  }

  [[nodiscard]] const_iterator_type cend() const noexcept {
    return const_iterator_type(root_.get(), root_->nkeys());
  }

  [[nodiscard]] reverse_iterator_type rbegin() noexcept {
    return reverse_iterator_type(begin());
  }

  [[nodiscard]] const_reverse_iterator_type rbegin() const noexcept {
    return const_reverse_iterator_type(begin());
  }

  [[nodiscard]] const_reverse_iterator_type crbegin() const noexcept {
    return const_reverse_iterator_type(cbegin());
  }

  [[nodiscard]] reverse_iterator_type rend() noexcept {
    return reverse_iterator_type(end());
  }

  [[nodiscard]] const_reverse_iterator_type rend() const noexcept {
    return const_reverse_iterator_type(end());
  }

  [[nodiscard]] const_reverse_iterator_type crend() const noexcept {
    return const_reverse_iterator_type(cend());
  }

  [[nodiscard]] bool empty() const noexcept { return root_->size_ == 0; }

  [[nodiscard]] size_type size() const noexcept {
    return static_cast<size_type>(root_->size_);
  }

  [[nodiscard]] attr_t height() const noexcept { return root_->height_; }

protected:
  [[nodiscard]] Node *get_root() noexcept { return root_.get(); }

  [[nodiscard]] Node *get_root() const noexcept { return root_.get(); }

public:
  void clear() {
    root_ = make_node();
    begin_ = iterator_type(root_.get(), 0);
  }

protected:
  static Node *rightmost_leaf(Node *curr) noexcept {
    while (curr && !curr->is_leaf()) {
      curr = curr->children_[std::ssize(curr->children_) - 1].get();
    }
    return curr;
  }

  static const Node *rightmost_leaf(const Node *curr) noexcept {
    while (curr && !curr->is_leaf()) {
      curr = curr->children_[std::ssize(curr->children_) - 1].get();
    }
    return curr;
  }

  static Node *leftmost_leaf(Node *curr) noexcept {
    while (curr && !curr->is_leaf()) {
      curr = curr->children_[0].get();
    }
    return curr;
  }

  static const Node *leftmost_leaf(const Node *curr) noexcept {
    while (curr && !curr->is_leaf()) {
      curr = curr->children_[0].get();
    }
    return curr;
  }

  void promote_root_if_necessary() {
    if (root_->empty()) {
      assert(std::ssize(root_->children_) == 1);
      root_ = std::move(root_->children_[0]);
      root_->index_ = 0;
      root_->parent_ = nullptr;
    }
  }

  void set_begin() { begin_ = iterator_type(leftmost_leaf(root_.get()), 0); }

  // node brings a key from parent
  // parent brings a key from right sibling
  // node brings a child from right sibling
  void left_rotate(Node *node) {
    auto parent = node->parent_;
    assert(node && parent && parent->children_[node->index_].get() == node &&
           node->index_ + 1 < std::ssize(parent->children_) &&
           parent->children_[node->index_ + 1]->can_take_key());
    auto sibling = parent->children_[node->index_ + 1].get();

    if constexpr (is_disk_) {
      node->keys_[node->num_keys_] = parent->keys_[node->index_];
      node->num_keys_++;
      parent->keys_[node->index_] = sibling->keys_[0];
      std::memmove(sibling->keys_.data(), sibling->keys_.data() + 1,
                   (sibling->num_keys_ - 1) * sizeof(V));
      sibling->num_keys_--;
      if constexpr (use_simd_) {
        sibling->keys_[sibling->num_keys_] = std::numeric_limits<K>::max();
      }
    } else {
      node->keys_.push_back(std::move(parent->keys_[node->index_]));
      parent->keys_[node->index_] = std::move(sibling->keys_.front());
      std::shift_left(sibling->keys_.begin(), sibling->keys_.end(), 1);
      sibling->keys_.pop_back();
    }

    node->size_++;
    sibling->size_--;

    if (!node->is_leaf()) {
      const auto orphan_size = sibling->children_.front()->size_;
      node->size_ += orphan_size;
      sibling->size_ -= orphan_size;

      sibling->children_.front()->parent_ = node;
      sibling->children_.front()->index_ =
          static_cast<attr_t>(std::ssize(node->children_));
      node->children_.push_back(std::move(sibling->children_.front()));
      std::shift_left(sibling->children_.begin(), sibling->children_.end(), 1);
      sibling->children_.pop_back();
      for (auto &&child : sibling->children_) {
        child->index_--;
      }
    }
  }

  // left_rotate() * n
  void left_rotate_n(Node *node, attr_t n) {
    assert(n >= 1);
    if (n == 1) {
      left_rotate(node);
      return;
    }

    auto parent = node->parent_;
    assert(node && parent && parent->children_[node->index_].get() == node &&
           node->index_ + 1 < std::ssize(parent->children_));
    auto sibling = parent->children_[node->index_ + 1].get();
    assert(sibling->nkeys() >= (Fanout - 1) + n);

    if constexpr (is_disk_) {
      // brings one key from parent
      node->keys_[node->num_keys_] = parent->keys_[node->index_];
      node->num_keys_++;
      // brings n - 1 keys from sibling
      std::memcpy(node->keys_.data() + node->num_keys_, sibling->keys_.data(),
                  (n - 1) * sizeof(V));
      // parent brings one key from sibling
      parent->keys_[node->index_] = sibling->keys_[n - 1];
      std::memmove(sibling->keys_.data(), sibling->keys_.data() + n,
                   (sibling->num_keys_ - n) * sizeof(V));
      sibling->num_keys_ -= n;
      if constexpr (use_simd_) {
        for (attr_t k = 0; k < n; ++k) {
          sibling->keys_[sibling->num_keys_ + k] =
              std::numeric_limits<K>::max();
        }
      }
    } else {
      // brings one key from parent
      node->keys_.push_back(std::move(parent->keys_[node->index_]));
      // brings n - 1 keys from sibling
      std::ranges::move(sibling->keys_ | std::views::take(n - 1),
                        std::back_inserter(node->keys_));
      // parent brings one key from sibling
      parent->keys_[node->index_] = std::move(sibling->keys_[n - 1]);
      std::shift_left(sibling->keys_.begin(), sibling->keys_.end(), n);
      sibling->keys_.resize(sibling->nkeys() - n);
    }

    node->size_ += n;
    sibling->size_ -= n;

    if (!node->is_leaf()) {
      // brings n children from sibling
      attr_t orphan_size = 0;
      attr_t immigrant_index = static_cast<attr_t>(std::ssize(node->children_));
      for (auto &&immigrant : sibling->children_ | std::views::take(n)) {
        immigrant->parent_ = node;
        immigrant->index_ = immigrant_index++;
        orphan_size += immigrant->size_;
      }
      node->size_ += orphan_size;
      sibling->size_ -= orphan_size;

      std::ranges::move(sibling->children_ | std::views::take(n),
                        std::back_inserter(node->children_));
      std::shift_left(sibling->children_.begin(), sibling->children_.end(), n);
      for (attr_t idx = 0; idx < n; ++idx) {
        sibling->children_.pop_back();
      }
      attr_t sibling_index = 0;
      for (auto &&child : sibling->children_) {
        child->index_ = sibling_index++;
      }
    }
  }

  // node brings a key from parent
  // parent brings a key from left sibling
  // node brings a child from left sibling
  void right_rotate(Node *node) {
    auto parent = node->parent_;
    assert(node && parent && parent->children_[node->index_].get() == node &&
           node->index_ - 1 >= 0 &&
           parent->children_[node->index_ - 1]->can_take_key());
    auto sibling = parent->children_[node->index_ - 1].get();

    if constexpr (is_disk_) {
      std::memmove(node->keys_.data() + 1, node->keys_.data(),
                   node->num_keys_ * sizeof(V));
      node->num_keys_++;
      node->keys_[0] = parent->keys_[node->index_ - 1];
      parent->keys_[node->index_ - 1] = sibling->keys_[sibling->num_keys_ - 1];
      sibling->num_keys_--;
      if constexpr (use_simd_) {
        sibling->keys_[sibling->num_keys_] = std::numeric_limits<K>::max();
      }
    } else {
      node->keys_.insert(node->keys_.begin(),
                         std::move(parent->keys_[node->index_ - 1]));
      parent->keys_[node->index_ - 1] = std::move(sibling->keys_.back());
      sibling->keys_.pop_back();
    }

    node->size_++;
    sibling->size_--;

    if (!node->is_leaf()) {
      const auto orphan_size = sibling->children_.back()->size_;
      node->size_ += orphan_size;
      sibling->size_ -= orphan_size;

      sibling->children_.back()->parent_ = node;
      sibling->children_.back()->index_ = 0;

      node->children_.insert(node->children_.begin(),
                             std::move(sibling->children_.back()));
      sibling->children_.pop_back();
      for (auto &&child : node->children_ | std::views::drop(1)) {
        child->index_++;
      }
    }
  }

  // right_rotate() * n
  void right_rotate_n(Node *node, attr_t n) {
    assert(n >= 1);
    if (n == 1) {
      right_rotate(node);
      return;
    }

    auto parent = node->parent_;
    assert(node && parent && parent->children_[node->index_].get() == node &&
           node->index_ - 1 >= 0);
    auto sibling = parent->children_[node->index_ - 1].get();
    assert(sibling->nkeys() >= (Fanout - 1) + n);

    if constexpr (is_disk_) {
      std::memcpy(node->keys_.data() + node->num_keys_,
                  sibling->keys_.data() + (sibling->num_keys_ - n),
                  (n - 1) * sizeof(V));
      node->num_keys_ += (n - 1);
      node->keys_[node->num_keys_] = parent->keys_[node->index_ - 1];
      node->num_keys_++;
      parent->keys_[node->index_ - 1] = sibling->keys_[sibling->num_keys_ - 1];

      std::rotate(
          std::make_reverse_iterator(node->keys_.begin() + node->num_keys_),
          std::make_reverse_iterator(node->keys_.begin() + node->num_keys_ - n),
          node->keys_.rend());
      sibling->num_keys_ -= n;
      if constexpr (use_simd_) {
        sibling->keys_[sibling->num_keys_] = std::numeric_limits<K>::max();
      }
    } else {
      // brings n - 1 keys from sibling
      std::ranges::move(sibling->keys_ |
                            std::views::drop(sibling->nkeys() - n) |
                            std::views::take(n - 1),
                        std::back_inserter(node->keys_));
      // brings one key from parent
      node->keys_.push_back(std::move(parent->keys_[node->index_ - 1]));
      // parent brings one key from sibling
      parent->keys_[node->index_ - 1] = std::move(sibling->keys_.back());
      // right rotate n
      std::ranges::rotate(node->keys_ | std::views::reverse,
                          node->keys_.rbegin() + n);
      sibling->keys_.resize(sibling->nkeys() - n);
    }

    node->size_ += n;
    sibling->size_ -= n;

    if (!node->is_leaf()) {
      // brings n children from sibling
      attr_t orphan_size = 0;
      attr_t immigrant_index = 0;
      for (auto &&immigrant :
           sibling->children_ |
               std::views::drop(std::ssize(sibling->children_) - n)) {
        immigrant->parent_ = node;
        immigrant->index_ = immigrant_index++;
        orphan_size += immigrant->size_;
      }
      node->size_ += orphan_size;
      sibling->size_ -= orphan_size;

      std::ranges::move(
          sibling->children_ |
              std::views::drop(std::ssize(sibling->children_) - n),
          std::back_inserter(node->children_));
      std::ranges::rotate(node->children_ | std::views::reverse,
                          node->children_.rbegin() + n);
      for (attr_t idx = 0; idx < n; ++idx) {
        sibling->children_.pop_back();
      }
      attr_t child_index = n;
      for (auto &&child : node->children_ | std::views::drop(n)) {
        child->index_ = child_index++;
      }
    }
  }

  auto get_lb(const K &key, const Node *x) const noexcept {
    if constexpr (use_simd_) {
      return get_lb_simd<K, CompIsLess>(key, x->keys_.data(),
                                        x->keys_.data() + 2 * Fanout);
    } else if constexpr (use_linsearch_) {
      auto lbcomp = [&key](const K &other) { return Comp{}(other, key); };
      return std::distance(
          x->keys_.begin(),
          std::ranges::find_if_not(
              x->keys_.begin(), x->keys_.begin() + x->nkeys(), lbcomp, Proj{}));
    } else {
      return std::distance(x->keys_.begin(),
                           std::ranges::lower_bound(
                               x->keys_.begin(), x->keys_.begin() + x->nkeys(),
                               key, Comp{}, Proj{}));
    }
  }

  auto get_ub(const K &key, const Node *x) const noexcept {
    if constexpr (use_simd_) {
      return get_ub_simd<K, CompIsLess>(key, x->keys_.data(),
                                        x->keys_.data() + 2 * Fanout);
    } else if constexpr (use_linsearch_) {
      auto ubcomp = [&key](const K &other) { return Comp{}(key, other); };
      return std::distance(x->keys_.begin(),
                           std::ranges::find_if(x->keys_.begin(),
                                                x->keys_.begin() + x->nkeys(),
                                                ubcomp, Proj{}));
    } else {
      return std::distance(x->keys_.begin(),
                           std::ranges::upper_bound(
                               x->keys_.begin(), x->keys_.begin() + x->nkeys(),
                               key, Comp{}, Proj{}));
    }
  }

  const_iterator_type search(const K &key) const {
    auto x = root_.get();
    while (x) {
      auto i = get_lb(key, x);
      if (i < x->nkeys() && key == Proj{}(x->keys_[i])) { // equal? key found
        return const_iterator_type(x, static_cast<attr_t>(i));
      } else if (x->is_leaf()) { // no child, key is not in the tree
        return cend();
      } else { // search on child between range
        x = x->children_[i].get();
      }
    }
    return cend();
  }

  nonconst_iterator_type find_lower_bound(const K &key, bool climb = true) {
    auto x = root_.get();
    while (x) {
      auto i = get_lb(key, x);
      if (x->is_leaf()) {
        auto it = nonconst_iterator_type(x, static_cast<attr_t>(i));
        if (climb) {
          it.climb();
        }
        return it;
      } else {
        x = x->children_[i].get();
      }
    }
    return nonconst_iterator_type(end());
  }

  const_iterator_type find_lower_bound(const K &key, bool climb = true) const {
    auto x = root_.get();
    while (x) {
      auto i = get_lb(key, x);
      if (x->is_leaf()) {
        auto it = const_iterator_type(x, static_cast<attr_t>(i));
        if (climb) {
          it.climb();
        }
        return it;
      } else {
        x = x->children_[i].get();
      }
    }
    return cend();
  }

  nonconst_iterator_type find_upper_bound(const K &key, bool climb = true) {
    auto x = root_.get();
    while (x) {
      auto i = get_ub(key, x);
      if (x->is_leaf()) {
        auto it = nonconst_iterator_type(x, static_cast<attr_t>(i));
        if (climb) {
          it.climb();
        }
        return it;
      } else {
        x = x->children_[i].get();
      }
    }
    return nonconst_iterator_type(end());
  }

  const_iterator_type find_upper_bound(const K &key, bool climb = true) const {
    auto x = root_.get();
    while (x) {
      auto i = get_ub(key, x);
      if (x->is_leaf()) {
        auto it = const_iterator_type(x, static_cast<attr_t>(i));
        if (climb) {
          it.climb();
        }
        return it;
      } else {
        x = x->children_[i].get();
      }
    }
    return cend();
  }

  // split child[i] to child[i], child[i + 1]
  void split_child(Node *y) {
    assert(y);
    auto i = y->index_;
    Node *x = y->parent_;
    assert(x && y == x->children_[i].get() && y->is_full() && !x->is_full());

    // split y's 2 * t keys
    // y will have left t - 1 keys
    // y->keys_[t - 1] will be a key of y->parent_
    // right t keys of y will be taken by y's right sibling

    auto z = make_node(); // will be y's right sibling
    z->parent_ = x;
    z->index_ = i + 1;
    z->height_ = y->height_;

    // bring right t keys from y
    if constexpr (is_disk_) {
      std::memcpy(z->keys_.data(), y->keys_.data() + Fanout,
                  (y->num_keys_ - Fanout) * sizeof(V));
      z->num_keys_ = y->num_keys_ - Fanout;
    } else {
      std::ranges::move(y->keys_ | std::views::drop(Fanout),
                        std::back_inserter(z->keys_));
    }
    auto z_size = z->nkeys();
    if (!y->is_leaf()) {
      z->children_.reserve(2 * Fanout);
      // bring right half children from y
      std::ranges::move(y->children_ | std::views::drop(Fanout),
                        std::back_inserter(z->children_));
      for (auto &&child : z->children_) {
        child->parent_ = z.get();
        child->index_ -= Fanout;
        z_size += child->size_;
      }
      while (static_cast<attr_t>(std::ssize(y->children_)) > Fanout) {
        y->children_.pop_back();
      }
    }
    z->size_ = z_size;
    y->size_ -= (z_size + 1);

    x->children_.insert(x->children_.begin() + i + 1, std::move(z));
    for (auto &&child : x->children_ | std::views::drop(i + 2)) {
      child->index_++;
    }

    if constexpr (is_disk_) {
      std::memmove(x->keys_.data() + i + 1, x->keys_.data() + i,
                   (x->num_keys_ - i) * sizeof(V));
      x->num_keys_++;
      x->keys_[i] = y->keys_[Fanout - 1];
      y->num_keys_ = Fanout - 1;
      if constexpr (use_simd_) {
        for (attr_t k = Fanout - 1; k < 2 * Fanout; ++k) {
          y->keys_[k] = std::numeric_limits<K>::max();
        }
      }
    } else {
      x->keys_.insert(x->keys_.begin() + i, std::move(y->keys_[Fanout - 1]));
      y->keys_.resize(Fanout - 1);
    }
  }

  // merge child[i + 1] and key[i] into child[i]
  void merge_child(Node *y) {
    assert(y);
    auto i = y->index_;
    Node *x = y->parent_;
    assert(x && y == x->children_[i].get() && !x->is_leaf() && i >= 0 &&
           i + 1 < std::ssize(x->children_));
    auto sibling = x->children_[i + 1].get();
    assert(y->nkeys() + sibling->nkeys() <= 2 * Fanout - 2);

    auto immigrated_size = sibling->nkeys();

    if constexpr (is_disk_) {
      y->keys_[y->num_keys_] = x->keys_[i];
      y->num_keys_++;
      std::memcpy(y->keys_.data() + y->num_keys_, sibling->keys_.data(),
                  sibling->num_keys_ * sizeof(V));
      y->num_keys_ += sibling->num_keys_;
    } else {
      y->keys_.push_back(std::move(x->keys_[i]));
      // bring keys of child[i + 1]
      std::ranges::move(sibling->keys_, std::back_inserter(y->keys_));
    }

    // bring children of child[i + 1]
    if (!y->is_leaf()) {
      attr_t immigrant_index = static_cast<attr_t>(std::ssize(y->children_));
      for (auto &&child : sibling->children_) {
        child->parent_ = y;
        child->index_ = immigrant_index++;
        immigrated_size += child->size_;
      }
      std::ranges::move(sibling->children_, std::back_inserter(y->children_));
    }
    y->size_ += immigrated_size + 1;

    // shift children from i + 1 left by 1 (because child[i + 1] is merged)
    std::shift_left(x->children_.begin() + i + 1, x->children_.end(), 1);
    x->children_.pop_back();
    if constexpr (is_disk_) {
      std::memmove(x->keys_.data() + i, x->keys_.data() + i + 1,
                   (x->num_keys_ - (i + 1)) * sizeof(V));
      x->num_keys_--;
      if constexpr (use_simd_) {
        x->keys_[x->num_keys_] = std::numeric_limits<K>::max();
      }
    } else {
      // shift keys from i left by 1 (because key[i] is merged)
      std::shift_left(x->keys_.begin() + i, x->keys_.end(), 1);
      x->keys_.pop_back();
    }

    for (auto &&child : x->children_ | std::views::drop(i + 1)) {
      child->index_--;
    }
  }

  // only used in join() when join() is called by split()
  // preinvariant: x is the leftmost of the root (left side)
  // or the rightmost (right side)

  // (left side) merge child[0], child[1] if necessary, and propagate to
  // possibly the root for right side it's child[n - 2], child[n - 1]
  void try_merge(Node *x, bool left_side) {
    assert(x && !x->is_leaf());
    if (std::ssize(x->children_) < 2) {
      return;
    }
    if (left_side) {
      auto first = x->children_[0].get();
      auto second = x->children_[1].get();

      if (first->nkeys() + second->nkeys() <= 2 * Fanout - 2) {
        // just merge to one node
        merge_child(first);
      } else if (first->nkeys() < Fanout - 1) {
        // first borrows key from second
        auto deficit = (Fanout - 1 - first->nkeys());

        // this is mathematically true, otherwise
        // #(first.keys) + #(second.keys) < 2 * t - 2, so it should be merged
        // before
        assert(second->nkeys() > deficit + (Fanout - 1));
        left_rotate_n(first, deficit);
      }
    } else {
      auto rfirst = x->children_.back().get();
      auto rsecond = x->children_[std::ssize(x->children_) - 2].get();

      if (rfirst->nkeys() + rsecond->nkeys() <= 2 * Fanout - 2) {
        // just merge to one node
        merge_child(rsecond);
      } else if (rfirst->nkeys() < Fanout - 1) {
        // rfirst borrows key from rsecond
        auto deficit = (Fanout - 1 - rfirst->nkeys());

        assert(rsecond->nkeys() > deficit + (Fanout - 1));
        right_rotate_n(rfirst, deficit);
      } else if (rsecond->nkeys() < Fanout - 1) {
        // rsecond borrows key from rfirst
        auto deficit = (Fanout - 1 - rsecond->nkeys());

        assert(rfirst->nkeys() > deficit + (Fanout - 1));
        left_rotate_n(rsecond, deficit);
      }
    }
  }

  template <typename T>
  iterator_type
  insert_leaf(Node *node, attr_t i,
              T &&value) requires std::is_same_v<std::remove_cvref_t<T>, V> {
    assert(node && node->is_leaf() && !node->is_full());
    bool update_begin = (empty() || Comp{}(Proj{}(value), Proj{}(*begin_)));

    if constexpr (is_disk_) {
      std::memmove(node->keys_.data() + i + 1, node->keys_.data() + i,
                   (node->num_keys_ - i) * sizeof(V));
      node->keys_[i] = std::forward<T>(value);
      node->num_keys_++;
    } else {
      node->keys_.insert(node->keys_.begin() + i, std::forward<T>(value));
    }
    iterator_type iter(node, i);
    if (update_begin) {
      assert(node == leftmost_leaf(root_.get()) && i == 0);
      begin_ = iter;
    }

    auto curr = node;
    while (curr) {
      curr->size_++;
      curr = curr->parent_;
    }

    assert(verify());
    return iter;
  }

  template <typename T>
  iterator_type insert_ub(T &&key) requires(
      AllowDup &&std::is_same_v<std::remove_cvref_t<T>, V>) {
    auto x = root_.get();
    while (true) {
      auto i = get_ub(Proj{}(key), x);
      if (x->is_leaf()) {
        return insert_leaf(x, static_cast<attr_t>(i), std::forward<T>(key));
      } else {
        if (x->children_[i]->is_full()) {
          split_child(x->children_[i].get());
          if (Comp{}(Proj{}(x->keys_[i]), Proj{}(key))) {
            ++i;
          }
        }
        x = x->children_[i].get();
      }
    }
  }

  template <typename T>
  std::pair<iterator_type, bool>
  insert_lb(T &&key) requires(!AllowDup &&
                              std::is_same_v<std::remove_cvref_t<T>, V>) {
    auto x = root_.get();
    while (true) {
      auto i = get_lb(Proj{}(key), x);
      if (i < x->nkeys() && Proj{}(key) == Proj{}(x->keys_[i])) {
        return {iterator_type(x, static_cast<attr_t>(i)), false};
      } else if (x->is_leaf()) {
        return {insert_leaf(x, static_cast<attr_t>(i), std::forward<T>(key)),
                true};
      } else {
        if (x->children_[i]->is_full()) {
          split_child(x->children_[i].get());
          if (Proj{}(key) == Proj{}(x->keys_[i])) {
            return {iterator_type(x, static_cast<attr_t>(i)), false};
          } else if (Comp{}(Proj{}(x->keys_[i]), Proj{}(key))) {
            ++i;
          }
        }
        x = x->children_[i].get();
      }
    }
  }

  iterator_type erase_leaf(Node *node, attr_t i) {
    assert(node && i >= 0 && i < node->nkeys() && node->is_leaf() &&
           !node->has_minimal_keys());
    bool update_begin = (begin_ == const_iterator_type(node, i));
    if constexpr (is_disk_) {
      std::memmove(node->keys_.data() + i, node->keys_.data() + i + 1,
                   (node->num_keys_ - (i + 1)) * sizeof(V));
      node->num_keys_--;
      if constexpr (use_simd_) {
        node->keys_[node->num_keys_] = std::numeric_limits<K>::max();
      }
    } else {
      std::shift_left(node->keys_.begin() + i, node->keys_.end(), 1);
      node->keys_.pop_back();
    }
    iterator_type iter(node, i);
    iter.climb();
    if (update_begin) {
      begin_ = iter;
    }
    auto curr = node;
    while (curr) {
      curr->size_--;
      curr = curr->parent_;
    }
    assert(verify());
    return iter;
  }

  size_t erase_lb(Node *x, const K &key) requires(!AllowDup) {
    while (true) {
      auto i = get_lb(key, x);
      if (i < x->nkeys() && key == Proj{}(x->keys_[i])) {
        // key found
        assert(x->is_leaf() || i + 1 < std::ssize(x->children_));
        if (x->is_leaf()) {
          erase_leaf(x, static_cast<attr_t>(i));
          return 1;
        } else if (x->children_[i]->can_take_key()) {
          // swap key with pred
          nonconst_iterator_type iter(x, static_cast<attr_t>(i));
          auto pred = std::prev(iter);
          assert(pred.node_ == rightmost_leaf(x->children_[i].get()));
          std::iter_swap(pred, iter);
          // search pred
          x = x->children_[i].get();
        } else if (x->children_[i + 1]->can_take_key()) {
          // swap key with succ
          nonconst_iterator_type iter(x, static_cast<attr_t>(i));
          auto succ = std::next(iter);
          assert(succ.node_ == leftmost_leaf(x->children_[i + 1].get()));
          std::iter_swap(succ, iter);
          // search succ
          x = x->children_[i + 1].get();
        } else {
          auto next = x->children_[i].get();
          merge_child(next);
          promote_root_if_necessary();
          x = next;
        }
      } else if (x->is_leaf()) {
        // no child, key is not in the tree
        return 0;
      } else {
        auto next = x->children_[i].get();
        if (x->children_[i]->has_minimal_keys()) {
          if (i + 1 < std::ssize(x->children_) &&
              x->children_[i + 1]->can_take_key()) {
            left_rotate(next);
          } else if (i - 1 >= 0 && x->children_[i - 1]->can_take_key()) {
            right_rotate(next);
          } else if (i + 1 < std::ssize(x->children_)) {
            merge_child(next);
            promote_root_if_necessary();
          } else if (i - 1 >= 0) {
            next = x->children_[i - 1].get();
            merge_child(next);
            promote_root_if_necessary();
          }
        }
        x = next;
      }
    }
  }

  iterator_type erase_hint([[maybe_unused]] const V &value,
                           std::vector<attr_t> &hints) {
    auto x = root_.get();
    while (true) {
      auto i = hints.back();
      hints.pop_back();
      if (hints.empty()) {
        // key found
        assert(i < x->nkeys() && value == x->keys_[i]);
        assert(x->is_leaf() || i + 1 < std::ssize(x->children_));
        if (x->is_leaf()) {
          return erase_leaf(x, i);
        } else if (x->children_[i]->can_take_key()) {
          // swap key with pred
          nonconst_iterator_type iter(x, i);
          auto pred = std::prev(iter);
          assert(pred.node_ == rightmost_leaf(x->children_[i].get()));
          std::iter_swap(pred, iter);
          // search pred
          x = x->children_[i].get();
          auto curr = x;
          assert(curr->index_ == i);
          while (!curr->is_leaf()) {
            hints.push_back(static_cast<attr_t>(std::ssize(curr->children_)) -
                            1);
            curr = curr->children_.back().get();
          }
          hints.push_back(curr->nkeys() - 1);
          std::ranges::reverse(hints);
        } else if (x->children_[i + 1]->can_take_key()) {
          // swap key with succ
          nonconst_iterator_type iter(x, i);
          auto succ = std::next(iter);
          assert(succ.node_ == leftmost_leaf(x->children_[i + 1].get()));
          std::iter_swap(succ, iter);
          // search succ
          x = x->children_[i + 1].get();
          auto curr = x;
          assert(curr->index_ == i + 1);
          while (!curr->is_leaf()) {
            hints.push_back(0);
            curr = curr->children_.front().get();
          }
          hints.push_back(0);
        } else {
          auto next = x->children_[i].get();
          merge_child(next);
          promote_root_if_necessary();
          x = next;
          // i'th key of x is now t - 1'th key of x->children_[i]
          hints.push_back(Fanout - 1);
        }
      } else {
        assert(!hints.empty());
        auto next = x->children_[i].get();
        if (x->children_[i]->has_minimal_keys()) {
          if (i + 1 < std::ssize(x->children_) &&
              x->children_[i + 1]->can_take_key()) {
            left_rotate(x->children_[i].get());
          } else if (i - 1 >= 0 && x->children_[i - 1]->can_take_key()) {
            right_rotate(x->children_[i].get());
            // x->children_[i] stuffs are shifted right by 1
            hints.back() += 1;
          } else if (i + 1 < std::ssize(x->children_)) {
            merge_child(next);
            promote_root_if_necessary();
          } else if (i - 1 >= 0) {
            next = x->children_[i - 1].get();
            merge_child(next);
            promote_root_if_necessary();
            // x->children_[i] stuffs are shifted right by t
            hints.back() += Fanout;
          }
        }
        x = next;
      }
    }
  }

private:
  static constexpr attr_t bulk_erase_threshold = 30;

protected:
  size_type erase_range(const_iterator_type first, const_iterator_type last) {
    if (first == cend()) {
      return 0;
    }
    if (first == begin_ && last == cend()) {
      auto cnt = size();
      clear();
      return cnt;
    }

    attr_t first_order = get_order(first);
    attr_t last_order = (last == cend()) ? root_->size_ : get_order(last);

    attr_t cnt = last_order - first_order;
    if (cnt < bulk_erase_threshold) {
      first.climb();
      for (attr_t i = 0; i < cnt; ++i) {
        first = erase(first);
      }
      return cnt;
    }

    K first_key = Proj{}(*first);

    auto [tree1, tree2] = split_to_two_trees(first, last);
    auto final_tree = join(std::move(tree1), first_key, std::move(tree2));
    final_tree.erase(final_tree.lower_bound(first_key));

    this->swap(final_tree);
    return cnt;
  }

  V get_kth(attr_t idx) const {
    auto x = root_.get();
    while (x) {
      if (x->is_leaf()) {
        assert(idx >= 0 && idx < x->nkeys());
        return x->keys_[idx];
      } else {
        assert(!x->children_.empty());
        attr_t i = 0;
        const auto n = x->nkeys();
        Node *next = nullptr;
        for (; i < n; ++i) {
          auto child_sz = x->children_[i]->size_;
          if (idx < child_sz) {
            next = x->children_[i].get();
            break;
          } else if (idx == child_sz) {
            return x->keys_[i];
          } else {
            idx -= child_sz + 1;
          }
        }
        if (i == n) {
          next = x->children_[n].get();
        }
        x = next;
      }
    }
    btree_fatal("unreachable");
  }

  attr_t get_order(const_iterator_type iter) const {
    auto [node, idx] = iter;
    attr_t order = 0;
    assert(node);
    if (!node->is_leaf()) {
      for (attr_t i = 0; i <= idx; ++i) {
        order += node->children_[i]->size_;
      }
    }
    order += idx;
    while (node->parent_) {
      for (attr_t i = 0; i < node->index_; ++i) {
        order += node->parent_->children_[i]->size_;
      }
      order += node->index_;
      node = node->parent_;
    }
    return order;
  }

public:
  iterator_type find(const K &key) { return iterator_type(search(key)); }

  const_iterator_type find(const K &key) const { return search(key); }

  bool contains(const K &key) const { return search(key) != cend(); }

  iterator_type lower_bound(const K &key) {
    return iterator_type(find_lower_bound(key));
  }

  const_iterator_type lower_bound(const K &key) const {
    return const_iterator_type(find_lower_bound(key));
  }

  iterator_type upper_bound(const K &key) {
    return iterator_type(find_upper_bound(key));
  }

  const_iterator_type upper_bound(const K &key) const {
    return const_iterator_type(find_upper_bound(key));
  }

  std::ranges::subrange<iterator_type> equal_range(const K &key) {
    return {iterator_type(find_lower_bound(key)),
            iterator_type(find_upper_bound(key))};
  }

  std::ranges::subrange<const_iterator_type> equal_range(const K &key) const {
    return {const_iterator_type(find_lower_bound(key)),
            const_iterator_type(find_upper_bound(key))};
  }

  std::ranges::subrange<const_iterator_type> enumerate(const K &a,
                                                       const K &b) const {
    if (Comp{}(b, a)) {
      btree_fatal("b < a in enumerate()");
    }
    return {const_iterator_type(find_lower_bound(a)),
            const_iterator_type(find_upper_bound(b))};
  }

  V kth(attr_t idx) const {
    if (idx >= root_->size_) {
      btree_fatal("in kth() k >= size()");
    }
    return get_kth(idx);
  }

  attr_t order(const_iterator_type iter) const {
    if (iter == cend()) {
      btree_fatal("attempt to get order in end()");
    }
    return get_order(iter);
  }

  attr_t count(const K &key) const requires(AllowDup) {
    auto first = find_lower_bound(key);
    auto last = find_upper_bound(key);
    attr_t first_order = get_order(first);
    attr_t last_order = (last == cend()) ? root_->size_ : get_order(last);
    return last_order - first_order;
  }

protected:
  template <typename T>
  std::conditional_t<AllowDup, iterator_type, std::pair<iterator_type, bool>>
  insert_value(T &&key) requires(std::is_same_v<std::remove_cvref_t<T>, V>) {
    if (root_->is_full()) {
      // if root is full then make it as a child of the new root
      auto new_root = make_node();
      root_->parent_ = new_root.get();
      new_root->size_ = root_->size_;
      new_root->height_ = root_->height_ + 1;
      new_root->children_.reserve(Fanout * 2);
      new_root->children_.push_back(std::move(root_));
      root_ = std::move(new_root);
      // and split
      split_child(root_->children_[0].get());
    }
    if constexpr (AllowDup) {
      return insert_ub(std::forward<T>(key));
    } else {
      return insert_lb(std::forward<T>(key));
    }
  }

  std::vector<attr_t> get_path_from_root(const_iterator_type iter) const {
    auto node = iter.node_;
    std::vector<attr_t> hints;
    hints.push_back(iter.index_);
    while (node && node->parent_) {
      hints.push_back(node->index_);
      node = node->parent_;
    }
    return hints;
  }

public:
  std::conditional_t<AllowDup, iterator_type, std::pair<iterator_type, bool>>
  insert(const V &key) {
    return insert_value(key);
  }

  std::conditional_t<AllowDup, iterator_type, std::pair<iterator_type, bool>>
  insert(V &&key) {
    return insert_value(std::move(key));
  }

  template <typename... Args>
  std::conditional_t<AllowDup, iterator_type, std::pair<iterator_type, bool>>
  emplace(Args &&...args) requires std::is_constructible_v<V, Args...> {
    V val{std::forward<Args>(args)...};
    return insert_value(std::move(val));
  }

  template <typename T>
  auto &operator[](T &&raw_key) requires(!is_set_ && !AllowDup) {
    if (root_->is_full()) {
      // if root is full then make it as a child of the new root
      auto new_root = make_node();

      root_->parent_ = new_root.get();
      new_root->size_ = root_->size_;
      new_root->height_ = root_->height_ + 1;
      new_root->children_.reserve(Fanout * 2);
      new_root->children_.push_back(std::move(root_));
      root_ = std::move(new_root);
      // and split
      split_child(root_->children_[0].get());
    }

    K key{std::forward<T>(raw_key)};
    auto x = root_.get();
    while (true) {
      auto i = get_lb(key, x);
      if (i < x->nkeys() && key == Proj{}(x->keys_[i])) {
        return iterator_type(x, static_cast<attr_t>(i))->second;
      } else if (x->is_leaf()) {
        V val{std::move(key), {}};
        return insert_leaf(x, static_cast<attr_t>(i), std::move(val))->second;
      } else {
        if (x->children_[i]->is_full()) {
          split_child(x->children_[i].get());
          if (key == Proj{}(x->keys_[i])) {
            return iterator_type(x, static_cast<attr_t>(i))->second;
          } else if (Comp{}(Proj{}(x->keys_[i]), key)) {
            ++i;
          }
        }
        x = x->children_[i].get();
      }
    }
  }

  template <std::forward_iterator Iter>
  requires std::is_constructible_v<V, std::iter_reference_t<Iter>>
      size_type insert_range(Iter first, Iter last) {
    auto [min_elem, max_elem] =
        std::ranges::minmax_element(first, last, Comp{}, Proj{});
    auto lb = find_lower_bound(*min_elem);
    auto ub = find_upper_bound(*max_elem);
    if (lb != ub) {
      size_type sz = 0;
      for (; first != last; ++first) {
        if constexpr (AllowDup) {
          insert(*first);
          sz++;
        } else {
          auto [_, inserted] = insert(*first);
          if (inserted) {
            sz++;
          }
        }
      }
      return sz;
    } else {
      BTreeBase tree_mid{alloc_};
      for (; first != last; ++first) {
        tree_mid.insert(*first);
      }
      auto sz = tree_mid.size();
      auto [tree_left, tree_right] = split_to_two_trees(lb, ub);
      auto tree_leftmid = join(std::move(tree_left), std::move(tree_mid));
      auto final_tree = join(std::move(tree_leftmid), std::move(tree_right));
      this->swap(final_tree);
      return sz;
    }
  }

  template <std::ranges::forward_range R>
  requires std::is_constructible_v<V, std::ranges::range_reference_t<R>>
      size_type insert_range(R &&r) {
    return insert_range(r.begin(), r.end());
  }

  const_iterator_type erase(const_iterator_type iter) {
    if (iter == cend()) {
      btree_fatal("attempt to erase cend()");
    }
    std::vector<attr_t> hints = get_path_from_root(iter);
    V value(std::move(*iter));
    return erase_hint(value, hints);
  }

  size_type erase(const K &key) {
    if constexpr (AllowDup) {
      return erase_range(const_iterator_type(find_lower_bound(key, false)),
                         const_iterator_type(find_upper_bound(key, false)));
    } else {
      return erase_lb(root_.get(), key);
    }
  }

  size_type erase_range(const K &a, const K &b) {
    if (Comp{}(b, a)) {
      btree_fatal("b < a in erase_range()");
    }
    return erase_range(const_iterator_type(find_lower_bound(a)),
                       const_iterator_type(find_upper_bound(b)));
  }

  template <typename Pred> size_type erase_if(Pred pred) {
    auto old_size = size();
    auto it = begin_;
    for (; it != end();) {
      if (pred(*it)) {
        it = erase(it);
      } else {
        ++it;
      }
    }
    return old_size - size();
  }

private:
  // serialization and deserialization
  static constexpr std::uint64_t begin_code = 0x6567696e; // 'begin'
  static constexpr std::uint64_t end_code = 0x656e64;     // 'end'

  // for tree, we write a root height

  // for each node, we only read/write two information:
  // 1. number of keys (attr_t, int32)
  // 2. byte stream for key data (sizeof(V) * nkeys())

  // all other information can be inferred during tree traversal

  // number of max bytes for serializing/deserializing a single node
  static constexpr std::size_t keydata_size = sizeof(V) * disk_max_nkeys;

  // maximum possible height for B-Tree
  // if height exceeds this value, this means that serialization/deserialization
  // size will exceed 16TB, much more likely a user mistake or a malicious
  // attack
  static constexpr std::size_t max_possible_height =
      (44UL - std::bit_width(static_cast<std::size_t>(2 * Fanout))) /
      std::bit_width(keydata_size);

public:
  friend std::istream &operator>>(std::istream &is,
                                  BTreeBase &tree) requires(is_disk_) {
    std::uint64_t tree_code = 0;
    if (!is.read(reinterpret_cast<char *>(&tree_code), sizeof(std::uint64_t))) {
      std::cerr << "Tree deserialization: begin code parse error\n";
      return is;
    }
    if (tree_code != begin_code) {
      std::cerr << "Tree deserialization: begin code is invalid\n";
      is.clear(std::ios_base::failbit);
      return is;
    }

    attr_t tree_height = 0;
    if (!is.read(reinterpret_cast<char *>(&tree_height), sizeof(attr_t))) {
      std::cerr << "Tree deserialization: tree height parse error\n";
      return is;
    }
    if (static_cast<std::size_t>(tree_height) > max_possible_height) {
      std::cerr << "Tree deserialization: height is invalid\n";
      is.clear(std::ios_base::failbit);
      return is;
    }

    auto node = tree.root_.get();
    assert(node);

    if (!tree.deserialize_node(is, node, 0, tree_height)) {
      return is;
    }
    if (!is.read(reinterpret_cast<char *>(&tree_code), sizeof(std::uint64_t))) {
      std::cerr << "Tree deserialization: end code parse error\n";
      tree.clear();
      return is;
    }
    if (tree_code != end_code) {
      std::cerr << "Tree deserialization: end code is invalid\n";
      tree.clear();
      is.clear(std::ios_base::failbit);
      return is;
    }
    tree.set_begin();
    assert(tree.verify());
    return is;
  }

protected:
  // preorder DFS traversal
  bool deserialize_node(std::istream &is, Node *node, attr_t node_index,
                        attr_t node_height) requires(is_disk_) {
    assert(node);
    node->index_ = node_index;
    node->height_ = node_height;
    if (!is.read(reinterpret_cast<char *>(&node->num_keys_), sizeof(attr_t))) {
      std::cerr << "Tree deserialization: nkeys parse error\n";
      return false;
    }
    if (node->num_keys_ >= 2 * Fanout ||
        (node != root_.get() && node->num_keys_ < Fanout - 1) ||
        node->num_keys_ < 0) {
      std::cerr << "Tree deserialization: nkeys is invalid\n";
      is.clear(std::ios_base::failbit);
      return false;
    }
    if (!is.read(reinterpret_cast<char *>(node->keys_.data()),
                 static_cast<std::size_t>(node->num_keys_) * sizeof(V))) {
      std::cerr << "Tree deserialization: key data read error\n";
      return false;
    }
    node->size_ = node->num_keys_;
    if (node_height > 0) {
      for (attr_t i = 0; i <= node->num_keys_; ++i) {
        node->children_.push_back(make_node());
        node->children_[i]->parent_ = node;
        if (!deserialize_node(is, node->children_[i].get(), i,
                              node_height - 1)) {
          return false;
        }
      }
    }
    if (node->parent_) {
      node->parent_->size_ += node->size_;
    }
    return true;
  }

public:
  friend std::ostream &operator<<(std::ostream &os,
                                  const BTreeBase &tree) requires(is_disk_) {
    std::uint64_t tree_code = begin_code;
    if (!os.write(reinterpret_cast<char *>(&tree_code),
                  sizeof(std::uint64_t))) {
      std::cerr << "Tree serialization: begin code write error\n";
      return os;
    }

    attr_t tree_height = tree.height();
    if (!os.write(reinterpret_cast<char *>(&tree_height), sizeof(attr_t))) {
      std::cerr << "Tree serialization: tree height write error\n";
      return os;
    }

    auto node = tree.root_.get();
    assert(node);

    if (!tree.serialize_node(os, node)) {
      return os;
    }
    tree_code = end_code;
    if (!os.write(reinterpret_cast<char *>(&tree_code),
                  sizeof(std::uint64_t))) {
      std::cerr << "Tree serialization: end code write error\n";
      return os;
    }
    return os;
  }

protected:
  // preorder DFS traversal
  bool serialize_node(std::ostream &os, const Node *node) const
      requires(is_disk_) {
    assert(node);
    if (!os.write(reinterpret_cast<const char *>(&node->num_keys_),
                  sizeof(attr_t))) {
      std::cerr << "Tree serialization: nkeys write error\n";
      return false;
    }
    if (!os.write(reinterpret_cast<const char *>(node->keys_.data()),
                  static_cast<std::size_t>(node->num_keys_) * sizeof(V))) {
      std::cerr << "Tree serialization: key data write error\n";
      return false;
    }
    if (node->height_ > 0) {
      for (attr_t i = 0; i <= node->num_keys_; ++i) {
        if (!serialize_node(os, node->children_[i].get())) {
          return false;
        }
      }
    }
    return true;
  }

public:
  template <Containable K_, typename V_, attr_t Fanout_, typename Comp_,
            bool AllowDup_, template <typename T> class AllocTemplate_>
  friend struct join_helper;

protected:
  std::pair<BTreeBase, BTreeBase>
  split_to_two_trees(const_iterator_type iter_lb, const_iterator_type iter_ub) {
    BTreeBase tree_left(alloc_);
    BTreeBase tree_right(alloc_);
    iter_lb.dig();

    auto lindices = get_path_from_root(iter_lb);
    auto xl = iter_lb.node_;
    std::ranges::reverse(lindices);
    auto rindices = get_path_from_root(iter_ub);
    auto xr = iter_ub.node_;
    std::ranges::reverse(rindices);

    while (!lindices.empty()) {
      auto il = lindices.back();
      lindices.pop_back();

      auto lroot = tree_left.root_.get();

      if (xl->is_leaf()) {
        assert(lroot->size_ == 0);

        if (il > 0) {
          if constexpr (is_disk_) {
            std::memcpy(lroot->keys_.data(), xl->keys_.data(), il * sizeof(V));
            lroot->num_keys_ += il;
          } else {
            // send left i keys to lroot
            std::ranges::move(xl->keys_ | std::views::take(il),
                              std::back_inserter(lroot->keys_));
          }
          lroot->size_ += il;
        }

        xl = xl->parent_;
      } else {
        if (il > 0) {
          BTreeBase supertree_left(alloc_);
          auto slroot = supertree_left.root_.get();
          // sltree takes left i - 1 keys, i children
          // middle key is key[i - 1]

          assert(slroot->size_ == 0);

          if constexpr (is_disk_) {
            std::memcpy(slroot->keys_.data(), xl->keys_.data(),
                        (il - 1) * sizeof(V));
            slroot->num_keys_ += (il - 1);
          } else {
            std::ranges::move(xl->keys_ | std::views::take(il - 1),
                              std::back_inserter(slroot->keys_));
          }
          slroot->size_ += (il - 1);

          slroot->children_.reserve(Fanout * 2);

          std::ranges::move(xl->children_ | std::views::take(il),
                            std::back_inserter(slroot->children_));
          slroot->height_ = slroot->children_[0]->height_ + 1;
          for (auto &&sl_child : slroot->children_) {
            sl_child->parent_ = slroot;
            slroot->size_ += sl_child->size_;
          }

          supertree_left.promote_root_if_necessary();
          supertree_left.set_begin();

          BTreeBase new_tree_left =
              join(std::move(supertree_left), std::move(xl->keys_[il - 1]), std::move(tree_left));
          tree_left = std::move(new_tree_left);
        }

        xl = xl->parent_;
      }
    }
    while (!rindices.empty()) {
      auto ir = rindices.back();
      rindices.pop_back();

      auto rroot = tree_right.root_.get();

      if (xr->is_leaf()) {
        assert(rroot->size_ == 0);

        if (ir < xr->nkeys()) {
          auto immigrants = xr->nkeys() - ir;
          if constexpr (is_disk_) {
            std::memcpy(rroot->keys_.data(), xr->keys_.data() + ir,
                        immigrants * sizeof(V));
            rroot->num_keys_ += immigrants;
          } else {
            // send right n - (i + 1) keys to rroot
            std::ranges::move(xr->keys_ | std::views::drop(ir),
                              std::back_inserter(rroot->keys_));
          }
          rroot->size_ += immigrants;
        }

        xr = xr->parent_;
      } else {

        if (ir + 1 < std::ssize(xr->children_)) {
          BTreeBase supertree_right(alloc_);
          auto srroot = supertree_right.root_.get();
          // srtree takes right n - (i + 1) keys, n - (i + 1) children
          // middle key is key[i]

          assert(srroot->size_ == 0);

          auto immigrants = xr->nkeys() - (ir + 1);
          if constexpr (is_disk_) {
            std::memcpy(srroot->keys_.data(), xr->keys_.data() + (ir + 1),
                        immigrants * sizeof(V));
            srroot->num_keys_ += immigrants;
          } else {
            std::ranges::move(xr->keys_ | std::views::drop(ir + 1),
                              std::back_inserter(srroot->keys_));
          }
          srroot->size_ += immigrants;

          srroot->children_.reserve(Fanout * 2);

          std::ranges::move(xr->children_ | std::views::drop(ir + 1),
                            std::back_inserter(srroot->children_));
          srroot->height_ = srroot->children_[0]->height_ + 1;
          attr_t sr_index = 0;
          for (auto &&sr_child : srroot->children_) {
            sr_child->parent_ = srroot;
            sr_child->index_ = sr_index++;
            srroot->size_ += sr_child->size_;
          }

          supertree_right.promote_root_if_necessary();
          supertree_right.set_begin();

          BTreeBase new_tree_right =
              join(std::move(tree_right), std::move(xr->keys_[ir]),
                   std::move(supertree_right));
          tree_right = std::move(new_tree_right);
        }

        xr = xr->parent_;
      }
    }
    assert(!xl && !xr && lindices.empty() && rindices.empty());
    assert(tree_left.verify());
    assert(tree_right.verify());
    clear();
    return {std::move(tree_left), std::move(tree_right)};
  }

public:
    template <Containable K_, typename V_, attr_t Fanout_, typename Comp_,
              bool AllowDup_, template <typename T> class AllocTemplate_, typename T>
    friend struct split_helper;
};

template <Containable K, typename V, attr_t Fanout, typename Comp,
          bool AllowDup, template <typename T> class AllocTemplate>
struct join_helper {
private:
    BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate> result_;

    using Tree = BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate>;
    using Node = typename Tree::node_type;
    using Proj = typename Tree::Proj;
    static constexpr bool is_disk_ = Tree::is_disk_;

public:
  join_helper(BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate> &&tree_left,
              BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate> &&tree_right) {
    if (tree_left.empty()) {
      result_ = std::move(tree_right);
    } else if (tree_right.empty()) {
      result_ = std::move(tree_left);
    } else {
      auto it = tree_right.begin();
      V mid_value = *it;
      tree_right.erase(it);
      result_ = join(std::move(tree_left), std::move(mid_value),
                     std::move(tree_right));
  }
}

  template <typename T_>
  requires std::is_constructible_v<V, std::remove_cvref_t<T_>>
  join_helper(BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate> &&tree_left,
              T_                                                     &&raw_value,
              BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate> &&tree_right)  {

  V mid_value{std::forward<T_>(raw_value)};
  if ((!tree_left.empty() &&
       Comp{}(Proj{}(mid_value), Proj{}(*tree_left.crbegin()))) ||
      (!tree_right.empty() &&
       Comp{}(Proj{}(*tree_right.cbegin()), Proj{}(mid_value)))) {
    btree_fatal("Join() key order is invalid\n");
  }
  if (tree_left.alloc_ != tree_right.alloc_) {
    btree_fatal("Join() two allocators are different\n");
  }

  auto height_left = tree_left.root_->height_;
  auto height_right = tree_right.root_->height_;
  auto size_left = tree_left.root_->size_;
  auto size_right = tree_right.root_->size_;

  if (height_left >= height_right) {
    Tree new_tree = std::move(tree_left);
    attr_t curr_height = height_left;
    Node *curr = new_tree.root_.get();
    if (new_tree.root_->is_full()) {
      // if root is full then make it as a child of the new root
      auto new_root = new_tree.make_node();
      new_tree.root_->index_ = 0;
      new_tree.root_->parent_ = new_root.get();
      new_root->size_ = new_tree.root_->size_;
      new_root->height_ = new_tree.root_->height_ + 1;
      new_root->children_.reserve(Fanout * 2);
      new_root->children_.push_back(std::move(new_tree.root_));
      new_tree.root_ = std::move(new_root);
      // and split
      new_tree.split_child(new_tree.root_->children_[0].get());
      curr = new_tree.root_->children_[1].get();
    }
    assert(curr->height_ == height_left);

    while (curr && curr_height > height_right) {
      assert(!curr->is_leaf());
      curr_height--;

      if (curr->children_.back()->is_full()) {
        new_tree.split_child(curr->children_.back().get());
      }
      curr = curr->children_.back().get();
    }
    assert(curr_height == height_right);
    auto parent = curr->parent_;
    if (!parent) {
      // tree_left was empty or height of two trees were the same
      auto new_root = tree_left.make_node();
      new_root->height_ = new_tree.root_->height_ + 1;

      if constexpr (is_disk_) {
        new_root->keys_[new_root->num_keys_] = mid_value;
        new_root->num_keys_++;
      } else {
        new_root->keys_.push_back(std::move(mid_value));
      }

      new_root->children_.reserve(Fanout * 2);

      new_tree.root_->parent_ = new_root.get();
      new_tree.root_->index_ = 0;
      new_root->children_.push_back(std::move(new_tree.root_));

      tree_right.root_->parent_ = new_root.get();
      tree_right.root_->index_ = 1;
      new_root->children_.push_back(std::move(tree_right.root_));

      new_tree.root_ = std::move(new_root);
      new_tree.try_merge(new_tree.root_.get(), false);
      new_tree.promote_root_if_necessary();
      new_tree.root_->size_ = size_left + size_right + 1;
    } else {
      if constexpr (is_disk_) {
        parent->keys_[parent->num_keys_] = mid_value;
        parent->num_keys_++;
      } else {
        parent->keys_.push_back(std::move(mid_value));
      }

      tree_right.root_->parent_ = parent;
      tree_right.root_->index_ =
          static_cast<attr_t>(std::ssize(parent->children_));
      parent->children_.push_back(std::move(tree_right.root_));

      while (parent) {
        parent->size_ += (size_right + 1);
        new_tree.try_merge(parent, false);
        parent = parent->parent_;
      }
      new_tree.promote_root_if_necessary();
    }
    assert(new_tree.root_->size_ == size_left + size_right + 1);
    assert(new_tree.verify());
    result_ = std::move(new_tree);
  } else {
    Tree new_tree = std::move(tree_right);
    attr_t curr_height = height_right;
    Node *curr = new_tree.root_.get();
    if (new_tree.root_->is_full()) {
      // if root is full then make it as a child of the new root
      auto new_root = new_tree.make_node();
      new_tree.root_->index_ = 0;
      new_tree.root_->parent_ = new_root.get();
      new_root->size_ = new_tree.root_->size_;
      new_root->height_ = new_tree.root_->height_ + 1;
      new_root->children_.reserve(Fanout * 2);
      new_root->children_.push_back(std::move(new_tree.root_));
      new_tree.root_ = std::move(new_root);
      // and split
      new_tree.split_child(new_tree.root_->children_[0].get());
      curr = new_tree.root_->children_[0].get();
    }
    assert(curr->height_ == height_right);

    while (curr && curr_height > height_left) {
      assert(!curr->is_leaf());
      curr_height--;

      if (curr->children_.front()->is_full()) {
        new_tree.split_child(curr->children_[0].get());
      }
      curr = curr->children_.front().get();
    }
    assert(curr_height == height_left);
    auto parent = curr->parent_;
    assert(parent);
    if constexpr (is_disk_) {
      std::memmove(parent->keys_.data() + 1, parent->keys_.data(),
                   parent->num_keys_ * sizeof(V));
      parent->keys_[0] = mid_value;
      parent->num_keys_++;
    } else {
      parent->keys_.insert(parent->keys_.begin(), std::move(mid_value));
    }

    auto new_begin = tree_left.begin();
    tree_left.root_->parent_ = parent;
    tree_left.root_->index_ = 0;
    parent->children_.insert(parent->children_.begin(),
                             std::move(tree_left.root_));
    for (auto &&child : parent->children_ | std::views::drop(1)) {
      child->index_++;
    }
    while (parent) {
      parent->size_ += (size_left + 1);
      new_tree.try_merge(parent, true);
      parent = parent->parent_;
    }
    new_tree.promote_root_if_necessary();
    new_tree.begin_ = new_begin;
    assert(new_tree.root_->size_ == size_left + size_right + 1);
    assert(new_tree.verify());
    result_ = std::move(new_tree);
  }
}
  BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate>&& result() { return std::move(result_); }
};
template <Containable K, typename V, attr_t Fanout, typename Comp,
          bool AllowDup, template <typename T> class AllocTemplate, typename T>
struct split_helper {
private:
  std::pair<BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate>,
            BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate>> result_;
public:
  using Tree = BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate>;

  split_helper(BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate> &&tree,
               T                                                      &&raw_key)
      requires(std::is_constructible_v<K, std::remove_cvref_t<T>>) {
    if (tree.empty()) {
      Tree tree_left(tree.alloc_);
      Tree tree_right(tree.alloc_);
      result_ = {std::move(tree_left), std::move(tree_right)};
    } else {
      K mid_key{std::forward<T>(raw_key)};
      result_ = tree.split_to_two_trees(tree.find_lower_bound(mid_key, false),
                                        tree.find_upper_bound(mid_key, false));
    }
  }
  split_helper(BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate> &&tree,
               T                                                      &&raw_key1,
               T                                                      &&raw_key2)
      requires(std::is_constructible_v<K, std::remove_cvref_t<T>>) {
    if (tree.empty()) {
      Tree tree_left(tree.alloc_);
      Tree tree_right(tree.alloc_);
      result_ = {std::move(tree_left), std::move(tree_right)};
    } else {
      K key1{std::forward<T>(raw_key1)};
      K key2{std::forward<T>(raw_key2)};
      if (Comp{}(key2, key1)) {
        btree_fatal("split() key order is invalid\n");
      }
      result_ = tree.split_to_two_trees(tree.find_lower_bound(key1, false),
                                        tree.find_upper_bound(key2, false));
    }
  }
  std::pair<BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate>,
            BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate>> &&
  result() { return std::move(result_); }

};
template <Containable K, attr_t t = 64, typename Comp = std::ranges::less,
          template <typename T> class AllocTemplate = std::allocator>
using BTreeSet = BTreeBase<K, K, t, Comp, false, AllocTemplate>;

template <Containable K, attr_t t = 64, typename Comp = std::ranges::less,
          template <typename T> class AllocTemplate = std::allocator>
using BTreeMultiSet = BTreeBase<K, K, t, Comp, true, AllocTemplate>;

template <Containable K, Containable V, attr_t t = 64,
          typename Comp = std::ranges::less,
          template <typename T> class AllocTemplate = std::allocator>
using BTreeMap = BTreeBase<K, BTreePair<K, V>, t, Comp, false, AllocTemplate>;

template <Containable K, Containable V, attr_t t = 64,
          typename Comp = std::ranges::less,
          template <typename T> class AllocTemplate = std::allocator>
using BTreeMultiMap =
    BTreeBase<K, BTreePair<K, V>, t, Comp, true, AllocTemplate>;

template <Containable K, typename V, attr_t Fanout, typename Comp,
          bool AllowDup, template <typename T> class AllocTemplate>
BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate> join(BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate> &&tree_left,
                                                            BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate> &&tree_right) {
  return join_helper(std::move(tree_left), std::move(tree_right)).result();
}

template <Containable K, typename V, attr_t Fanout, typename Comp,
          bool AllowDup, template <typename T> class AllocTemplate, typename T_>
BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate> join(BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate> &&tree_left,
                                                            T_                                                     &&raw_value,
                                                            BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate> &&tree_right) {
  return join_helper(std::move(tree_left), std::move(raw_value), std::move(tree_right)).result();
}

template <Containable K, typename V, attr_t Fanout, typename Comp,
          bool AllowDup, template <typename T> class AllocTemplate, typename T>
std::pair<BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate>,
          BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate>>
split(BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate> &&tree,
      T                                                      &&raw_key) {
  return split_helper(std::move(tree), std::move(raw_key)).result();
}

template <Containable K, typename V, attr_t Fanout, typename Comp,
          bool AllowDup, template <typename T> class AllocTemplate, typename T>
std::pair<BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate>,
          BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate>>
split(BTreeBase<K, V, Fanout, Comp, AllowDup, AllocTemplate> &&tree,
      T                                                      &&raw_key1,
      T                                                      &&raw_key2) {
  return split_helper(std::move(tree), std::move(raw_key1), std::move(raw_key2)).result();
}
} // namespace frozenca

#endif //__FC_BTREE_H__
