#pragma once
#include <vstl/config.h>
#include <type_traits>
#include <stdint.h>
#include <memory>
#include <vstl/Pool.h>
#include <vstl/vector.h>
#include <vstl/Hash.h>
#include <vstl/MetaLib.h>
#include <vstl/VAllocator.h>
#include <vstl/TreeMapBase.h>

namespace vstd {
template<typename A, typename B, bool value>
struct IfType;
template<typename A, typename B>
struct IfType<A, B, true> {
    using Type = A;
};
template<typename A, typename B>
struct IfType<A, B, false> {
    using Type = B;
};
template<typename K, typename V, typename Compare = compare<K>>
class SmallTreeMap {
    inline static const Compare comp;

public:
    using Element = TreeElement<K, V>;
    using ConstElement = ConstTreeElement<K, V>;
    // data structure that represents a node in the tree
    struct Node {
        bool color;  // 1 -> Red, 0 -> Black
        Node *parent;// pointer to the parent
        Node *left;  // pointer to left child
        Node *right; // pointer to right child
        size_t arrayIndex;
        size_t hashValue;
        Element data;// holds the key
        template<typename A, typename... B>
        Node(A &&a, B &&...b)
            : data(std::forward<A>(a), std::forward<B>(b)...) {
        }
    };
    typedef Node *NodePtr;

private:
    NodePtr root;
    template<typename Key>
    NodePtr searchTreeHelper(NodePtr node, Key const &key) const {
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
    std::pair<NodePtr, int32> searchClosestTreeHelper(NodePtr node, Key const &key, NodePtr lastNode, int32_t lastFlag) const {
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
    template<typename POOL>
    void deleteOneNode(NodePtr z, POOL &pool) {
        detail::TreeMapUtility::deleteOneNode(z, *reinterpret_cast<void **>(&root));
        pool.Delete(z);
    }

    template<typename Key, typename POOL>
    bool deleteNodeHelper(NodePtr node, Key const &key, POOL &pool, size_t &arrayIndex) {
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
        arrayIndex = z->arrayIndex;
        deleteOneNode(z, pool);
        return true;
    }

    void fixInsert(NodePtr k) {
        detail::TreeMapUtility::fixInsert(k, *reinterpret_cast<void **>(&root));
    }
    NodePtr getRoot() {
        return this->root;
    }

public:
    SmallTreeMap() {
        root = nullptr;
    }

    SmallTreeMap(SmallTreeMap const &) = delete;
    SmallTreeMap(SmallTreeMap &&o)
        : root(o.root) {
        o.root = nullptr;
    };
    template<typename Key>
    NodePtr find(Key const &k) const {
        return searchTreeHelper(this->root, k);
    }
    template<typename POOL, typename Key, typename... Value>
    std::pair<NodePtr, bool> try_insert(POOL &pool, Key &&key, Value &&...value) {
        // Ordinary Binary Search Insertion
        NodePtr y = nullptr;
        NodePtr x = this->root;
        int compResult;
        while (x != nullptr) {
            y = x;
            auto mCompResult = comp(x->data.first, key);
            static_assert(std::is_same_v<decltype(mCompResult), int32_t>, "compare result must be int32");
            compResult = mCompResult;
            if (compResult > 0) {
                x = x->left;
            } else if (compResult == 0) {
                return {x, false};

            } else {
                x = x->right;
            }
        }
        NodePtr node = pool.New(std::forward<Key>(key), std::forward<Value>(value)...);
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
            return {node, true};
        }

        // if the grandparent is null, simply return
        if (node->parent->parent == nullptr) {
            return {node, true};
        }
        // Fix the tree
        fixInsert(node);
        return {node, true};
    }
    template<typename POOL, typename Key, typename... Value>
    std::pair<NodePtr, bool> insert_or_assign(POOL &pool, Key &&key, Value &&...value) {
        // Ordinary Binary Search Insertion
        NodePtr y = nullptr;
        NodePtr x = this->root;
        int compResult;
        while (x != nullptr) {
            y = x;
            auto mCompResult = comp(x->data.first, key);
            static_assert(std::is_same_v<decltype(mCompResult), int32_t>, "compare result must be int32");
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
                return {x, false};

            } else {
                x = x->right;
            }
        }
        NodePtr node = pool.New(std::forward<Key>(key), std::forward<Value>(value)...);
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
            return {node, true};
        }

        // if the grandparent is null, simply return
        if (node->parent->parent == nullptr) {
            return {node, true};
        }
        // Fix the tree
        fixInsert(node);
        return {node, true};
    }
    template<typename POOL>
    NodePtr weak_insert(POOL &pool, NodePtr node) {
        // Ordinary Binary Search Insertion
        NodePtr y = nullptr;
        NodePtr x = this->root;
        int compResult;
        while (x != nullptr) {
            y = x;
            auto mCompResult = comp(x->data.first, node->data.first);
            static_assert(std::is_same_v<decltype(mCompResult), int32_t>, "compare result must be int32");
            compResult = mCompResult;
            if (compResult > 0) {
                x = x->left;
            } else if (compResult == 0) {
                return x;

            } else {
                x = x->right;
            }
        }
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
            return node;
        }

        // if the grandparent is null, simply return
        if (node->parent->parent == nullptr) {
            return node;
        }
        // Fix the tree
        fixInsert(node);
        return node;
    }
    template<typename POOL, typename Key>
    bool remove(POOL &pool, Key const &data, size_t &arrayIndex) {
        return deleteNodeHelper(this->root, data, pool, arrayIndex);
    }
    template<typename POOL>
    void remove(POOL &pool, NodePtr ite) {
        deleteOneNode(ite, pool);
    }
};
struct HashEqual {
    template<typename A, typename B>
    bool operator()(A const &a, B const &b) const {
        return a == b;
    }
};
struct HashValue {
    template<typename T>
    size_t operator()(T const &t) const {
        hash<std::remove_cvref_t<T>> h;
        return h(t);
    }
};
namespace detail {
template<typename T, typename... Args>
struct MapConstructible {
    static constexpr bool value = std::is_constructible_v<T, Args...>;
};

template<typename... Args>
struct MapConstructible<void, Args...> {
    static constexpr bool value = (sizeof...(Args) == 0);
};
};// namespace detail
template<typename K, typename V, typename Hash = HashValue, typename Compare = compare<K>, VEngine_AllocType allocType = VEngine_AllocType::VEngine>
class HashMap : public IOperatorNewBase {
public:
    using KeyType = K;
    using ValueType = V;
    using HashType = Hash;
    using Map = SmallTreeMap<K, V, Compare>;
    using LinkNode = typename Map::Node;
    using NodePair = typename Map::ConstElement;
    using MoveNodePair = typename Map::Element;
    struct Iterator {
    private:
        LinkNode **ii;

    public:
        Iterator(LinkNode **ii) : ii(ii) {}
        bool operator==(const Iterator &ite) const noexcept {
            return ii == ite.ii;
        }
        void operator++() noexcept {
            ++ii;
        }
        void operator++(int32_t) noexcept {
            ii++;
        }
        NodePair *operator->() const noexcept {
            return reinterpret_cast<NodePair *>(&(*ii)->data);
        }
        NodePair &operator*() const noexcept {
            return reinterpret_cast<NodePair &>((*ii)->data);
        }
    };
    struct MoveIterator {
    private:
        LinkNode **ii;

    public:
        MoveIterator(LinkNode **ii) : ii(ii) {}
        bool operator==(const MoveIterator &ite) const noexcept {
            return ii == ite.ii;
        }
        void operator++() noexcept {
            ++ii;
        }
        void operator++(int32_t) noexcept {
            ii++;
        }
        MoveNodePair *operator->() const noexcept {
            return &(*ii)->data;
        }
        MoveNodePair &&operator*() const noexcept {
            return std::move((*ii)->data);
        }
    };
    struct Index {
        friend class HashMap;

    private:
        const HashMap *map;
        LinkNode *node;
        Index(const HashMap *map, LinkNode *node) noexcept : map(map), node(node) {}
        using ValueType = typename IfType<void, std::add_lvalue_reference_t<V>, std::is_same_v<V, void>>::Type;

    public:
        Index() : map(nullptr), node(nullptr) {}
        bool operator==(const Index &a) const noexcept {
            return node == a.node;
        }
        operator bool() const noexcept {
            return node;
        }
        bool operator!() const noexcept {
            return !operator bool();
        }
        bool operator!=(const Index &a) const noexcept {
            return !operator==(a);
        }
        inline K const &Key() const noexcept;
        inline ValueType Value() const noexcept;
    };

private:
    LinkNode **nodeArray;
    Pool<LinkNode, allocType, true> pool;
    size_t mSize;
    size_t mCapacity;

    inline static const Hash hsFunc;
    LinkNode *GetNewLinkNode(size_t hashValue, LinkNode *newNode) {
        newNode->hashValue = hashValue;
        newNode->arrayIndex = mSize;
        nodeArray[mSize] = newNode;
        mSize++;
        return newNode;
    }
    void DeleteLinkNode(size_t arrayIndex) {
        if (arrayIndex != (mSize - 1)) {
            auto ite = nodeArray + (mSize - 1);
            (*ite)->arrayIndex = arrayIndex;
            nodeArray[arrayIndex] = *ite;
        }
        mSize--;
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
        if (mCapacity >= newCapacity) return;
        LinkNode **newNode = reinterpret_cast<LinkNode **>(pool.GetAllocator().Malloc(sizeof(LinkNode *) * newCapacity * 2));
        memcpy(newNode, nodeArray, sizeof(LinkNode *) * mSize);
        auto nodeVec = newNode + newCapacity;
        memset(nodeVec, 0, sizeof(LinkNode *) * newCapacity);
        for (auto node : ptr_range(nodeArray, nodeArray + mSize)) {
            size_t hashValue = node->hashValue;
            hashValue = GetHash(hashValue, newCapacity);
            Map *targetTree = reinterpret_cast<Map *>(&nodeVec[hashValue]);
            targetTree->weak_insert(pool, node);
        }
        pool.GetAllocator().Free(nodeArray);
        nodeArray = newNode;
        mCapacity = newCapacity;
    }
    static Index EmptyIndex() noexcept {
        return Index(nullptr, nullptr);
    }
    void Remove(LinkNode *node) {
        size_t hashValue = GetHash(node->hashValue, mCapacity);
        Map *targetTree = reinterpret_cast<Map *>(nodeArray + mCapacity + hashValue);
        auto arrayIndex = node->arrayIndex;
        targetTree->remove(pool, node);
        DeleteLinkNode(arrayIndex);
    }
    void TryResize() {
        size_t targetCapacity = (size_t)((mSize + 1));
        if (targetCapacity < 16) targetCapacity = 16;
        if (targetCapacity > mCapacity) {
            Resize(GetPow2Size(targetCapacity));
        }
    }

public:
    size_t Size() const {
        return mSize;
    }
    decltype(auto) begin() const & {
        return Iterator(nodeArray);
    }
    decltype(auto) begin() && {
        return MoveIterator(nodeArray);
    }
    decltype(auto) end() const & {
        return Iterator(nodeArray + mSize);
    }
    decltype(auto) end() && {
        return MoveIterator(nodeArray + mSize);
    }
    //////////////////Construct & Destruct
    HashMap(size_t capacity) noexcept : pool(capacity) {
        if (capacity < 2) capacity = 2;
        capacity = GetPow2Size(capacity);
        nodeArray = reinterpret_cast<LinkNode **>(pool.GetAllocator().Malloc(sizeof(LinkNode *) * capacity * 2));
        memset(nodeArray + capacity, 0, capacity * sizeof(LinkNode *));
        mCapacity = capacity;
        mSize = 0;
    }
    HashMap(HashMap &&map)
        : pool(std::move(map.pool)),
          mSize(map.mSize),
          mCapacity(map.mCapacity),
          nodeArray(map.nodeArray) {
        map.nodeArray = nullptr;
    }
    HashMap(HashMap const &map) = delete;

    template<typename Arg>
    void operator=(Arg &&map) {
        this->~HashMap();
        new (this) HashMap(std::forward<Arg>(map));
    }
    ~HashMap() noexcept {
        if (!nodeArray) return;
        for (auto i : ptr_range(nodeArray, nodeArray + mSize)) {
            i->~LinkNode();
        }
        pool.GetAllocator().Free(nodeArray);
    }
    HashMap() noexcept : HashMap(16) {}
    ///////////////////////
    template<typename Key, typename... ARGS>
        requires(std::is_constructible_v<K, Key &&> &&detail::MapConstructible<V, ARGS &&...>::value)
    Index ForceEmplace(Key &&key, ARGS &&...args) {
        TryResize();

        size_t hashOriginValue = hsFunc(std::forward<Key>(key));
        size_t hashValue = GetHash(hashOriginValue, mCapacity);
        auto nodeVec = nodeArray + mCapacity;

        Map *map = reinterpret_cast<Map *>(&nodeVec[hashValue]);
        auto insertResult = map->insert_or_assign(pool, std::forward<Key>(key), std::forward<ARGS>(args)...);
        //Add New
        if (insertResult.second) {
            GetNewLinkNode(hashOriginValue, insertResult.first);
        }
        return Index(this, insertResult.first);
    }

    template<typename Key, typename... ARGS>
        requires(std::is_constructible_v<K, Key &&> &&detail::MapConstructible<V, ARGS &&...>::value)
    Index Emplace(Key &&key, ARGS &&...args) {
        TryResize();

        size_t hashOriginValue = hsFunc(std::forward<Key>(key));
        size_t hashValue = GetHash(hashOriginValue, mCapacity);
        auto nodeVec = nodeArray + mCapacity;

        Map *map = reinterpret_cast<Map *>(&nodeVec[hashValue]);
        auto insertResult = map->try_insert(pool, std::forward<Key>(key), std::forward<ARGS>(args)...);
        //Add New
        if (insertResult.second) {
            GetNewLinkNode(hashOriginValue, insertResult.first);
        }
        return Index(this, insertResult.first);
    }

    template<typename Key, typename... ARGS>
        requires(std::is_constructible_v<K, Key &&> &&detail::MapConstructible<V, ARGS &&...>::value)
    std::pair<Index, bool> TryEmplace(Key &&key, ARGS &&...args) {
        TryResize();

        size_t hashOriginValue = hsFunc(std::forward<Key>(key));
        size_t hashValue = GetHash(hashOriginValue, mCapacity);
        auto nodeVec = nodeArray + mCapacity;

        Map *map = reinterpret_cast<Map *>(&nodeVec[hashValue]);
        auto insertResult = map->try_insert(pool, std::forward<Key>(key), std::forward<ARGS>(args)...);
        //Add New
        if (insertResult.second) {
            GetNewLinkNode(hashOriginValue, insertResult.first);
        }
        return {Index(this, insertResult.first), insertResult.second};
    }

    void Reserve(size_t capacity) noexcept {
        size_t newCapacity = GetPow2Size(capacity);
        Resize(newCapacity);
    }
    void reserve(size_t capacity) noexcept {
        Reserve(capacity);
    }
    template<typename Key>
    Index Find(Key &&key) const noexcept {
        size_t hashOriginValue = hsFunc(std::forward<Key>(key));
        size_t hashValue = GetHash(hashOriginValue, mCapacity);
        Map *map = reinterpret_cast<Map *>(nodeArray + mCapacity + hashValue);
        auto node = map->find(std::forward<Key>(key));
        if (node)
            return {this, node};
        return EmptyIndex();
    }

    template<typename Key>
    void Remove(Key &&key) noexcept {
        size_t hashOriginValue = hsFunc(key);
        size_t hashValue = GetHash(hashOriginValue, mCapacity);
        Map *map = reinterpret_cast<Map *>(nodeArray + mCapacity + hashValue);
        size_t arrayIndex;
        if (map->remove(pool, std::forward<Key>(key), arrayIndex)) {
            DeleteLinkNode(arrayIndex);
        }
    }

    void Remove(const Index &ite) noexcept {
        Remove(ite.node);
    }
    void Remove(Index &&ite) noexcept {
        Remove(ite.node);
    }
    void Remove(Index &ite) noexcept {
        Remove(ite.node);
    }
    void Remove(NodePair const &ite) noexcept {
        Remove(const_cast<LinkNode *>(static_cast<LinkNode const *>(&ite)));
    }

    void Clear() noexcept {
        if (mSize == 0) return;
        auto nodeVec = nodeArray + mCapacity;
        memset(nodeVec, 0, mCapacity * sizeof(LinkNode *));
        for (auto ite : ptr_range(nodeArray, nodeArray + mSize)) {
            pool.Delete(ite);
        }
        mSize = 0;
    }
    [[nodiscard]] size_t size() const noexcept { return mSize; }
    [[nodiscard]] size_t GetCapacity() const noexcept { return mCapacity; }
};

template<typename K, typename V, typename Hash, typename Equal, VEngine_AllocType allocType>
inline K const &HashMap<K, V, Hash, Equal, allocType>::Index::Key() const noexcept {
    return node->data.first;
}
template<typename K, typename V, typename Hash, typename Equal, VEngine_AllocType allocType>
inline typename HashMap<K, V, Hash, Equal, allocType>::Index::ValueType HashMap<K, V, Hash, Equal, allocType>::Index::Value() const noexcept {
    if constexpr (!std::is_same_v<V, void>) {
        return node->data.second;
    }
}
}// namespace vstd
