#pragma once
#include <vstl/config.h>
#include <core/stl/memory.h>
#include <type_traits>
#include <stdint.h>
#include <vstl/pool.h>
#include <vstl/hash.h>
#include <vstl/meta_lib.h>
#include <vstl/v_allocator.h>
#include <vstl/tree_map_base.h>

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
    using Element = typename decltype(TreeElementType<K, V>())::Type;
    using ConstElement = typename decltype(ConstTreeElementType<K, V>())::Type;
    static decltype(auto) GetFirst(auto &&ele) {
        if constexpr (std::is_same_v<V, void>) {
            return ele;
        } else {
            return ele.first;
        }
    }
    struct Node {
        size_t arrayIndex;
        size_t hashValue;
        Node *parent;// pointer to the parent
        Node *left;  // pointer to left child
        Node *right; // pointer to right child
        bool color;  // 1 -> Red, 0 -> Black
        Element data;// holds the key
        template<typename A, typename... B>
        Node(A &&a, B &&...b)
            : data(std::forward<A>(a), std::forward<B>(b)...) {
        }
    };

    // data structure that represents a node in the tree

private:
    using TreeMapUtility = detail::TreeMapUtility<Node>;

    Node *root;
    template<typename Key>
    Node *searchTreeHelper(Node *node, Key const &key) const {
        if (node == nullptr)
            return nullptr;
        auto compResult = comp(GetFirst(node->data), key);
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
    std::pair<Node *, int32> searchClosestTreeHelper(Node *node, Key const &key, Node *lastNode, int32_t lastFlag) const {
        if (node == nullptr) {
            return {lastNode, lastFlag};
        }
        auto compResult = comp(GetFirst(node->data), key);
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
    void deleteOneNode(Node *z, POOL &pool) {
        detail::TreeMapUtility<Node>::deleteOneNode(z, root);
        pool.destroy(z);
    }

    template<typename Key, typename POOL>
    bool deleteNodeHelper(Node *node, Key const &key, POOL &pool, size_t &arrayIndex) {
        // find the node containing key
        Node *z = nullptr;
        while (node != nullptr) {
            auto compResult = comp(GetFirst(node->data), key);
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

    void fixInsert(Node *k) {
        detail::TreeMapUtility<Node>::fixInsert(k, root);
    }
    Node *getRoot() {
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
    Node *find(Key const &k) const {
        return searchTreeHelper(this->root, k);
    }
    template<typename POOL, typename Key, typename... Value>
    std::pair<Node *, bool> try_insert(POOL &pool, Key &&key, Value &&...value) {
        // Ordinary Binary Search Insertion
        Node *y = nullptr;
        Node *x = this->root;
        int compResult;
        while (x != nullptr) {
            y = x;
            auto mCompResult = comp(GetFirst(x->data), key);
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
        Node *node = pool.create(std::forward<Key>(key), std::forward<Value>(value)...);
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
    std::pair<Node *, bool> insert_or_assign(POOL &pool, Key &&key, Value &&...value) {
        // Ordinary Binary Search Insertion
        Node *y = nullptr;
        Node *x = this->root;
        int compResult;
        while (x != nullptr) {
            y = x;
            auto mCompResult = comp(GetFirst(x->data), key);
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
        Node *node = pool.create(std::forward<Key>(key), std::forward<Value>(value)...);
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
    Node *weak_insert(POOL &pool, Node *node) {
        // Ordinary Binary Search Insertion
        Node *y = nullptr;
        Node *x = this->root;
        int compResult;
        while (x != nullptr) {
            y = x;
            auto mCompResult = comp(GetFirst(x->data), GetFirst(node->data));
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
    void remove(POOL &pool, Node *ite) {
        deleteOneNode(ite, pool);
    }
};
template<typename K, typename V = void, typename Hash = HashValue, typename Compare = compare<K>, VEngine_AllocType allocType = VEngine_AllocType::VEngine>
class HashMap {
public:
    using KeyType = K;
    using ValueType = V;
    using HashType = Hash;
    using Map = SmallTreeMap<K, V, Compare>;
    using LinkNode = typename Map::Node;
    using NodePair = typename Map::ConstElement;
    using MoveNodePair = typename Map::Element;
    using Allocator = VAllocHandle<allocType>;
    struct Iterator {
        friend class HashMap;

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
        friend class HashMap;

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

public:
    struct IndexBase {
        friend class HashMap;

    private:
        const HashMap *map;
        LinkNode *node;
        IndexBase(const HashMap *map, LinkNode *node) noexcept : map(map), node(node) {}

    public:
        IndexBase() : map(nullptr), node(nullptr) {}
        bool operator==(const IndexBase &a) const noexcept {
            return node == a.node;
        }
        operator bool() const noexcept {
            return node;
        }
        bool operator!() const noexcept {
            return !operator bool();
        }
        bool operator!=(const IndexBase &a) const noexcept {
            return !operator==(a);
        }
    };

    static consteval decltype(auto) IndexType() {
        if constexpr (std::is_same_v<V, void>) {
            struct IndexKey : public IndexBase {
                friend class HashMap;

            private:
                IndexKey(const HashMap *map, LinkNode *node) noexcept : IndexBase(map, node) {}

            public:
                IndexKey() {}
                K const &Get() const noexcept {
                    return Map::GetFirst(this->node->data);
                }
                K const *operator->() const noexcept {
                    return &Map::GetFirst(this->node->data);
                }
                K &operator*() const noexcept {
                    return Map::GetFirst(this->node->data);
                }
            };
            return TypeOf<IndexKey>{};
        } else {
            struct IndexKeyValue : public IndexBase {
                friend class HashMap;

            private:
                IndexKeyValue(const HashMap *map, LinkNode *node) noexcept : IndexBase(map, node) {}

            public:
                IndexKeyValue() {}
                K const &Key() const noexcept {
                    return Map::GetFirst(this->node->data);
                }
                inline V &Value() const noexcept {
                    return this->node->data.second;
                }
            };
            return TypeOf<IndexKeyValue>{};
        }
    }

    using Index = typename decltype(IndexType())::Type;
    Index get_index(Iterator const &ite) {
        return Index(this, *ite.ii);
    }
    Index get_index(MoveIterator const &ite) {
        return Index(this, *ite.ii);
    }

private:
    LinkNode **nodeArray;
    Pool<LinkNode, true> pool;
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
        LinkNode **newNode = reinterpret_cast<LinkNode **>(Allocator().Malloc(sizeof(LinkNode *) * newCapacity * 2));
        memcpy(newNode, nodeArray, sizeof(LinkNode *) * mSize);
        auto nodeVec = newNode + newCapacity;
        memset(nodeVec, 0, sizeof(LinkNode *) * newCapacity);
        for (auto node : ptr_range(nodeArray, nodeArray + mSize)) {
            size_t hashValue = node->hashValue;
            hashValue = GetHash(hashValue, newCapacity);
            Map *targetTree = reinterpret_cast<Map *>(&nodeVec[hashValue]);
            targetTree->weak_insert(pool, node);
        }
        Allocator().Free(nodeArray);
        nodeArray = newNode;
        mCapacity = newCapacity;
    }
    static Index EmptyIndex() noexcept {
        return Index(nullptr, nullptr);
    }
    void remove(LinkNode *node) {
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
        nodeArray = reinterpret_cast<LinkNode **>(Allocator().Malloc(sizeof(LinkNode *) * capacity * 2));
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

    HashMap &operator=(HashMap &&map) {
        this->~HashMap();
        new (this) HashMap(std::move(map));
        return *this;
    }
    ~HashMap() noexcept {
        if (!nodeArray) return;
        for (auto i : ptr_range(nodeArray, nodeArray + mSize)) {
            i->~LinkNode();
        }
        Allocator().Free(nodeArray);
    }
    HashMap() noexcept : HashMap(16) {}
    ///////////////////////
    template<typename Key, typename... ARGS>
        requires(std::is_constructible_v<K, Key &&> && detail::MapConstructible<V, ARGS && ...>::value)
    Index force_emplace(Key &&key, ARGS &&...args) {
        TryResize();

        size_t hashOriginValue = hsFunc(std::forward<Key>(key));
        size_t hashValue = GetHash(hashOriginValue, mCapacity);
        auto nodeVec = nodeArray + mCapacity;

        Map *map = reinterpret_cast<Map *>(&nodeVec[hashValue]);
        auto insertResult = map->insert_or_assign(pool, std::forward<Key>(key), std::forward<ARGS>(args)...);
        //Add create
        if (insertResult.second) {
            GetNewLinkNode(hashOriginValue, insertResult.first);
        }
        return Index(this, insertResult.first);
    }

    template<typename Key, typename... ARGS>
        requires(std::is_constructible_v<K, Key &&> && detail::MapConstructible<V, ARGS && ...>::value)
    Index emplace(Key &&key, ARGS &&...args) {
        TryResize();

        size_t hashOriginValue = hsFunc(std::forward<Key>(key));
        size_t hashValue = GetHash(hashOriginValue, mCapacity);
        auto nodeVec = nodeArray + mCapacity;

        Map *map = reinterpret_cast<Map *>(&nodeVec[hashValue]);
        auto insertResult = map->try_insert(pool, std::forward<Key>(key), std::forward<ARGS>(args)...);
        //Add create
        if (insertResult.second) {
            GetNewLinkNode(hashOriginValue, insertResult.first);
        }
        return Index(this, insertResult.first);
    }

    template<typename Key, typename... ARGS>
        requires(std::is_constructible_v<K, Key &&> && detail::MapConstructible<V, ARGS && ...>::value)
    std::pair<Index, bool> try_emplace(Key &&key, ARGS &&...args) {
        TryResize();

        size_t hashOriginValue = hsFunc(std::forward<Key>(key));
        size_t hashValue = GetHash(hashOriginValue, mCapacity);
        auto nodeVec = nodeArray + mCapacity;

        Map *map = reinterpret_cast<Map *>(&nodeVec[hashValue]);
        auto insertResult = map->try_insert(pool, std::forward<Key>(key), std::forward<ARGS>(args)...);
        //Add create
        if (insertResult.second) {
            GetNewLinkNode(hashOriginValue, insertResult.first);
        }
        return {Index(this, insertResult.first), insertResult.second};
    }

    void reserve(size_t capacity) noexcept {
        size_t newCapacity = GetPow2Size(capacity);
        Resize(newCapacity);
    }
    template<typename Key>
    Index find(Key &&key) const noexcept {
        size_t hashOriginValue = hsFunc(std::forward<Key>(key));
        size_t hashValue = GetHash(hashOriginValue, mCapacity);
        Map *map = reinterpret_cast<Map *>(nodeArray + mCapacity + hashValue);
        auto node = map->find(std::forward<Key>(key));
        if (node)
            return {this, node};
        return EmptyIndex();
    }

    template<typename Key>
    void remove(Key &&key) noexcept {
        size_t hashOriginValue = hsFunc(key);
        size_t hashValue = GetHash(hashOriginValue, mCapacity);
        Map *map = reinterpret_cast<Map *>(nodeArray + mCapacity + hashValue);
        size_t arrayIndex;
        if (map->remove(pool, std::forward<Key>(key), arrayIndex)) {
            DeleteLinkNode(arrayIndex);
        }
    }

    void remove(const Index &ite) noexcept {
        remove(ite.node);
    }
    void remove(NodePair const &ite) noexcept {
        remove(const_cast<LinkNode *>(static_cast<LinkNode const *>(&ite)));
    }

    void clear() noexcept {
        if (mSize == 0) return;
        auto nodeVec = nodeArray + mCapacity;
        memset(nodeVec, 0, mCapacity * sizeof(LinkNode *));
        for (auto ite : ptr_range(nodeArray, nodeArray + mSize)) {
            pool.destroy(ite);
        }
        mSize = 0;
    }
    [[nodiscard]] size_t size() const noexcept { return mSize; }
    [[nodiscard]] size_t capacity() const noexcept { return mCapacity; }
};
}// namespace vstd
