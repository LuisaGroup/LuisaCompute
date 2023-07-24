#pragma once
#include <luisa/vstl/hash_map.h>
namespace vstd {
template<typename Arena, typename K, typename V = void, typename Hash = HashValue, typename Compare = compare<K>>
    requires(std::is_trivially_destructible_v<K> && (std::is_void_v<V> || std::is_trivially_destructible_v<V>))
class ArenaHashMap {

public:
    using KeyType = K;
    using ValueType = V;
    using HashType = Hash;
    using Map = SmallTreeMap<K, V, Compare>;
    using LinkNode = typename Map::Node;
    using NodePair = typename Map::ConstElement;
    using MoveNodePair = typename Map::Element;

private:
    struct PseudoPool {
        Arena arena;
        PseudoPool(Arena &&arena) : arena(std::forward<Arena>(arena)) {}
        PseudoPool(PseudoPool const &) = delete;
        PseudoPool(PseudoPool &&) = default;
        void *allocate(size_t size_bytes) {
            return arena.allocate(size_bytes);
        }
        template<typename... Args>
            requires(std::is_constructible_v<LinkNode, Args && ...>)
        LinkNode *create(Args &&...args) {
            auto ptr = allocate(sizeof(LinkNode));
            return new (ptr) LinkNode(std::forward<Args>(args)...);
        }
        void destroy(LinkNode *ptr) {}
    };

public:
    struct Iterator {
        friend class ArenaHashMap;

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
        friend class ArenaHashMap;

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
        friend class ArenaHashMap;

        template<typename T>
        friend struct hashmap_detail::HashMapIndex;
        template<typename T>
        friend struct hashmap_detail::HashSetIndex;

    private:
        const ArenaHashMap *map;
        LinkNode *node;
        IndexBase(const ArenaHashMap *map, LinkNode *node) noexcept : map(map), node(node) {}

    public:
        IndexBase() noexcept : map(nullptr), node(nullptr) {}
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

    using Index = typename hashmap_detail::HashSetIndexType<V, ArenaHashMap>::type;

    Index get_index(Iterator const &ite) {
        return Index(this, *ite.ii);
    }
    Index get_index(MoveIterator const &ite) {
        return Index(this, *ite.ii);
    }

private:
    LinkNode **nodeArray;
    PseudoPool pool;
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
        LinkNode **newNode = reinterpret_cast<LinkNode **>(pool.allocate(sizeof(LinkNode *) * newCapacity * 2));
        memcpy(newNode, nodeArray, sizeof(LinkNode *) * mSize);
        auto nodeVec = newNode + newCapacity;
        memset(nodeVec, 0, sizeof(LinkNode *) * newCapacity);
        for (auto node : ptr_range(nodeArray, nodeArray + mSize)) {
            size_t hashValue = node->hashValue;
            hashValue = GetHash(hashValue, newCapacity);
            Map *targetTree = reinterpret_cast<Map *>(&nodeVec[hashValue]);
            targetTree->weak_insert(pool, node);
        }
        nodeArray = newNode;
        mCapacity = newCapacity;
    }
    static Index EmptyIndex() noexcept {
        return Index(nullptr, nullptr);
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
    ArenaHashMap(size_t capacity, Arena &&arena) noexcept : pool(std::move(arena)) {
        if (capacity < 2) capacity = 2;
        capacity = GetPow2Size(capacity);
        nodeArray = reinterpret_cast<LinkNode **>(this->pool.allocate(sizeof(LinkNode *) * capacity * 2));
        memset(nodeArray + capacity, 0, capacity * sizeof(LinkNode *));
        mCapacity = capacity;
        mSize = 0;
    }
    ArenaHashMap(ArenaHashMap &&map)
        : pool(std::move(map.pool)),
          mSize(map.mSize),
          mCapacity(map.mCapacity),
          nodeArray(map.nodeArray) {
        map.nodeArray = nullptr;
    }
    ArenaHashMap(ArenaHashMap const &map) = delete;

    ArenaHashMap &operator=(ArenaHashMap &&map) {
        this->~ArenaHashMap();
        new (this) ArenaHashMap(std::move(map));
        return *this;
    }
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

    void clear() noexcept {
        if (mSize == 0) return;
        auto nodeVec = nodeArray + mCapacity;
        memset(nodeVec, 0, mCapacity * sizeof(LinkNode *));
        mSize = 0;
    }
    [[nodiscard]] size_t size() const noexcept { return mSize; }
    [[nodiscard]] bool empty() const noexcept { return mSize == 0; }
    [[nodiscard]] size_t capacity() const noexcept { return mCapacity; }
};
}// namespace vstd