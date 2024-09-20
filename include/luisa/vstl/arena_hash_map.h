#pragma once
#include <luisa/core/stl/type_traits.h>
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
    static_assert(alignof(LinkNode) >= alignof(Map));
private:
    struct PseudoPool {
        Arena arena;
        LinkNode *elements;
        size_t mSize;
        PseudoPool(Arena &&arena) : arena(std::forward<Arena>(arena)) {}
        PseudoPool(PseudoPool const &) = delete;
        PseudoPool(PseudoPool &&) = default;
        void *allocate(size_t size_bytes) {
            return arena.allocate(size_bytes);
        }
        template<typename... Args>
            requires(luisa::is_constructible_v<LinkNode, Args && ...>)
        LinkNode *create(Args &&...args) {
            auto ptr = elements + mSize;
            mSize++;
            return new (ptr) LinkNode(std::forward<Args>(args)...);
        }
        void destroy(LinkNode *ptr) {}
    };

public:
    struct Iterator {
        friend class ArenaHashMap;

    private:
        LinkNode *ii;

    public:
        Iterator(LinkNode *ii) : ii(ii) {}
        bool operator==(const Iterator &ite) const {
            return ii == ite.ii;
        }
        void operator++() {
            ++ii;
        }
        void operator++(int32_t) {
            ii++;
        }
        NodePair *operator->() const {
            return reinterpret_cast<NodePair *>(&ii->data);
        }
        NodePair &operator*() const {
            return reinterpret_cast<NodePair &>(ii->data);
        }
    };
    struct MoveIterator {
        friend class ArenaHashMap;

    private:
        LinkNode *ii;

    public:
        MoveIterator(LinkNode *ii) : ii(ii) {}
        bool operator==(const MoveIterator &ite) const {
            return ii == ite.ii;
        }
        void operator++() {
            ++ii;
        }
        void operator++(int32_t) {
            ii++;
        }
        MoveNodePair *operator->() const {
            return &ii->data;
        }
        MoveNodePair &&operator*() const {
            return std::move(ii->data);
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
        IndexBase(const ArenaHashMap *map, LinkNode *node) : map(map), node(node) {}

    public:
        IndexBase() : map(nullptr), node(nullptr) {}
        bool operator==(const IndexBase &a) const {
            return node == a.node;
        }
        operator bool() const {
            return node;
        }
        bool operator!() const {
            return !operator bool();
        }
        bool operator!=(const IndexBase &a) const {
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
    PseudoPool pool;
    Map *GetNodeVec() const {
        return reinterpret_cast<Map *>(pool.elements + mCapacity);
    }
    size_t mCapacity;

    inline static const Hash hsFunc;
    void GetNewLinkNode(size_t hashValue, LinkNode *newNode) {
        newNode->hashValue = hashValue;
        newNode->arrayIndex = pool.mSize - 1;
    }
    static size_t GetPow2Size(size_t capacity) {
        size_t ssize = 1;
        while (ssize < capacity)
            ssize <<= 1;
        return ssize;
    }
    static size_t GetHash(size_t hash, size_t size) {
        return hash & (size - 1);
    }
    void Resize(size_t newCapacity) {
        if (mCapacity >= newCapacity) return;
        LinkNode *newElements = reinterpret_cast<LinkNode *>(pool.allocate((sizeof(Map) + sizeof(LinkNode)) * newCapacity));
        std::memcpy(newElements, pool.elements, sizeof(LinkNode) * pool.mSize);
        pool.elements = newElements;
        mCapacity = newCapacity;
        auto nodeVec = GetNodeVec();
        std::memset(nodeVec, 0, sizeof(Map) * newCapacity);
        for (auto &node : ptr_range(newElements, pool.mSize)) {
            size_t hashValue = node.hashValue;
            hashValue = GetHash(hashValue, newCapacity);
            Map *targetTree = nodeVec + hashValue;
            targetTree->weak_insert(pool, &node);
        }
    }
    static Index EmptyIndex() {
        return Index(nullptr, nullptr);
    }
    void TryResize() {
        size_t targetCapacity = (size_t)((pool.mSize + 1));
        if (targetCapacity < 16) targetCapacity = 16;
        if (targetCapacity > mCapacity) {
            Resize(GetPow2Size(targetCapacity));
        }
    }

public:
    decltype(auto) begin() const & {
        return Iterator(pool.elements);
    }
    decltype(auto) begin() && {
        return MoveIterator(pool.elements);
    }
    decltype(auto) end() const & {
        return Iterator(pool.elements + pool.mSize);
    }
    decltype(auto) end() && {
        return MoveIterator(pool.elements + pool.mSize);
    }
    //////////////////Construct & Destruct
    void clear() {
        auto nodeVec = GetNodeVec();
        std::memset(nodeVec, 0, mCapacity * sizeof(Map));
        pool.mSize = 0;
    }
    ArenaHashMap(size_t capacity, Arena &&arena) : pool(std::move(arena)) {
        if (capacity < 2) capacity = 2;
        capacity = GetPow2Size(capacity);
        pool.elements = reinterpret_cast<LinkNode *>(pool.allocate((sizeof(Map) + sizeof(LinkNode)) * capacity));
        mCapacity = capacity;
        clear();
    }
    ArenaHashMap(ArenaHashMap &&map)
        : pool(std::move(map.pool)),
          mCapacity(map.mCapacity) {
    }
    ArenaHashMap(ArenaHashMap const &map) = delete;

    ArenaHashMap &operator=(ArenaHashMap &&map) {
        this->~ArenaHashMap();
        new (this) ArenaHashMap(std::move(map));
        return *this;
    }
    ///////////////////////
    template<typename Key, typename... ARGS>
        requires(luisa::is_constructible_v<K, Key &&> && detail::MapConstructible<V, ARGS && ...>::value)
    Index force_emplace(Key &&key, ARGS &&...args) {
        TryResize();

        size_t hashOriginValue = hsFunc(std::forward<Key>(key));
        size_t hashValue = GetHash(hashOriginValue, mCapacity);
        auto nodeVec = GetNodeVec();
        Map *map = nodeVec + hashValue;
        auto insertResult = map->insert_or_assign(pool, std::forward<Key>(key), std::forward<ARGS>(args)...);
        //Add create
        if (insertResult.second) {
            GetNewLinkNode(hashOriginValue, insertResult.first);
        }
        return Index(this, insertResult.first);
    }

    template<typename Key, typename... ARGS>
        requires(luisa::is_constructible_v<K, Key &&> && detail::MapConstructible<V, ARGS && ...>::value)
    std::pair<Index, bool> try_emplace(Key &&key, ARGS &&...args) {
        TryResize();

        size_t hashOriginValue = hsFunc(std::forward<Key>(key));
        size_t hashValue = GetHash(hashOriginValue, mCapacity);
        Map *map = GetNodeVec() + hashValue;
        auto insertResult = map->try_insert(pool, std::forward<Key>(key), std::forward<ARGS>(args)...);
        //Add create
        if (insertResult.second) {
            GetNewLinkNode(hashOriginValue, insertResult.first);
        }
        return {Index(this, insertResult.first), insertResult.second};
    }

    template<typename Key, typename... ARGS>
        requires(luisa::is_constructible_v<K, Key &&> && detail::MapConstructible<V, ARGS && ...>::value)
    Index emplace(Key &&key, ARGS &&...args) {
        return try_emplace(std::forward<Key>(key), std::forward<ARGS>(args)...).first;
    }

    void reserve(size_t capacity) {
        size_t newCapacity = GetPow2Size(capacity);
        Resize(newCapacity);
    }
    template<typename Key>
    Index find(Key &&key) const {
        size_t hashOriginValue = hsFunc(std::forward<Key>(key));
        size_t hashValue = GetHash(hashOriginValue, mCapacity);
        Map *map = GetNodeVec() + hashValue;
        auto node = map->find(std::forward<Key>(key));
        if (node)
            return {this, node};
        return EmptyIndex();
    }
    [[nodiscard]] size_t size() const { return pool.mSize; }
    [[nodiscard]] bool empty() const { return pool.mSize == 0; }
    [[nodiscard]] size_t capacity() const { return mCapacity; }
};
}// namespace vstd