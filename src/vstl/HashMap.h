#pragma once
#include <memory>
#include <stdint.h>
#include <type_traits>
#include <vstl/Hash.h>
#include <vstl/MetaLib.h>
#include <vstl/Pool.h>
#include <vstl/VAllocator.h>
#include <vstl/config.h>
namespace vstd {
struct HashEqual {
    template<typename A, typename B>
    bool operator()(A const &a, B const &b) const {
        return a == b;
    }
};
struct HashValue {
    template<typename T>
    size_t operator()(T const &t) const {
        vstd::hash<std::remove_cvref_t<T>> h;
        return h(t);
    }
};

template<typename K, typename V, typename Hash = vstd::HashValue, typename Equal = vstd::HashEqual, VEngine_AllocType allocType = VEngine_AllocType::VEngine>
class HashMap : public vstd::IOperatorNewBase {
public:
    using KeyType = K;
    using ValueType = V;
    using HashType = Hash;
    using EqualType = Equal;
    struct NodePair {
    public:
        K first;
        mutable V second;
        NodePair() {}
        template<typename A, typename... B>
        NodePair(A &&a, B &&...b) : first(std::forward<A>(a)), second(std::forward<B>(b)...) {}
    };

    struct LinkNode : public NodePair {
        LinkNode *last = nullptr;
        LinkNode *next = nullptr;
        size_t arrayIndex;
        size_t hashValue;
        //LinkNode() noexcept {}
        template<typename A, typename... B>
        LinkNode(size_t hashValue, size_t arrayIndex, A &&key, B &&...args) noexcept : NodePair(std::forward<A>(key), std::forward<B>(args)...), arrayIndex(arrayIndex), hashValue(hashValue) {}

        static void Add(LinkNode *&source, LinkNode *dest) noexcept {
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
        typename std::vector<LinkNode *>::const_iterator ii;

    public:
        Iterator(typename std::vector<LinkNode *>::const_iterator ii) : ii(ii) {}
        bool operator==(const Iterator &ite) const noexcept {
            return ii == ite.ii;
        }
        bool operator!=(const Iterator &ite) const noexcept {
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
        void operator+=(size_t i) noexcept {
            ii += i;
        }
        void operator-=(size_t i) noexcept {
            ii -= i;
        }
        Iterator operator+(size_t i) noexcept {
            return Iterator(ii + i);
        }
        Iterator operator-(size_t i) noexcept {
            return Iterator(ii - i);
        }

        NodePair const *operator->() const noexcept {
            return *ii;
        }
        NodePair const &operator*() const noexcept {
            return *operator->();
        }
    };

    struct Index {
        friend class HashMap;

    private:
        const HashMap *map;
        HashMap::LinkNode *node;
        Index(const HashMap *map, HashMap::LinkNode *node) noexcept : map(map), node(node) {}

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
        inline V &Value() const noexcept;
    };

private:
    std::vector<LinkNode *> allocatedNodes;
    struct HashArray {
    private:
        LinkNode **nodesPtr = nullptr;
        size_t mSize;
        VAllocHandle<allocType> allocHandle;

    public:
        HashArray(HashArray &&map)
            : nodesPtr(map.nodesPtr),
              mSize(map.mSize) {
            map.mSize = 0;
            map.nodesPtr = nullptr;
        }
        size_t size() const noexcept { return mSize; }
        HashArray() noexcept : mSize(0) {}
        void ClearAll() {
            memset(nodesPtr, 0, sizeof(LinkNode *) * mSize);
        }

        HashArray(size_t mSize) noexcept : mSize(mSize) {
            nodesPtr = (LinkNode **)allocHandle.Malloc(sizeof(LinkNode *) * mSize);
            memset(nodesPtr, 0, sizeof(LinkNode *) * mSize);
        }
        HashArray(HashArray &arr) noexcept : nodesPtr(arr.nodesPtr) {

            mSize = arr.mSize;
            arr.nodesPtr = nullptr;
        }
        void operator=(HashArray &arr) noexcept {
            nodesPtr = arr.nodesPtr;
            mSize = arr.mSize;
            arr.nodesPtr = nullptr;
        }
        void operator=(HashArray &&arr) noexcept {
            operator=(arr);
        }
        ~HashArray() noexcept {
            if (nodesPtr)
                allocHandle.Free(nodesPtr);
        }
        LinkNode *const &operator[](size_t i) const noexcept {
            return nodesPtr[i];
        }
        LinkNode *&operator[](size_t i) noexcept {
            return nodesPtr[i];
        }
    };

    HashArray nodeVec;
    Pool<LinkNode, allocType, true> pool;
    inline static const Hash hsFunc;
    inline static const Equal eqFunc;
    template<typename A, typename... B>
    LinkNode *GetNewLinkNode(size_t hashValue, A &&key, B &&...args) {
        LinkNode *newNode = pool.New(hashValue, allocatedNodes.size(), std::forward<A>(key), std::forward<B>(args)...);
        allocatedNodes.push_back(newNode);
        return newNode;
    }
    void DeleteLinkNode(LinkNode *oldNode) {
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
        if (capacity >= newCapacity)
            return;
        allocatedNodes.reserve(newCapacity);
        HashArray newNode(newCapacity);
        for (auto node : allocatedNodes) {
            auto next = node->next;
            node->last = nullptr;
            node->next = nullptr;
            size_t hashValue = node->hashValue;
            hashValue = GetHash(hashValue, newCapacity);
            LinkNode *&targetHeaderLink = newNode[hashValue];
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
        return Index(nullptr, nullptr);
    }
    void Remove(LinkNode *node) {
        size_t hashValue = GetHash(node->hashValue, nodeVec.size());
        if (nodeVec[hashValue] == node) {
            nodeVec[hashValue] = node->next;
        }
        if (node->last)
            node->last->next = node->next;
        if (node->next)
            node->next->last = node->last;
        DeleteLinkNode(node);
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
        if (capacity < 2)
            capacity = 2;
        capacity = GetPow2Size(capacity);
        nodeVec = HashArray(capacity);
        allocatedNodes.reserve(capacity);
    }
    HashMap(HashMap &&map) = default;
    HashMap(HashMap const &map) = delete;

    template<typename Arg>
    void operator=(Arg &&map) {
        this->~HashMap();
        new (this) HashMap(std::forward<Arg>(map));
    }
    ~HashMap() noexcept {
        for (auto &ite : allocatedNodes) {
            pool.Delete(ite);
        }
    }
    HashMap() noexcept : HashMap(16) {}
    ///////////////////////
    template<typename Key, typename... ARGS>
    Index ForceEmplace(Key &&key, ARGS &&...args) {
        size_t hashOriginValue = hsFunc(std::forward<Key>(key));
        size_t hashValue;

        hashValue = GetHash(hashOriginValue, nodeVec.size());
        for (LinkNode *node = nodeVec[hashValue]; node != nullptr; node = node->next) {
            if (eqFunc(node->first, std::forward<Key>(key))) {
                node->second.~V();
                new (&node->second) V(std::forward<ARGS>(args)...);
                return Index(this, node);
            }
        }

        size_t targetCapacity = (size_t)((allocatedNodes.size() + 1) / 0.75);
        if (targetCapacity < 16)
            targetCapacity = 16;
        if (targetCapacity >= nodeVec.size()) {
            Resize(GetPow2Size(targetCapacity));
            hashValue = GetHash(hashOriginValue, nodeVec.size());
        }
        LinkNode *newNode = GetNewLinkNode(hashOriginValue, std::forward<Key>(key), std::forward<ARGS>(args)...);
        LinkNode::Add(nodeVec[hashValue], newNode);
        return Index(this, newNode);
    }

    template<typename Key, typename... ARGS>
    Index Emplace(Key &&key, ARGS &&...args) {
        size_t hashOriginValue = hsFunc(std::forward<Key>(key));
        size_t hashValue;

        hashValue = GetHash(hashOriginValue, nodeVec.size());
        for (LinkNode *node = nodeVec[hashValue]; node != nullptr; node = node->next) {
            if (eqFunc(node->first, std::forward<Key>(key))) {
                return Index(this, node);
            }
        }

        size_t targetCapacity = (size_t)((allocatedNodes.size() + 1) / 0.75);
        if (targetCapacity < 16)
            targetCapacity = 16;
        if (targetCapacity >= nodeVec.size()) {
            Resize(GetPow2Size(targetCapacity));
            hashValue = GetHash(hashOriginValue, nodeVec.size());
        }
        LinkNode *newNode = GetNewLinkNode(hashOriginValue, std::forward<Key>(key), std::forward<ARGS>(args)...);
        LinkNode::Add(nodeVec[hashValue], newNode);
        return Index(this, newNode);
    }

    template<typename Key, typename... ARGS>
    std::pair<Index, bool> TryEmplace(Key &&key, ARGS &&...args) {
        size_t hashOriginValue = hsFunc(std::forward<Key>(key));
        size_t hashValue;

        hashValue = GetHash(hashOriginValue, nodeVec.size());
        for (LinkNode *node = nodeVec[hashValue]; node != nullptr; node = node->next) {
            if (eqFunc(node->first, std::forward<Key>(key))) {
                return std::pair<Index, bool>(Index(this, node), false);
            }
        }

        size_t targetCapacity = (size_t)((allocatedNodes.size() + 1) / 0.75);
        if (targetCapacity < 16)
            targetCapacity = 16;
        if (targetCapacity >= nodeVec.size()) {
            Resize(GetPow2Size(targetCapacity));
            hashValue = GetHash(hashOriginValue, nodeVec.size());
        }
        LinkNode *newNode = GetNewLinkNode(hashOriginValue, std::forward<Key>(key), std::forward<ARGS>(args)...);
        LinkNode::Add(nodeVec[hashValue], newNode);
        return std::pair<Index, bool>(Index(this, newNode), true);
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
        size_t hashValue = GetHash(hashOriginValue, nodeVec.size());
        for (LinkNode *node = nodeVec[hashValue]; node != nullptr; node = node->next) {
            if (eqFunc(node->first, std::forward<Key>(key))) {
                return Index(this, node);
            }
        }

        return EmptyIndex();
    }

    void Remove(K const &key) noexcept {
        size_t hashOriginValue = hsFunc(key);
        size_t hashValue = GetHash(hashOriginValue, nodeVec.size());
        LinkNode *&startNode = nodeVec[hashValue];
        for (LinkNode *node = startNode; node != nullptr; node = node->next) {
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

    template<typename Key>
    void TRemove(Key &&key) noexcept {
        size_t hashOriginValue = hsFunc(std::forward<Key>(key));
        size_t hashValue = GetHash(hashOriginValue, nodeVec.size());
        LinkNode *&startNode = nodeVec[hashValue];
        for (LinkNode *node = startNode; node != nullptr; node = node->next) {
            if (eqFunc(node->first, std::forward<Key>(key))) {
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

    void Remove(const Index &ite) noexcept {
        Remove(ite.node);
    }
    void Remove(NodePair const &ite) noexcept {
        Remove(const_cast<LinkNode *>(static_cast<LinkNode const *>(&ite)));
    }
    //void Remove(NodePair*)
    template<typename Key>
    V &operator[](Key &&key) noexcept {

        size_t hashOriginValue = hsFunc(std::forward<Key>(key));
        size_t hashValue = GetHash(hashOriginValue, nodeVec.size());
        for (LinkNode *node = nodeVec[hashValue]; node != nullptr; node = node->next) {
            if (eqFunc(node->first, std::forward<Key>(key))) {
                return node->second;
            }
        }

        return *(V *)nullptr;
    }
    template<typename Key>
    V const &operator[](Key &&key) const noexcept {

        size_t hashOriginValue = hsFunc(std::forward<Key>(key));
        size_t hashValue = GetHash(hashOriginValue, nodeVec.size());
        for (LinkNode *node = nodeVec[hashValue]; node != nullptr; node = node->next) {
            if (eqFunc(node->first, std::forward<Key>(key))) {
                return node->second;
            }
        }

        return *(V const *)nullptr;
    }
    void Clear() noexcept {
        if (allocatedNodes.empty())
            return;
        nodeVec.ClearAll();
        for (auto &ite : allocatedNodes) {
            pool.Delete(ite);
        }
        allocatedNodes.clear();
    }
    [[nodiscard]] size_t size() const noexcept { return allocatedNodes.size(); }
    [[nodiscard]] size_t GetCapacity() const noexcept { return nodeVec.size(); }
    ////////////////////// Thread-Safe support
    template<typename Mutex, typename Key, typename... ARGS>
    Index Emplace_Lock(Mutex &mtx, Key &&key, ARGS &&...args) {
        size_t hashOriginValue = hsFunc(std::forward<Key>(key));
        size_t hashValue;
        std::lock_guard<Mutex> lck(mtx);

        hashValue = GetHash(hashOriginValue, nodeVec.size());
        for (LinkNode *node = nodeVec[hashValue]; node != nullptr; node = node->next) {
            if (eqFunc(node->first, std::forward<Key>(key))) {
                return Index(this, node);
            }
        }

        auto targetCapacity = (size_t)((allocatedNodes.size() + 1) / 0.75);
        if (targetCapacity < 16)
            targetCapacity = 16;
        if (targetCapacity >= nodeVec.size()) {
            Resize(GetPow2Size(targetCapacity));
            hashValue = GetHash(hashOriginValue, nodeVec.size());
        }
        LinkNode *newNode = GetNewLinkNode(hashOriginValue, std::forward<Key>(key), std::forward<ARGS>(args)...);
        LinkNode::Add(nodeVec[hashValue], newNode);
        return Index(this, newNode);
    }
    template<typename Mutex, typename Key>
    Index Find_Lock(Mutex &mtx, Key &&key) const noexcept {
        size_t hashOriginValue = hsFunc(std::forward<Key>(key));
        std::lock_guard<Mutex> lck(mtx);
        size_t hashValue = GetHash(hashOriginValue, nodeVec.size());
        for (LinkNode *node = nodeVec[hashValue]; node != nullptr; node = node->next) {
            if (eqFunc(node->first, std::forward<Key>(key))) {
                return Index(this, node);
            }
        }

        return EmptyIndex();
    }
    template<typename Mutex>
    void Remove_Lock(Mutex &mtx, K const &key) noexcept {
        size_t hashOriginValue = hsFunc(key);
        std::lock_guard<Mutex> lck(mtx);
        size_t hashValue = GetHash(hashOriginValue, nodeVec.size());
        LinkNode *&startNode = nodeVec[hashValue];
        for (LinkNode *node = startNode; node != nullptr; node = node->next) {
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
    template<typename Mutex, typename Key, typename... ARGS>
    Index ForceEmplace_Lock(Mutex &mtx, Key &&key, ARGS &&...args) {
        size_t hashOriginValue = hsFunc(std::forward<Key>(key));
        size_t hashValue;
        std::lock_guard<Mutex> lck(mtx);
        hashValue = GetHash(hashOriginValue, nodeVec.size());
        for (LinkNode *node = nodeVec[hashValue]; node != nullptr; node = node->next) {
            if (eqFunc(node->first, std::forward<Key>(key))) {
                node->second.~V();
                new (&node->second) V(std::forward<ARGS>(args)...);
                return Index(this, node);
            }
        }

        size_t targetCapacity = (size_t)((allocatedNodes.size() + 1) / 0.75);
        if (targetCapacity < 16)
            targetCapacity = 16;
        if (targetCapacity >= nodeVec.size()) {
            Resize(GetPow2Size(targetCapacity));
            hashValue = GetHash(hashOriginValue, nodeVec.size());
        }
        LinkNode *newNode = GetNewLinkNode(hashOriginValue, std::forward<Key>(key), std::forward<ARGS>(args)...);
        LinkNode::Add(nodeVec[hashValue], newNode);
        return Index(this, newNode);
    }
    template<typename Mutex, typename Key, typename... ARGS>
    std::pair<Index, bool> TryEmplace_Lock(Mutex &mtx, Key &&key, ARGS &&...args) {
        size_t hashOriginValue = hsFunc(std::forward<Key>(key));
        size_t hashValue;
        std::lock_guard<Mutex> lck(mtx);
        hashValue = GetHash(hashOriginValue, nodeVec.size());
        for (LinkNode *node = nodeVec[hashValue]; node != nullptr; node = node->next) {
            if (eqFunc(node->first, std::forward<Key>(key))) {
                return std::pair<Index, bool>(Index(this, node), false);
            }
        }

        size_t targetCapacity = (size_t)((allocatedNodes.size() + 1) / 0.75);
        if (targetCapacity < 16)
            targetCapacity = 16;
        if (targetCapacity >= nodeVec.size()) {
            Resize(GetPow2Size(targetCapacity));
            hashValue = GetHash(hashOriginValue, nodeVec.size());
        }
        LinkNode *newNode = GetNewLinkNode(hashOriginValue, std::forward<Key>(key), std::forward<ARGS>(args)...);
        LinkNode::Add(nodeVec[hashValue], newNode);
        return std::pair<Index, bool>(Index(this, newNode), true);
    }
    template<typename Mutex, typename Key>
    void TRemove_Lock(Mutex &mtx, Key &&key) noexcept {
        size_t hashOriginValue = hsFunc(std::forward<Key>(key));
        std::lock_guard<Mutex> lck(mtx);
        size_t hashValue = GetHash(hashOriginValue, nodeVec.size());
        LinkNode *&startNode = nodeVec[hashValue];
        for (LinkNode *node = startNode; node != nullptr; node = node->next) {
            if (eqFunc(node->first, std::forward<Key>(key))) {
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
};

template<typename K, typename V, typename Hash, typename Equal, VEngine_AllocType allocType>
inline K const &HashMap<K, V, Hash, Equal, allocType>::Index::Key() const noexcept {
    return node->first;
}
template<typename K, typename V, typename Hash, typename Equal, VEngine_AllocType allocType>
inline V &HashMap<K, V, Hash, Equal, allocType>::Index::Value() const noexcept {
    return node->second;
}
}// namespace vstd