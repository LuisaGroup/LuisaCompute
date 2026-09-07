#pragma once

#include <functional>
#include <utility>

#if __cpp_exceptions
#include <stdexcept>
#else
#include <cstdio>
#include <cstdlib>
#endif

#include <fc/btree.h>

#include <luisa/core/stl/memory.h>

namespace luisa {

namespace detail {

// Reports a missing key in luisa::map::at().
// Keeps throwing semantics when exceptions are enabled and falls back
// to a fatal abort so the header also compiles with -fno-exceptions.
[[noreturn]] inline void map_at_error()
#if !__cpp_exceptions
    noexcept
#endif
{
#if __cpp_exceptions
    throw std::out_of_range{"luisa::map::at"};
#else
    std::fprintf(stderr, "luisa::map::at: key not found\n");
    std::abort();
#endif
}

template<typename Key, typename Value, typename Compare, template<typename> class Allocator, bool AllowDup>
class btree_map_base
    : public std::conditional_t<AllowDup,
                                frozenca::BTreeMultiMap<Key, Value, 64, Compare, Allocator>,
                                frozenca::BTreeMap<Key, Value, 64, Compare, Allocator>> {
    using Base = std::conditional_t<AllowDup,
                                    frozenca::BTreeMultiMap<Key, Value, 64, Compare, Allocator>,
                                    frozenca::BTreeMap<Key, Value, 64, Compare, Allocator>>;

public:
    using Base::Base;
    using Base::emplace;

    // two-argument emplace that also accepts lvalue keys/values
    template<typename K_, typename V_>
    auto emplace(K_ &&k, V_ &&v) {
        Key kk{std::forward<K_>(k)};
        Value vv{std::forward<V_>(v)};
        return Base::emplace(std::move(kk), std::move(vv));
    }

    using mapped_type = Value;
    using key_type = typename Base::key_type;
    using value_type = typename Base::value_type;
    using size_type = typename Base::size_type;
    using iterator = typename Base::iterator_type;
    using const_iterator = typename Base::const_iterator_type;
    using reverse_iterator = typename Base::reverse_iterator_type;
    using const_reverse_iterator = typename Base::const_reverse_iterator_type;
};

} // namespace detail

template<typename Key, typename Value,
         typename Compare = std::less<Key>,
         template<typename> class Allocator = luisa::allocator>
class map : public detail::btree_map_base<Key, Value, Compare, Allocator, false> {
    using Base = detail::btree_map_base<Key, Value, Compare, Allocator, false>;

public:
    using Base::Base;
    [[nodiscard]] typename Base::size_type count(const Key &key) const {
        return Base::contains(key) ? 1u : 0u;
    }
    [[nodiscard]] Value &at(const Key &key) {
        auto it = Base::find(key);
        if (it == Base::end()) [[unlikely]] { luisa::detail::map_at_error(); }
        return it->second;
    }
    [[nodiscard]] const Value &at(const Key &key) const {
        auto it = Base::find(key);
        if (it == Base::end()) [[unlikely]] { luisa::detail::map_at_error(); }
        return it->second;
    }
};

template<typename Key, typename Value,
         typename Compare = std::less<Key>,
         template<typename> class Allocator = luisa::allocator>
class multimap : public detail::btree_map_base<Key, Value, Compare, Allocator, true> {
    using Base = detail::btree_map_base<Key, Value, Compare, Allocator, true>;

public:
    using Base::Base;
};

template<typename Key,
         typename Compare = std::less<Key>,
         template<typename> class Allocator = luisa::allocator>
class set : public frozenca::BTreeSet<Key, 64, Compare, Allocator> {
    using Base = frozenca::BTreeSet<Key, 64, Compare, Allocator>;

public:
    using Base::Base;
    [[nodiscard]] typename Base::size_type count(const Key &key) const {
        return Base::contains(key) ? 1u : 0u;
    }
};

template<typename Key,
         typename Compare = std::less<Key>,
         template<typename> class Allocator = luisa::allocator>
class multiset : public frozenca::BTreeMultiSet<Key, 64, Compare, Allocator> {
    using Base = frozenca::BTreeMultiSet<Key, 64, Compare, Allocator>;

public:
    using Base::Base;
};

} // namespace luisa
