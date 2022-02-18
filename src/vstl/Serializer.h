#pragma once
#include <vstl/Common.h>
#include <vstl/functional.h>
namespace vstd {
template<typename T>
void ReverseBytes(T &num) {
    using Type = std::remove_cvref_t<T>;
    if constexpr (!std::is_trivial_v<Type>) return;
    if constexpr (sizeof(Type) == 2) {
        uint8_t const *p = reinterpret_cast<uint8_t const *>(&num);
        uint16_t result = 0;
        uint8_t ofst = 8;
        for (uint8_t i = 0; i < 2; ++i) {
            result |= static_cast<uint16_t>(p[i]) << ofst;
            ofst -= 8;
        }
        num = *reinterpret_cast<Type *>(&result);

    } else if constexpr (sizeof(Type) == 4) {
        uint8_t const *p = reinterpret_cast<uint8_t const *>(&num);
        uint result = 0;
        uint8_t ofst = 24;
        for (uint8_t i = 0; i < 4; ++i) {
            result |= static_cast<uint>(p[i]) << ofst;
            ofst -= 8;
        }
        num = *reinterpret_cast<Type *>(&result);
    } else if constexpr (sizeof(Type) == 8) {
        uint8_t const *p = reinterpret_cast<uint8_t const *>(&num);
        uint64 result = 0;
        uint8_t ofst = 56;
        for (uint8_t i = 0; i < 8; ++i) {
            result |= static_cast<uint64>(p[i]) << ofst;
            ofst -= 8;
        }
        num = *reinterpret_cast<Type *>(&result);
    }
}

template<typename T, bool reverseBytes = false>
struct SerDe {
    static_assert(!std::is_pointer_v<T>, "pointer can not be serialized!");
    static T Get(vstd::span<uint8_t const> &data) {
        using Type = std::remove_cvref_t<T>;
        if (reverseBytes) {
            Type ptr = *reinterpret_cast<Type const *>(data.data());
            ReverseBytes(ptr);
            data = vstd::span<uint8_t const>(data.data() + sizeof(Type), data.size() - sizeof(Type));
            return ptr;
        } else {
            Type const *ptr = reinterpret_cast<Type const *>(data.data());
            data = vstd::span<uint8_t const>(data.data() + sizeof(Type), data.size() - sizeof(Type));
            return *ptr;
        }
    }
    static void Set(T const &data, vector<uint8_t> &vec) {
        using Type = std::remove_cvref_t<T>;
        if (reverseBytes) {
            Type tempData = data;
            ReverseBytes(tempData);
            vec.push_back_all(reinterpret_cast<uint8_t const *>(&tempData), sizeof(Type));
        } else {
            vec.push_back_all(reinterpret_cast<uint8_t const *>(&data), sizeof(Type));
        }
    }
};
template<bool reverseBytes>
struct SerDe<vstd::string, reverseBytes> {
    static vstd::string Get(vstd::span<uint8_t const> &sp) {
        auto strLen = SerDe<uint, reverseBytes>::Get(sp);
        auto ptr = sp.data();
        sp = vstd::span<uint8_t const>(ptr + strLen, sp.size() - strLen);
        return vstd::string(reinterpret_cast<char const *>(ptr), strLen);
    }
    static void Set(vstd::string const &data, vector<uint8_t> &arr) {
        SerDe<uint, reverseBytes>::Set(data.size(), arr);
        arr.push_back_all(reinterpret_cast<uint8_t const *>(data.data()), data.size());
    }
};
template<bool reverseBytes>
struct SerDe<std::string_view, reverseBytes> {
    static std::string_view Get(vstd::span<uint8_t const> &sp) {
        auto strLen = SerDe<uint, reverseBytes>::Get(sp);
        auto ptr = sp.data();
        sp = vstd::span<uint8_t const>(ptr + strLen, sp.size() - strLen);
        return std::string_view(
            reinterpret_cast<char const *>(ptr),
            strLen);
    }
    static void Set(std::string_view const &data, vector<uint8_t> &arr) {
        SerDe<uint, reverseBytes>::Set(data.size(), arr);
        arr.push_back_all(reinterpret_cast<uint8_t const *>(data.data()), data.size());
    }
};

template<typename T, VEngine_AllocType alloc, bool tri, bool reverseBytes>
struct SerDe<vector<T, alloc, tri>, reverseBytes> {
    using Value = vector<T, alloc, tri>;
    static Value Get(vstd::span<T const> &sp) {
        Value sz;
        auto s = SerDe<uint, reverseBytes>::Get(sp);
        sz.push_back_func(
            [&]() {
                return SerDe<T, reverseBytes>::Get(sp);
            },
            s);
        return sz;
    }
    static void Set(Value const &data, vector<T> &arr) {
        SerDe<uint, reverseBytes>::Set(data.size(), arr);
        for (auto &&i : data) {
            SerDe<T, reverseBytes>::Set(i, arr);
        }
    }
};

template<typename K, typename V, typename Hash, typename Equal, VEngine_AllocType alloc, bool reverseBytes>
struct SerDe<HashMap<K, V, Hash, Equal, alloc>, reverseBytes> {
    using Value = HashMap<K, V, Hash, Equal, alloc>;
    static Value Get(vstd::span<uint8_t const> &sp) {
        Value sz;
        auto capa = SerDe<uint, reverseBytes>::Get(sp);
        sz.reserve(capa);
        for (auto &&i : range(capa)) {
            auto key = SerDe<K, reverseBytes>::Get(sp);
            auto value = SerDe<V, reverseBytes>::Get(sp);
            sz.Emplace(
                std::move(key),
                std::move(value));
        }
        return sz;
    }
    static void Set(Value const &data, vector<uint8_t> &arr) {
        SerDe<uint, reverseBytes>::Set(data.size(), arr);
        for (auto &&i : data) {
            SerDe<K, reverseBytes>::Set(i.first, arr);
            SerDe<V, reverseBytes>::Set(i.second, arr);
        }
    }
};
template<typename... Args, bool reverseBytes>
struct SerDe<variant<Args...>, reverseBytes> {
    using Value = variant<Args...>;
    template<typename T>
    static void ExecuteGet(void *placePtr, vstd::span<uint8_t const> &sp) {
        new (placePtr) Value(SerDe<T, reverseBytes>::Get(sp));
    }
    template<typename T>
    static void ExecuteSet(void const *placePtr, vector<uint8_t> &sp) {
        SerDe<T, reverseBytes>::Set(*reinterpret_cast<T const *>(placePtr), sp);
    }
    static Value Get(vstd::span<uint8_t const> &sp) {
        auto type = SerDe<uint8_t, reverseBytes>::Get(sp);
        funcPtr_t<void(void *, vstd::span<uint8_t const> &)> ptrs[sizeof...(Args)] = {
            ExecuteGet<Args>...};
        Value v;
        v.update(type, [&](void *ptr) {
            ptrs[type](ptr, sp);
        });
        return v;
    }
    static void Set(Value const &data, vector<uint8_t> &arr) {
        SerDe<uint8_t, reverseBytes>::Set(data.GetType(), arr);
        funcPtr_t<void(void const *, vector<uint8_t> &)> ptrs[sizeof...(Args)] = {
            ExecuteSet<Args>...};
        ptrs[data.GetType()](&data, arr);
    }
};

template<typename A, typename B, bool reverseBytes>
struct SerDe<std::pair<A, B>, reverseBytes> {
    using Value = std::pair<A, B>;
    static Value Get(vstd::span<uint8_t const> &sp) {
        return Value{SerDe<A, reverseBytes>::Get(sp), SerDe<B, reverseBytes>::Get(sp)};
    }
    static void Set(Value const &data, vector<uint8_t> &arr) {
        SerDe<A, reverseBytes>::Set(data.first, arr);
        SerDe<B, reverseBytes>::Set(data.second, arr);
    }
};

template<bool reverseBytes>
struct SerDe<vstd::span<uint8_t const>, reverseBytes> {
    using Value = vstd::span<uint8_t const>;
    static Value Get(Value &sp) {
        auto sz = SerDe<uint, reverseBytes>::Get(sp);
        Value v(sp.data(), sz);
        sp = Value(sp.data() + sz, sp.size() - sz);
        return v;
    }
    static void Set(Value const &data, vector<uint8_t> &arr) {
        SerDe<uint, reverseBytes>::Set(data.size(), arr);
        arr.push_back_all(data);
    }
};
template<typename T, size_t sz, bool reverseBytes>
struct SerDe<std::array<T, sz>, reverseBytes> {
    using Value = std::array<T, sz>;
    static Value Get(vstd::span<uint8_t const> &sp) {
        Value v;
        for (auto &&i : v) {
            i = SerDe<T, reverseBytes>::Get(sp);
        }
        return v;
    }
    static void Set(Value const &data, vector<uint8_t> &arr) {
        for (auto &&i : data) {
            SerDe<T, reverseBytes>::Set(i);
        }
    }
};

template<typename Func>
struct SerDeAll_Impl;

template<typename Ret, typename... Args>
struct SerDeAll_Impl<Ret(Args...)> {
    template<typename Class, typename Func>
    static Ret CallMemberFunc(Class *ptr, Func func, vstd::span<uint8_t const> data) {
        auto closureFunc = [&](Args &&...args) {
            return (ptr->*func)(std::forward<Args>(args)...);
        };
        return std::apply(closureFunc, std::tuple<Args...>{SerDe<std::remove_cvref_t<Args>>::Get(data)...});
    }

    static vector<uint8_t> Ser(
        Args const &...args) {
        vector<uint8_t> vec;
        auto lst = {(SerDe<std::remove_cvref_t<Args>>::Set(args, vec), ' ')...};
        return vec;
    }

    template<typename Func>
    static decltype(auto) Call(
        Func &&func) {
        return [f = std::forward<Func>(func)](vstd::span<uint8_t const> data) {
            return std::apply(f, std::tuple<Args...>{SerDe<std::remove_cvref_t<Args>>::Get(data)...});
        };
    }
};
}// namespace vstd