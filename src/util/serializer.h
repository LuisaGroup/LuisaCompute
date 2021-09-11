#pragma once
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <util/HashMap.h>
#include <util/MetaLib.h>
#include <vector>
namespace vstd {
template<typename T>
struct SerDe {
    static_assert(std::is_trivial_v<T>, "only trivial type can be serialized!");
    static_assert(!std::is_pointer_v<T>, "pointer can not be serialized!");
    static T Get(std::span<uint8_t const> &data) {
        T const *ptr = reinterpret_cast<T const *>(data.data());
        data = std::span<uint8_t const>(data.data() + sizeof(T), data.size() - sizeof(T));
        return *ptr;
    }
    static void Set(T const &data, std::vector<uint8_t> &vec) {
        auto ptr = reinterpret_cast<uint8_t const *>(&data);
        vec.insert(vec.end(), ptr, ptr + sizeof(T));
    }
};
template<>
struct SerDe<std::string> {
    static std::string Get(std::span<uint8_t const> &sp) {
        auto strLen = SerDe<uint>::Get(sp);
        auto ptr = sp.data();
        sp = std::span<uint8_t const>(ptr + strLen, sp.size() - strLen);
        return std::string(std::string_view(
            reinterpret_cast<char const *>(ptr),
            strLen));
    }
    static void Set(std::string const &data, std::vector<uint8_t> &arr) {
        SerDe<uint>::Set(data.size(), arr);
        arr.insert(arr.end(), data.begin(), data.end());
    }
};
template<>
struct SerDe<std::string_view> {
    static std::string_view Get(std::span<uint8_t const> &sp) {
        auto strLen = SerDe<uint>::Get(sp);
        auto ptr = sp.data();
        sp = std::span<uint8_t const>(ptr + strLen, sp.size() - strLen);
        return std::string_view(
            reinterpret_cast<char const *>(ptr),
            strLen);
    }
    static void Set(std::string_view const &data, std::vector<uint8_t> &arr) {
        SerDe<uint>::Set(data.size(), arr);
        arr.insert(arr.end(), data.begin(), data.end());
    }
};

template<typename T>
struct SerDe<std::vector<T>> {
    using Value = std::vector<T>;
    static Value Get(std::span<uint8_t const> &sp) {
        Value sz;
        auto s = SerDe<uint>::Get(sp);
        for (auto i : range(s)) {
            sz.push_back(SerDe<T>::Get(sp));
        }

        return sz;
    }
    static void Set(Value const &data, std::vector<uint8_t> &arr) {
        SerDe<uint>::Set(data.size(), arr);
        for (auto &&i : data) {
            SerDe<T>::Set(i, arr);
        }
    }
};

template<typename K, typename V, typename Hash, typename Equal, VEngine_AllocType alloc>
struct SerDe<HashMap<K, V, Hash, Equal, alloc>> {
    using Value = HashMap<K, V, Hash, Equal, alloc>;
    static Value Get(std::span<uint8_t const> &sp) {
        Value sz;
        auto capa = SerDe<uint>::Get(sp);
        sz.reserve(capa);
        for (auto &&i : vstd::range(capa)) {
            auto key = SerDe<K>::Get(sp);
            auto value = SerDe<V>::Get(sp);
            sz.Emplace(
                std::move(key),
                std::move(value));
        }
        return sz;
    }
    static void Set(Value const &data, std::vector<uint8_t> &arr) {
        SerDe<uint>::Set(data.size(), arr);
        for (auto &&i : data) {
            SerDe<K>::Set(i.first, arr);
            SerDe<V>::Set(i.second, arr);
        }
    }
};

template<typename... Args>
struct SerDe<vstd::variant<Args...>> {
    using Value = vstd::variant<Args...>;
    template<typename T>
    static void ExecuteGet(void *placePtr, std::span<uint8_t const> &sp) {
        new (placePtr) Value(SerDe<T>::Get(sp));
    }
    template<typename T>
    static void ExecuteSet(void const *placePtr, std::vector<uint8_t> &sp) {
        SerDe<T>::Set(*reinterpret_cast<T const *>(placePtr), sp);
    }
    static Value Get(std::span<uint8_t const> &sp) {
        auto type = SerDe<uint8_t>::Get(sp);
        funcPtr_t<void(void *, std::span<uint8_t const> &)> ptrs[sizeof...(Args)] = {
            ExecuteGet<Args>...};
        Value v;
        v.update(type, [&](void *ptr) {
            ptrs[type](ptr, sp);
        });
        return v;
    }
    static void Set(Value const &data, std::vector<uint8_t> &arr) {
        SerDe<uint8_t>::Set(data.GetType(), arr);
        funcPtr_t<void(void const *, std::vector<uint8_t> &)> ptrs[sizeof...(Args)] = {
            ExecuteSet<Args>...};
        ptrs[data.GetType()](&data, arr);
    }
};

template<typename A, typename B>
struct SerDe<std::pair<A, B>> {
    using Value = std::pair<A, B>;
    static Value Get(std::span<uint8_t const> &sp) {
        return Value{SerDe<A>::Get(sp), SerDe<B>::Get(sp)};
    }
    static void Set(Value const &data, std::vector<uint8_t> &arr) {
        SerDe<A>::Set(data.first, arr);
        SerDe<B>::Set(data.second, arr);
    }
};

template<>
struct SerDe<std::span<uint8_t const>> {
    using Value = std::span<uint8_t const>;
    static Value Get(Value &sp) {
        auto sz = SerDe<uint>::Get(sp);
        Value v(sp.data(), sz);
        sp = Value(sp.data() + sz, sp.size() - sz);
        return v;
    }
    static void Set(Value const &data, std::vector<uint8_t> &arr) {
        SerDe<uint>::Set(data.size(), arr);
        arr.insert(arr.end(), data.begin(), data.end());
    }
};
template<typename T, size_t sz>
struct SerDe<std::array<T, sz>> {
    using Value = std::array<T, sz>;
    static Value Get(std::span<uint8_t const> &sp) {
        Value v;
        for (auto &&i : v) {
            i = vstd::SerDe<T>::Get(sp);
        }
        return v;
    }
    static void Set(Value const &data, std::vector<uint8_t> &arr) {
        for (auto &&i : data) {
            vstd::SerDe<T>::Set(i);
        }
    }
};

template<typename Func>
struct SerDeAll_Impl;

template<typename Ret, typename... Args>
struct SerDeAll_Impl<Ret(Args...)> {
    template<typename Class, typename Func>
    static Ret CallMemberFunc(Class *ptr, Func func, std::span<uint8_t const> data) {
        auto closureFunc = [&](Args &&...args) {
            (ptr->*func)(std::forward<Args>(args)...);
        };
        return std::apply(closureFunc, std::tuple<Args...>{SerDe<std::remove_cvref_t<Args>>::Get(data)...});
    }

    static std::vector<uint8_t> Ser(
        Args const &...args) {
        std::vector<uint8_t> vec;
        auto lst = {(SerDe<std::remove_cvref_t<Args>>::Set(args, vec), ' ')...};
        return vec;
    }

    template<typename Func>
    static decltype(auto) Call(
        Func &&func) {
        return [f = std::forward<Func>(func)](std::span<uint8_t const> data) {
            return std::apply(f, std::tuple<Args...>{SerDe<std::remove_cvref_t<Args>>::Get(data)...});
        };
    }
};

template<typename Func>
using SerDeAll = SerDeAll_Impl<FuncType<std::remove_cvref_t<Func>>>;
template<typename Func>
using SerDeAll_Member = SerDeAll_Impl<typename FunctionTemplateGlobal::memFuncPtr<std::remove_cvref_t<Func>>::Type::FuncType>;
}// namespace vstd