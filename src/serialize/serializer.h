#pragma once

#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vstl/HashMap.h>
#include <vstl/MetaLib.h>
#include <vstl/VGuid.h>
#include <vector>

namespace vstd {

template<typename T>
struct Serializer {
    static_assert(std::is_trivial_v<T>, "only trivial type can be serialized!");
    static_assert(!std::is_pointer_v<T>, "pointer can not be serialized!");
    static T Get(std::span<uint8_t const> &data) {
        T const *ptr = reinterpret_cast<T const *>(data.data());
        data = std::span<uint8_t const>(data.data() + sizeof(T), data.size() - sizeof(T));
        return *ptr;
    }
    static void Set(T const &data, luisa::vector<uint8_t> &vec) {
        auto ptr = reinterpret_cast<uint8_t const *>(&data);
        vec.insert(vec.end(), ptr, ptr + sizeof(T));
    }
};

template<>
struct Serializer<Guid> {
    using Value = Guid;
    static Value Get(std::span<uint8_t const> &sp) {
        return vstd::Serializer<Guid::GuidData>::Get(sp);
    }
    static void Set(Value const &data, luisa::vector<uint8_t> &arr) {
        vstd::Serializer<Guid::GuidData>::Set(data.ToBinary(), arr);
    }
};

template<>
struct Serializer<luisa::string> {
    static luisa::string Get(std::span<uint8_t const> &sp) {
        auto strLen = Serializer<uint>::Get(sp);
        auto ptr = sp.data();
        sp = std::span<uint8_t const>(ptr + strLen, sp.size() - strLen);
        return {reinterpret_cast<char const *>(ptr), strLen};
    }
    static void Set(luisa::string const &data, luisa::vector<uint8_t> &arr) {
        Serializer<uint>::Set(data.size(), arr);
        arr.insert(arr.end(), data.cbegin(), data.cend());
    }
};

template<>
struct Serializer<std::string_view> {
    static std::string_view Get(std::span<uint8_t const> &sp) {
        auto strLen = Serializer<uint>::Get(sp);
        auto ptr = sp.data();
        sp = std::span<uint8_t const>(ptr + strLen, sp.size() - strLen);
        return {reinterpret_cast<char const *>(ptr), strLen};
    }
    static void Set(std::string_view const &data, luisa::vector<uint8_t> &arr) {
        Serializer<uint>::Set(data.size(), arr);
        arr.insert(arr.end(), data.cbegin(), data.cend());
    }
};

template<typename T>
struct Serializer<luisa::vector<T>> {
    using Value = luisa::vector<T>;
    static Value Get(std::span<uint8_t const> &sp) {
        Value sz;
        auto s = Serializer<uint>::Get(sp);
        for (auto i : range(s)) {
            sz.push_back(Serializer<T>::Get(sp));
        }

        return sz;
    }
    static void Set(Value const &data, luisa::vector<uint8_t> &arr) {
        Serializer<uint>::Set(data.size(), arr);
        for (auto &&i : data) {
            Serializer<T>::Set(i, arr);
        }
    }
};

template<typename K, typename V, typename Hash, typename Equal, VEngine_AllocType alloc>
struct Serializer<HashMap<K, V, Hash, Equal, alloc>> {
    using Value = HashMap<K, V, Hash, Equal, alloc>;
    static Value Get(std::span<uint8_t const> &sp) {
        Value sz;
        auto capa = Serializer<uint>::Get(sp);
        sz.reserve(capa);
        for (auto &&i : vstd::range(capa)) {
            auto key = Serializer<K>::Get(sp);
            auto value = Serializer<V>::Get(sp);
            sz.Emplace(
                std::move(key),
                std::move(value));
        }
        return sz;
    }
    static void Set(Value const &data, luisa::vector<uint8_t> &arr) {
        Serializer<uint>::Set(data.size(), arr);
        for (auto &&i : data) {
            Serializer<K>::Set(i.first, arr);
            Serializer<V>::Set(i.second, arr);
        }
    }
};

template<typename... Args>
struct Serializer<vstd::variant<Args...>> {
    using Value = vstd::variant<Args...>;
    template<typename T>
    static void ExecuteGet(void *placePtr, std::span<uint8_t const> &sp) {
        new (placePtr) Value(Serializer<T>::Get(sp));
    }
    template<typename T>
    static void ExecuteSet(void const *placePtr, luisa::vector<uint8_t> &sp) {
        Serializer<T>::Set(*reinterpret_cast<T const *>(placePtr), sp);
    }
    static Value Get(std::span<uint8_t const> &sp) {
        auto type = Serializer<uint8_t>::Get(sp);
        funcPtr_t<void(void *, std::span<uint8_t const> &)> ptrs[sizeof...(Args)] = {
            ExecuteGet<Args>...};
        Value v;
        v.update(type, [&](void *ptr) {
            ptrs[type](ptr, sp);
        });
        return v;
    }
    static void Set(Value const &data, luisa::vector<uint8_t> &arr) {
        Serializer<uint8_t>::Set(data.GetType(), arr);
        funcPtr_t<void(void const *, luisa::vector<uint8_t> &)> ptrs[sizeof...(Args)] = {
            ExecuteSet<Args>...};
        ptrs[data.GetType()](&data, arr);
    }
};

template<typename A, typename B>
struct Serializer<std::pair<A, B>> {
    using Value = std::pair<A, B>;
    static Value Get(std::span<uint8_t const> &sp) {
        return Value{Serializer<A>::Get(sp), Serializer<B>::Get(sp)};
    }
    static void Set(Value const &data, luisa::vector<uint8_t> &arr) {
        Serializer<A>::Set(data.first, arr);
        Serializer<B>::Set(data.second, arr);
    }
};

template<>
struct Serializer<std::span<uint8_t const>> {
    using Value = std::span<uint8_t const>;
    static Value Get(Value &sp) {
        auto sz = Serializer<uint>::Get(sp);
        Value v(sp.data(), sz);
        sp = Value(sp.data() + sz, sp.size() - sz);
        return v;
    }
    static void Set(Value const &data, luisa::vector<uint8_t> &arr) {
        Serializer<uint>::Set(data.size(), arr);
        arr.insert(arr.end(), data.begin(), data.end());
    }
};

template<typename T, size_t sz>
struct Serializer<std::array<T, sz>> {
    using Value = std::array<T, sz>;
    static Value Get(std::span<uint8_t const> &sp) {
        Value v;
        for (auto &&i : v) {
            i = vstd::Serializer<T>::Get(sp);
        }
        return v;
    }
    static void Set(Value const &data, luisa::vector<uint8_t> &arr) {
        for (auto &&i : data) {
            vstd::Serializer<T>::Set(i);
        }
    }
};

template<typename Func>
struct SerializerAll_Impl;

template<typename Ret, typename... Args>
struct SerializerAll_Impl<Ret(Args...)> {

    template<typename Class, typename Func>
    static Ret CallMemberFunc(Class *ptr, Func func, std::span<uint8_t const> data) {
        auto closureFunc = [&](Args &&...args) {
            (ptr->*func)(std::forward<Args>(args)...);
        };
        return std::apply(closureFunc, std::tuple<Args...>{Serializer<std::remove_cvref_t<Args>>::Get(data)...});
    }

    static luisa::vector<uint8_t> Ser(
        Args const &...args) {
        luisa::vector<uint8_t> vec;
        (Serializer<std::remove_cvref_t<Args>>::Set(args, vec), ...);
        return vec;
    }

    template<typename Func>
    static decltype(auto) Call(Func &&func) {
        return [&func](std::span<uint8_t const> data) {
            return std::apply(func, std::tuple<Args...>{Serializer<std::remove_cvref_t<Args>>::Get(data)...});
        };
    }
};

template<typename Func>
using SerializerAll = SerializerAll_Impl<FuncType<std::remove_cvref_t<Func>>>;

template<typename Func>
using SerializerAll_Member = SerializerAll_Impl<typename FunctionTemplateGlobal::memFuncPtr<std::remove_cvref_t<Func>>::Type::FuncType>;

}// namespace vstd