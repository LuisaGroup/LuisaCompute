#pragma once
#include <vstl/common.h>
#include <vstl/functional.h>
#include <vstl/md5.h>
#include <vstl/v_guid.h>
#include <core/stl/vector.h>
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
    static T Get(vstd::span<std::byte const> &data) {
        using Type = std::remove_cvref_t<T>;
        if (reverseBytes) {
            Type ptr = *reinterpret_cast<Type const *>(data.data());
            ReverseBytes(ptr);
            data = vstd::span<std::byte const>(data.data() + sizeof(Type), data.size() - sizeof(Type));
            return ptr;
        } else {
            Type const *ptr = reinterpret_cast<Type const *>(data.data());
            data = vstd::span<std::byte const>(data.data() + sizeof(Type), data.size() - sizeof(Type));
            return *ptr;
        }
    }
    static void Set(T const &data, vector<std::byte> &vec) {
        using Type = std::remove_cvref_t<T>;
        if (reverseBytes) {
            Type tempData = data;
            ReverseBytes(tempData);
            auto beg = reinterpret_cast<std::byte const *>(&tempData);
            vec.insert(vec.end(), beg, beg + sizeof(Type));
        } else {
            auto beg = reinterpret_cast<std::byte const *>(&data);
            vec.insert(vec.end(), beg, beg + sizeof(Type));
        }
    }
};
template<bool reverseBytes>
struct SerDe<vstd::string, reverseBytes> {
    static vstd::string Get(vstd::span<std::byte const> &sp) {
        auto strLen = SerDe<uint, reverseBytes>::Get(sp);
        auto ptr = sp.data();
        sp = vstd::span<std::byte const>(ptr + strLen, sp.size() - strLen);
        return vstd::string(reinterpret_cast<char const *>(ptr), strLen);
    }
    static void Set(vstd::string const &data, vector<std::byte> &arr) {
        SerDe<uint, reverseBytes>::Set(data.size(), arr);
        auto beg = reinterpret_cast<std::byte const *>(data.data());
        arr.insert(arr.end(), beg, beg + data.size());
    }
};
template<bool reverseBytes>
struct SerDe<std::string_view, reverseBytes> {
    static std::string_view Get(vstd::span<std::byte const> &sp) {
        auto strLen = SerDe<uint, reverseBytes>::Get(sp);
        auto ptr = sp.data();
        sp = vstd::span<std::byte const>(ptr + strLen, sp.size() - strLen);
        return std::string_view(
            reinterpret_cast<char const *>(ptr),
            strLen);
    }
    static void Set(std::string_view const &data, vector<std::byte> &arr) {
        SerDe<uint, reverseBytes>::Set(data.size(), arr);
        auto beg = reinterpret_cast<std::byte const *>(data.data());
        arr.insert(arr.end(), beg, beg + data.size());
    }
};

template<typename T, bool reverseBytes>
struct SerDe<vector<T>, reverseBytes> {
    using Value = vector<T>;
    static Value Get(vstd::span<T const> &sp) {
        Value sz;
        auto s = SerDe<uint, reverseBytes>::Get(sp);
        push_back_func(
            sz,
            s,
            [&]() {
                return SerDe<T, reverseBytes>::Get(sp);
            });
        return sz;
    }
    static void Set(Value const &data, vector<std::byte> &arr) {
        SerDe<uint, reverseBytes>::Set(data.size(), arr);
        for(auto&& i : data){
            SerDe<T, reverseBytes>::Set(i, arr);
        }
    }
};
template<typename T, size_t idx, bool reverseBytes>
struct SerDe<fixed_vector<T, idx>, reverseBytes> {
    using Value = fixed_vector<T, idx>;
    static Value Get(vstd::span<T const> &sp) {
        Value sz;
        auto s = SerDe<uint, reverseBytes>::Get(sp);
        push_back_func(
            sz,
            s,
            [&]() {
                return SerDe<T, reverseBytes>::Get(sp);
            });
        return sz;
    }
    static void Set(Value const &data, vector<std::byte> &arr) {
        SerDe<uint, reverseBytes>::Set(data.size(), arr);
        for(auto&& i : data){
            SerDe<T, reverseBytes>::Set(i, arr);
        }
    }
};

template<typename K, typename V, typename Hash, typename Equal, VEngine_AllocType alloc, bool reverseBytes>
struct SerDe<HashMap<K, V, Hash, Equal, alloc>, reverseBytes> {
    using Value = HashMap<K, V, Hash, Equal, alloc>;
    static Value Get(vstd::span<std::byte const> &sp) {
        Value sz;
        auto capa = SerDe<uint, reverseBytes>::Get(sp);
        sz.reserve(capa);
        for (auto &&i : range(capa)) {
            auto key = SerDe<K, reverseBytes>::Get(sp);
            auto value = SerDe<V, reverseBytes>::Get(sp);
            sz.emplace(
                std::move(key),
                std::move(value));
        }
        return sz;
    }
    static void Set(Value const &data, vector<std::byte> &arr) {
        SerDe<uint, reverseBytes>::Set(data.size(), arr);
        for(auto&& i : data){
            SerDe<K, reverseBytes>::Set(i.first, arr);
            SerDe<V, reverseBytes>::Set(i.second, arr);
        }
    }
};
template<typename... Args, bool reverseBytes>
struct SerDe<variant<Args...>, reverseBytes> {
    using Value = variant<Args...>;
    template<typename T>
    static void ExecuteGet(void *placePtr, vstd::span<std::byte const> &sp) {
        new (placePtr) Value(SerDe<T, reverseBytes>::Get(sp));
    }
    template<typename T>
    static void ExecuteSet(void const *placePtr, vector<std::byte> &sp) {
        SerDe<T, reverseBytes>::Set(*reinterpret_cast<T const *>(placePtr), sp);
    }
    static Value Get(vstd::span<std::byte const> &sp) {
        auto type = SerDe<uint8_t, reverseBytes>::Get(sp);
        func_ptr_t<void(void *, vstd::span<std::byte const> &)> ptrs[sizeof...(Args)] = {
            ExecuteGet<Args>...};
        Value v;
        v.reset_as(type);
        ptrs[type](v.place_holder(), sp);
        return v;
    }
    static void Set(Value const &data, vector<std::byte> &arr) {
        SerDe<uint8_t, reverseBytes>::Set(data.GetType(), arr);
        func_ptr_t<void(void const *, vector<std::byte> &)> ptrs[sizeof...(Args)] = {
            ExecuteSet<Args>...};
        ptrs[data.GetType()](&data, arr);
    }
};

template<typename A, typename B, bool reverseBytes>
struct SerDe<std::pair<A, B>, reverseBytes> {
    using Value = std::pair<A, B>;
    static Value Get(vstd::span<std::byte const> &sp) {
        return Value{SerDe<A, reverseBytes>::Get(sp), SerDe<B, reverseBytes>::Get(sp)};
    }
    static void Set(Value const &data, vector<std::byte> &arr) {
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
    static void Set(Value const &data, vector<std::byte> &arr) {
        SerDe<uint, reverseBytes>::Set(data.size(), arr);
        auto beg = reinterpret_cast<std::byte const *>(data.data());
        arr.insert(arr.end(), beg, beg + data.size());
    }
};
template<bool reverseBytes>
struct SerDe<vstd::span<std::byte const>, reverseBytes> {
    using Value = vstd::span<std::byte const>;
    static Value Get(Value &sp) {
        auto sz = SerDe<uint, reverseBytes>::Get(sp);
        Value v(sp.data(), sz);
        sp = Value(sp.data() + sz, sp.size() - sz);
        return v;
    }
    static void Set(Value const &data, vector<std::byte> &arr) {
        SerDe<uint, reverseBytes>::Set(data.size(), arr);
        arr.insert(arr.end(), arr.begin(), arr.end());
    }
};
template<typename T, size_t sz, bool reverseBytes>
struct SerDe<std::array<T, sz>, reverseBytes> {
    using Value = std::array<T, sz>;
    static Value Get(vstd::span<std::byte const> &sp) {
        Value v;
        for (auto &&i : v) {
            i = SerDe<T, reverseBytes>::Get(sp);
        }
        return v;
    }
    static void Set(Value const &data, vector<std::byte> &arr) {
        for(auto&& i : data){
            SerDe<T, reverseBytes>::Set(i);
        }
    }
};

template<typename Func>
struct SerDeAll_Impl;

template<typename Ret, typename... Args>
struct SerDeAll_Impl<Ret(Args...)> {
    template<typename Class, typename Func>
    static Ret CallMemberFunc(Class *ptr, Func func, vstd::span<std::byte const> data) {
        auto closureFunc = [&](Args &&...args) {
            return (ptr->*func)(std::forward<Args>(args)...);
        };
        return std::apply(closureFunc, std::tuple<Args...>{SerDe<std::remove_cvref_t<Args>>::Get(data)...});
    }

    static vector<std::byte> Ser(
        Args const &...args) {
        vector<std::byte> vec;
        auto lst = {(SerDe<std::remove_cvref_t<Args>>::Set(args, vec), ' ')...};
        return vec;
    }

    template<typename Func>
    static decltype(auto) Call(
        Func &&func) {
        return [f = std::forward<Func>(func)](vstd::span<std::byte const> data) {
            return std::apply(f, std::tuple<Args...>{SerDe<std::remove_cvref_t<Args>>::Get(data)...});
        };
    }
};
template<>
struct SerDe<Guid::GuidData, true> {
    using Value = Guid::GuidData;
    static Value Get(span<std::byte const> &sp) {
        return Value{
            SerDe<uint64, true>::Get(sp),
            SerDe<uint64, true>::Get(sp)};
    }
    static void Set(Value const &data, vector<std::byte> &arr) {
        SerDe<uint64, true>::Set(data.data0, arr);
        SerDe<uint64, true>::Set(data.data1, arr);
    }
};

template<bool reverseBytes>
struct SerDe<Guid, reverseBytes> {
    using Value = Guid;
    static Value Get(span<std::byte const> &sp) {
        return SerDe<Guid::GuidData, reverseBytes>::Get(sp);
    }
    static void Set(Value const &data, vector<std::byte> &arr) {
        SerDe<Guid::GuidData, reverseBytes>::Set(data.ToBinary(), arr);
    }
};
template<>
struct SerDe<MD5> {
    static MD5 Get(eastl::span<std::byte const> &sp) {
        MD5::MD5Data data;
        data.data0 = SerDe<uint64>::Get(sp);
        data.data1 = SerDe<uint64>::Get(sp);
        return MD5(data);
    }
    static void Set(MD5 const &data, vector<std::byte> &arr) {
        auto &&dd = data.ToBinary();
        SerDe<uint64>::Set(dd.data0, arr);
        SerDe<uint64>::Set(dd.data1, arr);
    }
};
template<>
struct compare<MD5> {
    int32 operator()(MD5 const &a, MD5 const &b) const {
        if (a.ToBinary().data0 > b.ToBinary().data0) return 1;
        if (a.ToBinary().data0 < b.ToBinary().data0) return -1;
        if (a.ToBinary().data1 > b.ToBinary().data1) return 1;
        if (a.ToBinary().data1 < b.ToBinary().data1) return -1;
        return 0;
    }
};
}// namespace vstd