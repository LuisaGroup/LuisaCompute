#pragma once

#include <serialize/traits.h>
#include <serialize/key_value_pair.h>
#include <nlohmann/json.hpp>
#include <string_view>
#include <core/stl.h>
#include <iostream>
#include <utility>
#include <core/logging.h>

namespace luisa::compute{

template<typename T>
struct VariantPack {
    size_t index;
    T& value;
    VariantPack(size_t index, T& value) noexcept:index(index), value(value) {}
    template<typename S>
    void serialize(S& s) noexcept {
        s.serialize(MAKE_NAME_PAIR(index), KeyValuePair<decltype("value"), T>{"value", value});
    }
};

class Serializer{
private:
    nlohmann::json _data;
    luisa::vector<nlohmann::json*> _currentJson;

    auto& currentJson() const noexcept { return *_currentJson.back(); }

    template<typename K>
    auto enterScope(K key) noexcept {
        _currentJson.push_back(&currentJson()[key]);
    }

    void popScope() noexcept {
        _currentJson.pop_back();
    }

public:
    Serializer():_data(nlohmann::json::object()) { _currentJson.push_back(&_data); }

    template<typename K, typename Arg>
        requires has_member_function_save_load<Arg>
    void serialize(const KeyValuePair<K, Arg>& arg) {
        enterScope(arg.key());
        arg.value().save(*this);
        popScope();
    }

    template<typename K, typename Arg>
        requires has_member_function_serialize<Arg>
    void serialize(const KeyValuePair<K, Arg>& arg) {
        enterScope(arg.key());
        arg.value().serialize(*this);
        popScope();
    }

    template<typename K, typename Arg>
        requires can_directly_serialize<Arg>
    void serialize(const KeyValuePair<K, Arg>& arg) {
        currentJson()[arg.key()] = arg.value();
    }

    template<typename K, typename Arg>
    void serialize(const KeyValuePair<K, luisa::vector<Arg>>& arg) {
        enterScope(arg.key());
        for(size_t i = 0; i < arg.value().size(); i++){
            serialize(KeyValuePair{i, arg.value()[i]});
        }
        popScope();
    }

    template<typename K, typename Arg>
    void serialize(const KeyValuePair<K, luisa::span<Arg>>& arg) {
        enterScope(arg.key());
        using ArgNoConst = std::remove_cvref_t<Arg>;
        for(size_t i = 0; i < arg.value().size(); i++){
            ArgNoConst data = arg.value()[i];
            serialize(KeyValuePair{i, data});
        }
        popScope();
    }

    template<typename K, typename Arg>
        requires is_variant_v<std::remove_cvref_t<Arg>>
    void serialize(const KeyValuePair<K, Arg> &arg) {
        auto index = arg.value().index();
        luisa::visit([index, &arg, this]<typename T>(T &t) noexcept {
            auto pack = VariantPack<T>{index, t};
            serialize(KeyValuePair{arg.key(), pack});
        },
        arg.value());
    }

    template<typename K>
    void serialize(const KeyValuePair<K, luisa::monostate> &arg) {}

    template<typename K, typename Arg>
    void serialize(const KeyValuePair<K, luisa::unique_ptr<Arg>>& arg) {
        if(arg.value() == nullptr) {
            enterScope(arg.key());
            popScope();
            return;
        }
        if constexpr (enable_polymorphic_serialization<Arg>){
            static_assert(has_serialization_function<Arg>);
            enterScope(arg.key());
            if constexpr (has_member_function_save_load<Arg>)
                arg.value()->save(*this);
            else if constexpr (has_member_function_serialize<Arg>)
                arg.value()->serialize(*this);
            popScope();
        } else
            serialize(KeyValuePair{arg.key(), *(arg.value().get())});
    }

    template<typename K, typename Arg>
    void serialize(const KeyValuePair<K, luisa::shared_ptr<Arg>>& arg) {
        if(arg.value() == nullptr) {
            enterScope(arg.key());
            popScope();
            return;
        }
        serialize(KeyValuePair{arg.key(), *(arg.value().get())});
    }

    template<typename Arg, typename... Others>
        requires (sizeof...(Others) > 0)
    void serialize(const Arg& arg, const Others&... others) {
        serialize(arg);
        serialize(others...);
    }

    auto data() const noexcept {
        return _data;
    }
};

class Deserializer{
private:
    nlohmann::json _data;
    luisa::vector<nlohmann::json*> _currentJson;

    auto& currentJson() const noexcept { return *_currentJson.back(); }

    template<typename K>
    auto enterScope(K key) noexcept {
        _currentJson.push_back(&currentJson()[key]);
    }

    void popScope() noexcept {
        _currentJson.pop_back();
    }

    template<size_t I, typename K, typename Arg>
        requires is_variant_v<Arg>
    struct serializeVariant {
        void operator()(size_t index, const KeyValuePair<K, Arg>& arg, Deserializer* d) {
            if (I == index) {
                using T = luisa::variant_alternative_t<I, Arg>;
                T value;
                VariantPack<T> pack(0, value);
                d->serialize(KeyValuePair{arg.key(), pack});
                arg.value() = pack.value;
            }
            else serializeVariant<I - 1, K, Arg>()(index, arg, d);
        }
    };

    template<typename K, typename Arg>
        requires is_variant_v<Arg>
    struct serializeVariant<0, K, Arg> {
        void operator()(size_t index, const KeyValuePair<K, Arg>& arg, Deserializer* d) {
            if (index == 0) {
                using T = luisa::variant_alternative_t<0, Arg>;
                T value;
                VariantPack<T> pack(0, value);
                d->serialize(KeyValuePair{arg.key(), pack});
                arg.value() = pack.value;
            }
        }
    };

public:
    Deserializer(nlohmann::json data):_data(data) { _currentJson.push_back(&_data); }

    template<typename K, typename Arg>
        requires has_member_function_save_load<Arg>
    void serialize(const KeyValuePair<K, Arg>& arg) {
        enterScope(arg.key());
        arg.value().load(*this);
        popScope();
    }

    template<typename K, typename Arg>
        requires has_member_function_serialize<Arg>
    void serialize(const KeyValuePair<K, Arg>& arg) {
        enterScope(arg.key());
        arg.value().serialize(*this);
        popScope();
    }

    template<typename K, typename Arg>
        requires can_directly_serialize<Arg>
    void serialize(const KeyValuePair<K, Arg>& arg) {
        arg.value() = currentJson()[arg.key()];
    }

    template<typename K, typename Arg>
    void serialize(const KeyValuePair<K, luisa::vector<Arg>>& arg) {
        enterScope(arg.key());
        auto size = currentJson().size();
        arg.value().resize(size);
        for(auto i = 0; i < size; i++){
            serialize(KeyValuePair{i, arg.value()[i]});
        }
        popScope();
    }

    template<typename K, typename Arg>
    void serialize(const KeyValuePair<K, luisa::span<Arg>>& arg) {
        enterScope(arg.key());
        auto size = currentJson().size();
        luisa::vector<std::remove_cvref_t<Arg>> data(size);
        for(auto i = 0; i < size; i++){
            serialize(KeyValuePair{i, data[i]});
        }
        arg.value() = luisa::span<Arg>(data.begin(), data.end());
        popScope();
    }

    template<typename K, typename Arg>
        requires is_variant_v<std::remove_cvref_t<Arg>>
    void serialize(const KeyValuePair<K, Arg> &arg) {
        auto index = currentJson()[arg.key()]["index"];
        serializeVariant<luisa::variant_size_v<Arg> - 1, K, Arg>()(index, arg, this);
    }

    template<typename K>
    void serialize(const KeyValuePair<K, luisa::monostate> &arg) {}

    template<typename K, typename Arg>
    void serialize(const KeyValuePair<K, luisa::unique_ptr<Arg>>& arg) {
        if(currentJson()[arg.key()] == nullptr) return;
        if constexpr (enable_polymorphic_serialization<Arg>){
            enterScope(arg.key());
            static_assert(has_serialization_function<Arg>);
            arg.value() = std::move(Arg::create(currentJson()["tag"]));
            if constexpr (has_member_function_save_load<Arg>)
                arg.value()->load(*this);
            else if constexpr (has_member_function_serialize<Arg>)
                arg.value()->serialize(*this);
            popScope();
        } else {
            if(arg.value() == nullptr) arg.value() = std::move(luisa::make_unique<Arg>());
            serialize(KeyValuePair{arg.key(), *(arg.value().get())});
        }
    }

    template<typename K, typename Arg>
    void serialize(const KeyValuePair<K, luisa::shared_ptr<Arg>>& arg) {
        if(currentJson()[arg.key()] == nullptr) return;
        if(arg.value() == nullptr) arg.value() = luisa::make_shared<Arg>();
        serialize(KeyValuePair{arg.key(), *(arg.value().get())});
    }

    template<typename Arg, typename... Others>
        requires (sizeof...(Others) > 0)
    void serialize(const Arg& arg, const Others&... others) {
        serialize(arg);
        serialize(others...);
    }

};

}