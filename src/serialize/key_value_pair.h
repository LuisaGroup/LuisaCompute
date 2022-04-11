#pragma once

namespace luisa::compute{

template<typename K, typename T>
class KeyValuePair{
private:
    K _key;
    T& _value;
public:
    KeyValuePair(K key, T& value) noexcept:_key(key), _value(value){}
    auto key() const noexcept { return _key; }
    auto& value() const noexcept { return _value; }
};

template<typename K, typename T>
KeyValuePair(K, T &) -> KeyValuePair<K, std::remove_reference_t<T>>;

template<typename K, typename T>
KeyValuePair(K, T &&) -> KeyValuePair<K, std::remove_cvref_t<T>>;

#define MAKE_NAME_PAIR(var) (luisa::compute::KeyValuePair<decltype(#var), decltype(var)>{#var, var})

};