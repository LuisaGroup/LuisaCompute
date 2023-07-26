#pragma once

namespace luisa::compute::cuda::dcub {

template<typename _Key, typename _Value>
struct KeyValuePair {
    typedef _Key Key;    ///< Key data type
    typedef _Value Value;///< Value data type

    Key key;    ///< Item key
    Value value;///< Item value

    /// Constructor
    KeyValuePair() {}

    /// Constructor
    KeyValuePair(Key const &key, Value const &value) : key(key), value(value) {}
};

struct Equality {};

struct Max {};

struct Min {};

enum class BinaryOperator {
    Max,
    Min,
};
}// namespace luisa::compute::cuda::dcub