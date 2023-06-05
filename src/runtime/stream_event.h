//
// Created by Mike on 6/4/2023.
//

#pragma once

#include <type_traits>

namespace luisa::compute {
template<typename T>
struct is_stream_event : std::false_type {};

#define LUISA_MARK_STREAM_EVENT_TYPE(T) \
    template<>                          \
    struct luisa::compute::is_stream_event<T> : std::true_type {};

template<typename T>
constexpr auto is_stream_event_v = is_stream_event<T>::value;

}// namespace luisa::compute
