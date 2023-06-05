//
// Created by Mike on 6/4/2023.
//

#pragma once

#include <type_traits>

namespace luisa::compute {

namespace detail {
template<typename T>
struct is_stream_event_impl : std::false_type {};
}// namespace detail

#define LUISA_MARK_STREAM_EVENT_TYPE(T) \
    template<>                          \
    struct luisa::compute::detail::is_stream_event_impl<T &&> : std::true_type {};

template<typename T>
constexpr auto is_stream_event_v = detail::is_stream_event_impl<T>::value;

}// namespace luisa::compute
