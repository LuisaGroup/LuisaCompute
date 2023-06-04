//
// Created by Mike on 6/4/2023.
//

#pragma once

#include <type_traits>

namespace luisa::compute {

namespace detail {
template<typename T>
struct is_stream_event_impl : public std::false_type {};
}// namespace detail

// FIXME: GCC seems to have parsing issues if we derive the
//  class directly from std::true_type. So here we explicitly
//  define a constexpr value member to work around.
#define LUISA_MARK_STREAM_EVENT_TYPE(T)                        \
    template<>                                                 \
    struct ::luisa::compute::detail::is_stream_event_impl<T> { \
        static constexpr auto value = true;                    \
    };

template<typename T>
using is_stream_event = detail::is_stream_event_impl<std::remove_cvref_t<T>>;

template<typename T>
constexpr auto is_stream_event_v = is_stream_event<T>::value;

}// namespace luisa::compute
