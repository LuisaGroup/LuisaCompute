//
// Created by Mike Smith on 2020/8/8.
//

#pragma once

// From: https://github.com/Erlkoenig90/map-macro
#define LUISA_MACRO_EVAL0(...) __VA_ARGS__
#define LUISA_MACRO_EVAL1(...) LUISA_MACRO_EVAL0(LUISA_MACRO_EVAL0(LUISA_MACRO_EVAL0(__VA_ARGS__)))
#define LUISA_MACRO_EVAL2(...) LUISA_MACRO_EVAL1(LUISA_MACRO_EVAL1(LUISA_MACRO_EVAL1(__VA_ARGS__)))
#define LUISA_MACRO_EVAL3(...) LUISA_MACRO_EVAL2(LUISA_MACRO_EVAL2(LUISA_MACRO_EVAL2(__VA_ARGS__)))
#define LUISA_MACRO_EVAL4(...) LUISA_MACRO_EVAL3(LUISA_MACRO_EVAL3(LUISA_MACRO_EVAL3(__VA_ARGS__)))
#define LUISA_MACRO_EVAL5(...) LUISA_MACRO_EVAL4(LUISA_MACRO_EVAL4(LUISA_MACRO_EVAL4(__VA_ARGS__)))
#define LUISA_MACRO_EVAL(...) LUISA_MACRO_EVAL5(__VA_ARGS__)

#define LUISA_MACRO_EMPTY()
#define LUISA_MACRO_DEFER(id) id LUISA_MACRO_EMPTY()

// macro reverse
#define LUISA_REVERSE_END(...)
#define LUISA_REVERSE_OUT

#define LUISA_REVERSE_GET_END2() 0, LUISA_REVERSE_END
#define LUISA_REVERSE_GET_END1(...) LUISA_REVERSE_GET_END2
#define LUISA_REVERSE_GET_END(...) LUISA_REVERSE_GET_END1
#define LUISA_REVERSE_NEXT0(test, next, ...) next LUISA_REVERSE_OUT
#define LUISA_REVERSE_NEXT1(test, next) LUISA_MACRO_DEFER(LUISA_REVERSE_NEXT0)(test, next, 0)
#define LUISA_REVERSE_NEXT(test, next) LUISA_REVERSE_NEXT1(LUISA_REVERSE_GET_END test, next)

#define LUISA_REVERSE0(x, peek, ...) LUISA_MACRO_DEFER(LUISA_REVERSE_NEXT(peek, LUISA_REVERSE1))(peek, __VA_ARGS__) x,
#define LUISA_REVERSE1(x, peek, ...) LUISA_MACRO_DEFER(LUISA_REVERSE_NEXT(peek, LUISA_REVERSE0))(peek, __VA_ARGS__) x,
#define LUISA_REVERSE2(x, peek, ...) LUISA_MACRO_DEFER(LUISA_REVERSE_NEXT(peek, LUISA_REVERSE1))(peek, __VA_ARGS__) x
#define LUISA_REVERSE(...) LUISA_MACRO_EVAL(LUISA_REVERSE2(__VA_ARGS__, ()()(), ()()(), ()()(), 0))

// macro map
#define LUISA_MAP_END(...)
#define LUISA_MAP_OUT

#define LUISA_MAP_GET_END2() 0, LUISA_MAP_END
#define LUISA_MAP_GET_END1(...) LUISA_MAP_GET_END2
#define LUISA_MAP_GET_END(...) LUISA_MAP_GET_END1
#define LUISA_MAP_NEXT0(test, next, ...) next LUISA_MAP_OUT
#define LUISA_MAP_NEXT1(test, next)    \
    LUISA_MACRO_DEFER(LUISA_MAP_NEXT0) \
    (test, next, 0)
#define LUISA_MAP_NEXT(test, next) LUISA_MAP_NEXT1(LUISA_MAP_GET_END test, next)

#define LUISA_MAP0(f, x, peek, ...) f(x) LUISA_MACRO_DEFER(LUISA_MAP_NEXT(peek, LUISA_MAP1))(f, peek, __VA_ARGS__)
#define LUISA_MAP1(f, x, peek, ...) f(x) LUISA_MACRO_DEFER(LUISA_MAP_NEXT(peek, LUISA_MAP0))(f, peek, __VA_ARGS__)

#define LUISA_MAP_LIST0(f, x, peek, ...) , f(x) LUISA_MACRO_DEFER(LUISA_MAP_NEXT(peek, LUISA_MAP_LIST1))(f, peek, __VA_ARGS__)
#define LUISA_MAP_LIST1(f, x, peek, ...) , f(x) LUISA_MACRO_DEFER(LUISA_MAP_NEXT(peek, LUISA_MAP_LIST0))(f, peek, __VA_ARGS__)
#define LUISA_MAP_LIST2(f, x, peek, ...) f(x) LUISA_MACRO_DEFER(LUISA_MAP_NEXT(peek, LUISA_MAP_LIST1))(f, peek, __VA_ARGS__)

// Applies the function macro `f` to each of the remaining parameters.
#define LUISA_MAP(f, ...) LUISA_MACRO_EVAL(LUISA_MAP1(f, __VA_ARGS__, ()()(), ()()(), ()()(), 0))

// Applies the function macro `f` to each of the remaining parameters and inserts commas between the results.
#define LUISA_MAP_LIST(f, ...) LUISA_MACRO_EVAL(LUISA_MAP_LIST2(f, __VA_ARGS__, ()()(), ()()(), ()()(), 0))

#define LUISA_TAIL_IMPL(x, ...) __VA_ARGS__
#define LUISA_FIRST_IMPL(x, ...) x
#define LUISA_FIRST(...) LUISA_MACRO_EVAL(LUISA_MACRO_DEFER(LUISA_FIRST_IMPL)(__VA_ARGS__))
#define LUISA_TAIL(...) LUISA_MACRO_EVAL(LUISA_MACRO_DEFER(LUISA_TAIL_IMPL)(__VA_ARGS__))
#define LUISA_LAST(...) LUISA_FIRST(LUISA_REVERSE(__VA_ARGS__))
#define LUISA_POP_LAST(...) LUISA_REVERSE(LUISA_TAIL(LUISA_REVERSE(__VA_ARGS__)))
