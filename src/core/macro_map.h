//
// Created by Mike Smith on 2020/8/8.
//

#pragma once

// From: https://github.com/Erlkoenig90/map-macro
#define LUISA_MAP_EVAL0(...) __VA_ARGS__
#define LUISA_MAP_EVAL1(...) LUISA_MAP_EVAL0(LUISA_MAP_EVAL0(LUISA_MAP_EVAL0(__VA_ARGS__)))
#define LUISA_MAP_EVAL2(...) LUISA_MAP_EVAL1(LUISA_MAP_EVAL1(LUISA_MAP_EVAL1(__VA_ARGS__)))
#define LUISA_MAP_EVAL3(...) LUISA_MAP_EVAL2(LUISA_MAP_EVAL2(LUISA_MAP_EVAL2(__VA_ARGS__)))
#define LUISA_MAP_EVAL4(...) LUISA_MAP_EVAL3(LUISA_MAP_EVAL3(LUISA_MAP_EVAL3(__VA_ARGS__)))
#define LUISA_MAP_EVAL5(...) LUISA_MAP_EVAL4(LUISA_MAP_EVAL4(LUISA_MAP_EVAL4(__VA_ARGS__)))

#ifdef _MSC_VER
// MSVC needs more evaluations
#define LUISA_MAP_EVAL6(...) LUISA_MAP_EVAL5(LUISA_MAP_EVAL5(LUISA_MAP_EVAL5(__VA_ARGS__)))
#define LUISA_MAP_EVAL(...)  LUISA_MAP_EVAL6(LUISA_MAP_EVAL6(__VA_ARGS__))
#else
#define LUISA_MAP_EVAL(...)  LUISA_MAP_EVAL5(__VA_ARGS__)
#endif

#define LUISA_MAP_END(...)
#define LUISA_MAP_OUT

#define LUISA_MAP_EMPTY()
#define LUISA_MAP_DEFER(id) id LUISA_MAP_EMPTY()

#define LUISA_MAP_GET_END2() 0, LUISA_MAP_END
#define LUISA_MAP_GET_END1(...) LUISA_MAP_GET_END2
#define LUISA_MAP_GET_END(...) LUISA_MAP_GET_END1
#define LUISA_MAP_NEXT0(test, next, ...) next LUISA_MAP_OUT
#define LUISA_MAP_NEXT1(test, next) LUISA_MAP_DEFER ( LUISA_MAP_NEXT0 ) ( test, next, 0)
#define LUISA_MAP_NEXT(test, next)  LUISA_MAP_NEXT1(LUISA_MAP_GET_END test, next)

#define LUISA_MAP0(f, x, peek, ...) f(x) LUISA_MAP_DEFER ( LUISA_MAP_NEXT(peek, LUISA_MAP1) ) ( f, peek, __VA_ARGS__ )
#define LUISA_MAP1(f, x, peek, ...) f(x) LUISA_MAP_DEFER ( LUISA_MAP_NEXT(peek, LUISA_MAP0) ) ( f, peek, __VA_ARGS__ )

#define LUISA_MAP_LIST0(f, x, peek, ...) , f(x) LUISA_MAP_DEFER ( LUISA_MAP_NEXT(peek, LUISA_MAP_LIST1) ) ( f, peek, __VA_ARGS__ )
#define LUISA_MAP_LIST1(f, x, peek, ...) , f(x) LUISA_MAP_DEFER ( LUISA_MAP_NEXT(peek, LUISA_MAP_LIST0) ) ( f, peek, __VA_ARGS__ )
#define LUISA_MAP_LIST2(f, x, peek, ...)   f(x) LUISA_MAP_DEFER ( LUISA_MAP_NEXT(peek, LUISA_MAP_LIST1) ) ( f, peek, __VA_ARGS__ )

// Applies the function macro `f` to each of the remaining parameters.
#define LUISA_MAP(f, ...) LUISA_MAP_EVAL(LUISA_MAP1(f, __VA_ARGS__, ()()(), ()()(), ()()(), 0))

// Applies the function macro `f` to each of the remaining parameters and inserts commas between the results.
#define LUISA_MAP_LIST(f, ...) LUISA_MAP_EVAL(LUISA_MAP_LIST2(f, __VA_ARGS__, ()()(), ()()(), ()()(), 0))
