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
#define LUISA_REVERSE_NEXT1(test, next)    \
    LUISA_MACRO_DEFER(LUISA_REVERSE_NEXT0) \
    (test, next, 0)
#define LUISA_REVERSE_NEXT(test, next) LUISA_REVERSE_NEXT1(LUISA_REVERSE_GET_END test, next)

#define LUISA_REVERSE0(x, peek, ...)                            \
    LUISA_MACRO_DEFER(LUISA_REVERSE_NEXT(peek, LUISA_REVERSE1)) \
    (peek, __VA_ARGS__) x,
#define LUISA_REVERSE1(x, peek, ...)                            \
    LUISA_MACRO_DEFER(LUISA_REVERSE_NEXT(peek, LUISA_REVERSE0)) \
    (peek, __VA_ARGS__) x,
#define LUISA_REVERSE2(x, peek, ...)                            \
    LUISA_MACRO_DEFER(LUISA_REVERSE_NEXT(peek, LUISA_REVERSE1)) \
    (peek, __VA_ARGS__) x
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

// other useful list operations...
#define LUISA_TAIL_IMPL(x, ...) __VA_ARGS__
#define LUISA_HEAD_IMPL(x, ...) x
#define LUISA_HEAD(...) LUISA_HEAD_IMPL(__VA_ARGS__)
#define LUISA_TAIL(...) LUISA_TAIL_IMPL(__VA_ARGS__)
#define LUISA_LAST(...) LUISA_HEAD(LUISA_REVERSE(__VA_ARGS__))
#define LUISA_POP_LAST(...) LUISA_REVERSE(LUISA_TAIL(LUISA_REVERSE(__VA_ARGS__)))

// inc & dec
#define LUISA_INC_0() 1
#define LUISA_INC_1() 2
#define LUISA_INC_2() 3
#define LUISA_INC_3() 4
#define LUISA_INC_4() 5
#define LUISA_INC_5() 6
#define LUISA_INC_6() 7
#define LUISA_INC_7() 8
#define LUISA_INC_8() 9
#define LUISA_INC_9() 10
#define LUISA_INC_10() 11
#define LUISA_INC_11() 12
#define LUISA_INC_12() 13
#define LUISA_INC_13() 14
#define LUISA_INC_14() 15
#define LUISA_INC_15() 16
#define LUISA_INC_16() 17
#define LUISA_INC_17() 18
#define LUISA_INC_18() 19
#define LUISA_INC_19() 20
#define LUISA_INC_20() 21
#define LUISA_INC_21() 22
#define LUISA_INC_22() 23
#define LUISA_INC_23() 24
#define LUISA_INC_24() 25
#define LUISA_INC_25() 26
#define LUISA_INC_26() 27
#define LUISA_INC_27() 28
#define LUISA_INC_28() 29
#define LUISA_INC_29() 30
#define LUISA_INC_30() 31
#define LUISA_INC_31() 32
#define LUISA_INC_32() 33
#define LUISA_INC_33() 34
#define LUISA_INC_34() 35
#define LUISA_INC_35() 36
#define LUISA_INC_36() 37
#define LUISA_INC_37() 38
#define LUISA_INC_38() 39
#define LUISA_INC_39() 40
#define LUISA_INC_40() 41
#define LUISA_INC_41() 42
#define LUISA_INC_42() 43
#define LUISA_INC_43() 44
#define LUISA_INC_44() 45
#define LUISA_INC_45() 46
#define LUISA_INC_46() 47
#define LUISA_INC_47() 48
#define LUISA_INC_48() 49
#define LUISA_INC_49() 50
#define LUISA_INC_50() 51
#define LUISA_INC_51() 52
#define LUISA_INC_52() 53
#define LUISA_INC_53() 54
#define LUISA_INC_54() 55
#define LUISA_INC_55() 56
#define LUISA_INC_56() 57
#define LUISA_INC_57() 58
#define LUISA_INC_58() 59
#define LUISA_INC_59() 60
#define LUISA_INC_60() 61
#define LUISA_INC_61() 62
#define LUISA_INC_62() 63
#define LUISA_INC_63() 64
#define LUISA_INC_IMPL(x) LUISA_INC_##x()
#define LUISA_INC(x) LUISA_INC_IMPL(x)

#define LUISA_DEC_1() 0
#define LUISA_DEC_2() 1
#define LUISA_DEC_3() 2
#define LUISA_DEC_4() 3
#define LUISA_DEC_5() 4
#define LUISA_DEC_6() 5
#define LUISA_DEC_7() 6
#define LUISA_DEC_8() 7
#define LUISA_DEC_9() 8
#define LUISA_DEC_10() 9
#define LUISA_DEC_11() 10
#define LUISA_DEC_12() 11
#define LUISA_DEC_13() 12
#define LUISA_DEC_14() 13
#define LUISA_DEC_15() 14
#define LUISA_DEC_16() 15
#define LUISA_DEC_17() 16
#define LUISA_DEC_18() 17
#define LUISA_DEC_19() 18
#define LUISA_DEC_20() 19
#define LUISA_DEC_21() 20
#define LUISA_DEC_22() 21
#define LUISA_DEC_23() 22
#define LUISA_DEC_24() 23
#define LUISA_DEC_25() 24
#define LUISA_DEC_26() 25
#define LUISA_DEC_27() 26
#define LUISA_DEC_28() 27
#define LUISA_DEC_29() 28
#define LUISA_DEC_30() 29
#define LUISA_DEC_31() 30
#define LUISA_DEC_32() 31
#define LUISA_DEC_33() 32
#define LUISA_DEC_34() 33
#define LUISA_DEC_35() 34
#define LUISA_DEC_36() 35
#define LUISA_DEC_37() 36
#define LUISA_DEC_38() 37
#define LUISA_DEC_39() 38
#define LUISA_DEC_40() 39
#define LUISA_DEC_41() 40
#define LUISA_DEC_42() 41
#define LUISA_DEC_43() 42
#define LUISA_DEC_44() 43
#define LUISA_DEC_45() 44
#define LUISA_DEC_46() 45
#define LUISA_DEC_47() 46
#define LUISA_DEC_48() 47
#define LUISA_DEC_49() 48
#define LUISA_DEC_50() 49
#define LUISA_DEC_51() 50
#define LUISA_DEC_52() 51
#define LUISA_DEC_53() 52
#define LUISA_DEC_54() 53
#define LUISA_DEC_55() 54
#define LUISA_DEC_56() 55
#define LUISA_DEC_57() 56
#define LUISA_DEC_58() 57
#define LUISA_DEC_59() 58
#define LUISA_DEC_60() 59
#define LUISA_DEC_61() 60
#define LUISA_DEC_62() 61
#define LUISA_DEC_63() 62
#define LUISA_DEC_64() 63
#define LUISA_DEC_IMPL(x) LUISA_DEC_##x()
#define LUISA_DEC(x) LUISA_DEC_IMPL(x)

#define LUISA_RANGE_GEN_1() 0
#define LUISA_RANGE_GEN_2() 0, 1
#define LUISA_RANGE_GEN_3() 0, 1, 2
#define LUISA_RANGE_GEN_4() 0, 1, 2, 3
#define LUISA_RANGE_GEN_5() 0, 1, 2, 3, 4
#define LUISA_RANGE_GEN_6() 0, 1, 2, 3, 4, 5
#define LUISA_RANGE_GEN_7() 0, 1, 2, 3, 4, 5, 6
#define LUISA_RANGE_GEN_8() 0, 1, 2, 3, 4, 5, 6, 7
#define LUISA_RANGE_GEN_9() 0, 1, 2, 3, 4, 5, 6, 7, 8
#define LUISA_RANGE_GEN_10() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
#define LUISA_RANGE_GEN_11() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
#define LUISA_RANGE_GEN_12() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
#define LUISA_RANGE_GEN_13() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
#define LUISA_RANGE_GEN_14() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
#define LUISA_RANGE_GEN_15() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
#define LUISA_RANGE_GEN_16() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
#define LUISA_RANGE_GEN_17() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
#define LUISA_RANGE_GEN_18() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
#define LUISA_RANGE_GEN_19() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
#define LUISA_RANGE_GEN_20() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
#define LUISA_RANGE_GEN_21() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
#define LUISA_RANGE_GEN_22() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21
#define LUISA_RANGE_GEN_23() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
#define LUISA_RANGE_GEN_24() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
#define LUISA_RANGE_GEN_25() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
#define LUISA_RANGE_GEN_26() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
#define LUISA_RANGE_GEN_27() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26
#define LUISA_RANGE_GEN_28() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27
#define LUISA_RANGE_GEN_29() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28
#define LUISA_RANGE_GEN_30() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29
#define LUISA_RANGE_GEN_31() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30
#define LUISA_RANGE_GEN_32() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
#define LUISA_RANGE_GEN_33() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
#define LUISA_RANGE_GEN_34() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33
#define LUISA_RANGE_GEN_35() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34
#define LUISA_RANGE_GEN_36() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
#define LUISA_RANGE_GEN_37() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36
#define LUISA_RANGE_GEN_38() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37
#define LUISA_RANGE_GEN_39() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38
#define LUISA_RANGE_GEN_40() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39
#define LUISA_RANGE_GEN_41() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40
#define LUISA_RANGE_GEN_42() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41
#define LUISA_RANGE_GEN_43() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42
#define LUISA_RANGE_GEN_44() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43
#define LUISA_RANGE_GEN_45() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44
#define LUISA_RANGE_GEN_46() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45
#define LUISA_RANGE_GEN_47() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46
#define LUISA_RANGE_GEN_48() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47
#define LUISA_RANGE_GEN_49() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48
#define LUISA_RANGE_GEN_50() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49
#define LUISA_RANGE_GEN_51() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50
#define LUISA_RANGE_GEN_52() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51
#define LUISA_RANGE_GEN_53() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52
#define LUISA_RANGE_GEN_54() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53
#define LUISA_RANGE_GEN_55() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54
#define LUISA_RANGE_GEN_56() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55
#define LUISA_RANGE_GEN_57() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56
#define LUISA_RANGE_GEN_58() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57
#define LUISA_RANGE_GEN_59() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58
#define LUISA_RANGE_GEN_60() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59
#define LUISA_RANGE_GEN_61() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60
#define LUISA_RANGE_GEN_62() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61
#define LUISA_RANGE_GEN_63() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62
#define LUISA_RANGE_GEN_64() 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
#define LUISA_RANGE_GEN(N) LUISA_RANGE_GEN_##N()
#define LUISA_RANGE(N) LUISA_RANGE_GEN(N)

#define LUISA_STRINGIFY_IMPL(x) #x
#define LUISA_STRINGIFY(x) LUISA_STRINGIFY_IMPL(x)
