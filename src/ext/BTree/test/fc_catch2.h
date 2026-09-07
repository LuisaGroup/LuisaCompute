#ifndef FC_CATCH2_H
#define FC_CATCH2_H

#ifndef CATCH_CONFIG_ENABLE_BENCHMARKING
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#endif // CATCH_CONFIG_ENABLE_BENCHMARKING

#ifdef CATCH2_OLD
#include <catch2/catch.hpp>
#else
#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#endif

#endif // FC_CATCH2_H
