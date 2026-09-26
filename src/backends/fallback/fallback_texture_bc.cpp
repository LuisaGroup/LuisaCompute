//
// Created by swfly on 2024/11/16.
//

#include "fallback_texture_bc.h"
#if defined(_WIN32) && defined(_DEBUG)
#define NOMINMAX
#include <Windows.h>
#include <luisa/core/stl/algorithm.h>
#undef Yield
#undef AddPointer
#endif

namespace luisa::compute::fallback::bc {

// Partition, Shape, Pixel (index into 4x4 block)
const uint8_t g_aPartitionTable[3][64][16] =
    {
        {// 1 Region case has no subsets (all 0)
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},

        {
            // BC6HBlock/BC7 Partition Set for 2 Subsets
            {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1},// Shape 0
            {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1},// Shape 1
            {0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1},// Shape 2
            {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1},// Shape 3
            {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1},// Shape 4
            {0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1},// Shape 5
            {0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1},// Shape 6
            {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1},// Shape 7
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1},// Shape 8
            {0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},// Shape 9
            {0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1},// Shape 10
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1},// Shape 11
            {0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},// Shape 12
            {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},// Shape 13
            {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},// Shape 14
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1},// Shape 15
            {0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1},// Shape 16
            {0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},// Shape 17
            {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0},// Shape 18
            {0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0},// Shape 19
            {0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},// Shape 20
            {0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0},// Shape 21
            {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0},// Shape 22
            {0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1},// Shape 23
            {0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0},// Shape 24
            {0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0},// Shape 25
            {0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0},// Shape 26
            {0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0},// Shape 27
            {0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0},// Shape 28
            {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},// Shape 29
            {0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0},// Shape 30
            {0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0},// Shape 31

            // BC7 Partition Set for 2 Subsets (second-half)
            {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},// Shape 32
            {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1},// Shape 33
            {0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0},// Shape 34
            {0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0},// Shape 35
            {0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0},// Shape 36
            {0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0},// Shape 37
            {0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1},// Shape 38
            {0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1},// Shape 39
            {0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0},// Shape 40
            {0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0},// Shape 41
            {0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0},// Shape 42
            {0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0},// Shape 43
            {0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0},// Shape 44
            {0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1},// Shape 45
            {0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1},// Shape 46
            {0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0},// Shape 47
            {0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0},// Shape 48
            {0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0},// Shape 49
            {0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0},// Shape 50
            {0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0},// Shape 51
            {0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1},// Shape 52
            {0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1},// Shape 53
            {0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0},// Shape 54
            {0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0},// Shape 55
            {0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1},// Shape 56
            {0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1},// Shape 57
            {0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1},// Shape 58
            {0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1},// Shape 59
            {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1},// Shape 60
            {0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0},// Shape 61
            {0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0},// Shape 62
            {0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1} // Shape 63
        },

        {
            // BC7 Partition Set for 3 Subsets
            {0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 1, 2, 2, 2, 2},// Shape 0
            {0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1},// Shape 1
            {0, 0, 0, 0, 2, 0, 0, 1, 2, 2, 1, 1, 2, 2, 1, 1},// Shape 2
            {0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 1, 0, 1, 1, 1},// Shape 3
            {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2},// Shape 4
            {0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 2, 2},// Shape 5
            {0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1},// Shape 6
            {0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1},// Shape 7
            {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2},// Shape 8
            {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2},// Shape 9
            {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2},// Shape 10
            {0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2},// Shape 11
            {0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2},// Shape 12
            {0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2},// Shape 13
            {0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2},// Shape 14
            {0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0, 2, 2, 2, 0},// Shape 15
            {0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 2},// Shape 16
            {0, 1, 1, 1, 0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0},// Shape 17
            {0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2},// Shape 18
            {0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1},// Shape 19
            {0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2, 0, 2, 2, 2},// Shape 20
            {0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 1, 2, 2, 2, 1},// Shape 21
            {0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2},// Shape 22
            {0, 0, 0, 0, 1, 1, 0, 0, 2, 2, 1, 0, 2, 2, 1, 0},// Shape 23
            {0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1, 0, 0, 0, 0},// Shape 24
            {0, 0, 1, 2, 0, 0, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2},// Shape 25
            {0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1, 0, 1, 1, 0},// Shape 26
            {0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1},// Shape 27
            {0, 0, 2, 2, 1, 1, 0, 2, 1, 1, 0, 2, 0, 0, 2, 2},// Shape 28
            {0, 1, 1, 0, 0, 1, 1, 0, 2, 0, 0, 2, 2, 2, 2, 2},// Shape 29
            {0, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1},// Shape 30
            {0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 1, 1, 2, 2, 2, 1},// Shape 31
            {0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 2, 2, 2},// Shape 32
            {0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 2, 0, 0, 1, 1},// Shape 33
            {0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 2, 2, 0, 2, 2, 2},// Shape 34
            {0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0},// Shape 35
            {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0},// Shape 36
            {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0},// Shape 37
            {0, 1, 2, 0, 2, 0, 1, 2, 1, 2, 0, 1, 0, 1, 2, 0},// Shape 38
            {0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1},// Shape 39
            {0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1},// Shape 40
            {0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2},// Shape 41
            {0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1},// Shape 42
            {0, 0, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2, 1, 1, 2, 2},// Shape 43
            {0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 1},// Shape 44
            {0, 2, 2, 0, 1, 2, 2, 1, 0, 2, 2, 0, 1, 2, 2, 1},// Shape 45
            {0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 0, 1},// Shape 46
            {0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1},// Shape 47
            {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2},// Shape 48
            {0, 2, 2, 2, 0, 1, 1, 1, 0, 2, 2, 2, 0, 1, 1, 1},// Shape 49
            {0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 2, 1, 1, 1, 2},// Shape 50
            {0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2},// Shape 51
            {0, 2, 2, 2, 0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2},// Shape 52
            {0, 0, 0, 2, 1, 1, 1, 2, 1, 1, 1, 2, 0, 0, 0, 2},// Shape 53
            {0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2},// Shape 54
            {0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2},// Shape 55
            {0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2},// Shape 56
            {0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2},// Shape 57
            {0, 0, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2},// Shape 58
            {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2},// Shape 59
            {0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1},// Shape 60
            {0, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 1, 2, 2, 2},// Shape 61
            {0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},// Shape 62
            {0, 1, 1, 1, 2, 0, 1, 1, 2, 2, 0, 1, 2, 2, 2, 0} // Shape 63
        }};

// Partition, Shape, Fixup
const uint8_t g_aFixUp[3][64][3] =
    {
        {// No fix-ups for 1st subset for BC6HBlock or BC7
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0},
         {0, 0, 0}},

        {// BC6HBlock/BC7 Partition Set Fixups for 2 Subsets
         {0, 15, 0},
         {0, 15, 0},
         {0, 15, 0},
         {0, 15, 0},
         {0, 15, 0},
         {0, 15, 0},
         {0, 15, 0},
         {0, 15, 0},
         {0, 15, 0},
         {0, 15, 0},
         {0, 15, 0},
         {0, 15, 0},
         {0, 15, 0},
         {0, 15, 0},
         {0, 15, 0},
         {0, 15, 0},
         {0, 15, 0},
         {0, 2, 0},
         {0, 8, 0},
         {0, 2, 0},
         {0, 2, 0},
         {0, 8, 0},
         {0, 8, 0},
         {0, 15, 0},
         {0, 2, 0},
         {0, 8, 0},
         {0, 2, 0},
         {0, 2, 0},
         {0, 8, 0},
         {0, 8, 0},
         {0, 2, 0},
         {0, 2, 0},

         // BC7 Partition Set Fixups for 2 Subsets (second-half)
         {0, 15, 0},
         {0, 15, 0},
         {0, 6, 0},
         {0, 8, 0},
         {0, 2, 0},
         {0, 8, 0},
         {0, 15, 0},
         {0, 15, 0},
         {0, 2, 0},
         {0, 8, 0},
         {0, 2, 0},
         {0, 2, 0},
         {0, 2, 0},
         {0, 15, 0},
         {0, 15, 0},
         {0, 6, 0},
         {0, 6, 0},
         {0, 2, 0},
         {0, 6, 0},
         {0, 8, 0},
         {0, 15, 0},
         {0, 15, 0},
         {0, 2, 0},
         {0, 2, 0},
         {0, 15, 0},
         {0, 15, 0},
         {0, 15, 0},
         {0, 15, 0},
         {0, 15, 0},
         {0, 2, 0},
         {0, 2, 0},
         {0, 15, 0}},

        {// BC7 Partition Set Fixups for 3 Subsets
         {0, 3, 15},
         {0, 3, 8},
         {0, 15, 8},
         {0, 15, 3},
         {0, 8, 15},
         {0, 3, 15},
         {0, 15, 3},
         {0, 15, 8},
         {0, 8, 15},
         {0, 8, 15},
         {0, 6, 15},
         {0, 6, 15},
         {0, 6, 15},
         {0, 5, 15},
         {0, 3, 15},
         {0, 3, 8},
         {0, 3, 15},
         {0, 3, 8},
         {0, 8, 15},
         {0, 15, 3},
         {0, 3, 15},
         {0, 3, 8},
         {0, 6, 15},
         {0, 10, 8},
         {0, 5, 3},
         {0, 8, 15},
         {0, 8, 6},
         {0, 6, 10},
         {0, 8, 15},
         {0, 5, 15},
         {0, 15, 10},
         {0, 15, 8},
         {0, 8, 15},
         {0, 15, 3},
         {0, 3, 15},
         {0, 5, 10},
         {0, 6, 10},
         {0, 10, 8},
         {0, 8, 9},
         {0, 15, 10},
         {0, 15, 6},
         {0, 3, 15},
         {0, 15, 8},
         {0, 5, 15},
         {0, 15, 3},
         {0, 15, 6},
         {0, 15, 6},
         {0, 15, 8},
         {0, 3, 15},
         {0, 15, 3},
         {0, 5, 15},
         {0, 5, 15},
         {0, 5, 15},
         {0, 8, 15},
         {0, 5, 15},
         {0, 10, 15},
         {0, 5, 15},
         {0, 10, 15},
         {0, 8, 15},
         {0, 13, 15},
         {0, 15, 3},
         {0, 12, 15},
         {0, 3, 15},
         {0, 3, 8}}};

const int g_aWeights2[] = {0, 21, 43, 64};
const int g_aWeights3[] = {0, 9, 18, 27, 37, 46, 55, 64};
const int g_aWeights4[] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};

const BC7Block::ModeInfo BC7Block::ms_aInfo[BC7Block::c_NumModes] =
    {
        {2, 4, 6, 0, 0, 3, 0, LDRColorA(4, 4, 4, 0), LDRColorA(5, 5, 5, 0)},
        // Mode 0: Color only, 3 Subsets, RGBP 4441 (unique P-bit), 3-bit indecies, 16 partitions
        {1, 6, 2, 0, 0, 3, 0, LDRColorA(6, 6, 6, 0), LDRColorA(7, 7, 7, 0)},
        // Mode 1: Color only, 2 Subsets, RGBP 6661 (shared P-bit), 3-bit indecies, 64 partitions
        {2, 6, 0, 0, 0, 2, 0, LDRColorA(5, 5, 5, 0), LDRColorA(5, 5, 5, 0)},
        // Mode 2: Color only, 3 Subsets, RGB 555, 2-bit indecies, 64 partitions
        {1, 6, 4, 0, 0, 2, 0, LDRColorA(7, 7, 7, 0), LDRColorA(8, 8, 8, 0)},
        // Mode 3: Color only, 2 Subsets, RGBP 7771 (unique P-bit), 2-bits indecies, 64 partitions
        {0, 0, 0, 2, 1, 2, 3, LDRColorA(5, 5, 5, 6), LDRColorA(5, 5, 5, 6)},
        // Mode 4: Color w/ Separate Alpha, 1 Subset, RGB 555, A6, 16x2/16x3-bit indices, 2-bit rotation, 1-bit index selector
        {0, 0, 0, 2, 0, 2, 2, LDRColorA(7, 7, 7, 8), LDRColorA(7, 7, 7, 8)},
        // Mode 5: Color w/ Separate Alpha, 1 Subset, RGB 777, A8, 16x2/16x2-bit indices, 2-bit rotation
        {0, 0, 2, 0, 0, 4, 0, LDRColorA(7, 7, 7, 7), LDRColorA(8, 8, 8, 8)},
        // Mode 6: Color+Alpha, 1 Subset, RGBAP 77771 (unique P-bit), 16x4-bit indecies
        {1, 6, 4, 0, 0, 2, 0, LDRColorA(5, 5, 5, 5), LDRColorA(6, 6, 6, 6)}
        // Mode 7: Color+Alpha, 2 Subsets, RGBAP 55551 (unique P-bit), 2-bit indices, 64 partitions
};
inline HDRColorA::HDRColorA(const LDRColorA &c) noexcept {
    r = float(c.r) * (1.0f / 255.0f);
    g = float(c.g) * (1.0f / 255.0f);
    b = float(c.b) * (1.0f / 255.0f);
    a = float(c.a) * (1.0f / 255.0f);
}

void FillWithErrorColors(HDRColorA *pOut) noexcept {
    for (size_t i = 0; i < NUM_PIXELS_PER_BLOCK; ++i) {
#ifdef _DEBUG
        // Use Magenta in debug as a highly-visible error color
        pOut[i] = HDRColorA(1.0f, 0.0f, 1.0f, 1.0f);
#else
        // In production use, default to black
        pOut[i] = HDRColorA(0.0f, 0.0f, 0.0f, 1.0f);
#endif
    }
}
inline bool IsFixUpOffset(size_t uPartitions, size_t uShape, size_t uOffset) noexcept {
    assert(uPartitions < 3 && uShape < 64 && uOffset < 16);
    LUISA_ASSUME(uPartitions < 3 && uShape < 64 && uOffset < 16);
    for (size_t p = 0; p <= uPartitions; p++) {
        if (uOffset == g_aFixUp[uPartitions][uShape][p]) {
            return true;
        }
    }
    return false;
}
void BC7Block::Decode(HDRColorA *pOut) const noexcept {
    assert(pOut);

    size_t uFirst = 0;
    while (uFirst < 128 && !GetBit(uFirst)) {}
    const uint8_t uMode = uint8_t(uFirst - 1);

    if (uMode < 8) {
        const uint8_t uPartitions = ms_aInfo[uMode].uPartitions;
        assert(uPartitions < BC7_MAX_REGIONS);
        LUISA_ASSUME(uPartitions < BC7_MAX_REGIONS);

        auto const uNumEndPts = static_cast<const uint8_t>((unsigned(uPartitions) + 1u) << 1);
        const uint8_t uIndexPrec = ms_aInfo[uMode].uIndexPrec;
        const uint8_t uIndexPrec2 = ms_aInfo[uMode].uIndexPrec2;
        size_t i;
        size_t uStartBit = size_t(uMode) + 1;
        uint8_t P[6];
        const uint8_t uShape = GetBits(uStartBit, ms_aInfo[uMode].uPartitionBits);
        assert(uShape < BC7_MAX_SHAPES);
        LUISA_ASSUME(uShape < BC7_MAX_SHAPES);

        const uint8_t uRotation = GetBits(uStartBit, ms_aInfo[uMode].uRotationBits);
        assert(uRotation < 4);

        const uint8_t uIndexMode = GetBits(uStartBit, ms_aInfo[uMode].uIndexModeBits);
        assert(uIndexMode < 2);

        LDRColorA c[BC7_MAX_REGIONS << 1];
        const LDRColorA RGBAPrec = ms_aInfo[uMode].RGBAPrec;
        const LDRColorA RGBAPrecWithP = ms_aInfo[uMode].RGBAPrecWithP;

        assert(uNumEndPts <= (BC7_MAX_REGIONS << 1));

        // Red channel
        for (i = 0; i < uNumEndPts; i++) {
            if (uStartBit + RGBAPrec.r > 128) {
#if defined(_WIN32) && defined(_DEBUG)
                OutputDebugStringA("BC7: Invalid block encountered during decoding\n");
#endif
                FillWithErrorColors(pOut);
                return;
            }

            c[i].r = GetBits(uStartBit, RGBAPrec.r);
        }

        // Green channel
        for (i = 0; i < uNumEndPts; i++) {
            if (uStartBit + RGBAPrec.g > 128) {
#if defined(_WIN32) && defined(_DEBUG)
                OutputDebugStringA("BC7: Invalid block encountered during decoding\n");
#endif
                FillWithErrorColors(pOut);
                return;
            }

            c[i].g = GetBits(uStartBit, RGBAPrec.g);
        }

        // Blue channel
        for (i = 0; i < uNumEndPts; i++) {
            if (uStartBit + RGBAPrec.b > 128) {
#if defined(_WIN32) && defined(_DEBUG)
                OutputDebugStringA("BC7: Invalid block encountered during decoding\n");
#endif
                FillWithErrorColors(pOut);
                return;
            }

            c[i].b = GetBits(uStartBit, RGBAPrec.b);
        }

        // Alpha channel
        for (i = 0; i < uNumEndPts; i++) {
            if (uStartBit + RGBAPrec.a > 128) {
#if defined(_WIN32) && defined(_DEBUG)
                OutputDebugStringA("BC7: Invalid block encountered during decoding\n");
#endif
                FillWithErrorColors(pOut);
                return;
            }

            c[i].a = RGBAPrec.a ? GetBits(uStartBit, RGBAPrec.a) : 255u;
        }

        // P-bits
        assert(ms_aInfo[uMode].uPBits <= 6);
        LUISA_ASSUME(ms_aInfo[uMode].uPBits <= 6);
        for (i = 0; i < ms_aInfo[uMode].uPBits; i++) {
            if (uStartBit > 127) {
#if defined(_WIN32) && defined(_DEBUG)
                OutputDebugStringA("BC7: Invalid block encountered during decoding\n");
#endif
                FillWithErrorColors(pOut);
                return;
            }

            P[i] = GetBit(uStartBit);
        }

        if (ms_aInfo[uMode].uPBits) {
            for (i = 0; i < uNumEndPts; i++) {
                size_t pi = i * ms_aInfo[uMode].uPBits / uNumEndPts;
                for (uint8_t ch = 0; ch < BC7_NUM_CHANNELS; ch++) {
                    if (RGBAPrec[ch] != RGBAPrecWithP[ch]) {
                        c[i][ch] = static_cast<uint8_t>((unsigned(c[i][ch]) << 1) | P[pi]);
                    }
                }
            }
        }

        for (i = 0; i < uNumEndPts; i++) {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
            c[i] = Unquantize(c[i], RGBAPrecWithP);
#ifdef __GNUC_
#pragma GCC diagnostic pop
#endif
        }

        uint8_t w1[NUM_PIXELS_PER_BLOCK], w2[NUM_PIXELS_PER_BLOCK];

        // read color indices
        for (i = 0; i < NUM_PIXELS_PER_BLOCK; i++) {
            const size_t uNumBits = IsFixUpOffset(ms_aInfo[uMode].uPartitions, uShape, i) ? uIndexPrec - 1u : uIndexPrec;
            if (uStartBit + uNumBits > 128) {
#if defined(_WIN32) && defined(_DEBUG)
                OutputDebugStringA("BC7: Invalid block encountered during decoding\n");
#endif
                FillWithErrorColors(pOut);
                return;
            }
            w1[i] = GetBits(uStartBit, uNumBits);
        }

        // read alpha indices
        if (uIndexPrec2) {
            for (i = 0; i < NUM_PIXELS_PER_BLOCK; i++) {
                const size_t uNumBits = i ? uIndexPrec2 : uIndexPrec2 - 1u;
                if (uStartBit + uNumBits > 128) {
#if defined(_WIN32) && defined(_DEBUG)
                    OutputDebugStringA("BC7: Invalid block encountered during decoding\n");
#endif
                    FillWithErrorColors(pOut);
                    return;
                }
                w2[i] = GetBits(uStartBit, uNumBits);
            }
        }

        for (i = 0; i < NUM_PIXELS_PER_BLOCK; ++i) {
            const uint8_t uRegion = g_aPartitionTable[uPartitions][uShape][i];
            LDRColorA outPixel;
            if (uIndexPrec2 == 0) {
                LDRColorA::Interpolate(c[uRegion << 1], c[(uRegion << 1) + 1], w1[i], w1[i], uIndexPrec, uIndexPrec, outPixel);
            } else {
                if (uIndexMode == 0) {
                    LDRColorA::Interpolate(c[uRegion << 1], c[(uRegion << 1) + 1], w1[i], w2[i], uIndexPrec, uIndexPrec2, outPixel);
                } else {
                    LDRColorA::Interpolate(c[uRegion << 1], c[(uRegion << 1) + 1], w2[i], w1[i], uIndexPrec2, uIndexPrec, outPixel);
                }
            }

            switch (uRotation) {
                case 1: luisa::swap(outPixel.r, outPixel.a); break;
                case 2: luisa::swap(outPixel.g, outPixel.a); break;
                case 3: luisa::swap(outPixel.b, outPixel.a); break;
                default: break;
            }

            pOut[i] = HDRColorA(outPixel);
        }
    } else {
#if defined(_WIN32) && defined(_DEBUG)
        OutputDebugStringA("BC7: Reserved mode 8 encountered during decoding\n");
#endif
        // Per the BC7 format spec, we must return transparent black
        memset(pOut, 0, sizeof(HDRColorA) * NUM_PIXELS_PER_BLOCK);
    }
}

void ZeroOutPixel(float *pix) {
    pix[0] = 0.f;
    pix[1] = 0.f;
    pix[2] = 0.f;
    pix[3] = 0.f;
}

void BC7Block::Decode(int x, int y, float *out) const noexcept {
    assert(out);
    unsigned pos = x + y * 4;

    size_t uFirst = 0;
    while (uFirst < 128 && !GetBit(uFirst)) {}
    const uint8_t uMode = uint8_t(uFirst - 1);

    if (uMode < 8) {
		const uint8_t uPartitions = ms_aInfo[uMode].uPartitions;
		assert(uPartitions < BC7_MAX_REGIONS);
		LUISA_ASSUME(uPartitions < BC7_MAX_REGIONS);

		auto const uNumEndPts = static_cast<const uint8_t>((unsigned(uPartitions) + 1u) << 1);
		const uint8_t uIndexPrec = ms_aInfo[uMode].uIndexPrec;
		const uint8_t uIndexPrec2 = ms_aInfo[uMode].uIndexPrec2;
		size_t uStartBit = size_t(uMode) + 1;
		uint8_t P[6];
		const uint8_t uShape = GetBits(uStartBit, ms_aInfo[uMode].uPartitionBits);
		assert(uShape < BC7_MAX_SHAPES);
		LUISA_ASSUME(uShape < BC7_MAX_SHAPES);

		const uint8_t uRotation = GetBits(uStartBit, ms_aInfo[uMode].uRotationBits);
		assert(uRotation < 4);

		const uint8_t uIndexMode = GetBits(uStartBit, ms_aInfo[uMode].uIndexModeBits);
		assert(uIndexMode < 2);

		LDRColorA c[BC7_MAX_REGIONS << 1];
		const LDRColorA RGBAPrec = ms_aInfo[uMode].RGBAPrec;
		const LDRColorA RGBAPrecWithP = ms_aInfo[uMode].RGBAPrecWithP;

		assert(uNumEndPts <= (BC7_MAX_REGIONS << 1));

		// Red channel
		for (unsigned i = 0; i < uNumEndPts; i++) {
			if (uStartBit + RGBAPrec.r > 128) {
#if defined(_WIN32) && defined(_DEBUG)
				OutputDebugStringA("BC7: Invalid block encountered during decoding\n");
#endif
				ZeroOutPixel(out);
				return;
			}

			c[i].r = GetBits(uStartBit, RGBAPrec.r);
		}

		// Green channel
		for (unsigned i = 0; i < uNumEndPts; i++) {
			if (uStartBit + RGBAPrec.g > 128) {
#if defined(_WIN32) && defined(_DEBUG)
				OutputDebugStringA("BC7: Invalid block encountered during decoding\n");
#endif
				ZeroOutPixel(out);
				return;
			}

			c[i].g = GetBits(uStartBit, RGBAPrec.g);
		}

		// Blue channel
		for (unsigned i = 0; i < uNumEndPts; i++) {
			if (uStartBit + RGBAPrec.b > 128) {
#if defined(_WIN32) && defined(_DEBUG)
				OutputDebugStringA("BC7: Invalid block encountered during decoding\n");
#endif
				ZeroOutPixel(out);
				return;
			}

			c[i].b = GetBits(uStartBit, RGBAPrec.b);
		}

		// Alpha channel
		for (unsigned i = 0; i < uNumEndPts; i++) {
			if (uStartBit + RGBAPrec.a > 128) {
#if defined(_WIN32) && defined(_DEBUG)
				OutputDebugStringA("BC7: Invalid block encountered during decoding\n");
#endif
				ZeroOutPixel(out);
				return;
			}

			c[i].a = RGBAPrec.a ? GetBits(uStartBit, RGBAPrec.a) : 255u;
		}

		// P-bits
		assert(ms_aInfo[uMode].uPBits <= 6);
		LUISA_ASSUME(ms_aInfo[uMode].uPBits <= 6);
		for (unsigned i = 0; i < ms_aInfo[uMode].uPBits; i++) {
			if (uStartBit > 127) {
#if defined(_WIN32) && defined(_DEBUG)
				OutputDebugStringA("BC7: Invalid block encountered during decoding\n");
#endif
				ZeroOutPixel(out);
				return;
			}

			P[i] = GetBit(uStartBit);
		}

		if (ms_aInfo[uMode].uPBits) {
			for (unsigned i = 0; i < uNumEndPts; i++) {
				size_t pi = i * ms_aInfo[uMode].uPBits / uNumEndPts;
				for (uint8_t ch = 0; ch < BC7_NUM_CHANNELS; ch++) {
					if (RGBAPrec[ch] != RGBAPrecWithP[ch]) {
						c[i][ch] = static_cast<uint8_t>((unsigned(c[i][ch]) << 1) | P[pi]);
					}
				}
			}
		}

		for (unsigned i = 0; i < uNumEndPts; i++) {
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
			c[i] = Unquantize(c[i], RGBAPrecWithP);
#if defined(__GNUC_) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
		}

		uint8_t w1[NUM_PIXELS_PER_BLOCK], w2[NUM_PIXELS_PER_BLOCK];

		// read color indices
		for (int i = 0; i < NUM_PIXELS_PER_BLOCK; i++) {
			const size_t uNumBits = IsFixUpOffset(ms_aInfo[uMode].uPartitions, uShape, i) ? uIndexPrec - 1u
																						  : uIndexPrec;
			if (uStartBit + uNumBits > 128) {
#if defined(_WIN32) && defined(_DEBUG)
				OutputDebugStringA("BC7: Invalid block encountered during decoding\n");
#endif
				ZeroOutPixel(out);
				return;
			}
			w1[i] = GetBits(uStartBit, uNumBits);
		}

		// read alpha indices
		if (uIndexPrec2) {
			for (int i = 0; i < NUM_PIXELS_PER_BLOCK; i++) {
				const size_t uNumBits = i ? uIndexPrec2 : uIndexPrec2 - 1u;
				if (uStartBit + uNumBits > 128) {
#if defined(_WIN32) && defined(_DEBUG)
					OutputDebugStringA("BC7: Invalid block encountered during decoding\n");
#endif
					ZeroOutPixel(out);
					return;
				}
				w2[i] = GetBits(uStartBit, uNumBits);
			}
		}

		for (int i = 0; i < NUM_PIXELS_PER_BLOCK; ++i) {
			if (i != pos)
				continue;
			const uint8_t uRegion = g_aPartitionTable[uPartitions][uShape][i];
			LDRColorA outPixel;
			if (uIndexPrec2 == 0) {
				LDRColorA::Interpolate(c[uRegion << 1], c[(uRegion << 1) + 1], w1[i], w1[i], uIndexPrec, uIndexPrec,
						outPixel);
			} else {
				if (uIndexMode == 0) {
					LDRColorA::Interpolate(c[uRegion << 1], c[(uRegion << 1) + 1], w1[i], w2[i], uIndexPrec,
							uIndexPrec2, outPixel);
				} else {
					LDRColorA::Interpolate(c[uRegion << 1], c[(uRegion << 1) + 1], w2[i], w1[i], uIndexPrec2,
							uIndexPrec, outPixel);
				}
			}

			switch (uRotation) {
				case 1:
					luisa::swap(outPixel.r, outPixel.a);
					break;
				case 2:
					luisa::swap(outPixel.g, outPixel.a);
					break;
				case 3:
					luisa::swap(outPixel.b, outPixel.a);
					break;
				default:
					break;
			}

			out[0] = outPixel.r * (1.0f / 255.0f);
			out[1] = outPixel.g * (1.0f / 255.0f);
			out[2] = outPixel.b * (1.0f / 255.0f);
			out[3] = outPixel.a * (1.0f / 255.0f);
			break;
		}
	}
	else [[unlikely]]
	{
#if defined(_WIN32) && defined(_DEBUG)
        OutputDebugStringA("BC7: Reserved mode 8 encountered during decoding\n");
#endif
        ZeroOutPixel(out);
    }
}

using HALF = uint16_t;
class INTColor {
public:
    int r, g, b;
    int pad;
    INTColor() = default;
    INTColor(int nr, int ng, int nb) noexcept : r(nr), g(ng), b(nb), pad(0) {}

    INTColor(INTColor const &) = default;
    INTColor &operator=(const INTColor &) = default;

    INTColor(INTColor &&) = default;
    INTColor &operator=(INTColor &&) = default;

    INTColor &operator+=(const INTColor &c) noexcept {
        r += c.r;
        g += c.g;
        b += c.b;
        return *this;
    }

    INTColor &operator-=(const INTColor &c) noexcept {
        r -= c.r;
        g -= c.g;
        b -= c.b;
        return *this;
    }

    INTColor &operator&=(const INTColor &c) noexcept {
        r &= c.r;
        g &= c.g;
        b &= c.b;
        return *this;
    }

    int &operator[](uint8_t i) noexcept {
        assert(i < sizeof(INTColor) / sizeof(int));
        LUISA_ASSUME(i < sizeof(INTColor) / sizeof(int));
        return reinterpret_cast<int *>(this)[i];
    }

    INTColor &Clamp(int iMin, int iMax) noexcept {
        r = std::min<int>(iMax, std::max<int>(iMin, r));
        g = std::min<int>(iMax, std::max<int>(iMin, g));
        b = std::min<int>(iMax, std::max<int>(iMin, b));
        return *this;
    }

    INTColor &SignExtend(const LDRColorA &Prec) noexcept {
        r = SIGN_EXTEND(r, int(Prec.r));
        g = SIGN_EXTEND(g, int(Prec.g));
        b = SIGN_EXTEND(b, int(Prec.b));
        return *this;
    }

    void ToF16(HALF aF16[3], bool bSigned) const noexcept {
        aF16[0] = INT2F16(r, bSigned);
        aF16[1] = INT2F16(g, bSigned);
        aF16[2] = INT2F16(b, bSigned);
    }

private:

    static HALF INT2F16(int input, bool bSigned) noexcept {
        HALF h;
        uint16_t out;
        if (bSigned) {
            int s = 0;
            if (input < 0) {
                s = F16S_MASK;
                input = -input;
            }
            out = uint16_t(s | input);
        } else {
            assert(input >= 0 && input <= F16MAX);
            out = static_cast<uint16_t>(input);
        }

        *reinterpret_cast<uint16_t *>(&h) = out;
        return h;
    }
};
struct INTEndPntPair {
    INTColor A;
    INTColor B;
};
inline void TransformInverse(INTEndPntPair aEndPts[], const LDRColorA &Prec, bool bSigned) noexcept {
    const INTColor WrapMask((1 << Prec.r) - 1, (1 << Prec.g) - 1, (1 << Prec.b) - 1);
    aEndPts[0].B += aEndPts[0].A;
    aEndPts[0].B &= WrapMask;
    aEndPts[1].A += aEndPts[0].A;
    aEndPts[1].A &= WrapMask;
    aEndPts[1].B += aEndPts[0].A;
    aEndPts[1].B &= WrapMask;
    if (bSigned) {
        aEndPts[0].B.SignExtend(Prec);
        aEndPts[1].A.SignExtend(Prec);
        aEndPts[1].B.SignExtend(Prec);
    }
}
inline float XMConvertHalfToFloat(HALF Value) noexcept {
#if defined(_XM_F16C_INTRINSICS_) && !defined(_XM_NO_INTRINSICS_)
    __m128i V1 = _mm_cvtsi32_si128(static_cast<int>(Value));
    __m128 V2 = _mm_cvtph_ps(V1);
    return _mm_cvtss_f32(V2);
#elif defined(_XM_ARM_NEON_INTRINSICS_) && (defined(_M_ARM64) || defined(_M_HYBRID_X86_ARM64) || defined(_M_ARM64EC) || __aarch64__) && !defined(_XM_NO_INTRINSICS_) && (!defined(__GNUC__) || (__ARM_FP & 2))
    uint16x4_t vHalf = vdup_n_u16(Value);
    float32x4_t vFloat = vcvt_f32_f16(vreinterpret_f16_u16(vHalf));
    return vgetq_lane_f32(vFloat, 0);
#else
    auto Mantissa = static_cast<uint32_t>(Value & 0x03FF);

    uint32_t Exponent = (Value & 0x7C00);
    if (Exponent == 0x7C00)// INF/NAN
    {
        Exponent = 0x8f;
    } else if (Exponent != 0)// The value is normalized
    {
        Exponent = static_cast<uint32_t>((static_cast<int>(Value) >> 10) & 0x1F);
    } else if (Mantissa != 0)// The value is denormalized
    {
        // Normalize the value in the resulting float
        Exponent = 1;

        do {
            Exponent--;
            Mantissa <<= 1;
        } while ((Mantissa & 0x0400) == 0);

        Mantissa &= 0x03FF;
    } else// The value is zero
    {
        Exponent = static_cast<uint32_t>(-112);
    }

    uint32_t Result =
        ((static_cast<uint32_t>(Value) & 0x8000) << 16)// Sign
        | ((Exponent + 112) << 23)                     // Exponent
        | (Mantissa << 13);                            // Mantissa

    return reinterpret_cast<float *>(&Result)[0];
#endif// !_XM_F16C_INTRINSICS_
}
void BC6HBlock::Decode(bool bSigned, HDRColorA *pOut) const noexcept {
    assert(pOut);

    size_t uStartBit = 0;
    uint8_t uMode = GetBits(uStartBit, 2u);
    if (uMode != 0x00 && uMode != 0x01) {
        uMode = static_cast<uint8_t>((unsigned(GetBits(uStartBit, 3)) << 2) | uMode);
    }

    assert(uMode < c_NumModeInfo);
    LUISA_ASSUME(uMode < c_NumModeInfo);

    if (ms_aModeToInfo[uMode] >= 0) {
        assert(static_cast<unsigned int>(ms_aModeToInfo[uMode]) < c_NumModes);
        LUISA_ASSUME(ms_aModeToInfo[uMode] < c_NumModes);
        const ModeDescriptor *desc = ms_aDesc[ms_aModeToInfo[uMode]];
        const ModeInfo &info = ms_aInfo[ms_aModeToInfo[uMode]];

        INTEndPntPair aEndPts[BC6H_MAX_REGIONS] = {};
        uint32_t uShape = 0;

        // Read header
        const size_t uHeaderBits = info.uPartitions > 0 ? 82u : 65u;
        while (uStartBit < uHeaderBits) {
            const size_t uCurBit = uStartBit;
            if (GetBit(uStartBit)) {
                switch (desc[uCurBit].m_eField) {
                    case D: uShape |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case RW: aEndPts[0].A.r |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case RX: aEndPts[0].B.r |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case RY: aEndPts[1].A.r |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case RZ: aEndPts[1].B.r |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case GW: aEndPts[0].A.g |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case GX: aEndPts[0].B.g |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case GY: aEndPts[1].A.g |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case GZ: aEndPts[1].B.g |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case BW: aEndPts[0].A.b |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case BX: aEndPts[0].B.b |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case BY: aEndPts[1].A.b |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case BZ: aEndPts[1].B.b |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    default: {
#if defined(_WIN32) && defined(_DEBUG)
                        OutputDebugStringA("BC6HBlock: Invalid header bits encountered during decoding\n");
#endif
                        FillWithErrorColors(pOut);
                        return;
                    }
                }
            }
        }

        assert(uShape < 64);
        LUISA_ASSUME(uShape < 64);

        // Sign extend necessary end points
        if (bSigned) {
            aEndPts[0].A.SignExtend(info.RGBAPrec[0][0]);
        }
        if (bSigned || info.bTransformed) {
            assert(info.uPartitions < BC6H_MAX_REGIONS);
            LUISA_ASSUME(info.uPartitions < BC6H_MAX_REGIONS);
            for (size_t p = 0; p <= info.uPartitions; ++p) {
                if (p != 0) {
                    aEndPts[p].A.SignExtend(info.RGBAPrec[p][0]);
                }
                aEndPts[p].B.SignExtend(info.RGBAPrec[p][1]);
            }
        }

        // Inverse transform the end points
        if (info.bTransformed) {
            TransformInverse(aEndPts, info.RGBAPrec[0][0], bSigned);
        }

        // Read indices
        for (size_t i = 0; i < NUM_PIXELS_PER_BLOCK; ++i) {
            const size_t uNumBits = IsFixUpOffset(info.uPartitions, uShape, i) ? info.uIndexPrec - 1u : info.uIndexPrec;
            if (uStartBit + uNumBits > 128) {
#if defined(_WIN32) && defined(_DEBUG)
                OutputDebugStringA("BC6HBlock: Invalid block encountered during decoding\n");
#endif
                FillWithErrorColors(pOut);
                return;
            }
            const uint8_t uIndex = GetBits(uStartBit, uNumBits);

            if (uIndex >= ((info.uPartitions > 0) ? 8 : 16)) {
#if defined(_WIN32) && defined(_DEBUG)
                OutputDebugStringA("BC6HBlock: Invalid index encountered during decoding\n");
#endif
                FillWithErrorColors(pOut);
                return;
            }

            const size_t uRegion = g_aPartitionTable[info.uPartitions][uShape][i];
            assert(uRegion < BC6H_MAX_REGIONS);
            LUISA_ASSUME(uRegion < BC6H_MAX_REGIONS);

            // Unquantize endpoints and interpolate
            const int r1 = Unquantize(aEndPts[uRegion].A.r, info.RGBAPrec[0][0].r, bSigned);
            const int g1 = Unquantize(aEndPts[uRegion].A.g, info.RGBAPrec[0][0].g, bSigned);
            const int b1 = Unquantize(aEndPts[uRegion].A.b, info.RGBAPrec[0][0].b, bSigned);
            const int r2 = Unquantize(aEndPts[uRegion].B.r, info.RGBAPrec[0][0].r, bSigned);
            const int g2 = Unquantize(aEndPts[uRegion].B.g, info.RGBAPrec[0][0].g, bSigned);
            const int b2 = Unquantize(aEndPts[uRegion].B.b, info.RGBAPrec[0][0].b, bSigned);
            const int *aWeights = info.uPartitions > 0 ? g_aWeights3 : g_aWeights4;
            INTColor fc;
            fc.r = FinishUnquantize((r1 * (BC67_WEIGHT_MAX - aWeights[uIndex]) + r2 * aWeights[uIndex] + BC67_WEIGHT_ROUND) >> BC67_WEIGHT_SHIFT, bSigned);
            fc.g = FinishUnquantize((g1 * (BC67_WEIGHT_MAX - aWeights[uIndex]) + g2 * aWeights[uIndex] + BC67_WEIGHT_ROUND) >> BC67_WEIGHT_SHIFT, bSigned);
            fc.b = FinishUnquantize((b1 * (BC67_WEIGHT_MAX - aWeights[uIndex]) + b2 * aWeights[uIndex] + BC67_WEIGHT_ROUND) >> BC67_WEIGHT_SHIFT, bSigned);

            HALF rgb[3];
            fc.ToF16(rgb, bSigned);

            pOut[i].r = XMConvertHalfToFloat(rgb[0]);
            pOut[i].g = XMConvertHalfToFloat(rgb[1]);
            pOut[i].b = XMConvertHalfToFloat(rgb[2]);
            pOut[i].a = 1.0f;
        }
    } else {
#if defined(_WIN32) && defined(_DEBUG)
        const char *warnstr = "BC6HBlock: Invalid mode encountered during decoding\n";
        switch (uMode) {
            case 0x13: warnstr = "BC6HBlock: Reserved mode 10011 encountered during decoding\n"; break;
            case 0x17: warnstr = "BC6HBlock: Reserved mode 10111 encountered during decoding\n"; break;
            case 0x1B: warnstr = "BC6HBlock: Reserved mode 11011 encountered during decoding\n"; break;
            case 0x1F: warnstr = "BC6HBlock: Reserved mode 11111 encountered during decoding\n"; break;
            default: break;
        }
        OutputDebugStringA(warnstr);
#endif
        // Per the BC6HBlock format spec, we must return opaque black
        for (size_t i = 0; i < NUM_PIXELS_PER_BLOCK; ++i) {
            pOut[i] = HDRColorA(0.0f, 0.0f, 0.0f, 1.0f);
        }
    }
}

int BC6HBlock::Quantize(int iValue, int prec, bool bSigned) noexcept {
    assert(prec > 1);// didn't bother to make it work for 1
    int q, s = 0;
    if (bSigned) {
        assert(iValue >= -F16MAX && iValue <= F16MAX);
        if (iValue < 0) {
            s = 1;
            iValue = -iValue;
        }
        q = (prec >= 16) ? iValue : (iValue << (prec - 1)) / (F16MAX + 1);
        if (s)
            q = -q;
        assert(q > -(1 << (prec - 1)) && q < (1 << (prec - 1)));
    } else {
        assert(iValue >= 0 && iValue <= F16MAX);
        q = (prec >= 15) ? iValue : (iValue << prec) / (F16MAX + 1);
        assert(q >= 0 && q < (1 << prec));
    }

    return q;
}

int BC6HBlock::Unquantize(int comp, uint8_t uBitsPerComp, bool bSigned) noexcept {
    int unq = 0, s = 0;
    if (bSigned) {
        if (uBitsPerComp >= 16) {
            unq = comp;
        } else {
            if (comp < 0) {
                s = 1;
                comp = -comp;
            }

            if (comp == 0)
                unq = 0;
            else if (comp >= ((1 << (uBitsPerComp - 1)) - 1))
                unq = 0x7FFF;
            else
                unq = ((comp << 15) + 0x4000) >> (uBitsPerComp - 1);

            if (s) unq = -unq;
        }
    } else {
        if (uBitsPerComp >= 15)
            unq = comp;
        else if (comp == 0)
            unq = 0;
        else if (comp == ((1 << uBitsPerComp) - 1))
            unq = 0xFFFF;
        else
            unq = ((comp << 16) + 0x8000) >> uBitsPerComp;
    }

    return unq;
}

int BC6HBlock::FinishUnquantize(int comp, bool bSigned) noexcept {
    if (bSigned) {
        return (comp < 0) ? -(((-comp) * 31) >> 5) : (comp * 31) >> 5;// scale the magnitude by 31/32
    } else {
        return (comp * 31) >> 6;// scale the magnitude by 31/64
    }
}
const int BC6HBlock::ms_aModeToInfo[BC6HBlock::c_NumModeInfo] =
    {
        0, // Mode 1   - 0x00
        1, // Mode 2   - 0x01
        2, // Mode 3   - 0x02
        10,// Mode 11  - 0x03
        -1,// Invalid  - 0x04
        -1,// Invalid  - 0x05
        3, // Mode 4   - 0x06
        11,// Mode 12  - 0x07
        -1,// Invalid  - 0x08
        -1,// Invalid  - 0x09
        4, // Mode 5   - 0x0a
        12,// Mode 13  - 0x0b
        -1,// Invalid  - 0x0c
        -1,// Invalid  - 0x0d
        5, // Mode 6   - 0x0e
        13,// Mode 14  - 0x0f
        -1,// Invalid  - 0x10
        -1,// Invalid  - 0x11
        6, // Mode 7   - 0x12
        -1,// Reserved - 0x13
        -1,// Invalid  - 0x14
        -1,// Invalid  - 0x15
        7, // Mode 8   - 0x16
        -1,// Reserved - 0x17
        -1,// Invalid  - 0x18
        -1,// Invalid  - 0x19
        8, // Mode 9   - 0x1a
        -1,// Reserved - 0x1b
        -1,// Invalid  - 0x1c
        -1,// Invalid  - 0x1d
        9, // Mode 10  - 0x1e
        -1,// Resreved - 0x1f
};
const BC6HBlock::ModeDescriptor BC6HBlock::ms_aDesc[BC6HBlock::c_NumModes][82] =
    {
        {
            // Mode 1 (0x00) - 10 5 5 5
            {M, 0},
            {M, 1},
            {GY, 4},
            {BY, 4},
            {BZ, 4},
            {RW, 0},
            {RW, 1},
            {RW, 2},
            {RW, 3},
            {RW, 4},
            {RW, 5},
            {RW, 6},
            {RW, 7},
            {RW, 8},
            {RW, 9},
            {GW, 0},
            {GW, 1},
            {GW, 2},
            {GW, 3},
            {GW, 4},
            {GW, 5},
            {GW, 6},
            {GW, 7},
            {GW, 8},
            {GW, 9},
            {BW, 0},
            {BW, 1},
            {BW, 2},
            {BW, 3},
            {BW, 4},
            {BW, 5},
            {BW, 6},
            {BW, 7},
            {BW, 8},
            {BW, 9},
            {RX, 0},
            {RX, 1},
            {RX, 2},
            {RX, 3},
            {RX, 4},
            {GZ, 4},
            {GY, 0},
            {GY, 1},
            {GY, 2},
            {GY, 3},
            {GX, 0},
            {GX, 1},
            {GX, 2},
            {GX, 3},
            {GX, 4},
            {BZ, 0},
            {GZ, 0},
            {GZ, 1},
            {GZ, 2},
            {GZ, 3},
            {BX, 0},
            {BX, 1},
            {BX, 2},
            {BX, 3},
            {BX, 4},
            {BZ, 1},
            {BY, 0},
            {BY, 1},
            {BY, 2},
            {BY, 3},
            {RY, 0},
            {RY, 1},
            {RY, 2},
            {RY, 3},
            {RY, 4},
            {BZ, 2},
            {RZ, 0},
            {RZ, 1},
            {RZ, 2},
            {RZ, 3},
            {RZ, 4},
            {BZ, 3},
            {D, 0},
            {D, 1},
            {D, 2},
            {D, 3},
            {D, 4},
        },

        {
            // Mode 2 (0x01) - 7 6 6 6
            {M, 0},
            {M, 1},
            {GY, 5},
            {GZ, 4},
            {GZ, 5},
            {RW, 0},
            {RW, 1},
            {RW, 2},
            {RW, 3},
            {RW, 4},
            {RW, 5},
            {RW, 6},
            {BZ, 0},
            {BZ, 1},
            {BY, 4},
            {GW, 0},
            {GW, 1},
            {GW, 2},
            {GW, 3},
            {GW, 4},
            {GW, 5},
            {GW, 6},
            {BY, 5},
            {BZ, 2},
            {GY, 4},
            {BW, 0},
            {BW, 1},
            {BW, 2},
            {BW, 3},
            {BW, 4},
            {BW, 5},
            {BW, 6},
            {BZ, 3},
            {BZ, 5},
            {BZ, 4},
            {RX, 0},
            {RX, 1},
            {RX, 2},
            {RX, 3},
            {RX, 4},
            {RX, 5},
            {GY, 0},
            {GY, 1},
            {GY, 2},
            {GY, 3},
            {GX, 0},
            {GX, 1},
            {GX, 2},
            {GX, 3},
            {GX, 4},
            {GX, 5},
            {GZ, 0},
            {GZ, 1},
            {GZ, 2},
            {GZ, 3},
            {BX, 0},
            {BX, 1},
            {BX, 2},
            {BX, 3},
            {BX, 4},
            {BX, 5},
            {BY, 0},
            {BY, 1},
            {BY, 2},
            {BY, 3},
            {RY, 0},
            {RY, 1},
            {RY, 2},
            {RY, 3},
            {RY, 4},
            {RY, 5},
            {RZ, 0},
            {RZ, 1},
            {RZ, 2},
            {RZ, 3},
            {RZ, 4},
            {RZ, 5},
            {D, 0},
            {D, 1},
            {D, 2},
            {D, 3},
            {D, 4},
        },

        {
            // Mode 3 (0x02) - 11 5 4 4
            {M, 0},
            {M, 1},
            {M, 2},
            {M, 3},
            {M, 4},
            {RW, 0},
            {RW, 1},
            {RW, 2},
            {RW, 3},
            {RW, 4},
            {RW, 5},
            {RW, 6},
            {RW, 7},
            {RW, 8},
            {RW, 9},
            {GW, 0},
            {GW, 1},
            {GW, 2},
            {GW, 3},
            {GW, 4},
            {GW, 5},
            {GW, 6},
            {GW, 7},
            {GW, 8},
            {GW, 9},
            {BW, 0},
            {BW, 1},
            {BW, 2},
            {BW, 3},
            {BW, 4},
            {BW, 5},
            {BW, 6},
            {BW, 7},
            {BW, 8},
            {BW, 9},
            {RX, 0},
            {RX, 1},
            {RX, 2},
            {RX, 3},
            {RX, 4},
            {RW, 10},
            {GY, 0},
            {GY, 1},
            {GY, 2},
            {GY, 3},
            {GX, 0},
            {GX, 1},
            {GX, 2},
            {GX, 3},
            {GW, 10},
            {BZ, 0},
            {GZ, 0},
            {GZ, 1},
            {GZ, 2},
            {GZ, 3},
            {BX, 0},
            {BX, 1},
            {BX, 2},
            {BX, 3},
            {BW, 10},
            {BZ, 1},
            {BY, 0},
            {BY, 1},
            {BY, 2},
            {BY, 3},
            {RY, 0},
            {RY, 1},
            {RY, 2},
            {RY, 3},
            {RY, 4},
            {BZ, 2},
            {RZ, 0},
            {RZ, 1},
            {RZ, 2},
            {RZ, 3},
            {RZ, 4},
            {BZ, 3},
            {D, 0},
            {D, 1},
            {D, 2},
            {D, 3},
            {D, 4},
        },

        {
            // Mode 4 (0x06) - 11 4 5 4
            {M, 0},
            {M, 1},
            {M, 2},
            {M, 3},
            {M, 4},
            {RW, 0},
            {RW, 1},
            {RW, 2},
            {RW, 3},
            {RW, 4},
            {RW, 5},
            {RW, 6},
            {RW, 7},
            {RW, 8},
            {RW, 9},
            {GW, 0},
            {GW, 1},
            {GW, 2},
            {GW, 3},
            {GW, 4},
            {GW, 5},
            {GW, 6},
            {GW, 7},
            {GW, 8},
            {GW, 9},
            {BW, 0},
            {BW, 1},
            {BW, 2},
            {BW, 3},
            {BW, 4},
            {BW, 5},
            {BW, 6},
            {BW, 7},
            {BW, 8},
            {BW, 9},
            {RX, 0},
            {RX, 1},
            {RX, 2},
            {RX, 3},
            {RW, 10},
            {GZ, 4},
            {GY, 0},
            {GY, 1},
            {GY, 2},
            {GY, 3},
            {GX, 0},
            {GX, 1},
            {GX, 2},
            {GX, 3},
            {GX, 4},
            {GW, 10},
            {GZ, 0},
            {GZ, 1},
            {GZ, 2},
            {GZ, 3},
            {BX, 0},
            {BX, 1},
            {BX, 2},
            {BX, 3},
            {BW, 10},
            {BZ, 1},
            {BY, 0},
            {BY, 1},
            {BY, 2},
            {BY, 3},
            {RY, 0},
            {RY, 1},
            {RY, 2},
            {RY, 3},
            {BZ, 0},
            {BZ, 2},
            {RZ, 0},
            {RZ, 1},
            {RZ, 2},
            {RZ, 3},
            {GY, 4},
            {BZ, 3},
            {D, 0},
            {D, 1},
            {D, 2},
            {D, 3},
            {D, 4},
        },

        {
            // Mode 5 (0x0a) - 11 4 4 5
            {M, 0},
            {M, 1},
            {M, 2},
            {M, 3},
            {M, 4},
            {RW, 0},
            {RW, 1},
            {RW, 2},
            {RW, 3},
            {RW, 4},
            {RW, 5},
            {RW, 6},
            {RW, 7},
            {RW, 8},
            {RW, 9},
            {GW, 0},
            {GW, 1},
            {GW, 2},
            {GW, 3},
            {GW, 4},
            {GW, 5},
            {GW, 6},
            {GW, 7},
            {GW, 8},
            {GW, 9},
            {BW, 0},
            {BW, 1},
            {BW, 2},
            {BW, 3},
            {BW, 4},
            {BW, 5},
            {BW, 6},
            {BW, 7},
            {BW, 8},
            {BW, 9},
            {RX, 0},
            {RX, 1},
            {RX, 2},
            {RX, 3},
            {RW, 10},
            {BY, 4},
            {GY, 0},
            {GY, 1},
            {GY, 2},
            {GY, 3},
            {GX, 0},
            {GX, 1},
            {GX, 2},
            {GX, 3},
            {GW, 10},
            {BZ, 0},
            {GZ, 0},
            {GZ, 1},
            {GZ, 2},
            {GZ, 3},
            {BX, 0},
            {BX, 1},
            {BX, 2},
            {BX, 3},
            {BX, 4},
            {BW, 10},
            {BY, 0},
            {BY, 1},
            {BY, 2},
            {BY, 3},
            {RY, 0},
            {RY, 1},
            {RY, 2},
            {RY, 3},
            {BZ, 1},
            {BZ, 2},
            {RZ, 0},
            {RZ, 1},
            {RZ, 2},
            {RZ, 3},
            {BZ, 4},
            {BZ, 3},
            {D, 0},
            {D, 1},
            {D, 2},
            {D, 3},
            {D, 4},
        },

        {
            // Mode 6 (0x0e) - 9 5 5 5
            {M, 0},
            {M, 1},
            {M, 2},
            {M, 3},
            {M, 4},
            {RW, 0},
            {RW, 1},
            {RW, 2},
            {RW, 3},
            {RW, 4},
            {RW, 5},
            {RW, 6},
            {RW, 7},
            {RW, 8},
            {BY, 4},
            {GW, 0},
            {GW, 1},
            {GW, 2},
            {GW, 3},
            {GW, 4},
            {GW, 5},
            {GW, 6},
            {GW, 7},
            {GW, 8},
            {GY, 4},
            {BW, 0},
            {BW, 1},
            {BW, 2},
            {BW, 3},
            {BW, 4},
            {BW, 5},
            {BW, 6},
            {BW, 7},
            {BW, 8},
            {BZ, 4},
            {RX, 0},
            {RX, 1},
            {RX, 2},
            {RX, 3},
            {RX, 4},
            {GZ, 4},
            {GY, 0},
            {GY, 1},
            {GY, 2},
            {GY, 3},
            {GX, 0},
            {GX, 1},
            {GX, 2},
            {GX, 3},
            {GX, 4},
            {BZ, 0},
            {GZ, 0},
            {GZ, 1},
            {GZ, 2},
            {GZ, 3},
            {BX, 0},
            {BX, 1},
            {BX, 2},
            {BX, 3},
            {BX, 4},
            {BZ, 1},
            {BY, 0},
            {BY, 1},
            {BY, 2},
            {BY, 3},
            {RY, 0},
            {RY, 1},
            {RY, 2},
            {RY, 3},
            {RY, 4},
            {BZ, 2},
            {RZ, 0},
            {RZ, 1},
            {RZ, 2},
            {RZ, 3},
            {RZ, 4},
            {BZ, 3},
            {D, 0},
            {D, 1},
            {D, 2},
            {D, 3},
            {D, 4},
        },

        {
            // Mode 7 (0x12) - 8 6 5 5
            {M, 0},
            {M, 1},
            {M, 2},
            {M, 3},
            {M, 4},
            {RW, 0},
            {RW, 1},
            {RW, 2},
            {RW, 3},
            {RW, 4},
            {RW, 5},
            {RW, 6},
            {RW, 7},
            {GZ, 4},
            {BY, 4},
            {GW, 0},
            {GW, 1},
            {GW, 2},
            {GW, 3},
            {GW, 4},
            {GW, 5},
            {GW, 6},
            {GW, 7},
            {BZ, 2},
            {GY, 4},
            {BW, 0},
            {BW, 1},
            {BW, 2},
            {BW, 3},
            {BW, 4},
            {BW, 5},
            {BW, 6},
            {BW, 7},
            {BZ, 3},
            {BZ, 4},
            {RX, 0},
            {RX, 1},
            {RX, 2},
            {RX, 3},
            {RX, 4},
            {RX, 5},
            {GY, 0},
            {GY, 1},
            {GY, 2},
            {GY, 3},
            {GX, 0},
            {GX, 1},
            {GX, 2},
            {GX, 3},
            {GX, 4},
            {BZ, 0},
            {GZ, 0},
            {GZ, 1},
            {GZ, 2},
            {GZ, 3},
            {BX, 0},
            {BX, 1},
            {BX, 2},
            {BX, 3},
            {BX, 4},
            {BZ, 1},
            {BY, 0},
            {BY, 1},
            {BY, 2},
            {BY, 3},
            {RY, 0},
            {RY, 1},
            {RY, 2},
            {RY, 3},
            {RY, 4},
            {RY, 5},
            {RZ, 0},
            {RZ, 1},
            {RZ, 2},
            {RZ, 3},
            {RZ, 4},
            {RZ, 5},
            {D, 0},
            {D, 1},
            {D, 2},
            {D, 3},
            {D, 4},
        },

        {
            // Mode 8 (0x16) - 8 5 6 5
            {M, 0},
            {M, 1},
            {M, 2},
            {M, 3},
            {M, 4},
            {RW, 0},
            {RW, 1},
            {RW, 2},
            {RW, 3},
            {RW, 4},
            {RW, 5},
            {RW, 6},
            {RW, 7},
            {BZ, 0},
            {BY, 4},
            {GW, 0},
            {GW, 1},
            {GW, 2},
            {GW, 3},
            {GW, 4},
            {GW, 5},
            {GW, 6},
            {GW, 7},
            {GY, 5},
            {GY, 4},
            {BW, 0},
            {BW, 1},
            {BW, 2},
            {BW, 3},
            {BW, 4},
            {BW, 5},
            {BW, 6},
            {BW, 7},
            {GZ, 5},
            {BZ, 4},
            {RX, 0},
            {RX, 1},
            {RX, 2},
            {RX, 3},
            {RX, 4},
            {GZ, 4},
            {GY, 0},
            {GY, 1},
            {GY, 2},
            {GY, 3},
            {GX, 0},
            {GX, 1},
            {GX, 2},
            {GX, 3},
            {GX, 4},
            {GX, 5},
            {GZ, 0},
            {GZ, 1},
            {GZ, 2},
            {GZ, 3},
            {BX, 0},
            {BX, 1},
            {BX, 2},
            {BX, 3},
            {BX, 4},
            {BZ, 1},
            {BY, 0},
            {BY, 1},
            {BY, 2},
            {BY, 3},
            {RY, 0},
            {RY, 1},
            {RY, 2},
            {RY, 3},
            {RY, 4},
            {BZ, 2},
            {RZ, 0},
            {RZ, 1},
            {RZ, 2},
            {RZ, 3},
            {RZ, 4},
            {BZ, 3},
            {D, 0},
            {D, 1},
            {D, 2},
            {D, 3},
            {D, 4},
        },

        {
            // Mode 9 (0x1a) - 8 5 5 6
            {M, 0},
            {M, 1},
            {M, 2},
            {M, 3},
            {M, 4},
            {RW, 0},
            {RW, 1},
            {RW, 2},
            {RW, 3},
            {RW, 4},
            {RW, 5},
            {RW, 6},
            {RW, 7},
            {BZ, 1},
            {BY, 4},
            {GW, 0},
            {GW, 1},
            {GW, 2},
            {GW, 3},
            {GW, 4},
            {GW, 5},
            {GW, 6},
            {GW, 7},
            {BY, 5},
            {GY, 4},
            {BW, 0},
            {BW, 1},
            {BW, 2},
            {BW, 3},
            {BW, 4},
            {BW, 5},
            {BW, 6},
            {BW, 7},
            {BZ, 5},
            {BZ, 4},
            {RX, 0},
            {RX, 1},
            {RX, 2},
            {RX, 3},
            {RX, 4},
            {GZ, 4},
            {GY, 0},
            {GY, 1},
            {GY, 2},
            {GY, 3},
            {GX, 0},
            {GX, 1},
            {GX, 2},
            {GX, 3},
            {GX, 4},
            {BZ, 0},
            {GZ, 0},
            {GZ, 1},
            {GZ, 2},
            {GZ, 3},
            {BX, 0},
            {BX, 1},
            {BX, 2},
            {BX, 3},
            {BX, 4},
            {BX, 5},
            {BY, 0},
            {BY, 1},
            {BY, 2},
            {BY, 3},
            {RY, 0},
            {RY, 1},
            {RY, 2},
            {RY, 3},
            {RY, 4},
            {BZ, 2},
            {RZ, 0},
            {RZ, 1},
            {RZ, 2},
            {RZ, 3},
            {RZ, 4},
            {BZ, 3},
            {D, 0},
            {D, 1},
            {D, 2},
            {D, 3},
            {D, 4},
        },

        {
            // Mode 10 (0x1e) - 6 6 6 6
            {M, 0},
            {M, 1},
            {M, 2},
            {M, 3},
            {M, 4},
            {RW, 0},
            {RW, 1},
            {RW, 2},
            {RW, 3},
            {RW, 4},
            {RW, 5},
            {GZ, 4},
            {BZ, 0},
            {BZ, 1},
            {BY, 4},
            {GW, 0},
            {GW, 1},
            {GW, 2},
            {GW, 3},
            {GW, 4},
            {GW, 5},
            {GY, 5},
            {BY, 5},
            {BZ, 2},
            {GY, 4},
            {BW, 0},
            {BW, 1},
            {BW, 2},
            {BW, 3},
            {BW, 4},
            {BW, 5},
            {GZ, 5},
            {BZ, 3},
            {BZ, 5},
            {BZ, 4},
            {RX, 0},
            {RX, 1},
            {RX, 2},
            {RX, 3},
            {RX, 4},
            {RX, 5},
            {GY, 0},
            {GY, 1},
            {GY, 2},
            {GY, 3},
            {GX, 0},
            {GX, 1},
            {GX, 2},
            {GX, 3},
            {GX, 4},
            {GX, 5},
            {GZ, 0},
            {GZ, 1},
            {GZ, 2},
            {GZ, 3},
            {BX, 0},
            {BX, 1},
            {BX, 2},
            {BX, 3},
            {BX, 4},
            {BX, 5},
            {BY, 0},
            {BY, 1},
            {BY, 2},
            {BY, 3},
            {RY, 0},
            {RY, 1},
            {RY, 2},
            {RY, 3},
            {RY, 4},
            {RY, 5},
            {RZ, 0},
            {RZ, 1},
            {RZ, 2},
            {RZ, 3},
            {RZ, 4},
            {RZ, 5},
            {D, 0},
            {D, 1},
            {D, 2},
            {D, 3},
            {D, 4},
        },

        {
            // Mode 11 (0x03) - 10 10
            {M, 0},
            {M, 1},
            {M, 2},
            {M, 3},
            {M, 4},
            {RW, 0},
            {RW, 1},
            {RW, 2},
            {RW, 3},
            {RW, 4},
            {RW, 5},
            {RW, 6},
            {RW, 7},
            {RW, 8},
            {RW, 9},
            {GW, 0},
            {GW, 1},
            {GW, 2},
            {GW, 3},
            {GW, 4},
            {GW, 5},
            {GW, 6},
            {GW, 7},
            {GW, 8},
            {GW, 9},
            {BW, 0},
            {BW, 1},
            {BW, 2},
            {BW, 3},
            {BW, 4},
            {BW, 5},
            {BW, 6},
            {BW, 7},
            {BW, 8},
            {BW, 9},
            {RX, 0},
            {RX, 1},
            {RX, 2},
            {RX, 3},
            {RX, 4},
            {RX, 5},
            {RX, 6},
            {RX, 7},
            {RX, 8},
            {RX, 9},
            {GX, 0},
            {GX, 1},
            {GX, 2},
            {GX, 3},
            {GX, 4},
            {GX, 5},
            {GX, 6},
            {GX, 7},
            {GX, 8},
            {GX, 9},
            {BX, 0},
            {BX, 1},
            {BX, 2},
            {BX, 3},
            {BX, 4},
            {BX, 5},
            {BX, 6},
            {BX, 7},
            {BX, 8},
            {BX, 9},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
        },

        {
            // Mode 12 (0x07) - 11 9
            {M, 0},
            {M, 1},
            {M, 2},
            {M, 3},
            {M, 4},
            {RW, 0},
            {RW, 1},
            {RW, 2},
            {RW, 3},
            {RW, 4},
            {RW, 5},
            {RW, 6},
            {RW, 7},
            {RW, 8},
            {RW, 9},
            {GW, 0},
            {GW, 1},
            {GW, 2},
            {GW, 3},
            {GW, 4},
            {GW, 5},
            {GW, 6},
            {GW, 7},
            {GW, 8},
            {GW, 9},
            {BW, 0},
            {BW, 1},
            {BW, 2},
            {BW, 3},
            {BW, 4},
            {BW, 5},
            {BW, 6},
            {BW, 7},
            {BW, 8},
            {BW, 9},
            {RX, 0},
            {RX, 1},
            {RX, 2},
            {RX, 3},
            {RX, 4},
            {RX, 5},
            {RX, 6},
            {RX, 7},
            {RX, 8},
            {RW, 10},
            {GX, 0},
            {GX, 1},
            {GX, 2},
            {GX, 3},
            {GX, 4},
            {GX, 5},
            {GX, 6},
            {GX, 7},
            {GX, 8},
            {GW, 10},
            {BX, 0},
            {BX, 1},
            {BX, 2},
            {BX, 3},
            {BX, 4},
            {BX, 5},
            {BX, 6},
            {BX, 7},
            {BX, 8},
            {BW, 10},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
        },

        {
            // Mode 13 (0x0b) - 12 8
            {M, 0},
            {M, 1},
            {M, 2},
            {M, 3},
            {M, 4},
            {RW, 0},
            {RW, 1},
            {RW, 2},
            {RW, 3},
            {RW, 4},
            {RW, 5},
            {RW, 6},
            {RW, 7},
            {RW, 8},
            {RW, 9},
            {GW, 0},
            {GW, 1},
            {GW, 2},
            {GW, 3},
            {GW, 4},
            {GW, 5},
            {GW, 6},
            {GW, 7},
            {GW, 8},
            {GW, 9},
            {BW, 0},
            {BW, 1},
            {BW, 2},
            {BW, 3},
            {BW, 4},
            {BW, 5},
            {BW, 6},
            {BW, 7},
            {BW, 8},
            {BW, 9},
            {RX, 0},
            {RX, 1},
            {RX, 2},
            {RX, 3},
            {RX, 4},
            {RX, 5},
            {RX, 6},
            {RX, 7},
            {RW, 11},
            {RW, 10},
            {GX, 0},
            {GX, 1},
            {GX, 2},
            {GX, 3},
            {GX, 4},
            {GX, 5},
            {GX, 6},
            {GX, 7},
            {GW, 11},
            {GW, 10},
            {BX, 0},
            {BX, 1},
            {BX, 2},
            {BX, 3},
            {BX, 4},
            {BX, 5},
            {BX, 6},
            {BX, 7},
            {BW, 11},
            {BW, 10},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
        },

        {
            // Mode 14 (0x0f) - 16 4
            {M, 0},
            {M, 1},
            {M, 2},
            {M, 3},
            {M, 4},
            {RW, 0},
            {RW, 1},
            {RW, 2},
            {RW, 3},
            {RW, 4},
            {RW, 5},
            {RW, 6},
            {RW, 7},
            {RW, 8},
            {RW, 9},
            {GW, 0},
            {GW, 1},
            {GW, 2},
            {GW, 3},
            {GW, 4},
            {GW, 5},
            {GW, 6},
            {GW, 7},
            {GW, 8},
            {GW, 9},
            {BW, 0},
            {BW, 1},
            {BW, 2},
            {BW, 3},
            {BW, 4},
            {BW, 5},
            {BW, 6},
            {BW, 7},
            {BW, 8},
            {BW, 9},
            {RX, 0},
            {RX, 1},
            {RX, 2},
            {RX, 3},
            {RW, 15},
            {RW, 14},
            {RW, 13},
            {RW, 12},
            {RW, 11},
            {RW, 10},
            {GX, 0},
            {GX, 1},
            {GX, 2},
            {GX, 3},
            {GW, 15},
            {GW, 14},
            {GW, 13},
            {GW, 12},
            {GW, 11},
            {GW, 10},
            {BX, 0},
            {BX, 1},
            {BX, 2},
            {BX, 3},
            {BW, 15},
            {BW, 14},
            {BW, 13},
            {BW, 12},
            {BW, 11},
            {BW, 10},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
            {NA, 0},
        },
};

// Mode, Partitions, Transformed, IndexPrec, RGBAPrec
const BC6HBlock::ModeInfo BC6HBlock::ms_aInfo[BC6HBlock::c_NumModes] =
    {
        {0x00, 1, true, 3, {{LDRColorA(10, 10, 10, 0), LDRColorA(5, 5, 5, 0)}, {LDRColorA(5, 5, 5, 0), LDRColorA(5, 5, 5, 0)}}},    // Mode 1
        {0x01, 1, true, 3, {{LDRColorA(7, 7, 7, 0), LDRColorA(6, 6, 6, 0)}, {LDRColorA(6, 6, 6, 0), LDRColorA(6, 6, 6, 0)}}},       // Mode 2
        {0x02, 1, true, 3, {{LDRColorA(11, 11, 11, 0), LDRColorA(5, 4, 4, 0)}, {LDRColorA(5, 4, 4, 0), LDRColorA(5, 4, 4, 0)}}},    // Mode 3
        {0x06, 1, true, 3, {{LDRColorA(11, 11, 11, 0), LDRColorA(4, 5, 4, 0)}, {LDRColorA(4, 5, 4, 0), LDRColorA(4, 5, 4, 0)}}},    // Mode 4
        {0x0a, 1, true, 3, {{LDRColorA(11, 11, 11, 0), LDRColorA(4, 4, 5, 0)}, {LDRColorA(4, 4, 5, 0), LDRColorA(4, 4, 5, 0)}}},    // Mode 5
        {0x0e, 1, true, 3, {{LDRColorA(9, 9, 9, 0), LDRColorA(5, 5, 5, 0)}, {LDRColorA(5, 5, 5, 0), LDRColorA(5, 5, 5, 0)}}},       // Mode 6
        {0x12, 1, true, 3, {{LDRColorA(8, 8, 8, 0), LDRColorA(6, 5, 5, 0)}, {LDRColorA(6, 5, 5, 0), LDRColorA(6, 5, 5, 0)}}},       // Mode 7
        {0x16, 1, true, 3, {{LDRColorA(8, 8, 8, 0), LDRColorA(5, 6, 5, 0)}, {LDRColorA(5, 6, 5, 0), LDRColorA(5, 6, 5, 0)}}},       // Mode 8
        {0x1a, 1, true, 3, {{LDRColorA(8, 8, 8, 0), LDRColorA(5, 5, 6, 0)}, {LDRColorA(5, 5, 6, 0), LDRColorA(5, 5, 6, 0)}}},       // Mode 9
        {0x1e, 1, false, 3, {{LDRColorA(6, 6, 6, 0), LDRColorA(6, 6, 6, 0)}, {LDRColorA(6, 6, 6, 0), LDRColorA(6, 6, 6, 0)}}},      // Mode 10
        {0x03, 0, false, 4, {{LDRColorA(10, 10, 10, 0), LDRColorA(10, 10, 10, 0)}, {LDRColorA(0, 0, 0, 0), LDRColorA(0, 0, 0, 0)}}},// Mode 11
        {0x07, 0, true, 4, {{LDRColorA(11, 11, 11, 0), LDRColorA(9, 9, 9, 0)}, {LDRColorA(0, 0, 0, 0), LDRColorA(0, 0, 0, 0)}}},    // Mode 12
        {0x0b, 0, true, 4, {{LDRColorA(12, 12, 12, 0), LDRColorA(8, 8, 8, 0)}, {LDRColorA(0, 0, 0, 0), LDRColorA(0, 0, 0, 0)}}},    // Mode 13
        {0x0f, 0, true, 4, {{LDRColorA(16, 16, 16, 0), LDRColorA(4, 4, 4, 0)}, {LDRColorA(0, 0, 0, 0), LDRColorA(0, 0, 0, 0)}}},    // Mode 14
};
void BC6HBlock::Decode(bool bSigned, int x, int y, float *out) const noexcept {
    LUISA_ASSUME(out);
    const int pos = x + y * 4;
    size_t uStartBit = 0;
    uint8_t uMode = GetBits(uStartBit, 2u);
    if (uMode != 0x00 && uMode != 0x01) {
        uMode = static_cast<uint8_t>((unsigned(GetBits(uStartBit, 3)) << 2) | uMode);
    }

    LUISA_ASSUME(uMode < c_NumModeInfo);
    if (ms_aModeToInfo[uMode] >= 0) {
        LUISA_ASSUME(static_cast<unsigned int>(ms_aModeToInfo[uMode]) < c_NumModes);
        const ModeDescriptor *desc = ms_aDesc[ms_aModeToInfo[uMode]];
        const ModeInfo &info = ms_aInfo[ms_aModeToInfo[uMode]];

        INTEndPntPair aEndPts[BC6H_MAX_REGIONS] = {};
        uint32_t uShape = 0;

        // Read header
        const size_t uHeaderBits = info.uPartitions > 0 ? 82u : 65u;
        while (uStartBit < uHeaderBits) {
            const size_t uCurBit = uStartBit;
            if (GetBit(uStartBit)) {
                switch (desc[uCurBit].m_eField) {
                    case D: uShape |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case RW: aEndPts[0].A.r |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case RX: aEndPts[0].B.r |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case RY: aEndPts[1].A.r |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case RZ: aEndPts[1].B.r |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case GW: aEndPts[0].A.g |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case GX: aEndPts[0].B.g |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case GY: aEndPts[1].A.g |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case GZ: aEndPts[1].B.g |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case BW: aEndPts[0].A.b |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case BX: aEndPts[0].B.b |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case BY: aEndPts[1].A.b |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    case BZ: aEndPts[1].B.b |= 1 << uint32_t(desc[uCurBit].m_uBit); break;
                    default: {
                        ZeroOutPixel(out);
                        return;
                    }
                }
            }
        }

        LUISA_ASSUME(uShape < 64);

        // Sign extend necessary end points
        if (bSigned) {
            aEndPts[0].A.SignExtend(info.RGBAPrec[0][0]);
        }
        if (bSigned || info.bTransformed) {
            LUISA_ASSUME(info.uPartitions < BC6H_MAX_REGIONS);
            for (size_t p = 0; p <= info.uPartitions; ++p) {
                if (p != 0) {
                    aEndPts[p].A.SignExtend(info.RGBAPrec[p][0]);
                }
                aEndPts[p].B.SignExtend(info.RGBAPrec[p][1]);
            }
        }

        // Inverse transform the end points
        if (info.bTransformed) {
            TransformInverse(aEndPts, info.RGBAPrec[0][0], bSigned);
        }

        // Read indices
        for (size_t i = 0; i < NUM_PIXELS_PER_BLOCK; ++i) {
            const size_t uNumBits = IsFixUpOffset(info.uPartitions, uShape, i) ? info.uIndexPrec - 1u : info.uIndexPrec;
            if (uStartBit + uNumBits > 128) {
                ZeroOutPixel(out);
                return;
            }
            const uint8_t uIndex = GetBits(uStartBit, uNumBits);

            if (uIndex >= ((info.uPartitions > 0) ? 8 : 16)) {
                ZeroOutPixel(out);
                return;
            }
            if (i == pos) {
                const size_t uRegion = g_aPartitionTable[info.uPartitions][uShape][i];
                LUISA_ASSUME(uRegion < BC6H_MAX_REGIONS);
                // Unquantize endpoints and interpolate
                const int r1 = Unquantize(aEndPts[uRegion].A.r, info.RGBAPrec[0][0].r, bSigned);
                const int g1 = Unquantize(aEndPts[uRegion].A.g, info.RGBAPrec[0][0].g, bSigned);
                const int b1 = Unquantize(aEndPts[uRegion].A.b, info.RGBAPrec[0][0].b, bSigned);
                const int r2 = Unquantize(aEndPts[uRegion].B.r, info.RGBAPrec[0][0].r, bSigned);
                const int g2 = Unquantize(aEndPts[uRegion].B.g, info.RGBAPrec[0][0].g, bSigned);
                const int b2 = Unquantize(aEndPts[uRegion].B.b, info.RGBAPrec[0][0].b, bSigned);
                const int *aWeights = info.uPartitions > 0 ? g_aWeights3 : g_aWeights4;
                INTColor fc;
                fc.r = FinishUnquantize((r1 * (BC67_WEIGHT_MAX - aWeights[uIndex]) + r2 * aWeights[uIndex] + BC67_WEIGHT_ROUND) >> BC67_WEIGHT_SHIFT, bSigned);
                fc.g = FinishUnquantize((g1 * (BC67_WEIGHT_MAX - aWeights[uIndex]) + g2 * aWeights[uIndex] + BC67_WEIGHT_ROUND) >> BC67_WEIGHT_SHIFT, bSigned);
                fc.b = FinishUnquantize((b1 * (BC67_WEIGHT_MAX - aWeights[uIndex]) + b2 * aWeights[uIndex] + BC67_WEIGHT_ROUND) >> BC67_WEIGHT_SHIFT, bSigned);

                HALF rgb[3];
                fc.ToF16(rgb, bSigned);

                out[0] = XMConvertHalfToFloat(rgb[0]);
                out[1] = XMConvertHalfToFloat(rgb[1]);
                out[2] = XMConvertHalfToFloat(rgb[2]);
                out[3] = 1.0f;
                break;
            }
        }
    } else {
        ZeroOutPixel(out);
    }
}

}// namespace luisa::compute::fallback::bc
