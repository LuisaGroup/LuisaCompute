#include <luisa/core/basic_types.h>

namespace luisa {

template struct Vector<bool, 2>;
template struct Vector<bool, 3>;
template struct Vector<bool, 4>;
template struct Vector<short, 2>;
template struct Vector<short, 3>;
template struct Vector<short, 4>;
template struct Vector<ushort, 2>;
template struct Vector<ushort, 3>;
template struct Vector<ushort, 4>;
template struct Vector<int, 2>;
template struct Vector<int, 3>;
template struct Vector<int, 4>;
template struct Vector<uint, 2>;
template struct Vector<uint, 3>;
template struct Vector<uint, 4>;
template struct Vector<float, 2>;
template struct Vector<float, 3>;
template struct Vector<float, 4>;
template struct Vector<slong, 2>;
template struct Vector<slong, 3>;
template struct Vector<slong, 4>;
template struct Vector<ulong, 2>;
template struct Vector<ulong, 3>;
template struct Vector<ulong, 4>;
template struct Vector<half, 2>;
template struct Vector<half, 3>;
template struct Vector<half, 4>;

namespace detail {

// type check
static_assert(vector_alignment_v<float, 2> == 8u);
static_assert(vector_alignment_v<float, 3> == 16u);
static_assert(vector_alignment_v<float, 4> == 16u);
static_assert(vector_alignment_v<bool, 2> == 2u);
static_assert(vector_alignment_v<bool, 3> == 4u);
static_assert(vector_alignment_v<bool, 4> == 4u);
static_assert(vector_alignment_v<short, 2> == 4u);
static_assert(vector_alignment_v<short, 3> == 8u);
static_assert(vector_alignment_v<short, 4> == 8u);
static_assert(vector_alignment_v<int, 2> == 8u);
static_assert(vector_alignment_v<int, 3> == 16u);
static_assert(vector_alignment_v<int, 4> == 16u);
static_assert(vector_alignment_v<slong, 2> == 16u);
static_assert(vector_alignment_v<slong, 3> == 16u);
static_assert(vector_alignment_v<slong, 4> == 16u);

}// namespace detail

}// namespace luisa
