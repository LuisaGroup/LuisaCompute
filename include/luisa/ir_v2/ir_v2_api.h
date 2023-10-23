#pragma once
#include <luisa/ir_v2/ir_v2_fwd.h>

#ifndef LC_IR_API
#ifdef LC_IR_EXPORT_DLL
#define LC_IR_API __declspec(dllexport)
#else
#define LC_IR_API __declspec(dllimport)
#endif
#endif
namespace luisa::compute::ir_v2 {
template<class T>
struct Slice {
    T *data = nullptr;
    size_t len = 0;
    constexpr Slice() noexcept = default;
    constexpr Slice(T *data, size_t len) noexcept : data(data), len(len) {}
#ifndef BINDGEN
    // construct from array
    template<size_t N>
    constexpr Slice(T (&arr)[N]) noexcept : data(arr), len(N) {}
    // construct from std::array
    template<size_t N>
    constexpr Slice(std::array<T, N> &arr) noexcept : data(arr.data()), len(N) {}
    // construct from luisa::vector
    constexpr Slice(luisa::vector<T> &vec) noexcept : data(vec.data()), len(vec.size()) {}
    // construct from luisa::span
    constexpr Slice(luisa::span<T> &span) noexcept : data(span.data()), len(span.size()) {}
    // construct from luisa::string
    constexpr Slice(luisa::string &str) noexcept : data(str.data()), len(str.size()) {
        static_assert(std::is_same_v<T, char> || std::is_same_v<T, const char>);
    }
#endif
};
}// namespace luisa::compute::ir_v2

namespace luisa::compute::ir_v2 {
extern "C" LC_IR_API Zero *lc_ir_v2_Func_as_Zero(Func *self);
extern "C" LC_IR_API One *lc_ir_v2_Func_as_One(Func *self);
extern "C" LC_IR_API Assume *lc_ir_v2_Func_as_Assume(Func *self);
extern "C" LC_IR_API Unreachable *lc_ir_v2_Func_as_Unreachable(Func *self);
extern "C" LC_IR_API ThreadId *lc_ir_v2_Func_as_ThreadId(Func *self);
extern "C" LC_IR_API BlockId *lc_ir_v2_Func_as_BlockId(Func *self);
extern "C" LC_IR_API WarpSize *lc_ir_v2_Func_as_WarpSize(Func *self);
extern "C" LC_IR_API WarpLaneId *lc_ir_v2_Func_as_WarpLaneId(Func *self);
extern "C" LC_IR_API DispatchId *lc_ir_v2_Func_as_DispatchId(Func *self);
extern "C" LC_IR_API DispatchSize *lc_ir_v2_Func_as_DispatchSize(Func *self);
extern "C" LC_IR_API PropagateGradient *lc_ir_v2_Func_as_PropagateGradient(Func *self);
extern "C" LC_IR_API OutputGradient *lc_ir_v2_Func_as_OutputGradient(Func *self);
extern "C" LC_IR_API RequiresGradient *lc_ir_v2_Func_as_RequiresGradient(Func *self);
extern "C" LC_IR_API Backward *lc_ir_v2_Func_as_Backward(Func *self);
extern "C" LC_IR_API Gradient *lc_ir_v2_Func_as_Gradient(Func *self);
extern "C" LC_IR_API AccGrad *lc_ir_v2_Func_as_AccGrad(Func *self);
extern "C" LC_IR_API Detach *lc_ir_v2_Func_as_Detach(Func *self);
extern "C" LC_IR_API RayTracingInstanceTransform *lc_ir_v2_Func_as_RayTracingInstanceTransform(Func *self);
extern "C" LC_IR_API RayTracingInstanceVisibilityMask *lc_ir_v2_Func_as_RayTracingInstanceVisibilityMask(Func *self);
extern "C" LC_IR_API RayTracingInstanceUserId *lc_ir_v2_Func_as_RayTracingInstanceUserId(Func *self);
extern "C" LC_IR_API RayTracingSetInstanceTransform *lc_ir_v2_Func_as_RayTracingSetInstanceTransform(Func *self);
extern "C" LC_IR_API RayTracingSetInstanceOpacity *lc_ir_v2_Func_as_RayTracingSetInstanceOpacity(Func *self);
extern "C" LC_IR_API RayTracingSetInstanceVisibility *lc_ir_v2_Func_as_RayTracingSetInstanceVisibility(Func *self);
extern "C" LC_IR_API RayTracingSetInstanceUserId *lc_ir_v2_Func_as_RayTracingSetInstanceUserId(Func *self);
extern "C" LC_IR_API RayTracingTraceClosest *lc_ir_v2_Func_as_RayTracingTraceClosest(Func *self);
extern "C" LC_IR_API RayTracingTraceAny *lc_ir_v2_Func_as_RayTracingTraceAny(Func *self);
extern "C" LC_IR_API RayTracingQueryAll *lc_ir_v2_Func_as_RayTracingQueryAll(Func *self);
extern "C" LC_IR_API RayTracingQueryAny *lc_ir_v2_Func_as_RayTracingQueryAny(Func *self);
extern "C" LC_IR_API RayQueryWorldSpaceRay *lc_ir_v2_Func_as_RayQueryWorldSpaceRay(Func *self);
extern "C" LC_IR_API RayQueryProceduralCandidateHit *lc_ir_v2_Func_as_RayQueryProceduralCandidateHit(Func *self);
extern "C" LC_IR_API RayQueryTriangleCandidateHit *lc_ir_v2_Func_as_RayQueryTriangleCandidateHit(Func *self);
extern "C" LC_IR_API RayQueryCommittedHit *lc_ir_v2_Func_as_RayQueryCommittedHit(Func *self);
extern "C" LC_IR_API RayQueryCommitTriangle *lc_ir_v2_Func_as_RayQueryCommitTriangle(Func *self);
extern "C" LC_IR_API RayQueryCommitdProcedural *lc_ir_v2_Func_as_RayQueryCommitdProcedural(Func *self);
extern "C" LC_IR_API RayQueryTerminate *lc_ir_v2_Func_as_RayQueryTerminate(Func *self);
extern "C" LC_IR_API Load *lc_ir_v2_Func_as_Load(Func *self);
extern "C" LC_IR_API Store *lc_ir_v2_Func_as_Store(Func *self);
extern "C" LC_IR_API Cast *lc_ir_v2_Func_as_Cast(Func *self);
extern "C" LC_IR_API BitCast *lc_ir_v2_Func_as_BitCast(Func *self);
extern "C" LC_IR_API Add *lc_ir_v2_Func_as_Add(Func *self);
extern "C" LC_IR_API Sub *lc_ir_v2_Func_as_Sub(Func *self);
extern "C" LC_IR_API Mul *lc_ir_v2_Func_as_Mul(Func *self);
extern "C" LC_IR_API Div *lc_ir_v2_Func_as_Div(Func *self);
extern "C" LC_IR_API Rem *lc_ir_v2_Func_as_Rem(Func *self);
extern "C" LC_IR_API BitAnd *lc_ir_v2_Func_as_BitAnd(Func *self);
extern "C" LC_IR_API BitOr *lc_ir_v2_Func_as_BitOr(Func *self);
extern "C" LC_IR_API BitXor *lc_ir_v2_Func_as_BitXor(Func *self);
extern "C" LC_IR_API Shl *lc_ir_v2_Func_as_Shl(Func *self);
extern "C" LC_IR_API Shr *lc_ir_v2_Func_as_Shr(Func *self);
extern "C" LC_IR_API RotRight *lc_ir_v2_Func_as_RotRight(Func *self);
extern "C" LC_IR_API RotLeft *lc_ir_v2_Func_as_RotLeft(Func *self);
extern "C" LC_IR_API Eq *lc_ir_v2_Func_as_Eq(Func *self);
extern "C" LC_IR_API Ne *lc_ir_v2_Func_as_Ne(Func *self);
extern "C" LC_IR_API Lt *lc_ir_v2_Func_as_Lt(Func *self);
extern "C" LC_IR_API Le *lc_ir_v2_Func_as_Le(Func *self);
extern "C" LC_IR_API Gt *lc_ir_v2_Func_as_Gt(Func *self);
extern "C" LC_IR_API Ge *lc_ir_v2_Func_as_Ge(Func *self);
extern "C" LC_IR_API MatCompMul *lc_ir_v2_Func_as_MatCompMul(Func *self);
extern "C" LC_IR_API Neg *lc_ir_v2_Func_as_Neg(Func *self);
extern "C" LC_IR_API Not *lc_ir_v2_Func_as_Not(Func *self);
extern "C" LC_IR_API BitNot *lc_ir_v2_Func_as_BitNot(Func *self);
extern "C" LC_IR_API All *lc_ir_v2_Func_as_All(Func *self);
extern "C" LC_IR_API Any *lc_ir_v2_Func_as_Any(Func *self);
extern "C" LC_IR_API Select *lc_ir_v2_Func_as_Select(Func *self);
extern "C" LC_IR_API Clamp *lc_ir_v2_Func_as_Clamp(Func *self);
extern "C" LC_IR_API Lerp *lc_ir_v2_Func_as_Lerp(Func *self);
extern "C" LC_IR_API Step *lc_ir_v2_Func_as_Step(Func *self);
extern "C" LC_IR_API Saturate *lc_ir_v2_Func_as_Saturate(Func *self);
extern "C" LC_IR_API SmoothStep *lc_ir_v2_Func_as_SmoothStep(Func *self);
extern "C" LC_IR_API Abs *lc_ir_v2_Func_as_Abs(Func *self);
extern "C" LC_IR_API Min *lc_ir_v2_Func_as_Min(Func *self);
extern "C" LC_IR_API Max *lc_ir_v2_Func_as_Max(Func *self);
extern "C" LC_IR_API ReduceSum *lc_ir_v2_Func_as_ReduceSum(Func *self);
extern "C" LC_IR_API ReduceProd *lc_ir_v2_Func_as_ReduceProd(Func *self);
extern "C" LC_IR_API ReduceMin *lc_ir_v2_Func_as_ReduceMin(Func *self);
extern "C" LC_IR_API ReduceMax *lc_ir_v2_Func_as_ReduceMax(Func *self);
extern "C" LC_IR_API Clz *lc_ir_v2_Func_as_Clz(Func *self);
extern "C" LC_IR_API Ctz *lc_ir_v2_Func_as_Ctz(Func *self);
extern "C" LC_IR_API PopCount *lc_ir_v2_Func_as_PopCount(Func *self);
extern "C" LC_IR_API Reverse *lc_ir_v2_Func_as_Reverse(Func *self);
extern "C" LC_IR_API IsInf *lc_ir_v2_Func_as_IsInf(Func *self);
extern "C" LC_IR_API IsNan *lc_ir_v2_Func_as_IsNan(Func *self);
extern "C" LC_IR_API Acos *lc_ir_v2_Func_as_Acos(Func *self);
extern "C" LC_IR_API Acosh *lc_ir_v2_Func_as_Acosh(Func *self);
extern "C" LC_IR_API Asin *lc_ir_v2_Func_as_Asin(Func *self);
extern "C" LC_IR_API Asinh *lc_ir_v2_Func_as_Asinh(Func *self);
extern "C" LC_IR_API Atan *lc_ir_v2_Func_as_Atan(Func *self);
extern "C" LC_IR_API Atan2 *lc_ir_v2_Func_as_Atan2(Func *self);
extern "C" LC_IR_API Atanh *lc_ir_v2_Func_as_Atanh(Func *self);
extern "C" LC_IR_API Cos *lc_ir_v2_Func_as_Cos(Func *self);
extern "C" LC_IR_API Cosh *lc_ir_v2_Func_as_Cosh(Func *self);
extern "C" LC_IR_API Sin *lc_ir_v2_Func_as_Sin(Func *self);
extern "C" LC_IR_API Sinh *lc_ir_v2_Func_as_Sinh(Func *self);
extern "C" LC_IR_API Tan *lc_ir_v2_Func_as_Tan(Func *self);
extern "C" LC_IR_API Tanh *lc_ir_v2_Func_as_Tanh(Func *self);
extern "C" LC_IR_API Exp *lc_ir_v2_Func_as_Exp(Func *self);
extern "C" LC_IR_API Exp2 *lc_ir_v2_Func_as_Exp2(Func *self);
extern "C" LC_IR_API Exp10 *lc_ir_v2_Func_as_Exp10(Func *self);
extern "C" LC_IR_API Log *lc_ir_v2_Func_as_Log(Func *self);
extern "C" LC_IR_API Log2 *lc_ir_v2_Func_as_Log2(Func *self);
extern "C" LC_IR_API Log10 *lc_ir_v2_Func_as_Log10(Func *self);
extern "C" LC_IR_API Powi *lc_ir_v2_Func_as_Powi(Func *self);
extern "C" LC_IR_API Powf *lc_ir_v2_Func_as_Powf(Func *self);
extern "C" LC_IR_API Sqrt *lc_ir_v2_Func_as_Sqrt(Func *self);
extern "C" LC_IR_API Rsqrt *lc_ir_v2_Func_as_Rsqrt(Func *self);
extern "C" LC_IR_API Ceil *lc_ir_v2_Func_as_Ceil(Func *self);
extern "C" LC_IR_API Floor *lc_ir_v2_Func_as_Floor(Func *self);
extern "C" LC_IR_API Fract *lc_ir_v2_Func_as_Fract(Func *self);
extern "C" LC_IR_API Trunc *lc_ir_v2_Func_as_Trunc(Func *self);
extern "C" LC_IR_API Round *lc_ir_v2_Func_as_Round(Func *self);
extern "C" LC_IR_API Fma *lc_ir_v2_Func_as_Fma(Func *self);
extern "C" LC_IR_API Copysign *lc_ir_v2_Func_as_Copysign(Func *self);
extern "C" LC_IR_API Cross *lc_ir_v2_Func_as_Cross(Func *self);
extern "C" LC_IR_API Dot *lc_ir_v2_Func_as_Dot(Func *self);
extern "C" LC_IR_API OuterProduct *lc_ir_v2_Func_as_OuterProduct(Func *self);
extern "C" LC_IR_API Length *lc_ir_v2_Func_as_Length(Func *self);
extern "C" LC_IR_API LengthSquared *lc_ir_v2_Func_as_LengthSquared(Func *self);
extern "C" LC_IR_API Normalize *lc_ir_v2_Func_as_Normalize(Func *self);
extern "C" LC_IR_API Faceforward *lc_ir_v2_Func_as_Faceforward(Func *self);
extern "C" LC_IR_API Distance *lc_ir_v2_Func_as_Distance(Func *self);
extern "C" LC_IR_API Reflect *lc_ir_v2_Func_as_Reflect(Func *self);
extern "C" LC_IR_API Determinant *lc_ir_v2_Func_as_Determinant(Func *self);
extern "C" LC_IR_API Transpose *lc_ir_v2_Func_as_Transpose(Func *self);
extern "C" LC_IR_API Inverse *lc_ir_v2_Func_as_Inverse(Func *self);
extern "C" LC_IR_API WarpIsFirstActiveLane *lc_ir_v2_Func_as_WarpIsFirstActiveLane(Func *self);
extern "C" LC_IR_API WarpFirstActiveLane *lc_ir_v2_Func_as_WarpFirstActiveLane(Func *self);
extern "C" LC_IR_API WarpActiveAllEqual *lc_ir_v2_Func_as_WarpActiveAllEqual(Func *self);
extern "C" LC_IR_API WarpActiveBitAnd *lc_ir_v2_Func_as_WarpActiveBitAnd(Func *self);
extern "C" LC_IR_API WarpActiveBitOr *lc_ir_v2_Func_as_WarpActiveBitOr(Func *self);
extern "C" LC_IR_API WarpActiveBitXor *lc_ir_v2_Func_as_WarpActiveBitXor(Func *self);
extern "C" LC_IR_API WarpActiveCountBits *lc_ir_v2_Func_as_WarpActiveCountBits(Func *self);
extern "C" LC_IR_API WarpActiveMax *lc_ir_v2_Func_as_WarpActiveMax(Func *self);
extern "C" LC_IR_API WarpActiveMin *lc_ir_v2_Func_as_WarpActiveMin(Func *self);
extern "C" LC_IR_API WarpActiveProduct *lc_ir_v2_Func_as_WarpActiveProduct(Func *self);
extern "C" LC_IR_API WarpActiveSum *lc_ir_v2_Func_as_WarpActiveSum(Func *self);
extern "C" LC_IR_API WarpActiveAll *lc_ir_v2_Func_as_WarpActiveAll(Func *self);
extern "C" LC_IR_API WarpActiveAny *lc_ir_v2_Func_as_WarpActiveAny(Func *self);
extern "C" LC_IR_API WarpActiveBitMask *lc_ir_v2_Func_as_WarpActiveBitMask(Func *self);
extern "C" LC_IR_API WarpPrefixCountBits *lc_ir_v2_Func_as_WarpPrefixCountBits(Func *self);
extern "C" LC_IR_API WarpPrefixSum *lc_ir_v2_Func_as_WarpPrefixSum(Func *self);
extern "C" LC_IR_API WarpPrefixProduct *lc_ir_v2_Func_as_WarpPrefixProduct(Func *self);
extern "C" LC_IR_API WarpReadLaneAt *lc_ir_v2_Func_as_WarpReadLaneAt(Func *self);
extern "C" LC_IR_API WarpReadFirstLane *lc_ir_v2_Func_as_WarpReadFirstLane(Func *self);
extern "C" LC_IR_API SynchronizeBlock *lc_ir_v2_Func_as_SynchronizeBlock(Func *self);
extern "C" LC_IR_API AtomicExchange *lc_ir_v2_Func_as_AtomicExchange(Func *self);
extern "C" LC_IR_API AtomicCompareExchange *lc_ir_v2_Func_as_AtomicCompareExchange(Func *self);
extern "C" LC_IR_API AtomicFetchAdd *lc_ir_v2_Func_as_AtomicFetchAdd(Func *self);
extern "C" LC_IR_API AtomicFetchSub *lc_ir_v2_Func_as_AtomicFetchSub(Func *self);
extern "C" LC_IR_API AtomicFetchAnd *lc_ir_v2_Func_as_AtomicFetchAnd(Func *self);
extern "C" LC_IR_API AtomicFetchOr *lc_ir_v2_Func_as_AtomicFetchOr(Func *self);
extern "C" LC_IR_API AtomicFetchXor *lc_ir_v2_Func_as_AtomicFetchXor(Func *self);
extern "C" LC_IR_API AtomicFetchMin *lc_ir_v2_Func_as_AtomicFetchMin(Func *self);
extern "C" LC_IR_API AtomicFetchMax *lc_ir_v2_Func_as_AtomicFetchMax(Func *self);
extern "C" LC_IR_API BufferWrite *lc_ir_v2_Func_as_BufferWrite(Func *self);
extern "C" LC_IR_API BufferRead *lc_ir_v2_Func_as_BufferRead(Func *self);
extern "C" LC_IR_API BufferSize *lc_ir_v2_Func_as_BufferSize(Func *self);
extern "C" LC_IR_API ByteBufferWrite *lc_ir_v2_Func_as_ByteBufferWrite(Func *self);
extern "C" LC_IR_API ByteBufferRead *lc_ir_v2_Func_as_ByteBufferRead(Func *self);
extern "C" LC_IR_API ByteBufferSize *lc_ir_v2_Func_as_ByteBufferSize(Func *self);
extern "C" LC_IR_API Texture2dRead *lc_ir_v2_Func_as_Texture2dRead(Func *self);
extern "C" LC_IR_API Texture2dWrite *lc_ir_v2_Func_as_Texture2dWrite(Func *self);
extern "C" LC_IR_API Texture2dSize *lc_ir_v2_Func_as_Texture2dSize(Func *self);
extern "C" LC_IR_API Texture3dRead *lc_ir_v2_Func_as_Texture3dRead(Func *self);
extern "C" LC_IR_API Texture3dWrite *lc_ir_v2_Func_as_Texture3dWrite(Func *self);
extern "C" LC_IR_API Texture3dSize *lc_ir_v2_Func_as_Texture3dSize(Func *self);
extern "C" LC_IR_API BindlessTexture2dSample *lc_ir_v2_Func_as_BindlessTexture2dSample(Func *self);
extern "C" LC_IR_API BindlessTexture2dSampleLevel *lc_ir_v2_Func_as_BindlessTexture2dSampleLevel(Func *self);
extern "C" LC_IR_API BindlessTexture2dSampleGrad *lc_ir_v2_Func_as_BindlessTexture2dSampleGrad(Func *self);
extern "C" LC_IR_API BindlessTexture2dSampleGradLevel *lc_ir_v2_Func_as_BindlessTexture2dSampleGradLevel(Func *self);
extern "C" LC_IR_API BindlessTexture2dRead *lc_ir_v2_Func_as_BindlessTexture2dRead(Func *self);
extern "C" LC_IR_API BindlessTexture2dSize *lc_ir_v2_Func_as_BindlessTexture2dSize(Func *self);
extern "C" LC_IR_API BindlessTexture2dSizeLevel *lc_ir_v2_Func_as_BindlessTexture2dSizeLevel(Func *self);
extern "C" LC_IR_API BindlessTexture3dSample *lc_ir_v2_Func_as_BindlessTexture3dSample(Func *self);
extern "C" LC_IR_API BindlessTexture3dSampleLevel *lc_ir_v2_Func_as_BindlessTexture3dSampleLevel(Func *self);
extern "C" LC_IR_API BindlessTexture3dSampleGrad *lc_ir_v2_Func_as_BindlessTexture3dSampleGrad(Func *self);
extern "C" LC_IR_API BindlessTexture3dSampleGradLevel *lc_ir_v2_Func_as_BindlessTexture3dSampleGradLevel(Func *self);
extern "C" LC_IR_API BindlessTexture3dRead *lc_ir_v2_Func_as_BindlessTexture3dRead(Func *self);
extern "C" LC_IR_API BindlessTexture3dSize *lc_ir_v2_Func_as_BindlessTexture3dSize(Func *self);
extern "C" LC_IR_API BindlessTexture3dSizeLevel *lc_ir_v2_Func_as_BindlessTexture3dSizeLevel(Func *self);
extern "C" LC_IR_API BindlessBufferWrite *lc_ir_v2_Func_as_BindlessBufferWrite(Func *self);
extern "C" LC_IR_API BindlessBufferRead *lc_ir_v2_Func_as_BindlessBufferRead(Func *self);
extern "C" LC_IR_API BindlessBufferSize *lc_ir_v2_Func_as_BindlessBufferSize(Func *self);
extern "C" LC_IR_API BindlessByteBufferWrite *lc_ir_v2_Func_as_BindlessByteBufferWrite(Func *self);
extern "C" LC_IR_API BindlessByteBufferRead *lc_ir_v2_Func_as_BindlessByteBufferRead(Func *self);
extern "C" LC_IR_API BindlessByteBufferSize *lc_ir_v2_Func_as_BindlessByteBufferSize(Func *self);
extern "C" LC_IR_API Vec *lc_ir_v2_Func_as_Vec(Func *self);
extern "C" LC_IR_API Vec2 *lc_ir_v2_Func_as_Vec2(Func *self);
extern "C" LC_IR_API Vec3 *lc_ir_v2_Func_as_Vec3(Func *self);
extern "C" LC_IR_API Vec4 *lc_ir_v2_Func_as_Vec4(Func *self);
extern "C" LC_IR_API Permute *lc_ir_v2_Func_as_Permute(Func *self);
extern "C" LC_IR_API GetElementPtr *lc_ir_v2_Func_as_GetElementPtr(Func *self);
extern "C" LC_IR_API ExtractElement *lc_ir_v2_Func_as_ExtractElement(Func *self);
extern "C" LC_IR_API InsertElement *lc_ir_v2_Func_as_InsertElement(Func *self);
extern "C" LC_IR_API Array *lc_ir_v2_Func_as_Array(Func *self);
extern "C" LC_IR_API Struct *lc_ir_v2_Func_as_Struct(Func *self);
extern "C" LC_IR_API MatFull *lc_ir_v2_Func_as_MatFull(Func *self);
extern "C" LC_IR_API Mat2 *lc_ir_v2_Func_as_Mat2(Func *self);
extern "C" LC_IR_API Mat3 *lc_ir_v2_Func_as_Mat3(Func *self);
extern "C" LC_IR_API Mat4 *lc_ir_v2_Func_as_Mat4(Func *self);
extern "C" LC_IR_API BindlessAtomicFetchAdd *lc_ir_v2_Func_as_BindlessAtomicFetchAdd(Func *self);
extern "C" LC_IR_API BindlessAtomicFetchSub *lc_ir_v2_Func_as_BindlessAtomicFetchSub(Func *self);
extern "C" LC_IR_API BindlessAtomicFetchAnd *lc_ir_v2_Func_as_BindlessAtomicFetchAnd(Func *self);
extern "C" LC_IR_API BindlessAtomicFetchOr *lc_ir_v2_Func_as_BindlessAtomicFetchOr(Func *self);
extern "C" LC_IR_API BindlessAtomicFetchXor *lc_ir_v2_Func_as_BindlessAtomicFetchXor(Func *self);
extern "C" LC_IR_API BindlessAtomicFetchMin *lc_ir_v2_Func_as_BindlessAtomicFetchMin(Func *self);
extern "C" LC_IR_API BindlessAtomicFetchMax *lc_ir_v2_Func_as_BindlessAtomicFetchMax(Func *self);
extern "C" LC_IR_API Callable *lc_ir_v2_Func_as_Callable(Func *self);
extern "C" LC_IR_API CpuExt *lc_ir_v2_Func_as_CpuExt(Func *self);
extern "C" LC_IR_API ShaderExecutionReorder *lc_ir_v2_Func_as_ShaderExecutionReorder(Func *self);
extern "C" LC_IR_API FuncTag lc_ir_v2_Func_tag(Func *self);
extern "C" LC_IR_API void lc_ir_v2_Assume_msg(Assume *self, Slice<const char> *out);
extern "C" LC_IR_API void lc_ir_v2_Unreachable_msg(Unreachable *self, Slice<const char> *out);
extern "C" LC_IR_API void lc_ir_v2_BindlessAtomicFetchAdd_ty(BindlessAtomicFetchAdd *self, const Type **out);
extern "C" LC_IR_API void lc_ir_v2_BindlessAtomicFetchSub_ty(BindlessAtomicFetchSub *self, const Type **out);
extern "C" LC_IR_API void lc_ir_v2_BindlessAtomicFetchAnd_ty(BindlessAtomicFetchAnd *self, const Type **out);
extern "C" LC_IR_API void lc_ir_v2_BindlessAtomicFetchOr_ty(BindlessAtomicFetchOr *self, const Type **out);
extern "C" LC_IR_API void lc_ir_v2_BindlessAtomicFetchXor_ty(BindlessAtomicFetchXor *self, const Type **out);
extern "C" LC_IR_API void lc_ir_v2_BindlessAtomicFetchMin_ty(BindlessAtomicFetchMin *self, const Type **out);
extern "C" LC_IR_API void lc_ir_v2_BindlessAtomicFetchMax_ty(BindlessAtomicFetchMax *self, const Type **out);
extern "C" LC_IR_API void lc_ir_v2_Callable_module(Callable *self, CallableModule **out);
extern "C" LC_IR_API void lc_ir_v2_CpuExt_f(CpuExt *self, CpuExternFn *out);
extern "C" LC_IR_API Buffer *lc_ir_v2_Instruction_as_Buffer(Instruction *self);
extern "C" LC_IR_API Texture2d *lc_ir_v2_Instruction_as_Texture2d(Instruction *self);
extern "C" LC_IR_API Texture3d *lc_ir_v2_Instruction_as_Texture3d(Instruction *self);
extern "C" LC_IR_API BindlessArray *lc_ir_v2_Instruction_as_BindlessArray(Instruction *self);
extern "C" LC_IR_API Accel *lc_ir_v2_Instruction_as_Accel(Instruction *self);
extern "C" LC_IR_API Shared *lc_ir_v2_Instruction_as_Shared(Instruction *self);
extern "C" LC_IR_API Uniform *lc_ir_v2_Instruction_as_Uniform(Instruction *self);
extern "C" LC_IR_API Argument *lc_ir_v2_Instruction_as_Argument(Instruction *self);
extern "C" LC_IR_API Const *lc_ir_v2_Instruction_as_Const(Instruction *self);
extern "C" LC_IR_API Call *lc_ir_v2_Instruction_as_Call(Instruction *self);
extern "C" LC_IR_API Phi *lc_ir_v2_Instruction_as_Phi(Instruction *self);
extern "C" LC_IR_API BasicBlockSentinel *lc_ir_v2_Instruction_as_BasicBlockSentinel(Instruction *self);
extern "C" LC_IR_API If *lc_ir_v2_Instruction_as_If(Instruction *self);
extern "C" LC_IR_API GenericLoop *lc_ir_v2_Instruction_as_GenericLoop(Instruction *self);
extern "C" LC_IR_API Switch *lc_ir_v2_Instruction_as_Switch(Instruction *self);
extern "C" LC_IR_API Local *lc_ir_v2_Instruction_as_Local(Instruction *self);
extern "C" LC_IR_API Break *lc_ir_v2_Instruction_as_Break(Instruction *self);
extern "C" LC_IR_API Continue *lc_ir_v2_Instruction_as_Continue(Instruction *self);
extern "C" LC_IR_API Return *lc_ir_v2_Instruction_as_Return(Instruction *self);
extern "C" LC_IR_API Print *lc_ir_v2_Instruction_as_Print(Instruction *self);
extern "C" LC_IR_API InstructionTag lc_ir_v2_Instruction_tag(Instruction *self);
extern "C" LC_IR_API void lc_ir_v2_Argument_by_value(Argument *self, bool *out);
extern "C" LC_IR_API void lc_ir_v2_Const_ty(Const *self, const Type **out);
extern "C" LC_IR_API void lc_ir_v2_Const_value(Const *self, Slice<uint8_t> *out);
extern "C" LC_IR_API void lc_ir_v2_Call_func(Call *self, const Func **out);
extern "C" LC_IR_API void lc_ir_v2_Call_args(Call *self, Slice<Node *> *out);
extern "C" LC_IR_API void lc_ir_v2_Phi_incomings(Phi *self, Slice<PhiIncoming> *out);
extern "C" LC_IR_API void lc_ir_v2_If_cond(If *self, Node **out);
extern "C" LC_IR_API void lc_ir_v2_If_true_branch(If *self, BasicBlock **out);
extern "C" LC_IR_API void lc_ir_v2_If_false_branch(If *self, BasicBlock **out);
extern "C" LC_IR_API void lc_ir_v2_GenericLoop_prepare(GenericLoop *self, BasicBlock **out);
extern "C" LC_IR_API void lc_ir_v2_GenericLoop_cond(GenericLoop *self, Node **out);
extern "C" LC_IR_API void lc_ir_v2_GenericLoop_body(GenericLoop *self, BasicBlock **out);
extern "C" LC_IR_API void lc_ir_v2_GenericLoop_update(GenericLoop *self, BasicBlock **out);
extern "C" LC_IR_API void lc_ir_v2_Switch_value(Switch *self, Node **out);
extern "C" LC_IR_API void lc_ir_v2_Switch_cases(Switch *self, Slice<SwitchCase> *out);
extern "C" LC_IR_API void lc_ir_v2_Switch_default_(Switch *self, BasicBlock **out);
extern "C" LC_IR_API void lc_ir_v2_Local_init(Local *self, Node **out);
extern "C" LC_IR_API void lc_ir_v2_Return_value(Return *self, Node **out);
extern "C" LC_IR_API void lc_ir_v2_Print_fmt(Print *self, Slice<const char> *out);
extern "C" LC_IR_API void lc_ir_v2_Print_args(Print *self, Slice<Node *> *out);
extern "C" LC_IR_API BufferBinding *lc_ir_v2_Binding_as_BufferBinding(Binding *self);
extern "C" LC_IR_API TextureBinding *lc_ir_v2_Binding_as_TextureBinding(Binding *self);
extern "C" LC_IR_API BindlessArrayBinding *lc_ir_v2_Binding_as_BindlessArrayBinding(Binding *self);
extern "C" LC_IR_API AccelBinding *lc_ir_v2_Binding_as_AccelBinding(Binding *self);
extern "C" LC_IR_API BindingTag lc_ir_v2_Binding_tag(Binding *self);
extern "C" LC_IR_API void lc_ir_v2_BufferBinding_handle(BufferBinding *self, uint64_t *out);
extern "C" LC_IR_API void lc_ir_v2_BufferBinding_offset(BufferBinding *self, uint64_t *out);
extern "C" LC_IR_API void lc_ir_v2_BufferBinding_size(BufferBinding *self, uint64_t *out);
extern "C" LC_IR_API void lc_ir_v2_TextureBinding_handle(TextureBinding *self, uint64_t *out);
extern "C" LC_IR_API void lc_ir_v2_TextureBinding_level(TextureBinding *self, uint64_t *out);
extern "C" LC_IR_API void lc_ir_v2_BindlessArrayBinding_handle(BindlessArrayBinding *self, uint64_t *out);
extern "C" LC_IR_API void lc_ir_v2_AccelBinding_handle(AccelBinding *self, uint64_t *out);
}// namespace luisa::compute::ir_v2
