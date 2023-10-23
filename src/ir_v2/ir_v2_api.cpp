#include <luisa/ir_v2/ir_v2_api.h>
namespace luisa::compute::ir_v2 {
extern "C" LC_IR_API Zero *lc_ir_v2_Func_as_Zero(Func *self) {
    return self->as<Zero>();
}
extern "C" LC_IR_API One *lc_ir_v2_Func_as_One(Func *self) {
    return self->as<One>();
}
extern "C" LC_IR_API Assume *lc_ir_v2_Func_as_Assume(Func *self) {
    return self->as<Assume>();
}
extern "C" LC_IR_API Unreachable *lc_ir_v2_Func_as_Unreachable(Func *self) {
    return self->as<Unreachable>();
}
extern "C" LC_IR_API ThreadId *lc_ir_v2_Func_as_ThreadId(Func *self) {
    return self->as<ThreadId>();
}
extern "C" LC_IR_API BlockId *lc_ir_v2_Func_as_BlockId(Func *self) {
    return self->as<BlockId>();
}
extern "C" LC_IR_API WarpSize *lc_ir_v2_Func_as_WarpSize(Func *self) {
    return self->as<WarpSize>();
}
extern "C" LC_IR_API WarpLaneId *lc_ir_v2_Func_as_WarpLaneId(Func *self) {
    return self->as<WarpLaneId>();
}
extern "C" LC_IR_API DispatchId *lc_ir_v2_Func_as_DispatchId(Func *self) {
    return self->as<DispatchId>();
}
extern "C" LC_IR_API DispatchSize *lc_ir_v2_Func_as_DispatchSize(Func *self) {
    return self->as<DispatchSize>();
}
extern "C" LC_IR_API PropagateGradient *lc_ir_v2_Func_as_PropagateGradient(Func *self) {
    return self->as<PropagateGradient>();
}
extern "C" LC_IR_API OutputGradient *lc_ir_v2_Func_as_OutputGradient(Func *self) {
    return self->as<OutputGradient>();
}
extern "C" LC_IR_API RequiresGradient *lc_ir_v2_Func_as_RequiresGradient(Func *self) {
    return self->as<RequiresGradient>();
}
extern "C" LC_IR_API Backward *lc_ir_v2_Func_as_Backward(Func *self) {
    return self->as<Backward>();
}
extern "C" LC_IR_API Gradient *lc_ir_v2_Func_as_Gradient(Func *self) {
    return self->as<Gradient>();
}
extern "C" LC_IR_API AccGrad *lc_ir_v2_Func_as_AccGrad(Func *self) {
    return self->as<AccGrad>();
}
extern "C" LC_IR_API Detach *lc_ir_v2_Func_as_Detach(Func *self) {
    return self->as<Detach>();
}
extern "C" LC_IR_API RayTracingInstanceTransform *lc_ir_v2_Func_as_RayTracingInstanceTransform(Func *self) {
    return self->as<RayTracingInstanceTransform>();
}
extern "C" LC_IR_API RayTracingInstanceVisibilityMask *lc_ir_v2_Func_as_RayTracingInstanceVisibilityMask(Func *self) {
    return self->as<RayTracingInstanceVisibilityMask>();
}
extern "C" LC_IR_API RayTracingInstanceUserId *lc_ir_v2_Func_as_RayTracingInstanceUserId(Func *self) {
    return self->as<RayTracingInstanceUserId>();
}
extern "C" LC_IR_API RayTracingSetInstanceTransform *lc_ir_v2_Func_as_RayTracingSetInstanceTransform(Func *self) {
    return self->as<RayTracingSetInstanceTransform>();
}
extern "C" LC_IR_API RayTracingSetInstanceOpacity *lc_ir_v2_Func_as_RayTracingSetInstanceOpacity(Func *self) {
    return self->as<RayTracingSetInstanceOpacity>();
}
extern "C" LC_IR_API RayTracingSetInstanceVisibility *lc_ir_v2_Func_as_RayTracingSetInstanceVisibility(Func *self) {
    return self->as<RayTracingSetInstanceVisibility>();
}
extern "C" LC_IR_API RayTracingSetInstanceUserId *lc_ir_v2_Func_as_RayTracingSetInstanceUserId(Func *self) {
    return self->as<RayTracingSetInstanceUserId>();
}
extern "C" LC_IR_API RayTracingTraceClosest *lc_ir_v2_Func_as_RayTracingTraceClosest(Func *self) {
    return self->as<RayTracingTraceClosest>();
}
extern "C" LC_IR_API RayTracingTraceAny *lc_ir_v2_Func_as_RayTracingTraceAny(Func *self) {
    return self->as<RayTracingTraceAny>();
}
extern "C" LC_IR_API RayTracingQueryAll *lc_ir_v2_Func_as_RayTracingQueryAll(Func *self) {
    return self->as<RayTracingQueryAll>();
}
extern "C" LC_IR_API RayTracingQueryAny *lc_ir_v2_Func_as_RayTracingQueryAny(Func *self) {
    return self->as<RayTracingQueryAny>();
}
extern "C" LC_IR_API RayQueryWorldSpaceRay *lc_ir_v2_Func_as_RayQueryWorldSpaceRay(Func *self) {
    return self->as<RayQueryWorldSpaceRay>();
}
extern "C" LC_IR_API RayQueryProceduralCandidateHit *lc_ir_v2_Func_as_RayQueryProceduralCandidateHit(Func *self) {
    return self->as<RayQueryProceduralCandidateHit>();
}
extern "C" LC_IR_API RayQueryTriangleCandidateHit *lc_ir_v2_Func_as_RayQueryTriangleCandidateHit(Func *self) {
    return self->as<RayQueryTriangleCandidateHit>();
}
extern "C" LC_IR_API RayQueryCommittedHit *lc_ir_v2_Func_as_RayQueryCommittedHit(Func *self) {
    return self->as<RayQueryCommittedHit>();
}
extern "C" LC_IR_API RayQueryCommitTriangle *lc_ir_v2_Func_as_RayQueryCommitTriangle(Func *self) {
    return self->as<RayQueryCommitTriangle>();
}
extern "C" LC_IR_API RayQueryCommitdProcedural *lc_ir_v2_Func_as_RayQueryCommitdProcedural(Func *self) {
    return self->as<RayQueryCommitdProcedural>();
}
extern "C" LC_IR_API RayQueryTerminate *lc_ir_v2_Func_as_RayQueryTerminate(Func *self) {
    return self->as<RayQueryTerminate>();
}
extern "C" LC_IR_API Load *lc_ir_v2_Func_as_Load(Func *self) {
    return self->as<Load>();
}
extern "C" LC_IR_API Store *lc_ir_v2_Func_as_Store(Func *self) {
    return self->as<Store>();
}
extern "C" LC_IR_API Cast *lc_ir_v2_Func_as_Cast(Func *self) {
    return self->as<Cast>();
}
extern "C" LC_IR_API BitCast *lc_ir_v2_Func_as_BitCast(Func *self) {
    return self->as<BitCast>();
}
extern "C" LC_IR_API Add *lc_ir_v2_Func_as_Add(Func *self) {
    return self->as<Add>();
}
extern "C" LC_IR_API Sub *lc_ir_v2_Func_as_Sub(Func *self) {
    return self->as<Sub>();
}
extern "C" LC_IR_API Mul *lc_ir_v2_Func_as_Mul(Func *self) {
    return self->as<Mul>();
}
extern "C" LC_IR_API Div *lc_ir_v2_Func_as_Div(Func *self) {
    return self->as<Div>();
}
extern "C" LC_IR_API Rem *lc_ir_v2_Func_as_Rem(Func *self) {
    return self->as<Rem>();
}
extern "C" LC_IR_API BitAnd *lc_ir_v2_Func_as_BitAnd(Func *self) {
    return self->as<BitAnd>();
}
extern "C" LC_IR_API BitOr *lc_ir_v2_Func_as_BitOr(Func *self) {
    return self->as<BitOr>();
}
extern "C" LC_IR_API BitXor *lc_ir_v2_Func_as_BitXor(Func *self) {
    return self->as<BitXor>();
}
extern "C" LC_IR_API Shl *lc_ir_v2_Func_as_Shl(Func *self) {
    return self->as<Shl>();
}
extern "C" LC_IR_API Shr *lc_ir_v2_Func_as_Shr(Func *self) {
    return self->as<Shr>();
}
extern "C" LC_IR_API RotRight *lc_ir_v2_Func_as_RotRight(Func *self) {
    return self->as<RotRight>();
}
extern "C" LC_IR_API RotLeft *lc_ir_v2_Func_as_RotLeft(Func *self) {
    return self->as<RotLeft>();
}
extern "C" LC_IR_API Eq *lc_ir_v2_Func_as_Eq(Func *self) {
    return self->as<Eq>();
}
extern "C" LC_IR_API Ne *lc_ir_v2_Func_as_Ne(Func *self) {
    return self->as<Ne>();
}
extern "C" LC_IR_API Lt *lc_ir_v2_Func_as_Lt(Func *self) {
    return self->as<Lt>();
}
extern "C" LC_IR_API Le *lc_ir_v2_Func_as_Le(Func *self) {
    return self->as<Le>();
}
extern "C" LC_IR_API Gt *lc_ir_v2_Func_as_Gt(Func *self) {
    return self->as<Gt>();
}
extern "C" LC_IR_API Ge *lc_ir_v2_Func_as_Ge(Func *self) {
    return self->as<Ge>();
}
extern "C" LC_IR_API MatCompMul *lc_ir_v2_Func_as_MatCompMul(Func *self) {
    return self->as<MatCompMul>();
}
extern "C" LC_IR_API Neg *lc_ir_v2_Func_as_Neg(Func *self) {
    return self->as<Neg>();
}
extern "C" LC_IR_API Not *lc_ir_v2_Func_as_Not(Func *self) {
    return self->as<Not>();
}
extern "C" LC_IR_API BitNot *lc_ir_v2_Func_as_BitNot(Func *self) {
    return self->as<BitNot>();
}
extern "C" LC_IR_API All *lc_ir_v2_Func_as_All(Func *self) {
    return self->as<All>();
}
extern "C" LC_IR_API Any *lc_ir_v2_Func_as_Any(Func *self) {
    return self->as<Any>();
}
extern "C" LC_IR_API Select *lc_ir_v2_Func_as_Select(Func *self) {
    return self->as<Select>();
}
extern "C" LC_IR_API Clamp *lc_ir_v2_Func_as_Clamp(Func *self) {
    return self->as<Clamp>();
}
extern "C" LC_IR_API Lerp *lc_ir_v2_Func_as_Lerp(Func *self) {
    return self->as<Lerp>();
}
extern "C" LC_IR_API Step *lc_ir_v2_Func_as_Step(Func *self) {
    return self->as<Step>();
}
extern "C" LC_IR_API Saturate *lc_ir_v2_Func_as_Saturate(Func *self) {
    return self->as<Saturate>();
}
extern "C" LC_IR_API SmoothStep *lc_ir_v2_Func_as_SmoothStep(Func *self) {
    return self->as<SmoothStep>();
}
extern "C" LC_IR_API Abs *lc_ir_v2_Func_as_Abs(Func *self) {
    return self->as<Abs>();
}
extern "C" LC_IR_API Min *lc_ir_v2_Func_as_Min(Func *self) {
    return self->as<Min>();
}
extern "C" LC_IR_API Max *lc_ir_v2_Func_as_Max(Func *self) {
    return self->as<Max>();
}
extern "C" LC_IR_API ReduceSum *lc_ir_v2_Func_as_ReduceSum(Func *self) {
    return self->as<ReduceSum>();
}
extern "C" LC_IR_API ReduceProd *lc_ir_v2_Func_as_ReduceProd(Func *self) {
    return self->as<ReduceProd>();
}
extern "C" LC_IR_API ReduceMin *lc_ir_v2_Func_as_ReduceMin(Func *self) {
    return self->as<ReduceMin>();
}
extern "C" LC_IR_API ReduceMax *lc_ir_v2_Func_as_ReduceMax(Func *self) {
    return self->as<ReduceMax>();
}
extern "C" LC_IR_API Clz *lc_ir_v2_Func_as_Clz(Func *self) {
    return self->as<Clz>();
}
extern "C" LC_IR_API Ctz *lc_ir_v2_Func_as_Ctz(Func *self) {
    return self->as<Ctz>();
}
extern "C" LC_IR_API PopCount *lc_ir_v2_Func_as_PopCount(Func *self) {
    return self->as<PopCount>();
}
extern "C" LC_IR_API Reverse *lc_ir_v2_Func_as_Reverse(Func *self) {
    return self->as<Reverse>();
}
extern "C" LC_IR_API IsInf *lc_ir_v2_Func_as_IsInf(Func *self) {
    return self->as<IsInf>();
}
extern "C" LC_IR_API IsNan *lc_ir_v2_Func_as_IsNan(Func *self) {
    return self->as<IsNan>();
}
extern "C" LC_IR_API Acos *lc_ir_v2_Func_as_Acos(Func *self) {
    return self->as<Acos>();
}
extern "C" LC_IR_API Acosh *lc_ir_v2_Func_as_Acosh(Func *self) {
    return self->as<Acosh>();
}
extern "C" LC_IR_API Asin *lc_ir_v2_Func_as_Asin(Func *self) {
    return self->as<Asin>();
}
extern "C" LC_IR_API Asinh *lc_ir_v2_Func_as_Asinh(Func *self) {
    return self->as<Asinh>();
}
extern "C" LC_IR_API Atan *lc_ir_v2_Func_as_Atan(Func *self) {
    return self->as<Atan>();
}
extern "C" LC_IR_API Atan2 *lc_ir_v2_Func_as_Atan2(Func *self) {
    return self->as<Atan2>();
}
extern "C" LC_IR_API Atanh *lc_ir_v2_Func_as_Atanh(Func *self) {
    return self->as<Atanh>();
}
extern "C" LC_IR_API Cos *lc_ir_v2_Func_as_Cos(Func *self) {
    return self->as<Cos>();
}
extern "C" LC_IR_API Cosh *lc_ir_v2_Func_as_Cosh(Func *self) {
    return self->as<Cosh>();
}
extern "C" LC_IR_API Sin *lc_ir_v2_Func_as_Sin(Func *self) {
    return self->as<Sin>();
}
extern "C" LC_IR_API Sinh *lc_ir_v2_Func_as_Sinh(Func *self) {
    return self->as<Sinh>();
}
extern "C" LC_IR_API Tan *lc_ir_v2_Func_as_Tan(Func *self) {
    return self->as<Tan>();
}
extern "C" LC_IR_API Tanh *lc_ir_v2_Func_as_Tanh(Func *self) {
    return self->as<Tanh>();
}
extern "C" LC_IR_API Exp *lc_ir_v2_Func_as_Exp(Func *self) {
    return self->as<Exp>();
}
extern "C" LC_IR_API Exp2 *lc_ir_v2_Func_as_Exp2(Func *self) {
    return self->as<Exp2>();
}
extern "C" LC_IR_API Exp10 *lc_ir_v2_Func_as_Exp10(Func *self) {
    return self->as<Exp10>();
}
extern "C" LC_IR_API Log *lc_ir_v2_Func_as_Log(Func *self) {
    return self->as<Log>();
}
extern "C" LC_IR_API Log2 *lc_ir_v2_Func_as_Log2(Func *self) {
    return self->as<Log2>();
}
extern "C" LC_IR_API Log10 *lc_ir_v2_Func_as_Log10(Func *self) {
    return self->as<Log10>();
}
extern "C" LC_IR_API Powi *lc_ir_v2_Func_as_Powi(Func *self) {
    return self->as<Powi>();
}
extern "C" LC_IR_API Powf *lc_ir_v2_Func_as_Powf(Func *self) {
    return self->as<Powf>();
}
extern "C" LC_IR_API Sqrt *lc_ir_v2_Func_as_Sqrt(Func *self) {
    return self->as<Sqrt>();
}
extern "C" LC_IR_API Rsqrt *lc_ir_v2_Func_as_Rsqrt(Func *self) {
    return self->as<Rsqrt>();
}
extern "C" LC_IR_API Ceil *lc_ir_v2_Func_as_Ceil(Func *self) {
    return self->as<Ceil>();
}
extern "C" LC_IR_API Floor *lc_ir_v2_Func_as_Floor(Func *self) {
    return self->as<Floor>();
}
extern "C" LC_IR_API Fract *lc_ir_v2_Func_as_Fract(Func *self) {
    return self->as<Fract>();
}
extern "C" LC_IR_API Trunc *lc_ir_v2_Func_as_Trunc(Func *self) {
    return self->as<Trunc>();
}
extern "C" LC_IR_API Round *lc_ir_v2_Func_as_Round(Func *self) {
    return self->as<Round>();
}
extern "C" LC_IR_API Fma *lc_ir_v2_Func_as_Fma(Func *self) {
    return self->as<Fma>();
}
extern "C" LC_IR_API Copysign *lc_ir_v2_Func_as_Copysign(Func *self) {
    return self->as<Copysign>();
}
extern "C" LC_IR_API Cross *lc_ir_v2_Func_as_Cross(Func *self) {
    return self->as<Cross>();
}
extern "C" LC_IR_API Dot *lc_ir_v2_Func_as_Dot(Func *self) {
    return self->as<Dot>();
}
extern "C" LC_IR_API OuterProduct *lc_ir_v2_Func_as_OuterProduct(Func *self) {
    return self->as<OuterProduct>();
}
extern "C" LC_IR_API Length *lc_ir_v2_Func_as_Length(Func *self) {
    return self->as<Length>();
}
extern "C" LC_IR_API LengthSquared *lc_ir_v2_Func_as_LengthSquared(Func *self) {
    return self->as<LengthSquared>();
}
extern "C" LC_IR_API Normalize *lc_ir_v2_Func_as_Normalize(Func *self) {
    return self->as<Normalize>();
}
extern "C" LC_IR_API Faceforward *lc_ir_v2_Func_as_Faceforward(Func *self) {
    return self->as<Faceforward>();
}
extern "C" LC_IR_API Distance *lc_ir_v2_Func_as_Distance(Func *self) {
    return self->as<Distance>();
}
extern "C" LC_IR_API Reflect *lc_ir_v2_Func_as_Reflect(Func *self) {
    return self->as<Reflect>();
}
extern "C" LC_IR_API Determinant *lc_ir_v2_Func_as_Determinant(Func *self) {
    return self->as<Determinant>();
}
extern "C" LC_IR_API Transpose *lc_ir_v2_Func_as_Transpose(Func *self) {
    return self->as<Transpose>();
}
extern "C" LC_IR_API Inverse *lc_ir_v2_Func_as_Inverse(Func *self) {
    return self->as<Inverse>();
}
extern "C" LC_IR_API WarpIsFirstActiveLane *lc_ir_v2_Func_as_WarpIsFirstActiveLane(Func *self) {
    return self->as<WarpIsFirstActiveLane>();
}
extern "C" LC_IR_API WarpFirstActiveLane *lc_ir_v2_Func_as_WarpFirstActiveLane(Func *self) {
    return self->as<WarpFirstActiveLane>();
}
extern "C" LC_IR_API WarpActiveAllEqual *lc_ir_v2_Func_as_WarpActiveAllEqual(Func *self) {
    return self->as<WarpActiveAllEqual>();
}
extern "C" LC_IR_API WarpActiveBitAnd *lc_ir_v2_Func_as_WarpActiveBitAnd(Func *self) {
    return self->as<WarpActiveBitAnd>();
}
extern "C" LC_IR_API WarpActiveBitOr *lc_ir_v2_Func_as_WarpActiveBitOr(Func *self) {
    return self->as<WarpActiveBitOr>();
}
extern "C" LC_IR_API WarpActiveBitXor *lc_ir_v2_Func_as_WarpActiveBitXor(Func *self) {
    return self->as<WarpActiveBitXor>();
}
extern "C" LC_IR_API WarpActiveCountBits *lc_ir_v2_Func_as_WarpActiveCountBits(Func *self) {
    return self->as<WarpActiveCountBits>();
}
extern "C" LC_IR_API WarpActiveMax *lc_ir_v2_Func_as_WarpActiveMax(Func *self) {
    return self->as<WarpActiveMax>();
}
extern "C" LC_IR_API WarpActiveMin *lc_ir_v2_Func_as_WarpActiveMin(Func *self) {
    return self->as<WarpActiveMin>();
}
extern "C" LC_IR_API WarpActiveProduct *lc_ir_v2_Func_as_WarpActiveProduct(Func *self) {
    return self->as<WarpActiveProduct>();
}
extern "C" LC_IR_API WarpActiveSum *lc_ir_v2_Func_as_WarpActiveSum(Func *self) {
    return self->as<WarpActiveSum>();
}
extern "C" LC_IR_API WarpActiveAll *lc_ir_v2_Func_as_WarpActiveAll(Func *self) {
    return self->as<WarpActiveAll>();
}
extern "C" LC_IR_API WarpActiveAny *lc_ir_v2_Func_as_WarpActiveAny(Func *self) {
    return self->as<WarpActiveAny>();
}
extern "C" LC_IR_API WarpActiveBitMask *lc_ir_v2_Func_as_WarpActiveBitMask(Func *self) {
    return self->as<WarpActiveBitMask>();
}
extern "C" LC_IR_API WarpPrefixCountBits *lc_ir_v2_Func_as_WarpPrefixCountBits(Func *self) {
    return self->as<WarpPrefixCountBits>();
}
extern "C" LC_IR_API WarpPrefixSum *lc_ir_v2_Func_as_WarpPrefixSum(Func *self) {
    return self->as<WarpPrefixSum>();
}
extern "C" LC_IR_API WarpPrefixProduct *lc_ir_v2_Func_as_WarpPrefixProduct(Func *self) {
    return self->as<WarpPrefixProduct>();
}
extern "C" LC_IR_API WarpReadLaneAt *lc_ir_v2_Func_as_WarpReadLaneAt(Func *self) {
    return self->as<WarpReadLaneAt>();
}
extern "C" LC_IR_API WarpReadFirstLane *lc_ir_v2_Func_as_WarpReadFirstLane(Func *self) {
    return self->as<WarpReadFirstLane>();
}
extern "C" LC_IR_API SynchronizeBlock *lc_ir_v2_Func_as_SynchronizeBlock(Func *self) {
    return self->as<SynchronizeBlock>();
}
extern "C" LC_IR_API AtomicExchange *lc_ir_v2_Func_as_AtomicExchange(Func *self) {
    return self->as<AtomicExchange>();
}
extern "C" LC_IR_API AtomicCompareExchange *lc_ir_v2_Func_as_AtomicCompareExchange(Func *self) {
    return self->as<AtomicCompareExchange>();
}
extern "C" LC_IR_API AtomicFetchAdd *lc_ir_v2_Func_as_AtomicFetchAdd(Func *self) {
    return self->as<AtomicFetchAdd>();
}
extern "C" LC_IR_API AtomicFetchSub *lc_ir_v2_Func_as_AtomicFetchSub(Func *self) {
    return self->as<AtomicFetchSub>();
}
extern "C" LC_IR_API AtomicFetchAnd *lc_ir_v2_Func_as_AtomicFetchAnd(Func *self) {
    return self->as<AtomicFetchAnd>();
}
extern "C" LC_IR_API AtomicFetchOr *lc_ir_v2_Func_as_AtomicFetchOr(Func *self) {
    return self->as<AtomicFetchOr>();
}
extern "C" LC_IR_API AtomicFetchXor *lc_ir_v2_Func_as_AtomicFetchXor(Func *self) {
    return self->as<AtomicFetchXor>();
}
extern "C" LC_IR_API AtomicFetchMin *lc_ir_v2_Func_as_AtomicFetchMin(Func *self) {
    return self->as<AtomicFetchMin>();
}
extern "C" LC_IR_API AtomicFetchMax *lc_ir_v2_Func_as_AtomicFetchMax(Func *self) {
    return self->as<AtomicFetchMax>();
}
extern "C" LC_IR_API BufferWrite *lc_ir_v2_Func_as_BufferWrite(Func *self) {
    return self->as<BufferWrite>();
}
extern "C" LC_IR_API BufferRead *lc_ir_v2_Func_as_BufferRead(Func *self) {
    return self->as<BufferRead>();
}
extern "C" LC_IR_API BufferSize *lc_ir_v2_Func_as_BufferSize(Func *self) {
    return self->as<BufferSize>();
}
extern "C" LC_IR_API ByteBufferWrite *lc_ir_v2_Func_as_ByteBufferWrite(Func *self) {
    return self->as<ByteBufferWrite>();
}
extern "C" LC_IR_API ByteBufferRead *lc_ir_v2_Func_as_ByteBufferRead(Func *self) {
    return self->as<ByteBufferRead>();
}
extern "C" LC_IR_API ByteBufferSize *lc_ir_v2_Func_as_ByteBufferSize(Func *self) {
    return self->as<ByteBufferSize>();
}
extern "C" LC_IR_API Texture2dRead *lc_ir_v2_Func_as_Texture2dRead(Func *self) {
    return self->as<Texture2dRead>();
}
extern "C" LC_IR_API Texture2dWrite *lc_ir_v2_Func_as_Texture2dWrite(Func *self) {
    return self->as<Texture2dWrite>();
}
extern "C" LC_IR_API Texture2dSize *lc_ir_v2_Func_as_Texture2dSize(Func *self) {
    return self->as<Texture2dSize>();
}
extern "C" LC_IR_API Texture3dRead *lc_ir_v2_Func_as_Texture3dRead(Func *self) {
    return self->as<Texture3dRead>();
}
extern "C" LC_IR_API Texture3dWrite *lc_ir_v2_Func_as_Texture3dWrite(Func *self) {
    return self->as<Texture3dWrite>();
}
extern "C" LC_IR_API Texture3dSize *lc_ir_v2_Func_as_Texture3dSize(Func *self) {
    return self->as<Texture3dSize>();
}
extern "C" LC_IR_API BindlessTexture2dSample *lc_ir_v2_Func_as_BindlessTexture2dSample(Func *self) {
    return self->as<BindlessTexture2dSample>();
}
extern "C" LC_IR_API BindlessTexture2dSampleLevel *lc_ir_v2_Func_as_BindlessTexture2dSampleLevel(Func *self) {
    return self->as<BindlessTexture2dSampleLevel>();
}
extern "C" LC_IR_API BindlessTexture2dSampleGrad *lc_ir_v2_Func_as_BindlessTexture2dSampleGrad(Func *self) {
    return self->as<BindlessTexture2dSampleGrad>();
}
extern "C" LC_IR_API BindlessTexture2dSampleGradLevel *lc_ir_v2_Func_as_BindlessTexture2dSampleGradLevel(Func *self) {
    return self->as<BindlessTexture2dSampleGradLevel>();
}
extern "C" LC_IR_API BindlessTexture2dRead *lc_ir_v2_Func_as_BindlessTexture2dRead(Func *self) {
    return self->as<BindlessTexture2dRead>();
}
extern "C" LC_IR_API BindlessTexture2dSize *lc_ir_v2_Func_as_BindlessTexture2dSize(Func *self) {
    return self->as<BindlessTexture2dSize>();
}
extern "C" LC_IR_API BindlessTexture2dSizeLevel *lc_ir_v2_Func_as_BindlessTexture2dSizeLevel(Func *self) {
    return self->as<BindlessTexture2dSizeLevel>();
}
extern "C" LC_IR_API BindlessTexture3dSample *lc_ir_v2_Func_as_BindlessTexture3dSample(Func *self) {
    return self->as<BindlessTexture3dSample>();
}
extern "C" LC_IR_API BindlessTexture3dSampleLevel *lc_ir_v2_Func_as_BindlessTexture3dSampleLevel(Func *self) {
    return self->as<BindlessTexture3dSampleLevel>();
}
extern "C" LC_IR_API BindlessTexture3dSampleGrad *lc_ir_v2_Func_as_BindlessTexture3dSampleGrad(Func *self) {
    return self->as<BindlessTexture3dSampleGrad>();
}
extern "C" LC_IR_API BindlessTexture3dSampleGradLevel *lc_ir_v2_Func_as_BindlessTexture3dSampleGradLevel(Func *self) {
    return self->as<BindlessTexture3dSampleGradLevel>();
}
extern "C" LC_IR_API BindlessTexture3dRead *lc_ir_v2_Func_as_BindlessTexture3dRead(Func *self) {
    return self->as<BindlessTexture3dRead>();
}
extern "C" LC_IR_API BindlessTexture3dSize *lc_ir_v2_Func_as_BindlessTexture3dSize(Func *self) {
    return self->as<BindlessTexture3dSize>();
}
extern "C" LC_IR_API BindlessTexture3dSizeLevel *lc_ir_v2_Func_as_BindlessTexture3dSizeLevel(Func *self) {
    return self->as<BindlessTexture3dSizeLevel>();
}
extern "C" LC_IR_API BindlessBufferWrite *lc_ir_v2_Func_as_BindlessBufferWrite(Func *self) {
    return self->as<BindlessBufferWrite>();
}
extern "C" LC_IR_API BindlessBufferRead *lc_ir_v2_Func_as_BindlessBufferRead(Func *self) {
    return self->as<BindlessBufferRead>();
}
extern "C" LC_IR_API BindlessBufferSize *lc_ir_v2_Func_as_BindlessBufferSize(Func *self) {
    return self->as<BindlessBufferSize>();
}
extern "C" LC_IR_API BindlessByteBufferWrite *lc_ir_v2_Func_as_BindlessByteBufferWrite(Func *self) {
    return self->as<BindlessByteBufferWrite>();
}
extern "C" LC_IR_API BindlessByteBufferRead *lc_ir_v2_Func_as_BindlessByteBufferRead(Func *self) {
    return self->as<BindlessByteBufferRead>();
}
extern "C" LC_IR_API BindlessByteBufferSize *lc_ir_v2_Func_as_BindlessByteBufferSize(Func *self) {
    return self->as<BindlessByteBufferSize>();
}
extern "C" LC_IR_API Vec *lc_ir_v2_Func_as_Vec(Func *self) {
    return self->as<Vec>();
}
extern "C" LC_IR_API Vec2 *lc_ir_v2_Func_as_Vec2(Func *self) {
    return self->as<Vec2>();
}
extern "C" LC_IR_API Vec3 *lc_ir_v2_Func_as_Vec3(Func *self) {
    return self->as<Vec3>();
}
extern "C" LC_IR_API Vec4 *lc_ir_v2_Func_as_Vec4(Func *self) {
    return self->as<Vec4>();
}
extern "C" LC_IR_API Permute *lc_ir_v2_Func_as_Permute(Func *self) {
    return self->as<Permute>();
}
extern "C" LC_IR_API GetElementPtr *lc_ir_v2_Func_as_GetElementPtr(Func *self) {
    return self->as<GetElementPtr>();
}
extern "C" LC_IR_API ExtractElement *lc_ir_v2_Func_as_ExtractElement(Func *self) {
    return self->as<ExtractElement>();
}
extern "C" LC_IR_API InsertElement *lc_ir_v2_Func_as_InsertElement(Func *self) {
    return self->as<InsertElement>();
}
extern "C" LC_IR_API Array *lc_ir_v2_Func_as_Array(Func *self) {
    return self->as<Array>();
}
extern "C" LC_IR_API Struct *lc_ir_v2_Func_as_Struct(Func *self) {
    return self->as<Struct>();
}
extern "C" LC_IR_API MatFull *lc_ir_v2_Func_as_MatFull(Func *self) {
    return self->as<MatFull>();
}
extern "C" LC_IR_API Mat2 *lc_ir_v2_Func_as_Mat2(Func *self) {
    return self->as<Mat2>();
}
extern "C" LC_IR_API Mat3 *lc_ir_v2_Func_as_Mat3(Func *self) {
    return self->as<Mat3>();
}
extern "C" LC_IR_API Mat4 *lc_ir_v2_Func_as_Mat4(Func *self) {
    return self->as<Mat4>();
}
extern "C" LC_IR_API BindlessAtomicFetchAdd *lc_ir_v2_Func_as_BindlessAtomicFetchAdd(Func *self) {
    return self->as<BindlessAtomicFetchAdd>();
}
extern "C" LC_IR_API BindlessAtomicFetchSub *lc_ir_v2_Func_as_BindlessAtomicFetchSub(Func *self) {
    return self->as<BindlessAtomicFetchSub>();
}
extern "C" LC_IR_API BindlessAtomicFetchAnd *lc_ir_v2_Func_as_BindlessAtomicFetchAnd(Func *self) {
    return self->as<BindlessAtomicFetchAnd>();
}
extern "C" LC_IR_API BindlessAtomicFetchOr *lc_ir_v2_Func_as_BindlessAtomicFetchOr(Func *self) {
    return self->as<BindlessAtomicFetchOr>();
}
extern "C" LC_IR_API BindlessAtomicFetchXor *lc_ir_v2_Func_as_BindlessAtomicFetchXor(Func *self) {
    return self->as<BindlessAtomicFetchXor>();
}
extern "C" LC_IR_API BindlessAtomicFetchMin *lc_ir_v2_Func_as_BindlessAtomicFetchMin(Func *self) {
    return self->as<BindlessAtomicFetchMin>();
}
extern "C" LC_IR_API BindlessAtomicFetchMax *lc_ir_v2_Func_as_BindlessAtomicFetchMax(Func *self) {
    return self->as<BindlessAtomicFetchMax>();
}
extern "C" LC_IR_API Callable *lc_ir_v2_Func_as_Callable(Func *self) {
    return self->as<Callable>();
}
extern "C" LC_IR_API CpuExt *lc_ir_v2_Func_as_CpuExt(Func *self) {
    return self->as<CpuExt>();
}
extern "C" LC_IR_API ShaderExecutionReorder *lc_ir_v2_Func_as_ShaderExecutionReorder(Func *self) {
    return self->as<ShaderExecutionReorder>();
}
extern "C" LC_IR_API FuncTag lc_ir_v2_Func_tag(Func *self) {
    return self->tag();
}
extern "C" LC_IR_API void lc_ir_v2_Assume_msg(Assume *self, Slice<const char> *out) {
    *out = self->msg;
}
extern "C" LC_IR_API void lc_ir_v2_Unreachable_msg(Unreachable *self, Slice<const char> *out) {
    *out = self->msg;
}
extern "C" LC_IR_API void lc_ir_v2_BindlessAtomicFetchAdd_ty(BindlessAtomicFetchAdd *self, const Type **out) {
    *out = self->ty;
}
extern "C" LC_IR_API void lc_ir_v2_BindlessAtomicFetchSub_ty(BindlessAtomicFetchSub *self, const Type **out) {
    *out = self->ty;
}
extern "C" LC_IR_API void lc_ir_v2_BindlessAtomicFetchAnd_ty(BindlessAtomicFetchAnd *self, const Type **out) {
    *out = self->ty;
}
extern "C" LC_IR_API void lc_ir_v2_BindlessAtomicFetchOr_ty(BindlessAtomicFetchOr *self, const Type **out) {
    *out = self->ty;
}
extern "C" LC_IR_API void lc_ir_v2_BindlessAtomicFetchXor_ty(BindlessAtomicFetchXor *self, const Type **out) {
    *out = self->ty;
}
extern "C" LC_IR_API void lc_ir_v2_BindlessAtomicFetchMin_ty(BindlessAtomicFetchMin *self, const Type **out) {
    *out = self->ty;
}
extern "C" LC_IR_API void lc_ir_v2_BindlessAtomicFetchMax_ty(BindlessAtomicFetchMax *self, const Type **out) {
    *out = self->ty;
}
extern "C" LC_IR_API void lc_ir_v2_Callable_module(Callable *self, CallableModule **out) {
    *out = self->module.get();
}
extern "C" LC_IR_API void lc_ir_v2_CpuExt_f(CpuExt *self, CpuExternFn *out) {
    *out = self->f;
}
extern "C" LC_IR_API Buffer *lc_ir_v2_Instruction_as_Buffer(Instruction *self) {
    return self->as<Buffer>();
}
extern "C" LC_IR_API Texture2d *lc_ir_v2_Instruction_as_Texture2d(Instruction *self) {
    return self->as<Texture2d>();
}
extern "C" LC_IR_API Texture3d *lc_ir_v2_Instruction_as_Texture3d(Instruction *self) {
    return self->as<Texture3d>();
}
extern "C" LC_IR_API BindlessArray *lc_ir_v2_Instruction_as_BindlessArray(Instruction *self) {
    return self->as<BindlessArray>();
}
extern "C" LC_IR_API Accel *lc_ir_v2_Instruction_as_Accel(Instruction *self) {
    return self->as<Accel>();
}
extern "C" LC_IR_API Shared *lc_ir_v2_Instruction_as_Shared(Instruction *self) {
    return self->as<Shared>();
}
extern "C" LC_IR_API Uniform *lc_ir_v2_Instruction_as_Uniform(Instruction *self) {
    return self->as<Uniform>();
}
extern "C" LC_IR_API Argument *lc_ir_v2_Instruction_as_Argument(Instruction *self) {
    return self->as<Argument>();
}
extern "C" LC_IR_API Const *lc_ir_v2_Instruction_as_Const(Instruction *self) {
    return self->as<Const>();
}
extern "C" LC_IR_API Call *lc_ir_v2_Instruction_as_Call(Instruction *self) {
    return self->as<Call>();
}
extern "C" LC_IR_API Phi *lc_ir_v2_Instruction_as_Phi(Instruction *self) {
    return self->as<Phi>();
}
extern "C" LC_IR_API BasicBlockSentinel *lc_ir_v2_Instruction_as_BasicBlockSentinel(Instruction *self) {
    return self->as<BasicBlockSentinel>();
}
extern "C" LC_IR_API If *lc_ir_v2_Instruction_as_If(Instruction *self) {
    return self->as<If>();
}
extern "C" LC_IR_API GenericLoop *lc_ir_v2_Instruction_as_GenericLoop(Instruction *self) {
    return self->as<GenericLoop>();
}
extern "C" LC_IR_API Switch *lc_ir_v2_Instruction_as_Switch(Instruction *self) {
    return self->as<Switch>();
}
extern "C" LC_IR_API Local *lc_ir_v2_Instruction_as_Local(Instruction *self) {
    return self->as<Local>();
}
extern "C" LC_IR_API Break *lc_ir_v2_Instruction_as_Break(Instruction *self) {
    return self->as<Break>();
}
extern "C" LC_IR_API Continue *lc_ir_v2_Instruction_as_Continue(Instruction *self) {
    return self->as<Continue>();
}
extern "C" LC_IR_API Return *lc_ir_v2_Instruction_as_Return(Instruction *self) {
    return self->as<Return>();
}
extern "C" LC_IR_API Print *lc_ir_v2_Instruction_as_Print(Instruction *self) {
    return self->as<Print>();
}
extern "C" LC_IR_API InstructionTag lc_ir_v2_Instruction_tag(Instruction *self) {
    return self->tag();
}
extern "C" LC_IR_API void lc_ir_v2_Argument_by_value(Argument *self, bool *out) {
    *out = self->by_value;
}
extern "C" LC_IR_API void lc_ir_v2_Const_ty(Const *self, const Type **out) {
    *out = self->ty;
}
extern "C" LC_IR_API void lc_ir_v2_Const_value(Const *self, Slice<uint8_t> *out) {
    *out = self->value;
}
extern "C" LC_IR_API void lc_ir_v2_Call_func(Call *self, const Func **out) {
    *out = self->func;
}
extern "C" LC_IR_API void lc_ir_v2_Call_args(Call *self, Slice<Node *> *out) {
    *out = self->args;
}
extern "C" LC_IR_API void lc_ir_v2_Phi_incomings(Phi *self, Slice<PhiIncoming> *out) {
    *out = self->incomings;
}
extern "C" LC_IR_API void lc_ir_v2_If_cond(If *self, Node **out) {
    *out = self->cond;
}
extern "C" LC_IR_API void lc_ir_v2_If_true_branch(If *self, BasicBlock **out) {
    *out = self->true_branch;
}
extern "C" LC_IR_API void lc_ir_v2_If_false_branch(If *self, BasicBlock **out) {
    *out = self->false_branch;
}
extern "C" LC_IR_API void lc_ir_v2_GenericLoop_prepare(GenericLoop *self, BasicBlock **out) {
    *out = self->prepare;
}
extern "C" LC_IR_API void lc_ir_v2_GenericLoop_cond(GenericLoop *self, Node **out) {
    *out = self->cond;
}
extern "C" LC_IR_API void lc_ir_v2_GenericLoop_body(GenericLoop *self, BasicBlock **out) {
    *out = self->body;
}
extern "C" LC_IR_API void lc_ir_v2_GenericLoop_update(GenericLoop *self, BasicBlock **out) {
    *out = self->update;
}
extern "C" LC_IR_API void lc_ir_v2_Switch_value(Switch *self, Node **out) {
    *out = self->value;
}
extern "C" LC_IR_API void lc_ir_v2_Switch_cases(Switch *self, Slice<SwitchCase> *out) {
    *out = self->cases;
}
extern "C" LC_IR_API void lc_ir_v2_Switch_default_(Switch *self, BasicBlock **out) {
    *out = self->default_;
}
extern "C" LC_IR_API void lc_ir_v2_Local_init(Local *self, Node **out) {
    *out = self->init;
}
extern "C" LC_IR_API void lc_ir_v2_Return_value(Return *self, Node **out) {
    *out = self->value;
}
extern "C" LC_IR_API void lc_ir_v2_Print_fmt(Print *self, Slice<const char> *out) {
    *out = self->fmt;
}
extern "C" LC_IR_API void lc_ir_v2_Print_args(Print *self, Slice<Node *> *out) {
    *out = self->args;
}
extern "C" LC_IR_API BufferBinding *lc_ir_v2_Binding_as_BufferBinding(Binding *self) {
    return self->as<BufferBinding>();
}
extern "C" LC_IR_API TextureBinding *lc_ir_v2_Binding_as_TextureBinding(Binding *self) {
    return self->as<TextureBinding>();
}
extern "C" LC_IR_API BindlessArrayBinding *lc_ir_v2_Binding_as_BindlessArrayBinding(Binding *self) {
    return self->as<BindlessArrayBinding>();
}
extern "C" LC_IR_API AccelBinding *lc_ir_v2_Binding_as_AccelBinding(Binding *self) {
    return self->as<AccelBinding>();
}
extern "C" LC_IR_API BindingTag lc_ir_v2_Binding_tag(Binding *self) {
    return self->tag();
}
extern "C" LC_IR_API void lc_ir_v2_BufferBinding_handle(BufferBinding *self, uint64_t *out) {
    *out = self->handle;
}
extern "C" LC_IR_API void lc_ir_v2_BufferBinding_offset(BufferBinding *self, uint64_t *out) {
    *out = self->offset;
}
extern "C" LC_IR_API void lc_ir_v2_BufferBinding_size(BufferBinding *self, uint64_t *out) {
    *out = self->size;
}
extern "C" LC_IR_API void lc_ir_v2_TextureBinding_handle(TextureBinding *self, uint64_t *out) {
    *out = self->handle;
}
extern "C" LC_IR_API void lc_ir_v2_TextureBinding_level(TextureBinding *self, uint64_t *out) {
    *out = self->level;
}
extern "C" LC_IR_API void lc_ir_v2_BindlessArrayBinding_handle(BindlessArrayBinding *self, uint64_t *out) {
    *out = self->handle;
}
extern "C" LC_IR_API void lc_ir_v2_AccelBinding_handle(AccelBinding *self, uint64_t *out) {
    *out = self->handle;
}
}// namespace luisa::compute::ir_v2
