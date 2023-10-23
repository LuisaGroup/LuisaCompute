#include <luisa/ir_v2/ir_v2_api.h>
namespace luisa::compute::ir_v2 {
static Zero *Func_as_Zero(Func *self) {
    return self->as<Zero>();
}
static One *Func_as_One(Func *self) {
    return self->as<One>();
}
static Assume *Func_as_Assume(Func *self) {
    return self->as<Assume>();
}
static Unreachable *Func_as_Unreachable(Func *self) {
    return self->as<Unreachable>();
}
static ThreadId *Func_as_ThreadId(Func *self) {
    return self->as<ThreadId>();
}
static BlockId *Func_as_BlockId(Func *self) {
    return self->as<BlockId>();
}
static WarpSize *Func_as_WarpSize(Func *self) {
    return self->as<WarpSize>();
}
static WarpLaneId *Func_as_WarpLaneId(Func *self) {
    return self->as<WarpLaneId>();
}
static DispatchId *Func_as_DispatchId(Func *self) {
    return self->as<DispatchId>();
}
static DispatchSize *Func_as_DispatchSize(Func *self) {
    return self->as<DispatchSize>();
}
static PropagateGradient *Func_as_PropagateGradient(Func *self) {
    return self->as<PropagateGradient>();
}
static OutputGradient *Func_as_OutputGradient(Func *self) {
    return self->as<OutputGradient>();
}
static RequiresGradient *Func_as_RequiresGradient(Func *self) {
    return self->as<RequiresGradient>();
}
static Backward *Func_as_Backward(Func *self) {
    return self->as<Backward>();
}
static Gradient *Func_as_Gradient(Func *self) {
    return self->as<Gradient>();
}
static AccGrad *Func_as_AccGrad(Func *self) {
    return self->as<AccGrad>();
}
static Detach *Func_as_Detach(Func *self) {
    return self->as<Detach>();
}
static RayTracingInstanceTransform *Func_as_RayTracingInstanceTransform(Func *self) {
    return self->as<RayTracingInstanceTransform>();
}
static RayTracingInstanceVisibilityMask *Func_as_RayTracingInstanceVisibilityMask(Func *self) {
    return self->as<RayTracingInstanceVisibilityMask>();
}
static RayTracingInstanceUserId *Func_as_RayTracingInstanceUserId(Func *self) {
    return self->as<RayTracingInstanceUserId>();
}
static RayTracingSetInstanceTransform *Func_as_RayTracingSetInstanceTransform(Func *self) {
    return self->as<RayTracingSetInstanceTransform>();
}
static RayTracingSetInstanceOpacity *Func_as_RayTracingSetInstanceOpacity(Func *self) {
    return self->as<RayTracingSetInstanceOpacity>();
}
static RayTracingSetInstanceVisibility *Func_as_RayTracingSetInstanceVisibility(Func *self) {
    return self->as<RayTracingSetInstanceVisibility>();
}
static RayTracingSetInstanceUserId *Func_as_RayTracingSetInstanceUserId(Func *self) {
    return self->as<RayTracingSetInstanceUserId>();
}
static RayTracingTraceClosest *Func_as_RayTracingTraceClosest(Func *self) {
    return self->as<RayTracingTraceClosest>();
}
static RayTracingTraceAny *Func_as_RayTracingTraceAny(Func *self) {
    return self->as<RayTracingTraceAny>();
}
static RayTracingQueryAll *Func_as_RayTracingQueryAll(Func *self) {
    return self->as<RayTracingQueryAll>();
}
static RayTracingQueryAny *Func_as_RayTracingQueryAny(Func *self) {
    return self->as<RayTracingQueryAny>();
}
static RayQueryWorldSpaceRay *Func_as_RayQueryWorldSpaceRay(Func *self) {
    return self->as<RayQueryWorldSpaceRay>();
}
static RayQueryProceduralCandidateHit *Func_as_RayQueryProceduralCandidateHit(Func *self) {
    return self->as<RayQueryProceduralCandidateHit>();
}
static RayQueryTriangleCandidateHit *Func_as_RayQueryTriangleCandidateHit(Func *self) {
    return self->as<RayQueryTriangleCandidateHit>();
}
static RayQueryCommittedHit *Func_as_RayQueryCommittedHit(Func *self) {
    return self->as<RayQueryCommittedHit>();
}
static RayQueryCommitTriangle *Func_as_RayQueryCommitTriangle(Func *self) {
    return self->as<RayQueryCommitTriangle>();
}
static RayQueryCommitdProcedural *Func_as_RayQueryCommitdProcedural(Func *self) {
    return self->as<RayQueryCommitdProcedural>();
}
static RayQueryTerminate *Func_as_RayQueryTerminate(Func *self) {
    return self->as<RayQueryTerminate>();
}
static Load *Func_as_Load(Func *self) {
    return self->as<Load>();
}
static Cast *Func_as_Cast(Func *self) {
    return self->as<Cast>();
}
static BitCast *Func_as_BitCast(Func *self) {
    return self->as<BitCast>();
}
static Add *Func_as_Add(Func *self) {
    return self->as<Add>();
}
static Sub *Func_as_Sub(Func *self) {
    return self->as<Sub>();
}
static Mul *Func_as_Mul(Func *self) {
    return self->as<Mul>();
}
static Div *Func_as_Div(Func *self) {
    return self->as<Div>();
}
static Rem *Func_as_Rem(Func *self) {
    return self->as<Rem>();
}
static BitAnd *Func_as_BitAnd(Func *self) {
    return self->as<BitAnd>();
}
static BitOr *Func_as_BitOr(Func *self) {
    return self->as<BitOr>();
}
static BitXor *Func_as_BitXor(Func *self) {
    return self->as<BitXor>();
}
static Shl *Func_as_Shl(Func *self) {
    return self->as<Shl>();
}
static Shr *Func_as_Shr(Func *self) {
    return self->as<Shr>();
}
static RotRight *Func_as_RotRight(Func *self) {
    return self->as<RotRight>();
}
static RotLeft *Func_as_RotLeft(Func *self) {
    return self->as<RotLeft>();
}
static Eq *Func_as_Eq(Func *self) {
    return self->as<Eq>();
}
static Ne *Func_as_Ne(Func *self) {
    return self->as<Ne>();
}
static Lt *Func_as_Lt(Func *self) {
    return self->as<Lt>();
}
static Le *Func_as_Le(Func *self) {
    return self->as<Le>();
}
static Gt *Func_as_Gt(Func *self) {
    return self->as<Gt>();
}
static Ge *Func_as_Ge(Func *self) {
    return self->as<Ge>();
}
static MatCompMul *Func_as_MatCompMul(Func *self) {
    return self->as<MatCompMul>();
}
static Neg *Func_as_Neg(Func *self) {
    return self->as<Neg>();
}
static Not *Func_as_Not(Func *self) {
    return self->as<Not>();
}
static BitNot *Func_as_BitNot(Func *self) {
    return self->as<BitNot>();
}
static All *Func_as_All(Func *self) {
    return self->as<All>();
}
static Any *Func_as_Any(Func *self) {
    return self->as<Any>();
}
static Select *Func_as_Select(Func *self) {
    return self->as<Select>();
}
static Clamp *Func_as_Clamp(Func *self) {
    return self->as<Clamp>();
}
static Lerp *Func_as_Lerp(Func *self) {
    return self->as<Lerp>();
}
static Step *Func_as_Step(Func *self) {
    return self->as<Step>();
}
static Saturate *Func_as_Saturate(Func *self) {
    return self->as<Saturate>();
}
static SmoothStep *Func_as_SmoothStep(Func *self) {
    return self->as<SmoothStep>();
}
static Abs *Func_as_Abs(Func *self) {
    return self->as<Abs>();
}
static Min *Func_as_Min(Func *self) {
    return self->as<Min>();
}
static Max *Func_as_Max(Func *self) {
    return self->as<Max>();
}
static ReduceSum *Func_as_ReduceSum(Func *self) {
    return self->as<ReduceSum>();
}
static ReduceProd *Func_as_ReduceProd(Func *self) {
    return self->as<ReduceProd>();
}
static ReduceMin *Func_as_ReduceMin(Func *self) {
    return self->as<ReduceMin>();
}
static ReduceMax *Func_as_ReduceMax(Func *self) {
    return self->as<ReduceMax>();
}
static Clz *Func_as_Clz(Func *self) {
    return self->as<Clz>();
}
static Ctz *Func_as_Ctz(Func *self) {
    return self->as<Ctz>();
}
static PopCount *Func_as_PopCount(Func *self) {
    return self->as<PopCount>();
}
static Reverse *Func_as_Reverse(Func *self) {
    return self->as<Reverse>();
}
static IsInf *Func_as_IsInf(Func *self) {
    return self->as<IsInf>();
}
static IsNan *Func_as_IsNan(Func *self) {
    return self->as<IsNan>();
}
static Acos *Func_as_Acos(Func *self) {
    return self->as<Acos>();
}
static Acosh *Func_as_Acosh(Func *self) {
    return self->as<Acosh>();
}
static Asin *Func_as_Asin(Func *self) {
    return self->as<Asin>();
}
static Asinh *Func_as_Asinh(Func *self) {
    return self->as<Asinh>();
}
static Atan *Func_as_Atan(Func *self) {
    return self->as<Atan>();
}
static Atan2 *Func_as_Atan2(Func *self) {
    return self->as<Atan2>();
}
static Atanh *Func_as_Atanh(Func *self) {
    return self->as<Atanh>();
}
static Cos *Func_as_Cos(Func *self) {
    return self->as<Cos>();
}
static Cosh *Func_as_Cosh(Func *self) {
    return self->as<Cosh>();
}
static Sin *Func_as_Sin(Func *self) {
    return self->as<Sin>();
}
static Sinh *Func_as_Sinh(Func *self) {
    return self->as<Sinh>();
}
static Tan *Func_as_Tan(Func *self) {
    return self->as<Tan>();
}
static Tanh *Func_as_Tanh(Func *self) {
    return self->as<Tanh>();
}
static Exp *Func_as_Exp(Func *self) {
    return self->as<Exp>();
}
static Exp2 *Func_as_Exp2(Func *self) {
    return self->as<Exp2>();
}
static Exp10 *Func_as_Exp10(Func *self) {
    return self->as<Exp10>();
}
static Log *Func_as_Log(Func *self) {
    return self->as<Log>();
}
static Log2 *Func_as_Log2(Func *self) {
    return self->as<Log2>();
}
static Log10 *Func_as_Log10(Func *self) {
    return self->as<Log10>();
}
static Powi *Func_as_Powi(Func *self) {
    return self->as<Powi>();
}
static Powf *Func_as_Powf(Func *self) {
    return self->as<Powf>();
}
static Sqrt *Func_as_Sqrt(Func *self) {
    return self->as<Sqrt>();
}
static Rsqrt *Func_as_Rsqrt(Func *self) {
    return self->as<Rsqrt>();
}
static Ceil *Func_as_Ceil(Func *self) {
    return self->as<Ceil>();
}
static Floor *Func_as_Floor(Func *self) {
    return self->as<Floor>();
}
static Fract *Func_as_Fract(Func *self) {
    return self->as<Fract>();
}
static Trunc *Func_as_Trunc(Func *self) {
    return self->as<Trunc>();
}
static Round *Func_as_Round(Func *self) {
    return self->as<Round>();
}
static Fma *Func_as_Fma(Func *self) {
    return self->as<Fma>();
}
static Copysign *Func_as_Copysign(Func *self) {
    return self->as<Copysign>();
}
static Cross *Func_as_Cross(Func *self) {
    return self->as<Cross>();
}
static Dot *Func_as_Dot(Func *self) {
    return self->as<Dot>();
}
static OuterProduct *Func_as_OuterProduct(Func *self) {
    return self->as<OuterProduct>();
}
static Length *Func_as_Length(Func *self) {
    return self->as<Length>();
}
static LengthSquared *Func_as_LengthSquared(Func *self) {
    return self->as<LengthSquared>();
}
static Normalize *Func_as_Normalize(Func *self) {
    return self->as<Normalize>();
}
static Faceforward *Func_as_Faceforward(Func *self) {
    return self->as<Faceforward>();
}
static Distance *Func_as_Distance(Func *self) {
    return self->as<Distance>();
}
static Reflect *Func_as_Reflect(Func *self) {
    return self->as<Reflect>();
}
static Determinant *Func_as_Determinant(Func *self) {
    return self->as<Determinant>();
}
static Transpose *Func_as_Transpose(Func *self) {
    return self->as<Transpose>();
}
static Inverse *Func_as_Inverse(Func *self) {
    return self->as<Inverse>();
}
static WarpIsFirstActiveLane *Func_as_WarpIsFirstActiveLane(Func *self) {
    return self->as<WarpIsFirstActiveLane>();
}
static WarpFirstActiveLane *Func_as_WarpFirstActiveLane(Func *self) {
    return self->as<WarpFirstActiveLane>();
}
static WarpActiveAllEqual *Func_as_WarpActiveAllEqual(Func *self) {
    return self->as<WarpActiveAllEqual>();
}
static WarpActiveBitAnd *Func_as_WarpActiveBitAnd(Func *self) {
    return self->as<WarpActiveBitAnd>();
}
static WarpActiveBitOr *Func_as_WarpActiveBitOr(Func *self) {
    return self->as<WarpActiveBitOr>();
}
static WarpActiveBitXor *Func_as_WarpActiveBitXor(Func *self) {
    return self->as<WarpActiveBitXor>();
}
static WarpActiveCountBits *Func_as_WarpActiveCountBits(Func *self) {
    return self->as<WarpActiveCountBits>();
}
static WarpActiveMax *Func_as_WarpActiveMax(Func *self) {
    return self->as<WarpActiveMax>();
}
static WarpActiveMin *Func_as_WarpActiveMin(Func *self) {
    return self->as<WarpActiveMin>();
}
static WarpActiveProduct *Func_as_WarpActiveProduct(Func *self) {
    return self->as<WarpActiveProduct>();
}
static WarpActiveSum *Func_as_WarpActiveSum(Func *self) {
    return self->as<WarpActiveSum>();
}
static WarpActiveAll *Func_as_WarpActiveAll(Func *self) {
    return self->as<WarpActiveAll>();
}
static WarpActiveAny *Func_as_WarpActiveAny(Func *self) {
    return self->as<WarpActiveAny>();
}
static WarpActiveBitMask *Func_as_WarpActiveBitMask(Func *self) {
    return self->as<WarpActiveBitMask>();
}
static WarpPrefixCountBits *Func_as_WarpPrefixCountBits(Func *self) {
    return self->as<WarpPrefixCountBits>();
}
static WarpPrefixSum *Func_as_WarpPrefixSum(Func *self) {
    return self->as<WarpPrefixSum>();
}
static WarpPrefixProduct *Func_as_WarpPrefixProduct(Func *self) {
    return self->as<WarpPrefixProduct>();
}
static WarpReadLaneAt *Func_as_WarpReadLaneAt(Func *self) {
    return self->as<WarpReadLaneAt>();
}
static WarpReadFirstLane *Func_as_WarpReadFirstLane(Func *self) {
    return self->as<WarpReadFirstLane>();
}
static SynchronizeBlock *Func_as_SynchronizeBlock(Func *self) {
    return self->as<SynchronizeBlock>();
}
static AtomicExchange *Func_as_AtomicExchange(Func *self) {
    return self->as<AtomicExchange>();
}
static AtomicCompareExchange *Func_as_AtomicCompareExchange(Func *self) {
    return self->as<AtomicCompareExchange>();
}
static AtomicFetchAdd *Func_as_AtomicFetchAdd(Func *self) {
    return self->as<AtomicFetchAdd>();
}
static AtomicFetchSub *Func_as_AtomicFetchSub(Func *self) {
    return self->as<AtomicFetchSub>();
}
static AtomicFetchAnd *Func_as_AtomicFetchAnd(Func *self) {
    return self->as<AtomicFetchAnd>();
}
static AtomicFetchOr *Func_as_AtomicFetchOr(Func *self) {
    return self->as<AtomicFetchOr>();
}
static AtomicFetchXor *Func_as_AtomicFetchXor(Func *self) {
    return self->as<AtomicFetchXor>();
}
static AtomicFetchMin *Func_as_AtomicFetchMin(Func *self) {
    return self->as<AtomicFetchMin>();
}
static AtomicFetchMax *Func_as_AtomicFetchMax(Func *self) {
    return self->as<AtomicFetchMax>();
}
static BufferWrite *Func_as_BufferWrite(Func *self) {
    return self->as<BufferWrite>();
}
static BufferRead *Func_as_BufferRead(Func *self) {
    return self->as<BufferRead>();
}
static BufferSize *Func_as_BufferSize(Func *self) {
    return self->as<BufferSize>();
}
static ByteBufferWrite *Func_as_ByteBufferWrite(Func *self) {
    return self->as<ByteBufferWrite>();
}
static ByteBufferRead *Func_as_ByteBufferRead(Func *self) {
    return self->as<ByteBufferRead>();
}
static ByteBufferSize *Func_as_ByteBufferSize(Func *self) {
    return self->as<ByteBufferSize>();
}
static Texture2dRead *Func_as_Texture2dRead(Func *self) {
    return self->as<Texture2dRead>();
}
static Texture2dWrite *Func_as_Texture2dWrite(Func *self) {
    return self->as<Texture2dWrite>();
}
static Texture2dSize *Func_as_Texture2dSize(Func *self) {
    return self->as<Texture2dSize>();
}
static Texture3dRead *Func_as_Texture3dRead(Func *self) {
    return self->as<Texture3dRead>();
}
static Texture3dWrite *Func_as_Texture3dWrite(Func *self) {
    return self->as<Texture3dWrite>();
}
static Texture3dSize *Func_as_Texture3dSize(Func *self) {
    return self->as<Texture3dSize>();
}
static BindlessTexture2dSample *Func_as_BindlessTexture2dSample(Func *self) {
    return self->as<BindlessTexture2dSample>();
}
static BindlessTexture2dSampleLevel *Func_as_BindlessTexture2dSampleLevel(Func *self) {
    return self->as<BindlessTexture2dSampleLevel>();
}
static BindlessTexture2dSampleGrad *Func_as_BindlessTexture2dSampleGrad(Func *self) {
    return self->as<BindlessTexture2dSampleGrad>();
}
static BindlessTexture2dSampleGradLevel *Func_as_BindlessTexture2dSampleGradLevel(Func *self) {
    return self->as<BindlessTexture2dSampleGradLevel>();
}
static BindlessTexture2dRead *Func_as_BindlessTexture2dRead(Func *self) {
    return self->as<BindlessTexture2dRead>();
}
static BindlessTexture2dSize *Func_as_BindlessTexture2dSize(Func *self) {
    return self->as<BindlessTexture2dSize>();
}
static BindlessTexture2dSizeLevel *Func_as_BindlessTexture2dSizeLevel(Func *self) {
    return self->as<BindlessTexture2dSizeLevel>();
}
static BindlessTexture3dSample *Func_as_BindlessTexture3dSample(Func *self) {
    return self->as<BindlessTexture3dSample>();
}
static BindlessTexture3dSampleLevel *Func_as_BindlessTexture3dSampleLevel(Func *self) {
    return self->as<BindlessTexture3dSampleLevel>();
}
static BindlessTexture3dSampleGrad *Func_as_BindlessTexture3dSampleGrad(Func *self) {
    return self->as<BindlessTexture3dSampleGrad>();
}
static BindlessTexture3dSampleGradLevel *Func_as_BindlessTexture3dSampleGradLevel(Func *self) {
    return self->as<BindlessTexture3dSampleGradLevel>();
}
static BindlessTexture3dRead *Func_as_BindlessTexture3dRead(Func *self) {
    return self->as<BindlessTexture3dRead>();
}
static BindlessTexture3dSize *Func_as_BindlessTexture3dSize(Func *self) {
    return self->as<BindlessTexture3dSize>();
}
static BindlessTexture3dSizeLevel *Func_as_BindlessTexture3dSizeLevel(Func *self) {
    return self->as<BindlessTexture3dSizeLevel>();
}
static BindlessBufferWrite *Func_as_BindlessBufferWrite(Func *self) {
    return self->as<BindlessBufferWrite>();
}
static BindlessBufferRead *Func_as_BindlessBufferRead(Func *self) {
    return self->as<BindlessBufferRead>();
}
static BindlessBufferSize *Func_as_BindlessBufferSize(Func *self) {
    return self->as<BindlessBufferSize>();
}
static BindlessByteBufferWrite *Func_as_BindlessByteBufferWrite(Func *self) {
    return self->as<BindlessByteBufferWrite>();
}
static BindlessByteBufferRead *Func_as_BindlessByteBufferRead(Func *self) {
    return self->as<BindlessByteBufferRead>();
}
static BindlessByteBufferSize *Func_as_BindlessByteBufferSize(Func *self) {
    return self->as<BindlessByteBufferSize>();
}
static Vec *Func_as_Vec(Func *self) {
    return self->as<Vec>();
}
static Vec2 *Func_as_Vec2(Func *self) {
    return self->as<Vec2>();
}
static Vec3 *Func_as_Vec3(Func *self) {
    return self->as<Vec3>();
}
static Vec4 *Func_as_Vec4(Func *self) {
    return self->as<Vec4>();
}
static Permute *Func_as_Permute(Func *self) {
    return self->as<Permute>();
}
static GetElementPtr *Func_as_GetElementPtr(Func *self) {
    return self->as<GetElementPtr>();
}
static ExtractElement *Func_as_ExtractElement(Func *self) {
    return self->as<ExtractElement>();
}
static InsertElement *Func_as_InsertElement(Func *self) {
    return self->as<InsertElement>();
}
static Array *Func_as_Array(Func *self) {
    return self->as<Array>();
}
static Struct *Func_as_Struct(Func *self) {
    return self->as<Struct>();
}
static MatFull *Func_as_MatFull(Func *self) {
    return self->as<MatFull>();
}
static Mat2 *Func_as_Mat2(Func *self) {
    return self->as<Mat2>();
}
static Mat3 *Func_as_Mat3(Func *self) {
    return self->as<Mat3>();
}
static Mat4 *Func_as_Mat4(Func *self) {
    return self->as<Mat4>();
}
static BindlessAtomicExchange *Func_as_BindlessAtomicExchange(Func *self) {
    return self->as<BindlessAtomicExchange>();
}
static BindlessAtomicCompareExchange *Func_as_BindlessAtomicCompareExchange(Func *self) {
    return self->as<BindlessAtomicCompareExchange>();
}
static BindlessAtomicFetchAdd *Func_as_BindlessAtomicFetchAdd(Func *self) {
    return self->as<BindlessAtomicFetchAdd>();
}
static BindlessAtomicFetchSub *Func_as_BindlessAtomicFetchSub(Func *self) {
    return self->as<BindlessAtomicFetchSub>();
}
static BindlessAtomicFetchAnd *Func_as_BindlessAtomicFetchAnd(Func *self) {
    return self->as<BindlessAtomicFetchAnd>();
}
static BindlessAtomicFetchOr *Func_as_BindlessAtomicFetchOr(Func *self) {
    return self->as<BindlessAtomicFetchOr>();
}
static BindlessAtomicFetchXor *Func_as_BindlessAtomicFetchXor(Func *self) {
    return self->as<BindlessAtomicFetchXor>();
}
static BindlessAtomicFetchMin *Func_as_BindlessAtomicFetchMin(Func *self) {
    return self->as<BindlessAtomicFetchMin>();
}
static BindlessAtomicFetchMax *Func_as_BindlessAtomicFetchMax(Func *self) {
    return self->as<BindlessAtomicFetchMax>();
}
static Callable *Func_as_Callable(Func *self) {
    return self->as<Callable>();
}
static CpuExt *Func_as_CpuExt(Func *self) {
    return self->as<CpuExt>();
}
static ShaderExecutionReorder *Func_as_ShaderExecutionReorder(Func *self) {
    return self->as<ShaderExecutionReorder>();
}
static FuncTag Func_tag(Func *self) {
    return self->tag();
}
static Slice<const char> Assume_msg(Assume *self) {
    return self->msg;
}
static Slice<const char> Unreachable_msg(Unreachable *self) {
    return self->msg;
}
static const Type *BindlessAtomicExchange_ty(BindlessAtomicExchange *self) {
    return self->ty;
}
static const Type *BindlessAtomicCompareExchange_ty(BindlessAtomicCompareExchange *self) {
    return self->ty;
}
static const Type *BindlessAtomicFetchAdd_ty(BindlessAtomicFetchAdd *self) {
    return self->ty;
}
static const Type *BindlessAtomicFetchSub_ty(BindlessAtomicFetchSub *self) {
    return self->ty;
}
static const Type *BindlessAtomicFetchAnd_ty(BindlessAtomicFetchAnd *self) {
    return self->ty;
}
static const Type *BindlessAtomicFetchOr_ty(BindlessAtomicFetchOr *self) {
    return self->ty;
}
static const Type *BindlessAtomicFetchXor_ty(BindlessAtomicFetchXor *self) {
    return self->ty;
}
static const Type *BindlessAtomicFetchMin_ty(BindlessAtomicFetchMin *self) {
    return self->ty;
}
static const Type *BindlessAtomicFetchMax_ty(BindlessAtomicFetchMax *self) {
    return self->ty;
}
static CallableModule *Callable_module(Callable *self) {
    return self->module.get();
}
static CpuExternFn CpuExt_f(CpuExt *self) {
    return self->f;
}
static Buffer *Instruction_as_Buffer(Instruction *self) {
    return self->as<Buffer>();
}
static Texture2d *Instruction_as_Texture2d(Instruction *self) {
    return self->as<Texture2d>();
}
static Texture3d *Instruction_as_Texture3d(Instruction *self) {
    return self->as<Texture3d>();
}
static BindlessArray *Instruction_as_BindlessArray(Instruction *self) {
    return self->as<BindlessArray>();
}
static Accel *Instruction_as_Accel(Instruction *self) {
    return self->as<Accel>();
}
static Shared *Instruction_as_Shared(Instruction *self) {
    return self->as<Shared>();
}
static Uniform *Instruction_as_Uniform(Instruction *self) {
    return self->as<Uniform>();
}
static Argument *Instruction_as_Argument(Instruction *self) {
    return self->as<Argument>();
}
static Constant *Instruction_as_Constant(Instruction *self) {
    return self->as<Constant>();
}
static Call *Instruction_as_Call(Instruction *self) {
    return self->as<Call>();
}
static Phi *Instruction_as_Phi(Instruction *self) {
    return self->as<Phi>();
}
static BasicBlockSentinel *Instruction_as_BasicBlockSentinel(Instruction *self) {
    return self->as<BasicBlockSentinel>();
}
static If *Instruction_as_If(Instruction *self) {
    return self->as<If>();
}
static GenericLoop *Instruction_as_GenericLoop(Instruction *self) {
    return self->as<GenericLoop>();
}
static Switch *Instruction_as_Switch(Instruction *self) {
    return self->as<Switch>();
}
static Local *Instruction_as_Local(Instruction *self) {
    return self->as<Local>();
}
static Break *Instruction_as_Break(Instruction *self) {
    return self->as<Break>();
}
static Continue *Instruction_as_Continue(Instruction *self) {
    return self->as<Continue>();
}
static Return *Instruction_as_Return(Instruction *self) {
    return self->as<Return>();
}
static Print *Instruction_as_Print(Instruction *self) {
    return self->as<Print>();
}
static Update *Instruction_as_Update(Instruction *self) {
    return self->as<Update>();
}
static RayQuery *Instruction_as_RayQuery(Instruction *self) {
    return self->as<RayQuery>();
}
static RevAutodiff *Instruction_as_RevAutodiff(Instruction *self) {
    return self->as<RevAutodiff>();
}
static FwdAutodiff *Instruction_as_FwdAutodiff(Instruction *self) {
    return self->as<FwdAutodiff>();
}
static InstructionTag Instruction_tag(Instruction *self) {
    return self->tag();
}
static bool Argument_by_value(Argument *self) {
    return self->by_value;
}
static const Type *Constant_ty(Constant *self) {
    return self->ty;
}
static Slice<uint8_t> Constant_value(Constant *self) {
    return self->value;
}
static const Func *Call_func(Call *self) {
    return self->func;
}
static Slice<Node *> Call_args(Call *self) {
    return self->args;
}
static Slice<PhiIncoming> Phi_incomings(Phi *self) {
    return self->incomings;
}
static Node *If_cond(If *self) {
    return self->cond;
}
static BasicBlock *If_true_branch(If *self) {
    return self->true_branch;
}
static BasicBlock *If_false_branch(If *self) {
    return self->false_branch;
}
static BasicBlock *GenericLoop_prepare(GenericLoop *self) {
    return self->prepare;
}
static Node *GenericLoop_cond(GenericLoop *self) {
    return self->cond;
}
static BasicBlock *GenericLoop_body(GenericLoop *self) {
    return self->body;
}
static BasicBlock *GenericLoop_update(GenericLoop *self) {
    return self->update;
}
static Node *Switch_value(Switch *self) {
    return self->value;
}
static Slice<SwitchCase> Switch_cases(Switch *self) {
    return self->cases;
}
static BasicBlock *Switch_default_(Switch *self) {
    return self->default_;
}
static Node *Local_init(Local *self) {
    return self->init;
}
static Node *Return_value(Return *self) {
    return self->value;
}
static Slice<const char> Print_fmt(Print *self) {
    return self->fmt;
}
static Slice<Node *> Print_args(Print *self) {
    return self->args;
}
static Node *Update_var(Update *self) {
    return self->var;
}
static Node *Update_value(Update *self) {
    return self->value;
}
static Node *RayQuery_query(RayQuery *self) {
    return self->query;
}
static BasicBlock *RayQuery_on_triangle_hit(RayQuery *self) {
    return self->on_triangle_hit;
}
static BasicBlock *RayQuery_on_procedural_hit(RayQuery *self) {
    return self->on_procedural_hit;
}
static BasicBlock *RevAutodiff_body(RevAutodiff *self) {
    return self->body;
}
static BasicBlock *FwdAutodiff_body(FwdAutodiff *self) {
    return self->body;
}
static BufferBinding *Binding_as_BufferBinding(Binding *self) {
    return self->as<BufferBinding>();
}
static TextureBinding *Binding_as_TextureBinding(Binding *self) {
    return self->as<TextureBinding>();
}
static BindlessArrayBinding *Binding_as_BindlessArrayBinding(Binding *self) {
    return self->as<BindlessArrayBinding>();
}
static AccelBinding *Binding_as_AccelBinding(Binding *self) {
    return self->as<AccelBinding>();
}
static BindingTag Binding_tag(Binding *self) {
    return self->tag();
}
static uint64_t BufferBinding_handle(BufferBinding *self) {
    return self->handle;
}
static uint64_t BufferBinding_offset(BufferBinding *self) {
    return self->offset;
}
static uint64_t BufferBinding_size(BufferBinding *self) {
    return self->size;
}
static uint64_t TextureBinding_handle(TextureBinding *self) {
    return self->handle;
}
static uint64_t TextureBinding_level(TextureBinding *self) {
    return self->level;
}
static uint64_t BindlessArrayBinding_handle(BindlessArrayBinding *self) {
    return self->handle;
}
static uint64_t AccelBinding_handle(AccelBinding *self) {
    return self->handle;
}
extern "C" LC_IR_API IrV2BindingTable lc_ir_v2_binding_table() {
    return {
        Func_as_Zero,
        Func_as_One,
        Func_as_Assume,
        Func_as_Unreachable,
        Func_as_ThreadId,
        Func_as_BlockId,
        Func_as_WarpSize,
        Func_as_WarpLaneId,
        Func_as_DispatchId,
        Func_as_DispatchSize,
        Func_as_PropagateGradient,
        Func_as_OutputGradient,
        Func_as_RequiresGradient,
        Func_as_Backward,
        Func_as_Gradient,
        Func_as_AccGrad,
        Func_as_Detach,
        Func_as_RayTracingInstanceTransform,
        Func_as_RayTracingInstanceVisibilityMask,
        Func_as_RayTracingInstanceUserId,
        Func_as_RayTracingSetInstanceTransform,
        Func_as_RayTracingSetInstanceOpacity,
        Func_as_RayTracingSetInstanceVisibility,
        Func_as_RayTracingSetInstanceUserId,
        Func_as_RayTracingTraceClosest,
        Func_as_RayTracingTraceAny,
        Func_as_RayTracingQueryAll,
        Func_as_RayTracingQueryAny,
        Func_as_RayQueryWorldSpaceRay,
        Func_as_RayQueryProceduralCandidateHit,
        Func_as_RayQueryTriangleCandidateHit,
        Func_as_RayQueryCommittedHit,
        Func_as_RayQueryCommitTriangle,
        Func_as_RayQueryCommitdProcedural,
        Func_as_RayQueryTerminate,
        Func_as_Load,
        Func_as_Cast,
        Func_as_BitCast,
        Func_as_Add,
        Func_as_Sub,
        Func_as_Mul,
        Func_as_Div,
        Func_as_Rem,
        Func_as_BitAnd,
        Func_as_BitOr,
        Func_as_BitXor,
        Func_as_Shl,
        Func_as_Shr,
        Func_as_RotRight,
        Func_as_RotLeft,
        Func_as_Eq,
        Func_as_Ne,
        Func_as_Lt,
        Func_as_Le,
        Func_as_Gt,
        Func_as_Ge,
        Func_as_MatCompMul,
        Func_as_Neg,
        Func_as_Not,
        Func_as_BitNot,
        Func_as_All,
        Func_as_Any,
        Func_as_Select,
        Func_as_Clamp,
        Func_as_Lerp,
        Func_as_Step,
        Func_as_Saturate,
        Func_as_SmoothStep,
        Func_as_Abs,
        Func_as_Min,
        Func_as_Max,
        Func_as_ReduceSum,
        Func_as_ReduceProd,
        Func_as_ReduceMin,
        Func_as_ReduceMax,
        Func_as_Clz,
        Func_as_Ctz,
        Func_as_PopCount,
        Func_as_Reverse,
        Func_as_IsInf,
        Func_as_IsNan,
        Func_as_Acos,
        Func_as_Acosh,
        Func_as_Asin,
        Func_as_Asinh,
        Func_as_Atan,
        Func_as_Atan2,
        Func_as_Atanh,
        Func_as_Cos,
        Func_as_Cosh,
        Func_as_Sin,
        Func_as_Sinh,
        Func_as_Tan,
        Func_as_Tanh,
        Func_as_Exp,
        Func_as_Exp2,
        Func_as_Exp10,
        Func_as_Log,
        Func_as_Log2,
        Func_as_Log10,
        Func_as_Powi,
        Func_as_Powf,
        Func_as_Sqrt,
        Func_as_Rsqrt,
        Func_as_Ceil,
        Func_as_Floor,
        Func_as_Fract,
        Func_as_Trunc,
        Func_as_Round,
        Func_as_Fma,
        Func_as_Copysign,
        Func_as_Cross,
        Func_as_Dot,
        Func_as_OuterProduct,
        Func_as_Length,
        Func_as_LengthSquared,
        Func_as_Normalize,
        Func_as_Faceforward,
        Func_as_Distance,
        Func_as_Reflect,
        Func_as_Determinant,
        Func_as_Transpose,
        Func_as_Inverse,
        Func_as_WarpIsFirstActiveLane,
        Func_as_WarpFirstActiveLane,
        Func_as_WarpActiveAllEqual,
        Func_as_WarpActiveBitAnd,
        Func_as_WarpActiveBitOr,
        Func_as_WarpActiveBitXor,
        Func_as_WarpActiveCountBits,
        Func_as_WarpActiveMax,
        Func_as_WarpActiveMin,
        Func_as_WarpActiveProduct,
        Func_as_WarpActiveSum,
        Func_as_WarpActiveAll,
        Func_as_WarpActiveAny,
        Func_as_WarpActiveBitMask,
        Func_as_WarpPrefixCountBits,
        Func_as_WarpPrefixSum,
        Func_as_WarpPrefixProduct,
        Func_as_WarpReadLaneAt,
        Func_as_WarpReadFirstLane,
        Func_as_SynchronizeBlock,
        Func_as_AtomicExchange,
        Func_as_AtomicCompareExchange,
        Func_as_AtomicFetchAdd,
        Func_as_AtomicFetchSub,
        Func_as_AtomicFetchAnd,
        Func_as_AtomicFetchOr,
        Func_as_AtomicFetchXor,
        Func_as_AtomicFetchMin,
        Func_as_AtomicFetchMax,
        Func_as_BufferWrite,
        Func_as_BufferRead,
        Func_as_BufferSize,
        Func_as_ByteBufferWrite,
        Func_as_ByteBufferRead,
        Func_as_ByteBufferSize,
        Func_as_Texture2dRead,
        Func_as_Texture2dWrite,
        Func_as_Texture2dSize,
        Func_as_Texture3dRead,
        Func_as_Texture3dWrite,
        Func_as_Texture3dSize,
        Func_as_BindlessTexture2dSample,
        Func_as_BindlessTexture2dSampleLevel,
        Func_as_BindlessTexture2dSampleGrad,
        Func_as_BindlessTexture2dSampleGradLevel,
        Func_as_BindlessTexture2dRead,
        Func_as_BindlessTexture2dSize,
        Func_as_BindlessTexture2dSizeLevel,
        Func_as_BindlessTexture3dSample,
        Func_as_BindlessTexture3dSampleLevel,
        Func_as_BindlessTexture3dSampleGrad,
        Func_as_BindlessTexture3dSampleGradLevel,
        Func_as_BindlessTexture3dRead,
        Func_as_BindlessTexture3dSize,
        Func_as_BindlessTexture3dSizeLevel,
        Func_as_BindlessBufferWrite,
        Func_as_BindlessBufferRead,
        Func_as_BindlessBufferSize,
        Func_as_BindlessByteBufferWrite,
        Func_as_BindlessByteBufferRead,
        Func_as_BindlessByteBufferSize,
        Func_as_Vec,
        Func_as_Vec2,
        Func_as_Vec3,
        Func_as_Vec4,
        Func_as_Permute,
        Func_as_GetElementPtr,
        Func_as_ExtractElement,
        Func_as_InsertElement,
        Func_as_Array,
        Func_as_Struct,
        Func_as_MatFull,
        Func_as_Mat2,
        Func_as_Mat3,
        Func_as_Mat4,
        Func_as_BindlessAtomicExchange,
        Func_as_BindlessAtomicCompareExchange,
        Func_as_BindlessAtomicFetchAdd,
        Func_as_BindlessAtomicFetchSub,
        Func_as_BindlessAtomicFetchAnd,
        Func_as_BindlessAtomicFetchOr,
        Func_as_BindlessAtomicFetchXor,
        Func_as_BindlessAtomicFetchMin,
        Func_as_BindlessAtomicFetchMax,
        Func_as_Callable,
        Func_as_CpuExt,
        Func_as_ShaderExecutionReorder,
        Func_tag,
        Assume_msg,
        Unreachable_msg,
        BindlessAtomicExchange_ty,
        BindlessAtomicCompareExchange_ty,
        BindlessAtomicFetchAdd_ty,
        BindlessAtomicFetchSub_ty,
        BindlessAtomicFetchAnd_ty,
        BindlessAtomicFetchOr_ty,
        BindlessAtomicFetchXor_ty,
        BindlessAtomicFetchMin_ty,
        BindlessAtomicFetchMax_ty,
        Callable_module,
        CpuExt_f,
        Instruction_as_Buffer,
        Instruction_as_Texture2d,
        Instruction_as_Texture3d,
        Instruction_as_BindlessArray,
        Instruction_as_Accel,
        Instruction_as_Shared,
        Instruction_as_Uniform,
        Instruction_as_Argument,
        Instruction_as_Constant,
        Instruction_as_Call,
        Instruction_as_Phi,
        Instruction_as_BasicBlockSentinel,
        Instruction_as_If,
        Instruction_as_GenericLoop,
        Instruction_as_Switch,
        Instruction_as_Local,
        Instruction_as_Break,
        Instruction_as_Continue,
        Instruction_as_Return,
        Instruction_as_Print,
        Instruction_as_Update,
        Instruction_as_RayQuery,
        Instruction_as_RevAutodiff,
        Instruction_as_FwdAutodiff,
        Instruction_tag,
        Argument_by_value,
        Constant_ty,
        Constant_value,
        Call_func,
        Call_args,
        Phi_incomings,
        If_cond,
        If_true_branch,
        If_false_branch,
        GenericLoop_prepare,
        GenericLoop_cond,
        GenericLoop_body,
        GenericLoop_update,
        Switch_value,
        Switch_cases,
        Switch_default_,
        Local_init,
        Return_value,
        Print_fmt,
        Print_args,
        Update_var,
        Update_value,
        RayQuery_query,
        RayQuery_on_triangle_hit,
        RayQuery_on_procedural_hit,
        RevAutodiff_body,
        FwdAutodiff_body,
        Binding_as_BufferBinding,
        Binding_as_TextureBinding,
        Binding_as_BindlessArrayBinding,
        Binding_as_AccelBinding,
        Binding_tag,
        BufferBinding_handle,
        BufferBinding_offset,
        BufferBinding_size,
        TextureBinding_handle,
        TextureBinding_level,
        BindlessArrayBinding_handle,
        AccelBinding_handle,
    };
}
}// namespace luisa::compute::ir_v2
