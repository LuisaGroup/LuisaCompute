#include <luisa/ir_v2/ir_v2_api.h>
namespace luisa::compute::ir_v2 {
static ZeroFn *Func_as_ZeroFn(Func *self) {
    return self->as<ZeroFn>();
}
static OneFn *Func_as_OneFn(Func *self) {
    return self->as<OneFn>();
}
static AssumeFn *Func_as_AssumeFn(Func *self) {
    return self->as<AssumeFn>();
}
static UnreachableFn *Func_as_UnreachableFn(Func *self) {
    return self->as<UnreachableFn>();
}
static ThreadIdFn *Func_as_ThreadIdFn(Func *self) {
    return self->as<ThreadIdFn>();
}
static BlockIdFn *Func_as_BlockIdFn(Func *self) {
    return self->as<BlockIdFn>();
}
static WarpSizeFn *Func_as_WarpSizeFn(Func *self) {
    return self->as<WarpSizeFn>();
}
static WarpLaneIdFn *Func_as_WarpLaneIdFn(Func *self) {
    return self->as<WarpLaneIdFn>();
}
static DispatchIdFn *Func_as_DispatchIdFn(Func *self) {
    return self->as<DispatchIdFn>();
}
static DispatchSizeFn *Func_as_DispatchSizeFn(Func *self) {
    return self->as<DispatchSizeFn>();
}
static PropagateGradientFn *Func_as_PropagateGradientFn(Func *self) {
    return self->as<PropagateGradientFn>();
}
static OutputGradientFn *Func_as_OutputGradientFn(Func *self) {
    return self->as<OutputGradientFn>();
}
static RequiresGradientFn *Func_as_RequiresGradientFn(Func *self) {
    return self->as<RequiresGradientFn>();
}
static BackwardFn *Func_as_BackwardFn(Func *self) {
    return self->as<BackwardFn>();
}
static GradientFn *Func_as_GradientFn(Func *self) {
    return self->as<GradientFn>();
}
static AccGradFn *Func_as_AccGradFn(Func *self) {
    return self->as<AccGradFn>();
}
static DetachFn *Func_as_DetachFn(Func *self) {
    return self->as<DetachFn>();
}
static RayTracingInstanceTransformFn *Func_as_RayTracingInstanceTransformFn(Func *self) {
    return self->as<RayTracingInstanceTransformFn>();
}
static RayTracingInstanceVisibilityMaskFn *Func_as_RayTracingInstanceVisibilityMaskFn(Func *self) {
    return self->as<RayTracingInstanceVisibilityMaskFn>();
}
static RayTracingInstanceUserIdFn *Func_as_RayTracingInstanceUserIdFn(Func *self) {
    return self->as<RayTracingInstanceUserIdFn>();
}
static RayTracingSetInstanceTransformFn *Func_as_RayTracingSetInstanceTransformFn(Func *self) {
    return self->as<RayTracingSetInstanceTransformFn>();
}
static RayTracingSetInstanceOpacityFn *Func_as_RayTracingSetInstanceOpacityFn(Func *self) {
    return self->as<RayTracingSetInstanceOpacityFn>();
}
static RayTracingSetInstanceVisibilityFn *Func_as_RayTracingSetInstanceVisibilityFn(Func *self) {
    return self->as<RayTracingSetInstanceVisibilityFn>();
}
static RayTracingSetInstanceUserIdFn *Func_as_RayTracingSetInstanceUserIdFn(Func *self) {
    return self->as<RayTracingSetInstanceUserIdFn>();
}
static RayTracingTraceClosestFn *Func_as_RayTracingTraceClosestFn(Func *self) {
    return self->as<RayTracingTraceClosestFn>();
}
static RayTracingTraceAnyFn *Func_as_RayTracingTraceAnyFn(Func *self) {
    return self->as<RayTracingTraceAnyFn>();
}
static RayTracingQueryAllFn *Func_as_RayTracingQueryAllFn(Func *self) {
    return self->as<RayTracingQueryAllFn>();
}
static RayTracingQueryAnyFn *Func_as_RayTracingQueryAnyFn(Func *self) {
    return self->as<RayTracingQueryAnyFn>();
}
static RayQueryWorldSpaceRayFn *Func_as_RayQueryWorldSpaceRayFn(Func *self) {
    return self->as<RayQueryWorldSpaceRayFn>();
}
static RayQueryProceduralCandidateHitFn *Func_as_RayQueryProceduralCandidateHitFn(Func *self) {
    return self->as<RayQueryProceduralCandidateHitFn>();
}
static RayQueryTriangleCandidateHitFn *Func_as_RayQueryTriangleCandidateHitFn(Func *self) {
    return self->as<RayQueryTriangleCandidateHitFn>();
}
static RayQueryCommittedHitFn *Func_as_RayQueryCommittedHitFn(Func *self) {
    return self->as<RayQueryCommittedHitFn>();
}
static RayQueryCommitTriangleFn *Func_as_RayQueryCommitTriangleFn(Func *self) {
    return self->as<RayQueryCommitTriangleFn>();
}
static RayQueryCommitdProceduralFn *Func_as_RayQueryCommitdProceduralFn(Func *self) {
    return self->as<RayQueryCommitdProceduralFn>();
}
static RayQueryTerminateFn *Func_as_RayQueryTerminateFn(Func *self) {
    return self->as<RayQueryTerminateFn>();
}
static LoadFn *Func_as_LoadFn(Func *self) {
    return self->as<LoadFn>();
}
static CastFn *Func_as_CastFn(Func *self) {
    return self->as<CastFn>();
}
static BitCastFn *Func_as_BitCastFn(Func *self) {
    return self->as<BitCastFn>();
}
static AddFn *Func_as_AddFn(Func *self) {
    return self->as<AddFn>();
}
static SubFn *Func_as_SubFn(Func *self) {
    return self->as<SubFn>();
}
static MulFn *Func_as_MulFn(Func *self) {
    return self->as<MulFn>();
}
static DivFn *Func_as_DivFn(Func *self) {
    return self->as<DivFn>();
}
static RemFn *Func_as_RemFn(Func *self) {
    return self->as<RemFn>();
}
static BitAndFn *Func_as_BitAndFn(Func *self) {
    return self->as<BitAndFn>();
}
static BitOrFn *Func_as_BitOrFn(Func *self) {
    return self->as<BitOrFn>();
}
static BitXorFn *Func_as_BitXorFn(Func *self) {
    return self->as<BitXorFn>();
}
static ShlFn *Func_as_ShlFn(Func *self) {
    return self->as<ShlFn>();
}
static ShrFn *Func_as_ShrFn(Func *self) {
    return self->as<ShrFn>();
}
static RotRightFn *Func_as_RotRightFn(Func *self) {
    return self->as<RotRightFn>();
}
static RotLeftFn *Func_as_RotLeftFn(Func *self) {
    return self->as<RotLeftFn>();
}
static EqFn *Func_as_EqFn(Func *self) {
    return self->as<EqFn>();
}
static NeFn *Func_as_NeFn(Func *self) {
    return self->as<NeFn>();
}
static LtFn *Func_as_LtFn(Func *self) {
    return self->as<LtFn>();
}
static LeFn *Func_as_LeFn(Func *self) {
    return self->as<LeFn>();
}
static GtFn *Func_as_GtFn(Func *self) {
    return self->as<GtFn>();
}
static GeFn *Func_as_GeFn(Func *self) {
    return self->as<GeFn>();
}
static MatCompMulFn *Func_as_MatCompMulFn(Func *self) {
    return self->as<MatCompMulFn>();
}
static NegFn *Func_as_NegFn(Func *self) {
    return self->as<NegFn>();
}
static NotFn *Func_as_NotFn(Func *self) {
    return self->as<NotFn>();
}
static BitNotFn *Func_as_BitNotFn(Func *self) {
    return self->as<BitNotFn>();
}
static AllFn *Func_as_AllFn(Func *self) {
    return self->as<AllFn>();
}
static AnyFn *Func_as_AnyFn(Func *self) {
    return self->as<AnyFn>();
}
static SelectFn *Func_as_SelectFn(Func *self) {
    return self->as<SelectFn>();
}
static ClampFn *Func_as_ClampFn(Func *self) {
    return self->as<ClampFn>();
}
static LerpFn *Func_as_LerpFn(Func *self) {
    return self->as<LerpFn>();
}
static StepFn *Func_as_StepFn(Func *self) {
    return self->as<StepFn>();
}
static SaturateFn *Func_as_SaturateFn(Func *self) {
    return self->as<SaturateFn>();
}
static SmoothStepFn *Func_as_SmoothStepFn(Func *self) {
    return self->as<SmoothStepFn>();
}
static AbsFn *Func_as_AbsFn(Func *self) {
    return self->as<AbsFn>();
}
static MinFn *Func_as_MinFn(Func *self) {
    return self->as<MinFn>();
}
static MaxFn *Func_as_MaxFn(Func *self) {
    return self->as<MaxFn>();
}
static ReduceSumFn *Func_as_ReduceSumFn(Func *self) {
    return self->as<ReduceSumFn>();
}
static ReduceProdFn *Func_as_ReduceProdFn(Func *self) {
    return self->as<ReduceProdFn>();
}
static ReduceMinFn *Func_as_ReduceMinFn(Func *self) {
    return self->as<ReduceMinFn>();
}
static ReduceMaxFn *Func_as_ReduceMaxFn(Func *self) {
    return self->as<ReduceMaxFn>();
}
static ClzFn *Func_as_ClzFn(Func *self) {
    return self->as<ClzFn>();
}
static CtzFn *Func_as_CtzFn(Func *self) {
    return self->as<CtzFn>();
}
static PopCountFn *Func_as_PopCountFn(Func *self) {
    return self->as<PopCountFn>();
}
static ReverseFn *Func_as_ReverseFn(Func *self) {
    return self->as<ReverseFn>();
}
static IsInfFn *Func_as_IsInfFn(Func *self) {
    return self->as<IsInfFn>();
}
static IsNanFn *Func_as_IsNanFn(Func *self) {
    return self->as<IsNanFn>();
}
static AcosFn *Func_as_AcosFn(Func *self) {
    return self->as<AcosFn>();
}
static AcoshFn *Func_as_AcoshFn(Func *self) {
    return self->as<AcoshFn>();
}
static AsinFn *Func_as_AsinFn(Func *self) {
    return self->as<AsinFn>();
}
static AsinhFn *Func_as_AsinhFn(Func *self) {
    return self->as<AsinhFn>();
}
static AtanFn *Func_as_AtanFn(Func *self) {
    return self->as<AtanFn>();
}
static Atan2Fn *Func_as_Atan2Fn(Func *self) {
    return self->as<Atan2Fn>();
}
static AtanhFn *Func_as_AtanhFn(Func *self) {
    return self->as<AtanhFn>();
}
static CosFn *Func_as_CosFn(Func *self) {
    return self->as<CosFn>();
}
static CoshFn *Func_as_CoshFn(Func *self) {
    return self->as<CoshFn>();
}
static SinFn *Func_as_SinFn(Func *self) {
    return self->as<SinFn>();
}
static SinhFn *Func_as_SinhFn(Func *self) {
    return self->as<SinhFn>();
}
static TanFn *Func_as_TanFn(Func *self) {
    return self->as<TanFn>();
}
static TanhFn *Func_as_TanhFn(Func *self) {
    return self->as<TanhFn>();
}
static ExpFn *Func_as_ExpFn(Func *self) {
    return self->as<ExpFn>();
}
static Exp2Fn *Func_as_Exp2Fn(Func *self) {
    return self->as<Exp2Fn>();
}
static Exp10Fn *Func_as_Exp10Fn(Func *self) {
    return self->as<Exp10Fn>();
}
static LogFn *Func_as_LogFn(Func *self) {
    return self->as<LogFn>();
}
static Log2Fn *Func_as_Log2Fn(Func *self) {
    return self->as<Log2Fn>();
}
static Log10Fn *Func_as_Log10Fn(Func *self) {
    return self->as<Log10Fn>();
}
static PowiFn *Func_as_PowiFn(Func *self) {
    return self->as<PowiFn>();
}
static PowfFn *Func_as_PowfFn(Func *self) {
    return self->as<PowfFn>();
}
static SqrtFn *Func_as_SqrtFn(Func *self) {
    return self->as<SqrtFn>();
}
static RsqrtFn *Func_as_RsqrtFn(Func *self) {
    return self->as<RsqrtFn>();
}
static CeilFn *Func_as_CeilFn(Func *self) {
    return self->as<CeilFn>();
}
static FloorFn *Func_as_FloorFn(Func *self) {
    return self->as<FloorFn>();
}
static FractFn *Func_as_FractFn(Func *self) {
    return self->as<FractFn>();
}
static TruncFn *Func_as_TruncFn(Func *self) {
    return self->as<TruncFn>();
}
static RoundFn *Func_as_RoundFn(Func *self) {
    return self->as<RoundFn>();
}
static FmaFn *Func_as_FmaFn(Func *self) {
    return self->as<FmaFn>();
}
static CopysignFn *Func_as_CopysignFn(Func *self) {
    return self->as<CopysignFn>();
}
static CrossFn *Func_as_CrossFn(Func *self) {
    return self->as<CrossFn>();
}
static DotFn *Func_as_DotFn(Func *self) {
    return self->as<DotFn>();
}
static OuterProductFn *Func_as_OuterProductFn(Func *self) {
    return self->as<OuterProductFn>();
}
static LengthFn *Func_as_LengthFn(Func *self) {
    return self->as<LengthFn>();
}
static LengthSquaredFn *Func_as_LengthSquaredFn(Func *self) {
    return self->as<LengthSquaredFn>();
}
static NormalizeFn *Func_as_NormalizeFn(Func *self) {
    return self->as<NormalizeFn>();
}
static FaceforwardFn *Func_as_FaceforwardFn(Func *self) {
    return self->as<FaceforwardFn>();
}
static DistanceFn *Func_as_DistanceFn(Func *self) {
    return self->as<DistanceFn>();
}
static ReflectFn *Func_as_ReflectFn(Func *self) {
    return self->as<ReflectFn>();
}
static DeterminantFn *Func_as_DeterminantFn(Func *self) {
    return self->as<DeterminantFn>();
}
static TransposeFn *Func_as_TransposeFn(Func *self) {
    return self->as<TransposeFn>();
}
static InverseFn *Func_as_InverseFn(Func *self) {
    return self->as<InverseFn>();
}
static WarpIsFirstActiveLaneFn *Func_as_WarpIsFirstActiveLaneFn(Func *self) {
    return self->as<WarpIsFirstActiveLaneFn>();
}
static WarpFirstActiveLaneFn *Func_as_WarpFirstActiveLaneFn(Func *self) {
    return self->as<WarpFirstActiveLaneFn>();
}
static WarpActiveAllEqualFn *Func_as_WarpActiveAllEqualFn(Func *self) {
    return self->as<WarpActiveAllEqualFn>();
}
static WarpActiveBitAndFn *Func_as_WarpActiveBitAndFn(Func *self) {
    return self->as<WarpActiveBitAndFn>();
}
static WarpActiveBitOrFn *Func_as_WarpActiveBitOrFn(Func *self) {
    return self->as<WarpActiveBitOrFn>();
}
static WarpActiveBitXorFn *Func_as_WarpActiveBitXorFn(Func *self) {
    return self->as<WarpActiveBitXorFn>();
}
static WarpActiveCountBitsFn *Func_as_WarpActiveCountBitsFn(Func *self) {
    return self->as<WarpActiveCountBitsFn>();
}
static WarpActiveMaxFn *Func_as_WarpActiveMaxFn(Func *self) {
    return self->as<WarpActiveMaxFn>();
}
static WarpActiveMinFn *Func_as_WarpActiveMinFn(Func *self) {
    return self->as<WarpActiveMinFn>();
}
static WarpActiveProductFn *Func_as_WarpActiveProductFn(Func *self) {
    return self->as<WarpActiveProductFn>();
}
static WarpActiveSumFn *Func_as_WarpActiveSumFn(Func *self) {
    return self->as<WarpActiveSumFn>();
}
static WarpActiveAllFn *Func_as_WarpActiveAllFn(Func *self) {
    return self->as<WarpActiveAllFn>();
}
static WarpActiveAnyFn *Func_as_WarpActiveAnyFn(Func *self) {
    return self->as<WarpActiveAnyFn>();
}
static WarpActiveBitMaskFn *Func_as_WarpActiveBitMaskFn(Func *self) {
    return self->as<WarpActiveBitMaskFn>();
}
static WarpPrefixCountBitsFn *Func_as_WarpPrefixCountBitsFn(Func *self) {
    return self->as<WarpPrefixCountBitsFn>();
}
static WarpPrefixSumFn *Func_as_WarpPrefixSumFn(Func *self) {
    return self->as<WarpPrefixSumFn>();
}
static WarpPrefixProductFn *Func_as_WarpPrefixProductFn(Func *self) {
    return self->as<WarpPrefixProductFn>();
}
static WarpReadLaneAtFn *Func_as_WarpReadLaneAtFn(Func *self) {
    return self->as<WarpReadLaneAtFn>();
}
static WarpReadFirstLaneFn *Func_as_WarpReadFirstLaneFn(Func *self) {
    return self->as<WarpReadFirstLaneFn>();
}
static SynchronizeBlockFn *Func_as_SynchronizeBlockFn(Func *self) {
    return self->as<SynchronizeBlockFn>();
}
static AtomicExchangeFn *Func_as_AtomicExchangeFn(Func *self) {
    return self->as<AtomicExchangeFn>();
}
static AtomicCompareExchangeFn *Func_as_AtomicCompareExchangeFn(Func *self) {
    return self->as<AtomicCompareExchangeFn>();
}
static AtomicFetchAddFn *Func_as_AtomicFetchAddFn(Func *self) {
    return self->as<AtomicFetchAddFn>();
}
static AtomicFetchSubFn *Func_as_AtomicFetchSubFn(Func *self) {
    return self->as<AtomicFetchSubFn>();
}
static AtomicFetchAndFn *Func_as_AtomicFetchAndFn(Func *self) {
    return self->as<AtomicFetchAndFn>();
}
static AtomicFetchOrFn *Func_as_AtomicFetchOrFn(Func *self) {
    return self->as<AtomicFetchOrFn>();
}
static AtomicFetchXorFn *Func_as_AtomicFetchXorFn(Func *self) {
    return self->as<AtomicFetchXorFn>();
}
static AtomicFetchMinFn *Func_as_AtomicFetchMinFn(Func *self) {
    return self->as<AtomicFetchMinFn>();
}
static AtomicFetchMaxFn *Func_as_AtomicFetchMaxFn(Func *self) {
    return self->as<AtomicFetchMaxFn>();
}
static BufferWriteFn *Func_as_BufferWriteFn(Func *self) {
    return self->as<BufferWriteFn>();
}
static BufferReadFn *Func_as_BufferReadFn(Func *self) {
    return self->as<BufferReadFn>();
}
static BufferSizeFn *Func_as_BufferSizeFn(Func *self) {
    return self->as<BufferSizeFn>();
}
static ByteBufferWriteFn *Func_as_ByteBufferWriteFn(Func *self) {
    return self->as<ByteBufferWriteFn>();
}
static ByteBufferReadFn *Func_as_ByteBufferReadFn(Func *self) {
    return self->as<ByteBufferReadFn>();
}
static ByteBufferSizeFn *Func_as_ByteBufferSizeFn(Func *self) {
    return self->as<ByteBufferSizeFn>();
}
static Texture2dReadFn *Func_as_Texture2dReadFn(Func *self) {
    return self->as<Texture2dReadFn>();
}
static Texture2dWriteFn *Func_as_Texture2dWriteFn(Func *self) {
    return self->as<Texture2dWriteFn>();
}
static Texture2dSizeFn *Func_as_Texture2dSizeFn(Func *self) {
    return self->as<Texture2dSizeFn>();
}
static Texture3dReadFn *Func_as_Texture3dReadFn(Func *self) {
    return self->as<Texture3dReadFn>();
}
static Texture3dWriteFn *Func_as_Texture3dWriteFn(Func *self) {
    return self->as<Texture3dWriteFn>();
}
static Texture3dSizeFn *Func_as_Texture3dSizeFn(Func *self) {
    return self->as<Texture3dSizeFn>();
}
static BindlessTexture2dSampleFn *Func_as_BindlessTexture2dSampleFn(Func *self) {
    return self->as<BindlessTexture2dSampleFn>();
}
static BindlessTexture2dSampleLevelFn *Func_as_BindlessTexture2dSampleLevelFn(Func *self) {
    return self->as<BindlessTexture2dSampleLevelFn>();
}
static BindlessTexture2dSampleGradFn *Func_as_BindlessTexture2dSampleGradFn(Func *self) {
    return self->as<BindlessTexture2dSampleGradFn>();
}
static BindlessTexture2dSampleGradLevelFn *Func_as_BindlessTexture2dSampleGradLevelFn(Func *self) {
    return self->as<BindlessTexture2dSampleGradLevelFn>();
}
static BindlessTexture2dReadFn *Func_as_BindlessTexture2dReadFn(Func *self) {
    return self->as<BindlessTexture2dReadFn>();
}
static BindlessTexture2dSizeFn *Func_as_BindlessTexture2dSizeFn(Func *self) {
    return self->as<BindlessTexture2dSizeFn>();
}
static BindlessTexture2dSizeLevelFn *Func_as_BindlessTexture2dSizeLevelFn(Func *self) {
    return self->as<BindlessTexture2dSizeLevelFn>();
}
static BindlessTexture3dSampleFn *Func_as_BindlessTexture3dSampleFn(Func *self) {
    return self->as<BindlessTexture3dSampleFn>();
}
static BindlessTexture3dSampleLevelFn *Func_as_BindlessTexture3dSampleLevelFn(Func *self) {
    return self->as<BindlessTexture3dSampleLevelFn>();
}
static BindlessTexture3dSampleGradFn *Func_as_BindlessTexture3dSampleGradFn(Func *self) {
    return self->as<BindlessTexture3dSampleGradFn>();
}
static BindlessTexture3dSampleGradLevelFn *Func_as_BindlessTexture3dSampleGradLevelFn(Func *self) {
    return self->as<BindlessTexture3dSampleGradLevelFn>();
}
static BindlessTexture3dReadFn *Func_as_BindlessTexture3dReadFn(Func *self) {
    return self->as<BindlessTexture3dReadFn>();
}
static BindlessTexture3dSizeFn *Func_as_BindlessTexture3dSizeFn(Func *self) {
    return self->as<BindlessTexture3dSizeFn>();
}
static BindlessTexture3dSizeLevelFn *Func_as_BindlessTexture3dSizeLevelFn(Func *self) {
    return self->as<BindlessTexture3dSizeLevelFn>();
}
static BindlessBufferWriteFn *Func_as_BindlessBufferWriteFn(Func *self) {
    return self->as<BindlessBufferWriteFn>();
}
static BindlessBufferReadFn *Func_as_BindlessBufferReadFn(Func *self) {
    return self->as<BindlessBufferReadFn>();
}
static BindlessBufferSizeFn *Func_as_BindlessBufferSizeFn(Func *self) {
    return self->as<BindlessBufferSizeFn>();
}
static BindlessByteBufferWriteFn *Func_as_BindlessByteBufferWriteFn(Func *self) {
    return self->as<BindlessByteBufferWriteFn>();
}
static BindlessByteBufferReadFn *Func_as_BindlessByteBufferReadFn(Func *self) {
    return self->as<BindlessByteBufferReadFn>();
}
static BindlessByteBufferSizeFn *Func_as_BindlessByteBufferSizeFn(Func *self) {
    return self->as<BindlessByteBufferSizeFn>();
}
static VecFn *Func_as_VecFn(Func *self) {
    return self->as<VecFn>();
}
static Vec2Fn *Func_as_Vec2Fn(Func *self) {
    return self->as<Vec2Fn>();
}
static Vec3Fn *Func_as_Vec3Fn(Func *self) {
    return self->as<Vec3Fn>();
}
static Vec4Fn *Func_as_Vec4Fn(Func *self) {
    return self->as<Vec4Fn>();
}
static PermuteFn *Func_as_PermuteFn(Func *self) {
    return self->as<PermuteFn>();
}
static GetElementPtrFn *Func_as_GetElementPtrFn(Func *self) {
    return self->as<GetElementPtrFn>();
}
static ExtractElementFn *Func_as_ExtractElementFn(Func *self) {
    return self->as<ExtractElementFn>();
}
static InsertElementFn *Func_as_InsertElementFn(Func *self) {
    return self->as<InsertElementFn>();
}
static ArrayFn *Func_as_ArrayFn(Func *self) {
    return self->as<ArrayFn>();
}
static StructFn *Func_as_StructFn(Func *self) {
    return self->as<StructFn>();
}
static MatFullFn *Func_as_MatFullFn(Func *self) {
    return self->as<MatFullFn>();
}
static Mat2Fn *Func_as_Mat2Fn(Func *self) {
    return self->as<Mat2Fn>();
}
static Mat3Fn *Func_as_Mat3Fn(Func *self) {
    return self->as<Mat3Fn>();
}
static Mat4Fn *Func_as_Mat4Fn(Func *self) {
    return self->as<Mat4Fn>();
}
static BindlessAtomicExchangeFn *Func_as_BindlessAtomicExchangeFn(Func *self) {
    return self->as<BindlessAtomicExchangeFn>();
}
static BindlessAtomicCompareExchangeFn *Func_as_BindlessAtomicCompareExchangeFn(Func *self) {
    return self->as<BindlessAtomicCompareExchangeFn>();
}
static BindlessAtomicFetchAddFn *Func_as_BindlessAtomicFetchAddFn(Func *self) {
    return self->as<BindlessAtomicFetchAddFn>();
}
static BindlessAtomicFetchSubFn *Func_as_BindlessAtomicFetchSubFn(Func *self) {
    return self->as<BindlessAtomicFetchSubFn>();
}
static BindlessAtomicFetchAndFn *Func_as_BindlessAtomicFetchAndFn(Func *self) {
    return self->as<BindlessAtomicFetchAndFn>();
}
static BindlessAtomicFetchOrFn *Func_as_BindlessAtomicFetchOrFn(Func *self) {
    return self->as<BindlessAtomicFetchOrFn>();
}
static BindlessAtomicFetchXorFn *Func_as_BindlessAtomicFetchXorFn(Func *self) {
    return self->as<BindlessAtomicFetchXorFn>();
}
static BindlessAtomicFetchMinFn *Func_as_BindlessAtomicFetchMinFn(Func *self) {
    return self->as<BindlessAtomicFetchMinFn>();
}
static BindlessAtomicFetchMaxFn *Func_as_BindlessAtomicFetchMaxFn(Func *self) {
    return self->as<BindlessAtomicFetchMaxFn>();
}
static CallableFn *Func_as_CallableFn(Func *self) {
    return self->as<CallableFn>();
}
static CpuExtFn *Func_as_CpuExtFn(Func *self) {
    return self->as<CpuExtFn>();
}
static ShaderExecutionReorderFn *Func_as_ShaderExecutionReorderFn(Func *self) {
    return self->as<ShaderExecutionReorderFn>();
}
static FuncTag Func_tag(Func *self) {
    return self->tag();
}
static Slice<const char> AssumeFn_msg(AssumeFn *self) {
    return self->msg;
}
static void AssumeFn_set_msg(AssumeFn *self, Slice<const char> value) {
    self->msg = value.to_string();
}
static Slice<const char> UnreachableFn_msg(UnreachableFn *self) {
    return self->msg;
}
static void UnreachableFn_set_msg(UnreachableFn *self, Slice<const char> value) {
    self->msg = value.to_string();
}
static const Type *BindlessAtomicExchangeFn_ty(BindlessAtomicExchangeFn *self) {
    return self->ty;
}
static void BindlessAtomicExchangeFn_set_ty(BindlessAtomicExchangeFn *self, const Type *value) {
    self->ty = value;
}
static const Type *BindlessAtomicCompareExchangeFn_ty(BindlessAtomicCompareExchangeFn *self) {
    return self->ty;
}
static void BindlessAtomicCompareExchangeFn_set_ty(BindlessAtomicCompareExchangeFn *self, const Type *value) {
    self->ty = value;
}
static const Type *BindlessAtomicFetchAddFn_ty(BindlessAtomicFetchAddFn *self) {
    return self->ty;
}
static void BindlessAtomicFetchAddFn_set_ty(BindlessAtomicFetchAddFn *self, const Type *value) {
    self->ty = value;
}
static const Type *BindlessAtomicFetchSubFn_ty(BindlessAtomicFetchSubFn *self) {
    return self->ty;
}
static void BindlessAtomicFetchSubFn_set_ty(BindlessAtomicFetchSubFn *self, const Type *value) {
    self->ty = value;
}
static const Type *BindlessAtomicFetchAndFn_ty(BindlessAtomicFetchAndFn *self) {
    return self->ty;
}
static void BindlessAtomicFetchAndFn_set_ty(BindlessAtomicFetchAndFn *self, const Type *value) {
    self->ty = value;
}
static const Type *BindlessAtomicFetchOrFn_ty(BindlessAtomicFetchOrFn *self) {
    return self->ty;
}
static void BindlessAtomicFetchOrFn_set_ty(BindlessAtomicFetchOrFn *self, const Type *value) {
    self->ty = value;
}
static const Type *BindlessAtomicFetchXorFn_ty(BindlessAtomicFetchXorFn *self) {
    return self->ty;
}
static void BindlessAtomicFetchXorFn_set_ty(BindlessAtomicFetchXorFn *self, const Type *value) {
    self->ty = value;
}
static const Type *BindlessAtomicFetchMinFn_ty(BindlessAtomicFetchMinFn *self) {
    return self->ty;
}
static void BindlessAtomicFetchMinFn_set_ty(BindlessAtomicFetchMinFn *self, const Type *value) {
    self->ty = value;
}
static const Type *BindlessAtomicFetchMaxFn_ty(BindlessAtomicFetchMaxFn *self) {
    return self->ty;
}
static void BindlessAtomicFetchMaxFn_set_ty(BindlessAtomicFetchMaxFn *self, const Type *value) {
    self->ty = value;
}
static CallableModule *CallableFn_module(CallableFn *self) {
    return self->module.get();
}
static void CallableFn_set_module(CallableFn *self, CallableModule *value) {
    self->module = luisa::static_pointer_cast<std::decay_t<decltype(self->module)>::element_type>(value->shared_from_this());
}
static CpuExternFn CpuExtFn_f(CpuExtFn *self) {
    return self->f;
}
static void CpuExtFn_set_f(CpuExtFn *self, CpuExternFn value) {
    self->f = value;
}
static BufferInst *Instruction_as_BufferInst(Instruction *self) {
    return self->as<BufferInst>();
}
static Texture2dInst *Instruction_as_Texture2dInst(Instruction *self) {
    return self->as<Texture2dInst>();
}
static Texture3dInst *Instruction_as_Texture3dInst(Instruction *self) {
    return self->as<Texture3dInst>();
}
static BindlessArrayInst *Instruction_as_BindlessArrayInst(Instruction *self) {
    return self->as<BindlessArrayInst>();
}
static AccelInst *Instruction_as_AccelInst(Instruction *self) {
    return self->as<AccelInst>();
}
static SharedInst *Instruction_as_SharedInst(Instruction *self) {
    return self->as<SharedInst>();
}
static UniformInst *Instruction_as_UniformInst(Instruction *self) {
    return self->as<UniformInst>();
}
static ArgumentInst *Instruction_as_ArgumentInst(Instruction *self) {
    return self->as<ArgumentInst>();
}
static ConstantInst *Instruction_as_ConstantInst(Instruction *self) {
    return self->as<ConstantInst>();
}
static CallInst *Instruction_as_CallInst(Instruction *self) {
    return self->as<CallInst>();
}
static PhiInst *Instruction_as_PhiInst(Instruction *self) {
    return self->as<PhiInst>();
}
static BasicBlockSentinelInst *Instruction_as_BasicBlockSentinelInst(Instruction *self) {
    return self->as<BasicBlockSentinelInst>();
}
static IfInst *Instruction_as_IfInst(Instruction *self) {
    return self->as<IfInst>();
}
static GenericLoopInst *Instruction_as_GenericLoopInst(Instruction *self) {
    return self->as<GenericLoopInst>();
}
static SwitchInst *Instruction_as_SwitchInst(Instruction *self) {
    return self->as<SwitchInst>();
}
static LocalInst *Instruction_as_LocalInst(Instruction *self) {
    return self->as<LocalInst>();
}
static BreakInst *Instruction_as_BreakInst(Instruction *self) {
    return self->as<BreakInst>();
}
static ContinueInst *Instruction_as_ContinueInst(Instruction *self) {
    return self->as<ContinueInst>();
}
static ReturnInst *Instruction_as_ReturnInst(Instruction *self) {
    return self->as<ReturnInst>();
}
static PrintInst *Instruction_as_PrintInst(Instruction *self) {
    return self->as<PrintInst>();
}
static UpdateInst *Instruction_as_UpdateInst(Instruction *self) {
    return self->as<UpdateInst>();
}
static RayQueryInst *Instruction_as_RayQueryInst(Instruction *self) {
    return self->as<RayQueryInst>();
}
static RevAutodiffInst *Instruction_as_RevAutodiffInst(Instruction *self) {
    return self->as<RevAutodiffInst>();
}
static FwdAutodiffInst *Instruction_as_FwdAutodiffInst(Instruction *self) {
    return self->as<FwdAutodiffInst>();
}
static InstructionTag Instruction_tag(Instruction *self) {
    return self->tag();
}
static bool ArgumentInst_by_value(ArgumentInst *self) {
    return self->by_value;
}
static void ArgumentInst_set_by_value(ArgumentInst *self, bool value) {
    self->by_value = value;
}
static const Type *ConstantInst_ty(ConstantInst *self) {
    return self->ty;
}
static Slice<uint8_t> ConstantInst_value(ConstantInst *self) {
    return self->value;
}
static void ConstantInst_set_ty(ConstantInst *self, const Type *value) {
    self->ty = value;
}
static void ConstantInst_set_value(ConstantInst *self, Slice<uint8_t> value) {
    self->value = value.to_vector();
}
static const Func *CallInst_func(CallInst *self) {
    return self->func;
}
static Slice<Node *> CallInst_args(CallInst *self) {
    return self->args;
}
static void CallInst_set_func(CallInst *self, const Func *value) {
    self->func = value;
}
static void CallInst_set_args(CallInst *self, Slice<Node *> value) {
    self->args = value.to_vector();
}
static Slice<PhiIncoming> PhiInst_incomings(PhiInst *self) {
    return self->incomings;
}
static void PhiInst_set_incomings(PhiInst *self, Slice<PhiIncoming> value) {
    self->incomings = value.to_vector();
}
static Node *IfInst_cond(IfInst *self) {
    return self->cond;
}
static BasicBlock *IfInst_true_branch(IfInst *self) {
    return self->true_branch;
}
static BasicBlock *IfInst_false_branch(IfInst *self) {
    return self->false_branch;
}
static void IfInst_set_cond(IfInst *self, Node *value) {
    self->cond = value;
}
static void IfInst_set_true_branch(IfInst *self, BasicBlock *value) {
    self->true_branch = value;
}
static void IfInst_set_false_branch(IfInst *self, BasicBlock *value) {
    self->false_branch = value;
}
static BasicBlock *GenericLoopInst_prepare(GenericLoopInst *self) {
    return self->prepare;
}
static Node *GenericLoopInst_cond(GenericLoopInst *self) {
    return self->cond;
}
static BasicBlock *GenericLoopInst_body(GenericLoopInst *self) {
    return self->body;
}
static BasicBlock *GenericLoopInst_update(GenericLoopInst *self) {
    return self->update;
}
static void GenericLoopInst_set_prepare(GenericLoopInst *self, BasicBlock *value) {
    self->prepare = value;
}
static void GenericLoopInst_set_cond(GenericLoopInst *self, Node *value) {
    self->cond = value;
}
static void GenericLoopInst_set_body(GenericLoopInst *self, BasicBlock *value) {
    self->body = value;
}
static void GenericLoopInst_set_update(GenericLoopInst *self, BasicBlock *value) {
    self->update = value;
}
static Node *SwitchInst_value(SwitchInst *self) {
    return self->value;
}
static Slice<SwitchCase> SwitchInst_cases(SwitchInst *self) {
    return self->cases;
}
static BasicBlock *SwitchInst_default_(SwitchInst *self) {
    return self->default_;
}
static void SwitchInst_set_value(SwitchInst *self, Node *value) {
    self->value = value;
}
static void SwitchInst_set_cases(SwitchInst *self, Slice<SwitchCase> value) {
    self->cases = value.to_vector();
}
static void SwitchInst_set_default_(SwitchInst *self, BasicBlock *value) {
    self->default_ = value;
}
static Node *LocalInst_init(LocalInst *self) {
    return self->init;
}
static void LocalInst_set_init(LocalInst *self, Node *value) {
    self->init = value;
}
static Node *ReturnInst_value(ReturnInst *self) {
    return self->value;
}
static void ReturnInst_set_value(ReturnInst *self, Node *value) {
    self->value = value;
}
static Slice<const char> PrintInst_fmt(PrintInst *self) {
    return self->fmt;
}
static Slice<Node *> PrintInst_args(PrintInst *self) {
    return self->args;
}
static void PrintInst_set_fmt(PrintInst *self, Slice<const char> value) {
    self->fmt = value.to_string();
}
static void PrintInst_set_args(PrintInst *self, Slice<Node *> value) {
    self->args = value.to_vector();
}
static Node *UpdateInst_var(UpdateInst *self) {
    return self->var;
}
static Node *UpdateInst_value(UpdateInst *self) {
    return self->value;
}
static void UpdateInst_set_var(UpdateInst *self, Node *value) {
    self->var = value;
}
static void UpdateInst_set_value(UpdateInst *self, Node *value) {
    self->value = value;
}
static Node *RayQueryInst_query(RayQueryInst *self) {
    return self->query;
}
static BasicBlock *RayQueryInst_on_triangle_hit(RayQueryInst *self) {
    return self->on_triangle_hit;
}
static BasicBlock *RayQueryInst_on_procedural_hit(RayQueryInst *self) {
    return self->on_procedural_hit;
}
static void RayQueryInst_set_query(RayQueryInst *self, Node *value) {
    self->query = value;
}
static void RayQueryInst_set_on_triangle_hit(RayQueryInst *self, BasicBlock *value) {
    self->on_triangle_hit = value;
}
static void RayQueryInst_set_on_procedural_hit(RayQueryInst *self, BasicBlock *value) {
    self->on_procedural_hit = value;
}
static BasicBlock *RevAutodiffInst_body(RevAutodiffInst *self) {
    return self->body;
}
static void RevAutodiffInst_set_body(RevAutodiffInst *self, BasicBlock *value) {
    self->body = value;
}
static BasicBlock *FwdAutodiffInst_body(FwdAutodiffInst *self) {
    return self->body;
}
static void FwdAutodiffInst_set_body(FwdAutodiffInst *self, BasicBlock *value) {
    self->body = value;
}
Func *create_func_from_tag(Pool &pool, FuncTag tag) {
    switch (tag) {
        case FuncTag::ZERO: return pool.template alloc<ZeroFn>();
        case FuncTag::ONE: return pool.template alloc<OneFn>();
        case FuncTag::ASSUME: return pool.template alloc<AssumeFn>();
        case FuncTag::UNREACHABLE: return pool.template alloc<UnreachableFn>();
        case FuncTag::THREAD_ID: return pool.template alloc<ThreadIdFn>();
        case FuncTag::BLOCK_ID: return pool.template alloc<BlockIdFn>();
        case FuncTag::WARP_SIZE: return pool.template alloc<WarpSizeFn>();
        case FuncTag::WARP_LANE_ID: return pool.template alloc<WarpLaneIdFn>();
        case FuncTag::DISPATCH_ID: return pool.template alloc<DispatchIdFn>();
        case FuncTag::DISPATCH_SIZE: return pool.template alloc<DispatchSizeFn>();
        case FuncTag::PROPAGATE_GRADIENT: return pool.template alloc<PropagateGradientFn>();
        case FuncTag::OUTPUT_GRADIENT: return pool.template alloc<OutputGradientFn>();
        case FuncTag::REQUIRES_GRADIENT: return pool.template alloc<RequiresGradientFn>();
        case FuncTag::BACKWARD: return pool.template alloc<BackwardFn>();
        case FuncTag::GRADIENT: return pool.template alloc<GradientFn>();
        case FuncTag::ACC_GRAD: return pool.template alloc<AccGradFn>();
        case FuncTag::DETACH: return pool.template alloc<DetachFn>();
        case FuncTag::RAY_TRACING_INSTANCE_TRANSFORM: return pool.template alloc<RayTracingInstanceTransformFn>();
        case FuncTag::RAY_TRACING_INSTANCE_VISIBILITY_MASK: return pool.template alloc<RayTracingInstanceVisibilityMaskFn>();
        case FuncTag::RAY_TRACING_INSTANCE_USER_ID: return pool.template alloc<RayTracingInstanceUserIdFn>();
        case FuncTag::RAY_TRACING_SET_INSTANCE_TRANSFORM: return pool.template alloc<RayTracingSetInstanceTransformFn>();
        case FuncTag::RAY_TRACING_SET_INSTANCE_OPACITY: return pool.template alloc<RayTracingSetInstanceOpacityFn>();
        case FuncTag::RAY_TRACING_SET_INSTANCE_VISIBILITY: return pool.template alloc<RayTracingSetInstanceVisibilityFn>();
        case FuncTag::RAY_TRACING_SET_INSTANCE_USER_ID: return pool.template alloc<RayTracingSetInstanceUserIdFn>();
        case FuncTag::RAY_TRACING_TRACE_CLOSEST: return pool.template alloc<RayTracingTraceClosestFn>();
        case FuncTag::RAY_TRACING_TRACE_ANY: return pool.template alloc<RayTracingTraceAnyFn>();
        case FuncTag::RAY_TRACING_QUERY_ALL: return pool.template alloc<RayTracingQueryAllFn>();
        case FuncTag::RAY_TRACING_QUERY_ANY: return pool.template alloc<RayTracingQueryAnyFn>();
        case FuncTag::RAY_QUERY_WORLD_SPACE_RAY: return pool.template alloc<RayQueryWorldSpaceRayFn>();
        case FuncTag::RAY_QUERY_PROCEDURAL_CANDIDATE_HIT: return pool.template alloc<RayQueryProceduralCandidateHitFn>();
        case FuncTag::RAY_QUERY_TRIANGLE_CANDIDATE_HIT: return pool.template alloc<RayQueryTriangleCandidateHitFn>();
        case FuncTag::RAY_QUERY_COMMITTED_HIT: return pool.template alloc<RayQueryCommittedHitFn>();
        case FuncTag::RAY_QUERY_COMMIT_TRIANGLE: return pool.template alloc<RayQueryCommitTriangleFn>();
        case FuncTag::RAY_QUERY_COMMITD_PROCEDURAL: return pool.template alloc<RayQueryCommitdProceduralFn>();
        case FuncTag::RAY_QUERY_TERMINATE: return pool.template alloc<RayQueryTerminateFn>();
        case FuncTag::LOAD: return pool.template alloc<LoadFn>();
        case FuncTag::CAST: return pool.template alloc<CastFn>();
        case FuncTag::BIT_CAST: return pool.template alloc<BitCastFn>();
        case FuncTag::ADD: return pool.template alloc<AddFn>();
        case FuncTag::SUB: return pool.template alloc<SubFn>();
        case FuncTag::MUL: return pool.template alloc<MulFn>();
        case FuncTag::DIV: return pool.template alloc<DivFn>();
        case FuncTag::REM: return pool.template alloc<RemFn>();
        case FuncTag::BIT_AND: return pool.template alloc<BitAndFn>();
        case FuncTag::BIT_OR: return pool.template alloc<BitOrFn>();
        case FuncTag::BIT_XOR: return pool.template alloc<BitXorFn>();
        case FuncTag::SHL: return pool.template alloc<ShlFn>();
        case FuncTag::SHR: return pool.template alloc<ShrFn>();
        case FuncTag::ROT_RIGHT: return pool.template alloc<RotRightFn>();
        case FuncTag::ROT_LEFT: return pool.template alloc<RotLeftFn>();
        case FuncTag::EQ: return pool.template alloc<EqFn>();
        case FuncTag::NE: return pool.template alloc<NeFn>();
        case FuncTag::LT: return pool.template alloc<LtFn>();
        case FuncTag::LE: return pool.template alloc<LeFn>();
        case FuncTag::GT: return pool.template alloc<GtFn>();
        case FuncTag::GE: return pool.template alloc<GeFn>();
        case FuncTag::MAT_COMP_MUL: return pool.template alloc<MatCompMulFn>();
        case FuncTag::NEG: return pool.template alloc<NegFn>();
        case FuncTag::NOT: return pool.template alloc<NotFn>();
        case FuncTag::BIT_NOT: return pool.template alloc<BitNotFn>();
        case FuncTag::ALL: return pool.template alloc<AllFn>();
        case FuncTag::ANY: return pool.template alloc<AnyFn>();
        case FuncTag::SELECT: return pool.template alloc<SelectFn>();
        case FuncTag::CLAMP: return pool.template alloc<ClampFn>();
        case FuncTag::LERP: return pool.template alloc<LerpFn>();
        case FuncTag::STEP: return pool.template alloc<StepFn>();
        case FuncTag::SATURATE: return pool.template alloc<SaturateFn>();
        case FuncTag::SMOOTH_STEP: return pool.template alloc<SmoothStepFn>();
        case FuncTag::ABS: return pool.template alloc<AbsFn>();
        case FuncTag::MIN: return pool.template alloc<MinFn>();
        case FuncTag::MAX: return pool.template alloc<MaxFn>();
        case FuncTag::REDUCE_SUM: return pool.template alloc<ReduceSumFn>();
        case FuncTag::REDUCE_PROD: return pool.template alloc<ReduceProdFn>();
        case FuncTag::REDUCE_MIN: return pool.template alloc<ReduceMinFn>();
        case FuncTag::REDUCE_MAX: return pool.template alloc<ReduceMaxFn>();
        case FuncTag::CLZ: return pool.template alloc<ClzFn>();
        case FuncTag::CTZ: return pool.template alloc<CtzFn>();
        case FuncTag::POP_COUNT: return pool.template alloc<PopCountFn>();
        case FuncTag::REVERSE: return pool.template alloc<ReverseFn>();
        case FuncTag::IS_INF: return pool.template alloc<IsInfFn>();
        case FuncTag::IS_NAN: return pool.template alloc<IsNanFn>();
        case FuncTag::ACOS: return pool.template alloc<AcosFn>();
        case FuncTag::ACOSH: return pool.template alloc<AcoshFn>();
        case FuncTag::ASIN: return pool.template alloc<AsinFn>();
        case FuncTag::ASINH: return pool.template alloc<AsinhFn>();
        case FuncTag::ATAN: return pool.template alloc<AtanFn>();
        case FuncTag::ATAN2: return pool.template alloc<Atan2Fn>();
        case FuncTag::ATANH: return pool.template alloc<AtanhFn>();
        case FuncTag::COS: return pool.template alloc<CosFn>();
        case FuncTag::COSH: return pool.template alloc<CoshFn>();
        case FuncTag::SIN: return pool.template alloc<SinFn>();
        case FuncTag::SINH: return pool.template alloc<SinhFn>();
        case FuncTag::TAN: return pool.template alloc<TanFn>();
        case FuncTag::TANH: return pool.template alloc<TanhFn>();
        case FuncTag::EXP: return pool.template alloc<ExpFn>();
        case FuncTag::EXP2: return pool.template alloc<Exp2Fn>();
        case FuncTag::EXP10: return pool.template alloc<Exp10Fn>();
        case FuncTag::LOG: return pool.template alloc<LogFn>();
        case FuncTag::LOG2: return pool.template alloc<Log2Fn>();
        case FuncTag::LOG10: return pool.template alloc<Log10Fn>();
        case FuncTag::POWI: return pool.template alloc<PowiFn>();
        case FuncTag::POWF: return pool.template alloc<PowfFn>();
        case FuncTag::SQRT: return pool.template alloc<SqrtFn>();
        case FuncTag::RSQRT: return pool.template alloc<RsqrtFn>();
        case FuncTag::CEIL: return pool.template alloc<CeilFn>();
        case FuncTag::FLOOR: return pool.template alloc<FloorFn>();
        case FuncTag::FRACT: return pool.template alloc<FractFn>();
        case FuncTag::TRUNC: return pool.template alloc<TruncFn>();
        case FuncTag::ROUND: return pool.template alloc<RoundFn>();
        case FuncTag::FMA: return pool.template alloc<FmaFn>();
        case FuncTag::COPYSIGN: return pool.template alloc<CopysignFn>();
        case FuncTag::CROSS: return pool.template alloc<CrossFn>();
        case FuncTag::DOT: return pool.template alloc<DotFn>();
        case FuncTag::OUTER_PRODUCT: return pool.template alloc<OuterProductFn>();
        case FuncTag::LENGTH: return pool.template alloc<LengthFn>();
        case FuncTag::LENGTH_SQUARED: return pool.template alloc<LengthSquaredFn>();
        case FuncTag::NORMALIZE: return pool.template alloc<NormalizeFn>();
        case FuncTag::FACEFORWARD: return pool.template alloc<FaceforwardFn>();
        case FuncTag::DISTANCE: return pool.template alloc<DistanceFn>();
        case FuncTag::REFLECT: return pool.template alloc<ReflectFn>();
        case FuncTag::DETERMINANT: return pool.template alloc<DeterminantFn>();
        case FuncTag::TRANSPOSE: return pool.template alloc<TransposeFn>();
        case FuncTag::INVERSE: return pool.template alloc<InverseFn>();
        case FuncTag::WARP_IS_FIRST_ACTIVE_LANE: return pool.template alloc<WarpIsFirstActiveLaneFn>();
        case FuncTag::WARP_FIRST_ACTIVE_LANE: return pool.template alloc<WarpFirstActiveLaneFn>();
        case FuncTag::WARP_ACTIVE_ALL_EQUAL: return pool.template alloc<WarpActiveAllEqualFn>();
        case FuncTag::WARP_ACTIVE_BIT_AND: return pool.template alloc<WarpActiveBitAndFn>();
        case FuncTag::WARP_ACTIVE_BIT_OR: return pool.template alloc<WarpActiveBitOrFn>();
        case FuncTag::WARP_ACTIVE_BIT_XOR: return pool.template alloc<WarpActiveBitXorFn>();
        case FuncTag::WARP_ACTIVE_COUNT_BITS: return pool.template alloc<WarpActiveCountBitsFn>();
        case FuncTag::WARP_ACTIVE_MAX: return pool.template alloc<WarpActiveMaxFn>();
        case FuncTag::WARP_ACTIVE_MIN: return pool.template alloc<WarpActiveMinFn>();
        case FuncTag::WARP_ACTIVE_PRODUCT: return pool.template alloc<WarpActiveProductFn>();
        case FuncTag::WARP_ACTIVE_SUM: return pool.template alloc<WarpActiveSumFn>();
        case FuncTag::WARP_ACTIVE_ALL: return pool.template alloc<WarpActiveAllFn>();
        case FuncTag::WARP_ACTIVE_ANY: return pool.template alloc<WarpActiveAnyFn>();
        case FuncTag::WARP_ACTIVE_BIT_MASK: return pool.template alloc<WarpActiveBitMaskFn>();
        case FuncTag::WARP_PREFIX_COUNT_BITS: return pool.template alloc<WarpPrefixCountBitsFn>();
        case FuncTag::WARP_PREFIX_SUM: return pool.template alloc<WarpPrefixSumFn>();
        case FuncTag::WARP_PREFIX_PRODUCT: return pool.template alloc<WarpPrefixProductFn>();
        case FuncTag::WARP_READ_LANE_AT: return pool.template alloc<WarpReadLaneAtFn>();
        case FuncTag::WARP_READ_FIRST_LANE: return pool.template alloc<WarpReadFirstLaneFn>();
        case FuncTag::SYNCHRONIZE_BLOCK: return pool.template alloc<SynchronizeBlockFn>();
        case FuncTag::ATOMIC_EXCHANGE: return pool.template alloc<AtomicExchangeFn>();
        case FuncTag::ATOMIC_COMPARE_EXCHANGE: return pool.template alloc<AtomicCompareExchangeFn>();
        case FuncTag::ATOMIC_FETCH_ADD: return pool.template alloc<AtomicFetchAddFn>();
        case FuncTag::ATOMIC_FETCH_SUB: return pool.template alloc<AtomicFetchSubFn>();
        case FuncTag::ATOMIC_FETCH_AND: return pool.template alloc<AtomicFetchAndFn>();
        case FuncTag::ATOMIC_FETCH_OR: return pool.template alloc<AtomicFetchOrFn>();
        case FuncTag::ATOMIC_FETCH_XOR: return pool.template alloc<AtomicFetchXorFn>();
        case FuncTag::ATOMIC_FETCH_MIN: return pool.template alloc<AtomicFetchMinFn>();
        case FuncTag::ATOMIC_FETCH_MAX: return pool.template alloc<AtomicFetchMaxFn>();
        case FuncTag::BUFFER_WRITE: return pool.template alloc<BufferWriteFn>();
        case FuncTag::BUFFER_READ: return pool.template alloc<BufferReadFn>();
        case FuncTag::BUFFER_SIZE: return pool.template alloc<BufferSizeFn>();
        case FuncTag::BYTE_BUFFER_WRITE: return pool.template alloc<ByteBufferWriteFn>();
        case FuncTag::BYTE_BUFFER_READ: return pool.template alloc<ByteBufferReadFn>();
        case FuncTag::BYTE_BUFFER_SIZE: return pool.template alloc<ByteBufferSizeFn>();
        case FuncTag::TEXTURE2D_READ: return pool.template alloc<Texture2dReadFn>();
        case FuncTag::TEXTURE2D_WRITE: return pool.template alloc<Texture2dWriteFn>();
        case FuncTag::TEXTURE2D_SIZE: return pool.template alloc<Texture2dSizeFn>();
        case FuncTag::TEXTURE3D_READ: return pool.template alloc<Texture3dReadFn>();
        case FuncTag::TEXTURE3D_WRITE: return pool.template alloc<Texture3dWriteFn>();
        case FuncTag::TEXTURE3D_SIZE: return pool.template alloc<Texture3dSizeFn>();
        case FuncTag::BINDLESS_TEXTURE2D_SAMPLE: return pool.template alloc<BindlessTexture2dSampleFn>();
        case FuncTag::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: return pool.template alloc<BindlessTexture2dSampleLevelFn>();
        case FuncTag::BINDLESS_TEXTURE2D_SAMPLE_GRAD: return pool.template alloc<BindlessTexture2dSampleGradFn>();
        case FuncTag::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL: return pool.template alloc<BindlessTexture2dSampleGradLevelFn>();
        case FuncTag::BINDLESS_TEXTURE2D_READ: return pool.template alloc<BindlessTexture2dReadFn>();
        case FuncTag::BINDLESS_TEXTURE2D_SIZE: return pool.template alloc<BindlessTexture2dSizeFn>();
        case FuncTag::BINDLESS_TEXTURE2D_SIZE_LEVEL: return pool.template alloc<BindlessTexture2dSizeLevelFn>();
        case FuncTag::BINDLESS_TEXTURE3D_SAMPLE: return pool.template alloc<BindlessTexture3dSampleFn>();
        case FuncTag::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: return pool.template alloc<BindlessTexture3dSampleLevelFn>();
        case FuncTag::BINDLESS_TEXTURE3D_SAMPLE_GRAD: return pool.template alloc<BindlessTexture3dSampleGradFn>();
        case FuncTag::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL: return pool.template alloc<BindlessTexture3dSampleGradLevelFn>();
        case FuncTag::BINDLESS_TEXTURE3D_READ: return pool.template alloc<BindlessTexture3dReadFn>();
        case FuncTag::BINDLESS_TEXTURE3D_SIZE: return pool.template alloc<BindlessTexture3dSizeFn>();
        case FuncTag::BINDLESS_TEXTURE3D_SIZE_LEVEL: return pool.template alloc<BindlessTexture3dSizeLevelFn>();
        case FuncTag::BINDLESS_BUFFER_WRITE: return pool.template alloc<BindlessBufferWriteFn>();
        case FuncTag::BINDLESS_BUFFER_READ: return pool.template alloc<BindlessBufferReadFn>();
        case FuncTag::BINDLESS_BUFFER_SIZE: return pool.template alloc<BindlessBufferSizeFn>();
        case FuncTag::BINDLESS_BYTE_BUFFER_WRITE: return pool.template alloc<BindlessByteBufferWriteFn>();
        case FuncTag::BINDLESS_BYTE_BUFFER_READ: return pool.template alloc<BindlessByteBufferReadFn>();
        case FuncTag::BINDLESS_BYTE_BUFFER_SIZE: return pool.template alloc<BindlessByteBufferSizeFn>();
        case FuncTag::VEC: return pool.template alloc<VecFn>();
        case FuncTag::VEC2: return pool.template alloc<Vec2Fn>();
        case FuncTag::VEC3: return pool.template alloc<Vec3Fn>();
        case FuncTag::VEC4: return pool.template alloc<Vec4Fn>();
        case FuncTag::PERMUTE: return pool.template alloc<PermuteFn>();
        case FuncTag::GET_ELEMENT_PTR: return pool.template alloc<GetElementPtrFn>();
        case FuncTag::EXTRACT_ELEMENT: return pool.template alloc<ExtractElementFn>();
        case FuncTag::INSERT_ELEMENT: return pool.template alloc<InsertElementFn>();
        case FuncTag::ARRAY: return pool.template alloc<ArrayFn>();
        case FuncTag::STRUCT: return pool.template alloc<StructFn>();
        case FuncTag::MAT_FULL: return pool.template alloc<MatFullFn>();
        case FuncTag::MAT2: return pool.template alloc<Mat2Fn>();
        case FuncTag::MAT3: return pool.template alloc<Mat3Fn>();
        case FuncTag::MAT4: return pool.template alloc<Mat4Fn>();
        case FuncTag::BINDLESS_ATOMIC_EXCHANGE: return pool.template alloc<BindlessAtomicExchangeFn>();
        case FuncTag::BINDLESS_ATOMIC_COMPARE_EXCHANGE: return pool.template alloc<BindlessAtomicCompareExchangeFn>();
        case FuncTag::BINDLESS_ATOMIC_FETCH_ADD: return pool.template alloc<BindlessAtomicFetchAddFn>();
        case FuncTag::BINDLESS_ATOMIC_FETCH_SUB: return pool.template alloc<BindlessAtomicFetchSubFn>();
        case FuncTag::BINDLESS_ATOMIC_FETCH_AND: return pool.template alloc<BindlessAtomicFetchAndFn>();
        case FuncTag::BINDLESS_ATOMIC_FETCH_OR: return pool.template alloc<BindlessAtomicFetchOrFn>();
        case FuncTag::BINDLESS_ATOMIC_FETCH_XOR: return pool.template alloc<BindlessAtomicFetchXorFn>();
        case FuncTag::BINDLESS_ATOMIC_FETCH_MIN: return pool.template alloc<BindlessAtomicFetchMinFn>();
        case FuncTag::BINDLESS_ATOMIC_FETCH_MAX: return pool.template alloc<BindlessAtomicFetchMaxFn>();
        case FuncTag::CALLABLE: return pool.template alloc<CallableFn>();
        case FuncTag::CPU_EXT: return pool.template alloc<CpuExtFn>();
        case FuncTag::SHADER_EXECUTION_REORDER: return pool.template alloc<ShaderExecutionReorderFn>();
    }
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
static void BufferBinding_set_handle(BufferBinding *self, uint64_t value) {
    self->handle = value;
}
static void BufferBinding_set_offset(BufferBinding *self, uint64_t value) {
    self->offset = value;
}
static void BufferBinding_set_size(BufferBinding *self, uint64_t value) {
    self->size = value;
}
static uint64_t TextureBinding_handle(TextureBinding *self) {
    return self->handle;
}
static uint64_t TextureBinding_level(TextureBinding *self) {
    return self->level;
}
static void TextureBinding_set_handle(TextureBinding *self, uint64_t value) {
    self->handle = value;
}
static void TextureBinding_set_level(TextureBinding *self, uint64_t value) {
    self->level = value;
}
static uint64_t BindlessArrayBinding_handle(BindlessArrayBinding *self) {
    return self->handle;
}
static void BindlessArrayBinding_set_handle(BindlessArrayBinding *self, uint64_t value) {
    self->handle = value;
}
static uint64_t AccelBinding_handle(AccelBinding *self) {
    return self->handle;
}
static void AccelBinding_set_handle(AccelBinding *self, uint64_t value) {
    self->handle = value;
}
extern "C" LC_IR_API IrV2BindingTable lc_ir_v2_binding_table() {
    return {
        Func_as_ZeroFn,
        Func_as_OneFn,
        Func_as_AssumeFn,
        Func_as_UnreachableFn,
        Func_as_ThreadIdFn,
        Func_as_BlockIdFn,
        Func_as_WarpSizeFn,
        Func_as_WarpLaneIdFn,
        Func_as_DispatchIdFn,
        Func_as_DispatchSizeFn,
        Func_as_PropagateGradientFn,
        Func_as_OutputGradientFn,
        Func_as_RequiresGradientFn,
        Func_as_BackwardFn,
        Func_as_GradientFn,
        Func_as_AccGradFn,
        Func_as_DetachFn,
        Func_as_RayTracingInstanceTransformFn,
        Func_as_RayTracingInstanceVisibilityMaskFn,
        Func_as_RayTracingInstanceUserIdFn,
        Func_as_RayTracingSetInstanceTransformFn,
        Func_as_RayTracingSetInstanceOpacityFn,
        Func_as_RayTracingSetInstanceVisibilityFn,
        Func_as_RayTracingSetInstanceUserIdFn,
        Func_as_RayTracingTraceClosestFn,
        Func_as_RayTracingTraceAnyFn,
        Func_as_RayTracingQueryAllFn,
        Func_as_RayTracingQueryAnyFn,
        Func_as_RayQueryWorldSpaceRayFn,
        Func_as_RayQueryProceduralCandidateHitFn,
        Func_as_RayQueryTriangleCandidateHitFn,
        Func_as_RayQueryCommittedHitFn,
        Func_as_RayQueryCommitTriangleFn,
        Func_as_RayQueryCommitdProceduralFn,
        Func_as_RayQueryTerminateFn,
        Func_as_LoadFn,
        Func_as_CastFn,
        Func_as_BitCastFn,
        Func_as_AddFn,
        Func_as_SubFn,
        Func_as_MulFn,
        Func_as_DivFn,
        Func_as_RemFn,
        Func_as_BitAndFn,
        Func_as_BitOrFn,
        Func_as_BitXorFn,
        Func_as_ShlFn,
        Func_as_ShrFn,
        Func_as_RotRightFn,
        Func_as_RotLeftFn,
        Func_as_EqFn,
        Func_as_NeFn,
        Func_as_LtFn,
        Func_as_LeFn,
        Func_as_GtFn,
        Func_as_GeFn,
        Func_as_MatCompMulFn,
        Func_as_NegFn,
        Func_as_NotFn,
        Func_as_BitNotFn,
        Func_as_AllFn,
        Func_as_AnyFn,
        Func_as_SelectFn,
        Func_as_ClampFn,
        Func_as_LerpFn,
        Func_as_StepFn,
        Func_as_SaturateFn,
        Func_as_SmoothStepFn,
        Func_as_AbsFn,
        Func_as_MinFn,
        Func_as_MaxFn,
        Func_as_ReduceSumFn,
        Func_as_ReduceProdFn,
        Func_as_ReduceMinFn,
        Func_as_ReduceMaxFn,
        Func_as_ClzFn,
        Func_as_CtzFn,
        Func_as_PopCountFn,
        Func_as_ReverseFn,
        Func_as_IsInfFn,
        Func_as_IsNanFn,
        Func_as_AcosFn,
        Func_as_AcoshFn,
        Func_as_AsinFn,
        Func_as_AsinhFn,
        Func_as_AtanFn,
        Func_as_Atan2Fn,
        Func_as_AtanhFn,
        Func_as_CosFn,
        Func_as_CoshFn,
        Func_as_SinFn,
        Func_as_SinhFn,
        Func_as_TanFn,
        Func_as_TanhFn,
        Func_as_ExpFn,
        Func_as_Exp2Fn,
        Func_as_Exp10Fn,
        Func_as_LogFn,
        Func_as_Log2Fn,
        Func_as_Log10Fn,
        Func_as_PowiFn,
        Func_as_PowfFn,
        Func_as_SqrtFn,
        Func_as_RsqrtFn,
        Func_as_CeilFn,
        Func_as_FloorFn,
        Func_as_FractFn,
        Func_as_TruncFn,
        Func_as_RoundFn,
        Func_as_FmaFn,
        Func_as_CopysignFn,
        Func_as_CrossFn,
        Func_as_DotFn,
        Func_as_OuterProductFn,
        Func_as_LengthFn,
        Func_as_LengthSquaredFn,
        Func_as_NormalizeFn,
        Func_as_FaceforwardFn,
        Func_as_DistanceFn,
        Func_as_ReflectFn,
        Func_as_DeterminantFn,
        Func_as_TransposeFn,
        Func_as_InverseFn,
        Func_as_WarpIsFirstActiveLaneFn,
        Func_as_WarpFirstActiveLaneFn,
        Func_as_WarpActiveAllEqualFn,
        Func_as_WarpActiveBitAndFn,
        Func_as_WarpActiveBitOrFn,
        Func_as_WarpActiveBitXorFn,
        Func_as_WarpActiveCountBitsFn,
        Func_as_WarpActiveMaxFn,
        Func_as_WarpActiveMinFn,
        Func_as_WarpActiveProductFn,
        Func_as_WarpActiveSumFn,
        Func_as_WarpActiveAllFn,
        Func_as_WarpActiveAnyFn,
        Func_as_WarpActiveBitMaskFn,
        Func_as_WarpPrefixCountBitsFn,
        Func_as_WarpPrefixSumFn,
        Func_as_WarpPrefixProductFn,
        Func_as_WarpReadLaneAtFn,
        Func_as_WarpReadFirstLaneFn,
        Func_as_SynchronizeBlockFn,
        Func_as_AtomicExchangeFn,
        Func_as_AtomicCompareExchangeFn,
        Func_as_AtomicFetchAddFn,
        Func_as_AtomicFetchSubFn,
        Func_as_AtomicFetchAndFn,
        Func_as_AtomicFetchOrFn,
        Func_as_AtomicFetchXorFn,
        Func_as_AtomicFetchMinFn,
        Func_as_AtomicFetchMaxFn,
        Func_as_BufferWriteFn,
        Func_as_BufferReadFn,
        Func_as_BufferSizeFn,
        Func_as_ByteBufferWriteFn,
        Func_as_ByteBufferReadFn,
        Func_as_ByteBufferSizeFn,
        Func_as_Texture2dReadFn,
        Func_as_Texture2dWriteFn,
        Func_as_Texture2dSizeFn,
        Func_as_Texture3dReadFn,
        Func_as_Texture3dWriteFn,
        Func_as_Texture3dSizeFn,
        Func_as_BindlessTexture2dSampleFn,
        Func_as_BindlessTexture2dSampleLevelFn,
        Func_as_BindlessTexture2dSampleGradFn,
        Func_as_BindlessTexture2dSampleGradLevelFn,
        Func_as_BindlessTexture2dReadFn,
        Func_as_BindlessTexture2dSizeFn,
        Func_as_BindlessTexture2dSizeLevelFn,
        Func_as_BindlessTexture3dSampleFn,
        Func_as_BindlessTexture3dSampleLevelFn,
        Func_as_BindlessTexture3dSampleGradFn,
        Func_as_BindlessTexture3dSampleGradLevelFn,
        Func_as_BindlessTexture3dReadFn,
        Func_as_BindlessTexture3dSizeFn,
        Func_as_BindlessTexture3dSizeLevelFn,
        Func_as_BindlessBufferWriteFn,
        Func_as_BindlessBufferReadFn,
        Func_as_BindlessBufferSizeFn,
        Func_as_BindlessByteBufferWriteFn,
        Func_as_BindlessByteBufferReadFn,
        Func_as_BindlessByteBufferSizeFn,
        Func_as_VecFn,
        Func_as_Vec2Fn,
        Func_as_Vec3Fn,
        Func_as_Vec4Fn,
        Func_as_PermuteFn,
        Func_as_GetElementPtrFn,
        Func_as_ExtractElementFn,
        Func_as_InsertElementFn,
        Func_as_ArrayFn,
        Func_as_StructFn,
        Func_as_MatFullFn,
        Func_as_Mat2Fn,
        Func_as_Mat3Fn,
        Func_as_Mat4Fn,
        Func_as_BindlessAtomicExchangeFn,
        Func_as_BindlessAtomicCompareExchangeFn,
        Func_as_BindlessAtomicFetchAddFn,
        Func_as_BindlessAtomicFetchSubFn,
        Func_as_BindlessAtomicFetchAndFn,
        Func_as_BindlessAtomicFetchOrFn,
        Func_as_BindlessAtomicFetchXorFn,
        Func_as_BindlessAtomicFetchMinFn,
        Func_as_BindlessAtomicFetchMaxFn,
        Func_as_CallableFn,
        Func_as_CpuExtFn,
        Func_as_ShaderExecutionReorderFn,
        Func_tag,
        AssumeFn_msg,
        AssumeFn_set_msg,
        UnreachableFn_msg,
        UnreachableFn_set_msg,
        BindlessAtomicExchangeFn_ty,
        BindlessAtomicExchangeFn_set_ty,
        BindlessAtomicCompareExchangeFn_ty,
        BindlessAtomicCompareExchangeFn_set_ty,
        BindlessAtomicFetchAddFn_ty,
        BindlessAtomicFetchAddFn_set_ty,
        BindlessAtomicFetchSubFn_ty,
        BindlessAtomicFetchSubFn_set_ty,
        BindlessAtomicFetchAndFn_ty,
        BindlessAtomicFetchAndFn_set_ty,
        BindlessAtomicFetchOrFn_ty,
        BindlessAtomicFetchOrFn_set_ty,
        BindlessAtomicFetchXorFn_ty,
        BindlessAtomicFetchXorFn_set_ty,
        BindlessAtomicFetchMinFn_ty,
        BindlessAtomicFetchMinFn_set_ty,
        BindlessAtomicFetchMaxFn_ty,
        BindlessAtomicFetchMaxFn_set_ty,
        CallableFn_module,
        CallableFn_set_module,
        CpuExtFn_f,
        CpuExtFn_set_f,
        Instruction_as_BufferInst,
        Instruction_as_Texture2dInst,
        Instruction_as_Texture3dInst,
        Instruction_as_BindlessArrayInst,
        Instruction_as_AccelInst,
        Instruction_as_SharedInst,
        Instruction_as_UniformInst,
        Instruction_as_ArgumentInst,
        Instruction_as_ConstantInst,
        Instruction_as_CallInst,
        Instruction_as_PhiInst,
        Instruction_as_BasicBlockSentinelInst,
        Instruction_as_IfInst,
        Instruction_as_GenericLoopInst,
        Instruction_as_SwitchInst,
        Instruction_as_LocalInst,
        Instruction_as_BreakInst,
        Instruction_as_ContinueInst,
        Instruction_as_ReturnInst,
        Instruction_as_PrintInst,
        Instruction_as_UpdateInst,
        Instruction_as_RayQueryInst,
        Instruction_as_RevAutodiffInst,
        Instruction_as_FwdAutodiffInst,
        Instruction_tag,
        ArgumentInst_by_value,
        ArgumentInst_set_by_value,
        ConstantInst_ty,
        ConstantInst_value,
        ConstantInst_set_ty,
        ConstantInst_set_value,
        CallInst_func,
        CallInst_args,
        CallInst_set_func,
        CallInst_set_args,
        PhiInst_incomings,
        PhiInst_set_incomings,
        IfInst_cond,
        IfInst_true_branch,
        IfInst_false_branch,
        IfInst_set_cond,
        IfInst_set_true_branch,
        IfInst_set_false_branch,
        GenericLoopInst_prepare,
        GenericLoopInst_cond,
        GenericLoopInst_body,
        GenericLoopInst_update,
        GenericLoopInst_set_prepare,
        GenericLoopInst_set_cond,
        GenericLoopInst_set_body,
        GenericLoopInst_set_update,
        SwitchInst_value,
        SwitchInst_cases,
        SwitchInst_default_,
        SwitchInst_set_value,
        SwitchInst_set_cases,
        SwitchInst_set_default_,
        LocalInst_init,
        LocalInst_set_init,
        ReturnInst_value,
        ReturnInst_set_value,
        PrintInst_fmt,
        PrintInst_args,
        PrintInst_set_fmt,
        PrintInst_set_args,
        UpdateInst_var,
        UpdateInst_value,
        UpdateInst_set_var,
        UpdateInst_set_value,
        RayQueryInst_query,
        RayQueryInst_on_triangle_hit,
        RayQueryInst_on_procedural_hit,
        RayQueryInst_set_query,
        RayQueryInst_set_on_triangle_hit,
        RayQueryInst_set_on_procedural_hit,
        RevAutodiffInst_body,
        RevAutodiffInst_set_body,
        FwdAutodiffInst_body,
        FwdAutodiffInst_set_body,
        Binding_as_BufferBinding,
        Binding_as_TextureBinding,
        Binding_as_BindlessArrayBinding,
        Binding_as_AccelBinding,
        Binding_tag,
        BufferBinding_handle,
        BufferBinding_offset,
        BufferBinding_size,
        BufferBinding_set_handle,
        BufferBinding_set_offset,
        BufferBinding_set_size,
        TextureBinding_handle,
        TextureBinding_level,
        TextureBinding_set_handle,
        TextureBinding_set_level,
        BindlessArrayBinding_handle,
        BindlessArrayBinding_set_handle,
        AccelBinding_handle,
        AccelBinding_set_handle,
    };
}
}// namespace luisa::compute::ir_v2
