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
struct IrV2BindingTable {
    Zero *(*Func_as_Zero)(Func *self);
    One *(*Func_as_One)(Func *self);
    Assume *(*Func_as_Assume)(Func *self);
    Unreachable *(*Func_as_Unreachable)(Func *self);
    ThreadId *(*Func_as_ThreadId)(Func *self);
    BlockId *(*Func_as_BlockId)(Func *self);
    WarpSize *(*Func_as_WarpSize)(Func *self);
    WarpLaneId *(*Func_as_WarpLaneId)(Func *self);
    DispatchId *(*Func_as_DispatchId)(Func *self);
    DispatchSize *(*Func_as_DispatchSize)(Func *self);
    PropagateGradient *(*Func_as_PropagateGradient)(Func *self);
    OutputGradient *(*Func_as_OutputGradient)(Func *self);
    RequiresGradient *(*Func_as_RequiresGradient)(Func *self);
    Backward *(*Func_as_Backward)(Func *self);
    Gradient *(*Func_as_Gradient)(Func *self);
    AccGrad *(*Func_as_AccGrad)(Func *self);
    Detach *(*Func_as_Detach)(Func *self);
    RayTracingInstanceTransform *(*Func_as_RayTracingInstanceTransform)(Func *self);
    RayTracingInstanceVisibilityMask *(*Func_as_RayTracingInstanceVisibilityMask)(Func *self);
    RayTracingInstanceUserId *(*Func_as_RayTracingInstanceUserId)(Func *self);
    RayTracingSetInstanceTransform *(*Func_as_RayTracingSetInstanceTransform)(Func *self);
    RayTracingSetInstanceOpacity *(*Func_as_RayTracingSetInstanceOpacity)(Func *self);
    RayTracingSetInstanceVisibility *(*Func_as_RayTracingSetInstanceVisibility)(Func *self);
    RayTracingSetInstanceUserId *(*Func_as_RayTracingSetInstanceUserId)(Func *self);
    RayTracingTraceClosest *(*Func_as_RayTracingTraceClosest)(Func *self);
    RayTracingTraceAny *(*Func_as_RayTracingTraceAny)(Func *self);
    RayTracingQueryAll *(*Func_as_RayTracingQueryAll)(Func *self);
    RayTracingQueryAny *(*Func_as_RayTracingQueryAny)(Func *self);
    RayQueryWorldSpaceRay *(*Func_as_RayQueryWorldSpaceRay)(Func *self);
    RayQueryProceduralCandidateHit *(*Func_as_RayQueryProceduralCandidateHit)(Func *self);
    RayQueryTriangleCandidateHit *(*Func_as_RayQueryTriangleCandidateHit)(Func *self);
    RayQueryCommittedHit *(*Func_as_RayQueryCommittedHit)(Func *self);
    RayQueryCommitTriangle *(*Func_as_RayQueryCommitTriangle)(Func *self);
    RayQueryCommitdProcedural *(*Func_as_RayQueryCommitdProcedural)(Func *self);
    RayQueryTerminate *(*Func_as_RayQueryTerminate)(Func *self);
    Load *(*Func_as_Load)(Func *self);
    Cast *(*Func_as_Cast)(Func *self);
    BitCast *(*Func_as_BitCast)(Func *self);
    Add *(*Func_as_Add)(Func *self);
    Sub *(*Func_as_Sub)(Func *self);
    Mul *(*Func_as_Mul)(Func *self);
    Div *(*Func_as_Div)(Func *self);
    Rem *(*Func_as_Rem)(Func *self);
    BitAnd *(*Func_as_BitAnd)(Func *self);
    BitOr *(*Func_as_BitOr)(Func *self);
    BitXor *(*Func_as_BitXor)(Func *self);
    Shl *(*Func_as_Shl)(Func *self);
    Shr *(*Func_as_Shr)(Func *self);
    RotRight *(*Func_as_RotRight)(Func *self);
    RotLeft *(*Func_as_RotLeft)(Func *self);
    Eq *(*Func_as_Eq)(Func *self);
    Ne *(*Func_as_Ne)(Func *self);
    Lt *(*Func_as_Lt)(Func *self);
    Le *(*Func_as_Le)(Func *self);
    Gt *(*Func_as_Gt)(Func *self);
    Ge *(*Func_as_Ge)(Func *self);
    MatCompMul *(*Func_as_MatCompMul)(Func *self);
    Neg *(*Func_as_Neg)(Func *self);
    Not *(*Func_as_Not)(Func *self);
    BitNot *(*Func_as_BitNot)(Func *self);
    All *(*Func_as_All)(Func *self);
    Any *(*Func_as_Any)(Func *self);
    Select *(*Func_as_Select)(Func *self);
    Clamp *(*Func_as_Clamp)(Func *self);
    Lerp *(*Func_as_Lerp)(Func *self);
    Step *(*Func_as_Step)(Func *self);
    Saturate *(*Func_as_Saturate)(Func *self);
    SmoothStep *(*Func_as_SmoothStep)(Func *self);
    Abs *(*Func_as_Abs)(Func *self);
    Min *(*Func_as_Min)(Func *self);
    Max *(*Func_as_Max)(Func *self);
    ReduceSum *(*Func_as_ReduceSum)(Func *self);
    ReduceProd *(*Func_as_ReduceProd)(Func *self);
    ReduceMin *(*Func_as_ReduceMin)(Func *self);
    ReduceMax *(*Func_as_ReduceMax)(Func *self);
    Clz *(*Func_as_Clz)(Func *self);
    Ctz *(*Func_as_Ctz)(Func *self);
    PopCount *(*Func_as_PopCount)(Func *self);
    Reverse *(*Func_as_Reverse)(Func *self);
    IsInf *(*Func_as_IsInf)(Func *self);
    IsNan *(*Func_as_IsNan)(Func *self);
    Acos *(*Func_as_Acos)(Func *self);
    Acosh *(*Func_as_Acosh)(Func *self);
    Asin *(*Func_as_Asin)(Func *self);
    Asinh *(*Func_as_Asinh)(Func *self);
    Atan *(*Func_as_Atan)(Func *self);
    Atan2 *(*Func_as_Atan2)(Func *self);
    Atanh *(*Func_as_Atanh)(Func *self);
    Cos *(*Func_as_Cos)(Func *self);
    Cosh *(*Func_as_Cosh)(Func *self);
    Sin *(*Func_as_Sin)(Func *self);
    Sinh *(*Func_as_Sinh)(Func *self);
    Tan *(*Func_as_Tan)(Func *self);
    Tanh *(*Func_as_Tanh)(Func *self);
    Exp *(*Func_as_Exp)(Func *self);
    Exp2 *(*Func_as_Exp2)(Func *self);
    Exp10 *(*Func_as_Exp10)(Func *self);
    Log *(*Func_as_Log)(Func *self);
    Log2 *(*Func_as_Log2)(Func *self);
    Log10 *(*Func_as_Log10)(Func *self);
    Powi *(*Func_as_Powi)(Func *self);
    Powf *(*Func_as_Powf)(Func *self);
    Sqrt *(*Func_as_Sqrt)(Func *self);
    Rsqrt *(*Func_as_Rsqrt)(Func *self);
    Ceil *(*Func_as_Ceil)(Func *self);
    Floor *(*Func_as_Floor)(Func *self);
    Fract *(*Func_as_Fract)(Func *self);
    Trunc *(*Func_as_Trunc)(Func *self);
    Round *(*Func_as_Round)(Func *self);
    Fma *(*Func_as_Fma)(Func *self);
    Copysign *(*Func_as_Copysign)(Func *self);
    Cross *(*Func_as_Cross)(Func *self);
    Dot *(*Func_as_Dot)(Func *self);
    OuterProduct *(*Func_as_OuterProduct)(Func *self);
    Length *(*Func_as_Length)(Func *self);
    LengthSquared *(*Func_as_LengthSquared)(Func *self);
    Normalize *(*Func_as_Normalize)(Func *self);
    Faceforward *(*Func_as_Faceforward)(Func *self);
    Distance *(*Func_as_Distance)(Func *self);
    Reflect *(*Func_as_Reflect)(Func *self);
    Determinant *(*Func_as_Determinant)(Func *self);
    Transpose *(*Func_as_Transpose)(Func *self);
    Inverse *(*Func_as_Inverse)(Func *self);
    WarpIsFirstActiveLane *(*Func_as_WarpIsFirstActiveLane)(Func *self);
    WarpFirstActiveLane *(*Func_as_WarpFirstActiveLane)(Func *self);
    WarpActiveAllEqual *(*Func_as_WarpActiveAllEqual)(Func *self);
    WarpActiveBitAnd *(*Func_as_WarpActiveBitAnd)(Func *self);
    WarpActiveBitOr *(*Func_as_WarpActiveBitOr)(Func *self);
    WarpActiveBitXor *(*Func_as_WarpActiveBitXor)(Func *self);
    WarpActiveCountBits *(*Func_as_WarpActiveCountBits)(Func *self);
    WarpActiveMax *(*Func_as_WarpActiveMax)(Func *self);
    WarpActiveMin *(*Func_as_WarpActiveMin)(Func *self);
    WarpActiveProduct *(*Func_as_WarpActiveProduct)(Func *self);
    WarpActiveSum *(*Func_as_WarpActiveSum)(Func *self);
    WarpActiveAll *(*Func_as_WarpActiveAll)(Func *self);
    WarpActiveAny *(*Func_as_WarpActiveAny)(Func *self);
    WarpActiveBitMask *(*Func_as_WarpActiveBitMask)(Func *self);
    WarpPrefixCountBits *(*Func_as_WarpPrefixCountBits)(Func *self);
    WarpPrefixSum *(*Func_as_WarpPrefixSum)(Func *self);
    WarpPrefixProduct *(*Func_as_WarpPrefixProduct)(Func *self);
    WarpReadLaneAt *(*Func_as_WarpReadLaneAt)(Func *self);
    WarpReadFirstLane *(*Func_as_WarpReadFirstLane)(Func *self);
    SynchronizeBlock *(*Func_as_SynchronizeBlock)(Func *self);
    AtomicExchange *(*Func_as_AtomicExchange)(Func *self);
    AtomicCompareExchange *(*Func_as_AtomicCompareExchange)(Func *self);
    AtomicFetchAdd *(*Func_as_AtomicFetchAdd)(Func *self);
    AtomicFetchSub *(*Func_as_AtomicFetchSub)(Func *self);
    AtomicFetchAnd *(*Func_as_AtomicFetchAnd)(Func *self);
    AtomicFetchOr *(*Func_as_AtomicFetchOr)(Func *self);
    AtomicFetchXor *(*Func_as_AtomicFetchXor)(Func *self);
    AtomicFetchMin *(*Func_as_AtomicFetchMin)(Func *self);
    AtomicFetchMax *(*Func_as_AtomicFetchMax)(Func *self);
    BufferWrite *(*Func_as_BufferWrite)(Func *self);
    BufferRead *(*Func_as_BufferRead)(Func *self);
    BufferSize *(*Func_as_BufferSize)(Func *self);
    ByteBufferWrite *(*Func_as_ByteBufferWrite)(Func *self);
    ByteBufferRead *(*Func_as_ByteBufferRead)(Func *self);
    ByteBufferSize *(*Func_as_ByteBufferSize)(Func *self);
    Texture2dRead *(*Func_as_Texture2dRead)(Func *self);
    Texture2dWrite *(*Func_as_Texture2dWrite)(Func *self);
    Texture2dSize *(*Func_as_Texture2dSize)(Func *self);
    Texture3dRead *(*Func_as_Texture3dRead)(Func *self);
    Texture3dWrite *(*Func_as_Texture3dWrite)(Func *self);
    Texture3dSize *(*Func_as_Texture3dSize)(Func *self);
    BindlessTexture2dSample *(*Func_as_BindlessTexture2dSample)(Func *self);
    BindlessTexture2dSampleLevel *(*Func_as_BindlessTexture2dSampleLevel)(Func *self);
    BindlessTexture2dSampleGrad *(*Func_as_BindlessTexture2dSampleGrad)(Func *self);
    BindlessTexture2dSampleGradLevel *(*Func_as_BindlessTexture2dSampleGradLevel)(Func *self);
    BindlessTexture2dRead *(*Func_as_BindlessTexture2dRead)(Func *self);
    BindlessTexture2dSize *(*Func_as_BindlessTexture2dSize)(Func *self);
    BindlessTexture2dSizeLevel *(*Func_as_BindlessTexture2dSizeLevel)(Func *self);
    BindlessTexture3dSample *(*Func_as_BindlessTexture3dSample)(Func *self);
    BindlessTexture3dSampleLevel *(*Func_as_BindlessTexture3dSampleLevel)(Func *self);
    BindlessTexture3dSampleGrad *(*Func_as_BindlessTexture3dSampleGrad)(Func *self);
    BindlessTexture3dSampleGradLevel *(*Func_as_BindlessTexture3dSampleGradLevel)(Func *self);
    BindlessTexture3dRead *(*Func_as_BindlessTexture3dRead)(Func *self);
    BindlessTexture3dSize *(*Func_as_BindlessTexture3dSize)(Func *self);
    BindlessTexture3dSizeLevel *(*Func_as_BindlessTexture3dSizeLevel)(Func *self);
    BindlessBufferWrite *(*Func_as_BindlessBufferWrite)(Func *self);
    BindlessBufferRead *(*Func_as_BindlessBufferRead)(Func *self);
    BindlessBufferSize *(*Func_as_BindlessBufferSize)(Func *self);
    BindlessByteBufferWrite *(*Func_as_BindlessByteBufferWrite)(Func *self);
    BindlessByteBufferRead *(*Func_as_BindlessByteBufferRead)(Func *self);
    BindlessByteBufferSize *(*Func_as_BindlessByteBufferSize)(Func *self);
    Vec *(*Func_as_Vec)(Func *self);
    Vec2 *(*Func_as_Vec2)(Func *self);
    Vec3 *(*Func_as_Vec3)(Func *self);
    Vec4 *(*Func_as_Vec4)(Func *self);
    Permute *(*Func_as_Permute)(Func *self);
    GetElementPtr *(*Func_as_GetElementPtr)(Func *self);
    ExtractElement *(*Func_as_ExtractElement)(Func *self);
    InsertElement *(*Func_as_InsertElement)(Func *self);
    Array *(*Func_as_Array)(Func *self);
    Struct *(*Func_as_Struct)(Func *self);
    MatFull *(*Func_as_MatFull)(Func *self);
    Mat2 *(*Func_as_Mat2)(Func *self);
    Mat3 *(*Func_as_Mat3)(Func *self);
    Mat4 *(*Func_as_Mat4)(Func *self);
    BindlessAtomicExchange *(*Func_as_BindlessAtomicExchange)(Func *self);
    BindlessAtomicCompareExchange *(*Func_as_BindlessAtomicCompareExchange)(Func *self);
    BindlessAtomicFetchAdd *(*Func_as_BindlessAtomicFetchAdd)(Func *self);
    BindlessAtomicFetchSub *(*Func_as_BindlessAtomicFetchSub)(Func *self);
    BindlessAtomicFetchAnd *(*Func_as_BindlessAtomicFetchAnd)(Func *self);
    BindlessAtomicFetchOr *(*Func_as_BindlessAtomicFetchOr)(Func *self);
    BindlessAtomicFetchXor *(*Func_as_BindlessAtomicFetchXor)(Func *self);
    BindlessAtomicFetchMin *(*Func_as_BindlessAtomicFetchMin)(Func *self);
    BindlessAtomicFetchMax *(*Func_as_BindlessAtomicFetchMax)(Func *self);
    Callable *(*Func_as_Callable)(Func *self);
    CpuExt *(*Func_as_CpuExt)(Func *self);
    ShaderExecutionReorder *(*Func_as_ShaderExecutionReorder)(Func *self);
    FuncTag (*Func_tag)(Func *self);
    Slice<const char> (*Assume_msg)(Assume *self);
    Slice<const char> (*Unreachable_msg)(Unreachable *self);
    const Type *(*BindlessAtomicExchange_ty)(BindlessAtomicExchange *self);
    const Type *(*BindlessAtomicCompareExchange_ty)(BindlessAtomicCompareExchange *self);
    const Type *(*BindlessAtomicFetchAdd_ty)(BindlessAtomicFetchAdd *self);
    const Type *(*BindlessAtomicFetchSub_ty)(BindlessAtomicFetchSub *self);
    const Type *(*BindlessAtomicFetchAnd_ty)(BindlessAtomicFetchAnd *self);
    const Type *(*BindlessAtomicFetchOr_ty)(BindlessAtomicFetchOr *self);
    const Type *(*BindlessAtomicFetchXor_ty)(BindlessAtomicFetchXor *self);
    const Type *(*BindlessAtomicFetchMin_ty)(BindlessAtomicFetchMin *self);
    const Type *(*BindlessAtomicFetchMax_ty)(BindlessAtomicFetchMax *self);
    CallableModule *(*Callable_module)(Callable *self);
    CpuExternFn (*CpuExt_f)(CpuExt *self);
    Buffer *(*Instruction_as_Buffer)(Instruction *self);
    Texture2d *(*Instruction_as_Texture2d)(Instruction *self);
    Texture3d *(*Instruction_as_Texture3d)(Instruction *self);
    BindlessArray *(*Instruction_as_BindlessArray)(Instruction *self);
    Accel *(*Instruction_as_Accel)(Instruction *self);
    Shared *(*Instruction_as_Shared)(Instruction *self);
    Uniform *(*Instruction_as_Uniform)(Instruction *self);
    Argument *(*Instruction_as_Argument)(Instruction *self);
    Constant *(*Instruction_as_Constant)(Instruction *self);
    Call *(*Instruction_as_Call)(Instruction *self);
    Phi *(*Instruction_as_Phi)(Instruction *self);
    BasicBlockSentinel *(*Instruction_as_BasicBlockSentinel)(Instruction *self);
    If *(*Instruction_as_If)(Instruction *self);
    GenericLoop *(*Instruction_as_GenericLoop)(Instruction *self);
    Switch *(*Instruction_as_Switch)(Instruction *self);
    Local *(*Instruction_as_Local)(Instruction *self);
    Break *(*Instruction_as_Break)(Instruction *self);
    Continue *(*Instruction_as_Continue)(Instruction *self);
    Return *(*Instruction_as_Return)(Instruction *self);
    Print *(*Instruction_as_Print)(Instruction *self);
    Update *(*Instruction_as_Update)(Instruction *self);
    RayQuery *(*Instruction_as_RayQuery)(Instruction *self);
    RevAutodiff *(*Instruction_as_RevAutodiff)(Instruction *self);
    FwdAutodiff *(*Instruction_as_FwdAutodiff)(Instruction *self);
    InstructionTag (*Instruction_tag)(Instruction *self);
    bool (*Argument_by_value)(Argument *self);
    const Type *(*Constant_ty)(Constant *self);
    Slice<uint8_t> (*Constant_value)(Constant *self);
    const Func *(*Call_func)(Call *self);
    Slice<Node *> (*Call_args)(Call *self);
    Slice<PhiIncoming> (*Phi_incomings)(Phi *self);
    Node *(*If_cond)(If *self);
    BasicBlock *(*If_true_branch)(If *self);
    BasicBlock *(*If_false_branch)(If *self);
    BasicBlock *(*GenericLoop_prepare)(GenericLoop *self);
    Node *(*GenericLoop_cond)(GenericLoop *self);
    BasicBlock *(*GenericLoop_body)(GenericLoop *self);
    BasicBlock *(*GenericLoop_update)(GenericLoop *self);
    Node *(*Switch_value)(Switch *self);
    Slice<SwitchCase> (*Switch_cases)(Switch *self);
    BasicBlock *(*Switch_default_)(Switch *self);
    Node *(*Local_init)(Local *self);
    Node *(*Return_value)(Return *self);
    Slice<const char> (*Print_fmt)(Print *self);
    Slice<Node *> (*Print_args)(Print *self);
    Node *(*Update_var)(Update *self);
    Node *(*Update_value)(Update *self);
    Node *(*RayQuery_query)(RayQuery *self);
    BasicBlock *(*RayQuery_on_triangle_hit)(RayQuery *self);
    BasicBlock *(*RayQuery_on_procedural_hit)(RayQuery *self);
    BasicBlock *(*RevAutodiff_body)(RevAutodiff *self);
    BasicBlock *(*FwdAutodiff_body)(FwdAutodiff *self);
    BufferBinding *(*Binding_as_BufferBinding)(Binding *self);
    TextureBinding *(*Binding_as_TextureBinding)(Binding *self);
    BindlessArrayBinding *(*Binding_as_BindlessArrayBinding)(Binding *self);
    AccelBinding *(*Binding_as_AccelBinding)(Binding *self);
    BindingTag (*Binding_tag)(Binding *self);
    uint64_t (*BufferBinding_handle)(BufferBinding *self);
    uint64_t (*BufferBinding_offset)(BufferBinding *self);
    uint64_t (*BufferBinding_size)(BufferBinding *self);
    uint64_t (*TextureBinding_handle)(TextureBinding *self);
    uint64_t (*TextureBinding_level)(TextureBinding *self);
    uint64_t (*BindlessArrayBinding_handle)(BindlessArrayBinding *self);
    uint64_t (*AccelBinding_handle)(AccelBinding *self);
};
extern "C" LC_IR_API IrV2BindingTable lc_ir_v2_binding_table();
}// namespace luisa::compute::ir_v2
