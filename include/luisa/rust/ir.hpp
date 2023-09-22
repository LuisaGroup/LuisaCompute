#pragma once

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>
#include "ir_common.h"


namespace luisa::compute::ir {

enum class ModuleKind {
    Block,
    Function,
    Kernel,
};

enum class Primitive {
    Bool,
    Int8,
    Uint8,
    Int16,
    Uint16,
    Int32,
    Uint32,
    Int64,
    Uint64,
    Float16,
    Float32,
    Float64,
};

struct ModulePools;

struct TransformPipeline;

struct NodeRef {
    size_t _0;

    bool operator==(const NodeRef& other) const {
        return _0 == other._0;
    }
};

struct BasicBlock {
    NodeRef first;
    NodeRef last;
};

struct IrBuilder {
    Pooled<BasicBlock> bb;
    CArc<ModulePools> pools;
    NodeRef insert_point;
};

struct ModuleFlags {
    uint32_t bits;

    explicit operator bool() const {
        return !!bits;
    }
    ModuleFlags operator~() const {
        return ModuleFlags { static_cast<decltype(bits)>(~bits) };
    }
    ModuleFlags operator|(const ModuleFlags& other) const {
        return ModuleFlags { static_cast<decltype(bits)>(this->bits | other.bits) };
    }
    ModuleFlags& operator|=(const ModuleFlags& other) {
        *this = (*this | other);
        return *this;
    }
    ModuleFlags operator&(const ModuleFlags& other) const {
        return ModuleFlags { static_cast<decltype(bits)>(this->bits & other.bits) };
    }
    ModuleFlags& operator&=(const ModuleFlags& other) {
        *this = (*this & other);
        return *this;
    }
    ModuleFlags operator^(const ModuleFlags& other) const {
        return ModuleFlags { static_cast<decltype(bits)>(this->bits ^ other.bits) };
    }
    ModuleFlags& operator^=(const ModuleFlags& other) {
        *this = (*this ^ other);
        return *this;
    }
};
static const ModuleFlags ModuleFlags_NONE = ModuleFlags{ /* .bits = */ (uint32_t)0 };
static const ModuleFlags ModuleFlags_REQUIRES_REV_AD_TRANSFORM = ModuleFlags{ /* .bits = */ (uint32_t)1 };
static const ModuleFlags ModuleFlags_REQUIRES_FWD_AD_TRANSFORM = ModuleFlags{ /* .bits = */ (uint32_t)2 };

struct Module {
    ModuleKind kind;
    Pooled<BasicBlock> entry;
    ModuleFlags flags;
    CArc<ModulePools> pools;
};

struct VectorElementType {
    enum class Tag {
        Scalar,
        Vector,
    };

    struct Scalar_Body {
        Primitive _0;
    };

    struct Vector_Body {
        CArc<VectorType> _0;
    };

    Tag tag;
    union {
        Scalar_Body scalar;
        Vector_Body vector;
    };
};

struct VectorType {
    VectorElementType element;
    uint32_t length;
};

struct MatrixType {
    VectorElementType element;
    uint32_t dimension;
};

template<typename T>
struct CBoxedSlice {
    T *ptr;
    size_t len;
    void (*destructor)(T*, size_t);
};

struct StructType {
    CBoxedSlice<CArc<Type>> fields;
    size_t alignment;
    size_t size;
};

struct ArrayType {
    CArc<Type> element;
    size_t length;
};

struct Type {
    enum class Tag {
        Void,
        UserData,
        Primitive,
        Vector,
        Matrix,
        Struct,
        Array,
        Opaque,
    };

    struct Primitive_Body {
        Primitive _0;
    };

    struct Vector_Body {
        VectorType _0;
    };

    struct Matrix_Body {
        MatrixType _0;
    };

    struct Struct_Body {
        StructType _0;
    };

    struct Array_Body {
        ArrayType _0;
    };

    struct Opaque_Body {
        CBoxedSlice<uint8_t> _0;
    };

    Tag tag;
    union {
        Primitive_Body primitive;
        Vector_Body vector;
        Matrix_Body matrix;
        Struct_Body struct_;
        Array_Body array;
        Opaque_Body opaque;
    };
};

struct BufferBinding {
    uint64_t handle;
    uint64_t offset;
    size_t size;
};

struct TextureBinding {
    uint64_t handle;
    uint32_t level;
};

struct BindlessArrayBinding {
    uint64_t handle;
};

struct AccelBinding {
    uint64_t handle;
};

struct Binding {
    enum class Tag {
        Buffer,
        Texture,
        BindlessArray,
        Accel,
    };

    struct Buffer_Body {
        BufferBinding _0;
    };

    struct Texture_Body {
        TextureBinding _0;
    };

    struct BindlessArray_Body {
        BindlessArrayBinding _0;
    };

    struct Accel_Body {
        AccelBinding _0;
    };

    Tag tag;
    union {
        Buffer_Body buffer;
        Texture_Body texture;
        BindlessArray_Body bindless_array;
        Accel_Body accel;
    };
};

struct Capture {
    NodeRef node;
    Binding binding;
};

struct CpuCustomOp {
    uint8_t *data;
    /// func(data, args); func should modify args in place
    void (*func)(uint8_t*, uint8_t*);
    void (*destructor)(uint8_t*);
    CArc<Type> arg_type;
};

struct CallableModule {
    Module module;
    CArc<Type> ret_type;
    CBoxedSlice<NodeRef> args;
    CBoxedSlice<Capture> captures;
    CBoxedSlice<CArc<CpuCustomOp>> cpu_custom_ops;
    CArc<ModulePools> pools;
};

struct KernelModule {
    Module module;
    CBoxedSlice<Capture> captures;
    CBoxedSlice<NodeRef> args;
    CBoxedSlice<NodeRef> shared;
    CBoxedSlice<CArc<CpuCustomOp>> cpu_custom_ops;
    uint32_t block_size[3];
    CArc<ModulePools> pools;
};

struct CallableModuleRef {
    CArc<CallableModule> _0;
};

struct Func {
    enum class Tag {
        ZeroInitializer,
        Assume,
        Unreachable,
        Assert,
        ThreadId,
        BlockId,
        WarpSize,
        WarpLaneId,
        DispatchId,
        DispatchSize,
        /// (input, grads, ...) -> ()
        PropagateGrad,
        /// (var, idx) -> dvar/dinput_{idx}
        OutputGrad,
        RequiresGradient,
        Backward,
        Gradient,
        GradientMarker,
        AccGrad,
        Detach,
        RayTracingInstanceTransform,
        RayTracingInstanceUserId,
        RayTracingSetInstanceTransform,
        RayTracingSetInstanceOpacity,
        RayTracingSetInstanceVisibility,
        RayTracingSetInstanceUserId,
        RayTracingTraceClosest,
        RayTracingTraceAny,
        RayTracingQueryAll,
        RayTracingQueryAny,
        RayQueryWorldSpaceRay,
        RayQueryProceduralCandidateHit,
        RayQueryTriangleCandidateHit,
        RayQueryCommittedHit,
        RayQueryCommitTriangle,
        RayQueryCommitProcedural,
        RayQueryTerminate,
        RasterDiscard,
        IndirectDispatchSetCount,
        IndirectDispatchSetKernel,
        /// When referencing a Local in Call, it is always interpreted as a load
        /// However, there are cases you want to do this explicitly
        Load,
        Cast,
        Bitcast,
        Pack,
        Unpack,
        Add,
        Sub,
        Mul,
        Div,
        Rem,
        BitAnd,
        BitOr,
        BitXor,
        Shl,
        Shr,
        RotRight,
        RotLeft,
        Eq,
        Ne,
        Lt,
        Le,
        Gt,
        Ge,
        MatCompMul,
        Neg,
        Not,
        BitNot,
        All,
        Any,
        Select,
        Clamp,
        Lerp,
        Step,
        SmoothStep,
        Saturate,
        Abs,
        Min,
        Max,
        ReduceSum,
        ReduceProd,
        ReduceMin,
        ReduceMax,
        Clz,
        Ctz,
        PopCount,
        Reverse,
        IsInf,
        IsNan,
        Acos,
        Acosh,
        Asin,
        Asinh,
        Atan,
        Atan2,
        Atanh,
        Cos,
        Cosh,
        Sin,
        Sinh,
        Tan,
        Tanh,
        Exp,
        Exp2,
        Exp10,
        Log,
        Log2,
        Log10,
        Powi,
        Powf,
        Sqrt,
        Rsqrt,
        Ceil,
        Floor,
        Fract,
        Trunc,
        Round,
        Fma,
        Copysign,
        Cross,
        Dot,
        OuterProduct,
        Length,
        LengthSquared,
        Normalize,
        Faceforward,
        Distance,
        Reflect,
        Determinant,
        Transpose,
        Inverse,
        WarpIsFirstActiveLane,
        WarpFirstActiveLane,
        WarpActiveAllEqual,
        WarpActiveBitAnd,
        WarpActiveBitOr,
        WarpActiveBitXor,
        WarpActiveCountBits,
        WarpActiveMax,
        WarpActiveMin,
        WarpActiveProduct,
        WarpActiveSum,
        WarpActiveAll,
        WarpActiveAny,
        WarpActiveBitMask,
        WarpPrefixCountBits,
        WarpPrefixSum,
        WarpPrefixProduct,
        WarpReadLaneAt,
        WarpReadFirstLane,
        SynchronizeBlock,
        /// (buffer/smem, indices...): do not appear in the final IR, but will be lowered to an Atomic* instruction
        AtomicRef,
        /// (buffer/smem, indices..., desired) -> old: stores desired, returns old.
        AtomicExchange,
        /// (buffer/smem, indices..., expected, desired) -> old: stores (old == expected ? desired : old), returns old.
        AtomicCompareExchange,
        /// (buffer/smem, indices..., val) -> old: stores (old + val), returns old.
        AtomicFetchAdd,
        /// (buffer/smem, indices..., val) -> old: stores (old - val), returns old.
        AtomicFetchSub,
        /// (buffer/smem, indices..., val) -> old: stores (old & val), returns old.
        AtomicFetchAnd,
        /// (buffer/smem, indices..., val) -> old: stores (old | val), returns old.
        AtomicFetchOr,
        /// (buffer/smem, indices..., val) -> old: stores (old ^ val), returns old.
        AtomicFetchXor,
        /// (buffer/smem, indices..., val) -> old: stores min(old, val), returns old.
        AtomicFetchMin,
        /// (buffer/smem, indices..., val) -> old: stores max(old, val), returns old.
        AtomicFetchMax,
        /// (buffer, index) -> value: reads the index-th element in buffer
        BufferRead,
        /// (buffer, index, value) -> void: writes value into the indeex
        BufferWrite,
        /// buffer -> uint: returns buffer size in *elements*
        BufferSize,
        /// (buffer, index_bytes) -> value
        ByteBufferRead,
        /// (buffer, index_bytes, value) -> void
        ByteBufferWrite,
        /// buffer -> size in bytes
        ByteBufferSize,
        /// (texture, coord) -> value
        Texture2dRead,
        /// (texture, coord, value) -> void
        Texture2dWrite,
        /// (texture) -> uint2
        Texture2dSize,
        /// (texture, coord) -> value
        Texture3dRead,
        /// (texture, coord, value) -> void
        Texture3dWrite,
        /// (texture) -> uint3
        Texture3dSize,
        ///(bindless_array, index: uint, uv: float2) -> float4
        BindlessTexture2dSample,
        ///(bindless_array, index: uint, uv: float2, level: float) -> float4
        BindlessTexture2dSampleLevel,
        ///(bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2) -> float4
        BindlessTexture2dSampleGrad,
        ///(bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2, min_mip: float) -> float4
        BindlessTexture2dSampleGradLevel,
        ///(bindless_array, index: uint, uv: float3) -> float4
        BindlessTexture3dSample,
        ///(bindless_array, index: uint, uv: float3, level: float) -> float4
        BindlessTexture3dSampleLevel,
        ///(bindless_array, index: uint, uv: float3, ddx: float3, ddy: float3) -> float4
        BindlessTexture3dSampleGrad,
        ///(bindless_array, index: uint, uv: float2, ddx: float2, ddy: float2, min_mip: float) -> float4
        BindlessTexture3dSampleGradLevel,
        ///(bindless_array, index: uint, coord: uint2) -> float4
        BindlessTexture2dRead,
        ///(bindless_array, index: uint, coord: uint3) -> float4
        BindlessTexture3dRead,
        ///(bindless_array, index: uint, coord: uint2, level: uint) -> float4
        BindlessTexture2dReadLevel,
        ///(bindless_array, index: uint, coord: uint3, level: uint) -> float4
        BindlessTexture3dReadLevel,
        ///(bindless_array, index: uint) -> uint2
        BindlessTexture2dSize,
        ///(bindless_array, index: uint) -> uint3
        BindlessTexture3dSize,
        ///(bindless_array, index: uint, level: uint) -> uint2
        BindlessTexture2dSizeLevel,
        ///(bindless_array, index: uint, level: uint) -> uint3
        BindlessTexture3dSizeLevel,
        /// (bindless_array, index: uint, element: uint) -> T
        BindlessBufferRead,
        /// (bindless_array, index: uint, stride: uint) -> uint: returns the size of the buffer in *elements*
        BindlessBufferSize,
        BindlessBufferType,
        BindlessByteBufferRead,
        Vec,
        Vec2,
        Vec3,
        Vec4,
        Permute,
        InsertElement,
        ExtractElement,
        GetElementPtr,
        Struct,
        Array,
        Mat,
        Mat2,
        Mat3,
        Mat4,
        Callable,
        CpuCustomOp,
        ShaderExecutionReorder,
        Unknown0,
        Unknown1,
    };

    struct Unreachable_Body {
        CBoxedSlice<uint8_t> _0;
    };

    struct Assert_Body {
        CBoxedSlice<uint8_t> _0;
    };

    struct Callable_Body {
        CallableModuleRef _0;
    };

    struct CpuCustomOp_Body {
        CArc<CpuCustomOp> _0;
    };

    Tag tag;
    union {
        Unreachable_Body unreachable;
        Assert_Body assert;
        Callable_Body callable;
        CpuCustomOp_Body cpu_custom_op;
    };
};

template<typename T>
struct CSlice {
    const T *ptr;
    size_t len;
};

struct Const {
    enum class Tag {
        Zero,
        One,
        Bool,
        Int8,
        Uint8,
        Int16,
        Uint16,
        Int32,
        Uint32,
        Int64,
        Uint64,
        Float16,
        Float32,
        Float64,
        Generic,
    };

    struct Zero_Body {
        CArc<Type> _0;
    };

    struct One_Body {
        CArc<Type> _0;
    };

    struct Bool_Body {
        bool _0;
    };

    struct Int8_Body {
        int8_t _0;
    };

    struct Uint8_Body {
        uint8_t _0;
    };

    struct Int16_Body {
        int16_t _0;
    };

    struct Uint16_Body {
        uint16_t _0;
    };

    struct Int32_Body {
        int32_t _0;
    };

    struct Uint32_Body {
        uint32_t _0;
    };

    struct Int64_Body {
        int64_t _0;
    };

    struct Uint64_Body {
        uint64_t _0;
    };

    struct Float16_Body {
        c_half _0;
    };

    struct Float32_Body {
        float _0;
    };

    struct Float64_Body {
        double _0;
    };

    struct Generic_Body {
        CBoxedSlice<uint8_t> _0;
        CArc<Type> _1;
    };

    Tag tag;
    union {
        Zero_Body zero;
        One_Body one;
        Bool_Body bool_;
        Int8_Body int8;
        Uint8_Body uint8;
        Int16_Body int16;
        Uint16_Body uint16;
        Int32_Body int32;
        Uint32_Body uint32;
        Int64_Body int64;
        Uint64_Body uint64;
        Float16_Body float16;
        Float32_Body float32;
        Float64_Body float64;
        Generic_Body generic;
    };
};

struct PhiIncoming {
    NodeRef value;
    Pooled<BasicBlock> block;
};

struct SwitchCase {
    int32_t value;
    Pooled<BasicBlock> block;
};

struct BlockModule {
    Module module;
};

struct UserData {
    uint64_t tag;
    const uint8_t *data;
    bool (*eq)(const uint8_t*, const uint8_t*);
};

struct Instruction {
    enum class Tag {
        Buffer,
        Bindless,
        Texture2D,
        Texture3D,
        Accel,
        Shared,
        Uniform,
        Local,
        Argument,
        UserData,
        Invalid,
        Const,
        Update,
        Call,
        Phi,
        Return,
        Loop,
        GenericLoop,
        Break,
        Continue,
        If,
        Switch,
        AdScope,
        RayQuery,
        Print,
        AdDetach,
        Comment,
    };

    struct Local_Body {
        NodeRef init;
    };

    struct Argument_Body {
        bool by_value;
    };

    struct UserData_Body {
        CArc<UserData> _0;
    };

    struct Const_Body {
        Const _0;
    };

    struct Update_Body {
        NodeRef var;
        NodeRef value;
    };

    struct Call_Body {
        Func _0;
        CBoxedSlice<NodeRef> _1;
    };

    struct Phi_Body {
        CBoxedSlice<PhiIncoming> _0;
    };

    struct Return_Body {
        NodeRef _0;
    };

    struct Loop_Body {
        Pooled<BasicBlock> body;
        NodeRef cond;
    };

    struct GenericLoop_Body {
        Pooled<BasicBlock> prepare;
        NodeRef cond;
        Pooled<BasicBlock> body;
        Pooled<BasicBlock> update;
    };

    struct If_Body {
        NodeRef cond;
        Pooled<BasicBlock> true_branch;
        Pooled<BasicBlock> false_branch;
    };

    struct Switch_Body {
        NodeRef value;
        Pooled<BasicBlock> default_;
        CBoxedSlice<SwitchCase> cases;
    };

    struct AdScope_Body {
        Pooled<BasicBlock> body;
        bool forward;
        size_t n_forward_grads;
    };

    struct RayQuery_Body {
        NodeRef ray_query;
        Pooled<BasicBlock> on_triangle_hit;
        Pooled<BasicBlock> on_procedural_hit;
    };

    struct Print_Body {
        CBoxedSlice<uint8_t> fmt;
        CBoxedSlice<NodeRef> args;
    };

    struct AdDetach_Body {
        Pooled<BasicBlock> _0;
    };

    struct Comment_Body {
        CBoxedSlice<uint8_t> _0;
    };

    Tag tag;
    union {
        Local_Body local;
        Argument_Body argument;
        UserData_Body user_data;
        Const_Body const_;
        Update_Body update;
        Call_Body call;
        Phi_Body phi;
        Return_Body return_;
        Loop_Body loop;
        GenericLoop_Body generic_loop;
        If_Body if_;
        Switch_Body switch_;
        AdScope_Body ad_scope;
        RayQuery_Body ray_query;
        Print_Body print;
        AdDetach_Body ad_detach;
        Comment_Body comment;
    };
};

struct Node {
    CArc<Type> type_;
    NodeRef next;
    NodeRef prev;
    CArc<Instruction> instruction;
};

template<typename T>
struct CBox {
    T *ptr;
    void (*destructor)(T*);
};



static const NodeRef INVALID_REF = NodeRef{ /* ._0 = */ 0 };


extern "C" {

void luisa_compute_ir_append_node(IrBuilder *builder, NodeRef node_ref);

CArcSharedBlock<CallableModule> *luisa_compute_ir_ast_json_to_ir_callable(CBoxedSlice<uint8_t> j);

CArcSharedBlock<KernelModule> *luisa_compute_ir_ast_json_to_ir_kernel(CBoxedSlice<uint8_t> j);

NodeRef luisa_compute_ir_build_call(IrBuilder *builder,
                                    Func func,
                                    CSlice<NodeRef> args,
                                    CArc<Type> ret_type);

NodeRef luisa_compute_ir_build_const(IrBuilder *builder, Const const_);

Pooled<BasicBlock> luisa_compute_ir_build_finish(IrBuilder builder);

NodeRef luisa_compute_ir_build_generic_loop(IrBuilder *builder,
                                            Pooled<BasicBlock> prepare,
                                            NodeRef cond,
                                            Pooled<BasicBlock> body,
                                            Pooled<BasicBlock> update);

NodeRef luisa_compute_ir_build_if(IrBuilder *builder,
                                  NodeRef cond,
                                  Pooled<BasicBlock> true_branch,
                                  Pooled<BasicBlock> false_branch);

NodeRef luisa_compute_ir_build_local(IrBuilder *builder, NodeRef init);

NodeRef luisa_compute_ir_build_local_zero_init(IrBuilder *builder, CArc<Type> ty);

NodeRef luisa_compute_ir_build_loop(IrBuilder *builder, Pooled<BasicBlock> body, NodeRef cond);

NodeRef luisa_compute_ir_build_phi(IrBuilder *builder, CSlice<PhiIncoming> incoming, CArc<Type> t);

NodeRef luisa_compute_ir_build_switch(IrBuilder *builder,
                                      NodeRef value,
                                      CSlice<SwitchCase> cases,
                                      Pooled<BasicBlock> default_);

void luisa_compute_ir_build_update(IrBuilder *builder, NodeRef var, NodeRef value);

void luisa_compute_ir_builder_set_insert_point(IrBuilder *builder, NodeRef node_ref);

CBoxedSlice<uint8_t> luisa_compute_ir_dump_binary(const Module *module);

CBoxedSlice<uint8_t> luisa_compute_ir_dump_human_readable(const Module *module);

CBoxedSlice<uint8_t> luisa_compute_ir_dump_json(const Module *module);

CArcSharedBlock<BlockModule> *luisa_compute_ir_new_block_module(BlockModule m);

IrBuilder luisa_compute_ir_new_builder(CArc<ModulePools> pools);

CArcSharedBlock<CallableModule> *luisa_compute_ir_new_callable_module(CallableModule m);

CArcSharedBlock<Instruction> *luisa_compute_ir_new_instruction(Instruction inst);

CArcSharedBlock<KernelModule> *luisa_compute_ir_new_kernel_module(KernelModule m);

CArcSharedBlock<ModulePools> *luisa_compute_ir_new_module_pools();

NodeRef luisa_compute_ir_new_node(CArc<ModulePools> pools, Node node);

const Node *luisa_compute_ir_node_get(NodeRef node_ref);

void luisa_compute_ir_node_insert_after_self(NodeRef node_ref, NodeRef new_node);

void luisa_compute_ir_node_insert_before_self(NodeRef node_ref, NodeRef new_node);

void luisa_compute_ir_node_remove(NodeRef node_ref);

void luisa_compute_ir_node_replace_with(NodeRef node_ref, const Node *new_node);

CBoxedSlice<uint8_t> luisa_compute_ir_node_usage(const KernelModule *kernel);

CArcSharedBlock<Type> *luisa_compute_ir_register_type(const Type *ty);

Module luisa_compute_ir_transform_auto(Module module);

void luisa_compute_ir_transform_pipeline_add_transform(TransformPipeline *pipeline,
                                                       const char *name);

void luisa_compute_ir_transform_pipeline_destroy(TransformPipeline *pipeline);

TransformPipeline *luisa_compute_ir_transform_pipeline_new();

Module luisa_compute_ir_transform_pipeline_transform(TransformPipeline *pipeline, Module module);

size_t luisa_compute_ir_type_alignment(const CArc<Type> *ty);

size_t luisa_compute_ir_type_size(const CArc<Type> *ty);

} // extern "C"

} // namespace luisa::compute::ir
