#pragma once
#include <luisa/ir/fwd.h>
namespace luisa::compute::ir_v2 {
using raw::Primitive;
class LC_IR_API VectorElementType : concepts::Noncopyable {
    raw::VectorElementType _inner;
    class Marker{ };
public:
    using Tag = raw::VectorElementType::Tag;
    class LC_IR_API Scalar : Marker, concepts::Noncopyable {
        raw::VectorElementType::Scalar_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::VectorElementType::Tag::Scalar; }
    };
    class LC_IR_API Vector : Marker, concepts::Noncopyable {
        raw::VectorElementType::Vector_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::VectorElementType::Tag::Vector; }
    };
    template<class T>
    [[nodiscard]] bool isa() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        return _inner.tag == T::tag();
    }
    template<class T>
    [[nodiscard]] const T* as() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        if (!isa<T>()) return nullptr;
        if constexpr (std::is_same_v<T, Scalar>) {
            return reinterpret_cast<const Scalar*>(&_inner.scalar);
        }
        if constexpr (std::is_same_v<T, Vector>) {
            return reinterpret_cast<const Vector*>(&_inner.vector);
        }
        return reinterpret_cast<const T*>(this);
    }
};
static_assert(sizeof(VectorElementType) == sizeof(raw::VectorElementType));
namespace detail {
template<>struct FromInnerRef<raw::VectorElementType>{
    using Output = VectorElementType;
    static const Output& from(const raw::VectorElementType & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::VectorElementType>>{
    using Output = CArc<VectorElementType>;
    static const Output& from(const CArc<raw::VectorElementType> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::VectorElementType>>{
    using Output = Pooled<VectorElementType>;
    static const Output& from(const Pooled<raw::VectorElementType> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::VectorElementType>>{
    using Output = CBoxedSlice<VectorElementType>;
    static const Output& from(const CBoxedSlice<raw::VectorElementType> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
class LC_IR_API VectorType : concepts::Noncopyable {
    raw::VectorType _inner;
public:
        [[nodiscard]] const VectorElementType& element() const noexcept;
        [[nodiscard]] const uint32_t& length() const noexcept;
};
namespace detail {
template<>struct FromInnerRef<raw::VectorType>{
    using Output = VectorType;
    static const Output& from(const raw::VectorType & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::VectorType>>{
    using Output = CArc<VectorType>;
    static const Output& from(const CArc<raw::VectorType> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::VectorType>>{
    using Output = Pooled<VectorType>;
    static const Output& from(const Pooled<raw::VectorType> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::VectorType>>{
    using Output = CBoxedSlice<VectorType>;
    static const Output& from(const CBoxedSlice<raw::VectorType> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
class LC_IR_API MatrixType : concepts::Noncopyable {
    raw::MatrixType _inner;
public:
        [[nodiscard]] const VectorElementType& element() const noexcept;
        [[nodiscard]] const uint32_t& dimension() const noexcept;
};
namespace detail {
template<>struct FromInnerRef<raw::MatrixType>{
    using Output = MatrixType;
    static const Output& from(const raw::MatrixType & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::MatrixType>>{
    using Output = CArc<MatrixType>;
    static const Output& from(const CArc<raw::MatrixType> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::MatrixType>>{
    using Output = Pooled<MatrixType>;
    static const Output& from(const Pooled<raw::MatrixType> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::MatrixType>>{
    using Output = CBoxedSlice<MatrixType>;
    static const Output& from(const CBoxedSlice<raw::MatrixType> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
class LC_IR_API StructType : concepts::Noncopyable {
    raw::StructType _inner;
public:
        [[nodiscard]] luisa::span<const CArc < Type >> fields() const noexcept;
        [[nodiscard]] const size_t& alignment() const noexcept;
        [[nodiscard]] const size_t& size() const noexcept;
};
namespace detail {
template<>struct FromInnerRef<raw::StructType>{
    using Output = StructType;
    static const Output& from(const raw::StructType & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::StructType>>{
    using Output = CArc<StructType>;
    static const Output& from(const CArc<raw::StructType> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::StructType>>{
    using Output = Pooled<StructType>;
    static const Output& from(const Pooled<raw::StructType> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::StructType>>{
    using Output = CBoxedSlice<StructType>;
    static const Output& from(const CBoxedSlice<raw::StructType> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
class LC_IR_API ArrayType : concepts::Noncopyable {
    raw::ArrayType _inner;
public:
        [[nodiscard]] const CArc < Type >& element() const noexcept;
        [[nodiscard]] const size_t& length() const noexcept;
};
namespace detail {
template<>struct FromInnerRef<raw::ArrayType>{
    using Output = ArrayType;
    static const Output& from(const raw::ArrayType & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::ArrayType>>{
    using Output = CArc<ArrayType>;
    static const Output& from(const CArc<raw::ArrayType> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::ArrayType>>{
    using Output = Pooled<ArrayType>;
    static const Output& from(const Pooled<raw::ArrayType> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::ArrayType>>{
    using Output = CBoxedSlice<ArrayType>;
    static const Output& from(const CBoxedSlice<raw::ArrayType> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
class LC_IR_API Type : concepts::Noncopyable {
    raw::Type _inner;
    class Marker{ };
public:
    using Tag = raw::Type::Tag;
    class LC_IR_API Void : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API UserData : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Primitive : Marker, concepts::Noncopyable {
        raw::Type::Primitive_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Type::Tag::Primitive; }
    };
    class LC_IR_API Vector : Marker, concepts::Noncopyable {
        raw::Type::Vector_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Type::Tag::Vector; }
    };
    class LC_IR_API Matrix : Marker, concepts::Noncopyable {
        raw::Type::Matrix_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Type::Tag::Matrix; }
    };
    class LC_IR_API Struct : Marker, concepts::Noncopyable {
        raw::Type::Struct_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Type::Tag::Struct; }
    };
    class LC_IR_API Array : Marker, concepts::Noncopyable {
        raw::Type::Array_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Type::Tag::Array; }
    };
    class LC_IR_API Opaque : Marker, concepts::Noncopyable {
        raw::Type::Opaque_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Type::Tag::Opaque; }
    };
    template<class T>
    [[nodiscard]] bool isa() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        return _inner.tag == T::tag();
    }
    template<class T>
    [[nodiscard]] const T* as() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        if (!isa<T>()) return nullptr;
        if constexpr (std::is_same_v<T, Primitive>) {
            return reinterpret_cast<const Primitive*>(&_inner.primitive);
        }
        if constexpr (std::is_same_v<T, Vector>) {
            return reinterpret_cast<const Vector*>(&_inner.vector);
        }
        if constexpr (std::is_same_v<T, Matrix>) {
            return reinterpret_cast<const Matrix*>(&_inner.matrix);
        }
        if constexpr (std::is_same_v<T, Struct>) {
            return reinterpret_cast<const Struct*>(&_inner.struct_);
        }
        if constexpr (std::is_same_v<T, Array>) {
            return reinterpret_cast<const Array*>(&_inner.array);
        }
        if constexpr (std::is_same_v<T, Opaque>) {
            return reinterpret_cast<const Opaque*>(&_inner.opaque);
        }
        return reinterpret_cast<const T*>(this);
    }
};
static_assert(sizeof(Type) == sizeof(raw::Type));
namespace detail {
template<>struct FromInnerRef<raw::Type>{
    using Output = Type;
    static const Output& from(const raw::Type & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::Type>>{
    using Output = CArc<Type>;
    static const Output& from(const CArc<raw::Type> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::Type>>{
    using Output = Pooled<Type>;
    static const Output& from(const Pooled<raw::Type> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::Type>>{
    using Output = CBoxedSlice<Type>;
    static const Output& from(const CBoxedSlice<raw::Type> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
class LC_IR_API Node : concepts::Noncopyable {
    raw::Node _inner;
public:
        [[nodiscard]] const CArc < Type >& type_() const noexcept;
        [[nodiscard]] const NodeRef& next() const noexcept;
        [[nodiscard]] const NodeRef& prev() const noexcept;
        [[nodiscard]] const CArc < Instruction >& instruction() const noexcept;
};
namespace detail {
template<>struct FromInnerRef<raw::Node>{
    using Output = Node;
    static const Output& from(const raw::Node & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::Node>>{
    using Output = CArc<Node>;
    static const Output& from(const CArc<raw::Node> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::Node>>{
    using Output = Pooled<Node>;
    static const Output& from(const Pooled<raw::Node> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::Node>>{
    using Output = CBoxedSlice<Node>;
    static const Output& from(const CBoxedSlice<raw::Node> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
class LC_IR_API Func : concepts::Noncopyable {
    raw::Func _inner;
    class Marker{ };
public:
    using Tag = raw::Func::Tag;
    class LC_IR_API ZeroInitializer : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Assume : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Unreachable : Marker, concepts::Noncopyable {
        raw::Func::Unreachable_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Unreachable; }
    };
    class LC_IR_API Assert : Marker, concepts::Noncopyable {
        raw::Func::Assert_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Assert; }
    };
    class LC_IR_API ThreadId : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BlockId : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API DispatchId : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API DispatchSize : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API RequiresGradient : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Backward : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Gradient : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API GradientMarker : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API AccGrad : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Detach : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API RayTracingInstanceTransform : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API RayTracingSetInstanceTransform : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API RayTracingSetInstanceOpacity : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API RayTracingSetInstanceVisibility : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API RayTracingTraceClosest : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API RayTracingTraceAny : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API RayTracingQueryAll : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API RayTracingQueryAny : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API RayQueryWorldSpaceRay : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API RayQueryProceduralCandidateHit : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API RayQueryTriangleCandidateHit : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API RayQueryCommittedHit : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API RayQueryCommitTriangle : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API RayQueryCommitProcedural : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API RayQueryTerminate : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API RasterDiscard : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API IndirectClearDispatchBuffer : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API IndirectEmplaceDispatchKernel : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Load : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Cast : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Bitcast : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Add : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Sub : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Mul : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Div : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Rem : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BitAnd : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BitOr : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BitXor : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Shl : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Shr : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API RotRight : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API RotLeft : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Eq : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Ne : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Lt : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Le : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Gt : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Ge : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API MatCompMul : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Neg : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Not : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BitNot : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API All : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Any : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Select : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Clamp : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Lerp : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Step : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API SmoothStep : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Saturate : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Abs : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Min : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Max : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API ReduceSum : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API ReduceProd : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API ReduceMin : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API ReduceMax : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Clz : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Ctz : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API PopCount : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Reverse : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API IsInf : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API IsNan : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Acos : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Acosh : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Asin : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Asinh : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Atan : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Atan2 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Atanh : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Cos : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Cosh : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Sin : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Sinh : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Tan : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Tanh : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Exp : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Exp2 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Exp10 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Log : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Log2 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Log10 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Powi : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Powf : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Sqrt : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Rsqrt : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Ceil : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Floor : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Fract : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Trunc : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Round : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Fma : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Copysign : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Cross : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Dot : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API OuterProduct : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Length : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API LengthSquared : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Normalize : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Faceforward : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Reflect : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Determinant : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Transpose : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Inverse : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API SynchronizeBlock : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API AtomicExchange : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API AtomicCompareExchange : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API AtomicFetchAdd : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API AtomicFetchSub : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API AtomicFetchAnd : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API AtomicFetchOr : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API AtomicFetchXor : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API AtomicFetchMin : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API AtomicFetchMax : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BufferRead : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BufferWrite : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BufferSize : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Texture2dRead : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Texture2dWrite : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Texture3dRead : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Texture3dWrite : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BindlessTexture2dSample : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BindlessTexture2dSampleLevel : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BindlessTexture2dSampleGrad : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BindlessTexture2dSampleGradLevel : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BindlessTexture3dSample : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BindlessTexture3dSampleLevel : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BindlessTexture3dSampleGrad : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BindlessTexture3dSampleGradLevel : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BindlessTexture2dRead : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BindlessTexture3dRead : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BindlessTexture2dReadLevel : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BindlessTexture3dReadLevel : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BindlessTexture2dSize : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BindlessTexture3dSize : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BindlessTexture2dSizeLevel : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BindlessTexture3dSizeLevel : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BindlessBufferRead : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API BindlessBufferSize : Marker, concepts::Noncopyable {
        raw::Func::BindlessBufferSize_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BindlessBufferSize; }
    };
    class LC_IR_API BindlessBufferType : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Vec : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Vec2 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Vec3 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Vec4 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Permute : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API InsertElement : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API ExtractElement : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API GetElementPtr : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Struct : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Array : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Mat : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Mat2 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Mat3 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Mat4 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Callable : Marker, concepts::Noncopyable {
        raw::Func::Callable_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Callable; }
    };
    class LC_IR_API CpuCustomOp : Marker, concepts::Noncopyable {
        raw::Func::CpuCustomOp_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::CpuCustomOp; }
    };
    template<class T>
    [[nodiscard]] bool isa() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        return _inner.tag == T::tag();
    }
    template<class T>
    [[nodiscard]] const T* as() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        if (!isa<T>()) return nullptr;
        if constexpr (std::is_same_v<T, Unreachable>) {
            return reinterpret_cast<const Unreachable*>(&_inner.unreachable);
        }
        if constexpr (std::is_same_v<T, Assert>) {
            return reinterpret_cast<const Assert*>(&_inner.assert);
        }
        if constexpr (std::is_same_v<T, BindlessBufferSize>) {
            return reinterpret_cast<const BindlessBufferSize*>(&_inner.bindless_buffer_size);
        }
        if constexpr (std::is_same_v<T, Callable>) {
            return reinterpret_cast<const Callable*>(&_inner.callable);
        }
        if constexpr (std::is_same_v<T, CpuCustomOp>) {
            return reinterpret_cast<const CpuCustomOp*>(&_inner.cpu_custom_op);
        }
        return reinterpret_cast<const T*>(this);
    }
};
static_assert(sizeof(Func) == sizeof(raw::Func));
namespace detail {
template<>struct FromInnerRef<raw::Func>{
    using Output = Func;
    static const Output& from(const raw::Func & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::Func>>{
    using Output = CArc<Func>;
    static const Output& from(const CArc<raw::Func> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::Func>>{
    using Output = Pooled<Func>;
    static const Output& from(const Pooled<raw::Func> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::Func>>{
    using Output = CBoxedSlice<Func>;
    static const Output& from(const CBoxedSlice<raw::Func> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
class LC_IR_API Const : concepts::Noncopyable {
    raw::Const _inner;
    class Marker{ };
public:
    using Tag = raw::Const::Tag;
    class LC_IR_API Zero : Marker, concepts::Noncopyable {
        raw::Const::Zero_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Zero; }
    };
    class LC_IR_API One : Marker, concepts::Noncopyable {
        raw::Const::One_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::One; }
    };
    class LC_IR_API Bool : Marker, concepts::Noncopyable {
        raw::Const::Bool_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Bool; }
    };
    class LC_IR_API Int32 : Marker, concepts::Noncopyable {
        raw::Const::Int32_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Int32; }
    };
    class LC_IR_API Uint32 : Marker, concepts::Noncopyable {
        raw::Const::Uint32_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Uint32; }
    };
    class LC_IR_API Int64 : Marker, concepts::Noncopyable {
        raw::Const::Int64_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Int64; }
    };
    class LC_IR_API Uint64 : Marker, concepts::Noncopyable {
        raw::Const::Uint64_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Uint64; }
    };
    class LC_IR_API Float32 : Marker, concepts::Noncopyable {
        raw::Const::Float32_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Float32; }
    };
    class LC_IR_API Float64 : Marker, concepts::Noncopyable {
        raw::Const::Float64_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Float64; }
    };
    class LC_IR_API Generic : Marker, concepts::Noncopyable {
        raw::Const::Generic_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Generic; }
    };
    template<class T>
    [[nodiscard]] bool isa() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        return _inner.tag == T::tag();
    }
    template<class T>
    [[nodiscard]] const T* as() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        if (!isa<T>()) return nullptr;
        if constexpr (std::is_same_v<T, Zero>) {
            return reinterpret_cast<const Zero*>(&_inner.zero);
        }
        if constexpr (std::is_same_v<T, One>) {
            return reinterpret_cast<const One*>(&_inner.one);
        }
        if constexpr (std::is_same_v<T, Bool>) {
            return reinterpret_cast<const Bool*>(&_inner.bool_);
        }
        if constexpr (std::is_same_v<T, Int32>) {
            return reinterpret_cast<const Int32*>(&_inner.int32);
        }
        if constexpr (std::is_same_v<T, Uint32>) {
            return reinterpret_cast<const Uint32*>(&_inner.uint32);
        }
        if constexpr (std::is_same_v<T, Int64>) {
            return reinterpret_cast<const Int64*>(&_inner.int64);
        }
        if constexpr (std::is_same_v<T, Uint64>) {
            return reinterpret_cast<const Uint64*>(&_inner.uint64);
        }
        if constexpr (std::is_same_v<T, Float32>) {
            return reinterpret_cast<const Float32*>(&_inner.float32);
        }
        if constexpr (std::is_same_v<T, Float64>) {
            return reinterpret_cast<const Float64*>(&_inner.float64);
        }
        if constexpr (std::is_same_v<T, Generic>) {
            return reinterpret_cast<const Generic*>(&_inner.generic);
        }
        return reinterpret_cast<const T*>(this);
    }
};
static_assert(sizeof(Const) == sizeof(raw::Const));
namespace detail {
template<>struct FromInnerRef<raw::Const>{
    using Output = Const;
    static const Output& from(const raw::Const & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::Const>>{
    using Output = CArc<Const>;
    static const Output& from(const CArc<raw::Const> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::Const>>{
    using Output = Pooled<Const>;
    static const Output& from(const Pooled<raw::Const> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::Const>>{
    using Output = CBoxedSlice<Const>;
    static const Output& from(const CBoxedSlice<raw::Const> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
class LC_IR_API PhiIncoming : concepts::Noncopyable {
    raw::PhiIncoming _inner;
public:
        [[nodiscard]] const NodeRef& value() const noexcept;
        [[nodiscard]] const Pooled < BasicBlock >& block() const noexcept;
};
namespace detail {
template<>struct FromInnerRef<raw::PhiIncoming>{
    using Output = PhiIncoming;
    static const Output& from(const raw::PhiIncoming & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::PhiIncoming>>{
    using Output = CArc<PhiIncoming>;
    static const Output& from(const CArc<raw::PhiIncoming> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::PhiIncoming>>{
    using Output = Pooled<PhiIncoming>;
    static const Output& from(const Pooled<raw::PhiIncoming> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::PhiIncoming>>{
    using Output = CBoxedSlice<PhiIncoming>;
    static const Output& from(const CBoxedSlice<raw::PhiIncoming> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
class LC_IR_API SwitchCase : concepts::Noncopyable {
    raw::SwitchCase _inner;
public:
        [[nodiscard]] const int32_t& value() const noexcept;
        [[nodiscard]] const Pooled < BasicBlock >& block() const noexcept;
};
namespace detail {
template<>struct FromInnerRef<raw::SwitchCase>{
    using Output = SwitchCase;
    static const Output& from(const raw::SwitchCase & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::SwitchCase>>{
    using Output = CArc<SwitchCase>;
    static const Output& from(const CArc<raw::SwitchCase> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::SwitchCase>>{
    using Output = Pooled<SwitchCase>;
    static const Output& from(const Pooled<raw::SwitchCase> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::SwitchCase>>{
    using Output = CBoxedSlice<SwitchCase>;
    static const Output& from(const CBoxedSlice<raw::SwitchCase> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
class LC_IR_API Instruction : concepts::Noncopyable {
    raw::Instruction _inner;
    class Marker{ };
public:
    using Tag = raw::Instruction::Tag;
    class LC_IR_API Buffer : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Bindless : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Texture2D : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Texture3D : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Accel : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Shared : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Uniform : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Local : Marker, concepts::Noncopyable {
        raw::Instruction::Local_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Local; }
        [[nodiscard]] const NodeRef& init() const noexcept;
    };
    class LC_IR_API Argument : Marker, concepts::Noncopyable {
        raw::Instruction::Argument_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Argument; }
        [[nodiscard]] const bool& by_value() const noexcept;
    };
    class LC_IR_API UserData : Marker, concepts::Noncopyable {
        raw::Instruction::UserData_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::UserData; }
    };
    class LC_IR_API Invalid : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Const : Marker, concepts::Noncopyable {
        raw::Instruction::Const_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Const; }
    };
    class LC_IR_API Update : Marker, concepts::Noncopyable {
        raw::Instruction::Update_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Update; }
        [[nodiscard]] const NodeRef& var() const noexcept;
        [[nodiscard]] const NodeRef& value() const noexcept;
    };
    class LC_IR_API Call : Marker, concepts::Noncopyable {
        raw::Instruction::Call_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Call; }
    };
    class LC_IR_API Phi : Marker, concepts::Noncopyable {
        raw::Instruction::Phi_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Phi; }
    };
    class LC_IR_API Return : Marker, concepts::Noncopyable {
        raw::Instruction::Return_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Return; }
    };
    class LC_IR_API Loop : Marker, concepts::Noncopyable {
        raw::Instruction::Loop_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Loop; }
        [[nodiscard]] const Pooled < BasicBlock >& body() const noexcept;
        [[nodiscard]] const NodeRef& cond() const noexcept;
    };
    class LC_IR_API GenericLoop : Marker, concepts::Noncopyable {
        raw::Instruction::GenericLoop_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::GenericLoop; }
        [[nodiscard]] const Pooled < BasicBlock >& prepare() const noexcept;
        [[nodiscard]] const NodeRef& cond() const noexcept;
        [[nodiscard]] const Pooled < BasicBlock >& body() const noexcept;
        [[nodiscard]] const Pooled < BasicBlock >& update() const noexcept;
    };
    class LC_IR_API Break : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API Continue : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
    };
    class LC_IR_API If : Marker, concepts::Noncopyable {
        raw::Instruction::If_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::If; }
        [[nodiscard]] const NodeRef& cond() const noexcept;
        [[nodiscard]] const Pooled < BasicBlock >& true_branch() const noexcept;
        [[nodiscard]] const Pooled < BasicBlock >& false_branch() const noexcept;
    };
    class LC_IR_API Switch : Marker, concepts::Noncopyable {
        raw::Instruction::Switch_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Switch; }
        [[nodiscard]] const NodeRef& value() const noexcept;
        [[nodiscard]] const Pooled < BasicBlock >& default_() const noexcept;
        [[nodiscard]] luisa::span<const SwitchCase> cases() const noexcept;
    };
    class LC_IR_API AdScope : Marker, concepts::Noncopyable {
        raw::Instruction::AdScope_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::AdScope; }
        [[nodiscard]] const Pooled < BasicBlock >& body() const noexcept;
    };
    class LC_IR_API RayQuery : Marker, concepts::Noncopyable {
        raw::Instruction::RayQuery_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::RayQuery; }
        [[nodiscard]] const NodeRef& ray_query() const noexcept;
        [[nodiscard]] const Pooled < BasicBlock >& on_triangle_hit() const noexcept;
        [[nodiscard]] const Pooled < BasicBlock >& on_procedural_hit() const noexcept;
    };
    class LC_IR_API AdDetach : Marker, concepts::Noncopyable {
        raw::Instruction::AdDetach_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::AdDetach; }
    };
    class LC_IR_API Comment : Marker, concepts::Noncopyable {
        raw::Instruction::Comment_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Comment; }
    };
    template<class T>
    [[nodiscard]] bool isa() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        return _inner.tag == T::tag();
    }
    template<class T>
    [[nodiscard]] const T* as() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        if (!isa<T>()) return nullptr;
        if constexpr (std::is_same_v<T, Local>) {
            return reinterpret_cast<const Local*>(&_inner.local);
        }
        if constexpr (std::is_same_v<T, Argument>) {
            return reinterpret_cast<const Argument*>(&_inner.argument);
        }
        if constexpr (std::is_same_v<T, UserData>) {
            return reinterpret_cast<const UserData*>(&_inner.user_data);
        }
        if constexpr (std::is_same_v<T, Const>) {
            return reinterpret_cast<const Const*>(&_inner.const_);
        }
        if constexpr (std::is_same_v<T, Update>) {
            return reinterpret_cast<const Update*>(&_inner.update);
        }
        if constexpr (std::is_same_v<T, Call>) {
            return reinterpret_cast<const Call*>(&_inner.call);
        }
        if constexpr (std::is_same_v<T, Phi>) {
            return reinterpret_cast<const Phi*>(&_inner.phi);
        }
        if constexpr (std::is_same_v<T, Return>) {
            return reinterpret_cast<const Return*>(&_inner.return_);
        }
        if constexpr (std::is_same_v<T, Loop>) {
            return reinterpret_cast<const Loop*>(&_inner.loop);
        }
        if constexpr (std::is_same_v<T, GenericLoop>) {
            return reinterpret_cast<const GenericLoop*>(&_inner.generic_loop);
        }
        if constexpr (std::is_same_v<T, If>) {
            return reinterpret_cast<const If*>(&_inner.if_);
        }
        if constexpr (std::is_same_v<T, Switch>) {
            return reinterpret_cast<const Switch*>(&_inner.switch_);
        }
        if constexpr (std::is_same_v<T, AdScope>) {
            return reinterpret_cast<const AdScope*>(&_inner.ad_scope);
        }
        if constexpr (std::is_same_v<T, RayQuery>) {
            return reinterpret_cast<const RayQuery*>(&_inner.ray_query);
        }
        if constexpr (std::is_same_v<T, AdDetach>) {
            return reinterpret_cast<const AdDetach*>(&_inner.ad_detach);
        }
        if constexpr (std::is_same_v<T, Comment>) {
            return reinterpret_cast<const Comment*>(&_inner.comment);
        }
        return reinterpret_cast<const T*>(this);
    }
};
static_assert(sizeof(Instruction) == sizeof(raw::Instruction));
namespace detail {
template<>struct FromInnerRef<raw::Instruction>{
    using Output = Instruction;
    static const Output& from(const raw::Instruction & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::Instruction>>{
    using Output = CArc<Instruction>;
    static const Output& from(const CArc<raw::Instruction> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::Instruction>>{
    using Output = Pooled<Instruction>;
    static const Output& from(const Pooled<raw::Instruction> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::Instruction>>{
    using Output = CBoxedSlice<Instruction>;
    static const Output& from(const CBoxedSlice<raw::Instruction> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
class LC_IR_API BasicBlock : concepts::Noncopyable {
    raw::BasicBlock _inner;
public:
        [[nodiscard]] const NodeRef& first() const noexcept;
        [[nodiscard]] const NodeRef& last() const noexcept;
};
namespace detail {
template<>struct FromInnerRef<raw::BasicBlock>{
    using Output = BasicBlock;
    static const Output& from(const raw::BasicBlock & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::BasicBlock>>{
    using Output = CArc<BasicBlock>;
    static const Output& from(const CArc<raw::BasicBlock> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::BasicBlock>>{
    using Output = Pooled<BasicBlock>;
    static const Output& from(const Pooled<raw::BasicBlock> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::BasicBlock>>{
    using Output = CBoxedSlice<BasicBlock>;
    static const Output& from(const CBoxedSlice<raw::BasicBlock> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
using raw::ModuleKind;
class LC_IR_API Module : concepts::Noncopyable {
    raw::Module _inner;
public:
        [[nodiscard]] const ModuleKind& kind() const noexcept;
        [[nodiscard]] const Pooled < BasicBlock >& entry() const noexcept;
        [[nodiscard]] const CArc < ModulePools >& pools() const noexcept;
};
namespace detail {
template<>struct FromInnerRef<raw::Module>{
    using Output = Module;
    static const Output& from(const raw::Module & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::Module>>{
    using Output = CArc<Module>;
    static const Output& from(const CArc<raw::Module> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::Module>>{
    using Output = Pooled<Module>;
    static const Output& from(const Pooled<raw::Module> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::Module>>{
    using Output = CBoxedSlice<Module>;
    static const Output& from(const CBoxedSlice<raw::Module> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
class LC_IR_API CallableModule : concepts::Noncopyable {
    raw::CallableModule _inner;
public:
        [[nodiscard]] const Module& module() const noexcept;
        [[nodiscard]] const CArc < Type >& ret_type() const noexcept;
        [[nodiscard]] luisa::span<const NodeRef> args() const noexcept;
        [[nodiscard]] luisa::span<const Capture> captures() const noexcept;
        [[nodiscard]] luisa::span<const CallableModuleRef> callables() const noexcept;
        [[nodiscard]] luisa::span<const CArc < CpuCustomOp >> cpu_custom_ops() const noexcept;
        [[nodiscard]] const CArc < ModulePools >& pools() const noexcept;
};
namespace detail {
template<>struct FromInnerRef<raw::CallableModule>{
    using Output = CallableModule;
    static const Output& from(const raw::CallableModule & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::CallableModule>>{
    using Output = CArc<CallableModule>;
    static const Output& from(const CArc<raw::CallableModule> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::CallableModule>>{
    using Output = Pooled<CallableModule>;
    static const Output& from(const Pooled<raw::CallableModule> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::CallableModule>>{
    using Output = CBoxedSlice<CallableModule>;
    static const Output& from(const CBoxedSlice<raw::CallableModule> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
class LC_IR_API BufferBinding : concepts::Noncopyable {
    raw::BufferBinding _inner;
public:
        [[nodiscard]] const uint64_t& handle() const noexcept;
        [[nodiscard]] const uint64_t& offset() const noexcept;
        [[nodiscard]] const size_t& size() const noexcept;
};
namespace detail {
template<>struct FromInnerRef<raw::BufferBinding>{
    using Output = BufferBinding;
    static const Output& from(const raw::BufferBinding & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::BufferBinding>>{
    using Output = CArc<BufferBinding>;
    static const Output& from(const CArc<raw::BufferBinding> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::BufferBinding>>{
    using Output = Pooled<BufferBinding>;
    static const Output& from(const Pooled<raw::BufferBinding> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::BufferBinding>>{
    using Output = CBoxedSlice<BufferBinding>;
    static const Output& from(const CBoxedSlice<raw::BufferBinding> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
class LC_IR_API TextureBinding : concepts::Noncopyable {
    raw::TextureBinding _inner;
public:
        [[nodiscard]] const uint64_t& handle() const noexcept;
        [[nodiscard]] const uint32_t& level() const noexcept;
};
namespace detail {
template<>struct FromInnerRef<raw::TextureBinding>{
    using Output = TextureBinding;
    static const Output& from(const raw::TextureBinding & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::TextureBinding>>{
    using Output = CArc<TextureBinding>;
    static const Output& from(const CArc<raw::TextureBinding> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::TextureBinding>>{
    using Output = Pooled<TextureBinding>;
    static const Output& from(const Pooled<raw::TextureBinding> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::TextureBinding>>{
    using Output = CBoxedSlice<TextureBinding>;
    static const Output& from(const CBoxedSlice<raw::TextureBinding> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
class LC_IR_API BindlessArrayBinding : concepts::Noncopyable {
    raw::BindlessArrayBinding _inner;
public:
        [[nodiscard]] const uint64_t& handle() const noexcept;
};
namespace detail {
template<>struct FromInnerRef<raw::BindlessArrayBinding>{
    using Output = BindlessArrayBinding;
    static const Output& from(const raw::BindlessArrayBinding & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::BindlessArrayBinding>>{
    using Output = CArc<BindlessArrayBinding>;
    static const Output& from(const CArc<raw::BindlessArrayBinding> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::BindlessArrayBinding>>{
    using Output = Pooled<BindlessArrayBinding>;
    static const Output& from(const Pooled<raw::BindlessArrayBinding> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::BindlessArrayBinding>>{
    using Output = CBoxedSlice<BindlessArrayBinding>;
    static const Output& from(const CBoxedSlice<raw::BindlessArrayBinding> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
class LC_IR_API AccelBinding : concepts::Noncopyable {
    raw::AccelBinding _inner;
public:
        [[nodiscard]] const uint64_t& handle() const noexcept;
};
namespace detail {
template<>struct FromInnerRef<raw::AccelBinding>{
    using Output = AccelBinding;
    static const Output& from(const raw::AccelBinding & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::AccelBinding>>{
    using Output = CArc<AccelBinding>;
    static const Output& from(const CArc<raw::AccelBinding> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::AccelBinding>>{
    using Output = Pooled<AccelBinding>;
    static const Output& from(const Pooled<raw::AccelBinding> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::AccelBinding>>{
    using Output = CBoxedSlice<AccelBinding>;
    static const Output& from(const CBoxedSlice<raw::AccelBinding> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
class LC_IR_API Binding : concepts::Noncopyable {
    raw::Binding _inner;
    class Marker{ };
public:
    using Tag = raw::Binding::Tag;
    class LC_IR_API Buffer : Marker, concepts::Noncopyable {
        raw::Binding::Buffer_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Binding::Tag::Buffer; }
    };
    class LC_IR_API Texture : Marker, concepts::Noncopyable {
        raw::Binding::Texture_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Binding::Tag::Texture; }
    };
    class LC_IR_API BindlessArray : Marker, concepts::Noncopyable {
        raw::Binding::BindlessArray_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Binding::Tag::BindlessArray; }
    };
    class LC_IR_API Accel : Marker, concepts::Noncopyable {
        raw::Binding::Accel_Body _inner;
    public:
        static constexpr Tag tag() noexcept { return raw::Binding::Tag::Accel; }
    };
    template<class T>
    [[nodiscard]] bool isa() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        return _inner.tag == T::tag();
    }
    template<class T>
    [[nodiscard]] const T* as() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        if (!isa<T>()) return nullptr;
        if constexpr (std::is_same_v<T, Buffer>) {
            return reinterpret_cast<const Buffer*>(&_inner.buffer);
        }
        if constexpr (std::is_same_v<T, Texture>) {
            return reinterpret_cast<const Texture*>(&_inner.texture);
        }
        if constexpr (std::is_same_v<T, BindlessArray>) {
            return reinterpret_cast<const BindlessArray*>(&_inner.bindless_array);
        }
        if constexpr (std::is_same_v<T, Accel>) {
            return reinterpret_cast<const Accel*>(&_inner.accel);
        }
        return reinterpret_cast<const T*>(this);
    }
};
static_assert(sizeof(Binding) == sizeof(raw::Binding));
namespace detail {
template<>struct FromInnerRef<raw::Binding>{
    using Output = Binding;
    static const Output& from(const raw::Binding & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::Binding>>{
    using Output = CArc<Binding>;
    static const Output& from(const CArc<raw::Binding> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::Binding>>{
    using Output = Pooled<Binding>;
    static const Output& from(const Pooled<raw::Binding> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::Binding>>{
    using Output = CBoxedSlice<Binding>;
    static const Output& from(const CBoxedSlice<raw::Binding> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
class LC_IR_API Capture : concepts::Noncopyable {
    raw::Capture _inner;
public:
        [[nodiscard]] const NodeRef& node() const noexcept;
        [[nodiscard]] const Binding& binding() const noexcept;
};
namespace detail {
template<>struct FromInnerRef<raw::Capture>{
    using Output = Capture;
    static const Output& from(const raw::Capture & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::Capture>>{
    using Output = CArc<Capture>;
    static const Output& from(const CArc<raw::Capture> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::Capture>>{
    using Output = Pooled<Capture>;
    static const Output& from(const Pooled<raw::Capture> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::Capture>>{
    using Output = CBoxedSlice<Capture>;
    static const Output& from(const CBoxedSlice<raw::Capture> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
class LC_IR_API KernelModule : concepts::Noncopyable {
    raw::KernelModule _inner;
public:
        [[nodiscard]] const Module& module() const noexcept;
        [[nodiscard]] luisa::span<const Capture> captures() const noexcept;
        [[nodiscard]] luisa::span<const NodeRef> args() const noexcept;
        [[nodiscard]] luisa::span<const NodeRef> shared() const noexcept;
        [[nodiscard]] luisa::span<const CArc < CpuCustomOp >> cpu_custom_ops() const noexcept;
        [[nodiscard]] luisa::span<const CallableModuleRef> callables() const noexcept;
        [[nodiscard]] const std::array<uint32_t, 3>& block_size() const noexcept;
        [[nodiscard]] const CArc < ModulePools >& pools() const noexcept;
};
namespace detail {
template<>struct FromInnerRef<raw::KernelModule>{
    using Output = KernelModule;
    static const Output& from(const raw::KernelModule & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::KernelModule>>{
    using Output = CArc<KernelModule>;
    static const Output& from(const CArc<raw::KernelModule> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::KernelModule>>{
    using Output = Pooled<KernelModule>;
    static const Output& from(const Pooled<raw::KernelModule> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::KernelModule>>{
    using Output = CBoxedSlice<KernelModule>;
    static const Output& from(const CBoxedSlice<raw::KernelModule> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
class LC_IR_API BlockModule : concepts::Noncopyable {
    raw::BlockModule _inner;
public:
        [[nodiscard]] const Module& module() const noexcept;
};
namespace detail {
template<>struct FromInnerRef<raw::BlockModule>{
    using Output = BlockModule;
    static const Output& from(const raw::BlockModule & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::BlockModule>>{
    using Output = CArc<BlockModule>;
    static const Output& from(const CArc<raw::BlockModule> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::BlockModule>>{
    using Output = Pooled<BlockModule>;
    static const Output& from(const Pooled<raw::BlockModule> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::BlockModule>>{
    using Output = CBoxedSlice<BlockModule>;
    static const Output& from(const CBoxedSlice<raw::BlockModule> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
class LC_IR_API IrBuilder : concepts::Noncopyable {
    raw::IrBuilder _inner;
public:
        [[nodiscard]] const Pooled < BasicBlock >& bb() const noexcept;
        [[nodiscard]] const CArc < ModulePools >& pools() const noexcept;
        [[nodiscard]] const NodeRef& insert_point() const noexcept;
};
namespace detail {
template<>struct FromInnerRef<raw::IrBuilder>{
    using Output = IrBuilder;
    static const Output& from(const raw::IrBuilder & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CArc<raw::IrBuilder>>{
    using Output = CArc<IrBuilder>;
    static const Output& from(const CArc<raw::IrBuilder> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<Pooled<raw::IrBuilder>>{
    using Output = Pooled<IrBuilder>;
    static const Output& from(const Pooled<raw::IrBuilder> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
template<>struct FromInnerRef<CBoxedSlice<raw::IrBuilder>>{
    using Output = CBoxedSlice<IrBuilder>;
    static const Output& from(const CBoxedSlice<raw::IrBuilder> & _inner) noexcept { 
        return reinterpret_cast<const Output&>(_inner);
    }
};
}
}
