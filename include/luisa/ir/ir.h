#pragma once

#include <luisa/ir/fwd.h>

namespace luisa::compute::ir_v2 {
using raw::Primitive;
class LC_IR_API VectorElementType : concepts::Noncopyable {
    raw::VectorElementType _inner{};
    class Marker {};

public:
    friend class IrBuilder;
    using Tag = raw::VectorElementType::Tag;
    class LC_IR_API Scalar : Marker, concepts::Noncopyable {
        raw::VectorElementType::Scalar_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::VectorElementType::Tag::Scalar; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Vector : Marker, concepts::Noncopyable {
        raw::VectorElementType::Vector_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::VectorElementType::Tag::Vector; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
public:
    [[nodiscard]] auto tag() const noexcept { return _inner.tag; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    template<class T>
    [[nodiscard]] bool isa() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        return _inner.tag == T::tag();
    }
    template<class T>
    [[nodiscard]] const T *as() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        if (!isa<T>()) return nullptr;
        if constexpr (std::is_same_v<T, Scalar>) {
            return reinterpret_cast<const Scalar *>(&_inner.scalar);
        }
        if constexpr (std::is_same_v<T, Vector>) {
            return reinterpret_cast<const Vector *>(&_inner.vector);
        }
        return reinterpret_cast<const T *>(this);
    }
};
static_assert(sizeof(VectorElementType) == sizeof(raw::VectorElementType));

namespace detail {
template<>
struct FromInnerRef<raw::VectorElementType> {
    using Output = VectorElementType;
    static const Output &from(const raw::VectorElementType &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::VectorElementType>> {
    using Output = CArc<VectorElementType>;
    static const Output &from(const CArc<raw::VectorElementType> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::VectorElementType>> {
    using Output = Pooled<VectorElementType>;
    static const Output &from(const Pooled<raw::VectorElementType> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::VectorElementType>> {
    using Output = CBoxedSlice<VectorElementType>;
    static const Output &from(const CBoxedSlice<raw::VectorElementType> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API VectorType : concepts::Noncopyable {
    raw::VectorType _inner{};

public:
    friend class IrBuilder;
    [[nodiscard]] auto raw() noexcept { return &_inner; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    [[nodiscard]] const VectorElementType &element() const noexcept;
    [[nodiscard]] const uint32_t &length() const noexcept;
};

namespace detail {
template<>
struct FromInnerRef<raw::VectorType> {
    using Output = VectorType;
    static const Output &from(const raw::VectorType &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::VectorType>> {
    using Output = CArc<VectorType>;
    static const Output &from(const CArc<raw::VectorType> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::VectorType>> {
    using Output = Pooled<VectorType>;
    static const Output &from(const Pooled<raw::VectorType> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::VectorType>> {
    using Output = CBoxedSlice<VectorType>;
    static const Output &from(const CBoxedSlice<raw::VectorType> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API MatrixType : concepts::Noncopyable {
    raw::MatrixType _inner{};

public:
    friend class IrBuilder;
    [[nodiscard]] auto raw() noexcept { return &_inner; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    [[nodiscard]] const VectorElementType &element() const noexcept;
    [[nodiscard]] const uint32_t &dimension() const noexcept;
};

namespace detail {
template<>
struct FromInnerRef<raw::MatrixType> {
    using Output = MatrixType;
    static const Output &from(const raw::MatrixType &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::MatrixType>> {
    using Output = CArc<MatrixType>;
    static const Output &from(const CArc<raw::MatrixType> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::MatrixType>> {
    using Output = Pooled<MatrixType>;
    static const Output &from(const Pooled<raw::MatrixType> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::MatrixType>> {
    using Output = CBoxedSlice<MatrixType>;
    static const Output &from(const CBoxedSlice<raw::MatrixType> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API StructType : concepts::Noncopyable {
    raw::StructType _inner{};

public:
    friend class IrBuilder;
    [[nodiscard]] auto raw() noexcept { return &_inner; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    [[nodiscard]] luisa::span<const CArc<Type>> fields() const noexcept;
    [[nodiscard]] const size_t &alignment() const noexcept;
    [[nodiscard]] const size_t &size() const noexcept;
};

namespace detail {
template<>
struct FromInnerRef<raw::StructType> {
    using Output = StructType;
    static const Output &from(const raw::StructType &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::StructType>> {
    using Output = CArc<StructType>;
    static const Output &from(const CArc<raw::StructType> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::StructType>> {
    using Output = Pooled<StructType>;
    static const Output &from(const Pooled<raw::StructType> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::StructType>> {
    using Output = CBoxedSlice<StructType>;
    static const Output &from(const CBoxedSlice<raw::StructType> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API ArrayType : concepts::Noncopyable {
    raw::ArrayType _inner{};

public:
    friend class IrBuilder;
    [[nodiscard]] auto raw() noexcept { return &_inner; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    [[nodiscard]] const CArc<Type> &element() const noexcept;
    [[nodiscard]] const size_t &length() const noexcept;
};

namespace detail {
template<>
struct FromInnerRef<raw::ArrayType> {
    using Output = ArrayType;
    static const Output &from(const raw::ArrayType &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::ArrayType>> {
    using Output = CArc<ArrayType>;
    static const Output &from(const CArc<raw::ArrayType> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::ArrayType>> {
    using Output = Pooled<ArrayType>;
    static const Output &from(const Pooled<raw::ArrayType> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::ArrayType>> {
    using Output = CBoxedSlice<ArrayType>;
    static const Output &from(const CBoxedSlice<raw::ArrayType> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API Type : concepts::Noncopyable {
    raw::Type _inner{};
    class Marker {};

public:
    friend class IrBuilder;
    using Tag = raw::Type::Tag;
    class LC_IR_API Void : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Type::Tag::Void; }
    };
    explicit Type(Type::Void _) noexcept { _inner.tag = Void::tag(); }
    class LC_IR_API UserData : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Type::Tag::UserData; }
    };
    explicit Type(Type::UserData _) noexcept { _inner.tag = UserData::tag(); }
    class LC_IR_API Primitive : Marker, concepts::Noncopyable {
        raw::Type::Primitive_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Type::Tag::Primitive; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Vector : Marker, concepts::Noncopyable {
        raw::Type::Vector_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Type::Tag::Vector; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Matrix : Marker, concepts::Noncopyable {
        raw::Type::Matrix_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Type::Tag::Matrix; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Struct : Marker, concepts::Noncopyable {
        raw::Type::Struct_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Type::Tag::Struct; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Array : Marker, concepts::Noncopyable {
        raw::Type::Array_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Type::Tag::Array; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Opaque : Marker, concepts::Noncopyable {
        raw::Type::Opaque_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Type::Tag::Opaque; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
public:
    [[nodiscard]] auto tag() const noexcept { return _inner.tag; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    template<class T>
    [[nodiscard]] bool isa() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        return _inner.tag == T::tag();
    }
    template<class T>
    [[nodiscard]] const T *as() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        if (!isa<T>()) return nullptr;
        if constexpr (std::is_same_v<T, Primitive>) {
            return reinterpret_cast<const Primitive *>(&_inner.primitive);
        }
        if constexpr (std::is_same_v<T, Vector>) {
            return reinterpret_cast<const Vector *>(&_inner.vector);
        }
        if constexpr (std::is_same_v<T, Matrix>) {
            return reinterpret_cast<const Matrix *>(&_inner.matrix);
        }
        if constexpr (std::is_same_v<T, Struct>) {
            return reinterpret_cast<const Struct *>(&_inner.struct_);
        }
        if constexpr (std::is_same_v<T, Array>) {
            return reinterpret_cast<const Array *>(&_inner.array);
        }
        if constexpr (std::is_same_v<T, Opaque>) {
            return reinterpret_cast<const Opaque *>(&_inner.opaque);
        }
        return reinterpret_cast<const T *>(this);
    }
};
static_assert(sizeof(Type) == sizeof(raw::Type));

namespace detail {
template<>
struct FromInnerRef<raw::Type> {
    using Output = Type;
    static const Output &from(const raw::Type &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::Type>> {
    using Output = CArc<Type>;
    static const Output &from(const CArc<raw::Type> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::Type>> {
    using Output = Pooled<Type>;
    static const Output &from(const Pooled<raw::Type> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::Type>> {
    using Output = CBoxedSlice<Type>;
    static const Output &from(const CBoxedSlice<raw::Type> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API Node : concepts::Noncopyable {
    raw::Node _inner{};

public:
    friend class IrBuilder;
    [[nodiscard]] auto raw() noexcept { return &_inner; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    [[nodiscard]] const CArc<Type> &type_() const noexcept;
    [[nodiscard]] const NodeRef &next() const noexcept;
    [[nodiscard]] const NodeRef &prev() const noexcept;
    [[nodiscard]] const CArc<Instruction> &instruction() const noexcept;
};

namespace detail {
template<>
struct FromInnerRef<raw::Node> {
    using Output = Node;
    static const Output &from(const raw::Node &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::Node>> {
    using Output = CArc<Node>;
    static const Output &from(const CArc<raw::Node> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::Node>> {
    using Output = Pooled<Node>;
    static const Output &from(const Pooled<raw::Node> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::Node>> {
    using Output = CBoxedSlice<Node>;
    static const Output &from(const CBoxedSlice<raw::Node> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API Func : concepts::Noncopyable {
    raw::Func _inner{};
    class Marker {};

public:
    friend class IrBuilder;
    using Tag = raw::Func::Tag;
    class LC_IR_API ZeroInitializer : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::ZeroInitializer; }
    };
    explicit Func(Func::ZeroInitializer _) noexcept { _inner.tag = ZeroInitializer::tag(); }
    class LC_IR_API Assume : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Assume; }
    };
    explicit Func(Func::Assume _) noexcept { _inner.tag = Assume::tag(); }
    class LC_IR_API Unreachable : Marker, concepts::Noncopyable {
        raw::Func::Unreachable_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Unreachable; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Assert : Marker, concepts::Noncopyable {
        raw::Func::Assert_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Assert; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API ThreadId : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::ThreadId; }
    };
    explicit Func(Func::ThreadId _) noexcept { _inner.tag = ThreadId::tag(); }
    class LC_IR_API BlockId : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BlockId; }
    };
    explicit Func(Func::BlockId _) noexcept { _inner.tag = BlockId::tag(); }
    class LC_IR_API WarpSize : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::WarpSize; }
    };
    explicit Func(Func::WarpSize _) noexcept { _inner.tag = WarpSize::tag(); }
    class LC_IR_API WarpLaneId : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::WarpLaneId; }
    };
    explicit Func(Func::WarpLaneId _) noexcept { _inner.tag = WarpLaneId::tag(); }
    class LC_IR_API DispatchId : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::DispatchId; }
    };
    explicit Func(Func::DispatchId _) noexcept { _inner.tag = DispatchId::tag(); }
    class LC_IR_API DispatchSize : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::DispatchSize; }
    };
    explicit Func(Func::DispatchSize _) noexcept { _inner.tag = DispatchSize::tag(); }
    class LC_IR_API PropagateGrad : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::PropagateGrad; }
    };
    explicit Func(Func::PropagateGrad _) noexcept { _inner.tag = PropagateGrad::tag(); }
    class LC_IR_API OutputGrad : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::OutputGrad; }
    };
    explicit Func(Func::OutputGrad _) noexcept { _inner.tag = OutputGrad::tag(); }
    class LC_IR_API RequiresGradient : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::RequiresGradient; }
    };
    explicit Func(Func::RequiresGradient _) noexcept { _inner.tag = RequiresGradient::tag(); }
    class LC_IR_API Backward : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Backward; }
    };
    explicit Func(Func::Backward _) noexcept { _inner.tag = Backward::tag(); }
    class LC_IR_API Gradient : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Gradient; }
    };
    explicit Func(Func::Gradient _) noexcept { _inner.tag = Gradient::tag(); }
    class LC_IR_API GradientMarker : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::GradientMarker; }
    };
    explicit Func(Func::GradientMarker _) noexcept { _inner.tag = GradientMarker::tag(); }
    class LC_IR_API AccGrad : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::AccGrad; }
    };
    explicit Func(Func::AccGrad _) noexcept { _inner.tag = AccGrad::tag(); }
    class LC_IR_API Detach : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Detach; }
    };
    explicit Func(Func::Detach _) noexcept { _inner.tag = Detach::tag(); }
    class LC_IR_API RayTracingInstanceTransform : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::RayTracingInstanceTransform; }
    };
    explicit Func(Func::RayTracingInstanceTransform _) noexcept { _inner.tag = RayTracingInstanceTransform::tag(); }
    class LC_IR_API RayTracingInstanceUserId : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::RayTracingInstanceUserId; }
    };
    explicit Func(Func::RayTracingInstanceUserId _) noexcept { _inner.tag = RayTracingInstanceUserId::tag(); }
    class LC_IR_API RayTracingSetInstanceTransform : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::RayTracingSetInstanceTransform; }
    };
    explicit Func(Func::RayTracingSetInstanceTransform _) noexcept { _inner.tag = RayTracingSetInstanceTransform::tag(); }
    class LC_IR_API RayTracingSetInstanceOpacity : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::RayTracingSetInstanceOpacity; }
    };
    explicit Func(Func::RayTracingSetInstanceOpacity _) noexcept { _inner.tag = RayTracingSetInstanceOpacity::tag(); }
    class LC_IR_API RayTracingSetInstanceVisibility : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::RayTracingSetInstanceVisibility; }
    };
    explicit Func(Func::RayTracingSetInstanceVisibility _) noexcept { _inner.tag = RayTracingSetInstanceVisibility::tag(); }
    class LC_IR_API RayTracingSetInstanceUserId : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::RayTracingSetInstanceUserId; }
    };
    explicit Func(Func::RayTracingSetInstanceUserId _) noexcept { _inner.tag = RayTracingSetInstanceUserId::tag(); }
    class LC_IR_API RayTracingTraceClosest : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::RayTracingTraceClosest; }
    };
    explicit Func(Func::RayTracingTraceClosest _) noexcept { _inner.tag = RayTracingTraceClosest::tag(); }
    class LC_IR_API RayTracingTraceAny : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::RayTracingTraceAny; }
    };
    explicit Func(Func::RayTracingTraceAny _) noexcept { _inner.tag = RayTracingTraceAny::tag(); }
    class LC_IR_API RayTracingQueryAll : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::RayTracingQueryAll; }
    };
    explicit Func(Func::RayTracingQueryAll _) noexcept { _inner.tag = RayTracingQueryAll::tag(); }
    class LC_IR_API RayTracingQueryAny : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::RayTracingQueryAny; }
    };
    explicit Func(Func::RayTracingQueryAny _) noexcept { _inner.tag = RayTracingQueryAny::tag(); }
    class LC_IR_API RayQueryWorldSpaceRay : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::RayQueryWorldSpaceRay; }
    };
    explicit Func(Func::RayQueryWorldSpaceRay _) noexcept { _inner.tag = RayQueryWorldSpaceRay::tag(); }
    class LC_IR_API RayQueryProceduralCandidateHit : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::RayQueryProceduralCandidateHit; }
    };
    explicit Func(Func::RayQueryProceduralCandidateHit _) noexcept { _inner.tag = RayQueryProceduralCandidateHit::tag(); }
    class LC_IR_API RayQueryTriangleCandidateHit : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::RayQueryTriangleCandidateHit; }
    };
    explicit Func(Func::RayQueryTriangleCandidateHit _) noexcept { _inner.tag = RayQueryTriangleCandidateHit::tag(); }
    class LC_IR_API RayQueryCommittedHit : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::RayQueryCommittedHit; }
    };
    explicit Func(Func::RayQueryCommittedHit _) noexcept { _inner.tag = RayQueryCommittedHit::tag(); }
    class LC_IR_API RayQueryCommitTriangle : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::RayQueryCommitTriangle; }
    };
    explicit Func(Func::RayQueryCommitTriangle _) noexcept { _inner.tag = RayQueryCommitTriangle::tag(); }
    class LC_IR_API RayQueryCommitProcedural : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::RayQueryCommitProcedural; }
    };
    explicit Func(Func::RayQueryCommitProcedural _) noexcept { _inner.tag = RayQueryCommitProcedural::tag(); }
    class LC_IR_API RayQueryTerminate : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::RayQueryTerminate; }
    };
    explicit Func(Func::RayQueryTerminate _) noexcept { _inner.tag = RayQueryTerminate::tag(); }
    class LC_IR_API RasterDiscard : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::RasterDiscard; }
    };
    explicit Func(Func::RasterDiscard _) noexcept { _inner.tag = RasterDiscard::tag(); }
    class LC_IR_API IndirectDispatchSetCount : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::IndirectDispatchSetCount; }
    };
    explicit Func(Func::IndirectDispatchSetCount _) noexcept { _inner.tag = IndirectDispatchSetCount::tag(); }
    class LC_IR_API IndirectDispatchSetKernel : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::IndirectDispatchSetKernel; }
    };
    explicit Func(Func::IndirectDispatchSetKernel _) noexcept { _inner.tag = IndirectDispatchSetKernel::tag(); }
    class LC_IR_API Load : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Load; }
    };
    explicit Func(Func::Load _) noexcept { _inner.tag = Load::tag(); }
    class LC_IR_API Cast : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Cast; }
    };
    explicit Func(Func::Cast _) noexcept { _inner.tag = Cast::tag(); }
    class LC_IR_API Bitcast : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Bitcast; }
    };
    explicit Func(Func::Bitcast _) noexcept { _inner.tag = Bitcast::tag(); }
    class LC_IR_API Pack : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Pack; }
    };
    explicit Func(Func::Pack _) noexcept { _inner.tag = Pack::tag(); }
    class LC_IR_API Unpack : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Unpack; }
    };
    explicit Func(Func::Unpack _) noexcept { _inner.tag = Unpack::tag(); }
    class LC_IR_API Add : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Add; }
    };
    explicit Func(Func::Add _) noexcept { _inner.tag = Add::tag(); }
    class LC_IR_API Sub : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Sub; }
    };
    explicit Func(Func::Sub _) noexcept { _inner.tag = Sub::tag(); }
    class LC_IR_API Mul : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Mul; }
    };
    explicit Func(Func::Mul _) noexcept { _inner.tag = Mul::tag(); }
    class LC_IR_API Div : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Div; }
    };
    explicit Func(Func::Div _) noexcept { _inner.tag = Div::tag(); }
    class LC_IR_API Rem : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Rem; }
    };
    explicit Func(Func::Rem _) noexcept { _inner.tag = Rem::tag(); }
    class LC_IR_API BitAnd : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BitAnd; }
    };
    explicit Func(Func::BitAnd _) noexcept { _inner.tag = BitAnd::tag(); }
    class LC_IR_API BitOr : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BitOr; }
    };
    explicit Func(Func::BitOr _) noexcept { _inner.tag = BitOr::tag(); }
    class LC_IR_API BitXor : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BitXor; }
    };
    explicit Func(Func::BitXor _) noexcept { _inner.tag = BitXor::tag(); }
    class LC_IR_API Shl : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Shl; }
    };
    explicit Func(Func::Shl _) noexcept { _inner.tag = Shl::tag(); }
    class LC_IR_API Shr : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Shr; }
    };
    explicit Func(Func::Shr _) noexcept { _inner.tag = Shr::tag(); }
    class LC_IR_API RotRight : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::RotRight; }
    };
    explicit Func(Func::RotRight _) noexcept { _inner.tag = RotRight::tag(); }
    class LC_IR_API RotLeft : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::RotLeft; }
    };
    explicit Func(Func::RotLeft _) noexcept { _inner.tag = RotLeft::tag(); }
    class LC_IR_API Eq : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Eq; }
    };
    explicit Func(Func::Eq _) noexcept { _inner.tag = Eq::tag(); }
    class LC_IR_API Ne : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Ne; }
    };
    explicit Func(Func::Ne _) noexcept { _inner.tag = Ne::tag(); }
    class LC_IR_API Lt : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Lt; }
    };
    explicit Func(Func::Lt _) noexcept { _inner.tag = Lt::tag(); }
    class LC_IR_API Le : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Le; }
    };
    explicit Func(Func::Le _) noexcept { _inner.tag = Le::tag(); }
    class LC_IR_API Gt : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Gt; }
    };
    explicit Func(Func::Gt _) noexcept { _inner.tag = Gt::tag(); }
    class LC_IR_API Ge : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Ge; }
    };
    explicit Func(Func::Ge _) noexcept { _inner.tag = Ge::tag(); }
    class LC_IR_API MatCompMul : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::MatCompMul; }
    };
    explicit Func(Func::MatCompMul _) noexcept { _inner.tag = MatCompMul::tag(); }
    class LC_IR_API Neg : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Neg; }
    };
    explicit Func(Func::Neg _) noexcept { _inner.tag = Neg::tag(); }
    class LC_IR_API Not : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Not; }
    };
    explicit Func(Func::Not _) noexcept { _inner.tag = Not::tag(); }
    class LC_IR_API BitNot : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BitNot; }
    };
    explicit Func(Func::BitNot _) noexcept { _inner.tag = BitNot::tag(); }
    class LC_IR_API All : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::All; }
    };
    explicit Func(Func::All _) noexcept { _inner.tag = All::tag(); }
    class LC_IR_API Any : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Any; }
    };
    explicit Func(Func::Any _) noexcept { _inner.tag = Any::tag(); }
    class LC_IR_API Select : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Select; }
    };
    explicit Func(Func::Select _) noexcept { _inner.tag = Select::tag(); }
    class LC_IR_API Clamp : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Clamp; }
    };
    explicit Func(Func::Clamp _) noexcept { _inner.tag = Clamp::tag(); }
    class LC_IR_API Lerp : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Lerp; }
    };
    explicit Func(Func::Lerp _) noexcept { _inner.tag = Lerp::tag(); }
    class LC_IR_API Step : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Step; }
    };
    explicit Func(Func::Step _) noexcept { _inner.tag = Step::tag(); }
    class LC_IR_API SmoothStep : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::SmoothStep; }
    };
    explicit Func(Func::SmoothStep _) noexcept { _inner.tag = SmoothStep::tag(); }
    class LC_IR_API Saturate : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Saturate; }
    };
    explicit Func(Func::Saturate _) noexcept { _inner.tag = Saturate::tag(); }
    class LC_IR_API Abs : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Abs; }
    };
    explicit Func(Func::Abs _) noexcept { _inner.tag = Abs::tag(); }
    class LC_IR_API Min : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Min; }
    };
    explicit Func(Func::Min _) noexcept { _inner.tag = Min::tag(); }
    class LC_IR_API Max : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Max; }
    };
    explicit Func(Func::Max _) noexcept { _inner.tag = Max::tag(); }
    class LC_IR_API ReduceSum : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::ReduceSum; }
    };
    explicit Func(Func::ReduceSum _) noexcept { _inner.tag = ReduceSum::tag(); }
    class LC_IR_API ReduceProd : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::ReduceProd; }
    };
    explicit Func(Func::ReduceProd _) noexcept { _inner.tag = ReduceProd::tag(); }
    class LC_IR_API ReduceMin : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::ReduceMin; }
    };
    explicit Func(Func::ReduceMin _) noexcept { _inner.tag = ReduceMin::tag(); }
    class LC_IR_API ReduceMax : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::ReduceMax; }
    };
    explicit Func(Func::ReduceMax _) noexcept { _inner.tag = ReduceMax::tag(); }
    class LC_IR_API Clz : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Clz; }
    };
    explicit Func(Func::Clz _) noexcept { _inner.tag = Clz::tag(); }
    class LC_IR_API Ctz : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Ctz; }
    };
    explicit Func(Func::Ctz _) noexcept { _inner.tag = Ctz::tag(); }
    class LC_IR_API PopCount : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::PopCount; }
    };
    explicit Func(Func::PopCount _) noexcept { _inner.tag = PopCount::tag(); }
    class LC_IR_API Reverse : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Reverse; }
    };
    explicit Func(Func::Reverse _) noexcept { _inner.tag = Reverse::tag(); }
    class LC_IR_API IsInf : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::IsInf; }
    };
    explicit Func(Func::IsInf _) noexcept { _inner.tag = IsInf::tag(); }
    class LC_IR_API IsNan : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::IsNan; }
    };
    explicit Func(Func::IsNan _) noexcept { _inner.tag = IsNan::tag(); }
    class LC_IR_API Acos : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Acos; }
    };
    explicit Func(Func::Acos _) noexcept { _inner.tag = Acos::tag(); }
    class LC_IR_API Acosh : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Acosh; }
    };
    explicit Func(Func::Acosh _) noexcept { _inner.tag = Acosh::tag(); }
    class LC_IR_API Asin : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Asin; }
    };
    explicit Func(Func::Asin _) noexcept { _inner.tag = Asin::tag(); }
    class LC_IR_API Asinh : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Asinh; }
    };
    explicit Func(Func::Asinh _) noexcept { _inner.tag = Asinh::tag(); }
    class LC_IR_API Atan : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Atan; }
    };
    explicit Func(Func::Atan _) noexcept { _inner.tag = Atan::tag(); }
    class LC_IR_API Atan2 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Atan2; }
    };
    explicit Func(Func::Atan2 _) noexcept { _inner.tag = Atan2::tag(); }
    class LC_IR_API Atanh : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Atanh; }
    };
    explicit Func(Func::Atanh _) noexcept { _inner.tag = Atanh::tag(); }
    class LC_IR_API Cos : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Cos; }
    };
    explicit Func(Func::Cos _) noexcept { _inner.tag = Cos::tag(); }
    class LC_IR_API Cosh : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Cosh; }
    };
    explicit Func(Func::Cosh _) noexcept { _inner.tag = Cosh::tag(); }
    class LC_IR_API Sin : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Sin; }
    };
    explicit Func(Func::Sin _) noexcept { _inner.tag = Sin::tag(); }
    class LC_IR_API Sinh : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Sinh; }
    };
    explicit Func(Func::Sinh _) noexcept { _inner.tag = Sinh::tag(); }
    class LC_IR_API Tan : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Tan; }
    };
    explicit Func(Func::Tan _) noexcept { _inner.tag = Tan::tag(); }
    class LC_IR_API Tanh : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Tanh; }
    };
    explicit Func(Func::Tanh _) noexcept { _inner.tag = Tanh::tag(); }
    class LC_IR_API Exp : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Exp; }
    };
    explicit Func(Func::Exp _) noexcept { _inner.tag = Exp::tag(); }
    class LC_IR_API Exp2 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Exp2; }
    };
    explicit Func(Func::Exp2 _) noexcept { _inner.tag = Exp2::tag(); }
    class LC_IR_API Exp10 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Exp10; }
    };
    explicit Func(Func::Exp10 _) noexcept { _inner.tag = Exp10::tag(); }
    class LC_IR_API Log : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Log; }
    };
    explicit Func(Func::Log _) noexcept { _inner.tag = Log::tag(); }
    class LC_IR_API Log2 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Log2; }
    };
    explicit Func(Func::Log2 _) noexcept { _inner.tag = Log2::tag(); }
    class LC_IR_API Log10 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Log10; }
    };
    explicit Func(Func::Log10 _) noexcept { _inner.tag = Log10::tag(); }
    class LC_IR_API Powi : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Powi; }
    };
    explicit Func(Func::Powi _) noexcept { _inner.tag = Powi::tag(); }
    class LC_IR_API Powf : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Powf; }
    };
    explicit Func(Func::Powf _) noexcept { _inner.tag = Powf::tag(); }
    class LC_IR_API Sqrt : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Sqrt; }
    };
    explicit Func(Func::Sqrt _) noexcept { _inner.tag = Sqrt::tag(); }
    class LC_IR_API Rsqrt : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Rsqrt; }
    };
    explicit Func(Func::Rsqrt _) noexcept { _inner.tag = Rsqrt::tag(); }
    class LC_IR_API Ceil : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Ceil; }
    };
    explicit Func(Func::Ceil _) noexcept { _inner.tag = Ceil::tag(); }
    class LC_IR_API Floor : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Floor; }
    };
    explicit Func(Func::Floor _) noexcept { _inner.tag = Floor::tag(); }
    class LC_IR_API Fract : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Fract; }
    };
    explicit Func(Func::Fract _) noexcept { _inner.tag = Fract::tag(); }
    class LC_IR_API Trunc : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Trunc; }
    };
    explicit Func(Func::Trunc _) noexcept { _inner.tag = Trunc::tag(); }
    class LC_IR_API Round : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Round; }
    };
    explicit Func(Func::Round _) noexcept { _inner.tag = Round::tag(); }
    class LC_IR_API Fma : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Fma; }
    };
    explicit Func(Func::Fma _) noexcept { _inner.tag = Fma::tag(); }
    class LC_IR_API Copysign : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Copysign; }
    };
    explicit Func(Func::Copysign _) noexcept { _inner.tag = Copysign::tag(); }
    class LC_IR_API Cross : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Cross; }
    };
    explicit Func(Func::Cross _) noexcept { _inner.tag = Cross::tag(); }
    class LC_IR_API Dot : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Dot; }
    };
    explicit Func(Func::Dot _) noexcept { _inner.tag = Dot::tag(); }
    class LC_IR_API OuterProduct : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::OuterProduct; }
    };
    explicit Func(Func::OuterProduct _) noexcept { _inner.tag = OuterProduct::tag(); }
    class LC_IR_API Length : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Length; }
    };
    explicit Func(Func::Length _) noexcept { _inner.tag = Length::tag(); }
    class LC_IR_API LengthSquared : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::LengthSquared; }
    };
    explicit Func(Func::LengthSquared _) noexcept { _inner.tag = LengthSquared::tag(); }
    class LC_IR_API Normalize : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Normalize; }
    };
    explicit Func(Func::Normalize _) noexcept { _inner.tag = Normalize::tag(); }
    class LC_IR_API Faceforward : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Faceforward; }
    };
    explicit Func(Func::Faceforward _) noexcept { _inner.tag = Faceforward::tag(); }
    class LC_IR_API Reflect : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Reflect; }
    };
    explicit Func(Func::Reflect _) noexcept { _inner.tag = Reflect::tag(); }
    class LC_IR_API Determinant : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Determinant; }
    };
    explicit Func(Func::Determinant _) noexcept { _inner.tag = Determinant::tag(); }
    class LC_IR_API Transpose : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Transpose; }
    };
    explicit Func(Func::Transpose _) noexcept { _inner.tag = Transpose::tag(); }
    class LC_IR_API Inverse : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Inverse; }
    };
    explicit Func(Func::Inverse _) noexcept { _inner.tag = Inverse::tag(); }
    class LC_IR_API WarpIsFirstActiveLane : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::WarpIsFirstActiveLane; }
    };
    explicit Func(Func::WarpIsFirstActiveLane _) noexcept { _inner.tag = WarpIsFirstActiveLane::tag(); }
    class LC_IR_API WarpFirstActiveLane : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::WarpFirstActiveLane; }
    };
    explicit Func(Func::WarpFirstActiveLane _) noexcept { _inner.tag = WarpFirstActiveLane::tag(); }
    class LC_IR_API WarpActiveAllEqual : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::WarpActiveAllEqual; }
    };
    explicit Func(Func::WarpActiveAllEqual _) noexcept { _inner.tag = WarpActiveAllEqual::tag(); }
    class LC_IR_API WarpActiveBitAnd : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::WarpActiveBitAnd; }
    };
    explicit Func(Func::WarpActiveBitAnd _) noexcept { _inner.tag = WarpActiveBitAnd::tag(); }
    class LC_IR_API WarpActiveBitOr : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::WarpActiveBitOr; }
    };
    explicit Func(Func::WarpActiveBitOr _) noexcept { _inner.tag = WarpActiveBitOr::tag(); }
    class LC_IR_API WarpActiveBitXor : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::WarpActiveBitXor; }
    };
    explicit Func(Func::WarpActiveBitXor _) noexcept { _inner.tag = WarpActiveBitXor::tag(); }
    class LC_IR_API WarpActiveCountBits : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::WarpActiveCountBits; }
    };
    explicit Func(Func::WarpActiveCountBits _) noexcept { _inner.tag = WarpActiveCountBits::tag(); }
    class LC_IR_API WarpActiveMax : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::WarpActiveMax; }
    };
    explicit Func(Func::WarpActiveMax _) noexcept { _inner.tag = WarpActiveMax::tag(); }
    class LC_IR_API WarpActiveMin : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::WarpActiveMin; }
    };
    explicit Func(Func::WarpActiveMin _) noexcept { _inner.tag = WarpActiveMin::tag(); }
    class LC_IR_API WarpActiveProduct : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::WarpActiveProduct; }
    };
    explicit Func(Func::WarpActiveProduct _) noexcept { _inner.tag = WarpActiveProduct::tag(); }
    class LC_IR_API WarpActiveSum : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::WarpActiveSum; }
    };
    explicit Func(Func::WarpActiveSum _) noexcept { _inner.tag = WarpActiveSum::tag(); }
    class LC_IR_API WarpActiveAll : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::WarpActiveAll; }
    };
    explicit Func(Func::WarpActiveAll _) noexcept { _inner.tag = WarpActiveAll::tag(); }
    class LC_IR_API WarpActiveAny : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::WarpActiveAny; }
    };
    explicit Func(Func::WarpActiveAny _) noexcept { _inner.tag = WarpActiveAny::tag(); }
    class LC_IR_API WarpActiveBitMask : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::WarpActiveBitMask; }
    };
    explicit Func(Func::WarpActiveBitMask _) noexcept { _inner.tag = WarpActiveBitMask::tag(); }
    class LC_IR_API WarpPrefixCountBits : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::WarpPrefixCountBits; }
    };
    explicit Func(Func::WarpPrefixCountBits _) noexcept { _inner.tag = WarpPrefixCountBits::tag(); }
    class LC_IR_API WarpPrefixSum : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::WarpPrefixSum; }
    };
    explicit Func(Func::WarpPrefixSum _) noexcept { _inner.tag = WarpPrefixSum::tag(); }
    class LC_IR_API WarpPrefixProduct : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::WarpPrefixProduct; }
    };
    explicit Func(Func::WarpPrefixProduct _) noexcept { _inner.tag = WarpPrefixProduct::tag(); }
    class LC_IR_API WarpReadLaneAt : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::WarpReadLaneAt; }
    };
    explicit Func(Func::WarpReadLaneAt _) noexcept { _inner.tag = WarpReadLaneAt::tag(); }
    class LC_IR_API WarpReadFirstLane : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::WarpReadFirstLane; }
    };
    explicit Func(Func::WarpReadFirstLane _) noexcept { _inner.tag = WarpReadFirstLane::tag(); }
    class LC_IR_API SynchronizeBlock : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::SynchronizeBlock; }
    };
    explicit Func(Func::SynchronizeBlock _) noexcept { _inner.tag = SynchronizeBlock::tag(); }
    class LC_IR_API AtomicRef : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::AtomicRef; }
    };
    explicit Func(Func::AtomicRef _) noexcept { _inner.tag = AtomicRef::tag(); }
    class LC_IR_API AtomicExchange : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::AtomicExchange; }
    };
    explicit Func(Func::AtomicExchange _) noexcept { _inner.tag = AtomicExchange::tag(); }
    class LC_IR_API AtomicCompareExchange : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::AtomicCompareExchange; }
    };
    explicit Func(Func::AtomicCompareExchange _) noexcept { _inner.tag = AtomicCompareExchange::tag(); }
    class LC_IR_API AtomicFetchAdd : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::AtomicFetchAdd; }
    };
    explicit Func(Func::AtomicFetchAdd _) noexcept { _inner.tag = AtomicFetchAdd::tag(); }
    class LC_IR_API AtomicFetchSub : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::AtomicFetchSub; }
    };
    explicit Func(Func::AtomicFetchSub _) noexcept { _inner.tag = AtomicFetchSub::tag(); }
    class LC_IR_API AtomicFetchAnd : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::AtomicFetchAnd; }
    };
    explicit Func(Func::AtomicFetchAnd _) noexcept { _inner.tag = AtomicFetchAnd::tag(); }
    class LC_IR_API AtomicFetchOr : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::AtomicFetchOr; }
    };
    explicit Func(Func::AtomicFetchOr _) noexcept { _inner.tag = AtomicFetchOr::tag(); }
    class LC_IR_API AtomicFetchXor : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::AtomicFetchXor; }
    };
    explicit Func(Func::AtomicFetchXor _) noexcept { _inner.tag = AtomicFetchXor::tag(); }
    class LC_IR_API AtomicFetchMin : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::AtomicFetchMin; }
    };
    explicit Func(Func::AtomicFetchMin _) noexcept { _inner.tag = AtomicFetchMin::tag(); }
    class LC_IR_API AtomicFetchMax : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::AtomicFetchMax; }
    };
    explicit Func(Func::AtomicFetchMax _) noexcept { _inner.tag = AtomicFetchMax::tag(); }
    class LC_IR_API BufferRead : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BufferRead; }
    };
    explicit Func(Func::BufferRead _) noexcept { _inner.tag = BufferRead::tag(); }
    class LC_IR_API BufferWrite : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BufferWrite; }
    };
    explicit Func(Func::BufferWrite _) noexcept { _inner.tag = BufferWrite::tag(); }
    class LC_IR_API BufferSize : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BufferSize; }
    };
    explicit Func(Func::BufferSize _) noexcept { _inner.tag = BufferSize::tag(); }
    class LC_IR_API ByteBufferRead : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::ByteBufferRead; }
    };
    explicit Func(Func::ByteBufferRead _) noexcept { _inner.tag = ByteBufferRead::tag(); }
    class LC_IR_API ByteBufferWrite : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::ByteBufferWrite; }
    };
    explicit Func(Func::ByteBufferWrite _) noexcept { _inner.tag = ByteBufferWrite::tag(); }
    class LC_IR_API ByteBufferSize : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::ByteBufferSize; }
    };
    explicit Func(Func::ByteBufferSize _) noexcept { _inner.tag = ByteBufferSize::tag(); }
    class LC_IR_API Texture2dRead : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Texture2dRead; }
    };
    explicit Func(Func::Texture2dRead _) noexcept { _inner.tag = Texture2dRead::tag(); }
    class LC_IR_API Texture2dWrite : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Texture2dWrite; }
    };
    explicit Func(Func::Texture2dWrite _) noexcept { _inner.tag = Texture2dWrite::tag(); }
    class LC_IR_API Texture2dSize : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Texture2dSize; }
    };
    explicit Func(Func::Texture2dSize _) noexcept { _inner.tag = Texture2dSize::tag(); }
    class LC_IR_API Texture3dRead : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Texture3dRead; }
    };
    explicit Func(Func::Texture3dRead _) noexcept { _inner.tag = Texture3dRead::tag(); }
    class LC_IR_API Texture3dWrite : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Texture3dWrite; }
    };
    explicit Func(Func::Texture3dWrite _) noexcept { _inner.tag = Texture3dWrite::tag(); }
    class LC_IR_API Texture3dSize : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Texture3dSize; }
    };
    explicit Func(Func::Texture3dSize _) noexcept { _inner.tag = Texture3dSize::tag(); }
    class LC_IR_API BindlessTexture2dSample : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BindlessTexture2dSample; }
    };
    explicit Func(Func::BindlessTexture2dSample _) noexcept { _inner.tag = BindlessTexture2dSample::tag(); }
    class LC_IR_API BindlessTexture2dSampleLevel : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BindlessTexture2dSampleLevel; }
    };
    explicit Func(Func::BindlessTexture2dSampleLevel _) noexcept { _inner.tag = BindlessTexture2dSampleLevel::tag(); }
    class LC_IR_API BindlessTexture2dSampleGrad : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BindlessTexture2dSampleGrad; }
    };
    explicit Func(Func::BindlessTexture2dSampleGrad _) noexcept { _inner.tag = BindlessTexture2dSampleGrad::tag(); }
    class LC_IR_API BindlessTexture2dSampleGradLevel : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BindlessTexture2dSampleGradLevel; }
    };
    explicit Func(Func::BindlessTexture2dSampleGradLevel _) noexcept { _inner.tag = BindlessTexture2dSampleGradLevel::tag(); }
    class LC_IR_API BindlessTexture3dSample : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BindlessTexture3dSample; }
    };
    explicit Func(Func::BindlessTexture3dSample _) noexcept { _inner.tag = BindlessTexture3dSample::tag(); }
    class LC_IR_API BindlessTexture3dSampleLevel : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BindlessTexture3dSampleLevel; }
    };
    explicit Func(Func::BindlessTexture3dSampleLevel _) noexcept { _inner.tag = BindlessTexture3dSampleLevel::tag(); }
    class LC_IR_API BindlessTexture3dSampleGrad : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BindlessTexture3dSampleGrad; }
    };
    explicit Func(Func::BindlessTexture3dSampleGrad _) noexcept { _inner.tag = BindlessTexture3dSampleGrad::tag(); }
    class LC_IR_API BindlessTexture3dSampleGradLevel : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BindlessTexture3dSampleGradLevel; }
    };
    explicit Func(Func::BindlessTexture3dSampleGradLevel _) noexcept { _inner.tag = BindlessTexture3dSampleGradLevel::tag(); }
    class LC_IR_API BindlessTexture2dRead : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BindlessTexture2dRead; }
    };
    explicit Func(Func::BindlessTexture2dRead _) noexcept { _inner.tag = BindlessTexture2dRead::tag(); }
    class LC_IR_API BindlessTexture3dRead : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BindlessTexture3dRead; }
    };
    explicit Func(Func::BindlessTexture3dRead _) noexcept { _inner.tag = BindlessTexture3dRead::tag(); }
    class LC_IR_API BindlessTexture2dReadLevel : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BindlessTexture2dReadLevel; }
    };
    explicit Func(Func::BindlessTexture2dReadLevel _) noexcept { _inner.tag = BindlessTexture2dReadLevel::tag(); }
    class LC_IR_API BindlessTexture3dReadLevel : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BindlessTexture3dReadLevel; }
    };
    explicit Func(Func::BindlessTexture3dReadLevel _) noexcept { _inner.tag = BindlessTexture3dReadLevel::tag(); }
    class LC_IR_API BindlessTexture2dSize : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BindlessTexture2dSize; }
    };
    explicit Func(Func::BindlessTexture2dSize _) noexcept { _inner.tag = BindlessTexture2dSize::tag(); }
    class LC_IR_API BindlessTexture3dSize : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BindlessTexture3dSize; }
    };
    explicit Func(Func::BindlessTexture3dSize _) noexcept { _inner.tag = BindlessTexture3dSize::tag(); }
    class LC_IR_API BindlessTexture2dSizeLevel : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BindlessTexture2dSizeLevel; }
    };
    explicit Func(Func::BindlessTexture2dSizeLevel _) noexcept { _inner.tag = BindlessTexture2dSizeLevel::tag(); }
    class LC_IR_API BindlessTexture3dSizeLevel : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BindlessTexture3dSizeLevel; }
    };
    explicit Func(Func::BindlessTexture3dSizeLevel _) noexcept { _inner.tag = BindlessTexture3dSizeLevel::tag(); }
    class LC_IR_API BindlessBufferRead : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BindlessBufferRead; }
    };
    explicit Func(Func::BindlessBufferRead _) noexcept { _inner.tag = BindlessBufferRead::tag(); }
    class LC_IR_API BindlessBufferSize : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BindlessBufferSize; }
    };
    explicit Func(Func::BindlessBufferSize _) noexcept { _inner.tag = BindlessBufferSize::tag(); }
    class LC_IR_API BindlessBufferType : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BindlessBufferType; }
    };
    explicit Func(Func::BindlessBufferType _) noexcept { _inner.tag = BindlessBufferType::tag(); }
    class LC_IR_API BindlessByteBufferRead : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::BindlessByteBufferRead; }
    };
    explicit Func(Func::BindlessByteBufferRead _) noexcept { _inner.tag = BindlessByteBufferRead::tag(); }
    class LC_IR_API Vec : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Vec; }
    };
    explicit Func(Func::Vec _) noexcept { _inner.tag = Vec::tag(); }
    class LC_IR_API Vec2 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Vec2; }
    };
    explicit Func(Func::Vec2 _) noexcept { _inner.tag = Vec2::tag(); }
    class LC_IR_API Vec3 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Vec3; }
    };
    explicit Func(Func::Vec3 _) noexcept { _inner.tag = Vec3::tag(); }
    class LC_IR_API Vec4 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Vec4; }
    };
    explicit Func(Func::Vec4 _) noexcept { _inner.tag = Vec4::tag(); }
    class LC_IR_API Permute : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Permute; }
    };
    explicit Func(Func::Permute _) noexcept { _inner.tag = Permute::tag(); }
    class LC_IR_API InsertElement : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::InsertElement; }
    };
    explicit Func(Func::InsertElement _) noexcept { _inner.tag = InsertElement::tag(); }
    class LC_IR_API ExtractElement : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::ExtractElement; }
    };
    explicit Func(Func::ExtractElement _) noexcept { _inner.tag = ExtractElement::tag(); }
    class LC_IR_API GetElementPtr : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::GetElementPtr; }
    };
    explicit Func(Func::GetElementPtr _) noexcept { _inner.tag = GetElementPtr::tag(); }
    class LC_IR_API Struct : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Struct; }
    };
    explicit Func(Func::Struct _) noexcept { _inner.tag = Struct::tag(); }
    class LC_IR_API Array : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Array; }
    };
    explicit Func(Func::Array _) noexcept { _inner.tag = Array::tag(); }
    class LC_IR_API Mat : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Mat; }
    };
    explicit Func(Func::Mat _) noexcept { _inner.tag = Mat::tag(); }
    class LC_IR_API Mat2 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Mat2; }
    };
    explicit Func(Func::Mat2 _) noexcept { _inner.tag = Mat2::tag(); }
    class LC_IR_API Mat3 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Mat3; }
    };
    explicit Func(Func::Mat3 _) noexcept { _inner.tag = Mat3::tag(); }
    class LC_IR_API Mat4 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Mat4; }
    };
    explicit Func(Func::Mat4 _) noexcept { _inner.tag = Mat4::tag(); }
    class LC_IR_API Callable : Marker, concepts::Noncopyable {
        raw::Func::Callable_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Callable; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API CpuCustomOp : Marker, concepts::Noncopyable {
        raw::Func::CpuCustomOp_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::CpuCustomOp; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API ShaderExecutionReorder : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::ShaderExecutionReorder; }
    };
    explicit Func(Func::ShaderExecutionReorder _) noexcept { _inner.tag = ShaderExecutionReorder::tag(); }
    class LC_IR_API Unknown0 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Unknown0; }
    };
    explicit Func(Func::Unknown0 _) noexcept { _inner.tag = Unknown0::tag(); }
    class LC_IR_API Unknown1 : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Func::Tag::Unknown1; }
    };
    explicit Func(Func::Unknown1 _) noexcept { _inner.tag = Unknown1::tag(); }
public:
    [[nodiscard]] auto tag() const noexcept { return _inner.tag; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    template<class T>
    [[nodiscard]] bool isa() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        return _inner.tag == T::tag();
    }
    template<class T>
    [[nodiscard]] const T *as() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        if (!isa<T>()) return nullptr;
        if constexpr (std::is_same_v<T, Unreachable>) {
            return reinterpret_cast<const Unreachable *>(&_inner.unreachable);
        }
        if constexpr (std::is_same_v<T, Assert>) {
            return reinterpret_cast<const Assert *>(&_inner.assert);
        }
        if constexpr (std::is_same_v<T, Callable>) {
            return reinterpret_cast<const Callable *>(&_inner.callable);
        }
        if constexpr (std::is_same_v<T, CpuCustomOp>) {
            return reinterpret_cast<const CpuCustomOp *>(&_inner.cpu_custom_op);
        }
        return reinterpret_cast<const T *>(this);
    }
};
static_assert(sizeof(Func) == sizeof(raw::Func));

namespace detail {
template<>
struct FromInnerRef<raw::Func> {
    using Output = Func;
    static const Output &from(const raw::Func &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::Func>> {
    using Output = CArc<Func>;
    static const Output &from(const CArc<raw::Func> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::Func>> {
    using Output = Pooled<Func>;
    static const Output &from(const Pooled<raw::Func> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::Func>> {
    using Output = CBoxedSlice<Func>;
    static const Output &from(const CBoxedSlice<raw::Func> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API Const : concepts::Noncopyable {
    raw::Const _inner{};
    class Marker {};

public:
    friend class IrBuilder;
    using Tag = raw::Const::Tag;
    class LC_IR_API Zero : Marker, concepts::Noncopyable {
        raw::Const::Zero_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Zero; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API One : Marker, concepts::Noncopyable {
        raw::Const::One_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::One; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Bool : Marker, concepts::Noncopyable {
        raw::Const::Bool_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Bool; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Int8 : Marker, concepts::Noncopyable {
        raw::Const::Int8_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Int8; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Uint8 : Marker, concepts::Noncopyable {
        raw::Const::Uint8_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Uint8; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Int16 : Marker, concepts::Noncopyable {
        raw::Const::Int16_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Int16; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Uint16 : Marker, concepts::Noncopyable {
        raw::Const::Uint16_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Uint16; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Int32 : Marker, concepts::Noncopyable {
        raw::Const::Int32_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Int32; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Uint32 : Marker, concepts::Noncopyable {
        raw::Const::Uint32_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Uint32; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Int64 : Marker, concepts::Noncopyable {
        raw::Const::Int64_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Int64; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Uint64 : Marker, concepts::Noncopyable {
        raw::Const::Uint64_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Uint64; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Float16 : Marker, concepts::Noncopyable {
        raw::Const::Float16_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Float16; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Float32 : Marker, concepts::Noncopyable {
        raw::Const::Float32_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Float32; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Float64 : Marker, concepts::Noncopyable {
        raw::Const::Float64_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Float64; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Generic : Marker, concepts::Noncopyable {
        raw::Const::Generic_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Const::Tag::Generic; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
public:
    [[nodiscard]] auto tag() const noexcept { return _inner.tag; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    template<class T>
    [[nodiscard]] bool isa() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        return _inner.tag == T::tag();
    }
    template<class T>
    [[nodiscard]] const T *as() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        if (!isa<T>()) return nullptr;
        if constexpr (std::is_same_v<T, Zero>) {
            return reinterpret_cast<const Zero *>(&_inner.zero);
        }
        if constexpr (std::is_same_v<T, One>) {
            return reinterpret_cast<const One *>(&_inner.one);
        }
        if constexpr (std::is_same_v<T, Bool>) {
            return reinterpret_cast<const Bool *>(&_inner.bool_);
        }
        if constexpr (std::is_same_v<T, Int8>) {
            return reinterpret_cast<const Int8 *>(&_inner.int8);
        }
        if constexpr (std::is_same_v<T, Uint8>) {
            return reinterpret_cast<const Uint8 *>(&_inner.uint8);
        }
        if constexpr (std::is_same_v<T, Int16>) {
            return reinterpret_cast<const Int16 *>(&_inner.int16);
        }
        if constexpr (std::is_same_v<T, Uint16>) {
            return reinterpret_cast<const Uint16 *>(&_inner.uint16);
        }
        if constexpr (std::is_same_v<T, Int32>) {
            return reinterpret_cast<const Int32 *>(&_inner.int32);
        }
        if constexpr (std::is_same_v<T, Uint32>) {
            return reinterpret_cast<const Uint32 *>(&_inner.uint32);
        }
        if constexpr (std::is_same_v<T, Int64>) {
            return reinterpret_cast<const Int64 *>(&_inner.int64);
        }
        if constexpr (std::is_same_v<T, Uint64>) {
            return reinterpret_cast<const Uint64 *>(&_inner.uint64);
        }
        if constexpr (std::is_same_v<T, Float16>) {
            return reinterpret_cast<const Float16 *>(&_inner.float16);
        }
        if constexpr (std::is_same_v<T, Float32>) {
            return reinterpret_cast<const Float32 *>(&_inner.float32);
        }
        if constexpr (std::is_same_v<T, Float64>) {
            return reinterpret_cast<const Float64 *>(&_inner.float64);
        }
        if constexpr (std::is_same_v<T, Generic>) {
            return reinterpret_cast<const Generic *>(&_inner.generic);
        }
        return reinterpret_cast<const T *>(this);
    }
};
static_assert(sizeof(Const) == sizeof(raw::Const));

namespace detail {
template<>
struct FromInnerRef<raw::Const> {
    using Output = Const;
    static const Output &from(const raw::Const &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::Const>> {
    using Output = CArc<Const>;
    static const Output &from(const CArc<raw::Const> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::Const>> {
    using Output = Pooled<Const>;
    static const Output &from(const Pooled<raw::Const> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::Const>> {
    using Output = CBoxedSlice<Const>;
    static const Output &from(const CBoxedSlice<raw::Const> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API NodeRef {
    raw::NodeRef _inner{};

public:
    friend class IrBuilder;

    // including extra code from data/NodeRef.h
    [[nodiscard]] const Node *operator->() const noexcept;
    [[nodiscard]] const Node *get() const noexcept;
    static NodeRef from_raw(raw::NodeRef raw) noexcept {
        auto ret = NodeRef{};
        ret._inner = raw;
        return ret;
    }
    [[nodiscard]] auto raw() const noexcept { return _inner; }
    [[nodiscard]] auto operator==(const NodeRef &rhs) const noexcept { return raw() == rhs.raw(); }
    [[nodiscard]] auto valid() const noexcept { return raw() != raw::INVALID_REF; }
    void insert_before_self(NodeRef node) noexcept;
    void insert_after_self(NodeRef node) noexcept;
    void replace_with(NodeRef node) noexcept;
    void remove() noexcept;
    // end include
};

namespace detail {
template<>
struct FromInnerRef<raw::NodeRef> {
    using Output = NodeRef;
    static const Output &from(const raw::NodeRef &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::NodeRef>> {
    using Output = CArc<NodeRef>;
    static const Output &from(const CArc<raw::NodeRef> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::NodeRef>> {
    using Output = Pooled<NodeRef>;
    static const Output &from(const Pooled<raw::NodeRef> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::NodeRef>> {
    using Output = CBoxedSlice<NodeRef>;
    static const Output &from(const CBoxedSlice<raw::NodeRef> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API PhiIncoming : concepts::Noncopyable {
    raw::PhiIncoming _inner{};

public:
    friend class IrBuilder;
    [[nodiscard]] auto raw() noexcept { return &_inner; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    [[nodiscard]] const NodeRef &value() const noexcept;
    [[nodiscard]] const Pooled<BasicBlock> &block() const noexcept;
};

namespace detail {
template<>
struct FromInnerRef<raw::PhiIncoming> {
    using Output = PhiIncoming;
    static const Output &from(const raw::PhiIncoming &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::PhiIncoming>> {
    using Output = CArc<PhiIncoming>;
    static const Output &from(const CArc<raw::PhiIncoming> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::PhiIncoming>> {
    using Output = Pooled<PhiIncoming>;
    static const Output &from(const Pooled<raw::PhiIncoming> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::PhiIncoming>> {
    using Output = CBoxedSlice<PhiIncoming>;
    static const Output &from(const CBoxedSlice<raw::PhiIncoming> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API SwitchCase : concepts::Noncopyable {
    raw::SwitchCase _inner{};

public:
    friend class IrBuilder;
    [[nodiscard]] auto raw() noexcept { return &_inner; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    [[nodiscard]] const int32_t &value() const noexcept;
    [[nodiscard]] const Pooled<BasicBlock> &block() const noexcept;
};

namespace detail {
template<>
struct FromInnerRef<raw::SwitchCase> {
    using Output = SwitchCase;
    static const Output &from(const raw::SwitchCase &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::SwitchCase>> {
    using Output = CArc<SwitchCase>;
    static const Output &from(const CArc<raw::SwitchCase> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::SwitchCase>> {
    using Output = Pooled<SwitchCase>;
    static const Output &from(const Pooled<raw::SwitchCase> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::SwitchCase>> {
    using Output = CBoxedSlice<SwitchCase>;
    static const Output &from(const CBoxedSlice<raw::SwitchCase> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API Instruction : concepts::Noncopyable {
    raw::Instruction _inner{};
    class Marker {};

public:
    friend class IrBuilder;
    using Tag = raw::Instruction::Tag;
    class LC_IR_API Buffer : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Buffer; }
    };
    explicit Instruction(Instruction::Buffer _) noexcept { _inner.tag = Buffer::tag(); }
    class LC_IR_API Bindless : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Bindless; }
    };
    explicit Instruction(Instruction::Bindless _) noexcept { _inner.tag = Bindless::tag(); }
    class LC_IR_API Texture2D : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Texture2D; }
    };
    explicit Instruction(Instruction::Texture2D _) noexcept { _inner.tag = Texture2D::tag(); }
    class LC_IR_API Texture3D : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Texture3D; }
    };
    explicit Instruction(Instruction::Texture3D _) noexcept { _inner.tag = Texture3D::tag(); }
    class LC_IR_API Accel : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Accel; }
    };
    explicit Instruction(Instruction::Accel _) noexcept { _inner.tag = Accel::tag(); }
    class LC_IR_API Shared : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Shared; }
    };
    explicit Instruction(Instruction::Shared _) noexcept { _inner.tag = Shared::tag(); }
    class LC_IR_API Uniform : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Uniform; }
    };
    explicit Instruction(Instruction::Uniform _) noexcept { _inner.tag = Uniform::tag(); }
    class LC_IR_API Local : Marker, concepts::Noncopyable {
        raw::Instruction::Local_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Local; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
        [[nodiscard]] const NodeRef &init() const noexcept;
    };
    class LC_IR_API Argument : Marker, concepts::Noncopyable {
        raw::Instruction::Argument_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Argument; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
        [[nodiscard]] const bool &by_value() const noexcept;
    };
    class LC_IR_API UserData : Marker, concepts::Noncopyable {
        raw::Instruction::UserData_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::UserData; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Invalid : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Invalid; }
    };
    explicit Instruction(Instruction::Invalid _) noexcept { _inner.tag = Invalid::tag(); }
    class LC_IR_API Const : Marker, concepts::Noncopyable {
        raw::Instruction::Const_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Const; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Update : Marker, concepts::Noncopyable {
        raw::Instruction::Update_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Update; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
        [[nodiscard]] const NodeRef &var() const noexcept;
        [[nodiscard]] const NodeRef &value() const noexcept;
    };
    class LC_IR_API Call : Marker, concepts::Noncopyable {
        raw::Instruction::Call_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Call; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Phi : Marker, concepts::Noncopyable {
        raw::Instruction::Phi_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Phi; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Return : Marker, concepts::Noncopyable {
        raw::Instruction::Return_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Return; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Loop : Marker, concepts::Noncopyable {
        raw::Instruction::Loop_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Loop; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
        [[nodiscard]] const Pooled<BasicBlock> &body() const noexcept;
        [[nodiscard]] const NodeRef &cond() const noexcept;
    };
    class LC_IR_API GenericLoop : Marker, concepts::Noncopyable {
        raw::Instruction::GenericLoop_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::GenericLoop; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
        [[nodiscard]] const Pooled<BasicBlock> &prepare() const noexcept;
        [[nodiscard]] const NodeRef &cond() const noexcept;
        [[nodiscard]] const Pooled<BasicBlock> &body() const noexcept;
        [[nodiscard]] const Pooled<BasicBlock> &update() const noexcept;
    };
    class LC_IR_API Break : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Break; }
    };
    explicit Instruction(Instruction::Break _) noexcept { _inner.tag = Break::tag(); }
    class LC_IR_API Continue : Marker, concepts::Noncopyable {
        uint8_t _pad;
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Continue; }
    };
    explicit Instruction(Instruction::Continue _) noexcept { _inner.tag = Continue::tag(); }
    class LC_IR_API If : Marker, concepts::Noncopyable {
        raw::Instruction::If_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::If; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
        [[nodiscard]] const NodeRef &cond() const noexcept;
        [[nodiscard]] const Pooled<BasicBlock> &true_branch() const noexcept;
        [[nodiscard]] const Pooled<BasicBlock> &false_branch() const noexcept;
    };
    class LC_IR_API Switch : Marker, concepts::Noncopyable {
        raw::Instruction::Switch_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Switch; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
        [[nodiscard]] const NodeRef &value() const noexcept;
        [[nodiscard]] const Pooled<BasicBlock> &default_() const noexcept;
        [[nodiscard]] luisa::span<const SwitchCase> cases() const noexcept;
    };
    class LC_IR_API AdScope : Marker, concepts::Noncopyable {
        raw::Instruction::AdScope_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::AdScope; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
        [[nodiscard]] const Pooled<BasicBlock> &body() const noexcept;
        [[nodiscard]] const bool &forward() const noexcept;
        [[nodiscard]] const size_t &n_forward_grads() const noexcept;
    };
    class LC_IR_API RayQuery : Marker, concepts::Noncopyable {
        raw::Instruction::RayQuery_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::RayQuery; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
        [[nodiscard]] const NodeRef &ray_query() const noexcept;
        [[nodiscard]] const Pooled<BasicBlock> &on_triangle_hit() const noexcept;
        [[nodiscard]] const Pooled<BasicBlock> &on_procedural_hit() const noexcept;
    };
    class LC_IR_API Print : Marker, concepts::Noncopyable {
        raw::Instruction::Print_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Print; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
        [[nodiscard]] luisa::span<const uint8_t> fmt() const noexcept;
        [[nodiscard]] luisa::span<const NodeRef> args() const noexcept;
    };
    class LC_IR_API AdDetach : Marker, concepts::Noncopyable {
        raw::Instruction::AdDetach_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::AdDetach; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Comment : Marker, concepts::Noncopyable {
        raw::Instruction::Comment_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Instruction::Tag::Comment; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
public:
    [[nodiscard]] auto tag() const noexcept { return _inner.tag; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    template<class T>
    [[nodiscard]] bool isa() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        return _inner.tag == T::tag();
    }
    template<class T>
    [[nodiscard]] const T *as() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        if (!isa<T>()) return nullptr;
        if constexpr (std::is_same_v<T, Local>) {
            return reinterpret_cast<const Local *>(&_inner.local);
        }
        if constexpr (std::is_same_v<T, Argument>) {
            return reinterpret_cast<const Argument *>(&_inner.argument);
        }
        if constexpr (std::is_same_v<T, UserData>) {
            return reinterpret_cast<const UserData *>(&_inner.user_data);
        }
        if constexpr (std::is_same_v<T, Const>) {
            return reinterpret_cast<const Const *>(&_inner.const_);
        }
        if constexpr (std::is_same_v<T, Update>) {
            return reinterpret_cast<const Update *>(&_inner.update);
        }
        if constexpr (std::is_same_v<T, Call>) {
            return reinterpret_cast<const Call *>(&_inner.call);
        }
        if constexpr (std::is_same_v<T, Phi>) {
            return reinterpret_cast<const Phi *>(&_inner.phi);
        }
        if constexpr (std::is_same_v<T, Return>) {
            return reinterpret_cast<const Return *>(&_inner.return_);
        }
        if constexpr (std::is_same_v<T, Loop>) {
            return reinterpret_cast<const Loop *>(&_inner.loop);
        }
        if constexpr (std::is_same_v<T, GenericLoop>) {
            return reinterpret_cast<const GenericLoop *>(&_inner.generic_loop);
        }
        if constexpr (std::is_same_v<T, If>) {
            return reinterpret_cast<const If *>(&_inner.if_);
        }
        if constexpr (std::is_same_v<T, Switch>) {
            return reinterpret_cast<const Switch *>(&_inner.switch_);
        }
        if constexpr (std::is_same_v<T, AdScope>) {
            return reinterpret_cast<const AdScope *>(&_inner.ad_scope);
        }
        if constexpr (std::is_same_v<T, RayQuery>) {
            return reinterpret_cast<const RayQuery *>(&_inner.ray_query);
        }
        if constexpr (std::is_same_v<T, Print>) {
            return reinterpret_cast<const Print *>(&_inner.print);
        }
        if constexpr (std::is_same_v<T, AdDetach>) {
            return reinterpret_cast<const AdDetach *>(&_inner.ad_detach);
        }
        if constexpr (std::is_same_v<T, Comment>) {
            return reinterpret_cast<const Comment *>(&_inner.comment);
        }
        return reinterpret_cast<const T *>(this);
    }
};
static_assert(sizeof(Instruction) == sizeof(raw::Instruction));

namespace detail {
template<>
struct FromInnerRef<raw::Instruction> {
    using Output = Instruction;
    static const Output &from(const raw::Instruction &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::Instruction>> {
    using Output = CArc<Instruction>;
    static const Output &from(const CArc<raw::Instruction> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::Instruction>> {
    using Output = Pooled<Instruction>;
    static const Output &from(const Pooled<raw::Instruction> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::Instruction>> {
    using Output = CBoxedSlice<Instruction>;
    static const Output &from(const CBoxedSlice<raw::Instruction> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API BasicBlock : concepts::Noncopyable {
    raw::BasicBlock _inner{};

public:
    friend class IrBuilder;
    [[nodiscard]] auto raw() noexcept { return &_inner; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    [[nodiscard]] const NodeRef &first() const noexcept;
    [[nodiscard]] const NodeRef &last() const noexcept;

    // including extra code from data/BasicBlock.h
    class Iterator {
    public:
        struct Sentinel {};
    private:
        NodeRef _curr;
        NodeRef _end;
        friend class BasicBlock;
        Iterator(NodeRef curr, NodeRef end) noexcept
            : _curr{curr}, _end{end} {}
    public:
        [[nodiscard]] auto operator*() const noexcept { return _curr; }
        auto &operator++() noexcept {
            auto node = _curr->next();
            _curr = node;
            return *this;
        }
        auto operator++(int) noexcept {
            auto old = *this;
            ++(*this);
            return old;
        }
        [[nodiscard]] auto operator==(const Iterator &rhs) const noexcept { return _curr == rhs._curr; }
    };
    [[nodiscard]] auto begin() const noexcept { return Iterator{this->first()->next(), this->last()}; }
    [[nodiscard]] auto end() const noexcept { return Iterator{this->last(), this->last()}; }
    [[nodiscard]] auto cbegin() const noexcept { return this->begin(); }
    [[nodiscard]] auto cend() const noexcept { return this->end(); }
    // end include
};

namespace detail {
template<>
struct FromInnerRef<raw::BasicBlock> {
    using Output = BasicBlock;
    static const Output &from(const raw::BasicBlock &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::BasicBlock>> {
    using Output = CArc<BasicBlock>;
    static const Output &from(const CArc<raw::BasicBlock> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::BasicBlock>> {
    using Output = Pooled<BasicBlock>;
    static const Output &from(const Pooled<raw::BasicBlock> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::BasicBlock>> {
    using Output = CBoxedSlice<BasicBlock>;
    static const Output &from(const CBoxedSlice<raw::BasicBlock> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

using raw::ModuleKind;
class LC_IR_API Module : concepts::Noncopyable {
    raw::Module _inner{};

public:
    friend class IrBuilder;
    [[nodiscard]] auto raw() noexcept { return &_inner; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    [[nodiscard]] const ModuleKind &kind() const noexcept;
    [[nodiscard]] const Pooled<BasicBlock> &entry() const noexcept;
    [[nodiscard]] const ModuleFlags &flags() const noexcept;
    [[nodiscard]] const CArc<ModulePools> &pools() const noexcept;
};

namespace detail {
template<>
struct FromInnerRef<raw::Module> {
    using Output = Module;
    static const Output &from(const raw::Module &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::Module>> {
    using Output = CArc<Module>;
    static const Output &from(const CArc<raw::Module> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::Module>> {
    using Output = Pooled<Module>;
    static const Output &from(const Pooled<raw::Module> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::Module>> {
    using Output = CBoxedSlice<Module>;
    static const Output &from(const CBoxedSlice<raw::Module> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API CallableModule : concepts::Noncopyable {
    raw::CallableModule _inner{};

public:
    friend class IrBuilder;
    [[nodiscard]] auto raw() noexcept { return &_inner; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    [[nodiscard]] const Module &module() const noexcept;
    [[nodiscard]] const CArc<Type> &ret_type() const noexcept;
    [[nodiscard]] luisa::span<const NodeRef> args() const noexcept;
    [[nodiscard]] luisa::span<const Capture> captures() const noexcept;
    [[nodiscard]] luisa::span<const CArc<CpuCustomOp>> cpu_custom_ops() const noexcept;
    [[nodiscard]] const CArc<ModulePools> &pools() const noexcept;
};

namespace detail {
template<>
struct FromInnerRef<raw::CallableModule> {
    using Output = CallableModule;
    static const Output &from(const raw::CallableModule &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::CallableModule>> {
    using Output = CArc<CallableModule>;
    static const Output &from(const CArc<raw::CallableModule> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::CallableModule>> {
    using Output = Pooled<CallableModule>;
    static const Output &from(const Pooled<raw::CallableModule> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::CallableModule>> {
    using Output = CBoxedSlice<CallableModule>;
    static const Output &from(const CBoxedSlice<raw::CallableModule> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API BufferBinding : concepts::Noncopyable {
    raw::BufferBinding _inner{};

public:
    friend class IrBuilder;
    [[nodiscard]] auto raw() noexcept { return &_inner; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    [[nodiscard]] const uint64_t &handle() const noexcept;
    [[nodiscard]] const uint64_t &offset() const noexcept;
    [[nodiscard]] const size_t &size() const noexcept;
};

namespace detail {
template<>
struct FromInnerRef<raw::BufferBinding> {
    using Output = BufferBinding;
    static const Output &from(const raw::BufferBinding &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::BufferBinding>> {
    using Output = CArc<BufferBinding>;
    static const Output &from(const CArc<raw::BufferBinding> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::BufferBinding>> {
    using Output = Pooled<BufferBinding>;
    static const Output &from(const Pooled<raw::BufferBinding> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::BufferBinding>> {
    using Output = CBoxedSlice<BufferBinding>;
    static const Output &from(const CBoxedSlice<raw::BufferBinding> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API TextureBinding : concepts::Noncopyable {
    raw::TextureBinding _inner{};

public:
    friend class IrBuilder;
    [[nodiscard]] auto raw() noexcept { return &_inner; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    [[nodiscard]] const uint64_t &handle() const noexcept;
    [[nodiscard]] const uint32_t &level() const noexcept;
};

namespace detail {
template<>
struct FromInnerRef<raw::TextureBinding> {
    using Output = TextureBinding;
    static const Output &from(const raw::TextureBinding &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::TextureBinding>> {
    using Output = CArc<TextureBinding>;
    static const Output &from(const CArc<raw::TextureBinding> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::TextureBinding>> {
    using Output = Pooled<TextureBinding>;
    static const Output &from(const Pooled<raw::TextureBinding> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::TextureBinding>> {
    using Output = CBoxedSlice<TextureBinding>;
    static const Output &from(const CBoxedSlice<raw::TextureBinding> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API BindlessArrayBinding : concepts::Noncopyable {
    raw::BindlessArrayBinding _inner{};

public:
    friend class IrBuilder;
    [[nodiscard]] auto raw() noexcept { return &_inner; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    [[nodiscard]] const uint64_t &handle() const noexcept;
};

namespace detail {
template<>
struct FromInnerRef<raw::BindlessArrayBinding> {
    using Output = BindlessArrayBinding;
    static const Output &from(const raw::BindlessArrayBinding &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::BindlessArrayBinding>> {
    using Output = CArc<BindlessArrayBinding>;
    static const Output &from(const CArc<raw::BindlessArrayBinding> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::BindlessArrayBinding>> {
    using Output = Pooled<BindlessArrayBinding>;
    static const Output &from(const Pooled<raw::BindlessArrayBinding> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::BindlessArrayBinding>> {
    using Output = CBoxedSlice<BindlessArrayBinding>;
    static const Output &from(const CBoxedSlice<raw::BindlessArrayBinding> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API AccelBinding : concepts::Noncopyable {
    raw::AccelBinding _inner{};

public:
    friend class IrBuilder;
    [[nodiscard]] auto raw() noexcept { return &_inner; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    [[nodiscard]] const uint64_t &handle() const noexcept;
};

namespace detail {
template<>
struct FromInnerRef<raw::AccelBinding> {
    using Output = AccelBinding;
    static const Output &from(const raw::AccelBinding &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::AccelBinding>> {
    using Output = CArc<AccelBinding>;
    static const Output &from(const CArc<raw::AccelBinding> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::AccelBinding>> {
    using Output = Pooled<AccelBinding>;
    static const Output &from(const Pooled<raw::AccelBinding> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::AccelBinding>> {
    using Output = CBoxedSlice<AccelBinding>;
    static const Output &from(const CBoxedSlice<raw::AccelBinding> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API Binding : concepts::Noncopyable {
    raw::Binding _inner{};
    class Marker {};

public:
    friend class IrBuilder;
    using Tag = raw::Binding::Tag;
    class LC_IR_API Buffer : Marker, concepts::Noncopyable {
        raw::Binding::Buffer_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Binding::Tag::Buffer; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Texture : Marker, concepts::Noncopyable {
        raw::Binding::Texture_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Binding::Tag::Texture; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API BindlessArray : Marker, concepts::Noncopyable {
        raw::Binding::BindlessArray_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Binding::Tag::BindlessArray; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
    class LC_IR_API Accel : Marker, concepts::Noncopyable {
        raw::Binding::Accel_Body _inner{};
    public:
        static constexpr Tag tag() noexcept { return raw::Binding::Tag::Accel; }
        [[nodiscard]] auto raw() const noexcept { return &_inner; }
    };
public:
    [[nodiscard]] auto tag() const noexcept { return _inner.tag; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    template<class T>
    [[nodiscard]] bool isa() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        return _inner.tag == T::tag();
    }
    template<class T>
    [[nodiscard]] const T *as() const noexcept {
        static_assert(std::is_base_of_v<Marker, T>);
        if (!isa<T>()) return nullptr;
        if constexpr (std::is_same_v<T, Buffer>) {
            return reinterpret_cast<const Buffer *>(&_inner.buffer);
        }
        if constexpr (std::is_same_v<T, Texture>) {
            return reinterpret_cast<const Texture *>(&_inner.texture);
        }
        if constexpr (std::is_same_v<T, BindlessArray>) {
            return reinterpret_cast<const BindlessArray *>(&_inner.bindless_array);
        }
        if constexpr (std::is_same_v<T, Accel>) {
            return reinterpret_cast<const Accel *>(&_inner.accel);
        }
        return reinterpret_cast<const T *>(this);
    }
};
static_assert(sizeof(Binding) == sizeof(raw::Binding));

namespace detail {
template<>
struct FromInnerRef<raw::Binding> {
    using Output = Binding;
    static const Output &from(const raw::Binding &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::Binding>> {
    using Output = CArc<Binding>;
    static const Output &from(const CArc<raw::Binding> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::Binding>> {
    using Output = Pooled<Binding>;
    static const Output &from(const Pooled<raw::Binding> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::Binding>> {
    using Output = CBoxedSlice<Binding>;
    static const Output &from(const CBoxedSlice<raw::Binding> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API Capture : concepts::Noncopyable {
    raw::Capture _inner{};

public:
    friend class IrBuilder;
    [[nodiscard]] auto raw() noexcept { return &_inner; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    [[nodiscard]] const NodeRef &node() const noexcept;
    [[nodiscard]] const Binding &binding() const noexcept;
};

namespace detail {
template<>
struct FromInnerRef<raw::Capture> {
    using Output = Capture;
    static const Output &from(const raw::Capture &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::Capture>> {
    using Output = CArc<Capture>;
    static const Output &from(const CArc<raw::Capture> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::Capture>> {
    using Output = Pooled<Capture>;
    static const Output &from(const Pooled<raw::Capture> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::Capture>> {
    using Output = CBoxedSlice<Capture>;
    static const Output &from(const CBoxedSlice<raw::Capture> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API KernelModule : concepts::Noncopyable {
    raw::KernelModule _inner{};

public:
    friend class IrBuilder;
    [[nodiscard]] auto raw() noexcept { return &_inner; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    [[nodiscard]] const Module &module() const noexcept;
    [[nodiscard]] luisa::span<const Capture> captures() const noexcept;
    [[nodiscard]] luisa::span<const NodeRef> args() const noexcept;
    [[nodiscard]] luisa::span<const NodeRef> shared() const noexcept;
    [[nodiscard]] luisa::span<const CArc<CpuCustomOp>> cpu_custom_ops() const noexcept;
    [[nodiscard]] const std::array<uint32_t, 3> &block_size() const noexcept;
    [[nodiscard]] const CArc<ModulePools> &pools() const noexcept;
};

namespace detail {
template<>
struct FromInnerRef<raw::KernelModule> {
    using Output = KernelModule;
    static const Output &from(const raw::KernelModule &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::KernelModule>> {
    using Output = CArc<KernelModule>;
    static const Output &from(const CArc<raw::KernelModule> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::KernelModule>> {
    using Output = Pooled<KernelModule>;
    static const Output &from(const Pooled<raw::KernelModule> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::KernelModule>> {
    using Output = CBoxedSlice<KernelModule>;
    static const Output &from(const CBoxedSlice<raw::KernelModule> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API BlockModule : concepts::Noncopyable {
    raw::BlockModule _inner{};

public:
    friend class IrBuilder;
    [[nodiscard]] auto raw() noexcept { return &_inner; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    [[nodiscard]] const Module &module() const noexcept;
};

namespace detail {
template<>
struct FromInnerRef<raw::BlockModule> {
    using Output = BlockModule;
    static const Output &from(const raw::BlockModule &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::BlockModule>> {
    using Output = CArc<BlockModule>;
    static const Output &from(const CArc<raw::BlockModule> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::BlockModule>> {
    using Output = Pooled<BlockModule>;
    static const Output &from(const Pooled<raw::BlockModule> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::BlockModule>> {
    using Output = CBoxedSlice<BlockModule>;
    static const Output &from(const CBoxedSlice<raw::BlockModule> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

class LC_IR_API IrBuilder : concepts::Noncopyable {
    raw::IrBuilder _inner{};

public:
    friend class IrBuilder;
    [[nodiscard]] auto raw() noexcept { return &_inner; }
    [[nodiscard]] auto raw() const noexcept { return &_inner; }
    [[nodiscard]] const Pooled<BasicBlock> &bb() const noexcept;
    [[nodiscard]] const CArc<ModulePools> &pools() const noexcept;
    [[nodiscard]] const NodeRef &insert_point() const noexcept;

    // including extra code from data/IrBuilder.h
    IrBuilder(raw::IrBuilder inner) noexcept : _inner{inner} {}

    NodeRef call(const Func &f, luisa::span<const NodeRef> args, const CArc<Type> &type) noexcept;
    NodeRef phi(luisa::span<const PhiIncoming> incoming, const CArc<Type> &type) noexcept;
    NodeRef local(const CppOwnedCArc<Type> &type) noexcept;
    NodeRef local(NodeRef init) noexcept;
    NodeRef if_(const NodeRef &cond, const Pooled<BasicBlock> &true_branch, const Pooled<BasicBlock> &false_branch) noexcept;
    NodeRef switch_(const NodeRef &value, luisa::span<const SwitchCase> cases, const Pooled<BasicBlock> &default_) noexcept;
    NodeRef loop(const Pooled<BasicBlock> &body, const NodeRef &cond) noexcept;
    NodeRef generic_loop(const Pooled<BasicBlock> &prepare, const NodeRef &cond, const Pooled<BasicBlock> &body, const Pooled<BasicBlock> &update) noexcept;
    static Pooled<BasicBlock> finish(IrBuilder &&builder) noexcept;
    void set_insert_point(const NodeRef &node) noexcept;

    template<class F>
    static Pooled<BasicBlock> with(const CppOwnedCArc<ModulePools> &pools, F &&f) {
        static_assert(std::is_invocable_v<F, IrBuilder &>);
        static_assert(std::is_same_v<std::invoke_result_t<F, IrBuilder &>, void>);
        auto _inner = luisa_compute_ir_new_builder(pools.clone());
        auto builder = IrBuilder{_inner};
        f(builder);
        return IrBuilder::finish(std::move(builder));
    }
    // end include
};

namespace detail {
template<>
struct FromInnerRef<raw::IrBuilder> {
    using Output = IrBuilder;
    static const Output &from(const raw::IrBuilder &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CArc<raw::IrBuilder>> {
    using Output = CArc<IrBuilder>;
    static const Output &from(const CArc<raw::IrBuilder> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<Pooled<raw::IrBuilder>> {
    using Output = Pooled<IrBuilder>;
    static const Output &from(const Pooled<raw::IrBuilder> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<>
struct FromInnerRef<CBoxedSlice<raw::IrBuilder>> {
    using Output = CBoxedSlice<IrBuilder>;
    static const Output &from(const CBoxedSlice<raw::IrBuilder> &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
}//namespace detail

}// namespace luisa::compute::ir_v2
