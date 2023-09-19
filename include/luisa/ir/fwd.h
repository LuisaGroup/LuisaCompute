#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/stl/memory.h>// for span
#include <luisa/core/concepts.h>  // for Noncopyable
#include <luisa/rust/ir.hpp>

// deduction guide for CSlice
namespace luisa::compute::ir {
template<typename T>
CSlice(T *, size_t) -> CSlice<T>;
template<typename T>
CSlice(const T *, size_t) -> CSlice<T>;
}// namespace luisa::compute::ir

namespace luisa::compute::ir_v2 {
namespace raw = luisa::compute::ir;
using raw::CArc;
using raw::CBoxedSlice;
using raw::CppOwnedCArc;
using raw::Pooled;
using raw::ModulePools;
using raw::CallableModuleRef;
using raw::CpuCustomOp;
using raw::ModuleFlags;

namespace detail {
template<class T>
struct FromInnerRef {
    using Output = T;
    static const FromInnerRef::Output &from(const T &_inner) noexcept {
        return reinterpret_cast<const T &>(_inner);
    }
};
template<class T, size_t N>
struct FromInnerRef<T[N]> {
    using E = std::remove_extent_t<T>;
    using Output = std::array<E, N>;
    using A = T[N];
    static const Output &from(const A &_inner) noexcept {
        return reinterpret_cast<const Output &>(_inner);
    }
};
template<class T>
const typename FromInnerRef<T>::Output &from_inner_ref(const T &_inner) noexcept {
    return FromInnerRef<T>::from(_inner);
}
}// namespace detail

class VectorElementType;
class VectorType;
class MatrixType;
class StructType;
class ArrayType;
class Type;
class Node;
class Func;
class Const;
class NodeRef;
class PhiIncoming;
class SwitchCase;
class Instruction;
class BasicBlock;
class Module;
class CallableModule;
class BufferBinding;
class TextureBinding;
class BindlessArrayBinding;
class AccelBinding;
class Binding;
class Capture;
class KernelModule;
class BlockModule;
class IrBuilder;

}// namespace luisa::compute::ir_v2
