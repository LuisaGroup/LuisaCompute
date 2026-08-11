#include "llvm_native_math.h"

#include <array>
#include <string>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Module.h>

namespace luisa::compute::cpu {

namespace {

// The trigonometric range reduction and the exp/log approximations are
// adapted from SLEEF's single-precision implementations (Copyright Naoki
// Shibata and contributors 2010-2025, Boost Software License 1.0). See
// LICENSE.SLEEF.txt in this directory. These builders emit generic LLVM
// fixed-vector IR instead of any target-specific SLEEF helper layer.

struct DoubleFloat {
    ::llvm::Value *hi;
    ::llvm::Value *lo;
};

struct FloatInt {
    ::llvm::Value *value;
    ::llvm::Value *integer;
};

struct ReducedPi {
    DoubleFloat value;
    ::llvm::Value *quadrant;
};

enum struct TrigKind : uint8_t {
    sin,
    cos,
    tan,
};

class TrigF32IRBuilder {

private:
    ::llvm::Module &_module;
    ::llvm::IRBuilder<> _builder;
    ::llvm::FixedVectorType *_float_vector;
    ::llvm::FixedVectorType *_int_vector;
    bool _cosine;
    bool _tangent;

private:
    [[nodiscard]] ::llvm::Constant *_f32(double x) const {
        auto *scalar = ::llvm::ConstantFP::get(
            ::llvm::Type::getFloatTy(_module.getContext()), x);
        return ::llvm::ConstantVector::getSplat(
            _float_vector->getElementCount(), scalar);
    }

    [[nodiscard]] ::llvm::Constant *_i32(uint32_t x) const {
        auto *scalar = ::llvm::ConstantInt::get(
            ::llvm::Type::getInt32Ty(_module.getContext()), x);
        return ::llvm::ConstantVector::getSplat(
            _int_vector->getElementCount(), scalar);
    }

    [[nodiscard]] ::llvm::Value *_mla(
        ::llvm::Value *x, ::llvm::Value *y, ::llvm::Value *z) {
        // Keep the multiply and add distinct. The error-free transforms below
        // are the non-FMA SLEEF formulation and rely on this operation order.
        return _builder.CreateFAdd(_builder.CreateFMul(x, y), z);
    }

    [[nodiscard]] ::llvm::Value *_float_bits(::llvm::Value *x) {
        return _builder.CreateBitCast(x, _int_vector);
    }

    [[nodiscard]] ::llvm::Value *_bits_float(::llvm::Value *x) {
        return _builder.CreateBitCast(x, _float_vector);
    }

    [[nodiscard]] ::llvm::Value *_abs(::llvm::Value *x) {
        return _bits_float(_builder.CreateAnd(
            _float_bits(x), _i32(0x7fffffffu)));
    }

    [[nodiscard]] ::llvm::Value *_copy_sign(
        ::llvm::Value *magnitude, ::llvm::Value *sign) {
        auto *magnitude_bits = _builder.CreateAnd(
            _float_bits(magnitude), _i32(0x7fffffffu));
        auto *sign_bits = _builder.CreateAnd(
            _float_bits(sign), _i32(0x80000000u));
        return _bits_float(_builder.CreateOr(magnitude_bits, sign_bits));
    }

    [[nodiscard]] ::llvm::Value *_xor_sign(
        ::llvm::Value *x, ::llvm::Value *sign) {
        return _bits_float(_builder.CreateXor(
            _float_bits(x), _builder.CreateAnd(
                                _float_bits(sign), _i32(0x80000000u))));
    }

    [[nodiscard]] ::llvm::Value *_upper(::llvm::Value *x) {
        return _bits_float(_builder.CreateAnd(
            _float_bits(x), _i32(0xfffff000u)));
    }

    [[nodiscard]] ::llvm::Value *_round_nearest(::llvm::Value *x) {
        auto *bias = _copy_sign(_f32(8388608.0), x);
        return _builder.CreateFSub(
            _builder.CreateFAdd(x, bias), bias);
    }

    [[nodiscard]] DoubleFloat _normalize(DoubleFloat x) {
        auto *sum = _builder.CreateFAdd(x.hi, x.lo);
        auto *error = _builder.CreateFAdd(
            _builder.CreateFSub(x.hi, sum), x.lo);
        return {sum, error};
    }

    [[nodiscard]] DoubleFloat _add2(DoubleFloat x, DoubleFloat y) {
        auto *sum = _builder.CreateFAdd(x.hi, y.hi);
        auto *v = _builder.CreateFSub(sum, x.hi);
        auto *error = _builder.CreateFAdd(
            _builder.CreateFSub(
                x.hi, _builder.CreateFSub(sum, v)),
            _builder.CreateFSub(y.hi, v));
        error = _builder.CreateFAdd(
            error, _builder.CreateFAdd(x.lo, y.lo));
        return {sum, error};
    }

    [[nodiscard]] DoubleFloat _mul(
        ::llvm::Value *x, ::llvm::Value *y) {
        auto *xh = _upper(x);
        auto *xl = _builder.CreateFSub(x, xh);
        auto *yh = _upper(y);
        auto *yl = _builder.CreateFSub(y, yh);
        auto *product = _builder.CreateFMul(x, y);
        auto *error = _mla(xh, yh, _builder.CreateFNeg(product));
        error = _mla(xl, yh, error);
        error = _mla(xh, yl, error);
        error = _mla(xl, yl, error);
        return {product, error};
    }

    [[nodiscard]] DoubleFloat _mul(
        DoubleFloat x, ::llvm::Value *y) {
        auto result = _mul(x.hi, y);
        result.lo = _mla(x.lo, y, result.lo);
        return result;
    }

    [[nodiscard]] DoubleFloat _mul(DoubleFloat x, DoubleFloat y) {
        auto result = _mul(x.hi, y.hi);
        result.lo = _mla(x.hi, y.lo, result.lo);
        result.lo = _mla(x.lo, y.hi, result.lo);
        return result;
    }

    [[nodiscard]] FloatInt _reduce_quarter(::llvm::Value *x) {
        auto *four_x = _builder.CreateFMul(_f32(4.0), x);
        auto *rint_four = _round_nearest(four_x);
        auto *rint_x = _round_nearest(x);
        auto *value = _mla(_f32(-0.25), rint_four, x);
        auto *integer_float = _mla(_f32(-4.0), rint_x, rint_four);
        auto *integer = _builder.CreateFPToSI(integer_float, _int_vector);
        return {value, integer};
    }

    [[nodiscard]] ::llvm::GlobalVariable *_reduction_table();

    [[nodiscard]] ::llvm::Value *_table_gather(::llvm::Value *indices) {
        auto *table = _reduction_table();
        auto *array_type = table->getValueType();
        auto *base = _builder.CreateInBoundsGEP(
            array_type, table,
            {_builder.getInt32(0), _builder.getInt32(0)});
        auto *pointers = _builder.CreateGEP(
            _builder.getInt32Ty(), base, indices);
        auto *mask = ::llvm::ConstantVector::getSplat(
            _float_vector->getElementCount(), _builder.getTrue());
        auto *gathered = _builder.CreateMaskedGather(
            _int_vector, pointers, ::llvm::Align{4u}, mask,
            ::llvm::Constant::getNullValue(_int_vector));
        return _bits_float(gathered);
    }

    [[nodiscard]] ReducedPi _reduce_pi(::llvm::Value *input) {
        auto *bits = _float_bits(input);
        auto *exponent = _builder.CreateSub(
            _builder.CreateAnd(
                _builder.CreateLShr(bits, _i32(23u)), _i32(0xffu)),
            _i32(127u + 25u));
        auto *scale = _builder.CreateSelect(
            _builder.CreateICmpSGT(exponent, _i32(90u - 25u)),
            _i32(static_cast<uint32_t>(-64)), _i32(0u));
        auto *scaled_bits = _builder.CreateAdd(
            bits, _builder.CreateShl(scale, _i32(23u)));
        auto *scaled = _bits_float(scaled_bits);
        exponent = _builder.CreateSelect(
            _builder.CreateICmpSGT(exponent, _i32(0u)),
            exponent, _i32(0u));
        auto *indices = _builder.CreateShl(exponent, _i32(2u));

        auto result = _mul(scaled, _table_gather(indices));
        auto part = _reduce_quarter(result.hi);
        auto *quadrant = part.integer;
        result.hi = part.value;
        result = _normalize(result);

        auto *indices_1 = _builder.CreateAdd(indices, _i32(1u));
        result = _add2(
            result, _mul(scaled, _table_gather(indices_1)));
        part = _reduce_quarter(result.hi);
        quadrant = _builder.CreateAdd(quadrant, part.integer);
        result.hi = part.value;
        result = _normalize(result);

        auto *indices_2 = _builder.CreateAdd(indices, _i32(2u));
        auto *indices_3 = _builder.CreateAdd(indices, _i32(3u));
        DoubleFloat tail{
            _table_gather(indices_2), _table_gather(indices_3)};
        result = _normalize(_add2(result, _mul(tail, scaled)));
        result = _mul(
            result,
            DoubleFloat{_f32(6.283185482025146484375),
                        _f32(-1.7484555314695175772e-7)});

        auto *tiny = _builder.CreateFCmpOLT(
            _abs(scaled), _f32(0.7));
        result.hi = _builder.CreateSelect(tiny, scaled, result.hi);
        result.lo = _builder.CreateSelect(tiny, _f32(0.0), result.lo);
        return {result, quadrant};
    }

    void _build_tan(::llvm::Function *function) {
        auto *input = function->getArg(0u);
        input->setName("x");
        auto *moderate = _builder.CreateFCmpOLT(
            _abs(input), _f32(39000.0));
        auto *safe_input = _builder.CreateSelect(
            moderate, input, _f32(0.0));
        auto *q_float = _round_nearest(_builder.CreateFMul(
            safe_input, _f32(0.63661977236758134308)));
        auto *q = _builder.CreateFPToSI(q_float, _int_vector);
        auto *q_as_float = _builder.CreateSIToFP(q, _float_vector);
        auto *reduced = _mla(
            _f32(-1.5703125), q_as_float, safe_input);
        reduced = _mla(
            _f32(-0.00048351287841796875),
            q_as_float, reduced);
        reduced = _mla(
            _f32(-3.1385570764541625977e-7),
            q_as_float, reduced);
        reduced = _mla(
            _f32(-6.077100628276710381e-11),
            q_as_float, reduced);

        auto *all_moderate = _builder.CreateAndReduce(moderate);
        auto *slow = ::llvm::BasicBlock::Create(
            _module.getContext(), "large.reduce", function);
        auto *merge = ::llvm::BasicBlock::Create(
            _module.getContext(), "polynomial", function);
        auto *moderate_block = _builder.GetInsertBlock();
        _builder.CreateCondBr(all_moderate, merge, slow);

        _builder.SetInsertPoint(slow);
        auto large = _reduce_pi(input);
        auto *large_reduced = _builder.CreateFAdd(
            large.value.hi, large.value.lo);
        auto *invalid = _builder.CreateICmpUGE(
            _builder.CreateAnd(
                _float_bits(input), _i32(0x7fffffffu)),
            _i32(0x7f800000u));
        large_reduced = _builder.CreateSelect(
            invalid, _bits_float(_i32(0x7fc00000u)), large_reduced);
        auto *selected_q = _builder.CreateSelect(
            moderate, q, large.quadrant);
        auto *selected_reduced = _builder.CreateSelect(
            moderate, reduced, large_reduced);
        auto *slow_block = _builder.GetInsertBlock();
        _builder.CreateBr(merge);

        _builder.SetInsertPoint(merge);
        auto *q_phi = _builder.CreatePHI(
            _int_vector, 2u, "quadrant");
        q_phi->addIncoming(q, moderate_block);
        q_phi->addIncoming(selected_q, slow_block);
        auto *reduced_phi = _builder.CreatePHI(
            _float_vector, 2u, "reduced");
        reduced_phi->addIncoming(reduced, moderate_block);
        reduced_phi->addIncoming(selected_reduced, slow_block);

        auto *odd = _builder.CreateICmpEQ(
            _builder.CreateAnd(q_phi, _i32(1u)), _i32(1u));
        auto *flip_bits = _builder.CreateSelect(
            odd, _i32(0x80000000u), _i32(0u));
        auto *x = _bits_float(_builder.CreateXor(
            _float_bits(reduced_phi), flip_bits));
        auto *square = _builder.CreateFMul(x, x);
        ::llvm::Value *polynomial = _f32(0.0092724580317735672);
        polynomial = _mla(
            polynomial, square, _f32(0.0033198499586433172));
        polynomial = _mla(
            polynomial, square, _f32(0.024299807846546173));
        polynomial = _mla(
            polynomial, square, _f32(0.053449530154466629));
        polynomial = _mla(
            polynomial, square, _f32(0.13338300585746765));
        polynomial = _mla(
            polynomial, square, _f32(0.33333185315132141));
        auto *result = _mla(
            square, _builder.CreateFMul(polynomial, x), x);
        result = _builder.CreateSelect(
            odd, _builder.CreateFDiv(_f32(1.0), result), result);
        auto *negative_zero = _builder.CreateICmpEQ(
            _float_bits(input), _i32(0x80000000u));
        result = _builder.CreateSelect(negative_zero, input, result);
        _builder.CreateRet(result);
    }

public:
    TrigF32IRBuilder(
        ::llvm::Module &module, ::llvm::Function *function,
        uint32_t width, TrigKind kind)
        : _module{module},
          _builder{module.getContext()},
          _float_vector{::llvm::FixedVectorType::get(
              ::llvm::Type::getFloatTy(module.getContext()), width)},
          _int_vector{::llvm::FixedVectorType::get(
              ::llvm::Type::getInt32Ty(module.getContext()), width)},
          _cosine{kind == TrigKind::cos},
          _tangent{kind == TrigKind::tan} {
        auto *entry = ::llvm::BasicBlock::Create(
            module.getContext(), "entry", function);
        _builder.SetInsertPoint(entry);
    }

    void build(::llvm::Function *function) {
        if (_tangent) {
            _build_tan(function);
            return;
        }
        auto *input = function->getArg(0u);
        input->setName("x");
        auto *absolute = _abs(input);
        auto *moderate = _builder.CreateFCmpOLT(
            absolute, _f32(39000.0));
        auto *safe_input = _builder.CreateSelect(
            moderate, input, _f32(0.0));
        auto *q_float = _cosine ?
            _builder.CreateFAdd(
                _builder.CreateFMul(
                    _round_nearest(_builder.CreateFAdd(
                        _builder.CreateFMul(
                            safe_input,
                            _f32(0.31830988618379067154)),
                        _f32(-0.5))),
                    _f32(2.0)),
                _f32(1.0)) :
            _round_nearest(_builder.CreateFMul(
                safe_input, _f32(0.31830988618379067154)));
        auto *q = _builder.CreateFPToSI(q_float, _int_vector);
        auto *q_as_float = _builder.CreateSIToFP(q, _float_vector);
        auto *reduced = _mla(
            _f32(_cosine ? -1.5703125 : -3.140625),
            q_as_float, safe_input);
        reduced = _mla(
            _f32(_cosine ? -0.00048351287841796875 :
                           -0.0009670257568359375),
            q_as_float, reduced);
        reduced = _mla(
            _f32(_cosine ? -3.1385570764541625977e-7 :
                           -6.2771141529083251953e-7),
            q_as_float, reduced);
        reduced = _mla(
            _f32(_cosine ? -6.077100628276710381e-11 :
                           -1.2154201256553420762e-10),
            q_as_float, reduced);

        auto *all_moderate = _builder.CreateAndReduce(moderate);
        auto *slow = ::llvm::BasicBlock::Create(
            _module.getContext(), "large.reduce", function);
        auto *merge = ::llvm::BasicBlock::Create(
            _module.getContext(), "polynomial", function);
        auto *moderate_block = _builder.GetInsertBlock();
        _builder.CreateCondBr(all_moderate, merge, slow);

        _builder.SetInsertPoint(slow);
        auto large = _reduce_pi(input);
        auto *q_large = _builder.CreateAnd(
            large.quadrant, _i32(3u));
        q_large = _builder.CreateAdd(q_large, q_large);
        auto *positive = _builder.CreateFCmpOGT(
            large.value.hi, _f32(0.0));
        q_large = _builder.CreateAdd(
            q_large, _builder.CreateSelect(
                         positive,
                         _i32(_cosine ? 8u : 2u),
                         _i32(_cosine ? 7u : 1u)));
        q_large = _builder.CreateAShr(
            q_large, _i32(_cosine ? 1u : 2u));

        auto *adjust = _builder.CreateICmpEQ(
            _builder.CreateAnd(large.quadrant, _i32(1u)),
            _i32(_cosine ? 0u : 1u));
        auto *half_pi_sign = _cosine ?
            _builder.CreateSelect(positive, _f32(0.0), _f32(-1.0)) :
            large.value.hi;
        DoubleFloat half_pi{
            _xor_sign(_f32(-1.5707963705062866211), half_pi_sign),
            _xor_sign(_f32(4.3711388286737928865e-8), half_pi_sign)};
        auto adjusted = _add2(large.value, half_pi);
        large.value.hi = _builder.CreateSelect(
            adjust, adjusted.hi, large.value.hi);
        large.value.lo = _builder.CreateSelect(
            adjust, adjusted.lo, large.value.lo);
        auto *large_reduced = _builder.CreateFAdd(
            large.value.hi, large.value.lo);
        auto *invalid = _builder.CreateICmpUGE(
            _builder.CreateAnd(_float_bits(input), _i32(0x7fffffffu)),
            _i32(0x7f800000u));
        large_reduced = _builder.CreateSelect(
            invalid, _bits_float(_i32(0x7fc00000u)), large_reduced);
        auto *selected_q = _builder.CreateSelect(moderate, q, q_large);
        auto *selected_reduced = _builder.CreateSelect(
            moderate, reduced, large_reduced);
        auto *slow_block = _builder.GetInsertBlock();
        _builder.CreateBr(merge);

        _builder.SetInsertPoint(merge);
        auto *q_phi = _builder.CreatePHI(_int_vector, 2u, "quadrant");
        q_phi->addIncoming(q, moderate_block);
        q_phi->addIncoming(selected_q, slow_block);
        ::llvm::Value *reduced_phi = _builder.CreatePHI(
            _float_vector, 2u, "reduced");
        auto *reduced_node = ::llvm::cast<::llvm::PHINode>(reduced_phi);
        reduced_node->addIncoming(reduced, moderate_block);
        reduced_node->addIncoming(selected_reduced, slow_block);

        auto *flip = _builder.CreateICmpEQ(
            _builder.CreateAnd(q_phi, _i32(_cosine ? 2u : 1u)),
            _i32(_cosine ? 0u : 1u));
        auto *flip_bits = _builder.CreateSelect(
            flip, _i32(0x80000000u), _i32(0u));
        reduced_phi = _bits_float(_builder.CreateXor(
            _float_bits(reduced_phi), flip_bits));
        auto *square = _builder.CreateFMul(reduced_phi, reduced_phi);
        ::llvm::Value *polynomial = _f32(2.6083159809786593542e-6);
        polynomial = _mla(
            polynomial, square, _f32(-0.00019810690719168633223));
        polynomial = _mla(
            polynomial, square, _f32(0.0083330785855650901794));
        polynomial = _mla(
            polynomial, square, _f32(-0.16666659712791442871));
        auto *result = _builder.CreateFAdd(
            _builder.CreateFMul(
                square, _builder.CreateFMul(polynomial, reduced_phi)),
            reduced_phi);
        if (!_cosine) {
            auto *negative_zero = _builder.CreateICmpEQ(
                _float_bits(input), _i32(0x80000000u));
            result = _builder.CreateSelect(
                negative_zero, input, result);
        }
        _builder.CreateRet(result);
    }
};

::llvm::GlobalVariable *TrigF32IRBuilder::_reduction_table() {
    static constexpr std::array<uint32_t, 416u> bits{
        0x3e22f980u, 0x335b9390u, 0x2782a540u, 0x9b762a0cu,
        0x3d0be60cu, 0x31dc9c88u, 0x24a94fe0u, 0x191d5f48u,
        0x3d0be60cu, 0x31dc9c88u, 0x24a94fe0u, 0x191d5f48u,
        0x3b3e60dcu, 0xaed8ddf4u, 0xa33580f4u, 0x980a82e1u,
        0x3b3e60dcu, 0xaed8ddf4u, 0xa33580f4u, 0x980a82e1u,
        0x3b3e60dcu, 0xaed8ddf4u, 0xa33580f4u, 0x980a82e1u,
        0x3b3e60dcu, 0xaed8ddf4u, 0xa33580f4u, 0x980a82e1u,
        0x3a79836cu, 0x2f139104u, 0x23a53f84u, 0x17eafa3fu,
        0x3a79836cu, 0x2f139104u, 0x23a53f84u, 0x17eafa3fu,
        0x39f306dcu, 0x2d9c8828u, 0x2294fe14u, 0x96282e0bu,
        0x39660db8u, 0x2d9c8828u, 0x2294fe14u, 0x96282e0bu,
        0x38cc1b70u, 0x2d9c8828u, 0x2294fe14u, 0x96282e0bu,
        0x381836e4u, 0x2c644150u, 0x2127f09cu, 0x15afa3eau,
        0x36c1b724u, 0x2bc882a4u, 0x201fc274u, 0x14be8faau,
        0x36c1b724u, 0x2bc882a4u, 0x201fc274u, 0x14be8faau,
        0x36c1b724u, 0x2bc882a4u, 0x201fc274u, 0x14be8faau,
        0x36036e4cu, 0x2b110548u, 0x201fc274u, 0x14be8faau,
        0x335b9390u, 0x2782a540u, 0x9b762a0cu, 0x0efa9a6fu,
        0x335b9390u, 0x2782a540u, 0x9b762a0cu, 0x0efa9a6fu,
        0x335b9390u, 0x2782a540u, 0x9b762a0cu, 0x0efa9a6fu,
        0x335b9390u, 0x2782a540u, 0x9b762a0cu, 0x0efa9a6fu,
        0x335b9390u, 0x2782a540u, 0x9b762a0cu, 0x0efa9a6fu,
        0x335b9390u, 0x2782a540u, 0x9b762a0cu, 0x0efa9a6fu,
        0x32b72720u, 0x2782a540u, 0x9b762a0cu, 0x0efa9a6fu,
        0x31dc9c88u, 0x24a94fe0u, 0x191d5f48u, 0x8c2cb224u,
        0x31dc9c88u, 0x24a94fe0u, 0x191d5f48u, 0x8c2cb224u,
        0x31393910u, 0x24a94fe0u, 0x191d5f48u, 0x8c2cb224u,
        0x3064e440u, 0x24a94fe0u, 0x191d5f48u, 0x8c2cb224u,
        0x3064e440u, 0x24a94fe0u, 0x191d5f48u, 0x8c2cb224u,
        0x2fc9c880u, 0x24a94fe0u, 0x191d5f48u, 0x8c2cb224u,
        0x2f139104u, 0x23a53f84u, 0x17eafa3cu, 0x0ca9a6eeu,
        0x2d9c8828u, 0x2294fe14u, 0x96282e08u, 0x8b32c890u,
        0x2d9c8828u, 0x2294fe14u, 0x96282e08u, 0x8b32c890u,
        0x2d9c8828u, 0x2294fe14u, 0x96282e08u, 0x8b32c890u,
        0x2c644150u, 0x2127f09cu, 0x15afa3e8u, 0x0a9a6ee0u,
        0x2c644150u, 0x2127f09cu, 0x15afa3e8u, 0x0a9a6ee0u,
        0x2c644150u, 0x2127f09cu, 0x15afa3e8u, 0x0a9a6ee0u,
        0x2bc882a4u, 0x201fc274u, 0x14be8fa8u, 0x09537703u,
        0x2b110548u, 0x201fc274u, 0x14be8fa8u, 0x09537703u,
        0x29882a54u, 0x9b762a0cu, 0x0efa9a6cu, 0x03b81b6cu,
        0x29882a54u, 0x9b762a0cu, 0x0efa9a6cu, 0x03b81b6cu,
        0x29882a54u, 0x9b762a0cu, 0x0efa9a6cu, 0x03b81b6cu,
        0x2782a540u, 0x9b762a0cu, 0x0efa9a6cu, 0x03b81b6cu,
        0x2782a540u, 0x9b762a0cu, 0x0efa9a6cu, 0x03b81b6cu,
        0x2782a540u, 0x9b762a0cu, 0x0efa9a6cu, 0x03b81b6cu,
        0x2782a540u, 0x9b762a0cu, 0x0efa9a6cu, 0x03b81b6cu,
        0x24a94fe0u, 0x191d5f48u, 0x8c2cb224u, 0x0006db15u,
        0x24a94fe0u, 0x191d5f48u, 0x8c2cb224u, 0x0006db15u,
        0x24a94fe0u, 0x191d5f48u, 0x8c2cb224u, 0x0006db15u,
        0x24a94fe0u, 0x191d5f48u, 0x8c2cb224u, 0x0006db15u,
        0x24a94fe0u, 0x191d5f48u, 0x8c2cb224u, 0x0006db15u,
        0x24a94fe0u, 0x191d5f48u, 0x8c2cb224u, 0x0006db15u,
        0x23a53f84u, 0x17eafa3cu, 0x0ca9a6ecu, 0x0181b6c5u,
        0x23a53f84u, 0x17eafa3cu, 0x0ca9a6ecu, 0x0181b6c5u,
        0x2294fe14u, 0x96282e08u, 0x8b32c890u, 0x0006db15u,
        0x2294fe14u, 0x96282e08u, 0x8b32c890u, 0x0006db15u,
        0x2127f09cu, 0x15afa3e8u, 0x0a9a6ee0u, 0x0006db15u,
        0x2127f09cu, 0x15afa3e8u, 0x0a9a6ee0u, 0x0006db15u,
        0x2127f09cu, 0x15afa3e8u, 0x0a9a6ee0u, 0x0006db15u,
        0x201fc274u, 0x14be8fa8u, 0x09537700u, 0x0006db15u,
        0x201fc274u, 0x14be8fa8u, 0x09537700u, 0x0006db15u,
        0x1efe13acu, 0x91382b2cu, 0x8508fc90u, 0x800004ebu,
        0x1efe13acu, 0x91382b2cu, 0x8508fc90u, 0x800004ebu,
        0x1efe13acu, 0x91382b2cu, 0x8508fc90u, 0x800004ebu,
        0x1e7c2758u, 0x91382b2cu, 0x8508fc90u, 0x800004ebu,
        0x1df84eb0u, 0x91382b2cu, 0x8508fc90u, 0x800004ebu,
        0x3d709d5cu, 0x3251f534u, 0x265dc0d8u, 0x1b58a566u,
        0x3ce13abcu, 0x31a3ea68u, 0x265dc0d8u, 0x1b58a566u,
        0x3c42757cu, 0x308fa9a4u, 0x25bb81b4u, 0x1ab14acdu,
        0x3b84eaf8u, 0x308fa9a4u, 0x25bb81b4u, 0x1ab14acdu,
        0x391d5f48u, 0xac2cb224u, 0x1e5b6294u, 0x12cc9e22u,
        0x391d5f48u, 0xac2cb224u, 0x1e5b6294u, 0x12cc9e22u,
        0x391d5f48u, 0xac2cb224u, 0x1e5b6294u, 0x12cc9e22u,
        0x391d5f48u, 0xac2cb224u, 0x1e5b6294u, 0x12cc9e22u,
        0x391d5f48u, 0xac2cb224u, 0x1e5b6294u, 0x12cc9e22u,
        0x37eafa3cu, 0x2ca9a6ecu, 0x2181b6c4u, 0x1615993cu,
        0x37eafa3cu, 0x2ca9a6ecu, 0x2181b6c4u, 0x1615993cu,
        0x37eafa3cu, 0x2ca9a6ecu, 0x2181b6c4u, 0x1615993cu,
        0x3755f47cu, 0x2ba69bb8u, 0x1e5b6294u, 0x12cc9e22u,
        0x36abe8f8u, 0x2ba69bb8u, 0x1e5b6294u, 0x12cc9e22u,
        0x35afa3e8u, 0x2a9a6ee0u, 0x1e5b6294u, 0x12cc9e22u,
        0x35afa3e8u, 0x2a9a6ee0u, 0x1e5b6294u, 0x12cc9e22u,
        0x34be8fa8u, 0x29537700u, 0x1e5b6294u, 0x12cc9e22u,
        0x34be8fa8u, 0x29537700u, 0x1e5b6294u, 0x12cc9e22u,
        0x33fa3ea4u, 0x28a6ee04u, 0x1db6c528u, 0x12cc9e22u,
        0x33fa3ea4u, 0x28a6ee04u, 0x1db6c528u, 0x12cc9e22u,
        0x33747d4cu, 0x279bb818u, 0x1cdb14acu, 0x10c9e21du,
        0x32e8fa98u, 0x279bb818u, 0x1cdb14acu, 0x10c9e21du,
        0x3251f534u, 0x265dc0d8u, 0x1b58a564u, 0x1013c439u,
        0x31a3ea68u, 0x265dc0d8u, 0x1b58a564u, 0x1013c439u,
        0x308fa9a4u, 0x25bb81b4u, 0x1ab14accu, 0x0e9e21c8u,
        0x308fa9a4u, 0x25bb81b4u, 0x1ab14accu, 0x0e9e21c8u,
        0x2efa9a6cu, 0x23b81b6cu, 0x1725664cu, 0x0c443904u,
        0x2efa9a6cu, 0x23b81b6cu, 0x1725664cu, 0x0c443904u,
        0x2efa9a6cu, 0x23b81b6cu, 0x1725664cu, 0x0c443904u,
        0x2efa9a6cu, 0x23b81b6cu, 0x1725664cu, 0x0c443904u,
        0x2e7534dcu, 0x22e06db0u, 0x1725664cu, 0x0c443904u,
        0x2dea69bcu, 0xa17c9274u, 0x95d4cd84u, 0x8ade37dfu,
        0x2d54d374u, 0x2240db60u, 0x1725664cu, 0x0c443904u,
        0x2ca9a6ecu, 0x2181b6c4u, 0x1615993cu, 0x09872084u,
        0x2ba69bb8u, 0x1e5b6294u, 0x12cc9e20u, 0x07641080u,
        0x2ba69bb8u, 0x1e5b6294u, 0x12cc9e20u, 0x07641080u,
        0x2a9a6ee0u, 0x1e5b6294u, 0x12cc9e20u, 0x07641080u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
    };
    constexpr auto name = "__luisa_cpu_native_sin_rempi_f32";
    if (auto *table = _module.getGlobalVariable(name, true)) {
        return table;
    }
    auto *initializer = ::llvm::ConstantDataArray::get(
        _module.getContext(), ::llvm::ArrayRef<uint32_t>{bits});
    auto *table = new ::llvm::GlobalVariable(
        _module, initializer->getType(), true,
        ::llvm::GlobalValue::PrivateLinkage, initializer, name);
    table->setUnnamedAddr(::llvm::GlobalValue::UnnamedAddr::Global);
    table->setAlignment(::llvm::Align{64u});
    return table;
}

enum struct InverseTrigKind : uint8_t {
    asin,
    acos,
    atan,
};

class InverseTrigF32IRBuilder {

private:
    ::llvm::Module &_module;
    ::llvm::IRBuilder<> _builder;
    ::llvm::FixedVectorType *_float_vector;
    ::llvm::FixedVectorType *_int_vector;

private:
    [[nodiscard]] ::llvm::Constant *_f32(double x) const {
        auto *scalar = ::llvm::ConstantFP::get(
            ::llvm::Type::getFloatTy(_module.getContext()), x);
        return ::llvm::ConstantVector::getSplat(
            _float_vector->getElementCount(), scalar);
    }

    [[nodiscard]] ::llvm::Constant *_i32(uint32_t x) const {
        auto *scalar = ::llvm::ConstantInt::get(
            ::llvm::Type::getInt32Ty(_module.getContext()), x);
        return ::llvm::ConstantVector::getSplat(
            _int_vector->getElementCount(), scalar);
    }

    [[nodiscard]] ::llvm::Value *_mla(
        ::llvm::Value *x, ::llvm::Value *y, ::llvm::Value *z) {
        return _builder.CreateFAdd(_builder.CreateFMul(x, y), z);
    }

    [[nodiscard]] ::llvm::Value *_float_bits(::llvm::Value *x) {
        return _builder.CreateBitCast(x, _int_vector);
    }

    [[nodiscard]] ::llvm::Value *_bits_float(::llvm::Value *x) {
        return _builder.CreateBitCast(x, _float_vector);
    }

    [[nodiscard]] ::llvm::Value *_abs(::llvm::Value *x) {
        return _bits_float(_builder.CreateAnd(
            _float_bits(x), _i32(0x7fffffffu)));
    }

    [[nodiscard]] ::llvm::Value *_copy_sign(
        ::llvm::Value *magnitude, ::llvm::Value *sign) {
        return _bits_float(_builder.CreateOr(
            _builder.CreateAnd(
                _float_bits(magnitude), _i32(0x7fffffffu)),
            _builder.CreateAnd(
                _float_bits(sign), _i32(0x80000000u))));
    }

    [[nodiscard]] ::llvm::Value *_asin_polynomial(
        ::llvm::Value *x2) {
        ::llvm::Value *u = _f32(0.04197454825);
        u = _mla(u, x2, _f32(0.02424046025));
        u = _mla(u, x2, _f32(0.04547423869));
        u = _mla(u, x2, _f32(0.07495029271));
        return _mla(u, x2, _f32(0.1666677296));
    }

    void _build_asin_acos(
        ::llvm::Function *function, bool cosine) {
        auto *input = function->getArg(0u);
        input->setName("x");
        auto *absolute = _abs(input);
        auto *small = _builder.CreateFCmpOLT(
            absolute, _f32(0.5));
        auto *x2 = _builder.CreateSelect(
            small, _builder.CreateFMul(input, input),
            _builder.CreateFMul(
                _builder.CreateFSub(_f32(1.0), absolute),
                _f32(0.5)));
        auto *root = _builder.CreateUnaryIntrinsic(
            ::llvm::Intrinsic::sqrt, x2);
        auto *x = _builder.CreateSelect(small, absolute, root);
        if (cosine) {
            x = _builder.CreateSelect(
                _builder.CreateFCmpOEQ(absolute, _f32(1.0)),
                _f32(0.0), x);
        }
        auto *u = _builder.CreateFMul(
            _asin_polynomial(x2),
            _builder.CreateFMul(x, x2));
        ::llvm::Value *result;
        if (cosine) {
            auto *signed_x = _copy_sign(x, input);
            auto *signed_u = _copy_sign(u, input);
            auto *middle = _builder.CreateFSub(
                _f32(1.5707963267948966192),
                _builder.CreateFAdd(signed_x, signed_u));
            auto *edge = _builder.CreateFMul(
                _builder.CreateFAdd(x, u), _f32(2.0));
            result = _builder.CreateSelect(small, middle, edge);
            result = _builder.CreateSelect(
                _builder.CreateAnd(
                    _builder.CreateNot(small),
                    _builder.CreateFCmpOLT(input, _f32(0.0))),
                _builder.CreateFSub(
                    _f32(3.1415926535897932385), result),
                result);
        } else {
            auto *approximation = _builder.CreateFAdd(x, u);
            auto *edge = _mla(
                approximation, _f32(-2.0),
                _f32(1.5707963267948966192));
            result = _builder.CreateSelect(
                small, approximation, edge);
            result = _copy_sign(result, input);
        }
        _builder.CreateRet(result);
    }

    void _build_atan(::llvm::Function *function) {
        auto *input = function->getArg(0u);
        input->setName("x");
        auto *absolute = _abs(input);
        auto *large = _builder.CreateFCmpOGT(
            absolute, _f32(1.0));
        auto *s = _builder.CreateSelect(
            large,
            _builder.CreateFDiv(_f32(1.0), absolute), absolute);
        auto *t = _builder.CreateFMul(s, s);
        constexpr std::array coefficients{
            0.0028236389625817537308,
            -0.015956902876496315002,
            0.042504988610744476318,
            -0.074890092015266418457,
            0.10634793341159820557,
            -0.14202736318111419678,
            0.19992695748805999756,
            -0.33333101868629455566,
        };
        ::llvm::Value *u = _f32(coefficients.front());
        for (auto i = size_t{1u}; i < coefficients.size(); i++) {
            u = _mla(u, t, _f32(coefficients[i]));
        }
        auto *result = _mla(
            s, _builder.CreateFMul(t, u), s);
        result = _builder.CreateSelect(
            large,
            _builder.CreateFSub(
                _f32(1.5707963267948966192), result),
            result);
        result = _copy_sign(result, input);
        _builder.CreateRet(result);
    }

public:
    InverseTrigF32IRBuilder(
        ::llvm::Module &module, ::llvm::Function *function,
        uint32_t width)
        : _module{module},
          _builder{module.getContext()},
          _float_vector{::llvm::FixedVectorType::get(
              ::llvm::Type::getFloatTy(module.getContext()), width)},
          _int_vector{::llvm::FixedVectorType::get(
              ::llvm::Type::getInt32Ty(module.getContext()), width)} {
        auto *entry = ::llvm::BasicBlock::Create(
            module.getContext(), "entry", function);
        _builder.SetInsertPoint(entry);
    }

    void build(::llvm::Function *function, InverseTrigKind kind) {
        switch (kind) {
            case InverseTrigKind::asin:
                _build_asin_acos(function, false);
                break;
            case InverseTrigKind::acos:
                _build_asin_acos(function, true);
                break;
            case InverseTrigKind::atan:
                _build_atan(function);
                break;
        }
    }
};

class ExpLogF32IRBuilder {

private:
    ::llvm::Module &_module;
    ::llvm::IRBuilder<> _builder;
    ::llvm::FixedVectorType *_float_vector;
    ::llvm::FixedVectorType *_int_vector;

private:
    [[nodiscard]] ::llvm::Constant *_f32(double x) const {
        auto *scalar = ::llvm::ConstantFP::get(
            ::llvm::Type::getFloatTy(_module.getContext()), x);
        return ::llvm::ConstantVector::getSplat(
            _float_vector->getElementCount(), scalar);
    }

    [[nodiscard]] ::llvm::Constant *_i32(uint32_t x) const {
        auto *scalar = ::llvm::ConstantInt::get(
            ::llvm::Type::getInt32Ty(_module.getContext()), x);
        return ::llvm::ConstantVector::getSplat(
            _int_vector->getElementCount(), scalar);
    }

    [[nodiscard]] ::llvm::Value *_mla(
        ::llvm::Value *x, ::llvm::Value *y, ::llvm::Value *z) {
        return _builder.CreateFAdd(_builder.CreateFMul(x, y), z);
    }

    [[nodiscard]] ::llvm::Value *_float_bits(::llvm::Value *x) {
        return _builder.CreateBitCast(x, _int_vector);
    }

    [[nodiscard]] ::llvm::Value *_bits_float(::llvm::Value *x) {
        return _builder.CreateBitCast(x, _float_vector);
    }

    [[nodiscard]] ::llvm::Value *_round_nearest(::llvm::Value *x) {
        auto *sign = _builder.CreateAnd(
            _float_bits(x), _i32(0x80000000u));
        auto *bias = _bits_float(_builder.CreateOr(
            _i32(0x4b000000u), sign));
        return _builder.CreateFSub(
            _builder.CreateFAdd(x, bias), bias);
    }

    [[nodiscard]] ::llvm::Value *_pow2i(::llvm::Value *exponent) {
        return _bits_float(_builder.CreateShl(
            _builder.CreateAdd(exponent, _i32(127u)), _i32(23u)));
    }

    [[nodiscard]] ::llvm::Value *_ldexp2(
        ::llvm::Value *value, ::llvm::Value *exponent) {
        auto *half = _builder.CreateAShr(exponent, _i32(1u));
        value = _builder.CreateFMul(value, _pow2i(half));
        return _builder.CreateFMul(
            value,
            _pow2i(_builder.CreateSub(exponent, half)));
    }

    void _build_exp(::llvm::Function *function) {
        auto *input = function->getArg(0u);
        input->setName("x");
        auto *safe_range = _builder.CreateAnd(
            _builder.CreateFCmpOGE(input, _f32(-104.0)),
            _builder.CreateFCmpOLE(input, _f32(100.0)));
        auto *safe = _builder.CreateSelect(
            safe_range, input, _f32(0.0));
        auto *q_float = _round_nearest(_builder.CreateFMul(
            safe, _f32(1.4426950408889634074)));
        auto *q = _builder.CreateFPToSI(q_float, _int_vector);
        auto *q_as_float = _builder.CreateSIToFP(q, _float_vector);
        auto *reduced = _mla(
            q_as_float, _f32(-0.693145751953125), safe);
        reduced = _mla(
            q_as_float, _f32(-1.428606765330187045e-6), reduced);

        ::llvm::Value *polynomial =
            _f32(0.00019852761761285364628);
        polynomial = _mla(
            polynomial, reduced,
            _f32(0.0013930435525253415108));
        polynomial = _mla(
            polynomial, reduced,
            _f32(0.0083333607763051986694));
        polynomial = _mla(
            polynomial, reduced,
            _f32(0.041666485369205474854));
        polynomial = _mla(
            polynomial, reduced,
            _f32(0.16666667163372039795));
        polynomial = _mla(polynomial, reduced, _f32(0.5));
        auto *result = _builder.CreateFAdd(
            _f32(1.0),
            _builder.CreateFAdd(
                reduced,
                _builder.CreateFMul(
                    _builder.CreateFMul(reduced, reduced),
                    polynomial)));
        result = _ldexp2(result, q);
        result = _builder.CreateSelect(
            _builder.CreateFCmpOLT(input, _f32(-104.0)),
            _f32(0.0), result);
        result = _builder.CreateSelect(
            _builder.CreateFCmpOGT(input, _f32(100.0)),
            _bits_float(_i32(0x7f800000u)), result);
        result = _builder.CreateSelect(
            _builder.CreateFCmpUNO(input, input),
            _bits_float(_i32(0x7fc00000u)), result);
        _builder.CreateRet(result);
    }

    void _build_log(::llvm::Function *function) {
        auto *input = function->getArg(0u);
        input->setName("x");
        auto *subnormal = _builder.CreateAnd(
            _builder.CreateFCmpOGT(input, _f32(0.0)),
            _builder.CreateFCmpOLT(
                input, _f32(1.175494350822287508e-38)));
        auto *scaled = _builder.CreateSelect(
            subnormal,
            _builder.CreateFMul(
                input, _f32(18446744073709551616.0)),
            input);
        auto *probe = _builder.CreateFMul(
            scaled, _f32(1.3333333333333333333));
        auto *exponent = _builder.CreateSub(
            _builder.CreateAnd(
                _builder.CreateLShr(_float_bits(probe), _i32(23u)),
                _i32(0xffu)),
            _i32(127u));
        auto *mantissa = _bits_float(_builder.CreateAdd(
            _float_bits(scaled),
            _builder.CreateShl(
                _builder.CreateNeg(exponent), _i32(23u))));
        auto *log_exponent = _builder.CreateSelect(
            subnormal,
            _builder.CreateSub(exponent, _i32(64u)), exponent);

        auto *x = _builder.CreateFDiv(
            _builder.CreateFSub(mantissa, _f32(1.0)),
            _builder.CreateFAdd(_f32(1.0), mantissa));
        auto *x2 = _builder.CreateFMul(x, x);
        ::llvm::Value *polynomial = _f32(0.23928284645080566406);
        polynomial = _mla(
            polynomial, x2, _f32(0.28518211841583251953));
        polynomial = _mla(
            polynomial, x2, _f32(0.40000587701797485352));
        polynomial = _mla(
            polynomial, x2, _f32(0.6666666865348815918));
        polynomial = _mla(polynomial, x2, _f32(2.0));
        auto *result = _mla(
            x, polynomial,
            _builder.CreateFMul(
                _f32(0.69314718055994528623),
                _builder.CreateSIToFP(
                    log_exponent, _float_vector)));
        result = _builder.CreateSelect(
            _builder.CreateFCmpOEQ(input, _f32(0.0)),
            _bits_float(_i32(0xff800000u)), result);
        result = _builder.CreateSelect(
            _builder.CreateFCmpOEQ(
                input, _bits_float(_i32(0x7f800000u))),
            _bits_float(_i32(0x7f800000u)), result);
        result = _builder.CreateSelect(
            _builder.CreateOr(
                _builder.CreateFCmpOLT(input, _f32(0.0)),
                _builder.CreateFCmpUNO(input, input)),
            _bits_float(_i32(0x7fc00000u)), result);
        _builder.CreateRet(result);
    }

public:
    ExpLogF32IRBuilder(
        ::llvm::Module &module, ::llvm::Function *function,
        uint32_t width)
        : _module{module},
          _builder{module.getContext()},
          _float_vector{::llvm::FixedVectorType::get(
              ::llvm::Type::getFloatTy(module.getContext()), width)},
          _int_vector{::llvm::FixedVectorType::get(
              ::llvm::Type::getInt32Ty(module.getContext()), width)} {
        auto *entry = ::llvm::BasicBlock::Create(
            module.getContext(), "entry", function);
        _builder.SetInsertPoint(entry);
    }

    void build(::llvm::Function *function, bool logarithm) {
        if (logarithm) {
            _build_log(function);
        } else {
            _build_exp(function);
        }
    }
};

}// namespace

[[nodiscard]] static ::llvm::Value *emit_trig_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode, bool cosine) {
    auto *type = ::llvm::dyn_cast<::llvm::FixedVectorType>(
        vector == nullptr ? nullptr : vector->getType());
    if (type == nullptr || !type->getElementType()->isFloatTy()) {
        return nullptr;
    }
    auto width = type->getNumElements();
    auto suffix = mode == LLVMNativeMathMode::fast ? "fast" : "u10";
    auto name = std::string{"__luisa_cpu_native_"} +
                (cosine ? "cos" : "sin") + "_f32_v" +
                std::to_string(width) + "_" + suffix;
    auto *function = module.getFunction(name);
    if (function == nullptr) {
        auto *function_type = ::llvm::FunctionType::get(
            type, {type}, false);
        function = ::llvm::Function::Create(
            function_type, ::llvm::GlobalValue::InternalLinkage,
            name, module);
        function->addFnAttr(::llvm::Attribute::AlwaysInline);
        function->addFnAttr(::llvm::Attribute::NoUnwind);
        function->addFnAttr(::llvm::Attribute::NoRecurse);
        function->addFnAttr(::llvm::Attribute::WillReturn);
        function->addFnAttr("luisa.cpu.native_math");
        function->setOnlyReadsMemory();
        // The initial fast entry deliberately shares the audited u10 body.
        // A separately bounded approximation can replace it without changing
        // the provider ABI once ShaderOption plumbing is complete.
        TrigF32IRBuilder{
            module, function, width,
            cosine ? TrigKind::cos : TrigKind::sin}
            .build(function);
    }
    return builder.CreateCall(
        function, {vector}, cosine ? "native.cos" : "native.sin");
}

[[nodiscard]] static ::llvm::Value *emit_native_tan_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    auto *type = ::llvm::dyn_cast<::llvm::FixedVectorType>(
        vector == nullptr ? nullptr : vector->getType());
    if (type == nullptr || !type->getElementType()->isFloatTy()) {
        return nullptr;
    }
    auto width = type->getNumElements();
    auto suffix = mode == LLVMNativeMathMode::fast ? "fast" : "u35";
    auto name = std::string{"__luisa_cpu_native_tan_f32_v"} +
                std::to_string(width) + "_" + suffix;
    auto *function = module.getFunction(name);
    if (function == nullptr) {
        auto *function_type = ::llvm::FunctionType::get(
            type, {type}, false);
        function = ::llvm::Function::Create(
            function_type, ::llvm::GlobalValue::InternalLinkage,
            name, module);
        function->addFnAttr(::llvm::Attribute::AlwaysInline);
        function->addFnAttr(::llvm::Attribute::NoUnwind);
        function->addFnAttr(::llvm::Attribute::NoRecurse);
        function->addFnAttr(::llvm::Attribute::WillReturn);
        function->addFnAttr("luisa.cpu.native_math");
        function->setOnlyReadsMemory();
        TrigF32IRBuilder{module, function, width, TrigKind::tan}
            .build(function);
    }
    return builder.CreateCall(function, {vector}, "native.tan");
}

[[nodiscard]] static ::llvm::Value *emit_inverse_trig_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode,
    InverseTrigKind kind) {
    auto *type = ::llvm::dyn_cast<::llvm::FixedVectorType>(
        vector == nullptr ? nullptr : vector->getType());
    if (type == nullptr || !type->getElementType()->isFloatTy()) {
        return nullptr;
    }
    auto operation = kind == InverseTrigKind::asin ? "asin" :
                     kind == InverseTrigKind::acos ? "acos" : "atan";
    auto width = type->getNumElements();
    auto suffix = mode == LLVMNativeMathMode::fast ? "fast" : "u35";
    auto name = std::string{"__luisa_cpu_native_"} + operation +
                "_f32_v" + std::to_string(width) + "_" + suffix;
    auto *function = module.getFunction(name);
    if (function == nullptr) {
        auto *function_type = ::llvm::FunctionType::get(
            type, {type}, false);
        function = ::llvm::Function::Create(
            function_type, ::llvm::GlobalValue::InternalLinkage,
            name, module);
        function->addFnAttr(::llvm::Attribute::AlwaysInline);
        function->addFnAttr(::llvm::Attribute::NoUnwind);
        function->addFnAttr(::llvm::Attribute::NoRecurse);
        function->addFnAttr(::llvm::Attribute::WillReturn);
        function->addFnAttr("luisa.cpu.native_math");
        function->setDoesNotAccessMemory();
        InverseTrigF32IRBuilder{module, function, width}.build(
            function, kind);
    }
    return builder.CreateCall(
        function, {vector},
        std::string{"native."} + operation);
}

[[nodiscard]] static ::llvm::Value *emit_exp_log_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode, bool logarithm) {
    auto *type = ::llvm::dyn_cast<::llvm::FixedVectorType>(
        vector == nullptr ? nullptr : vector->getType());
    if (type == nullptr || !type->getElementType()->isFloatTy()) {
        return nullptr;
    }
    auto width = type->getNumElements();
    auto suffix = mode == LLVMNativeMathMode::fast ?
        "fast" : logarithm ? "u35" : "u10";
    auto name = std::string{"__luisa_cpu_native_"} +
                (logarithm ? "log" : "exp") + "_f32_v" +
                std::to_string(width) + "_" + suffix;
    auto *function = module.getFunction(name);
    if (function == nullptr) {
        auto *function_type = ::llvm::FunctionType::get(
            type, {type}, false);
        function = ::llvm::Function::Create(
            function_type, ::llvm::GlobalValue::InternalLinkage,
            name, module);
        function->addFnAttr(::llvm::Attribute::AlwaysInline);
        function->addFnAttr(::llvm::Attribute::NoUnwind);
        function->addFnAttr(::llvm::Attribute::NoRecurse);
        function->addFnAttr(::llvm::Attribute::WillReturn);
        function->addFnAttr("luisa.cpu.native_math");
        function->setDoesNotAccessMemory();
        ExpLogF32IRBuilder{module, function, width}.build(
            function, logarithm);
    }
    return builder.CreateCall(
        function, {vector}, logarithm ? "native.log" : "native.exp");
}

::llvm::Value *LLVMNativeMath::emit_sin_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_trig_f32(module, builder, vector, mode, false);
}

::llvm::Value *LLVMNativeMath::emit_cos_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_trig_f32(module, builder, vector, mode, true);
}

::llvm::Value *LLVMNativeMath::emit_tan_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_native_tan_f32(module, builder, vector, mode);
}

::llvm::Value *LLVMNativeMath::emit_asin_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_inverse_trig_f32(
        module, builder, vector, mode, InverseTrigKind::asin);
}

::llvm::Value *LLVMNativeMath::emit_acos_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_inverse_trig_f32(
        module, builder, vector, mode, InverseTrigKind::acos);
}

::llvm::Value *LLVMNativeMath::emit_atan_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_inverse_trig_f32(
        module, builder, vector, mode, InverseTrigKind::atan);
}

::llvm::Value *LLVMNativeMath::emit_exp_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_exp_log_f32(module, builder, vector, mode, false);
}

::llvm::Value *LLVMNativeMath::emit_log_f32(
    ::llvm::Module &module, ::llvm::IRBuilder<> &builder,
    ::llvm::Value *vector, LLVMNativeMathMode mode) {
    return emit_exp_log_f32(module, builder, vector, mode, true);
}

}// namespace luisa::compute::cpu
