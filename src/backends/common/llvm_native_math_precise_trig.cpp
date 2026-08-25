#include "llvm_native_math_internal.h"

#include <llvm/IR/GlobalVariable.h>

// The precise trigonometric range reduction and approximations are adapted
// from SLEEF's single-precision implementations (Copyright Naoki Shibata and
// contributors 2010-2025, Boost Software License 1.0). See LICENSE.SLEEF.txt.

namespace luisa::compute::cpu {

namespace {

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

    [[nodiscard]] ::llvm::GlobalVariable *_reduction_table() {
        return detail::get_trig_reduction_table(_module);
    }

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

}// namespace

namespace detail {

void build_precise_trig_f32(
    ::llvm::Module &module, ::llvm::Function *function, uint32_t width,
    NativeTrigKind kind) {
    auto precise_kind = kind == NativeTrigKind::sin ? TrigKind::sin :
                        kind == NativeTrigKind::cos ? TrigKind::cos :
                                                      TrigKind::tan;
    TrigF32IRBuilder{module, function, width, precise_kind}.build(function);
}

}// namespace detail

}// namespace luisa::compute::cpu
