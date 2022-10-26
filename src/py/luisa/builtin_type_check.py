from .types import *
import ast


class BasicTypeCheck:
    @staticmethod
    def is_arithmetic(dtype):
        return dtype in [int, float, uint]

    @staticmethod
    def is_bool(dtype):
        return dtype == bool

    @staticmethod
    def is_float(dtype):
        return dtype == float

    @staticmethod
    def is_scalar(dtype):
        return dtype in scalar_dtypes

    @staticmethod
    def is_vector(dtype):
        return dtype in vector_dtypes

    @staticmethod
    def is_matrix(dtype):
        return dtype in matrix_dtypes

    @staticmethod
    def is_basic(dtype):
        return dtype in basic_dtypes

    @staticmethod
    def is_integer(dtype):
        return dtype in [int, uint]

    @staticmethod
    def same_shape(dtype1, dtype2):
        if dtype1 in scalar_dtypes and dtype2 in scalar_dtypes:
            return True
        elif dtype1 in vector_dtypes and dtype2 in vector_dtypes:
            return length_of(dtype1) == length_of(dtype2)
        elif dtype1 in matrix_dtypes and dtype2 in matrix_dtypes:
            return length_of(dtype1) == length_of(dtype2)
        else:
            return False


class BinaryTypeInfer:
    def __call__(self, lhs_dtype, rhs_dtype, op):
        op_name = op.__name__.lower()
        checker = f"is_legal_{op_name}"
        try:
            return getattr(self, checker)(lhs_dtype, rhs_dtype, op)
        except AttributeError:
            print(f"Operator {op} not supported.")

    @staticmethod
    def get_coerced_arithmetic_dtype(lhs_dtype, rhs_dtype):
        if bool in [lhs_dtype, rhs_dtype]:
            raise TypeError("Type `bool` is not allowed in arithmetic operations.")
        if float in [lhs_dtype, rhs_dtype]:
            return float
        elif int in [lhs_dtype, rhs_dtype]:
            return int
        else:
            return uint

    @staticmethod
    def with_type(dtype, inner_dtype):
        if TC.is_scalar(dtype):
            return inner_dtype
        elif TC.is_vector(dtype):
            return vector(inner_dtype, length_of(dtype))
        elif TC.is_matrix(dtype):
            if inner_dtype is not float:
                raise TypeError("Matrix inner type must be float.")
            return dtype
        else:
            raise TypeError(f"Unknown type `{dtype}`.")

    # either of a and b is arithmetic
    # a and b are of the same size, inner type is arithmetic
    # returns coerced arithmetic type
    #
    # This applies to Add, Sub, Div, FloorDiv, Pow(this is a builtin function in luisa)
    @staticmethod
    def broadcast_arithmetic_op(lhs_dtype, rhs_dtype, op):
        if TC.is_arithmetic(lhs_dtype) and TC.is_arithmetic(element_of(rhs_dtype)):
            coerced_inner_type = BinaryTypeInfer.get_coerced_arithmetic_dtype(lhs_dtype, element_of(rhs_dtype))
            return BinaryTypeInfer.with_type(rhs_dtype, coerced_inner_type)
        elif TC.is_arithmetic(rhs_dtype) and TC.is_arithmetic(element_of(lhs_dtype)):
            coerced_inner_type = BinaryTypeInfer.get_coerced_arithmetic_dtype(rhs_dtype, element_of(lhs_dtype))
            return BinaryTypeInfer.with_type(lhs_dtype, coerced_inner_type)
        elif TC.same_shape(lhs_dtype, rhs_dtype) and \
                TC.is_arithmetic(element_of(lhs_dtype)) and TC.is_arithmetic(element_of(rhs_dtype)):
            coerced_inner_type = BinaryTypeInfer.get_coerced_arithmetic_dtype(element_of(lhs_dtype),
                                                                             element_of(rhs_dtype))
            return BinaryTypeInfer.with_type(lhs_dtype, coerced_inner_type)
        else:
            raise TypeError(f"Operator `{op}` between type `{lhs_dtype}` and `{rhs_dtype}` is not supported.")

    # either of a and b is arithmetic
    # a and b are of the same size, inner type is arithmetic
    # returns bool
    #
    # This applies to Lt, Gt, LtE, GtE
    @staticmethod
    def broadcast_compare_op(lhs_dtype, rhs_dtype, op):
        if TC.is_arithmetic(lhs_dtype) and TC.is_arithmetic(element_of(rhs_dtype)):
            return BinaryTypeInfer.with_type(rhs_dtype, bool)
        elif TC.is_arithmetic(rhs_dtype) and TC.is_arithmetic(element_of(lhs_dtype)):
            return BinaryTypeInfer.with_type(lhs_dtype, bool)
        elif TC.same_shape(lhs_dtype, rhs_dtype) and \
                TC.is_arithmetic(element_of(lhs_dtype)) and TC.is_arithmetic(element_of(rhs_dtype)):
            return BinaryTypeInfer.with_type(lhs_dtype, bool)
        else:
            raise TypeError(f"Operator `{op}` between type `{lhs_dtype}` and `{rhs_dtype}` is not supported.")

    # either of a and b is scalar (bool / arithmetic)
    # a and b are of the same size, inner type is scalar (bool / arithmetic)
    # returns bool
    #
    # This applies to Eq, NotEq
    @staticmethod
    def broadcast_eq_op(lhs_dtype, rhs_dtype, op):
        if TC.is_bool(lhs_dtype) and TC.is_bool(element_of(rhs_dtype)) or \
                TC.is_arithmetic(lhs_dtype) and TC.is_arithmetic(element_of(rhs_dtype)):
            return BinaryTypeInfer.with_type(rhs_dtype, bool)
        elif TC.is_bool(rhs_dtype) and TC.is_bool(element_of(lhs_dtype)) or \
                TC.is_arithmetic(rhs_dtype) and TC.is_arithmetic(element_of(lhs_dtype)):
            return BinaryTypeInfer.with_type(lhs_dtype, bool)
        elif TC.same_shape(lhs_dtype, rhs_dtype):
            if TC.is_bool(element_of(lhs_dtype)) and TC.is_bool(element_of(rhs_dtype)) or \
                    TC.is_arithmetic(element_of(lhs_dtype)) and TC.is_arithmetic(element_of(rhs_dtype)):
                return BinaryTypeInfer.with_type(lhs_dtype, bool)
        else:
            raise TypeError(f"Operator `{op}` between type `{lhs_dtype}` and `{rhs_dtype}` is not supported.")

    # either of a and b is integer && the other's inner type is integer
    # a and b are of the same size && inner type is integer
    # returns integer
    #
    # This applies to Mod, BitAnd, BitOr, BitXor, LShift, RShift
    # TODO: Distinguish between int and uint
    @staticmethod
    def broadcast_integer_op(lhs_dtype, rhs_dtype, op):
        if op is ast.LShift or op is ast.RShift:
            print("Luisa type check now regards int and uint as the same type, which makes invoking LShift and RShift "
                  "on negative numbers will not trigger a TypeError, resulting in potential bugs.")
            print("This behaviour will be fixed in the future.")
        if TC.is_integer(lhs_dtype) and TC.is_integer(element_of(rhs_dtype)):
            coerced_inner_type = BinaryTypeInfer.get_coerced_arithmetic_dtype(lhs_dtype, element_of(rhs_dtype))
            return BinaryTypeInfer.with_type(rhs_dtype, coerced_inner_type)
        elif TC.is_integer(rhs_dtype) and TC.is_integer(element_of(lhs_dtype)):
            coerced_inner_type = BinaryTypeInfer.get_coerced_arithmetic_dtype(rhs_dtype, element_of(lhs_dtype))
            return BinaryTypeInfer.with_type(lhs_dtype, coerced_inner_type)
        elif TC.same_shape(lhs_dtype, rhs_dtype) and \
                TC.is_integer(element_of(lhs_dtype)) and \
                TC.is_integer(element_of(rhs_dtype)):
            coerced_inner_type = BinaryTypeInfer.get_coerced_arithmetic_dtype(element_of(lhs_dtype),
                                                                              element_of(rhs_dtype))
            return BinaryTypeInfer.with_type(lhs_dtype, coerced_inner_type)
        else:
            raise TypeError(f"Operator `{op}` between type `{lhs_dtype}` and `{rhs_dtype}` is not supported.")

    # This applies to And, Or, Xor
    @staticmethod
    def broadcast_bool_op(lhs_dtype, rhs_dtype, op):
        if TC.is_bool(lhs_dtype) and TC.is_bool(element_of(rhs_dtype)):
            return BinaryTypeInfer.with_type(rhs_dtype, bool)
        elif TC.is_bool(rhs_dtype) and TC.is_bool(element_of(lhs_dtype)):
            return BinaryTypeInfer.with_type(lhs_dtype, bool)
        elif TC.same_shape(lhs_dtype, rhs_dtype) and \
                TC.is_bool(element_of(lhs_dtype)) and TC.is_bool(element_of(rhs_dtype)):
            return BinaryTypeInfer.with_type(lhs_dtype, bool)
        else:
            raise TypeError(f"Operator `{op}` between type `{lhs_dtype}` and `{rhs_dtype}` is not supported.")

    @staticmethod
    def is_legal_add(lhs_dtype, rhs_dtype, op):
        return BinaryTypeInfer.broadcast_arithmetic_op(lhs_dtype, rhs_dtype, op)

    @staticmethod
    def is_legal_sub(lhs_dtype, rhs_dtype, op):
        return BinaryTypeInfer.broadcast_arithmetic_op(lhs_dtype, rhs_dtype, op)

    # Besides _broadcast_arithmetic_op, mult also supports Matrix * Vector -> Vector
    @staticmethod
    def is_legal_mult(lhs_dtype, rhs_dtype, op):
        if TC.is_arithmetic(lhs_dtype) and TC.is_arithmetic(element_of(rhs_dtype)):
            coerced_inner_type = BinaryTypeInfer.get_coerced_arithmetic_dtype(lhs_dtype, element_of(rhs_dtype))
            return BinaryTypeInfer.with_type(rhs_dtype, coerced_inner_type)
        elif TC.is_arithmetic(rhs_dtype) and TC.is_arithmetic(element_of(lhs_dtype)):
            coerced_inner_type = BinaryTypeInfer.get_coerced_arithmetic_dtype(rhs_dtype, element_of(lhs_dtype))
            return BinaryTypeInfer.with_type(lhs_dtype, coerced_inner_type)
        elif TC.same_shape(lhs_dtype, rhs_dtype) and \
                TC.is_arithmetic(element_of(lhs_dtype)) and TC.is_arithmetic(element_of(rhs_dtype)):
            coerced_inner_type = BinaryTypeInfer.get_coerced_arithmetic_dtype(element_of(lhs_dtype),
                                                                             element_of(rhs_dtype))
            return BinaryTypeInfer.with_type(lhs_dtype, coerced_inner_type)
        elif TC.is_matrix(lhs_dtype) and TC.is_vector(rhs_dtype) and length_of(lhs_dtype) == length_of(rhs_dtype) and \
                TC.is_arithmetic(element_of(lhs_dtype)) and TC.is_arithmetic(element_of(rhs_dtype)):
            coerced_inner_type = BinaryTypeInfer.get_coerced_arithmetic_dtype(element_of(lhs_dtype),
                                                                             element_of(rhs_dtype))
            return BinaryTypeInfer.with_type(rhs_dtype, coerced_inner_type)
        else:
            raise TypeError(f"Operator `{op}` between type `{lhs_dtype}` and `{rhs_dtype}` is not supported.")

    @staticmethod
    def is_legal_div(lhs_dtype, rhs_dtype, op):
        return BinaryTypeInfer.broadcast_arithmetic_op(lhs_dtype, rhs_dtype, op)

    @staticmethod
    def is_legal_floordiv(lhs_dtype, rhs_dtype, op):
        return BinaryTypeInfer.broadcast_arithmetic_op(lhs_dtype, rhs_dtype, op)

    @staticmethod
    def is_legal_mod(lhs_dtype, rhs_dtype, op):
        return BinaryTypeInfer.broadcast_integer_op(lhs_dtype, rhs_dtype, op)

    @staticmethod
    def is_legal_lshift(lhs_dtype, rhs_dtype, op):
        return BinaryTypeInfer.broadcast_integer_op(lhs_dtype, rhs_dtype, op)

    @staticmethod
    def is_legal_rshift(lhs_dtype, rhs_dtype, op):
        return BinaryTypeInfer.broadcast_integer_op(lhs_dtype, rhs_dtype, op)

    @staticmethod
    def is_legal_pow(lhs_dtype, rhs_dtype, op):
        return BinaryTypeInfer.broadcast_arithmetic_op(lhs_dtype, rhs_dtype, op)

    @staticmethod
    def is_legal_and(lhs_dtype, rhs_dtype, op):
        return BinaryTypeInfer.broadcast_bool_op(lhs_dtype, rhs_dtype, op)

    @staticmethod
    def is_legal_or(lhs_dtype, rhs_dtype, op):
        return BinaryTypeInfer.broadcast_bool_op(lhs_dtype, rhs_dtype, op)

    @staticmethod
    def is_legal_bitor(lhs_dtype, rhs_dtype, op):
        return BinaryTypeInfer.broadcast_integer_op(lhs_dtype, rhs_dtype, op)

    @staticmethod
    def is_legal_bitxor(lhs_dtype, rhs_dtype, op):
        return BinaryTypeInfer.broadcast_integer_op(lhs_dtype, rhs_dtype, op)

    @staticmethod
    def is_legal_bitand(lhs_dtype, rhs_dtype, op):
        return BinaryTypeInfer.broadcast_integer_op(lhs_dtype, rhs_dtype, op)

    @staticmethod
    def is_legal_eq(lhs_dtype, rhs_dtype, op):
        return BinaryTypeInfer.broadcast_eq_op(lhs_dtype, rhs_dtype, op)

    @staticmethod
    def is_legal_noteq(lhs_dtype, rhs_dtype, op):
        return BinaryTypeInfer.broadcast_eq_op(lhs_dtype, rhs_dtype, op)

    @staticmethod
    def is_legal_lt(lhs_dtype, rhs_dtype, op):
        return BinaryTypeInfer.broadcast_compare_op(lhs_dtype, rhs_dtype, op)

    @staticmethod
    def is_legal_gt(lhs_dtype, rhs_dtype, op):
        return BinaryTypeInfer.broadcast_compare_op(lhs_dtype, rhs_dtype, op)

    @staticmethod
    def is_legal_lte(lhs_dtype, rhs_dtype, op):
        return BinaryTypeInfer.broadcast_compare_op(lhs_dtype, rhs_dtype, op)

    @staticmethod
    def is_legal_gte(lhs_dtype, rhs_dtype, op):
        return BinaryTypeInfer.broadcast_compare_op(lhs_dtype, rhs_dtype, op)


TC = BasicTypeCheck()
binary_type_infer = BinaryTypeInfer()
