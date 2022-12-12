import ast
from . import lcapi
import numpy
from types import SimpleNamespace
from .astbuilder import build
from .buffer import Buffer
from .types import dtype_of, to_lctype, from_lctype
from .builtin import wrap_with_tmp_var
from struct import pack, unpack

class Printer:
    def __init__(self, capacity = 2**16):
        self.capacity = capacity
        self.buffer = Buffer(size = capacity, dtype = int)
        # buffer layout: [tag, content..., tag, content..., ..., cursor]
        # tag is dtype or string const
        # content is all elements converted to int for expr, or empty for string const
        # cursor is next position available in buffer
        self.taglist = [] # stores tags
        self.reset()

    def reset(self):
        # clear buffer to zeros
        arr = numpy.zeros(self.capacity, dtype='int32')
        self.buffer.copy_from(arr)


    @staticmethod
    def get_expr_elements_count(dtype):
        if dtype is str:
            return 0
        lctype = to_lctype(dtype)
        if lctype.is_scalar():
            return 1
        if lctype.is_vector():
            return lctype.dimension()
        if lctype.is_matrix():
            return lctype.dimension()**2
        if lctype.is_array():
            return dtype.size * Printer.get_expr_elements_count(dtype.dtype)
        if lctype.is_structure():
            return sum([Printer.get_expr_elements_count(t) for t in dtype.membertype])
        raise Exception(f"print type {dtype} not supported")

    # return a list of elements (int-typed expressions)
    @staticmethod
    def get_expr_elements(dtype, expr):
        if dtype is str:
            return []
        lctype = to_lctype(dtype)
        # scalar types
        if dtype is int:
            return [expr]
        if dtype is bool:
            tmp = lcapi.builder().local(to_lctype(bool))
            lcapi.builder().assign(tmp, expr)
            return [lcapi.builder().cast(to_lctype(int), lcapi.CastOp.STATIC, tmp)]
        if dtype is float:
            tmp = lcapi.builder().local(to_lctype(float))
            lcapi.builder().assign(tmp, expr)
            return [lcapi.builder().cast(to_lctype(int), lcapi.CastOp.BITWISE, tmp)]
        assert not lctype.is_scalar()
        # vector & matrix
        if lctype.is_vector():
            res = []
            for idx in range(lctype.dimension()):
                idxexpr = lcapi.builder().literal(to_lctype(int), idx)
                element = lcapi.builder().access(lctype.element(), expr, idxexpr)
                res += Printer.get_expr_elements(from_lctype(lctype.element()), element)
            return res
        if lctype.is_matrix():
            column_lctype = lcapi.Type.from_(f"vector<float,{lctype.dimension()}>")
            res = []
            for idx in range(lctype.dimension()):
                idxexpr = lcapi.builder().literal(to_lctype(int), idx)
                column = lcapi.builder().access(column_lctype, expr, idxexpr)
                for idy in range(lctype.dimension()):
                    idyexpr = lcapi.builder().literal(to_lctype(int), idy)
                    element = lcapi.builder().access(lctype.element(), column, idyexpr)
                    res += Printer.get_expr_elements(from_lctype(lctype.element()), element)
            return res
        if lctype.is_array():
            res = []
            for idx in range(lctype.dimension()):
                idxexpr = lcapi.builder().literal(to_lctype(int), idx)
                element = lcapi.builder().access(lctype.element(), expr, idxexpr)
                res += Printer.get_expr_elements(dtype.dtype, element)
            return res
        if lctype.is_structure():
            res = []
            for idx, element_type in enumerate(dtype.membertype):
                element = lcapi.builder().member(to_lctype(element_type), expr, idx)
                res += Printer.get_expr_elements(element_type, element)
            return res
        raise Exception(f"print type {dtype} not supported")

    def get_tag_id(self, tag):
        if not tag in self.taglist:
            self.taglist.append(tag)
        return self.taglist.index(tag)

    def buffer_write(self, offset, expr): # offset is also expr
        lcapi.builder().call(lcapi.CallOp.BUFFER_WRITE, [self.buffer_expr, offset, expr])

    @staticmethod
    def addint(expr, k):
        kexpr = lcapi.builder().literal(to_lctype(int), k)
        return lcapi.builder().binary(to_lctype(int), lcapi.BinaryOp.ADD, expr, kexpr)

    def kernel_print(self, argnodes, sep=' ', end='\n'):
        # collect all nodes to print
        def tmp_str_node(text):
            obj = SimpleNamespace()
            obj.dtype = str
            obj.expr = text
            return obj
        elements = []
        for idx, x in enumerate(argnodes):
            if idx > 0:
                elements.append(tmp_str_node(sep))
            if hasattr(x, "joined"):
                elements += x.joined
            else:
                elements.append(x)
        elements.append(tmp_str_node(end))
        # get buffer and compute size to be used
        self.buffer_expr = lcapi.builder().buffer_binding(to_lctype(dtype_of(self.buffer)), self.buffer.handle, 0, self.buffer.bytesize)
        def intexpr(k):
            return lcapi.builder().literal(to_lctype(int), k)
        count = sum([1 + self.get_expr_elements_count(x.dtype) for x in elements]) # need to store tag+value for each element
        # reserve size of buffer to be used
        access_expr = lcapi.builder().access(to_lctype(int), self.buffer_expr, intexpr(self.capacity-1))
        tmp = lcapi.builder().call(to_lctype(int), lcapi.CallOp.ATOMIC_FETCH_ADD, [access_expr, intexpr(count)])
        start_pos = lcapi.builder().local(to_lctype(int))
        lcapi.builder().assign(start_pos, tmp)
        offset = 0
        if count >= 1023:
            raise ValueError("Printing too many elements!")
        # safeguard character: tag=-1
        # if start_pos > capacity-1024:
        cond1 = lcapi.builder().binary(to_lctype(bool), lcapi.BinaryOp.GREATER, start_pos, intexpr(self.capacity-1024))
        ifstmt1 = lcapi.builder().if_(cond1)
        with ifstmt1.true_branch():
            # if start_pos < capacity-1:
            cond2 = lcapi.builder().binary(to_lctype(bool), lcapi.BinaryOp.LESS, start_pos, intexpr(self.capacity-1))
            with lcapi.builder().if_(cond2).true_branch():
                self.buffer_write(start_pos, intexpr(-1))
        # if start_pos <= capacity-1024:
        with ifstmt1.false_branch():
            # write to buffer
            for idx, x in enumerate(elements):
                # write element
                self.buffer_write(self.addint(start_pos, offset), intexpr(self.get_tag_id(x.dtype if x.dtype != str else x.expr)))
                offset += 1
                # workaround: ISPC can't cast r-value; so creating a temporary variable
                if x.dtype != str:
                    wrap_with_tmp_var(x)
                elements = self.get_expr_elements(x.dtype, x.expr)
                assert len(elements) == self.get_expr_elements_count(x.dtype)
                for casted_expr in elements:
                    self.buffer_write(self.addint(start_pos, offset), casted_expr)
                    offset += 1
            assert offset == count

    # recover original data from stored buffer
    @staticmethod
    def recover(dtype, arr, idx):
        if dtype is int:
            return int(arr[idx])
        if dtype is float:
            b = pack('i', arr[idx])
            return unpack('f', b)[0]
        if dtype is bool:
            return bool(arr[idx])
        lctype = to_lctype(dtype)
        # vector
        if lctype.is_vector():
            res = []
            for i in range(lctype.dimension()):
                res.append(Printer.recover(from_lctype(lctype.element()), arr, i+idx))
            return dtype(*res)
        # matrix
        if lctype.is_matrix():
            res = []
            for i in range(lctype.dimension()**2):
                res.append(Printer.recover(from_lctype(lctype.element()), arr, i+idx))
            return dtype(*res)
        # array
        if lctype.is_array():
            res = []
            for i in range(dtype.size):
                res.append(Printer.recover(dtype.dtype, arr, idx))
                idx += Printer.get_expr_elements_count(dtype.dtype)
            return dtype(res)
        # struct
        if lctype.is_structure():
            res = {}
            for name in dtype.idx_dict:
                mtype = dtype.membertype[dtype.idx_dict[name]]
                res[name] = Printer.recover(mtype, arr, idx)
                idx += Printer.get_expr_elements_count(mtype)
            return dtype(**res)
        raise Exception("recovering data of unsupported dtype")

    def final_print(self):
        arr = numpy.zeros(self.capacity, dtype='int32')
        self.buffer.copy_to(arr, sync=True)
        idx = 0
        while idx < arr[-1]:
            if arr[idx] == -1:
                print("[Print buffer out of capacity]")
                return
            else:
                tag = self.taglist[arr[idx]]
                idx += 1
                if type(tag) is str:
                    print(tag, end="")
                    if tag == "[ABORT]":
                        quit()
                else:
                    print(self.recover(tag, arr, idx), end="")
                    idx += self.get_expr_elements_count(tag)
