
import os
import copy

'''
function build in names:
$RET$ $FUNC_NAME$ $FUNC_SIG_ARGS$
$FUNC_IMPL_ARGS$ 
$CLASS_NAME$
$INNER_FUNC_NAME$
$INNER_FUNC_INPUT_ARGS$

$INNER_FUNC_NAME$
$USER_FUNC_BODY$
$RETURN_CONVERT$
$RETURN_CAST$
$INNER_CLASS_NAME$
e.g. func_sig_template: 
$RET$ $FUNC_NAME$($FUNC_SIG_ARGS$);

e.g. func_impl_template:
$RET$ $CLASS_NAME$::$FUNC_NAME$($FUNC_IMPL_ARGS$){
    $USER_FUNC_BODY$
    return $RETURN_CONVERT$($INNER_FUNC_NAME$($INNER_FUNC_INPUT_ARGS$));
}
'''

class Util:
    @staticmethod
    def Replace(str:str, pairs = [(str,str)]):
        for pair in pairs:
            str = str.replace(pair[0], pair[1])
        return str

class ListMaker:
    def __init__(self, sep = ", "):
        self.buf:str=""
        self.sep = sep
    def push(self, s:str):
        if(s==""): return
        if(self.buf == ""):
            self.buf += s
        else:
            self.buf += self.sep + s
    def get(self):
        return self.buf

class Arg:
    def clone(self, type = "", name="", default_value = "", cast = "", user_convert="", auto_fill = ""):
        new = copy.deepcopy(self)
        if(type != ""): new.type = type
        if(name != ""): new.name = name
        if(default_value != ""): new.default_value = default_value
        if(cast != ""): new.cast = cast
        if(user_convert != ""): new.user_convert = user_convert
        if(auto_fill != ""): new.auto_fill = auto_fill
        return new
    
    def __init__(self, type = "", name="", default_value = "", cast = "", user_convert="", auto_fill = ""):
        self.type = type
        self.name = name
        self.default_value = default_value
        self.cast = cast
        self.auto_fill = auto_fill
        self.user_convert = user_convert
        self.return_type = False
    
    def func_sig_arg(self):
        if(self.auto_fill != ""):
            return ""
        buf = f"{self.type}  {self.name}"
        if(self.default_value != ""):
            buf += f" = {self.default_value}"
        return buf
    
    def func_impl_arg(self):
        if(self.auto_fill != ""): return ""
        buf = f"{self.type}  {self.name}"
        return buf
    
    def inner_func_input_arg(self):
        if(self.auto_fill != ""):
            return self.auto_fill
        buf = f"{self.name}"
        if(self.user_convert != ""):
            buf = f"{self.user_convert}({buf})"
        if(self.cast != ""):
            buf = f"({self.cast}) {buf}"
        return buf

def Ret(type = "void", user_convert = ""):
    a = Arg(type)
    a.return_type = True
    a.user_convert = user_convert
    return a

def arg_lists(args:list[Arg], pred):
    list_maker = ListMaker(", ")
    for arg in args:
        if(not arg.return_type):
            sig_arg = pred(arg)
            list_maker.push(sig_arg)
    return list_maker.get()

def func_sig_args(arg_list:list[Arg]):
    return arg_lists(arg_list, Arg.func_sig_arg)

def func_impl_args(arg_list:list[Arg]):
    return arg_lists(arg_list, Arg.func_impl_arg)

def inner_func_input_args(arg_list:list[Arg]):
    return arg_lists(arg_list, Arg.inner_func_input_arg)

def func_ret(arg:Arg):
    if(arg.return_type):
        return arg.type
    else:
        raise("Not a return type")

class Func:
    default_func_sig_template = '''$RET$ $FUNC_NAME$($FUNC_SIG_ARGS$);'''
    default_func_impl_template = '''$RET$ $CLASS_NAME$::$FUNC_NAME$($FUNC_IMPL_ARGS$)
{
    return $INNER_CLASS_NAME$::$INNER_FUNC_NAME$($INNER_FUNC_INPUT_ARGS$);
}'''
    def __init__(self, name:str,ret:Arg, args:list[Arg] = [],inner_func_name = "", user_func_body = ""):
        self.Class:Class = Class("")
        self.name:str = name
        self.args:list[Arg] = args
        self.ret:Arg = ret
        self.inner_func_name:str = inner_func_name if inner_func_name != "" else name
        self.user_func_body:str = user_func_body
        self.func_sig_template:str = Func.default_func_sig_template
        self.func_impl_template:str = Func.default_func_impl_template
    
    def set_template(self, template):
        self.func_sig_template = template.func_sig_template
        self.func_impl_template = template.func_impl_template
    
    def func_sig(self):
        return Util.Replace(self.func_sig_template,
                            [
                                ("$RET$", func_ret(self.ret)), 
                                ("$FUNC_NAME$", self.name), 
                                ("$FUNC_SIG_ARGS$", func_sig_args(self.args)),
                                ("$CLASS_NAME$", self.Class.name),
                                ("$FUNC_IMPL_ARGS$", func_impl_args(self.args)),
                                ("$USER_FUNC_BODY$", self.user_func_body),
                                ("$RETURN_CONVERT$", self.ret.user_convert),
                                ("$INNER_FUNC_NAME$", self.inner_func_name),
                                ("$INNER_FUNC_INPUT_ARGS$", inner_func_input_args(self.args)),
                                ("$INNER_CLASS_NAME$", self.Class.inner_class_name),
                                ("$RETURN_CAST$", self.ret.cast)
                            ])
    
    def func_impl(self):
        return Util.Replace(self.func_impl_template,
                    [
                        ("$RET$", func_ret(self.ret)), 
                        ("$FUNC_NAME$", self.name), 
                        ("$CLASS_NAME$", self.Class.name),
                        ("$FUNC_IMPL_ARGS$", func_impl_args(self.args)),
                        ("$USER_FUNC_BODY$", self.user_func_body),
                        ("$RETURN_CONVERT$", self.ret.user_convert),
                        ("$INNER_FUNC_NAME$", self.inner_func_name),
                        ("$INNER_FUNC_INPUT_ARGS$", inner_func_input_args(self.args)),
                        ("$INNER_CLASS_NAME$", self.Class.inner_class_name),
                        ("$RETURN_CAST$", self.ret.cast)
                    ])
    
    def instantiate(self, type_pairs:list[(str,str)]):
        # deep copy itself to a new func
        new_func = copy.deepcopy(self)
        new_func.name = Util.Replace(new_func.name, type_pairs)
        new_func.inner_func_name = Util.Replace(new_func.inner_func_name, type_pairs)
        new_func.user_func_body = Util.Replace(new_func.user_func_body, type_pairs)
        new_func.ret.type = Util.Replace(new_func.ret.type, type_pairs)
        new_func.ret.cast = Util.Replace(new_func.ret.cast, type_pairs)
        new_func.ret.user_convert = Util.Replace(new_func.ret.user_convert, type_pairs)
        
        for arg in new_func.args:
            arg.type = Util.Replace(arg.type, type_pairs)
            arg.cast = Util.Replace(arg.cast, type_pairs)
            arg.user_convert = Util.Replace(arg.user_convert, type_pairs)
            arg.auto_fill = Util.Replace(arg.auto_fill, type_pairs)
            arg.default_value = Util.Replace(arg.default_value, type_pairs)
            arg.name = Util.Replace(arg.name, type_pairs)
        return new_func
        
    
    def rename(self, name:str, inner_func_name:str = ""):
        new_func = copy.deepcopy(self)
        new_func.name = name
        new_func.inner_func_name = inner_func_name if inner_func_name != "" else name
        return new_func
    
class Class:
    default_header_template = '''// This file is generated by $FILE_NAME$.py
#pragma once

namespace $NAME_SPACE${

class $CLASS_NAME${
public:
$FUNC_SIGS$
};
}
'''

    default_src_template = ''' // This file is generated by $FILE_NAME$.py
#include "$FILE_NAME$.h"

namespace $NAME_SPACE${
$FUNC_IMPLS$
}
'''
    def __init__(self, name:str, inner_class_name:str="", 
                 namespace:str = "",
                 header_template = default_header_template, 
                 src_template = default_src_template):
        self.name:str = name
        self.namespace:str = namespace
        self.inner_class_name:str = inner_class_name if inner_class_name != "" else name
        self.funcs:list[Func] = []
        self.header_template = Class.default_header_template
        self.src_template = Class.default_src_template
    
    def set_template(self, template):
        self.header_template = template.header_template
        self.src_template = template.src_template
    
    def add_func(self, func:Func):
        func.Class = self
        self.funcs.append(func)
    
    def add_funcs(self, funcs:list[Func]):
        for func in funcs:
            self.add_func(func)
    
    def print_func_sigs(self):
        for func in self.funcs:
            print(func.func_sig())
    
    def print_func_impls(self):
        for func in self.funcs:
            print(func.func_impl())
    
    def print_funcs(self):
        for func in self.funcs:
            print(func.func_sig())
            print(func.func_impl())
    
    def __make_func_sigs(func_sigs):
        return "\n\n".join([func_sig for func_sig in func_sigs])

    def __make_func_impls(func_impls):
        return "\n\n".join([func_impl for func_impl in func_impls])
    
    def func_sigs(self):
        return Class.__make_func_sigs([func.func_sig() for func in self.funcs])
    
    def func_impls(self):
        return Class.__make_func_impls([func.func_impl() for func in self.funcs])
    
    def __make_file_name(CamelCaseName):
        first = True
        # seperate the name by uppercase
        # eg. AaBb -> Aa_Bb
        file_name = ""
        for c in CamelCaseName:
            if c.isupper() and not first:
                file_name += '_'
            file_name += c
            first = False
        # to lower case
        file_name = file_name.lower()
        return file_name
    
    def __write_file(file_name, file):
        f = open(file_name, "w")
        f.write(file)
        f.close()
    
    def write(self, file_name="", header_ext=".h", src_ext=".cpp", folder="", print_to_console=False):
        if(file_name == ""):
            file_name = Class.__make_file_name(self.name)
        header_file_name = folder + file_name + header_ext
        src_file_name = folder + file_name + src_ext
        
        header_file_content = Util.Replace(self.header_template, 
                                           [
                                               ("$CLASS_NAME$", self.name), 
                                               ("$NAME_SPACE$", self.namespace), 
                                               ("$FUNC_SIGS$", self.func_sigs()),
                                               ("$FILE_NAME$", file_name)
                                            ])
        
        src_file_content = Util.Replace(self.src_template,
                                        [
                                            ("$CLASS_NAME$", self.name),
                                            ("$NAME_SPACE$", self.namespace),
                                            ("$FUNC_IMPLS$", self.func_impls()),
                                            ("$FILE_NAME$", file_name)
                                        ])
        
        if(print_to_console):
            print(header_file_content)
            print(src_file_content)
        else:
            if(header_file_content!=""):
                Class.__write_file(header_file_name, header_file_content)
            if(src_file_content!=""):
                Class.__write_file(src_file_name, src_file_content)
