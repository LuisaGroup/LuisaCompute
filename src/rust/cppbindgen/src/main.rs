extern crate proc_macro;
extern crate syn;

use proc_macro2::TokenTree;
use std::{fs::File, io::Write, path::Path};
use syn::__private::ToTokens;

fn camel_case_to_snake_case(name: &str) -> String {
    let keywords = [
        "if", "else", "for", "while", "switch", "case", "default", "const", "struct", "bool",
        "return", "break", "continue", "true", "false", "int", "float", "double", "void",
    ];
    let mut result = String::new();
    for c in name.chars() {
        if c.is_uppercase() {
            if !result.is_empty() {
                result.push('_');
            }
            result.push(c.to_ascii_lowercase());
        } else {
            result.push(c);
        }
    }
    if keywords.contains(&result.as_str()) {
        result.push('_');
    }
    result
}

fn gen_span(class: &str, out_name: &str, fname: &str, ty_s: &str) -> (String, String) {
    use std::fmt::Write;
    let mut def = String::new();
    writeln!(
        def,
        "luisa::span<const {}> {}::{}() const noexcept {{",
        ty_s, class, out_name
    )
    .unwrap();
    writeln!(
        def,
        "    return {{reinterpret_cast<const {1} *>(_inner.{0}.ptr), _inner.{0}.len}};",
        fname, ty_s
    )
    .unwrap();
    writeln!(def, "}}").unwrap();
    let decl = format!(
        "    [[nodiscard]] luisa::span<const {}> {}() const noexcept;",
        ty_s, out_name
    );
    (decl, def)
}

fn has_repr_c(attrs: &[syn::Attribute]) -> bool {
    let mut has_repr_c = false;
    for attr in attrs {
        let meta = &attr.meta;
        match meta {
            syn::Meta::List(list) => {
                let path = &list.path;
                if path.is_ident("repr") {
                    for tok in list.tokens.clone().into_iter() {
                        match tok {
                            TokenTree::Ident(ident) => {
                                if ident == "C" {
                                    has_repr_c = true;
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            _ => {}
        }
    }
    has_repr_c
}

fn unwrap_generic_param(ty: &syn::Type) -> String {
    let ty_s = ty.to_token_stream().to_string();
    let i = ty_s.find('<').unwrap();
    let j = ty_s.rfind('>').unwrap();
    let t = ty_s[i + 1..j]
        .trim()
        .to_string()
        .replace(" < ", "<")
        .replace(" >", ">");
    convert_primitive_type_name(t.as_str())
}

fn convert_primitive_type_name(ty_s: &str) -> String {
    if ty_s == "bool" {
        return "bool".to_string();
    }
    if ty_s == "usize" {
        return "size_t".to_string();
    }
    if ty_s == "isize" {
        return "ssize_t".to_string();
    }
    if ty_s == "f32" {
        return "float".to_string();
    }
    if ty_s == "f64" {
        return "double".to_string();
    }
    if ty_s == "i8" {
        return "int8_t".to_string();
    }
    if ty_s == "i16" {
        return "int16_t".to_string();
    }
    if ty_s == "i32" {
        return "int32_t".to_string();
    }
    if ty_s == "i64" {
        return "int64_t".to_string();
    }
    if ty_s == "u8" {
        return "uint8_t".to_string();
    }
    if ty_s == "u16" {
        return "uint16_t".to_string();
    }
    if ty_s == "u32" {
        return "uint32_t".to_string();
    }
    if ty_s == "u64" {
        return "uint64_t".to_string();
    }
    ty_s.to_string()
}

fn gen_type(ty: &syn::Type) -> String {
    if let syn::Type::Array(a) = ty {
        return format!(
            "std::array<{}, {}>",
            gen_type(&a.elem),
            a.len.to_token_stream().to_string()
        );
    }
    let ty_s = ty
        .to_token_stream()
        .to_string()
        .replace(" < ", "<")
        .replace(" >", ">")
        .trim()
        .to_string();
    convert_primitive_type_name(ty_s.as_str())
}

fn gen_field(class: &str, out_name: &str, fname: &str, field: &syn::Field) -> (String, String) {
    let ty_s = field.ty.to_token_stream().to_string();
    let fname = &camel_case_to_snake_case(fname);
    let out_name = &camel_case_to_snake_case(out_name);
    if ty_s.starts_with("CBoxedSlice") {
        return gen_span(class, out_name, fname, &unwrap_generic_param(&field.ty));
    }
    (
        format!(
            "    [[nodiscard]] const {} &{}() const noexcept;",
            gen_type(&field.ty),
            out_name
        ),
        format!(
            "const {} &{}::{}() const noexcept {{ return detail::from_inner_ref(_inner.{}); }}",
            gen_type(&field.ty),
            class,
            out_name,
            fname
        ),
    )
}

fn gen_struct_binding(
    item: &syn::ItemStruct,
    fwd: &mut File,
    h: &mut File,
    cpp: &mut File,
) -> std::io::Result<()> {
    if !has_repr_c(&item.attrs) {
        return Ok(());
    }

    let name = item.ident.to_string();
    let exclude = ["CpuCustomOp", "CallableModuleRef", "UserData"];
    if exclude.contains(&name.as_str()) {
        return Ok(());
    }
    writeln!(fwd, "class {};", name)?;
    writeln!(
        h,
        "class LC_IR_API {}{} {{",
        name,
        if name.ends_with("Ref") {
            ""
        } else {
            " : concepts::Noncopyable"
        }
    )?;
    writeln!(h, "    raw::{} _inner{{}};\n", name)?;
    writeln!(h, "public:")?;
    writeln!(h, "    friend class IrBuilder;")?;
    if !name.ends_with("Ref") {
        writeln!(
            h,
            "    [[nodiscard]] auto raw() noexcept {{ return &_inner; }}"
        )?;
        writeln!(
            h,
            "    [[nodiscard]] auto raw() const noexcept {{ return &_inner; }}"
        )?;
    }
    let is_tuple = item.fields.iter().all(|f| match &f.ident {
        Some(_) => false,
        None => true,
    });
    if !is_tuple {
        for f in &item.fields {
            let fname = f.ident.as_ref().unwrap().to_string();
            let (decl, def) = gen_field(&name, &fname, &fname, &f);
            writeln!(h, "{}", decl)?;
            writeln!(cpp, "{}", def)?;
        }
    }
    {
        let extra_code = File::open(Path::new(&format!("data/{}.h", name)));
        if let Ok(mut extra_code) = extra_code {
            writeln!(h, "\n    // including extra code from data/{}.h", name)?;
            std::io::copy(&mut extra_code, h)?;
            writeln!(h, "\n    // end include")?;
        }
    }
    {
        let extra_code = File::open(Path::new(&format!("data/{}.cpp", name)));
        if let Ok(mut extra_code) = extra_code {
            writeln!(cpp, "\n// including extra code from data/{}.cpp", name)?;
            std::io::copy(&mut extra_code, cpp)?;
            writeln!(cpp, "\n// end include\n")?;
        }
    }
    writeln!(h, "}};")?;
    writeln!(h, "{}", gen_specialization(&name))?;
    Ok(())
}

fn gen_specialization(name: &str) -> String {
    format!(
        "
namespace detail {{
template<>
struct FromInnerRef<raw::{0}> {{
    using Output = {0};
    static const Output &from(const raw::{0} &_inner) noexcept {{
        return reinterpret_cast<const Output &>(_inner);
    }}
}};
template<>
struct FromInnerRef<CArc<raw::{0}>> {{
    using Output = CArc<{0}>;
    static const Output &from(const CArc<raw::{0}> &_inner) noexcept {{
        return reinterpret_cast<const Output &>(_inner);
    }}
}};
template<>
struct FromInnerRef<Pooled<raw::{0}>> {{
    using Output = Pooled<{0}>;
    static const Output &from(const Pooled<raw::{0}> &_inner) noexcept {{
        return reinterpret_cast<const Output &>(_inner);
    }}
}};
template<>
struct FromInnerRef<CBoxedSlice<raw::{0}>> {{
    using Output = CBoxedSlice<{0}>;
    static const Output &from(const CBoxedSlice<raw::{0}> &_inner) noexcept {{
        return reinterpret_cast<const Output &>(_inner);
    }}
}};
}}//namespace detail
",
        name
    )
}

fn gen_enum_binding(
    item: &syn::ItemEnum,
    fwd: &mut File,
    h: &mut File,
    cpp: &mut File,
) -> std::io::Result<()> {
    if !has_repr_c(&item.attrs) {
        return Ok(());
    }
    let name = item.ident.to_string();
    let exclude = ["Usage", "UsageMark"];
    if exclude.contains(&name.as_str()) {
        return Ok(());
    }
    // let recognized = [
    //     "Func",
    //     "Instruction",
    //     "Primitive",
    //     "Type",
    //     "Const",
    //     "VectorElementType",
    // ];
    // if !recognized.contains(&name.as_str()) {
    //     return Ok(());
    // }
    let has_body = item
        .variants
        .iter()
        .map(|v| match &v.fields {
            syn::Fields::Unit => false,
            _ => true,
        })
        .collect::<Vec<_>>();
    if has_body.iter().all(|&x| !x) {
        writeln!(h, "using raw::{};", name)?;
        return Ok(());
    }
    writeln!(fwd, "class {};", name)?;
    writeln!(h, "class LC_IR_API {0} : concepts::Noncopyable {{", name)?;
    writeln!(h, "    raw::{} _inner{{}};", name)?;
    writeln!(h, "    class Marker {{}};\n")?;
    writeln!(h, "public:")?;
    writeln!(h, "    friend class IrBuilder;")?;
    writeln!(h, "    using Tag = raw::{}::Tag;", name)?;

    for (i, variant) in item.variants.iter().enumerate() {
        let variant_name = variant.ident.to_string();
        let has_body = has_body[i];
        if has_body {
            writeln!(
                h,
                "    class LC_IR_API {} : Marker, concepts::Noncopyable {{",
                variant_name
            )?;
            writeln!(
                h,
                "        raw::{}::{}_Body _inner{{}};",
                name, variant_name
            )?;
            writeln!(h, "    public:")?;
            writeln!(
                h,
                "        static constexpr Tag tag() noexcept {{ return raw::{}::Tag::{}; }}",
                name, variant_name
            )?;
            writeln!(
                h,
                "        [[nodiscard]] auto raw() const noexcept {{ return &_inner; }}"
            )?;
            match &variant.fields {
                syn::Fields::Named(ref fields) => {
                    for field in &fields.named {
                        let fname = field.ident.as_ref().unwrap().to_string();
                        let (decl, def) = gen_field(
                            &format!("{}::{}", name, variant_name),
                            &fname,
                            &fname,
                            &field,
                        );
                        writeln!(h, "    {}", decl)?;
                        writeln!(cpp, "{}", def)?;
                    }
                }
                _ => {}
            }
            writeln!(h, "    }};")?;
        } else {
            writeln!(
                h,
                "    class LC_IR_API {} : Marker, concepts::Noncopyable {{",
                variant_name
            )?;
            writeln!(h, "        uint8_t _pad;")?;
            writeln!(h, "    public:")?;
            writeln!(
                h,
                "        static constexpr Tag tag() noexcept {{ return raw::{}::Tag::{}; }}",
                name, variant_name
            )?;
            writeln!(h, "    }};")?;
            writeln!(
                h,
                "    explicit {0}({0}::{1} _) noexcept {{ _inner.tag = {1}::tag(); }}",
                name, variant_name
            )?;
        }
    }
    writeln!(h, "public:")?;
    writeln!(
        h,
        "    [[nodiscard]] auto tag() const noexcept {{ return _inner.tag; }}"
    )?;
    writeln!(
        h,
        "    [[nodiscard]] auto raw() const noexcept {{ return &_inner; }}"
    )?;
    writeln!(
        h,
        "    template<class T>\n    [[nodiscard]] bool isa() const noexcept {{"
    )?;
    writeln!(h, "        static_assert(std::is_base_of_v<Marker, T>);")?;
    writeln!(h, "        return _inner.tag == T::tag();")?;
    writeln!(h, "    }}")?;
    writeln!(
        h,
        "    template<class T>\n    [[nodiscard]] const T *as() const noexcept {{"
    )?;
    writeln!(h, "        static_assert(std::is_base_of_v<Marker, T>);")?;
    writeln!(h, "        if (!isa<T>()) return nullptr;")?;
    for (i, variant) in item.variants.iter().enumerate() {
        let variant_name = variant.ident.to_string();
        let has_body = has_body[i];
        if has_body {
            writeln!(
                h,
                "        if constexpr (std::is_same_v<T, {}>) {{",
                variant_name
            )?;
            writeln!(
                h,
                "            return reinterpret_cast<const {} *>(&_inner.{});",
                variant_name,
                camel_case_to_snake_case(&variant_name)
            )?;
            writeln!(h, "        }}")?;
        }
    }
    writeln!(h, "        return reinterpret_cast<const T *>(this);")?;
    writeln!(h, "    }}")?;
    writeln!(h, "}};")?;
    writeln!(
        h,
        "static_assert(sizeof({}) == sizeof(raw::{}));",
        name, name
    )?;
    writeln!(h, "{}", gen_specialization(&name))?;
    Ok(())
}

fn gen_cpp_binding(file: syn::File, fwd: &mut File, h: &mut File, cpp: &mut File) {
    for item in &file.items {
        match item {
            syn::Item::Enum(e) => {
                gen_enum_binding(e, fwd, h, cpp).unwrap();
            }
            syn::Item::Struct(s) => gen_struct_binding(s, fwd, h, cpp).unwrap(),
            _ => {}
        }
    }
}

fn run_clang_format(path: &str) {
    use std::process::Command;
    Command::new("clang-format")
        .arg("-i")
        .arg(path)
        .output()
        .unwrap();
}

fn main() -> std::io::Result<()> {
    let source = include_str!("../../luisa_compute_ir/src/ir.rs");
    let file = syn::parse_file(source).unwrap();
    let mut cpp = std::fs::File::create("../../ir/ir.cpp")?;
    let mut h = std::fs::File::create("../../../include/luisa/ir/ir.h")?;
    let mut fwd: File = std::fs::File::create("../../../include/luisa/ir/fwd.h")?;
    writeln!(fwd, "#pragma once\n")?;
    writeln!(fwd, "#include <luisa/core/dll_export.h>")?;
    writeln!(fwd, "#include <luisa/core/stl/memory.h>// for span")?;
    writeln!(fwd, "#include <luisa/core/concepts.h>// for Noncopyable")?;
    writeln!(fwd, "#include <luisa/rust/ir.hpp>\n")?;
    writeln!(
        fwd,
        "{}",
        r#"// deduction guide for CSlice
namespace luisa::compute::ir {
template<typename T>
CSlice(T *, size_t) -> CSlice<T>;
template<typename T>
CSlice(const T *, size_t) -> CSlice<T>;
}// namespace luisa::compute::ir
"#
    )?;
    writeln!(fwd, "namespace luisa::compute::ir_v2 {{")?;
    writeln!(fwd, "namespace raw = luisa::compute::ir;")?;
    writeln!(
        fwd,
        "{}",
        r#"using raw::CArc;
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
"#
    )?;
    writeln!(h, "#pragma once\n")?;
    writeln!(h, "#include <luisa/ir/fwd.h>\n")?;
    writeln!(h, "namespace luisa::compute::ir_v2 {{")?;
    writeln!(
        cpp,
        r#"#include <luisa/ir/ir.h>

namespace luisa::compute::ir_v2 {{
"#
    )?;
    gen_cpp_binding(file, &mut fwd, &mut h, &mut cpp);
    writeln!(&mut h, "}}// namespace luisa::compute::ir_v2")?;
    writeln!(&mut cpp, "}}// namespace luisa::compute::ir_v2")?;
    writeln!(&mut fwd, "\n}}// namespace luisa::compute::ir_v2")?;
    run_clang_format("../../../include/luisa/ir/ir.h");
    run_clang_format("../../ir/ir.cpp");
    run_clang_format("../../../include/luisa/ir/fwd.h");
    Ok(())
}
