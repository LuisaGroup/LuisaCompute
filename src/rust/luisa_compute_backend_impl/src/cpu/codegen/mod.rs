use base64ct::Encoding;
use half::f16;
use luisa_compute_ir::{context, CArc, CBoxedSlice};
use luisa_compute_ir_v2::{TypeRef, TypeTag};
use sha2::{Digest, Sha256};
use std::ffi::CString;

use crate::ir::{Primitive, Type, VectorElementType};
use luisa_compute_ir::ir;

pub mod cpp;
pub mod cpp_v2;

pub fn sha256_full(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s);
    let hash = hasher.finalize();
    format!(
        "A{}",
        base64ct::Base64UrlUnpadded::encode_string(&hash).replace("-", "m_")
    )
}
pub fn sha256_short(s: &str) -> String {
    sha256_full(s)[0..17].to_string()
}

pub fn decode_const_data(
    data: &[u8],
    ty: &CArc<Type>,
    t2s: &impl Fn(&CArc<Type>) -> String,
) -> String {
    match ty.as_ref() {
        Type::Primitive(p) => match *p {
            Primitive::Bool => {
                format!("bool({})", if data[0] == 0 { "false" } else { "true" })
            }
            Primitive::Int8 => {
                format!("int8_t({})", data[0] as i8)
            }
            Primitive::Uint8 => {
                format!("uint8_t({})", data[0])
            }
            Primitive::Int16 => {
                format!("int16_t({})", i16::from_le_bytes([data[0], data[1]]))
            }
            Primitive::Uint16 => {
                format!("uint16_t({})", u16::from_le_bytes([data[0], data[1]]))
            }
            Primitive::Int32 => {
                format!(
                    "int32_t({})",
                    i32::from_le_bytes([data[0], data[1], data[2], data[3]])
                )
            }
            Primitive::Uint32 => {
                format!(
                    "uint32_t({})",
                    u32::from_le_bytes([data[0], data[1], data[2], data[3]])
                )
            }
            Primitive::Int64 => {
                format!(
                    "int64_t({})",
                    i64::from_le_bytes([
                        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]
                    ])
                )
            }
            Primitive::Uint64 => {
                format!(
                    "uint64_t({})",
                    u64::from_le_bytes([
                        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]
                    ])
                )
            }
            Primitive::Float16 => {
                format!("half({})", f16::from_le_bytes([data[0], data[1]]).to_f32())
            }
            Primitive::Float32 => {
                format!(
                    "float({})",
                    f32::from_le_bytes([data[0], data[1], data[2], data[3]])
                )
            }
            Primitive::Float64 => {
                format!(
                    "double({})",
                    f64::from_le_bytes([
                        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]
                    ])
                )
            }
        },
        Type::Vector(vt) => {
            let e = match &vt.element {
                VectorElementType::Scalar(p) => *p,
                _ => unimplemented!(),
            };
            let len = vt.length;
            match e {
                Primitive::Bool => {
                    format!(
                        "lc_bool{}({})",
                        len,
                        data.iter()
                            .take(len as usize)
                            .map(|x| if *x == 0 { "false" } else { "true" })
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                Primitive::Int8 => {
                    format!(
                        "lc_char{}({})",
                        len,
                        data.chunks(1)
                            .take(len as usize)
                            .map(|x| format!("{}", x[0] as i8))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                Primitive::Uint8 => {
                    format!(
                        "lc_uchar{}({})",
                        len,
                        data.chunks(1)
                            .take(len as usize)
                            .map(|x| format!("{}", x[0]))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                Primitive::Int16 => {
                    format!(
                        "lc_short{}({})",
                        len,
                        data.chunks(2)
                            .take(len as usize)
                            .map(|x| format!("{}", i16::from_le_bytes([x[0], x[1]])))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                Primitive::Uint16 => {
                    format!(
                        "lc_ushort{}({})",
                        len,
                        data.chunks(2)
                            .take(len as usize)
                            .map(|x| format!("{}", u16::from_le_bytes([x[0], x[1]])))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                Primitive::Int32 => {
                    format!(
                        "lc_int{}({})",
                        len,
                        data.chunks(4)
                            .take(len as usize)
                            .map(|x| format!("{}", i32::from_le_bytes([x[0], x[1], x[2], x[3]])))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                Primitive::Uint32 => {
                    format!(
                        "lc_uint{}({})",
                        len,
                        data.chunks(4)
                            .take(len as usize)
                            .map(|x| format!("{}", u32::from_le_bytes([x[0], x[1], x[2], x[3]])))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                Primitive::Int64 => {
                    format!(
                        "lc_long{}({})",
                        len,
                        data.chunks(8)
                            .take(len as usize)
                            .map(|x| format!(
                                "{}",
                                i64::from_le_bytes([
                                    x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
                                ])
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                Primitive::Uint64 => {
                    format!(
                        "lc_ulong{}({})",
                        len,
                        data.chunks(8)
                            .take(len as usize)
                            .map(|x| format!(
                                "{}",
                                u64::from_le_bytes([
                                    x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
                                ])
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                Primitive::Float16 => {
                    format!(
                        "lc_half{}({})",
                        len,
                        data.chunks(2)
                            .take(len as usize)
                            .map(|x| format!(
                                "lc_half({})",
                                f16::from_le_bytes([x[0], x[1]]).to_f32()
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                Primitive::Float32 => {
                    format!(
                        "lc_float{}({})",
                        len,
                        data.chunks(4)
                            .take(len as usize)
                            .map(|x| format!("{}", f32::from_le_bytes([x[0], x[1], x[2], x[3]])))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                Primitive::Float64 => {
                    format!(
                        "lc_double{}({})",
                        len,
                        data.chunks(8)
                            .take(len as usize)
                            .map(|x| format!(
                                "{}",
                                f64::from_le_bytes([
                                    x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
                                ])
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            }
        }
        Type::Matrix(mt) => {
            let e = match &mt.element {
                VectorElementType::Scalar(p) => *p,
                _ => unimplemented!(),
            };
            let dim = mt.dimension;
            let width = match dim {
                2 => 2,
                3 | 4 => 4,
                _ => unreachable!(),
            };
            match e {
                Primitive::Float32 => {
                    format!(
                        "lc_float{0}x{0}({1})",
                        dim,
                        data.chunks(4 * width as usize)
                            .take(dim as usize)
                            .map(|data| format!(
                                "lc_float{}({})",
                                dim,
                                data.chunks(4)
                                    .take(dim as usize)
                                    .map(|x| format!(
                                        "{}",
                                        f32::from_le_bytes([x[0], x[1], x[2], x[3]])
                                    ))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                _ => {
                    unimplemented!()
                }
            }
        }
        Type::Struct(s) => {
            let fields = s.fields.as_ref();
            let mut offset = 0usize;
            let out = format!(
                "{} {{ {} }}",
                t2s(ty),
                fields
                    .iter()
                    .map(|f| {
                        let len = f.size();
                        let data = &data[offset..offset + len];
                        offset += len;
                        decode_const_data(data, f, t2s)
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            assert_eq!(offset, data.len());
            out
        }
        Type::Array(at) => {
            format!(
                "{} ( {} )",
                t2s(ty),
                data.chunks(at.element.size())
                    .map(|data| decode_const_data(data, &at.element, t2s))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        _ => {
            todo!()
        }
    }
}
fn aggregate_printf(var: String, ty: &CArc<Type>) -> (String, String) {
    use std::fmt::Write;
    let mut printf_fmt = String::new();
    let mut printf_args = String::new();
    match ty.as_ref() {
        Type::Primitive(p) => match p {
            Primitive::Bool => {
                printf_fmt.push_str("%s");
                write!(printf_args, ",{} ? \"true\" : \"false\"", var).unwrap();
            }
            Primitive::Int8 => {
                printf_fmt.push_str("%d");
                write!(printf_args, ",static_cast<lc_int>({})", var).unwrap();
            }
            Primitive::Uint8 => {
                printf_fmt.push_str("%u");
                write!(printf_args, ",static_cast<lc_uint>({})", var).unwrap();
            }
            Primitive::Int16 => {
                printf_fmt.push_str("%d");
                write!(printf_args, ",static_cast<lc_int>({})", var).unwrap();
            }
            Primitive::Uint16 => {
                printf_fmt.push_str("%u");
                write!(printf_args, ",static_cast<lc_uint>({})", var).unwrap();
            }
            Primitive::Int32 => {
                printf_fmt.push_str("%d");
                write!(printf_args, ",static_cast<lc_int>({})", var).unwrap();
            }
            Primitive::Uint32 => {
                printf_fmt.push_str("%u");
                write!(printf_args, ",static_cast<lc_uint>({})", var).unwrap();
            }
            Primitive::Int64 => {
                printf_fmt.push_str("%lld");
                write!(printf_args, ",static_cast<lc_long>({})", var).unwrap();
            }
            Primitive::Uint64 => {
                printf_fmt.push_str("%llu");
                write!(printf_args, ",static_cast<lc_ulong>({})", var).unwrap();
            }
            Primitive::Float16 => {
                printf_fmt.push_str("%g");
                write!(printf_args, ",static_cast<float>({})", var).unwrap();
            }
            Primitive::Float32 => {
                printf_fmt.push_str("%g");
                write!(printf_args, ",static_cast<float>({})", var).unwrap();
            }
            Primitive::Float64 => {
                printf_fmt.push_str("%g");
                write!(printf_args, ",static_cast<double>({})", var).unwrap();
            }
        },
        Type::Array(a) => {
            printf_fmt.push_str("[");
            for i in 0..a.length {
                let (fmt, args) = aggregate_printf(format!("{}[{}]", var, i), &a.element);
                printf_fmt.push_str(&fmt);
                printf_args.push_str(&args);
                if i != a.length - 1 {
                    printf_fmt.push_str(", ");
                }
            }
            printf_fmt.push_str("]");
        }
        Type::Struct(s) => {
            printf_fmt.push_str("{");
            for (i, f) in s.fields.as_ref().iter().enumerate() {
                let (fmt, args) = aggregate_printf(format!("{}.f{}", var, i), f);
                printf_fmt.push_str(&fmt);
                printf_args.push_str(&args);
                if i != s.fields.as_ref().len() - 1 {
                    printf_fmt.push_str(", ");
                }
            }
            printf_fmt.push_str("}");
        }
        Type::Vector(v) => {
            let p = match v.element {
                VectorElementType::Scalar(p) => p,
                _ => unreachable!(),
            };
            let pt = context::register_type(Type::Primitive(p));
            printf_fmt.push_str("(");
            for i in 0..v.length {
                let (fmt, args) = aggregate_printf(format!("{}[{}]", var, i), &pt);
                printf_fmt.push_str(&fmt);
                printf_args.push_str(&args);
                if i != v.length - 1 {
                    printf_fmt.push_str(", ");
                }
            }
            printf_fmt.push_str(")");
        }
        Type::Matrix(mt) => {
            let p = match mt.element {
                VectorElementType::Scalar(p) => p,
                _ => unreachable!(),
            };
            let pt = Type::vector(p, mt.dimension);
            printf_fmt.push_str("<");
            for i in 0..mt.dimension {
                let (fmt, args) = aggregate_printf(format!("{}[{}]", var, i), &pt);
                printf_fmt.push_str(&fmt);
                printf_args.push_str(&args);
                if i != mt.dimension - 1 {
                    printf_fmt.push_str(", ");
                }
            }
            printf_fmt.push_str(">");
        }
        _ => unreachable!(),
    }
    (printf_fmt, printf_args)
}
pub fn decode_const_data_v2(data: &[u8], ty: TypeRef) -> String {
    match ty.tag() {
        TypeTag::Bool => {
            format!("bool({})", if data[0] == 0 { "false" } else { "true" })
        }
        TypeTag::Int8 => {
            format!("int8_t({})", data[0] as i8)
        }
        TypeTag::Uint8 => {
            format!("uint8_t({})", data[0])
        }
        TypeTag::Int16 => {
            format!("int16_t({})", i16::from_le_bytes([data[0], data[1]]))
        }
        TypeTag::Uint16 => {
            format!("uint16_t({})", u16::from_le_bytes([data[0], data[1]]))
        }
        TypeTag::Int32 => {
            format!(
                "int32_t({})",
                i32::from_le_bytes([data[0], data[1], data[2], data[3]])
            )
        }
        TypeTag::Uint32 => {
            format!(
                "uint32_t({})",
                u32::from_le_bytes([data[0], data[1], data[2], data[3]])
            )
        }
        TypeTag::Int64 => {
            format!(
                "int64_t({})",
                i64::from_le_bytes([
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]
                ])
            )
        }
        TypeTag::Uint64 => {
            format!(
                "uint64_t({})",
                u64::from_le_bytes([
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]
                ])
            )
        }
        TypeTag::Float16 => {
            format!("half({})", f16::from_le_bytes([data[0], data[1]]).to_f32())
        }
        TypeTag::Float32 => {
            format!(
                "float({})",
                f32::from_le_bytes([data[0], data[1], data[2], data[3]])
            )
        }
        TypeTag::Float64 => {
            format!(
                "double({})",
                f64::from_le_bytes([
                    data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]
                ])
            )
        }
        TypeTag::Vector => {
            let e = ty.element();
            let len = ty.dimension();
            match e.tag() {
                TypeTag::Bool => {
                    format!(
                        "lc_bool{}({})",
                        len,
                        data.iter()
                            .take(len as usize)
                            .map(|x| if *x == 0 { "false" } else { "true" })
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                TypeTag::Int8 => {
                    format!(
                        "lc_char{}({})",
                        len,
                        data.chunks(1)
                            .take(len as usize)
                            .map(|x| format!("{}", x[0] as i8))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                TypeTag::Uint8 => {
                    format!(
                        "lc_uchar{}({})",
                        len,
                        data.chunks(1)
                            .take(len as usize)
                            .map(|x| format!("{}", x[0]))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                TypeTag::Int16 => {
                    format!(
                        "lc_short{}({})",
                        len,
                        data.chunks(2)
                            .take(len as usize)
                            .map(|x| format!("{}", i16::from_le_bytes([x[0], x[1]])))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                TypeTag::Uint16 => {
                    format!(
                        "lc_ushort{}({})",
                        len,
                        data.chunks(2)
                            .take(len as usize)
                            .map(|x| format!("{}", u16::from_le_bytes([x[0], x[1]])))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                TypeTag::Int32 => {
                    format!(
                        "lc_int{}({})",
                        len,
                        data.chunks(4)
                            .take(len as usize)
                            .map(|x| format!("{}", i32::from_le_bytes([x[0], x[1], x[2], x[3]])))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                TypeTag::Uint32 => {
                    format!(
                        "lc_uint{}({})",
                        len,
                        data.chunks(4)
                            .take(len as usize)
                            .map(|x| format!("{}", u32::from_le_bytes([x[0], x[1], x[2], x[3]])))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                TypeTag::Int64 => {
                    format!(
                        "lc_long{}({})",
                        len,
                        data.chunks(8)
                            .take(len as usize)
                            .map(|x| format!(
                                "{}",
                                i64::from_le_bytes([
                                    x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
                                ])
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                TypeTag::Uint64 => {
                    format!(
                        "lc_ulong{}({})",
                        len,
                        data.chunks(8)
                            .take(len as usize)
                            .map(|x| format!(
                                "{}",
                                u64::from_le_bytes([
                                    x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
                                ])
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                TypeTag::Float16 => {
                    format!(
                        "lc_half{}({})",
                        len,
                        data.chunks(2)
                            .take(len as usize)
                            .map(|x| format!(
                                "lc_half({})",
                                f16::from_le_bytes([x[0], x[1]]).to_f32()
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                TypeTag::Float32 => {
                    format!(
                        "lc_float{}({})",
                        len,
                        data.chunks(4)
                            .take(len as usize)
                            .map(|x| format!("{}", f32::from_le_bytes([x[0], x[1], x[2], x[3]])))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                TypeTag::Float64 => {
                    format!(
                        "lc_double{}({})",
                        len,
                        data.chunks(8)
                            .take(len as usize)
                            .map(|x| format!(
                                "{}",
                                f64::from_le_bytes([
                                    x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
                                ])
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
                _ => unreachable!(),
            }
        }
        TypeTag::Matrix => {
            let e = ty.element();
            let dim = ty.dimension();
            let width = match dim {
                2 => 2,
                3 | 4 => 4,
                _ => unreachable!(),
            };
            format!(
                "lc_float{0}x{0}({1})",
                dim,
                data.chunks(4 * width as usize)
                    .take(dim as usize)
                    .map(|data| format!(
                        "lc_float{}({})",
                        dim,
                        data.chunks(4)
                            .take(dim as usize)
                            .map(|x| format!("{}", f32::from_le_bytes([x[0], x[1], x[2], x[3]])))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        _ => {
            panic!("const data of type {:?} is not supported", ty.description())
        }
    }
}