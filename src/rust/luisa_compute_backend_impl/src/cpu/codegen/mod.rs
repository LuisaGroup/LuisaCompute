use base64ct::Encoding;
use half::f16;
use luisa_compute_ir::CBoxedSlice;
use sha2::{Digest, Sha256};
use std::ffi::CString;

use crate::ir::{Primitive, Type, VectorElementType};
use luisa_compute_ir::ir;

pub mod cpp;

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

pub fn decode_const_data(data: &[u8], ty: &Type) -> String {
    match ty {
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
                "{{ {} }}",
                fields
                    .iter()
                    .map(|f| {
                        let len = f.size();
                        let data = &data[offset..offset + len];
                        offset += len;
                        decode_const_data(data, f)
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            assert_eq!(offset, data.len());
            out
        }
        Type::Array(at) => {
            format!(
                "{{ {} }}",
                data.chunks(at.element.size())
                    .map(|data| decode_const_data(data, &at.element))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        _ => {
            todo!()
        }
    }
}

#[no_mangle]
pub extern "C" fn luisa_compute_decode_const_data(
    data: *const u8,
    len: usize,
    ty: &ir::Type,
) -> CBoxedSlice<u8> {
    let data = unsafe { std::slice::from_raw_parts(data, len) };
    let out = decode_const_data(data, ty);
    let cstring = CString::new(out).unwrap();
    CBoxedSlice::new(cstring.as_bytes().to_vec())
}
