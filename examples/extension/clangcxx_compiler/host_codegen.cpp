#include "host_codegen.h"
#include <luisa/core/stl/format.h>
#include <luisa/core/logging.h>
#include <luisa/ast/type.h>
#include <luisa/vstl/md5.h>
#include <luisa/vstl/common.h>
#include <luisa/core/binary_file_stream.h>
#ifdef _WIN32
#include <windows.h>
#endif
#undef ERROR
#undef Yield
#undef byte
namespace detail {
using namespace luisa::compute;
static void _dummy_function_for_get_binary_path() {}
luisa::filesystem::path get_binary_path() {
	luisa::filesystem::path result;
#ifdef _WIN32
	TCHAR dllPath[MAX_PATH];
	{
		HMODULE hm = NULL;
		GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCWSTR)&_dummy_function_for_get_binary_path, &hm);
		GetModuleFileName(hm, dllPath, MAX_PATH);
	}
	result = luisa::filesystem::path(dllPath).parent_path();
	SetDllDirectoryW(result.c_str());
#else
	static_assert(false, "root-path not supported in current platform.");
#endif
	return result;
}
static void load_str(
	HostCodegen::MapType& map_type,
	luisa::span<luisa::clangcxx::BuildArgument const> build_args) {
	map_type.emplace("TypeNameDecl", [=](vstd::StringBuilder& sb) {
		for (auto& a : build_args) {
			auto desc = a.type->description();
			sb << luisa::format("char {}_desc[{}] = ", a.var_name, desc.size() + 1);
			sb << '{';
			for (auto& c : desc) {
				sb << luisa::format("{},", (int)c);
			}
			sb << "0};\n"sv;
		}
	});
	map_type.emplace("declArgTypes", [=](vstd::StringBuilder& sb) {
		bool is_first = true;
		for (auto& a : build_args) {
			auto desc = a.type->description();
			if (!is_first) [[likely]] {
				sb << ',';
			}
			is_first = false;
			sb << luisa::format("::luisa::compute::Type::from(luisa::string_view({}_desc, {}))", a.var_name, desc.size());
		}
	});
}
static void dispatch_str(
	uint32_t dimension,
	HostCodegen::MapType& map_type,
	vstd::unordered_map<Type const*, size_t>& struct_types,
	luisa::vector<Type const*>& struct_type_vecs,
	luisa::span<luisa::clangcxx::BuildArgument const> build_args) {

	auto add_struct = [&](auto& add_struct, Type const* type) -> void {
		switch (type->tag()) {
			case Type::Tag::STRUCTURE:
				struct_types.try_emplace(type, struct_types.size());
				break;
			case Type::Tag::ARRAY:
				add_struct(add_struct, type->element());
				break;
			case Type::Tag::BUFFER:
				if (type->element())
					add_struct(add_struct, type->element());

				break;
		}
	};
	for (auto& i : build_args) {
		add_struct(add_struct, i.type);
	}
	map_type.emplace("StructTemplate", [&struct_types, &struct_type_vecs](vstd::StringBuilder& result) {
		if (!struct_types.empty()) {
			struct_type_vecs.resize(struct_types.size());
			for (auto& i : struct_types) {
				struct_type_vecs[i.second] = i.first;
			}
			result << "template <";
			size_t idx = 0;
			bool is_first = true;
			for (auto& i : struct_type_vecs) {
				if (!is_first) {
					result << ',';
				}
				is_first = false;
				result << luisa::format("typename A{}", idx);
				idx += 1;
			}
			result << ">\nrequires(";
			idx = 0;
			is_first = true;
			for (auto& i : struct_type_vecs) {
				if (!is_first) {
					result << "&&\n";
				}
				is_first = false;
				auto type_name = luisa::format("A{}", idx);
				result << luisa::format("(sizeof({}) == {} && alignof({}) == {} && std::is_trivially_copyable_v<{}>)", type_name, i->size(), type_name, i->alignment(), type_name);
				idx += 1;
			}
			result << ")"sv;
		}
	});
	map_type.emplace("DispatchSizeType", [dimension](vstd::StringBuilder& result) {
		result << (dimension > 1 ? luisa::format("::luisa::uint{}", dimension) : "::luisa::uint");
	});
	map_type.emplace("DispatchArgs", [&struct_types, &struct_type_vecs, build_args](vstd::StringBuilder& result) {
		auto elem_name = [&](auto& elem_name, Type const* type) -> void {
			switch (type->tag()) {
				case Type::Tag::BUFFER:
					if (type->element()) {
						result << "::luisa::compute::BufferView<"sv;
						elem_name(elem_name, type->element());
						result << '>';
					} else {
						result << "ByteBufferView"sv;
					}
					break;
				case Type::Tag::TEXTURE: {
					if (type->dimension() == 3) {
						result << "::luisa::compute::VolumeView<"sv;
					} else {
						result << "::luisa::compute::ImageView<"sv;
					}
					elem_name(elem_name, type->element());
					result << '>';
				} break;
				case Type::Tag::BINDLESS_ARRAY:
					result << "::luisa::compute::BindlessArray"sv;
					break;
				case Type::Tag::ACCEL:
					result << "::luisa::compute::Accel"sv;
					break;
				case Type::Tag::BOOL: result << "bool"sv; break;
				case Type::Tag::INT8: result << "int8_t"sv; break;
				case Type::Tag::UINT8: result << "uint8_t"sv; break;
				case Type::Tag::INT16: result << "int16_t"sv; break;
				case Type::Tag::UINT16: result << "uint16_t"sv; break;
				case Type::Tag::INT32: result << "int32_t"sv; break;
				case Type::Tag::UINT32: result << "uint32_t"sv; break;
				case Type::Tag::INT64: result << "int64_t"sv; break;
				case Type::Tag::UINT64: result << "uint64_t"sv; break;
				case Type::Tag::FLOAT16: result << "::luisa::half"sv; break;
				case Type::Tag::FLOAT32: result << "float"sv; break;
				case Type::Tag::FLOAT64: result << "double"sv; break;
				case Type::Tag::VECTOR:
					result << "::luisa::Vector<"sv;
					elem_name(elem_name, type->element());
					result << luisa::format(",{}>", type->dimension());
					break;
				case Type::Tag::MATRIX:
					result << luisa::format("::luisa::float{}x{}", type->dimension(), type->dimension());
					break;
				case Type::Tag::ARRAY:
					result << "std::array<"sv;
					elem_name(elem_name, type->element());
					result << luisa::format(",{}>", type->dimension());
					break;
				case Type::Tag::STRUCTURE: {
					auto iter = struct_types.find(type);
					LUISA_ASSERT(iter != struct_types.end());
					result << luisa::format("A{}", iter->second);
				} break;
				default:
					LUISA_ERROR("Type not supported.");
					break;
			}
		};
		for (auto& i : build_args) {
			result << ",\n"sv;
			elem_name(elem_name, i.type);
			switch (i.type->tag()) {
				case Type::Tag::BINDLESS_ARRAY:
				case Type::Tag::ACCEL:
				case Type::Tag::ARRAY:
				case Type::Tag::STRUCTURE:
					result << " const &"sv;
					break;
			}
			result << ' ' << i.var_name;
		}
	});
	map_type.emplace("buildArgSize", [build_args](vstd::StringBuilder& result) {
		result << luisa::format("{}", build_args.size());
	});
	map_type.emplace("EncodeArg", [build_args, dimension](vstd::StringBuilder& result) {
		for (auto& i : build_args) {
			switch (i.type->tag()) {
				case Type::Tag::BUFFER:
					result << luisa::format("_cmd_encoder.encode_buffer({}.handle(),{}.offset_bytes(),{}.size_bytes());\n", i.var_name, i.var_name, i.var_name);
					break;
				case Type::Tag::TEXTURE:
					result << luisa::format("_cmd_encoder.encode_texture({}.handle(), {}.level());\n", i.var_name, i.var_name);
					break;
				case Type::Tag::BINDLESS_ARRAY:
					result << luisa::format("_cmd_encoder.encode_bindless_array({}.handle());\n", i.var_name);
					break;
				case Type::Tag::ACCEL:
					result << luisa::format("_cmd_encoder.encode_accel({}.handle());\n", i.var_name);
					break;
				case Type::Tag::ARRAY:
					result << luisa::format("_cmd_encoder.encode_uniform({}.data(), {}, {});\n", i.var_name, i.type->size(), i.type->alignment());
					break;
				default:
					result << luisa::format("_cmd_encoder.encode_uniform(std::addressof({}), {}, {});\n", i.var_name, i.type->size(), i.type->alignment());
					break;
			}
		}
		switch (dimension) {
			case 1:
				result << "_cmd_encoder.set_dispatch_size(::luisa::make_uint3(_dispatch_size,1u,1u));\n"sv;
				break;
			case 2:
				result << "_cmd_encoder.set_dispatch_size(::luisa::make_uint3(_dispatch_size,1u));\n"sv;
				break;
			default:
				result << "_cmd_encoder.set_dispatch_size(_dispatch_size);\n"sv;
				break;
		}
	});
	map_type.emplace("NamedHasArgs", [build_args](vstd::StringBuilder& result) {
		if (build_args.empty()) {
			result << "false"sv;
		} else {
			bool is_first = true;
			for (auto& arg : build_args) {
				if (!is_first) { result << " ||\n"sv; }
				is_first = false;
				result << luisa::format(
					"Name == \"{}\"",
					arg.var_name);
			}
		}
	});
	map_type.emplace("NamedRequiredArgs", [build_args](vstd::StringBuilder& result) {
		for (auto& arg : build_args) {
			result << luisa::format(
				" &&\nargument_count<\"{}\", NamedArgs...> == 1u",
				arg.var_name);
		}
	});
	map_type.emplace("NamedDispatchArgs", [build_args](vstd::StringBuilder& result) {
		for (auto& arg : build_args) {
			result << luisa::format(
				",\nargument<\"{}\">(\n"
				"::std::forward<NamedArgs>(args)...)",
				arg.var_name);
		}
	});
}
}// namespace detail
vstd::StringBuilder HostCodegen::codegen(
	uint32_t dimension,
	luisa::span<luisa::clangcxx::BuildArgument const> build_args) {
	vstd::StringBuilder result;
	result.reserve(1024);
	MapType map;
	luisa::BinaryFileStream file_stream{luisa::to_string(detail::get_binary_path() / "template.txt")};
	if (!file_stream.valid()) [[unlikely]] {
		LUISA_ERROR("template.txt file is invalid.");
	}
	luisa::vector<std::byte> template_text;
	template_text.resize_uninitialized(file_stream.length());
	file_stream.read(template_text);

	detail::load_str(map, build_args);
	vstd::unordered_map<luisa::compute::Type const*, size_t> struct_types;
	luisa::vector<luisa::compute::Type const*> struct_type_vecs;
	detail::dispatch_str(dimension, map, struct_types, struct_type_vecs, build_args);
	codegen_replace(result, map, luisa::string_view{(char const*)template_text.data(), template_text.size()}, '$');
	return result;
}
void HostCodegen::write_to(
	luisa::span<luisa::clangcxx::BuildArgument const> build_args,
	uint32_t dimension,
	luisa::filesystem::path const& path) {
	auto result = codegen(dimension, build_args);
	auto path_str = luisa::to_string(path);
	if (luisa::filesystem::exists(path)) {
		luisa::BinaryFileStream fs{path_str};
		luisa::vector<std::byte> dst;
		dst.resize(fs.length());
		fs.read(dst);
		vstd::MD5 src_md5{luisa::span{(uint8_t const*)dst.data(), dst.size()}};
		vstd::MD5 dst_md5{luisa::span{(uint8_t const*)result.data(), result.size()}};
		if (src_md5 == dst_md5) {
			return;
		}
	}
	auto f = fopen(path_str.c_str(), "wb");
	if (f) {
		fwrite(result.data(), result.size(), 1, f);
		fclose(f);
	}
}
void HostCodegen::codegen_replace(
	vstd::StringBuilder& sb,
	MapType const& replace_funcs,
	luisa::string_view template_type,
	char replace_char) {
	auto end_ptr = template_type.data() + template_type.size();
	char const* ptr = template_type.data();
	char const* last_cached_ptr = ptr;
	auto clear_cache = [&](char const* ptr) {
		if (last_cached_ptr < ptr) {
			sb.append(luisa::string_view{last_cached_ptr, ptr});
			last_cached_ptr = ptr;
		}
	};
	while (ptr < end_ptr) {
		if (*ptr == replace_char) {
			clear_cache(ptr);
			++ptr;
			auto key_begin = ptr;
			while (ptr < end_ptr) {
				if ((*ptr >= 'a' && *ptr <= 'z') || (*ptr >= 'A' && *ptr <= 'Z') || (*ptr >= '0' && *ptr <= '9') || *ptr == '_') {
					++ptr;
					continue;
				}
				break;
			}
			last_cached_ptr = ptr;
			auto iter = replace_funcs.find(luisa::string_view{key_begin, ptr});
			if (!iter) [[unlikely]] {
				LUISA_ERROR("Bad key {}", luisa::string_view{key_begin, ptr});
			}
			iter.value()(sb);
		} else {
			++ptr;
		}
	}
	clear_cache(ptr);
}
