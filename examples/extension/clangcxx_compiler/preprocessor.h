#pragma once
#include <luisa/core/clock.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/filesystem.h>
#include <luisa/vstl/common.h>
#include <luisa/vstl/functional.h>
#include <luisa/vstl/lmdb.hpp>
#include <luisa/vstl/md5.h>
#include <luisa/clangcxx/compiler.h>
#include <luisa/core/fiber.h>
#include <luisa/runtime/context.h>
#include <luisa/core/binary_file_stream.h>
#include <luisa/vstl/v_guid.h>
#include <luisa/core/stl/pdqsort.h>
#include <luisa/vstl/spin_mutex.h>
#include "simplecpp.h"
#include <mimalloc.h>
using namespace luisa;
template<typename Vec, typename T>
void add(Vec&& result, T c) {
	if constexpr (std::is_same_v<T, char const*> || std::is_same_v<T, char*>) {
		vstd::push_back_all(result, span<char const>(c, strlen(c)));
	} else if constexpr (std::is_same_v<T, char>) {
		result.emplace_back(c);
	} else {
		vstd::push_back_all(result, luisa::span<char const>(c.data(), c.size()));
	}
};

class Preprocessor {
	vstd::LMDB db;
	std::filesystem::path _cache_path;
	luisa::vector<luisa::string_view> _defines;
	luisa::vector<luisa::string> _inc_paths;
	vstd::spin_mutex _time_mtx;
	using DBValue = luisa::vector<std::byte>;
	vstd::HashMap<luisa::string, DBValue> _last_write_times;
	vstd::spin_mutex _remove_mtx;
	luisa::vector<luisa::vector<std::byte>> _remove_list;
	void update_file(luisa::string_view name, std::filesystem::file_time_type time, luisa::span<const std::byte> data) {
		DBValue vec;
		vec.reserve(data.size() + sizeof(time));
		vstd::push_back_all(vec, reinterpret_cast<std::byte const*>(&time), sizeof(time));
		if (!data.empty()) {
			vstd::push_back_all(vec, reinterpret_cast<std::byte const*>(data.data()), data.size());
		}
		std::lock_guard lck{_time_mtx};
		auto iter = _last_write_times.try_emplace(name);
		if (iter.second || (iter.first.value().size() < data.size() + sizeof(time))) {
			iter.first.value() = std::move(vec);
		}
	}
	bool file_is_new(luisa::string_view name) {
		luisa::span<const std::byte> db_value;
		return file_is_new(name, db_value);
	}
	bool file_is_new(luisa::string_view name, luisa::span<const std::byte>& db_value) {
		db_value = db.read(name);
		auto cur_time = std::filesystem::last_write_time(std::filesystem::path{name});
		if (db_value.size_bytes() >= sizeof(std::filesystem::file_time_type)) {
			std::filesystem::file_time_type old_time;
			memcpy(&old_time, db_value.data(), sizeof(old_time));
			if (cur_time <= old_time) {
				return false;
			}
		}
		update_file(name, cur_time, {});
		return true;
	}

public:
	Preprocessor(
		std::filesystem::path const& db_path,
		std::filesystem::path&& cache_path,
		vstd::IRange<luisa::string_view>& defines,
		vstd::IRange<luisa::string>& inc_paths)
		: db(db_path, std::max<size_t>(126ull, std::thread::hardware_concurrency() * 2)), _cache_path(std::move(cache_path)) {
		if (!std::filesystem::exists(_cache_path)) {
			std::error_code ec;
			std::filesystem::create_directories(_cache_path, ec);
			if (ec) [[unlikely]] {
				LUISA_ERROR("Create cache path '{}' failed, message: {}", luisa::to_string(_cache_path), ec.message());
			}
		}
		for (auto&& i : defines) {
			_defines.emplace_back(i);
		}
		for (auto&& i : inc_paths) {
			_inc_paths.emplace_back(std::move(i));
		}
	}
	void remove_file(luisa::string_view name) {
		std::lock_guard lck{_remove_mtx};
		auto& v = _remove_list.emplace_back();
		luisa::enlarge_by(v, name.size());
		memcpy(v.data(), name.data(), name.size());
	}
	void post_process() {
		luisa::vector<vstd::LMDBWriteCommand> write_cmds;
		write_cmds.reserve(_last_write_times.size());
		for (auto&& i : _last_write_times) {
			auto&& kv = write_cmds.emplace_back();
			vstd::push_back_all(kv.key, reinterpret_cast<std::byte const*>(i.first.data()), i.first.size());
			kv.value = std::move(i.second);
		}
		_last_write_times.clear();
		db.write_all(std::move(write_cmds));
		db.remove_all(std::move(_remove_list));
		_remove_list.clear();
		if (std::filesystem::exists(_cache_path)) {
			std::error_code ec;
			std::filesystem::remove_all(_cache_path, ec);
			if (ec) [[unlikely]] {
				LUISA_WARNING("Delete temp dir: {} failed: {}", luisa::to_string(_cache_path), ec.message());
			}
		}
	}
	static luisa::unordered_set<luisa::string> include_dirs(luisa::string_view text) {
		auto ptr = text.data();
		auto end = text.data() + text.size();
		luisa::unordered_set<luisa::string> r;
#define NEXT_PTR                     \
	do {                             \
		++ptr;                       \
		if (ptr == end) [[unlikely]] \
			return r;                \
	} while (false)
		for (; ptr != end; ++ptr) {
			if (*ptr != '#') [[likely]] {
				continue;
			}
			NEXT_PTR;
			while (*ptr != '"') {
				NEXT_PTR;
			}
			NEXT_PTR;
			auto path_begin = ptr;
			while (*ptr != '"') {
				if (*ptr == '<') [[unlikely]] {
					++ptr;
					goto CONTINUE;
				}
				NEXT_PTR;
			}
			r.emplace(luisa::string{path_begin, ptr});
		CONTINUE:
			continue;
		}
		return r;
#undef NEXT_PTR
	}
	bool require_recompile(
		std::filesystem::path const& src_dir,
		std::filesystem::path const& file_dir) {
		std::error_code ec;
		auto file_abs_dir = std::filesystem::canonical(src_dir / file_dir, ec);
		if (ec) [[unlikely]] {
			LUISA_ERROR("Invalid canonical file path '{}' failed, message: {}", luisa::to_string(file_abs_dir), ec.message());
		}
		if (ec) [[unlikely]] {
			LUISA_ERROR("Get file last write time '{}' failed, message: {}", luisa::to_string(_cache_path), ec.message());
		}
		auto file_abs_dir_str = luisa::to_string(file_abs_dir);
		if (ec) [[unlikely]] {
			LUISA_ERROR("Invalid canonical file path '{}' failed, message: {}", luisa::to_string(file_dir), ec.message());
		}
		vstd::Guid guid{true};
		if (ec) [[unlikely]] {
			LUISA_ERROR("Invalid canonical file path '{}' failed, message: {}", luisa::to_string(_cache_path / file_dir), ec.message());
		}
		luisa::span<const std::byte> db_value;
		auto preprocess = [&]<bool check_preprocess = true>() {
			auto preprocess_path = std::filesystem::weakly_canonical(_cache_path / guid.to_string(false), ec);
			vstd::MD5 old_md5;
			if constexpr (check_preprocess) {
				int64_t lefted_size = db_value.size_bytes() - sizeof(std::filesystem::file_time_type);
				if (lefted_size >= int64_t(sizeof(vstd::MD5))) {
					memcpy(&old_md5, db_value.data() + sizeof(std::filesystem::file_time_type), sizeof(vstd::MD5));
				}
			}
			std::vector<std::string> files;
			std::string preprocessed_path;
			{
				simplecpp::DUI dui;
				dui.removeComments = true;
				for (auto&& i : _inc_paths) {
					dui.includePaths.emplace_back(i);
				}
				for (auto&& i : _defines) {
					dui.defines.emplace_back(i);
				}
				std::map<std::string, simplecpp::TokenList*> filedata;
				std::string filename{file_abs_dir_str};
				simplecpp::OutputList outputList;
				simplecpp::TokenList rawtokens(filename, files, &outputList);
				rawtokens.removeComments();
				simplecpp::TokenList outputTokens(files);
				simplecpp::preprocess(outputTokens, rawtokens, files, filedata, dui, &outputList);
				preprocessed_path = outputTokens.stringify();
				simplecpp::cleanup(filedata);
			}
			{
				vstd::MD5 md5{{reinterpret_cast<uint8_t const*>(preprocessed_path.data()), preprocessed_path.size()}};
				if constexpr (check_preprocess) {
					if (old_md5 == md5) {
						return false;
					}
				}
				luisa::vector<std::byte> vec;
  				auto push = [&]<typename T>(T const& a) {
  					auto last_size = vec.size();
  					if constexpr (std::is_trivial_v<T>) {
  						luisa::enlarge_by(vec, sizeof(T));
  						memcpy(vec.data() + last_size, &a, sizeof(T));
  					} else if constexpr (std::is_same_v<T, luisa::string_view> || std::is_same_v<T, luisa::string>) {
  						luisa::enlarge_by(vec, a.size());
  						memcpy(vec.data() + last_size, a.data(), a.size());
  					} else {
  						luisa::enlarge_by(vec, a.size_bytes());
  						memcpy(vec.data() + last_size, a.data(), a.size_bytes());
  					}
  				};
				push(md5.to_binary());
				// nlohmann::json js_arr;
				for (auto&& i : files) {
					auto inc_path = std::filesystem::weakly_canonical(i, ec);
					auto path = luisa::to_string(inc_path);
					update_file(path, std::filesystem::last_write_time(inc_path), {});
					push(path.size());
					push(path);
				}
				// auto js_str = js_arr.dump();
				update_file(file_abs_dir_str, std::filesystem::last_write_time(file_abs_dir), vec);
			}
			return true;
		};
		if (file_is_new(file_abs_dir_str, db_value)) {
			return preprocess();
		}
		// check include files
		auto header_size = sizeof(std::filesystem::file_time_type) + sizeof(vstd::MD5);
		int64_t data_size = db_value.size() - header_size;
		if (data_size > 0) {
			auto ptr = db_value.data() + header_size;
			auto end_ptr = db_value.data() + db_value.size();
			bool force_check = false;
			while (ptr < end_ptr) {
				size_t str_size;
				memcpy(&str_size, ptr, sizeof(size_t));
				ptr += sizeof(size_t);
				if (ptr >= end_ptr) [[unlikely]] {
					force_check = true;
					break;
				}
				luisa::string_view name = {reinterpret_cast<char const*>(ptr), reinterpret_cast<char const*>(ptr + str_size)};
				ptr += str_size;
				if (ptr > end_ptr) [[unlikely]] {
					LUISA_WARNING("Invalid cache data.");
					force_check = true;
					break;
				}
				if (file_is_new(name)) {
					force_check = true;
					break;
				}
			}
			if (force_check) {
				return preprocess();
			} else {
				return false;
			}
		}
		return false;
	}
};