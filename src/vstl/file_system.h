#pragma once
#include <vstl/Common.h>
#include <vstl/functional.h>
namespace vstl {

class FileSystem {
public:
	struct FileTime {
		uint64 creationTime;
		uint64 lastAccessTime;
		uint64 lastWriteTime;
	};
	static bool IsFileExists(std::string const& path);
	static void GetFiles(
		std::string const& path,
		vstd::function<bool(std::string&&)> const& callBack,
		bool recursionFolder);
};
}// namespace vstl