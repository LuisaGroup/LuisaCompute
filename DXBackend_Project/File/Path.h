#pragma once
#include <Common/Common.h>
class Path
{
private:
	vengine::string pathData;
public:
	static void SeparatePath(vengine::string const& path, vengine::vector<vengine::string>& blocks);
	Path(vengine::string const& path);
	Path(Path const& absolutePath, vengine::string const& relativePath);
	Path() {}
	bool IsEmpty() const
	{
		return pathData.empty();
	}
	static Path GetProgramPath();
	Path GetParentLevelPath() const;
	Path GetSubLevelPath(vengine::string const& subName) const;
	bool IsFile() const;
	bool IsDirectory() const;
	bool Exists() const;
	vengine::string const& GetPathStr() const
	{
		return pathData;
	}
	vengine::string GetExtension();
	Path& operator=(Path const& v);
	Path& operator=(vengine::string const& path);
	bool operator==(Path const& a) const;
	bool operator!=(Path const& a) const;
	bool IsSubPathOf(Path const& parentPath) const;
	bool IsParentPathOf(Path const& subPath) const;
	vengine::string TryGetSubPath(Path const& subPath) const;
	void TryCreateDirectory();
};
namespace vengine
{
	template <>
	struct hash<Path>
	{
		size_t operator()(Path const& p) const noexcept
		{
			static const hash<vengine::string> vc;
			return vc(p.GetPathStr());
		}
	};
}