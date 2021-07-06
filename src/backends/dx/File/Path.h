#pragma once
#include <Common/Common.h>
class Path
{
private:
	vstd::string pathData;
public:
	static void SeparatePath(vstd::string const& path, vstd::vector<vstd::string>& blocks);
	Path(vstd::string const& path);
	Path(Path const& absolutePath, vstd::string const& relativePath);
	Path() {}
	bool IsEmpty() const
	{
		return pathData.empty();
	}
	static Path GetProgramPath();
	Path GetParentLevelPath() const;
	Path GetSubLevelPath(vstd::string const& subName) const;
	bool IsFile() const;
	bool IsDirectory() const;
	bool Exists() const;
	vstd::string const& GetPathStr() const
	{
		return pathData;
	}
	vstd::string GetExtension();
	Path& operator=(Path const& v);
	Path& operator=(vstd::string const& path);
	bool operator==(Path const& a) const;
	bool operator!=(Path const& a) const;
	bool IsSubPathOf(Path const& parentPath) const;
	bool IsParentPathOf(Path const& subPath) const;
	vstd::string TryGetSubPath(Path const& subPath) const;
	void TryCreateDirectory();
};
namespace vstd
{
	template <>
	struct hash<Path>
	{
		size_t operator()(Path const& p) const noexcept
		{
			static const hash<vstd::string> vc;
			return vc(p.GetPathStr());
		}
	};
}