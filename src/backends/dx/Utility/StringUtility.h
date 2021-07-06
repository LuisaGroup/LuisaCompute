#pragma once
#include <Common/vector.h>
#include <fstream>
#include <Common/vstring.h>
#include <Common/MetaLib.h>
#include <Common/string_view.h>
typedef unsigned int uint;
enum class CharCutState : uint8_t
{
	Same,
	Different,
	Cull
};
struct StringUtil
{
	enum class CodeType : uint8_t
	{
		Code,
		Comment,
		String
	};
	struct CodeChunk
	{
		vstd::string str;
		CodeType type;
		CodeChunk() {}
		CodeChunk(
			vstd::string str,
			CodeType type
		) : str(str), type(type) {}
		CodeChunk(
			char const* start,
			char const* end,
			CodeType type
		) : str(start, end), type(type) {}
	};
	static void SplitCodeString(
		char const* beg,
		char const* end,
		vstd::vector<vstd::string_view>& results,
		void* ptr,
		funcPtr_t<CharCutState(void*, char)> func
	);
	template <typename Func>
	static void SplitCodeString(
		char const* beg,
		char const* end,
		vstd::vector<vstd::string_view>& results,
		Func& func)
	{
		SplitCodeString(
			beg,
			end,
			results,
			&func,
			[](void* pp, char c)->CharCutState
			{
				return static_cast<CharCutState>(((Func*)pp)->operator()(c));
			}
		);
	}
	template <typename Func>
	static void SplitCodeString(
		char const* beg,
		char const* end,
		vstd::vector<vstd::string_view>& results,
		Func&& func)
	{
		SplitCodeString(
			beg,
			end,
			results,
			func
		);
	}
	template <typename Func>
	static void SplitCodeString(
		vstd::string const& str,
		vstd::vector<vstd::string_view>& results,
		Func& func)
	{
		results.clear();
		char const* beg = str.data();
		char const* end = beg + str.size();
		SplitCodeString<Func>(
			beg,
			end,
			results,
			func);
	}
	template <typename Func>
	static void SplitCodeString(
		vstd::string const& str,
		vstd::vector<vstd::string_view>& results,
		Func&& func)
	{
		SplitCodeString(
			str,
			results,
			func
		);
	}

	static void IndicesOf(const vstd::string& str, const vstd::string& sign, vstd::vector<uint>& v);
	static void IndicesOf(const vstd::string& str, char, vstd::vector<uint>& v);
	static void CutToLine(const vstd::string& str, vstd::vector<vstd::string>& lines);
	static void CutToLine_ContainEmptyLine(const vstd::string& str, vstd::vector<vstd::string>& lines);
	static void CutToLine(const char* str, const char* end, vstd::vector<vstd::string>& lines);
	static void CutToLine_ContainEmptyLine(const char* str, const char* end, vstd::vector<vstd::string>& lines);

	static void CutToLine(const vstd::string& str, vstd::vector<vstd::string_view>& lines);
	static void CutToLine_ContainEmptyLine(const vstd::string& str, vstd::vector<vstd::string_view>& lines);
	static void CutToLine(const char* str, const char* end, vstd::vector<vstd::string_view>& lines);
	static void CutToLine_ContainEmptyLine(const char* str, const char* end, vstd::vector<vstd::string_view>& lines);

	static void ReadLines(std::ifstream& ifs, vstd::vector<vstd::string>& lines);
	static int GetFirstIndexOf(const vstd::string& str, const vstd::string& sign);
	static int GetFirstIndexOf(const vstd::string& str, char sign);
	static void Split(const vstd::string& str, const vstd::string& sign, vstd::vector<vstd::string>& v);
	static void Split(const vstd::string& str, char sign, vstd::vector<vstd::string>& v);
	static void Split(const vstd::string& str, const vstd::string& sign, vstd::vector<vstd::string_view>& v);
	static void Split(const vstd::string& str, char sign, vstd::vector<vstd::string_view>& v);
	static bool CheckStringIsInteger(const char* beg, const char* end);
	static bool CheckStringIsFloat(const char* beg, const char* end);
	static void ToLower(vstd::string& str);
	static void ToUpper(vstd::string& str);
	static void CullCharacater(vstd::string const& source, vstd::string& dest, std::initializer_list<char> const& lists);
	static void CullCharacater(vstd::string_view const& source, vstd::string& dest, std::initializer_list<char> const& lists);
	static void CullCodeSpace(vstd::string const& source, vstd::string& dest)
	{
		CullCharacater(
			source, dest,
			{
				' ', '\t', '\r', '\n', '\\'
			}
		);
	}
	static bool IsCharSpace(char c);
	static bool IsCharNumber(char c);
	static bool IsCharCharacter(char c);
	static bool IsCharAvaliableCode(char c);
	static bool ReadStringFromFile(vstd::string const& path, vstd::string& data);
	template <uint chrLen>
	static constexpr bool CompareCharArray(char const* first, char const* second)
	{
		for (uint i = 0; i < chrLen; ++i)
		{
			if (first[i] != second[i]) return false;
		}
		return true;
	}
	static bool CompareCharArray(char const* first, char const* second, size_t chrLen);
	static bool CompareCharArray(
		char const* first,
		char const* second,
		char const* secondEnd
	);
	static bool CompareCharArray(
		char const* first,
		size_t len,
		char const* second,
		char const* secondEnd
	);
	static void SampleCodeFile(vstd::string const& fileData, vstd::vector<CodeChunk>& results, bool separateCodeAndString = true, bool disposeComment = false);
	static void SeparateString(
		vstd::string const& str,
		funcPtr_t<bool(char const*, char const*)> judgeFunc,
		vstd::vector<std::pair<vstd::string, bool>>& vec
	);

	static void SeparateString(
		vstd::string const& str,
		funcPtr_t<bool(char const*, char const*)> judgeFunc,
		vstd::vector<vstd::string>& vec
	);
	static int64_t StringToInt(const vstd::string& str);
	static int64_t StringToInt(const char* start, const char* end);
	KILL_COPY_CONSTRUCT(StringUtil)
};