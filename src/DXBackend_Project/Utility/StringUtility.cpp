
#include "StringUtility.h"
#include "../Common/BitArray.h"
void StringUtil::IndicesOf(const vengine::string& str, const vengine::string& sign, vengine::vector<uint>& v) {
	v.clear();
	if (str.empty()) return;
	int count = str.length() - sign.length() + 1;
	v.reserve(10);
	for (int i = 0; i < count; ++i) {
		bool success = true;
		for (int j = 0; j < sign.length(); ++j) {
			if (sign[j] != str[i + j]) {
				success = false;
				break;
			}
		}
		if (success)
			v.push_back(i);
	}
}

void StringUtil::SplitCodeString(
	char const* beg,
	char const* end,
	vengine::vector<vengine::string_view>& results,
	void* ptr,
	funcPtr_t<CharCutState(void*, char)> func) {
	for (char const* i = beg; i < end; ++i) {
		CharCutState state = func(ptr, *i);
		switch (state) {
			case CharCutState::Different:
				if (beg < i)
					results.emplace_back(beg, i);
				beg = i;
				break;
			case CharCutState::Cull:
				if (beg < i)
					results.emplace_back(beg, i);
				beg = i + 1;
				break;
		}
	}
	if (beg < end)
		results.emplace_back(beg, end);
}

void StringUtil::IndicesOf(const vengine::string& str, char sign, vengine::vector<uint>& v) {
	v.clear();
	int count = str.length();
	v.reserve(10);
	for (int i = 0; i < count; ++i) {
		if (sign == str[i]) {
			v.push_back(i);
		}
	}
}

void StringUtil::CutToLine(const char* str, const char* end, vengine::vector<vengine::string>& lines) {
	lines.clear();
	lines.reserve(32);
	vengine::vector<char> c;
	c.reserve(32);
	for (char const* ite = str; ite < end; ++ite) {
		if (*ite == '\n') {
			if (!c.empty()) {
				lines.emplace_back(c.data(), c.data() + c.size());
				c.clear();
			}
		} else if (*ite != '\r')
			c.push_back(*ite);
	}
	if (!c.empty()) {
		lines.emplace_back(c.data(), c.data() + c.size());
	}
}

void StringUtil::CutToLine_ContainEmptyLine(const char* str, const char* end, vengine::vector<vengine::string>& lines) {
	lines.clear();
	lines.reserve(32);
	vengine::vector<char> c;
	c.reserve(32);
	for (char const* ite = str; ite < end; ++ite) {
		if (*ite == '\n') {
			if (!c.empty()) {
				lines.emplace_back(c.data(), c.data() + c.size());
				c.clear();
			} else {
				lines.emplace_back();
			}
		} else if (*ite != '\r')
			c.push_back(*ite);
	}
	if (!c.empty()) {
		lines.emplace_back(c.data(), c.data() + c.size());
	}
}

void StringUtil::CutToLine_ContainEmptyLine(const char* str, const char* end, vengine::vector<vengine::string_view>& lines) {
	lines.clear();
	lines.reserve(32);
	char const* start = str;
	for (char const* ite = str; ite < end; ++ite) {
		if (*ite == '\n') {
			if (start != ite) {
				if (*(ite - 1) == '\r') {
					auto a = ite - 1;
					lines.emplace_back(start, a);
				} else
					lines.emplace_back(start, ite);
				start = ite + 1;
			} else {
				lines.emplace_back();
			}
		}
	}
	if (start != end) {
		if (*(end - 1) == '\r') {
			auto a = end - 1;
			if (start != a)
				lines.emplace_back(start, a);
		} else
			lines.emplace_back(start, end);
	}
}

void StringUtil::CutToLine(const char* str, const char* end, vengine::vector<vengine::string_view>& lines) {
	lines.clear();
	lines.reserve(32);
	char const* start = str;
	for (char const* ite = str; ite < end; ++ite) {
		if (*ite == '\n') {
			if (start != ite) {
				if (*(ite - 1) == '\r') {
					auto a = ite - 1;
					if (start != a)
						lines.emplace_back(start, a);
				} else
					lines.emplace_back(start, ite);
				start = ite + 1;
			}
		}
	}
	if (start != end) {
		if (*(end - 1) == '\r') {
			auto a = end - 1;
			if (start != a)
				lines.emplace_back(start, a);
		} else
			lines.emplace_back(start, end);
	}
}

void StringUtil::CutToLine(const vengine::string& str, vengine::vector<vengine::string>& lines) {
	return CutToLine(str.data(), str.data() + str.size(), lines);
}

void StringUtil::CutToLine_ContainEmptyLine(const vengine::string& str, vengine::vector<vengine::string>& lines) {
	return CutToLine_ContainEmptyLine(str.data(), str.data() + str.size(), lines);
}

void StringUtil::CutToLine(const vengine::string& str, vengine::vector<vengine::string_view>& lines) {
	return CutToLine(str.data(), str.data() + str.size(), lines);
}
void StringUtil::CutToLine_ContainEmptyLine(const vengine::string& str, vengine::vector<vengine::string_view>& lines) {
	return CutToLine_ContainEmptyLine(str.data(), str.data() + str.size(), lines);
}

void StringUtil::ReadLines(std::ifstream& ifs, vengine::vector<vengine::string>& lines) {
	ifs.seekg(0, std::ios::end);
	int64_t size = ifs.tellg();
	ifs.seekg(0, std::ios::beg);
	vengine::vector<char> buffer(size);
	memset(buffer.data(), 0, size);
	ifs.read(buffer.data(), size);
	for (int64_t sz = buffer.size() - 1; sz >= 0; --sz) {
		if (buffer[sz] != '\0') {
			buffer.resize(sz + 1);
			break;
		}
	}
	CutToLine(buffer.data(), buffer.data() + buffer.size(), lines);
}

int StringUtil::GetFirstIndexOf(const vengine::string& str, char sign) {
	int count = str.length();
	for (int i = 0; i < count; ++i) {
		if (sign == str[i]) {
			return i;
		}
	}
	return -1;
}

int StringUtil::GetFirstIndexOf(const vengine::string& str, const vengine::string& sign) {
	int count = str.length() - sign.length() + 1;
	for (int i = 0; i < count; ++i) {
		bool success = true;
		for (int j = 0; j < sign.length(); ++j) {
			if (sign[j] != str[i + j]) {
				success = false;
				break;
			}
		}
		if (success)
			return i;
	}
	return -1;
}

void StringUtil::Split(const vengine::string& str, char sign, vengine::vector<vengine::string>& v) {
	vengine::vector<uint> indices;
	IndicesOf(str, sign, indices);
	v.clear();
	v.reserve(10);
	vengine::string s;
	s.reserve(str.size());
	uint startPos = 0;
	for (auto index = indices.begin(); index != indices.end(); ++index) {
		s.clear();
		s.push_back_all(&str[startPos], *index - startPos);
		startPos = *index + 1;
		if (!s.empty())
			v.push_back(s);
	}
	s.clear();
	s.push_back_all(&str[startPos], str.length() - startPos);
	if (!s.empty())
		v.push_back(s);
}

void StringUtil::Split(const vengine::string& str, char sign, vengine::vector<vengine::string_view>& v) {
	vengine::vector<uint> indices;
	IndicesOf(str, sign, indices);
	v.clear();
	v.reserve(10);
	vengine::string_view s;
	int startPos = 0;
	for (auto index = indices.begin(); index != indices.end(); ++index) {
		s = vengine::string_view(&str[startPos], *index - startPos);
		startPos = *index + 1;
		if (s.size() > 0)
			v.push_back(s);
	}
	s = vengine::string_view(&str[startPos], str.length() - startPos);
	if (s.size() > 0)
		v.push_back(s);
}

void StringUtil::Split(const vengine::string& str, const vengine::string& sign, vengine::vector<vengine::string>& v) {
	vengine::vector<uint> indices;
	IndicesOf(str, sign, indices);
	v.clear();
	v.reserve(10);
	vengine::string s;
	s.reserve(str.size());
	uint startPos = 0;
	for (auto index = indices.begin(); index != indices.end(); ++index) {
		s.clear();
		s.push_back_all(&str[startPos], *index - startPos);
		startPos = *index + 1;
		if (!s.empty())
			v.push_back(s);
	}
	s.clear();
	s.push_back_all(&str[startPos], str.length() - startPos);
	if (!s.empty())
		v.push_back(s);
}

void StringUtil::Split(const vengine::string& str, const vengine::string& sign, vengine::vector<vengine::string_view>& v) {
	vengine::vector<uint> indices;
	IndicesOf(str, sign, indices);
	v.clear();
	v.reserve(10);
	vengine::string_view s;
	int startPos = 0;
	for (auto index = indices.begin(); index != indices.end(); ++index) {
		s = vengine::string_view(&str[startPos], *index - startPos);
		startPos = *index + 1;
		if (s.size() > 0)
			v.push_back(s);
	}
	s = vengine::string_view(&str[startPos], str.length() - startPos);
	if (s.size() > 0)
		v.push_back(s);
}
inline constexpr void mtolower(char& c) {
	if ((c >= 'A') && (c <= 'Z'))
		c = c + ('a' - 'A');
}
inline constexpr void mtoupper(char& c) {
	if ((c >= 'a') && (c <= 'z'))
		c = c + ('A' - 'a');
}

void StringUtil::ToLower(vengine::string& str) {
	char* c = str.data();
	const uint size = str.length();
	for (uint i = 0; i < size; ++i) {
		mtolower(c[i]);
	}
}
bool StringUtil::CheckStringIsInteger(const char* beg, const char* end) {
	if (end - beg == 0) return true;
	if (*beg == '-') beg++;
	for (const char* i = beg; i != end; ++i) {
		if (*i < '0' || *i > '9') return false;
	}
	return true;
}
bool StringUtil::CheckStringIsFloat(const char* beg, const char* end) {
	if (end - beg == 0) return true;
	if (*beg == '-') beg++;
	const char* i = beg;
	for (; i != end; ++i) {
		if (*i == '.') {
			++i;
			break;
		}
		if (*i < '0' || *i > '9') return false;
	}
	for (; i != end; ++i) {
		if (*i < '0' || *i > '9') return false;
	}
	return true;
}
void StringUtil::ToUpper(vengine::string& str) {
	char* c = str.data();
	const uint size = str.length();
	for (uint i = 0; i < size; ++i) {
		mtoupper(c[i]);
	}
}
void StringUtil::CullCharacater(vengine::string const& source, vengine::string& dest, std::initializer_list<char> const& lists) {
	BitArray bit(256);
	for (auto i : lists) {
		bit[(uint8_t)i] = 1;
	}
	dest.clear();
	char* end = source.data() + source.size();
	char* last = source.data();
	size_t sz = 0;
	for (char* ite = source.data(); ite != end; ++ite) {
		if (bit[(uint8_t)*ite]) {
			if (sz > 0) {
				dest.push_back_all(last, sz);
				sz = 0;
			}
			last = ite + 1;
		} else {
			sz++;
		}
	}
	if (sz > 0) {
		dest.push_back_all(last, sz);
	}
}
void StringUtil::CullCharacater(vengine::string_view const& source, vengine::string& dest, std::initializer_list<char> const& lists) {
	BitArray bit(256);
	for (auto i : lists) {
		bit[(uint8_t)i] = 1;
	}
	dest.clear();
	char const* end = source.end();
	char const* last = source.begin();
	size_t sz = 0;
	for (char const* ite = last; ite != end; ++ite) {
		if (bit[(uint8_t)*ite]) {
			if (sz > 0) {
				dest.push_back_all(last, sz);
				sz = 0;
			}
			last = ite + 1;
		} else {
			sz++;
		}
	}
	if (sz > 0) {
		dest.push_back_all(last, sz);
	}
}

struct CharControl {
	BitArray spaceChar;

	CharControl() : spaceChar(256) {
		spaceChar[(uint8_t)' '] = 1;
		spaceChar[(uint8_t)'\t'] = 1;
		spaceChar[(uint8_t)'\r'] = 1;
		spaceChar[(uint8_t)'\n'] = 1;
		spaceChar[(uint8_t)'\\'] = 1;
	}
};
bool StringUtil::IsCharSpace(char c) {
	static CharControl charControl;
	return charControl.spaceChar[(uint8_t)c];
}
bool StringUtil::IsCharNumber(char c) {
	return c >= '0' && c <= '9';
}
bool StringUtil::IsCharCharacter(char c) {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
bool StringUtil::IsCharAvaliableCode(char c) {
	char cs[] = {' ', '\t', '\r', '\n', '\\'};
	for (auto& i : cs) {
		if (i == c) return false;
	}
	return true;
}
bool StringUtil::ReadStringFromFile(vengine::string const& path, vengine::string& data) {
	std::ifstream ifs(path.c_str());
	if (!ifs) return false;
	ifs.seekg(0, std::ios::end);
	auto sz = ifs.tellg();
	ifs.seekg(0, std::ios::beg);
	data.clear();
	data.resize(sz);
	memset(data.data(), 0, data.size());
	ifs.read(data.data(), data.size());
	for (int64_t sz = data.size() - 1; sz >= 0; --sz) {
		if (data[sz] != '\0') {
			data.resize(sz + 1);
			return true;
		}
	}
	return true;
}
namespace StringUtilGlobal {
template<bool containType>
void InputCommentFunc(
	vengine::vector<StringUtil::CodeChunk>& vec,
	StringUtil::CodeType type,
	char const*& lastPtr,
	char const*& ite,
	char const* end,
	char const* leftSign,
	size_t const leftSignSize,
	char const* rightSign,
	size_t const rightSignSize) {
	if (end - ite < leftSignSize) return;
	if (StringUtil::CompareCharArray(ite, leftSign, leftSignSize)) {
		if (
			reinterpret_cast<int64_t>(ite) - reinterpret_cast<int64_t>(lastPtr) > 0) {
			vec.emplace_back(lastPtr, ite, StringUtil::CodeType::Code);
		}
		char const* commentStart = ite;
		ite += leftSignSize;
		char const* iteEnd = end - (rightSignSize - 1);
		while (ite < iteEnd) {
			if (StringUtil::CompareCharArray(ite, rightSign, rightSignSize)) {
				ite += rightSignSize;
				if constexpr (containType)
					auto&& code = vec.emplace_back(commentStart, ite, type);
				lastPtr = ite;
				return;
			}
			ite++;
		}
		if (
			reinterpret_cast<int64_t>(end) - reinterpret_cast<int64_t>(commentStart) > 0) {
			vec.emplace_back(commentStart, end, type);
		}
		lastPtr = end;
	}
}
}// namespace StringUtilGlobal

bool StringUtil::CompareCharArray(char const* first, char const* second, size_t chrLen) {
	return BinaryEqualTo_Size(first, second, chrLen);
}
bool StringUtil::CompareCharArray(
	char const* first,
	char const* second,
	char const* secondEnd) {
	auto len = strlen(first);
	if (secondEnd - second < len) return false;
	return CompareCharArray(first, second, len);
}
bool StringUtil::CompareCharArray(
	char const* first,
	size_t len,
	char const* second,
	char const* secondEnd) {
	if (secondEnd - second < len) return false;
	return CompareCharArray(first, second, len);
}

void StringUtil::SampleCodeFile(vengine::string const& fileData, vengine::vector<CodeChunk>& results, bool separateCodeAndString, bool disposeComment) {
	using namespace StringUtilGlobal;
	results.clear();
	if (fileData.empty()) return;
	char const* begin = fileData.data();
	char const* end = fileData.data() + fileData.size();
	char const* codeStart = begin;
	auto CollectTailFunc = [&](vengine::vector<CodeChunk>& res) -> void {
		if (
			reinterpret_cast<int64_t>(end) - reinterpret_cast<int64_t>(codeStart) > 0) {
			res.emplace_back(codeStart, end, CodeType::Code);
		}
	};
	if (separateCodeAndString) {
		vengine::vector<CodeChunk> codeAndComments;
		for (char const* ite = begin; ite < end; ++ite) {
			if (disposeComment) {
				InputCommentFunc<false>(
					codeAndComments,
					CodeType::Comment,
					codeStart,
					ite,
					end,
					"//",
					2,
					"\n",
					1);
				InputCommentFunc<false>(
					codeAndComments,
					CodeType::Comment,
					codeStart,
					ite,
					end,
					"/*",
					2,
					"*/",
					2);
			} else {
				InputCommentFunc<true>(
					codeAndComments,
					CodeType::Comment,
					codeStart,
					ite,
					end,
					"//",
					2,
					"\n",
					1);
				InputCommentFunc<true>(
					codeAndComments,
					CodeType::Comment,
					codeStart,
					ite,
					end,
					"/*",
					2,
					"*/",
					2);
			}
		}
		CollectTailFunc(codeAndComments);
		for (auto& chunk : codeAndComments) {
			if (chunk.type == CodeType::Code) {
				begin = chunk.str.data();
				codeStart = begin;
				end = chunk.str.data() + chunk.str.size();
				for (char const* ite = begin; ite < end; ++ite) {
					InputCommentFunc<true>(
						results,
						CodeType::String,
						codeStart,
						ite,
						end,
						"\"",
						1,
						"\"",
						1);
					InputCommentFunc<true>(
						results,
						CodeType::String,
						codeStart,
						ite,
						end,
						"'",
						1,
						"'",
						1);
				}
				CollectTailFunc(results);
			} else {
				results.push_back(chunk);
			}
		}
	} else {
		for (char const* ite = begin; ite < end; ++ite) {
			if (disposeComment) {
				InputCommentFunc<false>(
					results,
					CodeType::Comment,
					codeStart,
					ite,
					end,
					"//",
					2,
					"\n",
					1);
				InputCommentFunc<false>(
					results,
					CodeType::Comment,
					codeStart,
					ite,
					end,
					"/*",
					2,
					"*/",
					2);
			} else {
				InputCommentFunc<true>(
					results,
					CodeType::Comment,
					codeStart,
					ite,
					end,
					"//",
					2,
					"\n",
					1);
				InputCommentFunc<true>(
					results,
					CodeType::Comment,
					codeStart,
					ite,
					end,
					"/*",
					2,
					"*/",
					2);
			}
		}
		CollectTailFunc(results);
	}
}
void StringUtil::SeparateString(
	vengine::string const& str,
	funcPtr_t<bool(char const*, char const*)> judgeFunc,
	vengine::vector<std::pair<vengine::string, bool>>& vec) {
	vec.clear();
	char const* lastBeg = str.data();
	bool flag = false;
	char const* end = str.data() + str.size();
	char const* ite;
	for (ite = lastBeg; ite < end; ++ite) {
		bool cur = judgeFunc(ite, end);
		if (cur != flag) {
			if (
				reinterpret_cast<int64_t>(ite) - reinterpret_cast<int64_t>(lastBeg) > 0) {
				auto&& pair = vec.emplace_back();
				pair.second = flag;
				pair.first.push_back_all(lastBeg, ite - lastBeg);
			}
			lastBeg = ite;
			flag = cur;
		}
	}
	if (
		reinterpret_cast<int64_t>(ite) - reinterpret_cast<int64_t>(lastBeg) > 0) {
		auto&& pair = vec.emplace_back();
		pair.second = flag;
		pair.first.push_back_all(lastBeg, ite - lastBeg);
	}
}
void StringUtil::SeparateString(
	vengine::string const& str,
	funcPtr_t<bool(char const*, char const*)> judgeFunc,
	vengine::vector<vengine::string>& vec) {
	vec.clear();
	char const* lastBeg = str.data();
	bool flag = false;
	char const* end = str.data() + str.size();
	char const* ite;
	for (ite = lastBeg; ite < end; ++ite) {
		bool cur = judgeFunc(ite, end);
		if (cur != flag) {
			if (flag && reinterpret_cast<int64_t>(ite) - reinterpret_cast<int64_t>(lastBeg) > 0) {
				vec.emplace_back(lastBeg, ite);
			}
			flag = cur;
			lastBeg = ite;
		}
	}
	if (flag && reinterpret_cast<int64_t>(ite) - reinterpret_cast<int64_t>(lastBeg) > 0) {
		vec.emplace_back(lastBeg, ite);
	}
}
int64_t StringUtil::StringToInt(const vengine::string& str) {
	if (str.empty()) return 0;
	int64_t v;
	int64_t result = 0;
	uint start = 0;
	if (str[0] == '-') {
		v = -1;
		start = 1;
	} else
		v = 1;
	for (; start < str.size(); ++start) {
		result = result * 10 + ((int)str[start] - 48);
	}
	return result * v;
}
int64_t StringUtil::StringToInt(const char* chr, const char* end) {
	size_t size = end - chr;
	if (size == 0) return 0;
	int64_t v;
	int64_t result = 0;
	size_t start = 0;
	if (*chr == '-') {
		v = -1;
		start = 1;
	} else
		v = 1;
	for (; start < size; ++start) {
		result = result * 10 + ((int)chr[start] - 48);
	}
	return result * v;
}