#pragma vengine_package vengine_database

#include <serde/config.h>
#include <serde/DatabaseInclude.h>
#include <serde/Parser/StateRecorder.h>
#include <vstl/StringUtility.h>
#include <serde/SimpleJsonValue.h>
namespace toolhub::db {

struct YamlStateKeys {
    parser::StateRecorder<bool> spaces;
    parser::StateRecorder<bool> endOfValue;
    parser::StateRecorder<bool> endOfValueKey;
    parser::StateRecorder<bool> isNumber;
    parser::StateRecorder<bool> numCheck;
    using BoolArrayType = typename parser::StateRecorder<bool>::ArrayType;
    YamlStateKeys()
        : spaces(
              [&](BoolArrayType &ptr) {
                  ptr[' '] = true;
                  ptr['\t'] = true;
              }),
          endOfValue([&](BoolArrayType &ptr) {
              ptr[','] = true;
              ptr[']'] = true;
              ptr['['] = true;
              ptr['}'] = true;
              ptr['{'] = true;
              ptr[':'] = true;
          }),
          endOfValueKey([&](BoolArrayType &ptr) {
              ptr[','] = true;
              ptr[':'] = true;
          }),
          isNumber([&](BoolArrayType &ptr) {
              for (auto i : vstd::range(48, 58)) {
                  ptr[i] = true;
              }
              ptr['-'] = true;
          }),
          numCheck(
              [&](BoolArrayType &ptr) {
                  std::string_view sv = "0123456789.eE-+"sv;
                  for (auto i : sv) {
                      ptr[i] = true;
                  }
              }) {
    }
};
class YamlParser {
public:
    static const size_t lineSpaces = 2;
    using MapType = vstd::variant<IJsonArray *, IJsonDict *>;
    using StackElement = std::tuple<std::string_view, MapType, bool>;
    vstd::vector<StackElement> stack;
    vstd::vector<std::pair<MapType, bool>> localStack;
    vstd::vector<std::pair<vstd::string, MapType>> valueStrStack;
    std::string_view lastLineComment;
    std::string_view fullText;
    YamlStateKeys recorders;
    size_t jumpLine = 0;
    YamlParser() {
    }
    char const *GetNextChar(char const *ptr, char const *end) {
        if (!ptr) return nullptr;
        auto &&s = recorders.spaces;
        while (ptr != end) {
            if (!s[*ptr]) return ptr;
            ptr++;
        }
        return nullptr;
    }

    vstd::string GetString(char const *&ptr, char const *end, size_t currLineSpace) {
        char const *start = ptr;
        while (ptr != end) {
            if (recorders.endOfValue[*ptr]) {
                return vstd::string(start, ptr);
            }
            ++ptr;
        }
        //TODO: switch line
        auto textEnd = fullText.data() + fullText.size();
        if (end == textEnd)
            return vstd::string(start, end);
        vstd::string value = vstd::string(start, end);
        ptr = end;
        char const *nextPtr = end;
        currLineSpace *= lineSpaces;
        while (true) {
            nextPtr = GetNextNewLine(ptr, textEnd);
            if (!nextPtr) return value;
            size_t spaceCount = 0;
            while (*ptr == ' ' && ptr != nextPtr) {
                spaceCount++;
                ptr++;
            }
            if (spaceCount <= currLineSpace)
                return value;
            jumpLine++;
            value << ' ' << std::string_view(ptr, nextPtr - ptr);
            ptr = nextPtr;
        }
    }
    std::string_view GetStringKey(char const *&ptr, char const *end) {
        char const *start = ptr;
        while (ptr != end) {
            if (recorders.endOfValueKey[*ptr]) {
                return std::string_view(start, ptr - start);
            }
            ++ptr;
        }
        return std::string_view(start, end - start);
    }

    char const *GetNextNewLine(char const *&ptr, char const *end) {
        while (*ptr == '\r' || *ptr == '\n') {
            if (ptr == end) return nullptr;
            ++ptr;
        }
        auto nextPtr = ptr;
        while (nextPtr != end) {
            if (*nextPtr == '\r' || *nextPtr == '\n') return nextPtr;
            ++nextPtr;
        }
        return end;
    }

    template<char beginEnd>
    bool ParseString(char const *&ptr, char const *const end, vstd::string &chars, vstd::string &errorStr) {
        ptr++;
        char const *start = ptr;
        bool isSlash = false;
        chars.reserve(32);
        auto func = [&]() {
            while (true) {
                if (isSlash) {
                    isSlash = false;
                    switch (*ptr) {
                        case '\\':
                            chars.push_back('\\');
                            break;
                        case 't':
                            chars.push_back('\t');
                            break;
                        case 'r':
                            chars.push_back('\r');
                            break;
                        case 'n':
                            chars.push_back('\n');
                            break;
                        case '\'':
                            chars.push_back('\'');
                            break;
                        case '\"':
                            chars.push_back('\"');
                            break;
                    }
                } else {
                    switch (*ptr) {
                        case '\\':

                            isSlash = true;
                            break;
                        case beginEnd:
                            return true;
                        default:
                            chars.push_back(*ptr);
                            break;
                    }
                }
                ++ptr;
                if (ptr == end) {
                    errorStr = ("Error end!");
                    return false;
                }
            }
            return true;
        };
        if (!func()) return false;
        ++ptr;
        return true;
    }
    void ClearOneStack(IJsonDatabase *db) {
        auto lastEle = stack.erase_last();
        auto &&lastParent = *(stack.end() - 1);
        auto setEle = [&](auto &&ele) {
            if (!std::get<1>(ele).valid()) {
                std::get<1>(ele) = db->CreateDict_RawPtr();
            }
        };
        setEle(lastEle);
        setEle(lastParent);
        std::get<1>(lastParent).multi_visit([&](IJsonArray *parentArr) {
				if (!parentArr) return;
				std::get<1>(lastEle).multi_visit(
					[&](auto&& arr) {
						if (arr) {
							auto newArr = db->CreateArray();
							newArr->Add(std::get<0>(lastEle));
							newArr->Add(vstd::make_unique(arr));
							parentArr->Add(std::move(newArr));
						}
					},
					[&](auto&& dict) {
						if (dict) {
							auto newArr = db->CreateArray();
							newArr->Add(std::get<0>(lastEle));
							newArr->Add(vstd::make_unique(dict));
							parentArr->Add(std::move(newArr));
						}
					}); }, [&](auto &&parentDictKV) {
				if (!parentDictKV) return;
			std::get<1>(lastEle).visit(
					[&](auto&& arr) {
						if (arr)
							parentDictKV->Set(std::get<0>(lastEle), vstd::make_unique(arr));
					}); });
    }
    void ClearStack(int64 targetStackSize, IJsonDatabase *db) {
        auto target = (int64)stack.size() - targetStackSize;
        if (target < 0) return;
        for (auto i : vstd::range(0, target)) {
            ClearOneStack(db);
        }
    }
    vstd::string combineStr;
    bool ProcessLine(char const *ptr, char const *end, vstd::string &errorMsg, IJsonDatabase *db) {
        //count spaces
        bool commentRefresh = false;
        auto start = ptr;
        size_t horiSize = 0;
        while (true) {
            if (ptr == end) {
                return true;
            }
            if (*ptr == '-') {
                if (++horiSize >= 3) {
                    ++ptr;
                    ptr = GetNextChar(ptr, end);
                    if (ptr != nullptr) {
                        lastLineComment = std::string_view(ptr, end - ptr);
                    }
                    return true;
                }

            } else if (*ptr == '%') {
                return true;
            } else if (*ptr != ' ' && *ptr != '\t') {
                break;
            }
            ptr++;
        }
        size_t spaceCount = ptr - start;
        spaceCount /= lineSpaces;
        spaceCount += 1;
        if (spaceCount > (localStack.size() + stack.size())) {
            errorMsg = "line error align";
            return false;
        }
        ClearStack(spaceCount, db);
        using ValueType = vstd::variant<
            vstd::string,
            int64,
            double,
            vstd::Guid,
            IJsonDict *,
            IJsonArray *>;

        auto getDictValue = [&](auto &&ele) -> WriteJsonVariant {
            if (!ele.valid()) return {};
            return ele.multi_visit_or(
                WriteJsonVariant(),
                [&](auto &&s) {
                    return std::move(s);
                },
                [&](auto &&s) {
                    return std::move(s);
                },
                [&](auto &&s) {
                    return std::move(s);
                },
                [&](auto &&s) {
                    return std::move(s);
                },
                [&](auto &&v) {
                    return vstd::make_unique(v);
                },
                [&](auto &&v) {
                    return vstd::make_unique(v);
                });
        };
        auto parseDict = [&](auto &parseKey, auto &parseValue) -> IJsonDict * {
            auto dict = db->CreateDict_RawPtr();
            localStack.emplace_back(dict, false);
            while (true) {
                ptr = GetNextChar(ptr, end);
                if (ptr == nullptr) {
                    return dict;
                } else if (*ptr == '}') {
                    ptr++;
                    localStack.erase_last();
                    return dict;
                } else if (*ptr == ',') {
                    (localStack.end() - 1)->second = true;
                    ptr++;
                    ptr = GetNextChar(ptr, end);
                    if (ptr == nullptr) return dict;
                    continue;
                } else {
                    if (!parseKey(parseKey, parseValue, dict)) {
                        errorMsg = "error value ending";
                        return nullptr;
                    }
                }
                (localStack.end() - 1)->second = false;
            }
            return dict;
        };
        auto parseArray = [&](auto &parseValue) -> IJsonArray * {
            auto arr = db->CreateArray_RawPtr();
            localStack.emplace_back(arr, false);
            while (true) {
                ptr = GetNextChar(ptr, end);
                auto vari = parseValue(parseValue);
                if (!vari.valid()) return nullptr;
                vari.multi_visit(
                    [&](auto &&s) { arr->Add(std::move(s)); },
                    [&](auto &&s) { arr->Add(std::move(s)); },
                    [&](auto &&s) { arr->Add(std::move(s)); },
                    [&](auto &&s) { arr->Add(std::move(s)); },
                    [&](auto &&s) {
                        arr->Add(vstd::make_unique(s));
                    },
                    [&](auto &&s) { arr->Add(vstd::make_unique(s)); });
                ptr = GetNextChar(ptr, end);
                if (ptr == nullptr) {
                    return arr;
                }
                auto &&pp = ptr;
                if (*ptr == ']') {
                    ptr++;
                    localStack.erase_last();
                    return arr;
                } else if (*ptr == ',') {
                    ptr++;
                    (localStack.end() - 1)->second = true;
                    continue;
                } else {
                    errorMsg = "error value ending";
                    return nullptr;
                }
                (localStack.end() - 1)->second = false;
            }
        };
        bool lineInit = false;
        auto parseKey = [&](auto &parseKey, auto &parseValue, MapType const &arrOrDict) -> bool {
            ptr = GetNextChar(ptr, end);
            if (ptr == nullptr) {
                errorMsg = "error ending";
                return false;
            }
            if (horiSize > 0) {
                if (*ptr == '{') {
                    lineInit = true;
                    ptr++;
                    auto parseResult = parseDict(parseKey, parseValue);
                    parseResult->Set(0, nullptr);
                    if (!parseResult) {
                        return false;
                    }
                    arrOrDict.multi_visit(
                        [&](auto &&arr) {
                            arr->Add(parseResult);
                        },
                        [&](auto &&dict) {

                        });
                    return true;
                } else if (*ptr == '[') {
                    ptr++;
                    auto parseResult = parseArray(parseValue);
                    if (!parseResult) {
                        return false;
                    }
                    arrOrDict.multi_visit(
                        [&](auto &&arr) {
                            arr->Add(parseResult);
                        },
                        [&](auto &&dict) {});
                    return true;
                }
            }
            auto strv = GetStringKey(ptr, end);
            if (lastLineComment.data() != nullptr) {
                combineStr.clear();
                combineStr << lastLineComment << '|' << strv;
                strv = combineStr;
                lastLineComment = std::string_view(nullptr, 0);
            }
            if (strv.empty()) {
                errorMsg = "error ending";
                return false;
            }

            if (ptr == end) {
                if (!horiSize) {
                    errorMsg = "error ending";
                    return false;
                }
                auto add = [&](auto &&ele) {
                    arrOrDict.multi_visit(
                        [&](auto &&arr) {
                            arr->Add(ele);
                        },
                        [&](auto &&dict) {});
                };

                auto num = vstd::StringUtil::StringToNumber(strv);
                if (num.valid()) {
                    num.visit(
                        [&](auto &&n) { add(n); });
                } else {
                    add(strv);
                }

                return true;
            } else if (*ptr == ':') {
                ptr++;
                ptr = GetNextChar(ptr, end);
                lineInit = true;
                if (ptr == nullptr) {
                    stack.emplace_back(strv, MapType{}, horiSize);
                } else {
                    ptr = GetNextChar(ptr, end);
                    auto ele = parseValue(parseValue);
                    if (ele.valid()) {
                        arrOrDict.multi_visit(
                            [&](auto &&arr) {
                                auto keyValueArray = db->CreateArray();
                                keyValueArray->Add(strv);
                                keyValueArray->Add(getDictValue(ele));
                                arr->Add(std::move(keyValueArray));
                            },
                            [&](IJsonDict *dict) {
                                dict->Set(strv, getDictValue(ele));
                            });
                    } else {
                        return false;
                    }
                }
                return true;
            } else {
                auto pp = ptr;
                errorMsg = "error key ending";
                return false;
            }
        };

        auto parseValue = [&](auto &parseValue) -> ValueType {
            if (!ptr) {
                errorMsg = "error ending";
                return {};
            }
            switch (*ptr) {
                //dict
                case '{': {
                    lineInit = true;
                    ++ptr;
                    auto d = parseDict(parseKey, parseValue);
                    d->Set(0, nullptr);
                    if (!d) return {};
                    return d;
                }
                case '[': {
                    lineInit = true;
                    ++ptr;
                    auto arr = parseArray(parseValue);
                    if (!arr) return {};
                    return arr;
                } break;
                case '\"': {
                    vstd::string result;
                    if (!ParseString<'\"'>(ptr, end, result, errorMsg)) {
                        return {};
                    }
                    return result;
                }
                case '\'': {
                    vstd::string result;
                    if (!ParseString<'\''>(ptr, end, result, errorMsg)) {
                        return {};
                    }
                    return result;
                }
                default: {
                    auto ss = GetString(ptr, end, spaceCount - 1);
                    if (ss.empty()) return nullptr;
                    if (ss.size() > 10) {
                        return ss;
                    } else {
                        auto num = vstd::StringUtil::StringToNumber(ss);

                        return num.visit_or(
                            vstd::MakeLazyEval([&] { return ValueType{std::move(ss)}; }),
                            [&](auto &&n) {
                                return n;
                            });
                    }
                }
            }
        };
        if (localStack.empty()) {
            auto &&arrOrDict = std::get<1>(stack[spaceCount - 1]);
            if (!arrOrDict.valid()) {
                if (horiSize)
                    arrOrDict = db->CreateArray_RawPtr();
                else
                    arrOrDict = db->CreateDict_RawPtr();
            }
            if (!parseKey(parseKey, parseValue, arrOrDict)) {
                return false;
            }
        } else {
            lineInit = true;
            while (!localStack.empty()) {
                auto last = localStack.end() - 1;
                ptr = GetNextChar(ptr, end);
                if (ptr == nullptr) return true;
                if (last->second) {
                    char endChar = ' ';
                    if (!last->first.multi_visit_or(
                            true,
                            [&](IJsonArray *arr) {
                                endChar = ']';
                                ptr = GetNextChar(ptr, end);
                                auto ele = parseValue(parseValue);
                                if (!ele.valid()) return false;
                                arr->Add(getDictValue(ele));
                                return true;
                            },
                            [&](IJsonDict *dict) {
                                endChar = '}';
                                return parseKey(parseKey, parseValue, last->first);
                            }))
                        return false;
                    ptr = GetNextChar(ptr, end);
                    if (ptr != nullptr) {
                        if (*ptr == ',') {
                            ptr++;
                        } else if (*ptr == endChar) {
                            ptr++;
                            localStack.erase(last);
                        } else {
                            localStack.erase(last);
                        }
                    }
                } else {
                    if (*ptr != ',') {
                        errorMsg = "error value ending";
                        return false;
                    }
                    ptr++;
                    last->second = true;
                    continue;
                }
            }
        }
        return true;
    }
    vstd::optional<ParsingException> Parse(std::string_view strv, IJsonDict *dict) {
        fullText = strv;
        stack.clear();
        stack.emplace_back(std::string_view(nullptr, 0), dict, false);
        auto ptr = strv.data();
        auto end = ptr + strv.size();
        vstd::string errorMsg;
        for (char const *nextPtr = nullptr; (nextPtr = GetNextNewLine(ptr, end)); ptr = nextPtr) {
            if (ptr == end) {
                break;
            }
            if (jumpLine) {
                jumpLine--;
                continue;
            }
            if (!ProcessLine(ptr, nextPtr, errorMsg, dict->GetDB())) {
                return std::move(errorMsg);
            }
        }
        ClearStack(1, dict->GetDB());
        return {};
    }
};
vstd::optional<ParsingException> SimpleJsonValueDict::ParseYaml(
    std::string_view str,
    bool clearLast) {
    if (clearLast) {
        Reset();
    }
    YamlParser parser;
    return parser.Parse(str, this);
}
}// namespace toolhub::db