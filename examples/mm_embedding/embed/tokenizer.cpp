// mm_embedding engine -- Gemma BPE Tokenizer implementation using yyjson.
#include "mm_embedding/embed/tokenizer.h"

#include <atomic>
#include <cstdio>
#include <queue>
#include <string_view>
#include <thread>

#include <luisa/core/logging.h>
#include <luisa/core/stl/format.h>
#include <luisa/core/stl/unordered_map.h>

#include <yyjson.h>

namespace luisa::mm_embed {

namespace {

constexpr uint32_t kDefaultBosId = 2u;
constexpr uint32_t kDefaultEosId = 1u;
constexpr uint32_t kDefaultPadId = 0u;
constexpr uint32_t kDefaultUnkId = 3u;
constexpr uint32_t kDefaultVocabSize = 262144u;
constexpr luisa::string_view kMetaspace = "\xe2\x96\x81";

class TokenizerImpl final : public Tokenizer {
public:
    luisa::unordered_map<luisa::string, uint32_t> vocab;
    luisa::unordered_map<uint32_t, luisa::string> id_to_piece;
    luisa::unordered_map<luisa::string, uint32_t> bpe_ranks;

    uint32_t byte_tokens[256]{};
    uint32_t _bos_id{kDefaultBosId};
    uint32_t _eos_id{kDefaultEosId};
    uint32_t _pad_id{kDefaultPadId};
    uint32_t _unk_id{kDefaultUnkId};
    uint32_t _vocab_size{kDefaultVocabSize};

    [[nodiscard]] uint32_t bos_id() const noexcept override { return _bos_id; }
    [[nodiscard]] uint32_t eos_id() const noexcept override { return _eos_id; }
    [[nodiscard]] uint32_t pad_id() const noexcept override { return _pad_id; }
    [[nodiscard]] uint32_t unk_id() const noexcept override { return _unk_id; }
    [[nodiscard]] uint32_t vocab_size() const noexcept override { return _vocab_size; }

    static luisa::string make_pair_key(luisa::string_view a, luisa::string_view b) {
        luisa::string key;
        key.reserve(a.size() + 1u + b.size());
        key.append(a.data(), a.size());
        key.push_back('\0');
        key.append(b.data(), b.size());
        return key;
    }

    [[nodiscard]] luisa::vector<uint32_t> encode(luisa::string_view utf8, uint32_t max_len) const override {
        luisa::vector<uint32_t> result;
        result.push_back(_bos_id);

        if (utf8.empty()) {
            if (result.size() < max_len) {
                result.push_back(_eos_id);
            }
            return result;
        }

        if (max_len > 0 && utf8.size() > max_len * 5u) {
            size_t safe_len = max_len * 5u;
            while (safe_len > 0 && (static_cast<uint8_t>(utf8[safe_len]) & 0xC0) == 0x80) {
                safe_len--;
            }
            utf8 = utf8.substr(0, safe_len);
        }

        // Convert string to initial pieces with byte fallback
        struct Node {
            luisa::string piece;
            int32_t prev{-1};
            int32_t next{-1};
            uint32_t version{0};
            bool deleted{false};
        };

        luisa::vector<Node> nodes;
        nodes.reserve(utf8.size() + 1u);

        size_t idx = 0;
        while (idx < utf8.size()) {
            if (utf8[idx] == ' ') {
                nodes.push_back(Node{luisa::string(kMetaspace.data(), kMetaspace.size()), -1, -1, 0, false});
                idx++;
                continue;
            }

            // UTF-8 lead byte length
            uint8_t b0 = static_cast<uint8_t>(utf8[idx]);
            size_t char_len = 1;
            if ((b0 & 0x80) == 0) {
                char_len = 1;
            } else if ((b0 & 0xE0) == 0xC0) {
                char_len = 2;
            } else if ((b0 & 0xF0) == 0xE0) {
                char_len = 3;
            } else if ((b0 & 0xF8) == 0xF0) {
                char_len = 4;
            }

            if (idx + char_len <= utf8.size()) {
                luisa::string_view cp(utf8.data() + idx, char_len);
                auto it = vocab.find(luisa::string(cp));
                if (it != vocab.end()) {
                    nodes.push_back(Node{luisa::string(cp.data(), cp.size()), -1, -1, 0, false});
                    idx += char_len;
                    continue;
                }
            }

            // Byte fallback
            char buf[16];
            snprintf(buf, sizeof(buf), "<0x%02X>", b0);
            nodes.push_back(Node{luisa::string(buf), -1, -1, 0, false});
            idx++;
        }

        if (nodes.empty()) {
            if (result.size() < max_len) result.push_back(_eos_id);
            return result;
        }

        int32_t num_nodes = static_cast<int32_t>(nodes.size());
        for (int32_t i = 0; i < num_nodes; i++) {
            nodes[i].prev = i - 1;
            nodes[i].next = (i + 1 < num_nodes) ? (i + 1) : -1;
        }

        struct MergeCandidate {
            uint32_t rank;
            int32_t left;
            int32_t right;
            uint32_t left_v;
            uint32_t right_v;
            bool operator>(const MergeCandidate &o) const noexcept {
                return rank > o.rank;
            }
        };

        // O(N log N) priority-queue-driven BPE merge
        std::priority_queue<MergeCandidate, std::vector<MergeCandidate>, std::greater<MergeCandidate>> pq;
        for (int32_t i = 0; i + 1 < num_nodes; i++) {
            auto key = make_pair_key(nodes[i].piece, nodes[i + 1].piece);
            auto it = bpe_ranks.find(key);
            if (it != bpe_ranks.end()) {
                pq.push(MergeCandidate{it->second, i, i + 1, 0, 0});
            }
        }

        while (!pq.empty()) {
            auto top = pq.top();
            pq.pop();
            int32_t l = top.left;
            int32_t r = top.right;
            if (nodes[l].deleted || nodes[r].deleted) continue;
            if (nodes[l].version != top.left_v || nodes[r].version != top.right_v) continue;
            if (nodes[l].next != r) continue;

            // Merge right node into left node
            nodes[l].piece.append(nodes[r].piece);
            nodes[r].deleted = true;
            nodes[l].next = nodes[r].next;
            if (nodes[r].next != -1) {
                nodes[nodes[r].next].prev = l;
            }
            nodes[l].version++;

            // Register newly formed adjacent pairs into priority queue
            if (nodes[l].prev != -1) {
                int32_t p = nodes[l].prev;
                auto key = make_pair_key(nodes[p].piece, nodes[l].piece);
                auto it = bpe_ranks.find(key);
                if (it != bpe_ranks.end()) {
                    pq.push(MergeCandidate{it->second, p, l, nodes[p].version, nodes[l].version});
                }
            }
            if (nodes[l].next != -1) {
                int32_t n = nodes[l].next;
                auto key = make_pair_key(nodes[l].piece, nodes[n].piece);
                auto it = bpe_ranks.find(key);
                if (it != bpe_ranks.end()) {
                    pq.push(MergeCandidate{it->second, l, n, nodes[l].version, nodes[n].version});
                }
            }
        }

        // Map active pieces to token IDs
        int32_t curr = 0;
        while (curr != -1 && nodes[curr].deleted) {
            curr = nodes[curr].next;
        }
        while (curr != -1) {
            if (!nodes[curr].deleted) {
                auto it = vocab.find(nodes[curr].piece);
                if (it != vocab.end()) {
                    result.push_back(it->second);
                } else {
                    result.push_back(_unk_id);
                }
                if (result.size() >= max_len) {
                    break;
                }
            }
            curr = nodes[curr].next;
        }

        if (result.size() < max_len) {
            result.push_back(_eos_id);
        }

        return result;
    }

    [[nodiscard]] luisa::vector<luisa::vector<uint32_t>> encode_batch(
        luisa::span<const luisa::string_view> texts,
        uint32_t max_len) const override {
        size_t n = texts.size();
        luisa::vector<luisa::vector<uint32_t>> batch(n);
        if (n <= 1) {
            for (size_t i = 0; i < n; i++) {
                batch[i] = encode(texts[i], max_len);
            }
            return batch;
        }

        unsigned int num_workers = std::min<unsigned int>(
            std::thread::hardware_concurrency(), static_cast<unsigned int>(n));
        if (num_workers <= 1) num_workers = 1;

        std::vector<std::thread> workers;
        workers.reserve(num_workers);
        std::atomic<size_t> next_idx{0};

        for (unsigned int w = 0; w < num_workers; w++) {
            workers.emplace_back([&]() {
                while (true) {
                    size_t i = next_idx.fetch_add(1, std::memory_order_relaxed);
                    if (i >= n) break;
                    batch[i] = encode(texts[i], max_len);
                }
            });
        }
        for (auto &t : workers) {
            t.join();
        }
        return batch;
    }
};

}// namespace

luisa::unique_ptr<Tokenizer> Tokenizer::load(const std::filesystem::path &tokenizer_json) {
    yyjson_read_err err;
    yyjson_doc *doc = yyjson_read_file(tokenizer_json.string().c_str(), 0, nullptr, &err);
    if (!doc) {
        LUISA_ERROR("Tokenizer::load: failed to parse JSON in '{}': {} at pos {}",
                    tokenizer_json.string(), err.msg, err.pos);
    }

    auto tok = luisa::make_unique<TokenizerImpl>();
    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *model = yyjson_obj_get(root, "model");
    if (!model) {
        yyjson_doc_free(doc);
        LUISA_ERROR("Tokenizer::load: missing 'model' in '{}'", tokenizer_json.string());
    }

    // Load vocabulary
    yyjson_val *vocab_obj = yyjson_obj_get(model, "vocab");
    if (vocab_obj && yyjson_is_obj(vocab_obj)) {
        size_t idx, max;
        yyjson_val *key, *val;
        yyjson_obj_foreach(vocab_obj, idx, max, key, val) {
            const char *piece = yyjson_get_str(key);
            uint32_t id = static_cast<uint32_t>(yyjson_get_uint(val));
            tok->vocab[piece] = id;
            tok->id_to_piece[id] = piece;
        }
    }

    tok->_vocab_size = static_cast<uint32_t>(tok->vocab.size());
    if (tok->_vocab_size == 0) {
        tok->_vocab_size = kDefaultVocabSize;
    }

    // Load added_tokens (HuggingFace special tokens)
    yyjson_val *added_arr = yyjson_obj_get(root, "added_tokens");
    if (added_arr && yyjson_is_arr(added_arr)) {
        size_t idx, max;
        yyjson_val *item;
        yyjson_arr_foreach(added_arr, idx, max, item) {
            yyjson_val *id_val = yyjson_obj_get(item, "id");
            yyjson_val *content_val = yyjson_obj_get(item, "content");
            if (id_val && content_val) {
                uint32_t id = static_cast<uint32_t>(yyjson_get_uint(id_val));
                const char *piece = yyjson_get_str(content_val);
                if (piece) {
                    tok->vocab[piece] = id;
                    tok->id_to_piece[id] = piece;
                }
            }
        }
    }

    // Load BPE merges (supports both [\"a\", \"b\"] arrays and \"a b\" strings)
    yyjson_val *merges_arr = yyjson_obj_get(model, "merges");
    if (merges_arr && yyjson_is_arr(merges_arr)) {
        size_t idx, max;
        yyjson_val *val;
        uint32_t rank = 0;
        yyjson_arr_foreach(merges_arr, idx, max, val) {
            if (yyjson_is_arr(val) && yyjson_arr_size(val) >= 2) {
                const char *a = yyjson_get_str(yyjson_arr_get(val, 0));
                const char *b = yyjson_get_str(yyjson_arr_get(val, 1));
                if (a && b) {
                    auto key = TokenizerImpl::make_pair_key(a, b);
                    tok->bpe_ranks[key] = rank++;
                }
            } else if (yyjson_is_str(val)) {
                const char *merge_str = yyjson_get_str(val);
                if (merge_str) {
                    std::string_view sv(merge_str);
                    auto space_pos = sv.find(' ');
                    if (space_pos != std::string_view::npos) {
                        auto a = sv.substr(0, space_pos);
                        auto b = sv.substr(space_pos + 1);
                        auto key = TokenizerImpl::make_pair_key(a, b);
                        tok->bpe_ranks[key] = rank++;
                    }
                }
            }
        }
    }

    // Initialize byte fallback tokens
    for (uint32_t b = 0; b < 256; b++) {
        char buf[16];
        snprintf(buf, sizeof(buf), "<0x%02X>", b);
        auto it = tok->vocab.find(buf);
        if (it != tok->vocab.end()) {
            tok->byte_tokens[b] = it->second;
        } else {
            tok->byte_tokens[b] = tok->_unk_id;
        }
    }

    // Lookup special tokens
    auto find_tok = [&](const char *name, uint32_t def_val) {
        auto it = tok->vocab.find(name);
        return it != tok->vocab.end() ? it->second : def_val;
    };
    tok->_bos_id = find_tok("<bos>", kDefaultBosId);
    tok->_eos_id = find_tok("<eos>", kDefaultEosId);
    tok->_pad_id = find_tok("<pad>", kDefaultPadId);
    tok->_unk_id = find_tok("<unk>", kDefaultUnkId);

    yyjson_doc_free(doc);

    LUISA_INFO("Tokenizer::load: successfully loaded '{}' (vocab={}, merges={}, bos={}, eos={}, pad={})",
               tokenizer_json.string(), tok->vocab.size(), tok->bpe_ranks.size(),
               tok->_bos_id, tok->_eos_id, tok->_pad_id);

    return tok;
}

}// namespace luisa::mm_embed
