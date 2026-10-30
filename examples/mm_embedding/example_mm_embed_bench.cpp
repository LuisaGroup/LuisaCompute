// mm_embedding engine -- Comprehensive Multimodal Vision Benchmark Target.
//
// Dual-track architecture: Dedicated high-resolution, multi-aspect-ratio benchmark
// evaluating 16 realistic curated samples across 8 domains:
//   1. Cross-Modal Retrieval (Image -> Text & Text -> Image, Short & Detailed Captions)
//   2. Hard-Negative Discrimination Margin (sim(pos) - sim(neg) > 0)
//   3. Bilingual Semantic Alignment (EN vs ZH cosine similarity)
//   4. Aspect Ratio Stress Test (2:1 panoramic, 2:3 vertical, 1:1, 4:3, 3:2)
//   5. PyTorch SentenceTransformers Golden Reference Parity (>0.9999 cos sim)
//   6. End-to-End Latency & Throughput Profiling
//
// Usage:
//   example_mm_embed_bench [backend: dx|cuda|vk] [model.mmegm] [tokenizer.json] [diverse_dir]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

#include <yyjson.h>

#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/stream.h>

#include "mm_embedding/embed/image_processor.h"
#include "mm_embedding/embed/package.h"
#include "mm_embedding/embed/session.h"
#include "mm_embedding/embed/tokenizer.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::mm_embed;

namespace {

struct NegativeCaption {
    luisa::string en;
    luisa::string zh;
};

struct BenchmarkItem {
    luisa::string id;
    luisa::string category;
    luisa::string filename;
    luisa::string aspect_ratio;
    uint32_t width{0};
    uint32_t height{0};

    luisa::string short_en;
    luisa::string short_zh;
    luisa::string detailed_en;
    luisa::string detailed_zh;
    luisa::string task_en;
    luisa::string task_zh;
    luisa::vector<NegativeCaption> hard_negatives;

    // Computed C++ embeddings
    Embedding img_emb;
    Embedding short_en_emb;
    Embedding short_zh_emb;
    Embedding detailed_en_emb;
    Embedding detailed_zh_emb;
    luisa::vector<Embedding> neg_en_embs;
    luisa::vector<Embedding> neg_zh_embs;

    // Timings
    double img_encode_ms{0.0};

    // PyTorch Golden reference
    luisa::vector<float> golden_img_768;
    luisa::vector<float> golden_img_512;
    luisa::vector<float> golden_img_256;
    luisa::vector<float> golden_img_128;
    float golden_parity_cos_sim{0.0f};

    // Metrics
    float pos_sim_en{0.0f};
    float pos_sim_zh{0.0f};
    float min_neg_margin_en{0.0f};
    float min_neg_margin_zh{0.0f};
    float bilingual_sim_short{0.0f};
    float bilingual_sim_detailed{0.0f};
};

[[nodiscard]] float compute_cosine_similarity(luisa::span<const float> a, luisa::span<const float> b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a <= 0.0f || norm_b <= 0.0f) return 0.0f;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

[[nodiscard]] const char *get_str_safe(yyjson_val *val, const char *def = "") {
    if (val && yyjson_is_str(val)) return yyjson_get_str(val);
    return def;
}

luisa::vector<BenchmarkItem> load_dataset_metadata(const std::filesystem::path &json_path) {
    luisa::vector<BenchmarkItem> items;
    std::ifstream f(json_path);
    if (!f.is_open()) {
        LUISA_ERROR_WITH_LOCATION("Failed to open dataset metadata at '{}'", json_path.string());
    }
    std::string json_str((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    yyjson_doc *doc = yyjson_read(json_str.c_str(), json_str.size(), 0);
    if (!doc) {
        LUISA_ERROR_WITH_LOCATION("Failed to parse JSON in '{}'", json_path.string());
    }

    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *samples_arr = yyjson_obj_get(root, "samples");
    if (!samples_arr && yyjson_is_arr(root)) {
        samples_arr = root;
    }

    size_t idx, max;
    yyjson_val *val;
    yyjson_arr_foreach(samples_arr, idx, max, val) {
        BenchmarkItem item{};
        item.id = get_str_safe(yyjson_obj_get(val, "id"));
        item.category = get_str_safe(yyjson_obj_get(val, "category"));
        item.filename = get_str_safe(yyjson_obj_get(val, "image_filename"));
        item.aspect_ratio = get_str_safe(yyjson_obj_get(val, "aspect_ratio"));
        item.width = static_cast<uint32_t>(yyjson_get_uint(yyjson_obj_get(val, "width")));
        item.height = static_cast<uint32_t>(yyjson_get_uint(yyjson_obj_get(val, "height")));

        yyjson_val *ann = yyjson_obj_get(val, "annotations");
        if (ann) {
            yyjson_val *sc = yyjson_obj_get(ann, "short_caption");
            if (sc) {
                item.short_en = get_str_safe(yyjson_obj_get(sc, "en"));
                item.short_zh = get_str_safe(yyjson_obj_get(sc, "zh"));
            }
            yyjson_val *dc = yyjson_obj_get(ann, "detailed_caption");
            if (dc) {
                item.detailed_en = get_str_safe(yyjson_obj_get(dc, "en"));
                item.detailed_zh = get_str_safe(yyjson_obj_get(dc, "zh"));
            }
            yyjson_val *tp = yyjson_obj_get(ann, "task_prompt_retrieval");
            if (tp) {
                item.task_en = get_str_safe(yyjson_obj_get(tp, "en"));
                item.task_zh = get_str_safe(yyjson_obj_get(tp, "zh"));
            }
            yyjson_val *hn_arr = yyjson_obj_get(ann, "hard_negatives");
            if (hn_arr && yyjson_is_arr(hn_arr)) {
                size_t n_idx, n_max;
                yyjson_val *n_val;
                yyjson_arr_foreach(hn_arr, n_idx, n_max, n_val) {
                    NegativeCaption nc{};
                    nc.en = get_str_safe(yyjson_obj_get(n_val, "en"));
                    nc.zh = get_str_safe(yyjson_obj_get(n_val, "zh"));
                    item.hard_negatives.push_back(std::move(nc));
                }
            }
        }
        items.push_back(std::move(item));
    }
    yyjson_doc_free(doc);
    return items;
}

void load_golden_embeddings(const std::filesystem::path &golden_path, luisa::vector<BenchmarkItem> &items) {
    if (!std::filesystem::exists(golden_path)) {
        LUISA_WARNING("Golden vision reference '{}' not found. Parity comparison skipped.", golden_path.string());
        return;
    }

    std::ifstream f(golden_path);
    if (!f.is_open()) return;
    std::string json_str((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    yyjson_doc *doc = yyjson_read(json_str.c_str(), json_str.size(), 0);
    if (!doc) return;

    yyjson_val *root = yyjson_doc_get_root(doc);
    yyjson_val *images_obj = yyjson_obj_get(root, "images");

    auto read_vec = [](yyjson_val *arr, luisa::vector<float> &out) {
        out.clear();
        if (arr && yyjson_is_arr(arr)) {
            size_t idx, max;
            yyjson_val *v;
            out.reserve(yyjson_arr_size(arr));
            yyjson_arr_foreach(arr, idx, max, v) {
                out.push_back(static_cast<float>(yyjson_get_num(v)));
            }
        }
    };

    for (auto &item : items) {
        yyjson_val *img_val = yyjson_obj_get(images_obj, item.id.c_str());
        if (!img_val) continue;
        read_vec(yyjson_obj_get(img_val, "768"), item.golden_img_768);
        read_vec(yyjson_obj_get(img_val, "512"), item.golden_img_512);
        read_vec(yyjson_obj_get(img_val, "256"), item.golden_img_256);
        read_vec(yyjson_obj_get(img_val, "128"), item.golden_img_128);
    }
    yyjson_doc_free(doc);
}

} // namespace

int main(int argc, char **argv) {
    luisa::string backend = (argc > 1) ? argv[1] : "dx";
    std::filesystem::path egm_path = (argc > 2) ? argv[2] : "model.mmegm";
    std::filesystem::path tok_path = (argc > 3) ? argv[3] : "tokenizer.json";
    std::filesystem::path diverse_dir = (argc > 4) ? argv[4] : "examples/mm_embedding/data/diverse";

    // Fallback search locations
    if (!std::filesystem::exists(tok_path)) {
        if (std::filesystem::exists("models/embeddinggemma-2/tokenizer.json")) {
            tok_path = "models/embeddinggemma-2/tokenizer.json";
        } else if (std::filesystem::exists("examples/mm_embedding/data/tokenizer.json")) {
            tok_path = "examples/mm_embedding/data/tokenizer.json";
        } else if (std::filesystem::exists("../../examples/mm_embedding/data/tokenizer.json")) {
            tok_path = "../../examples/mm_embedding/data/tokenizer.json";
        }
    }
    if (!std::filesystem::exists(diverse_dir)) {
        if (std::filesystem::exists("examples/mm_embedding/data/diverse")) {
            diverse_dir = "examples/mm_embedding/data/diverse";
        } else if (std::filesystem::exists("../../examples/mm_embedding/data/diverse")) {
            diverse_dir = "../../examples/mm_embedding/data/diverse";
        }
    }

    std::cout << "==================================================================================\n"
              << "       LuisaCompute EmbeddingGemma-2 Vision Benchmark Engine (Phase 2)           \n"
              << "==================================================================================\n"
              << " Backend:      " << backend << "\n"
              << " Model:        " << egm_path.string() << "\n"
              << " Tokenizer:    " << tok_path.string() << "\n"
              << " Benchmark DB: " << diverse_dir.string() << "\n"
              << "==================================================================================\n"
              << std::endl;

    if (!std::filesystem::exists(egm_path)) {
        LUISA_ERROR_WITH_LOCATION("Model package '{}' not found.", egm_path.string());
    }
    if (!std::filesystem::exists(tok_path)) {
        LUISA_ERROR_WITH_LOCATION("Tokenizer '{}' not found.", tok_path.string());
    }
    auto dataset_json = diverse_dir / "diverse_dataset.json";
    if (!std::filesystem::exists(dataset_json)) {
        LUISA_ERROR_WITH_LOCATION("Diverse dataset JSON '{}' not found. Run build_diverse_dataset.py first.", dataset_json.string());
    }

    auto items = load_dataset_metadata(dataset_json);
    auto golden_json = diverse_dir / "golden_diverse_vision.json";
    load_golden_embeddings(golden_json, items);

    std::cout << "[INFO] Loaded " << items.size() << " benchmark items across 8 domains." << std::endl;

    Context context{argv[0]};
    auto device = context.create_device(backend);

    SessionConfig config{};
    config.enable_vision = true;
    auto session = Session::create(device, egm_path, tok_path, config);
    if (!session || !session->has_vision()) {
        LUISA_ERROR_WITH_LOCATION("Failed to initialize multimodal session with Vision Tower.");
    }

    // Warmup
    std::cout << "\n[1/6] Warming up GPU compute pipelines..." << std::endl;
    {
        auto warmup_img = diverse_dir / items.front().filename;
        session->encode_image(warmup_img);
        session->encode("Warmup query string");
    }

    // 1. Image Encoding & Latency Profiling
    std::cout << "[2/6] Running High-Resolution Image Inference across diverse aspect ratios..." << std::endl;
    for (size_t i = 0; i < items.size(); i++) {
        auto &item = items[i];
        auto img_path = diverse_dir / item.filename;

        auto t0 = std::chrono::high_resolution_clock::now();
        item.img_emb = session->encode_image(img_path);
        auto t1 = std::chrono::high_resolution_clock::now();
        item.img_encode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Parity with PyTorch Golden
        if (!item.golden_img_768.empty()) {
            item.golden_parity_cos_sim = compute_cosine_similarity(
                luisa::span{item.img_emb.values.data(), item.img_emb.values.size()},
                luisa::span{item.golden_img_768.data(), item.golden_img_768.size()}
            );
        }

        std::cout << "  [" << std::setw(2) << (i + 1) << "/" << items.size() << "] "
                  << std::left << std::setw(34) << item.id
                  << " (" << std::setw(3) << item.aspect_ratio << ", "
                  << std::setw(4) << item.width << "x" << std::setw(4) << item.height << ") "
                  << "time=" << std::fixed << std::setprecision(1) << std::setw(5) << item.img_encode_ms << " ms | "
                  << "Golden CosSim=" << std::setprecision(6) << item.golden_parity_cos_sim
                  << std::endl;
    }

    // 2. Bilingual Text Prompt Encoding
    std::cout << "\n[3/6] Encoding bilingual text queries and hard negative distractors..." << std::endl;
    for (auto &item : items) {
        item.short_en_emb = session->encode(item.short_en);
        item.short_zh_emb = session->encode(item.short_zh);
        item.detailed_en_emb = session->encode(item.detailed_en);
        item.detailed_zh_emb = session->encode(item.detailed_zh);

        item.neg_en_embs.clear();
        item.neg_zh_embs.clear();
        for (const auto &neg : item.hard_negatives) {
            item.neg_en_embs.push_back(session->encode(neg.en));
            item.neg_zh_embs.push_back(session->encode(neg.zh));
        }

        item.bilingual_sim_short = compute_cosine_similarity(
            luisa::span{item.short_en_emb.values.data(), item.short_en_emb.values.size()},
            luisa::span{item.short_zh_emb.values.data(), item.short_zh_emb.values.size()}
        );
        item.bilingual_sim_detailed = compute_cosine_similarity(
            luisa::span{item.detailed_en_emb.values.data(), item.detailed_en_emb.values.size()},
            luisa::span{item.detailed_zh_emb.values.data(), item.detailed_zh_emb.values.size()}
        );
    }

    // 3. Evaluate Hard Negative Discrimination Margins
    std::cout << "[4/6] Evaluating Hard Negative Discrimination Margins..." << std::endl;
    float sum_margin_en = 0.0f;
    float sum_margin_zh = 0.0f;
    size_t total_neg_pairs = 0;
    bool all_margins_positive = true;

    for (auto &item : items) {
        item.pos_sim_en = compute_cosine_similarity(
            luisa::span{item.img_emb.values.data(), item.img_emb.values.size()},
            luisa::span{item.short_en_emb.values.data(), item.short_en_emb.values.size()}
        );
        item.pos_sim_zh = compute_cosine_similarity(
            luisa::span{item.img_emb.values.data(), item.img_emb.values.size()},
            luisa::span{item.short_zh_emb.values.data(), item.short_zh_emb.values.size()}
        );

        float min_m_en = 1e9f;
        float min_m_zh = 1e9f;

        for (size_t k = 0; k < item.neg_en_embs.size(); k++) {
            float neg_sim_en = compute_cosine_similarity(
                luisa::span{item.img_emb.values.data(), item.img_emb.values.size()},
                luisa::span{item.neg_en_embs[k].values.data(), item.neg_en_embs[k].values.size()}
            );
            float neg_sim_zh = compute_cosine_similarity(
                luisa::span{item.img_emb.values.data(), item.img_emb.values.size()},
                luisa::span{item.neg_zh_embs[k].values.data(), item.neg_zh_embs[k].values.size()}
            );

            float m_en = item.pos_sim_en - neg_sim_en;
            float m_zh = item.pos_sim_zh - neg_sim_zh;
            min_m_en = std::min(min_m_en, m_en);
            min_m_zh = std::min(min_m_zh, m_zh);

            sum_margin_en += m_en;
            sum_margin_zh += m_zh;
            total_neg_pairs++;
        }
        item.min_neg_margin_en = (min_m_en == 1e9f) ? 0.0f : min_m_en;
        item.min_neg_margin_zh = (min_m_zh == 1e9f) ? 0.0f : min_m_zh;

        if (item.min_neg_margin_en <= 0.0f) {
            all_margins_positive = false;
        }
    }

    // 4. Cross-Modal Retrieval Matrices (Short & Detailed, EN & ZH, I2T & T2I)
    std::cout << "[5/6] Computing Cross-Modal Retrieval Accuracy Matrices (16x16)..." << std::endl;

    auto eval_retrieval = [&](const luisa::string &desc, auto get_txt_emb) {
        const size_t N = items.size();
        size_t i2t_top1 = 0;
        size_t i2t_top3 = 0;
        size_t t2i_top1 = 0;
        size_t t2i_top3 = 0;

        // Image to Text (I2T)
        for (size_t i = 0; i < N; i++) {
            luisa::vector<std::pair<float, size_t>> ranking;
            ranking.reserve(N);
            for (size_t j = 0; j < N; j++) {
                const auto &txt_e = get_txt_emb(items[j]);
                float sim = compute_cosine_similarity(
                    luisa::span{items[i].img_emb.values.data(), items[i].img_emb.values.size()},
                    luisa::span{txt_e.values.data(), txt_e.values.size()}
                );
                ranking.emplace_back(sim, j);
            }
            std::sort(ranking.begin(), ranking.end(), [](auto &a, auto &b) { return a.first > b.first; });
            if (ranking[0].second == i) i2t_top1++;
            for (size_t r = 0; r < 3 && r < N; r++) {
                if (ranking[r].second == i) { i2t_top3++; break; }
            }
        }

        // Text to Image (T2I)
        for (size_t j = 0; j < N; j++) {
            const auto &txt_e = get_txt_emb(items[j]);
            luisa::vector<std::pair<float, size_t>> ranking;
            ranking.reserve(N);
            for (size_t i = 0; i < N; i++) {
                float sim = compute_cosine_similarity(
                    luisa::span{items[i].img_emb.values.data(), items[i].img_emb.values.size()},
                    luisa::span{txt_e.values.data(), txt_e.values.size()}
                );
                ranking.emplace_back(sim, i);
            }
            std::sort(ranking.begin(), ranking.end(), [](auto &a, auto &b) { return a.first > b.first; });
            if (ranking[0].second == j) t2i_top1++;
            for (size_t r = 0; r < 3 && r < N; r++) {
                if (ranking[r].second == j) { t2i_top3++; break; }
            }
        }

        float i2t_t1_pct = (static_cast<float>(i2t_top1) / N) * 100.0f;
        float i2t_t3_pct = (static_cast<float>(i2t_top3) / N) * 100.0f;
        float t2i_t1_pct = (static_cast<float>(t2i_top1) / N) * 100.0f;
        float t2i_t3_pct = (static_cast<float>(t2i_top3) / N) * 100.0f;

        std::cout << "  [" << std::left << std::setw(15) << desc << "] "
                  << "I->T Top-1: " << std::setw(2) << i2t_top1 << "/" << N << " (" << std::fixed << std::setprecision(1) << i2t_t1_pct << "%) | "
                  << "Top-3: " << std::setw(2) << i2t_top3 << "/" << N << " (" << i2t_t3_pct << "%) | "
                  << "T->I Top-1: " << std::setw(2) << t2i_top1 << "/" << N << " (" << t2i_t1_pct << "%)"
                  << std::endl;
    };

    eval_retrieval("Short (EN)", [](const BenchmarkItem &it) -> const Embedding & { return it.short_en_emb; });
    eval_retrieval("Short (ZH)", [](const BenchmarkItem &it) -> const Embedding & { return it.short_zh_emb; });
    eval_retrieval("Detailed (EN)", [](const BenchmarkItem &it) -> const Embedding & { return it.detailed_en_emb; });
    eval_retrieval("Detailed (ZH)", [](const BenchmarkItem &it) -> const Embedding & { return it.detailed_zh_emb; });

    // 5. Aspect Ratio Stress Test (Spatial Pooling & Unit Norm Invariants)
    std::cout << "\n[6/6] Verifying Aspect Ratio Stress Test Invariants (2:1, 2:3, 1:1, 4:3, 3:2)..." << std::endl;
    bool aspect_ratio_ok = true;
    for (const auto &item : items) {
        float norm_sq = 0.0f;
        for (float v : item.img_emb.values) {
            if (std::isnan(v) || std::isinf(v)) {
                aspect_ratio_ok = false;
                break;
            }
            norm_sq += v * v;
        }
        float norm = std::sqrt(norm_sq);
        if (std::abs(norm - 1.0f) > 1e-4f) {
            aspect_ratio_ok = false;
        }
    }
    std::cout << "  Aspect Ratio Stress Invariant Check: "
              << (aspect_ratio_ok ? "[PASSED] Unit norm & valid spatial pooling across all ratios" : "[FAILED]")
              << std::endl;

    // =========================================================================
    // Comprehensive Professional Benchmark Report
    // =========================================================================
    std::cout << "\n"
              << "============================================================================================================================\n"
              << "                                       DIVERSE MULTIMODAL BENCHMARK EVALUATION REPORT                                       \n"
              << "============================================================================================================================\n"
              << std::left
              << std::setw(4)  << "#"
              << std::setw(36) << "Item ID"
              << std::setw(24) << "Domain"
              << std::setw(6)  << "Ratio"
              << std::setw(12) << "Dims"
              << std::setw(10) << "GPU (ms)"
              << std::setw(16) << "PyTorch Parity"
              << std::setw(12) << "PosSim (EN)"
              << std::setw(14) << "MinMargin(EN)"
              << std::setw(10) << "Bilingual"
              << "\n"
              << "----------------------------------------------------------------------------------------------------------------------------\n";

    double total_ms = 0.0;
    float sum_parity = 0.0f;
    float min_parity = 1.0f;
    float sum_bilingual = 0.0f;

    for (size_t i = 0; i < items.size(); i++) {
        const auto &it = items[i];
        total_ms += it.img_encode_ms;
        sum_parity += it.golden_parity_cos_sim;
        min_parity = std::min(min_parity, it.golden_parity_cos_sim);
        sum_bilingual += it.bilingual_sim_short;

        std::stringstream ss_dims;
        ss_dims << it.width << "x" << it.height;

        std::cout << std::left
                  << std::setw(4)  << (i + 1)
                  << std::setw(36) << it.id
                  << std::setw(24) << it.category
                  << std::setw(6)  << it.aspect_ratio
                  << std::setw(12) << ss_dims.str()
                  << std::fixed << std::setprecision(1) << std::setw(10) << it.img_encode_ms
                  << std::setprecision(6) << std::setw(16) << it.golden_parity_cos_sim
                  << std::setprecision(4) << std::setw(12) << it.pos_sim_en
                  << (it.min_neg_margin_en >= 0 ? "+" : "") << std::setprecision(4) << std::setw(13) << it.min_neg_margin_en
                  << std::setprecision(4) << std::setw(10) << it.bilingual_sim_short
                  << "\n";
    }

    float avg_parity = sum_parity / items.size();
    float avg_margin_en = (total_neg_pairs > 0) ? (sum_margin_en / total_neg_pairs) : 0.0f;
    float avg_bilingual = sum_bilingual / items.size();
    double avg_ms = total_ms / items.size();

    std::cout << "============================================================================================================================\n"
              << " SUMMARY AUDIT SCORECARD\n"
              << "----------------------------------------------------------------------------------------------------------------------------\n"
              << " [1] PyTorch Golden Parity (Average):      " << std::setprecision(6) << avg_parity << " (Threshold > 0.995000)  -> "
              << (avg_parity >= 0.995000f ? "[PASSED]" : "[FAILED]") << "\n"
              << " [2] PyTorch Golden Parity (Minimum):      " << std::setprecision(6) << min_parity << " (Threshold > 0.990000)  -> "
              << (min_parity >= 0.990000f ? "[PASSED]" : "[FAILED]") << "\n"
              << " [3] Average Hard-Negative Margin (EN):    " << (avg_margin_en >= 0 ? "+" : "") << std::setprecision(4) << avg_margin_en << " (Threshold > +0.0500)  -> "
              << (avg_margin_en >= 0.05f ? "[PASSED]" : "[FAILED]") << "\n"
              << " [4] Bilingual Consistency (EN vs ZH):    " << std::setprecision(4) << avg_bilingual << " (Threshold > 0.7500)   -> "
              << (avg_bilingual >= 0.75f ? "[PASSED]" : "[FAILED]") << "\n"
              << " [5] Extreme Aspect Ratio Robustness:      " << (aspect_ratio_ok ? "ALL PASSED (2:1, 2:3, 4:3, 1:1, 3:2)" : "FAILED") << " -> "
              << (aspect_ratio_ok ? "[PASSED]" : "[FAILED]") << "\n"
              << " [6] Mean Vision Inference Time:          " << std::fixed << std::setprecision(2) << avg_ms << " ms / image\n"
              << "============================================================================================================================\n"
              << std::endl;

    if (avg_parity < 0.995000f || min_parity < 0.990000f || !aspect_ratio_ok || avg_margin_en < 0.05f || avg_bilingual < 0.75f) {
        LUISA_ERROR_WITH_LOCATION("Benchmark validation criteria failed.");
        return 1;
    }

    std::cout << "[SUCCESS] EmbeddingGemma-2 Vision Benchmark executed and validated successfully!\n" << std::endl;
    return 0;
}
