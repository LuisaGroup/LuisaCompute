// OCR engine -- deterministic reading order implementation (Phase 5).
#include "ocr/output/reading_order.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include "ocr/output/ctc_decode.h"

namespace luisa::ocr::output {

namespace {

struct LineBounds {
    float x0{0}, y0{0}, x1{0}, y1{0};
    bool degenerate{false};
};

[[nodiscard]] LineBounds bounds_of(const ReadingOrderLine &line) noexcept {
    LineBounds b;
    if (!line.box_empty()) {
        b.x0 = line.x0;
        b.y0 = line.y0;
        b.x1 = line.x1;
        b.y1 = line.y1;
        return b;
    }
    if (!line.polygon.empty()) {
        b.x0 = b.x1 = line.polygon[0].first;
        b.y0 = b.y1 = line.polygon[0].second;
        for (const auto &p : line.polygon) {
            b.x0 = std::min(b.x0, p.first);
            b.y0 = std::min(b.y0, p.second);
            b.x1 = std::max(b.x1, p.first);
            b.y1 = std::max(b.y1, p.second);
        }
        if (b.x1 > b.x0 && b.y1 > b.y0) { return b; }
    }
    b.degenerate = true;
    return b;
}

[[nodiscard]] float center_y(const LineBounds &b) noexcept { return 0.5f * (b.y0 + b.y1); }
[[nodiscard]] float height_of(const LineBounds &b) noexcept { return b.y1 - b.y0; }

[[nodiscard]] float overlap_ratio(const LineBounds &a, const LineBounds &b) noexcept {
    const float ix0 = std::max(a.x0, b.x0);
    const float ix1 = std::min(a.x1, b.x1);
    const float inter = std::max(0.0f, ix1 - ix0);
    const float narrower = std::min(a.x1 - a.x0, b.x1 - b.x0);
    return narrower <= 0.0f ? 0.0f : inter / narrower;
}

}// namespace

[[nodiscard]] Status ReadingOrderOptions::validate() const {
    if (!std::isfinite(column_overlap_ratio) || column_overlap_ratio < 0.0f || column_overlap_ratio > 1.0f) {
        return Status::error(StatusCode::invalid_argument, "column_overlap_ratio must be in [0,1]", "column_overlap_ratio");
    }
    if (!std::isfinite(row_tolerance) || row_tolerance < 0.0f) {
        return Status::error(StatusCode::invalid_argument, "row_tolerance must be >= 0", "row_tolerance");
    }
    return Status::success();
}

[[nodiscard]] Result<luisa::vector<ReadingOrderLine>> order_lines(luisa::span<const ReadingOrderLine> lines,
                                                                  const ReadingOrderOptions &options) {
    using R = Result<luisa::vector<ReadingOrderLine>>;
    if (auto s = options.validate(); !s.ok()) { return R::failure(s); }
    if (lines.empty()) { return R::success({}); }

    struct Item {
        uint32_t index{0};
        LineBounds bounds;
    };
    luisa::vector<uint32_t> pages;
    for (const auto &line : lines) { pages.push_back(line.page_index); }
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());

    luisa::vector<ReadingOrderLine> ordered;
    ordered.reserve(lines.size());
    uint32_t next_order = 0;

    for (auto page : pages) {
        luisa::vector<Item> items;
        for (uint32_t i = 0; i < lines.size(); i++) {
            if (lines[i].page_index != page) { continue; }
            items.push_back(Item{i, bounds_of(lines[i])});
        }
        luisa::vector<Item> degenerate;
        luisa::vector<Item> normal;
        for (auto &it : items) {
            if (it.bounds.degenerate) { degenerate.push_back(it); } else { normal.push_back(it); }
        }

        float median_height = 0.0f;
        if (!normal.empty()) {
            luisa::vector<float> heights;
            heights.reserve(normal.size());
            for (const auto &it : normal) { heights.push_back(height_of(it.bounds)); }
            std::sort(heights.begin(), heights.end());
            median_height = heights[heights.size() / 2u];
        }
        const float row_band = options.row_tolerance * median_height;

        struct Column {
            float x0{0}, x1{0};
            luisa::vector<uint32_t> members;
        };
        luisa::vector<uint32_t> seed_order(normal.size());
        for (uint32_t i = 0; i < normal.size(); i++) { seed_order[i] = i; }
        std::stable_sort(seed_order.begin(), seed_order.end(), [&](uint32_t a, uint32_t b) {
            const auto &ba = normal[a].bounds;
            const auto &bb = normal[b].bounds;
            if (ba.x0 != bb.x0) { return ba.x0 < bb.x0; }
            if (ba.y0 != bb.y0) { return ba.y0 < bb.y0; }
            return normal[a].index < normal[b].index;
        });
        luisa::vector<Column> columns;
        for (auto idx : seed_order) {
            const auto &b = normal[idx].bounds;
            bool placed = false;
            for (auto &col : columns) {
                LineBounds cb;
                cb.x0 = col.x0;
                cb.x1 = col.x1;
                cb.y0 = b.y0;
                cb.y1 = b.y1;
                if (overlap_ratio(cb, b) >= options.column_overlap_ratio) {
                    col.x0 = std::min(col.x0, b.x0);
                    col.x1 = std::max(col.x1, b.x1);
                    col.members.push_back(idx);
                    placed = true;
                    break;
                }
            }
            if (!placed) {
                Column col;
                col.x0 = b.x0;
                col.x1 = b.x1;
                col.members.push_back(idx);
                columns.push_back(std::move(col));
            }
        }
        std::stable_sort(columns.begin(), columns.end(), [](const Column &a, const Column &b) {
            return a.x0 < b.x0;
        });

        for (auto &col : columns) {
            std::stable_sort(col.members.begin(), col.members.end(), [&](uint32_t a, uint32_t b) {
                const auto &ba = normal[a].bounds;
                const auto &bb = normal[b].bounds;
                const float ca = center_y(ba), cb = center_y(bb);
                if (ca != cb) { return ca < cb; }
                if (ba.x0 != bb.x0) { return ba.x0 < bb.x0; }
                return normal[a].index < normal[b].index;
            });
            size_t i = 0;
            while (i < col.members.size()) {
                size_t j = i + 1u;
                const float row_center = center_y(normal[col.members[i]].bounds);
                while (j < col.members.size() &&
                       std::abs(center_y(normal[col.members[j]].bounds) - row_center) <= row_band) {
                    j++;
                }
                std::stable_sort(col.members.begin() + static_cast<ptrdiff_t>(i),
                                 col.members.begin() + static_cast<ptrdiff_t>(j),
                                 [&](uint32_t a, uint32_t b) {
                                     const auto &ba = normal[a].bounds;
                                     const auto &bb = normal[b].bounds;
                                     if (ba.x0 != bb.x0) { return ba.x0 < bb.x0; }
                                     if (ba.y0 != bb.y0) { return ba.y0 < bb.y0; }
                                     return normal[a].index < normal[b].index;
                                 });
                i = j;
            }
            for (auto idx : col.members) {
                ReadingOrderLine line = lines[normal[idx].index];
                line.reading_order = next_order++;
                ordered.push_back(std::move(line));
            }
        }
        std::stable_sort(degenerate.begin(), degenerate.end(), [](const Item &a, const Item &b) {
            return a.index < b.index;
        });
        for (const auto &it : degenerate) {
            ReadingOrderLine line = lines[it.index];
            line.reading_order = next_order++;
            ordered.push_back(std::move(line));
        }
    }
    return R::success(std::move(ordered));
}

[[nodiscard]] luisa::string join_ordered_text(luisa::span<const ReadingOrderLine> ordered_lines) {
    luisa::vector<luisa::string> texts;
    texts.reserve(ordered_lines.size());
    for (const auto &line : ordered_lines) { texts.push_back(line.text); }
    return join_line_texts(texts);
}

}// namespace luisa::ocr::output