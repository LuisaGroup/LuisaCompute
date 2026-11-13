#pragma once
// Internal family entry points for the benchmark-support target.
#include "benchmark_common.h"
#include "ocr/models/pp_ocr_size.h"

namespace luisa::ocr::bench {

void append_ops_rows(compute::Device &device, const BenchmarkOptions &options, luisa::vector<Row> &rows);
void append_decode_rows(compute::Device &device, const BenchmarkOptions &options, luisa::vector<Row> &rows);
void append_cnn_rows(compute::Device &device, const BenchmarkOptions &options, luisa::vector<Row> &rows);
void append_layout_rows(compute::Device &device, const BenchmarkOptions &options, luisa::vector<Row> &rows);
void append_gdn_rows(compute::Device &device, const BenchmarkOptions &options, luisa::vector<Row> &rows);
void append_moe_rows(compute::Device &device, const BenchmarkOptions &options, luisa::vector<Row> &rows);
void append_model_rows(compute::Device &device, const BenchmarkOptions &options, luisa::vector<Row> &rows);

// Complete synthetic slices shared by family and L3 model catalogs.
Row run_ovis_row(compute::Device &device, const BenchmarkOptions &options, BenchmarkSuite suite,
                 luisa::string case_name, luisa::string series);
Row run_deepseek_row(compute::Device &device, const BenchmarkOptions &options, BenchmarkSuite suite,
                     luisa::string case_name, luisa::string series);
Row run_pp_row(compute::Device &device, const BenchmarkOptions &options, BenchmarkSuite suite,
               PpOcrSize size, luisa::string series, uint32_t tier);
void append_paddle_model_rows(compute::Device &device, const BenchmarkOptions &options, luisa::vector<Row> &rows);
void append_opendoc_model_rows(compute::Device &device, const BenchmarkOptions &options, luisa::vector<Row> &rows);
void append_p7_model_rows(compute::Device &device, const BenchmarkOptions &options, luisa::vector<Row> &rows);
void append_surya_model_rows(compute::Device &device, const BenchmarkOptions &options, luisa::vector<Row> &rows);

}// namespace luisa::ocr::bench
