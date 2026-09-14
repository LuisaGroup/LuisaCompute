#include "ngram_kernels.h"

#include "ngram_library.h"

#include <luisa/dsl/sugar.h>

namespace tokenize {

using namespace luisa;
using namespace luisa::compute;

// NOTE on DSL control flow: runtime branching/looping uses $if/$for with
// Var-typed conditions; there is no break, so loop bodies are guarded by
// flag variables ("done"/"found"). Scalar boolean logic uses bitwise & |
// (the DSL && || are vector-only and unary ! is unavailable).

RetrieveKernel make_retrieve_naive_kernel() noexcept {
    return [](BufferUInt lib_buf, Var<uint32_t> lib_len,
              BufferUInt query_buf, BufferUInt qlen_buf,
              Var<uint32_t> query_stride,
              BufferUInt draft_buf, BufferUInt draft_len_buf,
              Var<uint32_t> min_n, Var<uint32_t> max_n, Var<uint32_t> k,
              BufferUInt req_off_buf) noexcept {
        Var<uint32_t> qid = req_off_buf->read(kernel_id()) + dispatch_x();
        Var<uint32_t> qlen = qlen_buf->read(qid);

        // every draft slot starts invalid; every query gets a count
        $for (i, 0u, k) {
            draft_buf->write(qid * k + i, ngram_invalid_id);
        };
        draft_len_buf->write(qid, 0u);

        Var<uint32_t> best_n = 0u;
        Var<uint32_t> best_pos = 0u;
        Var<uint32_t> done = 0u;

        // longest n first: the first length with any match wins
        $for (nn, 0u, max_n - min_n + 1u) {
            Var<uint32_t> n = max_n - nn;
            // n must fit the query and leave >= 1 corpus token after it
            $if (done == 0u & n <= qlen & n < lib_len) {
                Var<uint32_t> found = 0u;
                Var<uint32_t> found_pos = 0u;
                // earliest p with p + n < lib_len
                $for (p, 0u, lib_len - n) {
                    $if (found == 0u) {
                        Var<bool> match = true;
                        $for (j, 0u, n) {
                            $if (match) {
                                match = lib_buf->read(p + j) ==
                                        query_buf->read(qid * query_stride + qlen - n + j);
                            };
                        };
                        $if (match) {
                            found = 1u;
                            found_pos = p;
                        };
                    };
                };
                $if (found != 0u) {
                    done = 1u;
                    best_n = n;
                    best_pos = found_pos;
                };
            };
        };

        $if (done != 0u) {
            Var<uint32_t> start = best_pos + best_n;
            Var<uint32_t> avail = lib_len - start;// >= 1 by construction
            Var<uint32_t> cnt = min(k, avail);
            $for (i, 0u, cnt) {
                draft_buf->write(qid * k + i, lib_buf->read(start + i));
            };
            draft_len_buf->write(qid, cnt);
        };
    };
}

RetrieveKernel make_retrieve_parallel_kernel(uint32_t block_size) noexcept {
    return [block_size](BufferUInt lib_buf, Var<uint32_t> lib_len,
                        BufferUInt query_buf, BufferUInt qlen_buf,
                        Var<uint32_t> query_stride,
                        BufferUInt draft_buf, BufferUInt draft_len_buf,
                        Var<uint32_t> min_n, Var<uint32_t> max_n, Var<uint32_t> k,
                        BufferUInt req_off_buf) noexcept {
        set_block_size(block_size, 1u, 1u);
        Var<uint32_t> qid = req_off_buf->read(kernel_id()) + block_id().x;
        Var<uint32_t> lane = thread_id().x;
        Var<uint32_t> qlen = qlen_buf->read(qid);

        Shared<uint32_t> s_suffix{ngram_max_suffix_tokens};
        Shared<uint32_t> s_pos{1u};

        Var<uint32_t> best_n = 0u;
        Var<uint32_t> best_pos = 0u;
        Var<uint32_t> done = 0u;

        $if (lane == 0u) {
            $for (i, 0u, k) {
                draft_buf->write(qid * k + i, ngram_invalid_id);
            };
            draft_len_buf->write(qid, 0u);
            s_pos.write(0u, ngram_invalid_id);
        };
        sync_block();

        // longest n first: the first length with any match wins
        $for (nn, 0u, max_n - min_n + 1u) {
            Var<uint32_t> n = max_n - nn;
            $if (done == 0u & n <= qlen & n < lib_len) {
                // stage the query suffix q[qlen-n .. qlen) into shared memory
                $for (idx, lane, n, block_size) {
                    s_suffix.write(idx, query_buf->read(qid * query_stride + qlen - n + idx));
                };
                $if (lane == 0u) {
                    s_pos.write(0u, ngram_invalid_id);
                };
                sync_block();
                // strided scan: thread `lane` checks positions lane, lane+bs, ...
                Var<bool> mine = false;
                Var<uint32_t> my_pos = 0u;
                $for (p, lane, lib_len - n, block_size) {
                    $if (mine == false) {
                        // first-token filter: nearly all corpus positions
                        // fail here, costing a single global read instead
                        // of a full n-token compare
                        $if (lib_buf->read(p) == s_suffix.read(0)) {
                            Var<bool> match = true;
                            $for (j, 1u, n) {
                                $if (match) {
                                    match = lib_buf->read(p + j) == s_suffix.read(j);
                                };
                            };
                            $if (match) {
                                mine = true;
                                my_pos = p;
                            };
                        };
                    };
                };
                // earliest match across the block wins
                $if (mine) {
                    s_pos.atomic(0u).fetch_min(my_pos);
                };
                sync_block();
                Var<uint32_t> pos = s_pos.read(0u);
                $if (pos != ngram_invalid_id) {
                    done = 1u;
                    best_n = n;
                    best_pos = pos;
                };
            };
        };

        // lane 0 extracts the draft tokens
        $if (done != 0u & lane == 0u) {
            Var<uint32_t> start = best_pos + best_n;
            Var<uint32_t> avail = lib_len - start;// >= 1 by construction
            Var<uint32_t> cnt = min(k, avail);
            $for (i, 0u, cnt) {
                draft_buf->write(qid * k + i, lib_buf->read(start + i));
            };
            draft_len_buf->write(qid, cnt);
        };
    };
}

RetrieveKernelHash make_retrieve_hash_kernel(uint32_t cap_log2) noexcept {
    return [cap_log2](BufferUInt lib_buf, Var<uint32_t> lib_len,
                      BufferUInt query_buf, BufferUInt qlen_buf,
                      Var<uint32_t> query_stride,
                      BufferUInt draft_buf, BufferUInt draft_len_buf,
                      Var<uint32_t> min_n, Var<uint32_t> max_n, Var<uint32_t> k,
                      BufferVar<uint64_t> keys_buf, BufferUInt pos_buf,
                      BufferUInt req_off_buf) noexcept {
        Var<uint32_t> qid = req_off_buf->read(kernel_id()) + dispatch_x();
        Var<uint32_t> qlen = qlen_buf->read(qid);

        // every draft slot starts invalid; every query gets a count
        $for (i, 0u, k) {
            draft_buf->write(qid * k + i, ngram_invalid_id);
        };
        draft_len_buf->write(qid, 0u);

        Var<uint32_t> best_n = 0u;
        Var<uint32_t> best_pos = 0u;
        Var<uint32_t> done = 0u;

        // longest n first: the first length present in the index wins
        $for (nn, 0u, max_n - min_n + 1u) {
            Var<uint32_t> n = max_n - nn;
            $if (done == 0u & n <= qlen) {
                // FNV-1a 64 over the query suffix q[qlen-n .. qlen),
                // byte-identical to the host-side index builder
                Var<uint64_t> key = 14695981039346656037ull;
                $for (j, 0u, n) {
                    Var<uint64_t> tok = cast<uint64_t>(
                        query_buf->read(qid * query_stride + qlen - n + j));
                    $for (b, 0u, 4u) {
                        key = (key ^ ((tok >> (b * 8)) & 0xFFull)) * 1099511628211ull;
                    };
                };
                $if (key == 0ull) {
                    key = 1ull;// reserve 0 for empty slots
                };
                // linear-probe the index; content-verify key collisions
                Var<uint32_t> found = 0u;
                Var<uint32_t> found_pos = 0u;
                Var<uint64_t> slot = (key * 0x9E3779B97F4A7C15ull) >> (64 - cap_log2);
                $while (found == 0u) {
                    Var<uint64_t> k2 = keys_buf->read(slot.cast<uint32_t>());
                    $if (k2 == key) {
                        Var<uint32_t> p = pos_buf->read(slot.cast<uint32_t>());
                        Var<bool> same = true;
                        $for (j, 0u, n) {
                            $if (same) {
                                same = lib_buf->read(p + j) ==
                                       query_buf->read(qid * query_stride + qlen - n + j);
                            };
                        };
                        $if (same) {
                            found = 1u;
                            found_pos = p;
                        };
                    };
                    $if (found == 0u) {
                        $if (k2 == 0ull) {
                            found = 2u;// empty slot: not in the corpus
                        };
                    };
                    $if (found == 0u) {
                        slot = (slot + 1ull) & ((1ull << cap_log2) - 1ull);
                    };
                };
                $if (found == 1u) {
                    done = 1u;
                    best_n = n;
                    best_pos = found_pos;
                };
            };
        };

        $if (done != 0u) {
            Var<uint32_t> start = best_pos + best_n;
            Var<uint32_t> avail = lib_len - start;// >= 1: indexed positions keep p + n < lib_len
            Var<uint32_t> cnt = min(k, avail);
            $for (i, 0u, cnt) {
                draft_buf->write(qid * k + i, lib_buf->read(start + i));
            };
            draft_len_buf->write(qid, cnt);
        };
    };
}

}// namespace tokenize
