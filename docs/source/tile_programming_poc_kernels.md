# Luisa Tile DSL: Kernel Syntax Gallery

- Status: syntax exploration, not an implemented API
- Goal: stress the execution, reduction, pipeline, memory, and layout model
  across representative neural-network kernels
- Default: values stay virtual Tile SSA; compiler planning decides whether and
  where to materialize them

The complete GEMM sketches are in
[tile_programming_poc.cpp](tile_programming_poc.cpp). The examples below use
`nest`, `subnest`, and `leaf` for execution handles. `tile` is reserved for
data `Tile<T, R>` values and tensor tiles, so no execution name suggests a
memory or hardware level.

There is no predefined axis vocabulary. A kernel creates fresh local dimension
symbols with `dim()` / `dims(...)`, relates separate views with `with_dims`, or
obtains an existing symbol with `value.dim<I>()`. A label such as `"feature"`
is diagnostic text only; all semantics use symbol identity and typed maps.

## 1. The reduction surface

The general region form updates an ordinary outer Tile variable:

~~~cpp
auto feature_dim = x.dim<1>();
auto sum = zeros<f32>(x.shape().without(feature_dim));

for (auto &item : nest.reduce(x.domain(feature_dim))) {
    sum += x.at(item);
}

use(sum); // no result() or yield in the C++ surface
~~~

`item` is a `ReduceScope`, exactly as `nest`, `subnest`, and `k` are scope
handles. The range body is captured once. On close, the compiler sees that the
outer `sum` is updated by canonical `+=`, makes it a ReduceOp state, and infers
the registered additive merge contract. Floating-point reassociation still
requires the selected math policy. `x.domain(feature_dim)` only constructs a
typed `IndexSet`; `x.at(item)` is a pure projection/reindex of the Tile.

The common pure form is shorter:

~~~cpp
auto feature_dim = x.dim<1>();
auto sum = reduce(x, feature_dim, add);
auto peak = reduce(x, feature_dim, maximum);
~~~

Both forms create the same region-shaped ReduceOp. The expression form must name
`add` or `maximum` because it has no body from which to infer the update.

An index-domain overload fuses arbitrary element generation with the fold:

~~~cpp
auto sum = scalar<f32>(0.0f);
for (auto &item : nest.reduce(range(0, count))) {
    auto i = item.index();
    sum += square(prediction[i] - target[i]);
}
~~~

`reduce` is nest-like but does not extend the spatial memory-owner prefix. Its
logical reduction domain may later be split across participants, serial steps,
and a merge tree. The primitive signature is
`nest.reduce(domain, custom_contract?, policy?)`; states and canonical built-in
contracts are inferred from direct outer-Tile updates.

## 2. Elementwise bias + GELU + residual

Common scalar operators lift directly to Tiles; `map` is not required:

~~~cpp
struct ElementwiseConfig {
    int32_t rows_per_nest;
};

auto make_bias_gelu_residual(ElementwiseConfig cfg) {
    return tile_kernel([=](TensorView<f16, 2> X,
                           TensorView<f16, 1> bias,
                           TensorView<f16, 2> residual,
                           TensorView<f16, 2> Y) {
        auto [row_dim, feature_dim] = dims("row", "feature");
        auto X_rf = X.with_dims(row_dim, feature_dim);
        auto bias_f = bias.with_dims(feature_dim);
        auto residual_rf = residual.with_dims(row_dim, feature_dim);
        auto Y_rf = Y.with_dims(row_dim, feature_dim);

        auto rows = X_rf.extent(row_dim);
        auto features = X_rf.extent(feature_dim);

        for (auto &nest : parallel(
                 ceil_div(rows, cfg.rows_per_nest))) {
            auto row0 = nest.index() * cfg.rows_per_nest;
            auto tile_shape = shape(
                row_dim(cfg.rows_per_nest), feature_dim(features));

            auto x = cast<f32>(X_rf.tile(
                                    coord(row0, 0),
                                    tile_shape,
                                    bounds::zero)
                                   .load());
            auto r = cast<f32>(residual_rf.tile(
                                           coord(row0, 0),
                                           tile_shape,
                                           bounds::zero)
                                          .load());
            auto b = cast<f32>(bias_f.tile(
                                       coord(0),
                                       shape(feature_dim(features)),
                                       bounds::assume)
                                      .load());

            // `b` broadcasts by its named feature axis. Physical layouts need
            // not match; scheduling inserts a Repartition only when required.
            auto y = gelu(x + b) + r;

            for (auto &leaf : nest.parallel(exec::infer)) {
                Y_rf.tile(
                     coord(row0, 0),
                     tile_shape,
                     bounds::predicate)
                    .store(cast<f16>(y));
            }
        }
    });
}
~~~

The TileIR elementwise scalar region contains the full expression DAG, so a
backend can fuse it into one loop or vector instruction sequence.

## 3. Sum, mean, maximum, and argmax

One kernel can exercise built-in and custom-state reducers without naming a
warp, subgroup, or shared allocation:

~~~cpp
auto make_row_statistics(ReductionConfig cfg) {
    return tile_kernel([=](TensorView<f16, 2> X,
                           TensorView<f32, 1> Sum,
                           TensorView<f32, 1> Mean,
                           TensorView<f32, 1> Peak,
                           TensorView<i32, 1> ArgMax) {
        auto [row_dim, feature_dim] = dims("row", "feature");
        auto X_rf = X.with_dims(row_dim, feature_dim);
        auto Sum_r = Sum.with_dims(row_dim);
        auto Mean_r = Mean.with_dims(row_dim);
        auto Peak_r = Peak.with_dims(row_dim);
        auto ArgMax_r = ArgMax.with_dims(row_dim);

        auto rows = X_rf.extent(row_dim);
        auto features = X_rf.extent(feature_dim);

        for (auto &nest : parallel(rows)) {
            auto row = nest.index();
            auto x = cast<f32>(X_rf.tile(
                                    coord(row, 0),
                                    shape(1, features),
                                    bounds::assume)
                                   .load()
                                   .squeeze(row_dim));

            auto sum = scalar<f32>(0.0f);
            for (auto &item : nest.reduce(x.domain(feature_dim))) {
                sum += x.at(item);
            }

            auto peak = reduce(x, feature_dim, maximum);

            auto best = scalar(ArgMaxState<f32>{
                .value = -infinity<f32>(),
                .index = 0});
            for (auto &candidate : nest.reduce(
                     x.domain(feature_dim),
                     argmax(tie::lowest_index))) {
                auto item = indexed_value(
                    x.at(candidate), candidate.index());
                best = argmax_push(best, item);
            }

            Sum_r[row] = sum;
            Mean_r[row] = sum / cast<f32>(features);
            Peak_r[row] = peak;
            ArgMax_r[row] = best.index;
        }
    });
}
~~~

`argmax(tie::lowest_index)` is a monoid over `{value, index}` with a stable tie
rule. It can use a tree without making the chosen index backend-dependent.
Storage replicas of `x` are never counted as extra logical elements.

## 4. Streaming and whole-tensor reduction

The reduction range can iterate explicit chunks. The body may itself contain a
Tile-level reduction:

~~~cpp
auto make_streaming_row_sum(StreamReductionConfig cfg) {
    return tile_kernel([=](TensorView<f32, 2> X,
                           TensorView<f32, 1> Y) {
        auto [row_dim, feature_dim] = dims("row", "feature");
        auto X_rf = X.with_dims(row_dim, feature_dim);
        auto Y_r = Y.with_dims(row_dim);
        auto rows = X_rf.extent(row_dim);
        auto features = X_rf.extent(feature_dim);

        for (auto &nest : parallel(rows)) {
            auto row = nest.index();
            auto sum = scalar<f32>(0.0f);

            for (auto &chunk : nest.reduce(
                     range(0, features, cfg.chunk_size))) {
                auto feature0 = chunk.index();
                auto x = X_rf.tile(
                              coord(row, feature0),
                              shape(1, cfg.chunk_size),
                              bounds::zero)
                             .load()
                             .squeeze(row_dim);
                sum += reduce(x, feature_dim, add);
            }

            Y_r[row] = sum;
        }
    });
}
~~~

The outer ReduceOp says that chunks may be evaluated and merged in any legal
tree. This is different from `serial`, which would preserve an arbitrary
ordered recurrence.

An ordinary GPU kernel cannot silently synchronize independent root instances.
A large global reduction therefore exposes a partial-reduction pass:

~~~cpp
auto make_sum_partials(int32_t elements_per_nest) {
    return tile_kernel([=](TensorView<f32, 1> input,
                           TensorView<f32, 1> partials) {
        auto count = input.extent(0);

        for (auto &nest : parallel(
                 ceil_div(count, elements_per_nest))) {
            auto begin = nest.index() * elements_per_nest;
            auto x = input.tile(
                              coord(begin),
                              shape(elements_per_nest),
                              bounds::zero)
                         .load();
            partials[nest.index()] =
                reduce(x, x.dim<0>(), add);
        }
    });
}
~~~

The host repeatedly JITs or reuses that pass until one value remains:

~~~cpp
auto count = input_count;
auto source = input.view();
auto destination = scratch_a.view();
auto write_a = true;
while (count > 1) {
    auto pass = device.jit(make_sum_partials(cfg.elements_per_nest));
    auto next_count = ceil_div(count, cfg.elements_per_nest);
    pass(source.subview(0, count),
         destination.subview(0, next_count));
    source = destination;
    write_a = !write_a;
    destination = write_a ? scratch_a.view() : scratch_b.view();
    count = next_count;
}
~~~

A future multi-dispatch `tile_program` may encapsulate this launch graph. A
single `tile_kernel` must instead use a target-proved cooperative collective,
an explicitly permitted atomic, or the visible partial-output strategy above.

## 5. MSE, MAE, Huber, and binary cross-entropy losses

Point losses are ordinary elementwise expressions followed by the same partial
reduction kernel:

~~~cpp
template<typename PointLoss>
auto make_point_loss_partials(LossConfig cfg, PointLoss point_loss) {
    return tile_kernel([=](TensorView<f32, 1> prediction,
                           TensorView<f32, 1> target,
                           TensorView<f32, 1> partials) {
        auto element_dim = dim("element");
        auto prediction_e = prediction.with_dims(element_dim);
        auto target_e = target.with_dims(element_dim);
        auto count = prediction_e.extent(element_dim);

        for (auto &nest : parallel(
                 ceil_div(count, cfg.elements_per_nest))) {
            auto begin = nest.index() * cfg.elements_per_nest;
            auto extent = shape(element_dim(cfg.elements_per_nest));
            auto p = prediction_e.tile(
                                   coord(begin), extent, bounds::zero)
                         .load();
            auto t = target_e.tile(
                              coord(begin), extent, bounds::zero)
                         .load();

            auto contribution = point_loss(p, t);
            auto element =
                iota<i32>(element_dim, cfg.elements_per_nest) + begin;
            contribution = select(element < count, contribution, 0.0f);
            partials[nest.index()] =
                reduce(contribution, element_dim, add);
        }
    });
}

auto mse = make_point_loss_partials(cfg, [](auto p, auto t) {
    return square(p - t);
});

auto mae = make_point_loss_partials(cfg, [](auto p, auto t) {
    return abs(p - t);
});

auto huber = make_point_loss_partials(cfg, [delta](auto p, auto t) {
    auto d = abs(p - t);
    return select(d <= delta,
                  0.5f * square(d),
                  delta * (d - 0.5f * delta));
});

// Numerically stable binary cross entropy from logits.
auto bce = make_point_loss_partials(cfg, [](auto logit, auto label) {
    return maximum(logit, 0.0f) - logit * label +
           log1p(exp(-abs(logit)));
});
~~~

The same `make_sum_partials` kernel finishes the scalar loss. Division by the
true element count happens once after the final sum, so padded zero
contributions do not bias a mean.

## 6. Sparse softmax cross-entropy

This kernel reduces classes inside each row, gathers the target class, then
reduces several row losses into one root partial:

~~~cpp
auto make_cross_entropy_partials(CrossEntropyConfig cfg) {
    return tile_kernel([=](TensorView<f16, 2> logits,
                           TensorView<i32, 1> labels,
                           TensorView<f32, 1> partials) {
        auto [row_dim, class_dim] = dims("row", "class");
        auto logits_rc = logits.with_dims(row_dim, class_dim);
        auto labels_r = labels.with_dims(row_dim);
        auto rows = logits_rc.extent(row_dim);
        auto classes = logits_rc.extent(class_dim);

        for (auto &nest : parallel(
                 ceil_div(rows, cfg.rows_per_nest))) {
            auto row0 = nest.index() * cfg.rows_per_nest;
            auto tile_shape = shape(
                row_dim(cfg.rows_per_nest), class_dim(classes));
            auto x = cast<f32>(logits_rc.tile(
                                       coord(row0, 0),
                                       tile_shape,
                                       bounds::zero)
                                      .load());
            auto target = labels_r.tile(
                                    coord(row0),
                                    shape(row_dim(cfg.rows_per_nest)),
                                    bounds::zero)
                              .load();

            auto row_max = reduce(x, class_dim, maximum);
            auto log_z = row_max +
                         log(reduce(exp(x - row_max),
                                    class_dim,
                                    add));
            auto target_logit = gather(x, target, class_dim);
            auto row_id = iota<i32>(row_dim, cfg.rows_per_nest) + row0;
            auto row_loss = select(row_id < rows,
                                   log_z - target_logit,
                                   0.0f);

            partials[nest.index()] =
                reduce(row_loss, row_dim, add);
        }
    });
}
~~~

Dimension-identity broadcasting makes `x - row_max` unambiguous. Neither the class
reduction nor the row reduction commits to one target collective.

## 7. Row softmax with explicit reduction regions

~~~cpp
struct SoftmaxConfig {
    int32_t rows_per_nest;
};

auto make_softmax(SoftmaxConfig cfg) {
    return tile_kernel([=](TensorView<f32, 2> X,
                           TensorView<f32, 2> Y) {
        auto [row_dim, feature_dim] = dims("row", "feature");
        auto X_rf = X.with_dims(row_dim, feature_dim);
        auto Y_rf = Y.with_dims(row_dim, feature_dim);
        auto rows = X_rf.extent(row_dim);
        auto features = X_rf.extent(feature_dim);

        for (auto &nest : parallel(
                 ceil_div(rows, cfg.rows_per_nest))) {
            auto row0 = nest.index() * cfg.rows_per_nest;
            auto tile_shape = shape(
                row_dim(cfg.rows_per_nest), feature_dim(features));
            auto x = X_rf.tile(
                          coord(row0, 0),
                          tile_shape,
                          bounds::zero)
                         .load();

            auto row_max = full<f32>(
                shape(row_dim(cfg.rows_per_nest)), -infinity<f32>());
            for (auto &item : nest.reduce(x.domain(feature_dim))) {
                row_max = maximum(row_max, x.at(item));
            }

            auto e = exp(x - row_max);
            auto row_sum = zeros<f32>(
                shape(row_dim(cfg.rows_per_nest)));
            for (auto &item : nest.reduce(e.domain(feature_dim))) {
                row_sum += e.at(item);
            }

            auto y = e / row_sum;
            for (auto &leaf : nest.parallel(exec::infer)) {
                Y_rf.tile(
                     coord(row0, 0),
                     tile_shape,
                     bounds::predicate)
                    .store(y);
            }
        }
    });
}
~~~

The result along the local `row_dim` may stay sharded. If the selected
implementation needs row statistics on several participants, their Distribution
gains an explicit replica fiber; those replicas are placement, not extra
reductions.

## 8. LayerNorm and RMSNorm

Welford's mergeable state demonstrates why ReduceOp owns a captured update
region rather than being limited to a fixed list of opcodes:

~~~cpp
auto make_layer_norm(NormConfig cfg) {
    return tile_kernel([=](TensorView<f16, 2> X,
                           TensorView<f16, 1> gamma,
                           TensorView<f16, 1> beta,
                           TensorView<f16, 2> Y) {
        auto [row_dim, feature_dim] = dims("row", "feature");
        auto X_rf = X.with_dims(row_dim, feature_dim);
        auto gamma_f = gamma.with_dims(feature_dim);
        auto beta_f = beta.with_dims(feature_dim);
        auto Y_rf = Y.with_dims(row_dim, feature_dim);
        auto rows = X_rf.extent(row_dim);
        auto features = X_rf.extent(feature_dim);

        for (auto &nest : parallel(rows)) {
            auto row = nest.index();
            auto x = cast<f32>(X_rf.tile(
                                    coord(row, 0),
                                    shape(1, features),
                                    bounds::assume)
                                   .load()
                                   .squeeze(row_dim));
            auto stats = scalar(WelfordState<f32>::identity());

            for (auto &item : nest.reduce(
                     x.domain(feature_dim), welford)) {
                stats = welford_push(stats, x.at(item));
            }

            auto g = cast<f32>(gamma_f.tile(
                                       coord(0),
                                       shape(features),
                                       bounds::assume)
                                      .load());
            auto b = cast<f32>(beta_f.tile(
                                      coord(0),
                                      shape(features),
                                      bounds::assume)
                                     .load());
            auto y = (x - stats.mean) *
                         rsqrt(stats.variance() + cfg.epsilon) *
                         g +
                     b;

            Y_rf.tile(
                 coord(row, 0),
                 shape(1, features),
                 bounds::assume)
                .squeeze(row_dim)
                .store(cast<f16>(y));
        }
    });
}
~~~

RMSNorm needs only the expression shorthand:

~~~cpp
auto square_sum = reduce(x * x, x.dim<0>(), add);
auto inv_rms = rsqrt(square_sum / cast<f32>(features) + epsilon);
auto y = x * inv_rms * gamma;
~~~

The compiler may fuse elementwise producers into the reduction input and fuse
normalization consumers after the collective.

## 9. FlashAttention-style online softmax

The online-softmax state is an ordered recurrence across key blocks, while each
block contains ordinary logical-axis reductions. Pipeline stages overlap future
loads with the recurrence when dependences allow:

~~~cpp
auto make_flash_attention(AttentionConfig cfg) {
    return tile_kernel([=](TensorView<f16, 4> Q,
                           TensorView<f16, 4> K,
                           TensorView<f16, 4> V,
                           TensorView<f16, 4> O) {
        auto [batch_dim, head_dim, query_dim, key_dim, channel_dim] =
            dims("batch", "head", "query", "key", "channel");
        auto Q_bhqd = Q.with_dims(
            batch_dim, head_dim, query_dim, channel_dim);
        auto K_bhkd = K.with_dims(
            batch_dim, head_dim, key_dim, channel_dim);
        auto V_bhkd = V.with_dims(
            batch_dim, head_dim, key_dim, channel_dim);
        auto O_bhqd = O.with_dims(
            batch_dim, head_dim, query_dim, channel_dim);

        auto batch = Q_bhqd.extent(batch_dim);
        auto heads = Q_bhqd.extent(head_dim);
        auto queries = Q_bhqd.extent(query_dim);
        auto keys = K_bhkd.extent(key_dim);
        auto channels = Q_bhqd.extent(channel_dim);

        for (auto &nest : parallel(
                 shape(batch,
                       heads,
                       ceil_div(queries, cfg.block_q)))) {
            auto [b, h, query_block] = nest.index();
            auto query0 = query_block * cfg.block_q;
            auto q = Q_bhqd.tile(
                          coord(b, h, query0, 0),
                          shape(batch_dim(1),
                                head_dim(1),
                                query_dim(cfg.block_q),
                                channel_dim(channels)),
                          bounds::zero)
                         .load()
                         .squeeze(batch_dim, head_dim);

            auto row_max = full<f32>(
                shape(query_dim(cfg.block_q)), -infinity<f32>());
            auto row_sum = zeros<f32>(shape(query_dim(cfg.block_q)));
            auto output = zeros<f32>(shape(
                query_dim(cfg.block_q), channel_dim(channels)));

            for (auto &subnest : nest.parallel(cfg.groups)) {
                for (auto &key_block : subnest.pipeline(
                         range(0, keys, cfg.block_k),
                         pipeline_policy{
                             .max_in_flight = cfg.max_in_flight,
                             .initiation_interval = 1})) {
                    auto key0 = key_block.index();

                    key_block.stage("load");
                    auto k = K_bhkd.tile(
                                  coord(b, h, key0, 0),
                                  shape(batch_dim(1),
                                        head_dim(1),
                                        key_dim(cfg.block_k),
                                        channel_dim(channels)),
                                  bounds::zero)
                                 .load()
                                 .squeeze(batch_dim, head_dim);
                    auto v = V_bhkd.tile(
                                  coord(b, h, key0, 0),
                                  shape(batch_dim(1),
                                        head_dim(1),
                                        key_dim(cfg.block_k),
                                        channel_dim(channels)),
                                  bounds::zero)
                                 .load()
                                 .squeeze(batch_dim, head_dim);

                    key_block.stage("score");
                    auto scores = mma(
                        q, transpose(k), zeros<f32>(shape(
                                             query_dim(cfg.block_q),
                                             key_dim(cfg.block_k))));
                    scores *= rsqrt(cast<f32>(channels));

                    auto query_id =
                        iota<i32>(query_dim, cfg.block_q) + query0;
                    auto key_id =
                        iota<i32>(key_dim, cfg.block_k) + key0;
                    auto valid = key_id < keys;
                    if (cfg.causal) {
                        valid = valid && key_id <= query_id;
                    }
                    scores = select(
                        valid, scores, -infinity<f32>());

                    key_block.stage("update");
                    auto block_max =
                        reduce(scores, key_dim, maximum);
                    auto next_max = maximum(row_max, block_max);
                    auto old_scale = exp(row_max - next_max);
                    auto probability = exp(scores - next_max);

                    row_sum = row_sum * old_scale +
                              reduce(probability, key_dim, add);
                    output = mma(
                        probability, v, output * old_scale);
                    row_max = next_max;
                }
            }

            auto normalized = output / row_sum;
            for (auto &leaf : nest.parallel(exec::infer)) {
                O_bhqd.tile(
                     coord(b, h, query0, 0),
                     shape(batch_dim(1),
                           head_dim(1),
                           query_dim(cfg.block_q),
                           channel_dim(channels)),
                     bounds::predicate)
                    .squeeze(batch_dim, head_dim)
                    .store(cast<f16>(normalized));
            }
        }
    });
}
~~~

`k`, `v`, and `scores` are cross-stage Tile SSA values. Their uses describe
producer/consumer edges; the compiler decides whether to materialize them,
their storage layouts, resource classes, live version counts, and barriers. The
loop-carried `row_max`, `row_sum`, and `output` assignments become distance-one
dependences. No explicit Memory is required.

## 10. Strided, dilated Conv2D

A convolution is a reduction over filter taps and input-channel chunks. The
output-space hierarchy remains independent of that reduction domain:

~~~cpp
struct Conv2DConfig {
    int32_t block_h;
    int32_t block_w;
    int32_t block_in_channels;
    int32_t block_out_channels;
    int32_t stride_h;
    int32_t stride_w;
    int32_t dilation_h;
    int32_t dilation_w;
    int32_t pad_h;
    int32_t pad_w;
};

// X is NHWC; W is filter_h × filter_w × input_channel × output_channel.
auto make_conv2d(Conv2DConfig cfg) {
    return tile_kernel([=](TensorView<f16, 4> X,
                           TensorView<f16, 4> W,
                           TensorView<f16, 1> bias,
                           TensorView<f16, 4> Y) {
        auto [batch_dim,
              input_y_dim,
              input_x_dim,
              output_y_dim,
              output_x_dim,
              filter_y_dim,
              filter_x_dim,
              input_channel_dim,
              output_channel_dim] =
            dims("batch",
                 "input_y",
                 "input_x",
                 "output_y",
                 "output_x",
                 "filter_y",
                 "filter_x",
                 "input_channel",
                 "output_channel");
        auto X_nyxc = X.with_dims(
            batch_dim, input_y_dim, input_x_dim, input_channel_dim);
        auto W_rscd = W.with_dims(filter_y_dim,
                                  filter_x_dim,
                                  input_channel_dim,
                                  output_channel_dim);
        auto bias_d = bias.with_dims(output_channel_dim);
        auto Y_nhwd = Y.with_dims(
            batch_dim, output_y_dim, output_x_dim, output_channel_dim);

        auto batch = X_nyxc.extent(batch_dim);
        auto input_channels = X_nyxc.extent(input_channel_dim);
        auto filter_h = W_rscd.extent(filter_y_dim);
        auto filter_w = W_rscd.extent(filter_x_dim);
        auto output_channels = W_rscd.extent(output_channel_dim);
        auto output_h = Y_nhwd.extent(output_y_dim);
        auto output_w = Y_nhwd.extent(output_x_dim);

        for (auto &nest : parallel(shape(
                 batch,
                 ceil_div(output_h, cfg.block_h),
                 ceil_div(output_w, cfg.block_w),
                 ceil_div(output_channels,
                          cfg.block_out_channels)))) {
            auto [n, tile_h, tile_w, tile_c] = nest.index();
            auto output_y0 = tile_h * cfg.block_h;
            auto output_x0 = tile_w * cfg.block_w;
            auto output_c0 = tile_c * cfg.block_out_channels;
            auto output_shape = shape(
                output_y_dim(cfg.block_h),
                output_x_dim(cfg.block_w),
                output_channel_dim(cfg.block_out_channels));
            auto output = zeros<f32>(output_shape);

            auto taps = product(
                range(0, filter_h),
                range(0, filter_w),
                range(0,
                      input_channels,
                      cfg.block_in_channels));

            for (auto &tap : nest.reduce(taps)) {
                auto [r, s, input_c0] = tap.index();
                auto input_y0 = output_y0 * cfg.stride_h +
                                r * cfg.dilation_h - cfg.pad_h;
                auto input_x0 = output_x0 * cfg.stride_w +
                                s * cfg.dilation_w - cfg.pad_w;

                // `window` is a strided TensorView projection. It does not
                // allocate a convolution-specific im2col buffer.
                auto x = X_nyxc.window(
                              coord(n, input_y0, input_x0, input_c0),
                              shape(batch_dim(1),
                                    output_y_dim(cfg.block_h),
                                    output_x_dim(cfg.block_w),
                                    input_channel_dim(
                                        cfg.block_in_channels)),
                              step(1,
                                   cfg.stride_h,
                                   cfg.stride_w,
                                   1),
                             bounds::zero)
                             .load()
                             .squeeze(batch_dim);
                auto w = W_rscd.tile(
                              coord(r, s, input_c0, output_c0),
                              shape(filter_y_dim(1),
                                    filter_x_dim(1),
                                    input_channel_dim(
                                        cfg.block_in_channels),
                                    output_channel_dim(
                                        cfg.block_out_channels)),
                              bounds::zero)
                             .load()
                             .squeeze(filter_y_dim, filter_x_dim);

                output = mma(x, w, output);
            }

            auto b = cast<f32>(bias_d.tile(
                                       coord(output_c0),
                                       shape(output_channel_dim(
                                           cfg.block_out_channels)),
                                       bounds::zero)
                                      .load());
            auto y = maximum(output + b, 0.0f);

            for (auto &leaf : nest.parallel(exec::infer)) {
                Y_nhwd.tile(
                     coord(n, output_y0, output_x0, output_c0),
                     shape(batch_dim(1),
                           output_y_dim(cfg.block_h),
                           output_x_dim(cfg.block_w),
                           output_channel_dim(cfg.block_out_channels)),
                     bounds::predicate)
                    .squeeze(batch_dim)
                    .store(cast<f16>(y));
            }
        }
    });
}
~~~

`window` is a library view helper over a core subview/index map. Each tap uses
the semantic `mma` value operation; bias/ReLU remains ordinary lifted Tile
arithmetic. A schedule may realize this as direct convolution, vector FMA,
implicit GEMM, or a proved target convolution atom. No `conv_team`, core
ConvOp, or im2col semantic object appears in the frontend.

## 11. Depthwise convolution and max-pooling

Depthwise convolution exercises the same tap reduction without an MMA-like
channel contraction:

~~~cpp
auto make_depthwise_conv2d(DepthwiseConfig cfg) {
    return tile_kernel([=](TensorView<f16, 4> X,
                           TensorView<f16, 3> W,
                           TensorView<f16, 4> Y) {
        auto [batch_dim,
              input_y_dim,
              input_x_dim,
              output_y_dim,
              output_x_dim,
              filter_y_dim,
              filter_x_dim,
              channel_dim] =
            dims("batch",
                 "input_y",
                 "input_x",
                 "output_y",
                 "output_x",
                 "filter_y",
                 "filter_x",
                 "channel");
        auto X_nyxc = X.with_dims(
            batch_dim, input_y_dim, input_x_dim, channel_dim);
        auto W_rsc = W.with_dims(
            filter_y_dim, filter_x_dim, channel_dim);
        auto Y_nhwc = Y.with_dims(
            batch_dim, output_y_dim, output_x_dim, channel_dim);

        auto batch = X_nyxc.extent(batch_dim);
        auto channels = X_nyxc.extent(channel_dim);
        auto output_h = Y_nhwc.extent(output_y_dim);
        auto output_w = Y_nhwc.extent(output_x_dim);
        auto filter_h = W_rsc.extent(filter_y_dim);
        auto filter_w = W_rsc.extent(filter_x_dim);

        for (auto &nest : parallel(shape(
                 batch,
                 ceil_div(output_h, cfg.block_h),
                 ceil_div(output_w, cfg.block_w),
                 ceil_div(channels, cfg.block_channels)))) {
            auto [n, tile_h, tile_w, tile_c] = nest.index();
            auto y0 = tile_h * cfg.block_h;
            auto x0 = tile_w * cfg.block_w;
            auto c0 = tile_c * cfg.block_channels;
            auto output = zeros<f32>(shape(
                output_y_dim(cfg.block_h),
                output_x_dim(cfg.block_w),
                channel_dim(cfg.block_channels)));

            for (auto &tap : nest.reduce(
                     product(range(0, filter_h),
                             range(0, filter_w)))) {
                auto [r, s] = tap.index();
                auto x = cast<f32>(X_nyxc.window(
                                           coord(n,
                                                 y0 * cfg.stride_h + r -
                                                     cfg.pad_h,
                                                 x0 * cfg.stride_w + s -
                                                     cfg.pad_w,
                                                 c0),
                                           shape(batch_dim(1),
                                                 output_y_dim(cfg.block_h),
                                                 output_x_dim(cfg.block_w),
                                                 channel_dim(
                                                     cfg.block_channels)),
                                           step(1,
                                                cfg.stride_h,
                                                cfg.stride_w,
                                                1),
                                          bounds::zero)
                                          .load()
                                          .squeeze(batch_dim));
                auto w = cast<f32>(W_rsc.tile(
                                           coord(r, s, c0),
                                           shape(filter_y_dim(1),
                                                 filter_x_dim(1),
                                                 channel_dim(
                                                     cfg.block_channels)),
                                           bounds::zero)
                                          .load()
                                          .squeeze(filter_y_dim,
                                                   filter_x_dim));
                output += x * w;
            }

            Y_nhwc.tile(
                 coord(n, y0, x0, c0),
                 shape(batch_dim(1),
                       output_y_dim(cfg.block_h),
                       output_x_dim(cfg.block_w),
                       channel_dim(cfg.block_channels)),
                 bounds::predicate)
                .squeeze(batch_dim)
                .store(cast<f16>(output));
        }
    });
}
~~~

Max-pooling merely changes the initial state and captured update:

~~~cpp
auto pooled = full<f32>(output_shape, -infinity<f32>());
for (auto &tap : nest.reduce(
         product(range(0, window_h), range(0, window_w)))) {
    auto [dy, dx] = tap.index();
    pooled = maximum(pooled, input_window(dy, dx).load());
}
~~~

The backend may select entirely different collectives for the inferred additive
and maximum contracts, but the execution and view model does not change.

## 12. Sobel and a deliberately ordered median filter

Sobel uses a pair-valued reducer over a generic two-dimensional stencil:

~~~cpp
auto make_sobel(FilterConfig cfg) {
    return tile_kernel([=](TensorView<f32, 2> input,
                           TensorView<f32, 2> output) {
        auto height = input.extent(0);
        auto width = input.extent(1);

        for (auto &nest : parallel(shape(
                 ceil_div(height, cfg.block_h),
                 ceil_div(width, cfg.block_w)))) {
            auto [tile_y, tile_x] = nest.index();
            auto y0 = tile_y * cfg.block_h;
            auto x0 = tile_x * cfg.block_w;
            auto gradient = GradientTile{
                .x = zeros<f32>(shape(cfg.block_h, cfg.block_w)),
                .y = zeros<f32>(shape(cfg.block_h, cfg.block_w))};

            for (auto &tap : nest.reduce(
                     product(range(-1, 2), range(-1, 2)),
                     componentwise_add)) {
                auto [dy, dx] = tap.index();
                auto pixel = input.tile(
                                      coord(y0 + dy, x0 + dx),
                                      shape(cfg.block_h, cfg.block_w),
                                      bounds::clamp)
                                 .load();
                gradient += GradientTile{
                    .x = pixel * sobel_x(dy, dx),
                    .y = pixel * sobel_y(dy, dx)};
            }

            auto magnitude = sqrt(
                square(gradient.x) + square(gradient.y));
            output.tile(
                      coord(y0, x0),
                      shape(cfg.block_h, cfg.block_w),
                      bounds::predicate)
                .store(magnitude);
        }
    });
}
~~~

By contrast, an arbitrary stateful filter is not falsely declared associative.
This pedagogical median implementation uses `serial` and may be unrolled, but
its update order is preserved:

~~~cpp
auto state = Median9State<f32>::empty(output_shape);
for (auto &tap : nest.serial(
         product(range(-1, 2), range(-1, 2)))) {
    auto [dy, dx] = tap.index();
    auto pixel = input_window(dy, dx).load();
    state = median9_insert(state, pixel);
}
auto median = median9_value(state);
~~~

If a future mergeable selection-state contract is provided, the same algorithm
may be rewritten as `reduce`. Until then, `serial` is the honest semantic form.

## 13. Stable row Top-K

Top-K is a custom reduction over a bounded candidate state. The total ordering
includes validity, value, NaN policy, and original index, so merging two states
and truncating to K is associative and deterministic:

~~~cpp
template<size_t K>
auto make_row_topk(TopKConfig cfg) {
    return tile_kernel([=](TensorView<f16, 2> X,
                           TensorView<f32, 2> Values,
                           TensorView<i32, 2> Indices) {
        auto [row_dim, feature_dim, rank_dim] =
            dims("row", "feature", "rank");
        auto X_rf = X.with_dims(row_dim, feature_dim);
        auto Values_rr = Values.with_dims(row_dim, rank_dim);
        auto Indices_rr = Indices.with_dims(row_dim, rank_dim);
        auto rows = X_rf.extent(row_dim);
        auto features = X_rf.extent(feature_dim);

        for (auto &nest : parallel(rows)) {
            auto row = nest.index();
            auto x = cast<f32>(X_rf.tile(
                                    coord(row, 0),
                                    shape(1, features),
                                    bounds::assume)
                                   .load()
                                   .squeeze(row_dim));
            auto best = topk_identity<f32, K>(
                descending,
                tie::lowest_index,
                nan::last);

            for (auto &candidate : nest.reduce(
                     x.domain(feature_dim),
                     topk_merge<K>)) {
                auto item = indexed_value(
                    x.at(candidate), candidate.index());
                best = topk_insert<K>(best, item);
            }

            Values_rr.tile(
                      coord(row, 0),
                      shape(row_dim(1), rank_dim(K)),
                      bounds::assume)
                .squeeze(row_dim)
                .store(best.values);
            Indices_rr.tile(
                       coord(row, 0),
                       shape(row_dim(1), rank_dim(K)),
                       bounds::assume)
                .squeeze(row_dim)
                .store(best.indices);
        }
    });
}
~~~

The common form is only sugar:

~~~cpp
auto best = topk<K>(x,
                    x.dim<0>(),
                    descending,
                    tie::lowest_index,
                    nan::last);
~~~

The scheduler may use per-participant insertion, bitonic candidate merges,
shuffle exchanges, shared-memory merging, or a serial fallback. `K` is an
ordinary host/JIT specialization because it changes Tile shape and state type.
A runtime `k` uses `topk<KMax>` plus a staged valid count; it does not create a
dynamically sized fragment.

## 14. Block sort and large multi-pass sort

Full sort is a logical permutation, not a reduction. The Tile library expresses
one root-local block sort with an explicit total-order contract:

~~~cpp
auto make_block_sort(SortConfig cfg) {
    return tile_kernel([=](TensorView<f32, 1> Input,
                           TensorView<f32, 1> Values,
                           TensorView<i32, 1> Indices) {
        auto element_dim = dim("element");
        auto Input_e = Input.with_dims(element_dim);
        auto Values_e = Values.with_dims(element_dim);
        auto Indices_e = Indices.with_dims(element_dim);
        auto count = Input_e.extent(element_dim);

        for (auto &nest : parallel(
                 ceil_div(count, cfg.elements_per_nest))) {
            auto begin = nest.index() * cfg.elements_per_nest;
            auto logical_index =
                iota<i32>(element_dim, cfg.elements_per_nest) + begin;
            auto valid = logical_index < count;
            auto value = Input_e.tile(
                                  coord(begin),
                                  shape(element_dim(cfg.elements_per_nest)),
                                  bounds::zero)
                             .load();
            auto items = SortItem{
                .valid = valid,
                .value = value,
                .original_index = logical_index};

            auto sorted = sort(
                items,
                element_dim,
                total_order{
                    .valid = valid_first,
                    .value = ascending,
                    .nan = nan::last,
                    .tie = tie::lowest_index});

            Values_e.tile(
                      coord(begin),
                      shape(element_dim(cfg.elements_per_nest)),
                      bounds::predicate)
                .store(sorted.value);
            Indices_e.tile(
                       coord(begin),
                       shape(element_dim(cfg.elements_per_nest)),
                       bounds::predicate)
                .store(sorted.original_index);
        }
    });
}
~~~

The valid bit, rather than a magic numeric sentinel, places padded elements
last even in the presence of infinities and NaNs. The reference implementation
expands to core regions and value operations; a target may replace the proved
expansion with a sorting network, bitonic exchange, radix, or registered atom.

A large global sort exposes its merge-pass boundary:

~~~cpp
auto make_merge_sorted_runs(int32_t run_size) {
    return tile_kernel([=](TensorView<SortItem, 1> input,
                           TensorView<SortItem, 1> output) {
        auto element_dim = dim("element");
        auto input_e = input.with_dims(element_dim);
        auto output_e = output.with_dims(element_dim);
        auto count = input_e.extent(element_dim);
        auto pair_size = 2 * run_size;

        for (auto &nest : parallel(ceil_div(count, pair_size))) {
            auto begin = nest.index() * pair_size;
            auto left = input_e.tile(
                                 coord(begin),
                                 shape(element_dim(run_size)),
                                 bounds::invalid)
                            .load();
            auto right = input_e.tile(
                                  coord(begin + run_size),
                                  shape(element_dim(run_size)),
                                  bounds::invalid)
                             .load();
            auto merged = merge_sorted(
                left,
                right,
                element_dim,
                stable_total_order);

            output_e.tile(
                      coord(begin),
                      shape(element_dim(pair_size)),
                      bounds::predicate)
                .store(merged);
        }
    });
}
~~~

Host orchestration ping-pongs buffers between visible launches:

~~~cpp
device.jit(make_block_sort(cfg))(input, scratch_a);
for (auto run = cfg.elements_per_nest; run < count; run *= 2) {
    device.jit(make_merge_sorted_runs(run))(scratch_a, scratch_b);
    std::swap(scratch_a, scratch_b);
}
~~~

A radix implementation uses the same core pieces rather than another frontend
hierarchy:

~~~cpp
for (auto &digit : nest.serial(
         range(0, key_bits, cfg.radix_bits))) {
    auto bucket = radix_digit(keys, digit.index(), cfg.radix_bits);
    auto counts = histogram(bucket, cfg.radix_size);
    auto offsets = exclusive_scan(counts, counts.dim<0>(), add);
    keys = stable_bucket_scatter(keys, bucket, offsets);
}
~~~

Here `serial` preserves digit-pass order; histogram reduction, scan, and scatter
may still be parallelized internally. Device-wide versions use visible partial
histogram/scan/scatter launches or a target-proved cooperative schedule.

## 15. Manual Memory escape hatch

Stable addressable temporaries are declared only when address identity or a
manual protocol is intentional. Ownership is inferred from the nearest lexical
`parallel` scope:

~~~cpp
for (auto &nest : parallel(grid_shape)) {
    for (auto &subnest : nest.parallel(subnest_shape)) {
        auto mailbox = memory<f16>(mailbox_layout, mem::shared);
        auto exchange = memory<f32>(exchange_layout);
        auto output = zeros<f32>(output_shape);

        for (auto &k : subnest.pipeline(iterations, policy)) {
            k.stage("produce");
            for (auto &producer : k.parallel(producer_shape)) {
                mailbox.store(input_view(k.index()).load());
            }

            k.stage("consume");
            for (auto &consumer : k.parallel(consumer_shape)) {
                auto partial = transform(mailbox.load());
                exchange.store(partial);
                output = combine(output, exchange.load());
            }
        }
    }
}
~~~

The `Memory` type itself does not mean shared memory, registers, or SRAM. Here
`mem::shared` is an optional target resource-class constraint on `mailbox`,
while `exchange` remains open for scheduling. Resource classes are not a
memory-order lattice; the target checks a general access/capability relation.
Compared with virtual Tile SSA, the user has deliberately pinned stable
identity, address layout, and mutation effects. Reads and writes remain explicit
`load()` / `store(tile)` effects, while MemorySSA and the pipeline planner still
derive physical versions.

## 16. Scatter and segmented accumulation

~~~cpp
auto make_segment_sum(ScatterConfig cfg) {
    return tile_kernel([=](TensorView<i32, 1> segment,
                           TensorView<f32, 1> value,
                           TensorView<f32, 1> output) {
        auto count = value.extent(0);

        for (auto &nest : parallel(
                 ceil_div(count, cfg.chunk_size))) {
            auto begin = nest.index() * cfg.chunk_size;
            auto ids = segment.tile(
                                  coord(begin),
                                  shape(cfg.chunk_size),
                                  bounds::predicate)
                               .load();
            auto values = value.tile(
                                   coord(begin),
                                   shape(cfg.chunk_size),
                                   bounds::predicate)
                                .load();

            for (auto &leaf : nest.parallel(exec::infer)) {
                output.scatter(ids, values, add);
            }
        }
    });
}
~~~

The data-dependent IDs are semantic values. `scatter` is a library wrapper over
an indexed store/atomic effect, not a core ScatterOp. Indirection is not
smuggled into layout algebra, and the atomic/segmented realization is selected
from target capabilities.

## 17. All four structured regions in one skeleton

~~~cpp
for (auto &nest : parallel(grid_shape)) {
    auto output = zeros<f32>(output_shape);

    for (auto &subnest : nest.parallel(
             subnest_shape, exec::warp)) {
        for (auto &k : subnest.pipeline(k_domain, policy)) {
            k.stage("load");
            auto a = load_a(k.index());
            auto b = load_b(k.index());

            k.stage("compute");
            output = mma(a, b, output);
        }

        for (auto &i : subnest.serial(tail_domain)) {
            output = ordered_tail(output, i.index());
        }

        auto total = zeros<f32>(result_shape);
        for (auto &elem_nest :
             subnest.reduce(output.domain(reduction_axes))) {
            total += output.at(elem_nest);
        }
        consume(total);
    }
}
~~~

- `parallel` extends the spatial participant/owner prefix;
- `serial` extends strict time and permits arbitrary recurrence;
- `pipeline` extends time with a producer/consumer dependence graph;
- `reduce` introduces a mergeable algebraic domain that scheduling may place
  across space, time, or both.

For an explicitly scalar implementation, descend with `serial`:

~~~cpp
for (auto &leaf : nest.parallel(exec::infer)) {
    auto begin = leaf.index();
    auto step = leaf.extent();

    for (auto &i : leaf.serial(range(begin, count, step))) {
        Y[base + i.index()] = gelu(X[base + i.index()]);
    }
}
~~~

`leaf` is still a logical participant. A later binding decides whether it is a
GPU lane, CPU vector lane, accelerator thread, or serialized logical instance.

## 18. Layout construction and target binding

~~~cpp
auto row_major = layout(shape(M, N), stride(N, 1));
auto column_major = layout(shape(M, N), stride(1, M));

auto transpose_map = permute<1, 0>(shape(M, N));
auto transposed = compose(row_major, transpose_map);

auto tiled = logical_divide(
    row_major,
    layout(shape(BM, BN), stride(BN, 1)));

auto swizzled = compose(
    xor_swizzle<3, 4, 3>(),
    layout(shape(BM, BK), stride(BK, 1)));

auto fragment = index_map(
    domain(subnest, leaf, local(4)),
    codomain(shape(BM, BN)),
    [=](auto subnest_id, auto leaf_id, auto slot) {
        auto linear =
            (subnest_id * leaf.extent() + leaf_id) * 4 + slot;
        return coord(linear / BN, linear % BN);
    });
~~~

All constructors produce typed LayoutMap DAGs. The lambda is a pure symbolic
index definition; it is not a data-dependent device callback.

The same portable hierarchy may receive either binding:

~~~text
ExecBinding(nest, subnest, leaf)
  -> (workgroup_xy, subgroup_id, lane_id, serial_k)

ExecBinding(nest, subnest, leaf)
  -> (thread_team, serial_team, vector_lane, scalar_tail)
~~~

Reduction placement uses the same algebra but has a distinct type:

~~~text
ReductionPlacement(reduction_axis)
  -> (participant_fiber, local_serial_step)
~~~

That distinction prevents reduction coordinates from accidentally becoming
memory-owner coordinates.

## 19. Straightforward JIT autotuning

~~~cpp
auto candidates = make_candidates(problem, target);

for (auto cfg : candidates) {
    auto kernel = make_flash_attention(cfg);
    auto executable = device.jit(kernel);
    database.record(cfg, benchmark(executable, Q, K, V, O));
}
~~~

Repeated construction and JIT is the baseline. Candidates may vary logical
nest shapes, reduction factorization, Tile distributions, atom selection,
pipeline policy, and materialization plan. Layout proofs and target capacities
reject illegal candidates before compilation. A later symbolic family optimizer
must preserve this observable model and cache behavior.
