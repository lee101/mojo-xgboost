"""Dense histogram tree kernels exposed through a stable C ABI."""

from max.algorithm import parallelize
from std.math import iota
from std.sys.info import simd_width_of as simdwidthof
from std.utils.numerics import isnan

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime PARALLEL_WORK = 100_000
comptime PREDICT_CHUNK = 4_096


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def soft_threshold(g: Float64, alpha: Float64) -> Float64:
    if g > alpha:
        return g - alpha
    if g < -alpha:
        return g + alpha
    return 0.0


def node_score(g: Float64, h: Float64, reg_lambda: Float64, alpha: Float64) -> Float64:
    var s = soft_threshold(g, alpha)
    return s * s / (h + reg_lambda)


def leaf_value(
    g: Float64,
    h: Float64,
    reg_lambda: Float64,
    alpha: Float64,
    max_delta_step: Float64,
) -> Float64:
    var value = -soft_threshold(g, alpha) / (h + reg_lambda)
    if max_delta_step > 0.0:
        if value > max_delta_step:
            value = max_delta_step
        elif value < -max_delta_step:
            value = -max_delta_step
    return value


def predict_margin_rows(
    x: FPtr,
    features: IPtr,
    thresholds: FPtr,
    defaults: IPtr,
    leaves: FPtr,
    pred: FPtr,
    start: Int,
    end: Int,
    d: Int,
    n_trees: Int,
    max_nodes: Int,
    base_margin: Float64,
):
    comptime W = simdwidthof[DType.float64]()
    var vector_trees = n_trees - n_trees % W
    for r in range(start, end):
        var margin = base_margin
        for tree in range(0, vector_trees, W):
            var tree_base = iota[DType.int, W](tree) * max_nodes
            var node = SIMD[DType.int, W](0)
            var pos = tree_base
            var feature = features.gather(pos).cast[DType.int]()
            var active = feature.ge(0)
            while any(active):
                var value = x.gather(
                    r * d + feature, mask=active
                )
                var threshold = thresholds.gather(
                    pos, mask=active
                )
                var default_left = defaults.gather(
                    pos, mask=active
                )
                var go_left = isnan(value).select(
                    default_left.ne(0), value.le(threshold)
                )
                var next_node = go_left.select(2 * node + 1, 2 * node + 2)
                node = active.select(next_node, node)
                pos = tree_base + node
                feature = features.gather(pos).cast[DType.int]()
                active = feature.ge(0)
            margin += leaves.gather(pos).reduce_add()
        for tree in range(vector_trees, n_trees):
            var tree_base = tree * max_nodes
            var node = 0
            while features[tree_base + node] >= 0:
                var f = Int(features[tree_base + node])
                var value = x[r * d + f]
                var go_left = False
                if value != value:
                    go_left = defaults[tree_base + node] != 0
                else:
                    go_left = value <= thresholds[tree_base + node]
                node = 2 * node + 1 if go_left else 2 * node + 2
            margin += leaves[tree_base + node]
        pred[r] = margin


def predict_add_rows(
    x: FPtr,
    features: IPtr,
    thresholds: FPtr,
    defaults: IPtr,
    leaves: FPtr,
    pred: FPtr,
    start: Int,
    end: Int,
    d: Int,
):
    for r in range(start, end):
        var node = 0
        while features[node] >= 0:
            var f = Int(features[node])
            var value = x[r * d + f]
            var go_left = False
            if value != value:
                go_left = defaults[node] != 0
            else:
                go_left = value <= thresholds[node]
            node = 2 * node + 1 if go_left else 2 * node + 2
        pred[r] += leaves[node]


def predict_leaf_rows(
    x: FPtr,
    features: IPtr,
    thresholds: FPtr,
    defaults: IPtr,
    leaf_indices: IPtr,
    start: Int,
    end: Int,
    d: Int,
    n_trees: Int,
    max_nodes: Int,
):
    for r in range(start, end):
        for tree in range(n_trees):
            var tree_base = tree * max_nodes
            var node = 0
            while features[tree_base + node] >= 0:
                var f = Int(features[tree_base + node])
                var value = x[r * d + f]
                var go_left = False
                if value != value:
                    go_left = defaults[tree_base + node] != 0
                else:
                    go_left = value <= thresholds[tree_base + node]
                node = 2 * node + 1 if go_left else 2 * node + 2
            leaf_indices[r * n_trees + tree] = Int64(node)


def partition_row_range(
    bins: IPtr,
    grad: FPtr,
    hess: FPtr,
    features: IPtr,
    split_bins: IPtr,
    defaults: IPtr,
    row_nodes: IPtr,
    node_grad: FPtr,
    node_hess: FPtr,
    start: Int,
    end: Int,
    d: Int,
    level_start: Int,
    level_end: Int,
    update_totals: Bool,
):
    for r in range(start, end):
        var node = Int(row_nodes[r])
        if node < level_start or node >= level_end:
            continue
        var feature = Int(features[node])
        if feature < 0:
            row_nodes[r] = -1
            continue
        var b = Int(bins[r * d + feature])
        var go_left = False
        if b < 0:
            go_left = defaults[node] != 0
        else:
            go_left = b <= Int(split_bins[node])
        var child = 2 * node + 1 if go_left else 2 * node + 2
        row_nodes[r] = Int64(child)
        if update_totals:
            node_grad[child] += grad[r]
            node_hess[child] += hess[r]


@export("mxgb_quantize")
def mxgb_quantize(
    x_addr: Int,
    cuts_addr: Int,
    bins_addr: Int,
    n: Int,
    d: Int,
    max_bin: Int,
) abi("C"):
    if n <= 0 or d <= 0:
        return
    if max_bin < 2 or x_addr == 0 or cuts_addr == 0 or bins_addr == 0:
        return
    var x = fp(x_addr)
    var cuts = fp(cuts_addr)
    var bins = ip(bins_addr)
    var cut_count = max_bin - 1
    comptime W = simdwidthof[DType.float64]()
    var vector_features = d - d % W
    for r in range(n):
        for f in range(0, vector_features, W):
            var value = x.load[width=W](r * d + f)
            var missing = isnan(value)
            var feature = iota[DType.int, W](f)
            var lo = SIMD[DType.int, W](0)
            var hi = SIMD[DType.int, W](cut_count)
            var active = lo.lt(hi) & ~missing
            while any(active):
                var mid = (lo + hi) // 2
                var cut = cuts.gather(feature * cut_count + mid, mask=active)
                var lower = value.le(cut)
                var next_lo = lower.select(lo, mid + 1)
                var next_hi = lower.select(mid, hi)
                lo = active.select(next_lo, lo)
                hi = active.select(next_hi, hi)
                active = lo.lt(hi) & ~missing
            bins.store(
                r * d + f,
                missing.select(SIMD[DType.int, W](-1), lo).cast[DType.int64](),
            )
        for f in range(vector_features, d):
            var value = x[r * d + f]
            if value != value:
                bins[r * d + f] = -1
                continue
            var lo = 0
            var hi = cut_count
            while lo < hi:
                var mid = (lo + hi) // 2
                if value <= cuts[f * cut_count + mid]:
                    hi = mid
                else:
                    lo = mid + 1
            bins[r * d + f] = Int64(lo)


@export("mxgb_build_tree")
def mxgb_build_tree(
    bins_addr: Int,
    grad_addr: Int,
    hess_addr: Int,
    feature_addr: Int,
    split_bin_addr: Int,
    default_left_addr: Int,
    leaf_addr: Int,
    gain_addr: Int,
    cover_addr: Int,
    row_node_addr: Int,
    node_grad_addr: Int,
    node_hess_addr: Int,
    hist_grad_addr: Int,
    hist_hess_addr: Int,
    n: Int,
    d: Int,
    max_bin: Int,
    max_depth: Int,
    min_child_weight: Float64,
    reg_lambda: Float64,
    reg_alpha: Float64,
    gamma: Float64,
    max_delta_step: Float64,
    n_threads: Int,
) abi("C") -> Int:
    if (
        n <= 0
        or d <= 0
        or max_bin < 2
        or max_depth < 1
        or bins_addr == 0
        or grad_addr == 0
        or hess_addr == 0
        or feature_addr == 0
        or split_bin_addr == 0
        or default_left_addr == 0
        or leaf_addr == 0
        or gain_addr == 0
        or cover_addr == 0
        or row_node_addr == 0
        or node_grad_addr == 0
        or node_hess_addr == 0
        or hist_grad_addr == 0
        or hist_hess_addr == 0
    ):
        return -1
    var bins = ip(bins_addr)
    var grad = fp(grad_addr)
    var hess = fp(hess_addr)
    var features = ip(feature_addr)
    var split_bins = ip(split_bin_addr)
    var defaults = ip(default_left_addr)
    var leaves = fp(leaf_addr)
    var gains = fp(gain_addr)
    var covers = fp(cover_addr)
    var row_nodes = ip(row_node_addr)
    var node_grad = fp(node_grad_addr)
    var node_hess = fp(node_hess_addr)
    var hist_grad = fp(hist_grad_addr)
    var hist_hess = fp(hist_hess_addr)
    var max_nodes = (1 << (max_depth + 1)) - 1

    for node in range(max_nodes):
        features[node] = -1
        split_bins[node] = -1
        defaults[node] = 0
        leaves[node] = 0.0
        gains[node] = 0.0
        covers[node] = 0.0
        node_grad[node] = 0.0
        node_hess[node] = 0.0
    for r in range(n):
        row_nodes[r] = 0

    var split_count = 0
    for depth in range(max_depth + 1):
        var level_start = (1 << depth) - 1
        var level_end = (1 << (depth + 1)) - 1
        var hist_start = level_start * d * max_bin
        var hist_end = level_end * d * max_bin
        for i in range(hist_start, hist_end):
            hist_grad[i] = 0.0
            hist_hess[i] = 0.0
        if n_threads > 1 and n * d >= PARALLEL_WORK and d > 1:
            if depth == 0:
                for r in range(n):
                    node_grad[0] += grad[r]
                    node_hess[0] += hess[r]

            @parameter
            def build_feature(f: Int):
                for r in range(n):
                    var node = Int(row_nodes[r])
                    if node < level_start or node >= level_end:
                        continue
                    if depth > 0:
                        var parent = (node - 1) // 2
                        var left = 2 * parent + 1
                        var build_node = (
                            left
                            if node_hess[left] <= node_hess[left + 1]
                            else left + 1
                        )
                        if node != build_node:
                            continue
                    var b = Int(bins[r * d + f])
                    if b >= 0:
                        var pos = (node * d + f) * max_bin + b
                        hist_grad[pos] += grad[r]
                        hist_hess[pos] += hess[r]
                if depth > 0:
                    comptime W = simdwidthof[DType.float64]()
                    var vector_end = max_bin - max_bin % W
                    var parent_start = (level_start - 1) // 2
                    for parent in range(parent_start, level_start):
                        var left = 2 * parent + 1
                        var built = (
                            left
                            if node_hess[left] <= node_hess[left + 1]
                            else left + 1
                        )
                        var sibling = left + 1 if built == left else left
                        var parent_base = (parent * d + f) * max_bin
                        var built_base = (built * d + f) * max_bin
                        var sibling_base = (sibling * d + f) * max_bin
                        for b in range(0, vector_end, W):
                            var grad_vec = (
                                hist_grad.load[width=W](parent_base + b)
                                - hist_grad.load[width=W](built_base + b)
                            )
                            var hess_vec = (
                                hist_hess.load[width=W](parent_base + b)
                                - hist_hess.load[width=W](built_base + b)
                            )
                            hist_grad.store(sibling_base + b, grad_vec)
                            hist_hess.store(sibling_base + b, hess_vec)
                        for b in range(vector_end, max_bin):
                            hist_grad[sibling_base + b] = (
                                hist_grad[parent_base + b]
                                - hist_grad[built_base + b]
                            )
                            hist_hess[sibling_base + b] = (
                                hist_hess[parent_base + b]
                                - hist_hess[built_base + b]
                            )

            parallelize[build_feature](d, n_threads)
        else:
            for r in range(n):
                var node = Int(row_nodes[r])
                if node < level_start or node >= level_end:
                    continue
                if depth == 0:
                    node_grad[0] += grad[r]
                    node_hess[0] += hess[r]
                elif depth > 0:
                    var parent = (node - 1) // 2
                    var left = 2 * parent + 1
                    var build_node = (
                        left
                        if node_hess[left] <= node_hess[left + 1]
                        else left + 1
                    )
                    if node != build_node:
                        continue
                for f in range(d):
                    var b = Int(bins[r * d + f])
                    if b >= 0:
                        var pos = (node * d + f) * max_bin + b
                        hist_grad[pos] += grad[r]
                        hist_hess[pos] += hess[r]
            if depth > 0:
                comptime W = simdwidthof[DType.float64]()
                var vector_end = max_bin - max_bin % W
                var parent_start = (level_start - 1) // 2
                for parent in range(parent_start, level_start):
                    var left = 2 * parent + 1
                    var built = (
                        left
                        if node_hess[left] <= node_hess[left + 1]
                        else left + 1
                    )
                    var sibling = left + 1 if built == left else left
                    for f in range(d):
                        var parent_base = (parent * d + f) * max_bin
                        var built_base = (built * d + f) * max_bin
                        var sibling_base = (sibling * d + f) * max_bin
                        for b in range(0, vector_end, W):
                            var grad_vec = (
                                hist_grad.load[width=W](parent_base + b)
                                - hist_grad.load[width=W](built_base + b)
                            )
                            var hess_vec = (
                                hist_hess.load[width=W](parent_base + b)
                                - hist_hess.load[width=W](built_base + b)
                            )
                            hist_grad.store(sibling_base + b, grad_vec)
                            hist_hess.store(sibling_base + b, hess_vec)
                        for b in range(vector_end, max_bin):
                            hist_grad[sibling_base + b] = (
                                hist_grad[parent_base + b]
                                - hist_grad[built_base + b]
                            )
                            hist_hess[sibling_base + b] = (
                                hist_hess[parent_base + b]
                                - hist_hess[built_base + b]
                            )

        for node in range(level_start, level_end):
            var total_g = node_grad[node]
            var total_h = node_hess[node]
            covers[node] = total_h
            if total_h + reg_lambda > 0.0:
                leaves[node] = leaf_value(
                    total_g, total_h, reg_lambda, reg_alpha, max_delta_step
                )
            if depth == max_depth or total_h < 2.0 * min_child_weight:
                continue

            var parent_score = node_score(total_g, total_h, reg_lambda, reg_alpha)
            var best_gain = 0.0
            var best_feature = -1
            var best_bin = -1
            var best_default = 0
            for f in range(d):
                var base = (node * d + f) * max_bin
                var present_g = 0.0
                var present_h = 0.0
                comptime W = simdwidthof[DType.float64]()
                var vector_end = max_bin - max_bin % W
                for b in range(0, vector_end, W):
                    present_g += hist_grad.load[width=W](base + b).reduce_add()
                    present_h += hist_hess.load[width=W](base + b).reduce_add()
                for b in range(vector_end, max_bin):
                    present_g += hist_grad[base + b]
                    present_h += hist_hess[base + b]
                var missing_g = total_g - present_g
                var missing_h = total_h - present_h
                var prefix_g = 0.0
                var prefix_h = 0.0
                for b in range(max_bin - 1):
                    prefix_g += hist_grad[base + b]
                    prefix_h += hist_hess[base + b]

                    var left_g = prefix_g
                    var left_h = prefix_h
                    var right_g = total_g - left_g
                    var right_h = total_h - left_h
                    if left_h >= min_child_weight and right_h >= min_child_weight:
                        var candidate = 0.5 * (
                            node_score(left_g, left_h, reg_lambda, reg_alpha)
                            + node_score(right_g, right_h, reg_lambda, reg_alpha)
                            - parent_score
                        ) - gamma
                        if candidate > best_gain:
                            best_gain = candidate
                            best_feature = f
                            best_bin = b
                            best_default = 0

                    left_g = prefix_g + missing_g
                    left_h = prefix_h + missing_h
                    right_g = total_g - left_g
                    right_h = total_h - left_h
                    if left_h >= min_child_weight and right_h >= min_child_weight:
                        var candidate = 0.5 * (
                            node_score(left_g, left_h, reg_lambda, reg_alpha)
                            + node_score(right_g, right_h, reg_lambda, reg_alpha)
                            - parent_score
                        ) - gamma
                        if candidate > best_gain:
                            best_gain = candidate
                            best_feature = f
                            best_bin = b
                            best_default = 1

            if best_feature >= 0:
                features[node] = Int64(best_feature)
                split_bins[node] = Int64(best_bin)
                defaults[node] = Int64(best_default)
                gains[node] = best_gain
                split_count += 1

        if depth < max_depth:
            var next_level_end = (1 << (depth + 2)) - 1
            for node in range(level_end, next_level_end):
                node_grad[node] = 0.0
                node_hess[node] = 0.0
            if n_threads > 1 and n >= PARALLEL_WORK:
                var chunks = (n + PREDICT_CHUNK - 1) // PREDICT_CHUNK

                @parameter
                def partition_chunk(chunk: Int):
                    var start = chunk * PREDICT_CHUNK
                    partition_row_range(
                        bins,
                        grad,
                        hess,
                        features,
                        split_bins,
                        defaults,
                        row_nodes,
                        node_grad,
                        node_hess,
                        start,
                        min(start + PREDICT_CHUNK, n),
                        d,
                        level_start,
                        level_end,
                        False,
                    )

                parallelize[partition_chunk](chunks, n_threads)
                for r in range(n):
                    var node = Int(row_nodes[r])
                    if node >= level_end and node < next_level_end:
                        node_grad[node] += grad[r]
                        node_hess[node] += hess[r]
            else:
                partition_row_range(
                    bins,
                    grad,
                    hess,
                    features,
                    split_bins,
                    defaults,
                    row_nodes,
                    node_grad,
                    node_hess,
                    0,
                    n,
                    d,
                    level_start,
                    level_end,
                    True,
                )

    return split_count


@export("mxgb_predict")
def mxgb_predict(
    x_addr: Int,
    feature_addr: Int,
    threshold_addr: Int,
    default_left_addr: Int,
    leaf_addr: Int,
    pred_addr: Int,
    n: Int,
    d: Int,
    n_trees: Int,
    max_nodes: Int,
    base_margin: Float64,
    n_threads: Int,
) abi("C"):
    if n <= 0 or n_trees <= 0:
        return
    if (
        d <= 0
        or max_nodes <= 0
        or x_addr == 0
        or feature_addr == 0
        or threshold_addr == 0
        or default_left_addr == 0
        or leaf_addr == 0
        or pred_addr == 0
    ):
        return
    var x = fp(x_addr)
    var features = ip(feature_addr)
    var thresholds = fp(threshold_addr)
    var defaults = ip(default_left_addr)
    var leaves = fp(leaf_addr)
    var pred = fp(pred_addr)
    if n_threads > 1 and n * n_trees >= PARALLEL_WORK:
        var chunks = (n + PREDICT_CHUNK - 1) // PREDICT_CHUNK

        @parameter
        def predict_chunk(chunk: Int):
            var start = chunk * PREDICT_CHUNK
            predict_margin_rows(
                x,
                features,
                thresholds,
                defaults,
                leaves,
                pred,
                start,
                min(start + PREDICT_CHUNK, n),
                d,
                n_trees,
                max_nodes,
                base_margin,
            )

        parallelize[predict_chunk](chunks, n_threads)
    else:
        predict_margin_rows(
            x,
            features,
            thresholds,
            defaults,
            leaves,
            pred,
            0,
            n,
            d,
            n_trees,
            max_nodes,
            base_margin,
        )


@export("mxgb_predict_add")
def mxgb_predict_add(
    x_addr: Int,
    feature_addr: Int,
    threshold_addr: Int,
    default_left_addr: Int,
    leaf_addr: Int,
    pred_addr: Int,
    n: Int,
    d: Int,
    n_threads: Int,
) abi("C"):
    if n <= 0:
        return
    if (
        d <= 0
        or x_addr == 0
        or feature_addr == 0
        or threshold_addr == 0
        or default_left_addr == 0
        or leaf_addr == 0
        or pred_addr == 0
    ):
        return
    var x = fp(x_addr)
    var features = ip(feature_addr)
    var thresholds = fp(threshold_addr)
    var defaults = ip(default_left_addr)
    var leaves = fp(leaf_addr)
    var pred = fp(pred_addr)
    if n_threads > 1 and n >= PARALLEL_WORK:
        var chunks = (n + PREDICT_CHUNK - 1) // PREDICT_CHUNK

        @parameter
        def predict_add_chunk(chunk: Int):
            var start = chunk * PREDICT_CHUNK
            predict_add_rows(
                x,
                features,
                thresholds,
                defaults,
                leaves,
                pred,
                start,
                min(start + PREDICT_CHUNK, n),
                d,
            )

        parallelize[predict_add_chunk](chunks, n_threads)
    else:
        predict_add_rows(
            x, features, thresholds, defaults, leaves, pred, 0, n, d
        )


@export("mxgb_predict_leaf")
def mxgb_predict_leaf(
    x_addr: Int,
    feature_addr: Int,
    threshold_addr: Int,
    default_left_addr: Int,
    leaf_index_addr: Int,
    n: Int,
    d: Int,
    n_trees: Int,
    max_nodes: Int,
    n_threads: Int,
) abi("C"):
    if n <= 0 or n_trees <= 0:
        return
    if (
        d <= 0
        or max_nodes <= 0
        or x_addr == 0
        or feature_addr == 0
        or threshold_addr == 0
        or default_left_addr == 0
        or leaf_index_addr == 0
    ):
        return
    var x = fp(x_addr)
    var features = ip(feature_addr)
    var thresholds = fp(threshold_addr)
    var defaults = ip(default_left_addr)
    var leaf_indices = ip(leaf_index_addr)
    if n_threads > 1 and n * n_trees >= PARALLEL_WORK:
        var chunks = (n + PREDICT_CHUNK - 1) // PREDICT_CHUNK

        @parameter
        def predict_leaf_chunk(chunk: Int):
            var start = chunk * PREDICT_CHUNK
            predict_leaf_rows(
                x,
                features,
                thresholds,
                defaults,
                leaf_indices,
                start,
                min(start + PREDICT_CHUNK, n),
                d,
                n_trees,
                max_nodes,
            )

        parallelize[predict_leaf_chunk](chunks, n_threads)
    else:
        predict_leaf_rows(
            x,
            features,
            thresholds,
            defaults,
            leaf_indices,
            0,
            n,
            d,
            n_trees,
            max_nodes,
        )
