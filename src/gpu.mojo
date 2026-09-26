"""GPU kernels for the histogram-boosting core.

Three stages of `hist` boosting are wide, regular and independent per element,
which is exactly what a GPU wants:

* **quantize** — a binary search over the cut vector for every one of `n * d`
  values. Pure ALU on a small, cached `cuts` array.
* **histogram** — the gradient/hessian scatter-add that dominates tree building.
  One block per (feature, row-chunk) accumulates into a shared-memory histogram
  and commits it once with global atomics, so global traffic is
  `blocks * max_bin` instead of `n * d`.
* **predict** — a per-row tree walk over the whole ensemble; every row is
  independent and the tree arrays stay hot in L2.

Layout note that is worth more than any tuning: the histogram reads bins
**feature-major** (`bins_t[f * n + r]`), so adjacent threads touch adjacent
addresses. The row-major layout the CPU path uses would cost `d` transactions per
warp. `mxgb_gpu_quantize` writes both layouts in one pass, so the transpose is
free.

Every entry point returns `0` on success and a negative code otherwise —
`DeviceContext()` raises (including OOM when another process owns the card) and an
`@export ... abi("C")` function cannot, so each wrapper catches internally and the
Python layer falls back to the CPU kernels.
"""

from std.atomic import Atomic
from max.gpu import barrier, block_dim, block_idx, global_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from std.math import isnan
from std.memory import stack_allocation

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]

comptime BLOCK = 256
# Upper bound on bins per feature; XGBoost's default max_bin is 256 and the
# shared histogram is 2 * MAX_BIN * 8 bytes = 4 KiB per block at this size.
comptime MAX_BIN = 256
comptime ROWS_PER_BLOCK = 4096

comptime ERR_UNAVAILABLE = -1
comptime ERR_ARGS = -2
comptime ERR_MAX_BIN = -3


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------


def quantize_kernel(
    x: UnsafePointer[Float64, AnyOrigin[mut=True]],
    cuts: UnsafePointer[Float64, AnyOrigin[mut=True]],
    bins: UnsafePointer[Int64, AnyOrigin[mut=True]],
    bins_t: UnsafePointer[Int64, AnyOrigin[mut=True]],
    n: Int32,
    d: Int32,
    cut_count: Int32,
):
    var rows = Int(n)
    var cols = Int(d)
    var per_feature = Int(cut_count)
    var i = Int(global_idx.x)
    if i >= rows * cols:
        return
    var r = i // cols
    var f = i % cols
    var value = x[i]
    if isnan(value):
        bins[i] = -1
        bins_t[f * rows + r] = -1
        return
    var lo = 0
    var hi = per_feature
    while lo < hi:
        var mid = (lo + hi) // 2
        if value <= cuts[f * per_feature + mid]:
            hi = mid
        else:
            lo = mid + 1
    bins[i] = Int64(lo)
    bins_t[f * rows + r] = Int64(lo)


def hist_kernel(
    bins_t: UnsafePointer[Int64, AnyOrigin[mut=True]],
    grad: UnsafePointer[Float64, AnyOrigin[mut=True]],
    hess: UnsafePointer[Float64, AnyOrigin[mut=True]],
    row_nodes: UnsafePointer[Int64, AnyOrigin[mut=True]],
    hist_grad: UnsafePointer[Float64, AnyOrigin[mut=True]],
    hist_hess: UnsafePointer[Float64, AnyOrigin[mut=True]],
    n: Int32,
    d: Int32,
    max_bin: Int32,
    node: Int32,
):
    """Accumulate one node's gradient/hessian histogram for one feature.

    grid = (row chunks, d); each block owns feature `block_idx.y`. `node < 0`
    means "every row" (the depth-0 histogram).
    """
    var rows = Int(n)
    var cols = Int(d)
    var bins_per_feature = Int(max_bin)
    var target = Int(node)
    var f = Int(block_idx.y)
    if f >= cols:
        return
    var sg = stack_allocation[MAX_BIN, Float64, address_space = AddressSpace.SHARED]()
    var sh = stack_allocation[MAX_BIN, Float64, address_space = AddressSpace.SHARED]()
    var tx = Int(thread_idx.x)
    var b = tx
    while b < bins_per_feature:
        sg[b] = 0.0
        sh[b] = 0.0
        b += BLOCK
    barrier()

    var chunk_start = Int(block_idx.x) * ROWS_PER_BLOCK
    var chunk_end = chunk_start + ROWS_PER_BLOCK
    if chunk_end > rows:
        chunk_end = rows
    var r = chunk_start + tx
    while r < chunk_end:
        var take = True
        if target >= 0:
            take = Int(row_nodes[r]) == target
        if take:
            var bin = Int(bins_t[f * rows + r])
            if bin >= 0:
                _ = Atomic.fetch_add(sg + bin, grad[r])
                _ = Atomic.fetch_add(sh + bin, hess[r])
        r += BLOCK
    barrier()

    var base = f * bins_per_feature
    var k = tx
    while k < bins_per_feature:
        var g = sg[k]
        var h = sh[k]
        if g != 0.0 or h != 0.0:
            _ = Atomic.fetch_add(hist_grad + base + k, g)
            _ = Atomic.fetch_add(hist_hess + base + k, h)
        k += BLOCK


def predict_kernel(
    x: UnsafePointer[Float64, AnyOrigin[mut=True]],
    x_t: UnsafePointer[Float64, AnyOrigin[mut=True]],
    features: UnsafePointer[Int64, AnyOrigin[mut=True]],
    thresholds: UnsafePointer[Float64, AnyOrigin[mut=True]],
    defaults: UnsafePointer[Int64, AnyOrigin[mut=True]],
    leaves: UnsafePointer[Float64, AnyOrigin[mut=True]],
    pred: UnsafePointer[Float64, AnyOrigin[mut=True]],
    n: Int32,
    d: Int32,
    n_trees: Int32,
    max_nodes: Int32,
    base_margin: Float64,
    feature_major: Int32,
):
    var rows = Int(n)
    var cols = Int(d)
    var stride = Int(max_nodes)
    var transposed = feature_major != 0
    var r = Int(global_idx.x)
    if r >= rows:
        return
    var margin = base_margin
    for tree in range(Int(n_trees)):
        var tree_base = tree * stride
        var node = 0
        while features[tree_base + node] >= 0:
            var f = Int(features[tree_base + node])
            # Row-major `x[r * d + f]` makes adjacent threads read addresses `d`
            # apart — one memory transaction per thread instead of per warp.
            # Feature-major `x_t[f * n + r]` is fully coalesced, because every
            # thread in a warp walks the same node and therefore the same f.
            var value = x_t[f * rows + r] if transposed else x[r * cols + f]
            var go_left = False
            if isnan(value):
                go_left = defaults[tree_base + node] != 0
            else:
                go_left = value <= thresholds[tree_base + node]
            node = 2 * node + 1 if go_left else 2 * node + 2
        margin += leaves[tree_base + node]
    pred[r] = margin


# --------------------------------------------------------------------------
# C ABI
# --------------------------------------------------------------------------


@export("mxgb_gpu_available")
def mxgb_gpu_available() abi("C") -> Int:
    """1 when a usable accelerator context can be created right now, else 0.

    Checked rather than assumed: the card may be owned by another process, in
    which case `DeviceContext()` raises CUDA_ERROR_OUT_OF_MEMORY and the caller
    must silently use the CPU kernels.
    """
    try:
        var ctx = DeviceContext()
        _ = ctx.name()
        return 1
    except:
        return 0


@export("mxgb_gpu_quantize")
def mxgb_gpu_quantize(
    x_addr: Int,
    cuts_addr: Int,
    bins_addr: Int,
    bins_t_addr: Int,
    n: Int,
    d: Int,
    max_bin: Int,
) abi("C") -> Int:
    if n <= 0 or d <= 0 or max_bin < 2:
        return ERR_ARGS
    if x_addr == 0 or cuts_addr == 0 or bins_addr == 0 or bins_t_addr == 0:
        return ERR_ARGS
    var cut_count = max_bin - 1
    var total = n * d
    try:
        var ctx = DeviceContext()
        var dx = ctx.enqueue_create_buffer[DType.float64](total)
        var dc = ctx.enqueue_create_buffer[DType.float64](d * cut_count)
        var db = ctx.enqueue_create_buffer[DType.int64](total)
        var dbt = ctx.enqueue_create_buffer[DType.int64](total)
        ctx.enqueue_copy(dx, fp(x_addr))
        ctx.enqueue_copy(dc, fp(cuts_addr))
        ctx.enqueue_function[quantize_kernel](
            dx,
            dc,
            db,
            dbt,
            Int32(n),
            Int32(d),
            Int32(cut_count),
            grid_dim=(total + BLOCK - 1) // BLOCK,
            block_dim=BLOCK,
        )
        ctx.enqueue_copy(ip(bins_addr), db)
        ctx.enqueue_copy(ip(bins_t_addr), dbt)
        ctx.synchronize()
        return 0
    except:
        return ERR_UNAVAILABLE


@export("mxgb_gpu_hist")
def mxgb_gpu_hist(
    bins_t_addr: Int,
    grad_addr: Int,
    hess_addr: Int,
    row_nodes_addr: Int,
    hist_grad_addr: Int,
    hist_hess_addr: Int,
    n: Int,
    d: Int,
    max_bin: Int,
    node: Int,
) abi("C") -> Int:
    """Gradient/hessian histograms for one node, all features, in one launch."""
    if n <= 0 or d <= 0 or max_bin < 2:
        return ERR_ARGS
    if max_bin > MAX_BIN:
        return ERR_MAX_BIN
    if bins_t_addr == 0 or grad_addr == 0 or hess_addr == 0:
        return ERR_ARGS
    if hist_grad_addr == 0 or hist_hess_addr == 0:
        return ERR_ARGS
    var hist_len = d * max_bin
    var row_chunks = (n + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK
    try:
        var ctx = DeviceContext()
        var dbt = ctx.enqueue_create_buffer[DType.int64](n * d)
        var dg = ctx.enqueue_create_buffer[DType.float64](n)
        var dh = ctx.enqueue_create_buffer[DType.float64](n)
        var dn = ctx.enqueue_create_buffer[DType.int64](n)
        var dhg = ctx.enqueue_create_buffer[DType.float64](hist_len)
        var dhh = ctx.enqueue_create_buffer[DType.float64](hist_len)
        ctx.enqueue_copy(dbt, ip(bins_t_addr))
        ctx.enqueue_copy(dg, fp(grad_addr))
        ctx.enqueue_copy(dh, fp(hess_addr))
        if row_nodes_addr != 0:
            ctx.enqueue_copy(dn, ip(row_nodes_addr))
        else:
            ctx.enqueue_memset(dn, Int64(0))
        ctx.enqueue_memset(dhg, Float64(0.0))
        ctx.enqueue_memset(dhh, Float64(0.0))
        ctx.enqueue_function[hist_kernel](
            dbt,
            dg,
            dh,
            dn,
            dhg,
            dhh,
            Int32(n),
            Int32(d),
            Int32(max_bin),
            Int32(node),
            grid_dim=(row_chunks, d),
            block_dim=BLOCK,
        )
        ctx.enqueue_copy(fp(hist_grad_addr), dhg)
        ctx.enqueue_copy(fp(hist_hess_addr), dhh)
        ctx.synchronize()
        return 0
    except:
        return ERR_UNAVAILABLE


@export("mxgb_gpu_predict")
def mxgb_gpu_predict(
    x_addr: Int,
    feature_addr: Int,
    threshold_addr: Int,
    default_addr: Int,
    leaf_addr: Int,
    pred_addr: Int,
    n: Int,
    d: Int,
    n_trees: Int,
    max_nodes: Int,
    base_margin: Float64,
    x_t_addr: Int,
) abi("C") -> Int:
    if n <= 0 or d <= 0 or n_trees <= 0 or max_nodes <= 0:
        return ERR_ARGS
    if x_addr == 0 or feature_addr == 0 or threshold_addr == 0:
        return ERR_ARGS
    if default_addr == 0 or leaf_addr == 0 or pred_addr == 0:
        return ERR_ARGS
    var tree_len = n_trees * max_nodes
    try:
        var ctx = DeviceContext()
        var dx = ctx.enqueue_create_buffer[DType.float64](n * d)
        var dxt = ctx.enqueue_create_buffer[DType.float64](n * d if x_t_addr != 0 else 1)
        var df = ctx.enqueue_create_buffer[DType.int64](tree_len)
        var dt = ctx.enqueue_create_buffer[DType.float64](tree_len)
        var dd = ctx.enqueue_create_buffer[DType.int64](tree_len)
        var dl = ctx.enqueue_create_buffer[DType.float64](tree_len)
        var dp = ctx.enqueue_create_buffer[DType.float64](n)
        ctx.enqueue_copy(dx, fp(x_addr))
        if x_t_addr != 0:
            ctx.enqueue_copy(dxt, fp(x_t_addr))
        ctx.enqueue_copy(df, ip(feature_addr))
        ctx.enqueue_copy(dt, fp(threshold_addr))
        ctx.enqueue_copy(dd, ip(default_addr))
        ctx.enqueue_copy(dl, fp(leaf_addr))
        ctx.enqueue_function[predict_kernel](
            dx,
            dxt,
            df,
            dt,
            dd,
            dl,
            dp,
            Int32(n),
            Int32(d),
            Int32(n_trees),
            Int32(max_nodes),
            base_margin,
            Int32(1) if x_t_addr != 0 else Int32(0),
            grid_dim=(n + BLOCK - 1) // BLOCK,
            block_dim=BLOCK,
        )
        ctx.enqueue_copy(fp(pred_addr), dp)
        ctx.synchronize()
        return 0
    except:
        return ERR_UNAVAILABLE
