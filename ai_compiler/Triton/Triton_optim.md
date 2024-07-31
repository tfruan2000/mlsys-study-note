# Triton kernel optim

本文记录下本人优化 `Triton Kernel` 的思路，由于不了解 `Cuda` 编程以及对 `GPU` 体系结构知识只是一知半解，所以本文设计的优化思路都比较通用(aka **naive**)。

kernel写法上请参考 [triton language guide](https://triton-lang.org/main/python-api/triton.language.html)、[triton tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials)、以及[flaggems](https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops)等项目，网络资料很不错~

---

IMO，对 `Triton Kernel` 的优化过程可以简单分为以下两种(因为我目前只会这两步)，本文只涉及第一种：

- 浅层优化：通过替换算子、合并kernel、拆时间片循环等方式实现初步优化。
- 深层优化：分析下降所得IR，使用perf工具，对照算子库实现等方式，优化kernel的下降行为。

---

以优化 [flaggems](https://github.com/FlagOpen/FlagGems/tree/master) 中的 [layernorm kernel](https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/layernorm.py) 的 `backward` 为主线讲解。因为我只用了较为简单通用的方法，使用前向也是一个样优化，就不再说明。

## perf test

本文中不介绍相关环境的配置，想上手的同学根据 [README](https://github.com/FlagOpen/FlagGems/blob/master/README.md) 配置就应该不会有啥问题。

正好最近的 [commit](https://github.com/FlagOpen/FlagGems/commit/8435781a6dfb0f52b88ba7f917d3b34141373c05) 支持了对算子的 `backward` 进行 `perf` 测试，只不过现在测试函数不全，得自己添加一下，例如测试 `layernorm backward kernel`  可以在 `benchmark/test_reduction_perf.py` 中添加：

```python
def test_perf_layernorm_backward():
    def layer_norm_args(dtype, batch, size):
        inp = torch.randn([batch, size], dtype=dtype, device="cuda")
        weight = torch.randn([size,], dtype=dtype, device="cuda",)
        bias = torch.randn([size,], dtype=dtype,device="cuda",)
        return (inp, [size,], weight, bias, )
    bench = Benchmark(
        op_name="layernorm",
        torch_op=torch.layer_norm,
        arg_func=layer_norm_args,
        dtypes=FLOAT_DTYPES,
        batch=REDUCTION_BATCH,
        sizes=SIZES,
        is_backward=True,
    )
    bench.run()
```

然后运行

```bash
cd benchmark
pytest test_reduction_perf.py::test_perf_layernorm_backward -s
```

## kernel optim

`layernorm` 的 `backward kernel` 在 `flaggems` 中的[实现](https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/layernorm.py#L230)分成了两个，一个计算 `in_grad`、一个计算 `weight_grad` 和 `bias_grad`。

因为 `in_grad` 的每个值都需要完整地遍历 `col`（即N），而 `weight_grad` 和 `bias_grad`的每个值需要完整地遍历 `row`（即M）。为了更清晰理解计算行为，可以看：[这篇blog](https://blog.csdn.net/pgsld2333/article/details/122576365)中`layernorm backward` 的计算推导。

当前实现功能上基本能cover所有的case，**性能上我也不知道如何，因为我还没在GPU测过hhh**。但还是可以强行优化一下，而且在我的环境下确实有性能提升叻，并且精度测试没问题。


### 合并 kernel

当看到 `kernel` 分为了两个，第一反应是**合并**一下，但是由于 `in_grad`、 `weight_grad` 和 `bias_grad` 的计算行为分别依赖不同的遍历，导致难以合并。

这时候翻看下官方 `tutorial` [layernorm backward](https://github.com/triton-lang/triton/blob/main/python/tutorials/05-layer-norm.py#L275)，虽然也是两个kernel，但是第二个kernel[本质上只做了sum](https://github.com/triton-lang/triton/blob/main/python/tutorials/05-layer-norm.py#L211)，那么我们在第一个kernel中对 `partial_dw` 和 `partial_db` **使用 `atomic_add` 就可以合并为一个kernel**，`atomic_add` 在完成 `add` 后会有 `store` 的行为。

- kernel

优化后的kernel和 `tutorial` 中的实现相似：

```python
@triton.jit
def layer_norm_backward_kernel(DX,  # pointer to the input gradient
                               DY,  # pointer to the output gradient
                               DW,  # pointer to the partial sum of weights gradient
                               DB,  # pointer to the partial sum of biases gradient
                               X,  # pointer to the input
                               W,  # pointer to the weights
                               Mean,  # pointer to the mean
                               Rstd,  # pointer to the 1/std
                               stride,  # how much to increase the pointer when moving by 1 row
                               N,  # number of columns in X
                               BLOCK_COL_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_COL_SIZE)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    # Compute dx
    xhat = (x - mean) * rstd
    wdy = w * dy
    c1 = tl.sum(xhat * wdy, axis=0)
    c2 = tl.sum(wdy, axis=0)
    dx = (wdy - (xhat * c1 + c2) / N) * rstd
    # Write dx
    tl.store(DX + cols, dx, mask=mask)
    # Accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(tl.float32)
    partial_db = (dy).to(tl.float32)
    # 使用 atomic_add 合并第二个 sum kernel
    tl.atomic_add(DW + cols, partial_dw)
    tl.atomic_add(DB + cols, partial_db)
```

- launch func

```python
class LayerNorm(torch.autograd.Function):
    ... # 这里是forward
    @staticmethod
    def backward(ctx, out_grad, mean_grad, rstd_grad):
        logging.debug("GEMS LAYERNORM BACKWARD")
        out_grad = out_grad.contiguous()
        x, weight, mean, rstd = ctx.saved_tensors
        M, N = ctx.M, ctx.N

        # tutorial 中设置的超参数，这里的参数也需要根据硬件来改！！
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        # alloc for out
        in_grad = torch.empty_like(x)
        # 为了保证使用 atomic_add 支持的数据类型，所以直接使用float32
        weight_grad = torch.zeros((weight.shape[0],), dtype=torch.float, device=weight.device)
        bias_grad = torch.zeros((weight.shape[0],), dtype=torch.float, device=weight.device)
        layer_norm_backward_kernel[(M, )](
            in_grad, out_grad, weight_grad, bias_grad, x, weight, mean, rstd,
            N, BLOCK_COL_SIZE=BLOCK_SIZE, num_warps = num_warps,
        )
        weight_grad = weight_grad.to(x.dtype)
        bias_grad = bias_grad.to(x.dtype)
```

- tuning config

由于当前kernel没有需要tuning的超参数，所以不需要设置

- 问题分析

根据上文的 `launch` 函数可以注意到，当前kernel主要存在以下两个问题：

(1)**对M是有大小限制**，launch grid直接为(M, 1, 1)，可能超出 `grid` 的限制。

(2)**对N是有大小限制**，导致kernel无法覆盖所有case

针对问题(1)，我们选择**对M进行拆时间片循环**。

针对问题(2)，我们选择修改`flaggems`官方实现（也增加拆时间片循环），作为 `fallback kernel`，然后根据N的大小去选择最终使用的kernel。

下节我们将依次解决这两个问题。

### 拆时间片循环

将kernel的`grid`按如下设置，保证不超过`grid`的最大限制，其中`MAX_GRID_NUM`是一个人为设置的超参数，根据硬件设置就好。

```python
grid = lambda META: (min(triton.cdiv(M, META['BLOCK_ROW_SIZE']), MAX_GRID_NUM),)
```

使用该 `grid` 后，每个kernel处理的数据大小就不一定为一个 `[1, BLOCK_COL_SIZE]`。每个`pid`处理1或多个大小为 `[BLOCK_ROW_SIZE, BLOCK_COL_SIZE]` 的数据块：

```bash
pid = tl.program_id(0)
row_start = pid * BLOCK_ROW_SIZE
total_num = min(triton.cdiv(M, META['BLOCK_ROW_SIZE'])
step = total_num * BLOCK_ROW_SIZE
cols = tl.arange(0, BLOCK_COL_SIZE)
for row in range(row_start, M, step):
    # 每次处理 [BLOCK_ROW_SIZE, BLOCK_COL_SIZE]
    row_off = row + tl.arange(0, BLOCK_ROW_SIZE)
```

- kernel

然后以上一步的kernel为基础，增加拆 `row`（即M）的循环，循环中一次处理 `[BLOCK_ROW_SIZE, BLOCK_COL_SIZE]` 大小的数据：

```python
@triton.jit
def layer_norm_backward_kernel(
        DX,  # pointer to the input gradient
        DY,  # pointer to the output gradient
        DW,  # pointer to the partial sum of weights gradient
        DB,  # pointer to the partial sum of biases gradient
        X,  # pointer to the input
        W,  # pointer to the weights
        Mean,  # pointer to the mean
        Rstd,  # pointer to the 1/std
        M,  # number of rows in X
        N,  # number of columns in X
        BLOCK_ROW_SIZE: tl.constexpr, BLOCK_COL_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_ROW_SIZE
    cols = tl.arange(0, BLOCK_COL_SIZE)
    num_jobs = tl.num_programs(axis=0)
    step = num_jobs * BLOCK_ROW_SIZE
    col_mask = cols < N

    X += cols[None, :]
    DY += cols[None, :]
    W += cols[None, :]
    DX += cols[None, :]
    w = tl.load(W, mask = col_mask, other = 0.0).to(tl.float32)

    partial_dw = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    partial_db = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
    for row in range(row_start, M, step):
        row_off = row + tl.arange(0, BLOCK_ROW_SIZE)
        row_mask = row_off < M
        # Load data to SRAM
        off = row_off[:, None] * N # row的stride为 BLOCK_ROW_SIZE * N
        mask = row_mask[:, None] and col_mask
        x = tl.load(X + off, mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + off, mask, other=0.0).to(tl.float32)
        mean = tl.load(Mean + row_off, mask = row_mask)[:, None].to(tl.float32)
        rstd = tl.load(Rstd + row_off, mask = row_mask)[:, None].to(tl.float32)
        # Compute dx
        x_hat = (x - mean) * rstd
        wdy = w * dy
        #  [BLOCK_ROW_SIZE, BLOCK_COL_SIZE] -> [BLOCK_ROW_SIZE]
        c1 = tl.sum(x_hat * wdy, axis=1)[:, None]
        c2 = tl.sum(wdy, axis=1)[:, None]
        dx = (wdy - (x_hat * c1 + c2) / N) * rstd
        # Accumulate partial sums for dw/db
        partial_dw += (dy * x_hat).to(tl.float32)
        partial_db += (dy).to(tl.float32)
        # Write dx
        tl.store(DX + off, dx.to(x.dtype), mask=mask)

    #  [BLOCK_ROW_SIZE, BLOCK_COL_SIZE] -> [BLOCK_COL_SIZE]
    dw = tl.sum(partial_dw, axis=0)
    db = tl.sum(partial_db, axis=0)
    tl.atomic_add(DW + cols, dw)
    tl.atomic_add(DB + cols, db)
```

- launch func

backward的launch函数部分也是模仿tutorial写的，人为设置一些超参数

```python
class LayerNorm(torch.autograd.Function):
    ... # 这里是forward
    @staticmethod
    def backward(ctx, out_grad, mean_grad, rstd_grad):
        logging.debug("GEMS LAYERNORM BACKWARD")
        out_grad = out_grad.contiguous()
        x, weight, mean, rstd = ctx.saved_tensors
        M, N = ctx.M, ctx.N

        # tutorial 中设置的超参数，这里的参数也需要根据硬件来改！！
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        in_grad = torch.empty_like(x)
        # 为了保证使用 atomic_add 支持的数据类型，所以直接使用float32
        weight_grad = torch.zeros((weight.shape[0],), dtype=torch.float, device=weight.device)
        bias_grad = torch.zeros((weight.shape[0],), dtype=torch.float, device=weight.device)
        grid = lambda META: (min(triton.cdiv(M, META['BLOCK_ROW_SIZE']), MAX_GRID_NUM),)
        layer_norm_backward_kernel[grid](
            in_grad, out_grad, weight_grad, bias_grad, x, weight, mean, rstd,
            M, N, BLOCK_COL_SIZE=BLOCK_SIZE, num_warps = num_warps,
        )
        weight_grad = weight_grad.to(x.dtype)
        bias_grad = bias_grad.to(x.dtype)
```

- tuning config

对 `row` 进行拆分后，我们就需要tuning `BLOCK_ROW_SIZE`，但由于kernel一次还是处理完整的 `col`，所以 `BLOCK_ROW_SIZE` 也不能设置多大。tuning 参数仁者见仁，根据场景做 编译时间和性能的trade-off 就好

```python
def cfggen_bw():
    block_m = [1, 4, 16, 32]
    # num_stages 这里就不提供大概设置多少了
    num_stages = [...]
    configs=[
        triton.Config({"BLOCK_ROW_SIZE": m}, num_stages=s)
        for m in block_m
        for s in num_stages
    ],
    return configs
```

需要注意的是，使用 `atomic_add` 后，若同时设置了多个 `tuning config` ，会有精度问题，因为每次选择新的 `config` 时没有对 `atomic_add` 的 `target` 重置为0。需要在设计 `tuning config` 时加一个 `reset_to_zero`，大致如下。（这个是大佬告诉我的）

```python
@libentry() # 这是 flaggems 需要加的
@triton.autotune(configs=cfggen_bw(), key=["M", "N"], reset_to_zero=["DW", "DB"])
```

---

让我们回顾下初次优化kernel后提出的两个问题：

(1)**对M是有大小限制**，launch grid直接为(M, 1, 1)，可能超出 `grid` 的限制。

(2)**对N是有大小限制**，导致kernel无法覆盖所有case

前文已经解决了问题(1)，现在我们来考虑问题(2)。我们选择修改`flaggems`官方原本的实现，作为 `fallback kernel`。

- kernel

实现和[官方](https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/layernorm.py#L102)比较相似，只是增加了时间片拆分，不再展开讲解。

```python
def cfggen_input_bw():
    block_m = [1, 4, 16, 32]
    block_n = [32, 256, 1024, 2048]
    # num_stages 和 num_warps 这里就不提供大概设置多少了
    num_stages = [...]
    num_warps = [...]
    configs = [
        triton.Config({"BLOCK_ROW_SIZE": m, "BLOCK_COL_SIZE": n}, num_warps=w, num_stages=s)
        for m in block_m
        for n in block_n
        for s in num_stages
        for w in num_warps
    ]
    return configs

@libentry() # 这是 flaggems 需要加的
@triton.autotune(configs=cfggen_input_bw(), key=["M", "N"])
@triton.jit
def input_backward_kernel(
    dY,
    X,
    W,
    Mean,
    Rstd,
    dX,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_ROW_SIZE
    num_jobs = tl.num_programs(axis=0)
    step = num_jobs * BLOCK_ROW_SIZE

    for row in range(row_start, M, step):
        row_off = row + tl.arange(0, BLOCK_ROW_SIZE)
        mean = tl.load(Mean + row_off, mask = row_off < M, other = 0.0)[:, None].to(tl.float32)
        rstd = tl.load(Rstd + row_off, mask = row_off < M, other = 0.0)[:, None].to(tl.float32)

        row_mask = row_off[:, None] < M
        off = row_off[:, None] * N
        new_dY = dY + off
        new_X = X + off
        new_DX = dX + off

        dx_part2 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
        dx_part3 = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)

        for off in range(0, N, BLOCK_COL_SIZE):
            cols = off + tl.arange(0, BLOCK_COL_SIZE)
            col_mask = cols[None, :] < N
            mask = row_mask and col_mask
            dy = tl.load(new_dY + cols[None, :], mask, other = 0.0).to(tl.float32)
            x = tl.load(new_X + cols[None, :], mask, other = 0.0).to(tl.float32)
            x_hat = (x - mean) * rstd
            w = tl.load(W + cols, mask=cols < N).to(tl.float32)
            wdy = dy * w
            dx_part2 += wdy
            dx_part3 += wdy * x_hat

        dx_2 = tl.sum(dx_part2, axis=1)[:, None]
        dx_3 = tl.sum(dx_part3, axis=1)[:, None]

        for off in range(0, N, BLOCK_COL_SIZE):
            cols = off + tl.arange(0, BLOCK_COL_SIZE)
            col_mask = cols[None, :] < N
            mask = row_mask and col_mask
            dy = tl.load(new_dY + cols[None, :], mask, other = 0.0).to(tl.float32)
            x = tl.load(new_X + cols[None, :], mask, other = 0.0).to(tl.float32)
            w = tl.load(W + cols, mask=cols < N, other = 0.0).to(tl.float32)
            x_hat = (x - mean) * rstd
            wdy = dy * w
            dx = rstd * (wdy - (dx_2 + x_hat * dx_3) / N)
            tl.store(new_DX + cols, dx.to(x.dtype), mask=mask)


def cfggen_wb_bw():
    block_m = [32, 64, 128, 512, 1024]
    block_n = [1, 4, 16, 32]
    # num_stages 和 num_warps 这里就不提供大概设置多少了
    num_stages = [...]
    num_warps = [...]
    configs = [
        triton.Config({"BLOCK_ROW_SIZE": m, "BLOCK_COL_SIZE": n}, num_stages=s)
        for m in block_m
        for n in block_n
        for s in num_stages
        for w in num_warps
    ]
    return configs

@libentry()
@triton.autotune(configs=cfggen_wb_bw(), key=["M", "N"])
@triton.jit
def weight_bias_backward_kernel(
    dY,
    X,
    Mean,
    Rstd,
    dW,
    dB,
    M,
    N,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    col_start = pid * BLOCK_COL_SIZE
    num_jobs = tl.num_programs(axis=0)
    step = num_jobs * BLOCK_COL_SIZE

    for col in range(col_start, N, step):
        col_off = col + tl.arange(0, BLOCK_COL_SIZE)[None, :]
        col_mask = col_off < N

        new_dY = dY + col_off
        new_X = X + col_off
        new_dW = dW + col_off
        new_dB = dB + col_off

        accW = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)
        accB = tl.zeros([BLOCK_ROW_SIZE, BLOCK_COL_SIZE], dtype=tl.float32)

        for off in range(0, M, BLOCK_ROW_SIZE):
            rows = off + tl.arange(0, BLOCK_ROW_SIZE)
            row_mask = rows[:, None] < M
            mask = row_mask and col_mask
            dy = tl.load(new_dY + rows[:, None] * N, mask, other = 0.0).to(tl.float32)
            x = tl.load(new_X + rows[:, None] * N, mask, other = 0.0).to(tl.float32)
            mean = tl.load(Mean + rows, mask = rows < M, other = 0.0)[:, None].to(tl.float32)
            rstd = tl.load(Rstd + rows, mask = rows < M, other = 0.0)[:, None].to(tl.float32)
            x_hat = (x - mean) * rstd
            accW += dy * x_hat
            accB += dy
        dw = tl.sum(accW, axis=0)
        db = tl.sum(accB, axis=0)
        tl.store(new_dW, dw[None, :], mask=col_mask)
        tl.store(new_dB, db[None, :], mask=col_mask)
```

- launch func

```python
# 人为设置超参数，这些都和硬件参数有关，在这里我都是乱设，一学就废
MAX_COL_LEN_BACKWARD = 16392
MAX_GRID_NUM = 65535

class LayerNorm(torch.autograd.Function):
    ... # 这里是forward
    @staticmethod
    def backward(ctx, out_grad, mean_grad, rstd_grad):
        logging.debug("GEMS LAYERNORM BACKWARD")
        out_grad = out_grad.contiguous()
        x, weight, mean, rstd = ctx.saved_tensors
        M, N = ctx.M, ctx.N
        in_grad = torch.empty_like(x)
        # tutorial 中设置的超参数，这里的参数也需要根据硬件来改！！
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if (N <= BLOCK_SIZE) and (BLOCK_SIZE <= MAX_COL_LEN_BACKWARD):
            num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
            # 为了保证使用 atomic_add 支持的数据类型，所以直接使用float32
            weight_grad = torch.zeros((weight.shape[0],), dtype=torch.float, device=weight.device)
            bias_grad = torch.zeros((weight.shape[0],), dtype=torch.float, device=weight.device)
            grid = lambda META: (min(triton.cdiv(M, META['BLOCK_ROW_SIZE']), MAX_GRID_NUM),)
            layer_norm_backward_kernel[grid](
                in_grad, out_grad, weight_grad, bias_grad, x, weight, mean, rstd,
                M, N, BLOCK_COL_SIZE=BLOCK_SIZE, num_warps = num_warps,
            )
        else:
            grid = lambda META: (min(triton.cdiv(M, META['BLOCK_ROW_SIZE']), MAX_GRID_NUM),)
            input_backward_kernel[grid](
                out_grad, x, weight, mean, rstd, in_grad, M, N,
            )
            weight_grad = torch.empty_like(weight)
            bias_grad = torch.empty_like(weight)
            grid = lambda META: (min(triton.cdiv(N, META['BLOCK_COL_SIZE']), MAX_GRID_NUM),)
            weight_bias_backward_kernel[grid](
                out_grad, x, mean, rstd, weight_grad, bias_grad, M, N,
            )
        return in_grad, None, weight_grad, bias_grad, None, None
```

### 替换算子

简单的算子替换:

- `tl.max(a, 0.0)` 可以换成 `tl.where(a > 0, a, 0.0)`
- `x` 和 `y` 在 `tl.load` 时用了mask，随后的 `tl.where(mask, x - y, 0.0)` 可以删除
- 大规模 `reduce(10000->1)` -> 多级 `reduce(10000->100->1)`
- ...

算法替换：

例如：累乘 -> 二分乘法

算法实现上

```python
tmp = 1
def mul_acc(x, l, h):
    tmp = 1
    for i in rang(l, h)
        tmp *= i
    return tmp

->

def binary_mul(x, l, h):
    if l >= h:
        return 1
    if h - l == 1:
        return x[l]
    mid = (l + h) // 2
    return binary_mul(x, l, mid) + binary_mul(x, mid, h)
```

以优化 flaggems 中的 [prod](https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/prod.py#L18) 为例：

```python
@triton.jit
def prod_kernel_mid(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=1.0).to(tl.float32)
    mid_value = tl.reduce(inp_val, axis=0, combine_fn=reduce_mul)
    mid_ptr = mid + pid
    tl.store(mid_ptr, mid_value.to(inp_val.dtype))
```

首先拆时间片循环：

```python
@triton.jit
def prod_kernel_mid(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(axis=0)
    block_start = pid * BLOCK_SIZE
    step = num_jobs * BLOCK_SIZE
    _tmp = tl.full([BLOCK_SIZE], value=1.0, dtype=tl.float32)
    for off in range(block_start, M, step):
        offset = off + tl.arange(0, BLOCK_SIZE)
        mask = offset < M
        inp_val = tl.load(inp_ptrs, mask=mask, other=1.0).to(tl.float32)
        _tmp = _tmp * input_val
    mid_value = tl.reduce(_tmp, axis=0, combine_fn=reduce_mul)
    tl.store(mid_ptr + pid, mid_value.to(inp_val.dtype))

# launch func
# grid = lambda META: min((triton.cdiv(M, MEAT['BLOCK_SIZE']), MAX_GRID_NUM),)
```

然后将 `_tmp` 的 累乘优化为二分乘法（`reduce_mul`->归约规约）

```python
mid_value = tl.reduce(_tmp, axis=0, combine_fn=reduce_mul)
tl.store(mid_ptr + pid, mid_value.to(inp_val.dtype))

->

# triton.Config({"BLOCK_SIZE": m, "ITER_NUM": math.log2(m)} for m in [...])
# 将数组 _tmp 前一半的元素与后一半的元素相乘，并将结果存储在前一半的位置
# 以 BLOCK_SIZE = 16 为例，ITER_NUM=4
# 例： x   _tmp[:BLOCK_SIZE // (2 ** 1)]   _tmp[BLOCK_SIZE // (2 ** 1):BLOCK_SIZE // (2 ** (x - 1))]
#     1   _tmp[:8]                        _tmp[8:16]
#     2   _tmp[:4]                        _tmp[4:8]
#     3   _tmp[:2]                        _tmp[2:4]
#     4   _tmp[:1]                        _tmp[1:2]
for x in tl.static_range(1, int(ITER_NUM), 1):
    # 等下于 _tmp[:BLOCK_SIZE // (2 ** x)] = reduce_mul(_tmp[:BLOCK_SIZE // (2 ** x)], _tmp[BLOCK_SIZE // (2 ** x):BLOCK_SIZE // (2 ** (x - 1))])
    _tmp[:BLOCK_SIZE // (2 ** x)] = _tmp[:BLOCK_SIZE // (2 ** x)] * _tmp[BLOCK_SIZE // (2 ** x):BLOCK_SIZE // (2 ** (x - 1))]
# reduce(_tmp[:2])
res = tl.reduce(_tmp[:BLOCK_SIZE // (2 ** (ITER_NUM - 1))], axis=0, combine_fn=reduce_mul)
tl.store(mid_ptr + pid, res)

# 如果BLOCK_SIZE设置的都是二次幂，并且 {"ITER_NUM": math.log2(m)+1} ，则直接store即可
# tl.store(mid_ptr + pid, _tmp[0])
```

需要注意的是，上述并行归约优化在`tl.reduce`下降过程完成更具泛化性。

---

自此，本文对 `Triton Kernel` 的优化行为只涉及**替换算子、合并kernel、拆时间片循环**等初步优化，并不包含：

- 使用硬件特性优化
- 修改lowering源码

这两者和硬件相关性太大了，各家有各家的说法，再说我也确实不太懂硬件架构（修改源码的地方这里也不好放出来），只能用此文记录下自己naive的优化行为，期望有更多大佬分享优化的经验。