# PyTorch2.0

DL framework三种运行加速

![Untitled](PyTorch2%200%205cd6be82123c423cb7658c65a754283f/Untitled.png)

Torch Compiler分3步

1. Graph Acquisition: Dynamo (forward) + AOTAutograd (backward)
2. Graph Lowering: ATen / Prim IR
3. Graph Compilation: TorchInductor

![Untitled](PyTorch2%200%205cd6be82123c423cb7658c65a754283f/Untitled%201.png)

TorchInductor分为四层，三大核心技术

1. 核心 IR 是 loop level IR，是 python callable。做 codegen 或者 analysis，只需要直接 execute。
2. 用 SymPy（一个符号计算库）支持动态 shape。
3. 在 CPU 上用的 OpenMP，跟 intel 一起搞的。GPU 选了 OpenAI Triton。

AOTInductor = torch.export (whole graph capture) + Inductor (AOT compilation)

![Untitled](PyTorch2%200%205cd6be82123c423cb7658c65a754283f/Untitled%202.png)

Dynamo 的 2 个精髓：

1. partial graph capture。遇到不支持的，就保留并拆分出前后的子图，分别编译。
2. guard。解决了 trace 的经典难题 -- control flow 等导致 capture 不能用。guard 会自动报警，不会出错。

三种模式

![Untitled](PyTorch2%200%205cd6be82123c423cb7658c65a754283f/Untitled%203.png)