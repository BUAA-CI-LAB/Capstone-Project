# CPU+GPU大模型协同推理技术教程-以FlexGen为例
## 1. 协同推理基本原理
### 1.1 大模型推理的显存挑战
随着大语言模型规模的持续增长，推理阶段的显存需求已成为实际部署的主要瓶颈：

| 模型 | 参数量 | FP16显存需求 |
|------|------|------|
| OPT-125M | 125M | 250MB |
| OPT-1.3B | 1.3B | 2.6GB |
| OPT-13B | 13B | 26GB |
| OPT-175B | 175B | 350GB |
> 关键问题：以上仅计算权重开销，实际开销还包括KV Cache、激活值等开销。即使是最小的125M参数模型，其完整推理也需要最少250MB显存，而当模型规模达到13B+时，单卡消费级GPU（通常4-8GB显存）根本无法加载完整模型。

### 1.2 大模型常见协同推理技术
协同推理的核心思想是将计算、存储等任务分布到多个设备（如GPU、CPU、磁盘）上，通过智能调度实现高效推理。主要技术包括：

1. **张量并行** (Tensor Parallelism, TP)

![这是图片](/Co-Inference_Tutorial/images/图一张量并行（候选）.jpg "张量并行")
    - 原理：将单个张量操作（如矩阵乘法）拆分到多个设备上并行计算
    - 适用场景：单层计算过于庞大
    - **优点**：减少单设备计算负载
    - **缺点**：需要设备间频繁通信，延迟高

2. **流水线并行** (Pipeline Parallelism, PP)
![这是图片](/Co-Inference_Tutorial/images/图二流水线并行PP.jpg "张量并行")
    - 原理：将模型按层拆分为多个阶段，各阶段在不同设备上执行，形成流水线
    - 适用场景：超大规模模型推理
    - **优点**：适合大模型，减少单设备内存压力
    - **缺点**：流水线气泡（bubble）导致设备利用率低

3. **序列并行** (Sequence Parallelism, SP)
![这是图片](/Co-Inference_Tutorial/images/序列并行.png "序列并行")
    - 原理：将长序列输入拆分为多个片段，分别在不同设备上处理
    - 适用场景：长文本生成、超长上下文处理
    - **优点**：适合处理超长序列
    - **缺点**：需要跨设备通信处理序列依赖

4. **参数卸载** (Parameter/Data Offloading)
    - 原理：将不活跃的模型参数临时卸载到CPU或磁盘，需要时再加载回GPU
    - 适用场景：单层计算过于庞大
    - **优点**：减少单设备计算负载
    - **缺点**：需要设备间频繁通信，延迟高

5. **激活值检查点** (Activation Checkpointing)
    - 原理：不保存所有中间激活值，需要时重新计算
    - 适用场景：内存受限的推理环境
    - **优点**：以计算换内存，显著降低显存需求
    - **缺点**：增加计算量、仅对推理有部分帮助（主要用于训练）
    - 实现方式：
```python
# 伪代码：激活值检查点
def checkpointed_layer_forward(layer, x):
    # 1. 不保存中间激活值
    if torch.is_grad_enabled():
        # 仅保存输入和层参数
        return CheckpointFunction.apply(layer, x)
    else:
        # 推理模式下正常计算
        return layer(x)

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, layer, x):
        # 保存必要信息用于反向传播
        ctx.layer = layer
        ctx.save_for_backward(x)
        return layer(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 重新计算前向传播
        x, = ctx.saved_tensors
        with torch.enable_grad():
            x.requires_grad = True
            y = ctx.layer(x)
            # 计算梯度
            grad_input = torch.autograd.grad(
                y, x, grad_output, retain_graph=False)
        return None, grad_input[0]
```
1. **PD分离、混合并行等各种策略**


## 2.FlexGen核心技术解析（Offloading）
### 2.1 论文核心思想
FlexGen的核心目标是：在***单GPU***上实现***高吞吐***大语言模型推理，***特别针对"后台型"任务***（如模型评测、批量文本生成）优化。
> ***关键创新***：通过***线性规划优化张量放置***、***4-bit压缩***和***Zig-zag/对角块调度***，在有限GPU内存下实现最大批量大小，从而显著提升吞吐量。

- 线性规划优化张量放置：为各种offloading策略定义了一个搜索空间(search space)，即基于某种数学模型约束各个参数取值范围的调度算法，也就是要在数学模型限定的取值范围这个space里search出最优的参数组合。
- 4-bit压缩：通过group-wise的量化方法把weights和KV cache压缩成4-bit格式，减小了offloading时的通信开销和存储开销。
- Zig-zag/对角块调度策略（并行策略）：
  - 关键insight：用batch维度并行打破同batch内seq token的generating过程中时序依赖导致的并发度不足。
  - 通过流水等延迟隐藏方法不断向GPU输送batch样本，且GPU端仅仅进行一个Transformer layer的计算，一旦计算完成就对KVcache、激活、weight权重参数进行checkpoint，也是流水化overlapping的将数据转移到CPU DRAM和磁盘；当batch纬度并行达到一定的阶段后，再进入下一个layer的计算，且也是高优先batch纬度并行，同样利用overlapping技术隐藏CPU Dram和磁盘的访存延迟；依次进入下一个layer；直到所有layer完成。到这里肯可能prompt计算完了，然后Generating的token同样按照这样的并行技巧，优先batch并行，然后进行layer并行



### 2.2 三级存储协同架构
![这是图片](/Co-Inference_Tutorial/images/图6.png "三层示意")
三级存储角色定义：
1. GPU内存：
    - 仅保留当前计算所需的最小张量集
    - 在推理过程中动态管理
2. CPU内存：
    - 作为高速缓存，存储近期使用过的张量
    - 用于存储激活值和KV缓存
3. Disk存储：
    - 容纳完整模型，作为最后的后备存储
    - 在内存不足时使用

### 2.3 核心技术突破
#### 2.3.1. 线性规划张量调度（LP-based Tensor Scheduling）
***LP优化问题形式化***：
FlexGen将张量放置问题建模为线性规划问题，目标是最小化I/O成本：

$$
\begin{align*}
\min_{p} &\quad T/\text{bl}s \\
\text{s.t.} &\quad \text{gpu peak memory} < \text{gpu mem capacity} \\
&\quad \text{cpu peak memory} < \text{cpu mem capacity} \\
&\quad \text{disk peak memory} < \text{disk mem capacity} \\
&\quad wg + wc + wd = 1 \\
&\quad cg + cc + cd = 1 \\
&\quad hg + hc + hd = 1 \\
\end{align*}
$$

***变量说明***：

- p：优化目标（I/O成本）
- T：总token数
- bls：块大小（block size）
- wg,wc,wd：权重在GPU、CPU、磁盘上的分布比例
- cg,cc,cd：KV缓存在GPU、CPU、磁盘上的分布比例
- hg,hc,hd：激活值在GPU、CPU、磁盘上的分布比例

优化过程如下：

```python
#伪代码
def solve_placement_problem(config, model, batch_size):
    # 1. 定义LP问题
    problem = pulp.LpProblem("Tensor_Placement", pulp.LpMinimize)

    # 2. 定义决策变量
    wg = pulp.LpVariable("wg", 0, 1)
    wc = pulp.LpVariable("wc", 0, 1)
    wd = pulp.LpVariable("wd", 0, 1)
    cg = pulp.LpVariable("cg", 0, 1)
    cc = pulp.LpVariable("cc", 0, 1)
    cd = pulp.LpVariable("cd", 0, 1)
    hg = pulp.LpVariable("hg", 0, 1)
    hc = pulp.LpVariable("hc", 0, 1)
    hd = pulp.LpVariable("hd", 0, 1)

    # 3. 定义目标函数
    problem += wg * config["gpu_weight_cost"] + 
               wc * config["cpu_weight_cost"] + 
               wd * config["disk_weight_cost"] +
               cg * config["gpu_cache_cost"] + 
               cc * config["cpu_cache_cost"] + 
               cd * config["disk_cache_cost"] +
               hg * config["gpu_activation_cost"] + 
               hc * config["cpu_activation_cost"] + 
               hd * config["disk_activation_cost"]

    # 4. 定义约束
    # 内存约束
    problem += wg * model.weight_size * config["gpu_memory_factor"] + \
               hg * batch_size * config["activation_memory_factor"] + \
               cg * batch_size * config["cache_memory_factor"] <= config["gpu_memory_capacity"]

    problem += wc * model.weight_size + \
               hc * batch_size + \
               cc * batch_size <= config["cpu_memory_capacity"]

    problem += wd * model.weight_size + \
               hd * batch_size + \
               cd * batch_size <= config["disk_memory_capacity"]

    # 比例约束
    problem += wg + wc + wd == 1
    problem += cg + cc + cd == 1
    problem += hg + hc + hd == 1

    # 5. 求解
    problem.solve()

    # 6. 返回最优解
    return {
        "weight_placement": {"gpu": wg.varValue, "cpu": wc.varValue, "disk": wd.varValue},
        "cache_placement": {"gpu": cg.varValue, "cpu": cc.varValue, "disk": cd.varValue},
        "activation_placement": {"gpu": hg.varValue, "cpu": hc.varValue, "disk": hd.varValue}
    }

```
***LP优化的意义***：
- 通过求解这个LP问题，FlexGen能够找到在给定内存约束下I/O成本最小的张量放置策略
- 9个超参数（wg,wc,wd,cg,cc,cd,hg,hc,hd）共同决定了张量在三级存储中的分布
- 这种优化使FlexGen能够充分利用有限的GPU内存，实现最大可能的批量大小

#### 2.3.2  Zig-zag block Schedule vs Row-by-row Schedule
Zig-zag块调度是FlexGen提出的核心创新，其工作原理和行调度的区别如下：
![这是图片](/Co-Inference_Tutorial/images/zig-zag.png "zig-zag")
***行调度的问题***：
传统的行调度（row-wise scheduling）在生成推理中效率低下
- 每生成一个token都需要重新加载整个模型权重
- 无法有效重用权重，导致大量I/O开销
- 内存使用效率低，限制了批量大小

***Zig-zag块调度的优势***：
1. ***权重复用***：同一列中的所有计算共享权重，权重可以保留在GPU上重复使用
    - 生成 n⋅bls 个token只需加载 n 次完整模型权重
    - 而行调度需要加载 n⋅bls 次
2. ***CPU内存***：
    - 激活值I/O总量：2(2h⋅s⋅bls⋅l+2h⋅bls⋅l⋅(n−1)) 字节
    - KV缓存I/O总量：4h⋅bls⋅l⋅(s⋅n+n(n−1)/2) 字节
    - 相比行调度，I/O操作减少 bls 倍
3. ***内存管理***：
    - 通过限制列的高度，确保激活值和KV缓存不会超出CPU和磁盘内存
    - 峰值内存使用：peak_mem=w+2h⋅bls+4h⋅bls⋅l(s+n)
	
#### 2.3.3 Overlap
![这是图片](/Co-Inference_Tutorial/images/algorithm1.png "算法一")
***重叠技术的关键点***：
- 将权重加载、缓存/激活加载、计算等操作并行化
- 利用计算与I/O的重叠，隐藏传输延迟
- 六个操作可以并行执行（因为它们之间没有依赖关系）：
    1. 加载下一层权重
    2. 存储前一批的激活
    3. 存储前一批的缓存
    4. 加载下一批的激活
    5. 加载下一批的缓存
    6. 计算当前批次

#### 2.3.4 外围参数优化：GPU-batch和Blocksize
lexGen的搜索空间包含两个关键外围参数：
1. ***GPU batch size***：单个GPU批次的token数量（***计算的单位***）
2. ***Number of GPU batches in a block***：块中的GPU批次数量
这两个参数的乘积称为***block size***（***调度的单位***） block_size=gpu_batch_size×num_gpu_batches

***参数优化策略***：
- 外层枚举GPU batch和num of block内层LP，隐性优化外层的两个枚举参数
- 在内存约束下最大化吞吐量

***优化过程***：

```python
def find_optimal_batching(config, model):
    # 1. 初始化参数范围
    gpu_batch_sizes = [2**i for i in range(0, 8)]  # 1, 2, 4, ..., 128
    num_gpu_batches_list = [2**i for i in range(0, 8)]

    best_throughput = 0
    best_params = None

    # 2. 搜索最优参数组合
    for gpu_batch_size in gpu_batch_sizes:
        for num_gpu_batches in num_gpu_batches_list:
            block_size = gpu_batch_size * num_gpu_batches

            # 3. 检查内存约束
            if not check_memory_constraints(config, model, gpu_batch_size, num_gpu_batches):
                continue

            # 4. 计算吞吐量
            throughput = estimate_throughput(config, model, gpu_batch_size, num_gpu_batches)

            # 5. 更新最优参数
            if throughput > best_throughput:
                best_throughput = throughput
                best_params = (gpu_batch_size, num_gpu_batches)

    return best_params

```

***隐性参数优化的意义***：
- GPU batch size影响计算效率和内存使用模式
- Num GPU batches影响I/O模式和权重复用效率
- 两者的组合对系统性能有重大影响

### 2.4 FlexGen的创新组合
FlexGen的真正创新在于将多种技术有机组合，形成一个协同工作的系统：
1. ***三级存储层次***：GPU+CPU+磁盘，突破显存限制
2. ***LP优化***：全局优化9个超参数，最小化I/O开销
3. ***4-bit量化***：减少75%内存需求，精度损失<0.5%
4. ***Zig-zag block Schedule***：通过列式遍历和权重复用减少I/O
5. ***Overlap***：隐藏传输延迟，提升设备利用率
6. ***外围参数优化***：调整GPU-batch和blocksize以最大化吞吐

***实验结果***
![这是图片](/Co-Inference_Tutorial/images/table2.png "表2")
在模型参数量和已有计算资源差距极为悬殊的情况下，FlexGen展现出来其优势。相比于DeepSpeed、Accelerate，吞吐提升100倍，首次在单16GB GPU上达到1 token/s吞吐。

## 3实现流程及对应核心代码
### 3.1 安装环境
Miniconda是轻量级的Python环境管理工具，比完整Anaconda更轻便。
**步骤1：安装miniconda**
```bash
# Linux系统
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Linux/macOS
bash Miniconda3-latest-*.sh -b

# 清理安装脚本
rm Miniconda3-latest-*.sh

# 初始化conda（仅首次安装需要）
~/miniconda3/bin/conda init bash
```
**步骤2：创建环境**

```bash
# 1. 创建虚拟环境
conda create -n flexgen python=3.10 -y

# 2. 激活环境
conda activate flexgen

# 3. 验证环境
python -c "import sys; print(f'Python {sys.version}')"
```
### 3.2安装依赖
```bash
pip install -r requirements.txt

#添加镜像
export HF_ENDPOINT=https://hf-mirror.com
```
### 3.3 核心配置详解
#### 3.3.1 文件结构
```
FLEXGEN
├── flexllmgen/          # FlexGen相关模块目录
├── images/              # 实验相关图片目录
├── flexgen_demo.py      # FlexGen协同推理脚本
├── onlygpu_demo.py      # 纯GPU推理脚本
├── README.md            # 实验指南文档
└── requirements.txt     # 依赖包清单
```
#### 3.3.2 demo配置接口


| 参数 | 归属脚本 | 作用描述 |
|------|------|------|
| --model | 通用 | Hugging Face 模型 ID（从仓库加载模型） |
| --prompt | 通用 | 输入的提示文本（模型生成的起始文本） |
| --max-new-tokens | 通用 | 生成的新 token 数量（控制输出文本长度） |
| --gpu-batch-size	 | 通用 | 单 GPU 的批处理大小（一次推理的样本数量） |
| --num-gpu-batches | FlexGen | GPU 上的总批次数（计算批次） |
| --gpu-memory | FlexGen | GPU 显存分配上限（GB，用于控制 Offloading 时的 GPU 资源） |
| --cpu-memory | FlexGen | CPU 内存分配上限（GB，用于控制 Offloading 时的 CPU 资源） |
| --percent | FlexGen | Offloading 策略（6 个百分比，依次为：权重 GPU%、权重 CPU%、缓存 GPU%、缓存 CPU%、激活值 GPU%、激活值 CPU%）***(受flexgen源代码支持问题，对于缓存和激活值同时放入gpu和cpu的混合策略并不支持)*** |
| --max-input-length | FlexGen | 输入文本的最大长度（超过则截断，确保输入格式统一） |
| --temperature | FlexGen | 生成温度（0-1，值越高输出多样性越强，越低越确定） |

### 3.4 完整运行流程
#### 3.4.1 运行第一个推理任务
```bash
##第一次使用会下载对应模型
python flexgen_demo.py
```
![这是图片](/Co-Inference_Tutorial/images/默认flexgen.png "默认运行")

#### 
3.4.2 代码
**1：flexgen_demo.py**
```python
import argparse
import torch
import time  # 用于计时
from transformers import AutoTokenizer
from flexllmgen.flex_opt import OptLM, Policy, ExecutionEnv
from flexllmgen.opt_config import get_opt_config
from flexllmgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice

def main():
    # ========== 外部参数接口定义（所有可调整参数） ==========
    parser = argparse.ArgumentParser(description="FlexGen参数化推理脚本（支持OPT模型）")
    
    # 1. 模型与硬件配置
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b",
                        help="Hugging Face模型ID（如facebook/opt-1.3b）")
    parser.add_argument("--gpu-memory", type=int, default=6,
                        help="GPU显存分配上限（GB）")
    parser.add_argument("--cpu-memory", type=int, default=16,
                        help="CPU内存分配上限（GB）")
    parser.add_argument("--percent", nargs="+", type=int, 
                        default=[10, 90, 100, 0, 100, 0],
                        help="6个数字：[权重GPU%, 权重CPU%, 缓存GPU%, 缓存CPU%, 激活值GPU%, 激活值CPU%]")
    
    # 2. 批量处理参数
    parser.add_argument("--gpu-batch-size", type=int, default=2,
                        help="单GPU的批处理大小（原hardcode的2）")
    parser.add_argument("--num-gpu-batches", type=int, default=1,
                        help="GPU上的总批次数（原hardcode的1，单卡场景通常为1）")
    
    # 3. 输入与生成配置
    parser.add_argument("--prompt", type=str, default="what is AI",
                        help="输入的提示文本（原hardcode的'what is AI'）")
    parser.add_argument("--max-input-length", type=int, default=32,
                        help="输入文本的最大长度（超过截断，原hardcode的32）")
    parser.add_argument("--max-new-tokens", type=int, default=32,
                        help="生成的新token数量（原hardcode的32）")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="生成温度（0-1，越高多样性越强，原hardcode的0.7）")
    
    # 解析参数
    args = parser.parse_args()

    # ========== 2. 初始化设备（CPU+GPU协同） ==========
    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk("./offload_dir")
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    # ========== 3. 定义Offloading策略（使用外部参数） ==========
    policy = Policy(
        gpu_batch_size=args.gpu_batch_size,          # 外部参数：单GPU批大小
        num_gpu_batches=args.num_gpu_batches,        # 外部参数：总批次数
        w_gpu_percent=args.percent[0],
        w_cpu_percent=args.percent[1],
        cache_gpu_percent=args.percent[2],
        cache_cpu_percent=args.percent[3],
        act_gpu_percent=args.percent[4],
        act_cpu_percent=args.percent[5],
        overlap=True,
        sep_layer=True,
        pin_weight=True,
        cpu_cache_compute=False,
        attn_sparsity=1.0,
        compress_weight=False,
        comp_weight_config=None,
        compress_cache=False,
        comp_cache_config=None
    )

    # 打印当前参数配置（方便实验记录）
    print("===== 当前运行参数 =====")
    print(f"模型：{args.model} | GPU显存：{args.gpu_memory}GB | CPU内存：{args.cpu_memory}GB")
    print(f"Offloading策略：权重[{args.percent[0]}%GPU, {args.percent[1]}%CPU] | "
          f"缓存[{args.percent[2]}%GPU, {args.percent[3]}%CPU] | "
          f"激活值[{args.percent[4]}%GPU, {args.percent[5]}%CPU]")
    print(f"批量配置：单GPU批大小={args.gpu_batch_size} | 总批次数={args.num_gpu_batches}")
    print(f"生成配置：输入长度={args.max_input_length} | 新token数={args.max_new_tokens} | 温度={args.temperature}")
    print(f"输入prompt：{args.prompt}")
    print("=========================\n")

    # ========== 4. 加载模型配置和分词器 ==========
    opt_config = get_opt_config(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # 修复OPT模型无pad_token的问题
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"已自动设置pad_token为：{tokenizer.pad_token}")

    # ========== 5. 初始化模型（计时） ==========
    print(f"从镜像站下载模型 {args.model}（CPU+GPU协同）...")
    start_load = time.time()  # 记录加载开始时间
    model = OptLM(
        config=opt_config,
        env=env,
        path="~/.cache/huggingface/hub",
        policy=policy
    )
    load_time = time.time() - start_load  # 模型加载时间
    print(f"模型加载完成，耗时：{load_time:.2f}秒")

    # ========== 6. 准备输入（使用外部prompt参数） ==========
    inputs = tokenizer(
        args.prompt,  # 外部参数：输入文本
        padding="max_length",
        max_length=args.max_input_length,  # 外部参数：输入长度
        truncation=True,
        return_tensors="pt"
    ).input_ids.numpy()
    # 计算总样本数（批大小×批次数）
    total_samples = args.gpu_batch_size * args.num_gpu_batches
    inputs = [inputs[0]] * total_samples  # 适配总样本数
    print(f"\n输入文本处理完成，总样本数：{total_samples}")

    # ========== 7. 推理（计时+统计吞吐） ==========
    print("开始推理...")
    start_infer = time.time()  # 记录推理开始时间
    output_ids = model.generate(
        inputs=inputs,
        max_new_tokens=args.max_new_tokens,  # 外部参数：新token数
        do_sample=True,
        temperature=args.temperature,        # 外部参数：温度
    )
    infer_time = time.time() - start_infer  # 推理总时间

    # 计算吞吐率（总生成token数 / 推理时间）
    total_generated_tokens = total_samples * args.max_new_tokens  # 总生成token数
    throughput = total_generated_tokens / infer_time  # 吞吐率（tokens/秒）

    # ========== 8. 输出结果 ==========
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print("\n===== 生成结果 =====")
    for i, out in enumerate(outputs):
        print(f"样本 {i+1}:\n{out}\n")

    # 输出时间和吞吐统计
    print("===== 性能指标 =====")
    print(f"模型加载时间：{load_time:.2f}秒")
    print(f"推理总时间：{infer_time:.2f}秒")
    print(f"总生成token数：{total_generated_tokens}")
    print(f"吞吐率：{throughput:.2f} tokens/秒")  # 每秒生成的token数
    

if __name__ == "__main__":
    main()
```
**2：onlygpu_demo.py**
```python
import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    # 与FlexGen版本保持一致的参数接口
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--prompt", type=str, default="what is AI", help="输入提示文本")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="生成的新token数量")
    parser.add_argument("--gpu-batch-size", type=int, default=2, help="单GPU批大小")
    args = parser.parse_args()

    # 检查GPU是否可用
    if not torch.cuda.is_available():
        raise RuntimeError("纯GPU推理需要CUDA环境，请确保GPU可用")
    device = torch.device("cuda:0")
    print(f"使用GPU设备：{device}")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 修复pad_token

    # 加载模型（纯GPU，全部参数放GPU）
    print("\n===== 开始加载模型（纯GPU） =====")
    start_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,  # 用半精度节省显存
        device_map="auto"  # 自动将模型加载到GPU
    ).to(device)
    model.eval()  # 推理模式
    load_time = time.time() - start_load
    print(f"模型加载完成，耗时：{load_time:.2f}秒")

    # 准备输入（批量处理）
    inputs = tokenizer(
        [args.prompt] * args.gpu_batch_size,  # 复制prompt生成批量输入
        padding="max_length",
        max_length=32,
        truncation=True,
        return_tensors="pt"
    ).to(device)  # 输入放GPU
    total_samples = args.gpu_batch_size
    print(f"\n输入文本：{args.prompt} | 总样本数：{total_samples}")

    # 推理（计时）
    print("\n===== 开始推理（纯GPU） =====")
    start_infer = time.time()
    with torch.no_grad():  # 关闭梯度计算，节省显存
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.pad_token_id
        )
    infer_time = time.time() - start_infer  # 推理耗时

    # 计算吞吐（总生成token数 / 推理时间）
    total_generated_tokens = total_samples * args.max_new_tokens  # 总生成token数
    throughput = total_generated_tokens / infer_time  # 吞吐率（tokens/秒）

    # 输出结果
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print("\n===== 生成结果 =====")
    for i, out in enumerate(outputs):
        print(f"样本 {i+1}:\n{out}\n")

    # 输出时间和吞吐统计
    print("===== 性能指标 =====")
    print(f"模型加载时间：{load_time:.2f}秒")
    print(f"推理时间（总）：{infer_time:.2f}秒")
    print(f"总生成token数：{total_generated_tokens}")
    print(f"吞吐率：{throughput:.2f} tokens/秒")

if __name__ == "__main__":
    main()
```

## 4 对比实验（示例）
验证 FlexGen 相比纯 GPU 推理的三大优势：
- **显存节省能力**：在相同模型下，GPU 显存占用显著更低；

- **超显存运行能力**：能运行纯 GPU 因显存不足无法加载的大模型；

- **资源灵活性**：通过调整 Offloading 策略，平衡显存占用与推理速度。

### 4.1 相同小规模模型下的 “显存 - 速度” 对比

**目的**：验证 FlexGen 在显存占用上的优势，以及速度的可接受损失。

**模型**：facebook/opt-1.3b（纯 GPU 可正常运行）。

#### 4.1.1 运行纯 GPU 脚本：
```bash
python onlygpu_demo.py --model facebook/opt-1.3b --max-new-tokens 64
```
![这是图片](/Co-Inference_Tutorial/images/显存速度图一.png "显存速度图一")
![这是图片](/Co-Inference_Tutorial/images/显存速度图一附属.png "显存速度图一附属")

#### 4.1.2 运行 FlexGen 脚本（低 GPU 占比策略）
```bash
python flexgen_demo.py --model facebook/opt-1.3b --max-new-tokens 64 --percent 10 90 0 100 0 100 --cpu-memory 30
```
![这是图片](/Co-Inference_Tutorial/images/显存速度图二.png "显存速度图二")
![这是图片](/Co-Inference_Tutorial/images/显存速度图二附属.png "显存速度图二附属")

#### 4.1.3 运行加大 FlexGen 的 GPU 占比
```bash
python flexgen_demo.py --model facebook/opt-1.3b --max-new-tokens 64 --percent 90 10 100 0 100 0 --cpu-memory 30
```
![这是图片](/Co-Inference_Tutorial/images//显存速度图三.png "显存速度图三")
![这是图片](/Co-Inference_Tutorial/images/显存速度图三附属.png "显存速度图三附属")

**结论**：FlexGen当GPU占比较小时推理速度略慢，但生成结果质量一致，当GPU占比上升时和纯GPU差距不大。

### 4.2 超显存模型的 “可行性” 对比

**目的**：验证 FlexGen 能运行纯 GPU 因显存不足无法加载的模型（核心优势）。

**模型**：facebook/opt-6.7b（13GB 权重，8GB GPU 纯跑 OOM）。

#### 4.2.1 尝试运行纯 GPU 脚本：
```bash
python onlygpu_demo.py --model facebook/opt-6.7b
```
![这是图片](/Co-Inference_Tutorial/images/爆显存图一.png "爆显存图一")

#### 4.2.2 运行 FlexGen 脚本（高 CPU / 磁盘 Offloading）：
```bash
python flexgen_demo.py --model facebook/opt-6.7b --max-new-tokens 64 --percent 20 80 0 100 0 100 --cpu-memory 30 --gpu-memory 8
```
![这是图片](/Co-Inference_Tutorial/images/大显存速度.png "大显存速度")
![这是图片](/Co-Inference_Tutorial/images/大显存flex.png "大显存")

**结论**：纯 GPU 直接报错，无法运行；
FlexGen 通过将大部分权重放到 CPU，成功加载并生成结果，证明其 “超显存运行能力”。

### 4.3 不同 Offloading 策略的 “灵活性” 验证

**目的**：展示 FlexGen 可通过调整--percent参数，平衡显存占用与速度（纯 GPU 无此灵活性）。

**模型**：facebook/opt-1.3b。

#### 4.3.1 极端 CPU Offloading：
```bash
python flexgen_demo.py --model facebook/opt-1.3b --max-new-tokens 64 --percent 5 95 0 100 0 100 --cpu-memory 30 --gpu-memory 8
```
![这是图片](/Co-Inference_Tutorial/images/低gpu.png "低gpu")

#### 4.3.1 平衡策略：
```bash
python flexgen_demo.py --model facebook/opt-1.3b --max-new-tokens 64 --percent 70 30 0 100 0 100 --cpu-memory 30 --gpu-memory 8
```
![这是图片](/Co-Inference_Tutorial/images/平衡.png "平衡")

#### 4.3.1 高 GPU 占比：
```bash
python flexgen_demo.py --model facebook/opt-1.3b --max-new-tokens 64 --percent 90 10 100 0 100 0 --cpu-memory 30 --gpu-memory 8
```
![这是图片](/Co-Inference_Tutorial/images/极限.png "极限")


**结论**：随着 GPU 占比升高，GPU 显存占用递增，推理时间递减（灵活性体现）

## 5 常遇问题
若首次安装conda可能需手动接受条款
![这是图片](/Co-Inference_Tutorial/images/接受条款.png "接受条款")
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

若无法找到某个依赖如
![这是图片](/Co-Inference_Tutorial/images/pip版本低.png "pip版本低")
```bash
pip install --upgrade pip
```

若运行onlygpu_demo发现找不到对应模块，可能是模块冲突
```bash
pip install \
  huggingface-hub==0.20.3 \
  accelerate==0.24.1 \
  transformers==4.35.2 \
  --force-reinstall
```
服务器退出后也需要重新添加镜像
```bash
#添加镜像
export HF_ENDPOINT=https://hf-mirror.com
```
## 6 总结
FlexGen通过以下核心技术实现了单GPU上的高吞吐大语言模型推理：
1. **三级存储协同**：GPU+CPU+磁盘的分层内存架构
2. **LP优化张量调度**：通过求解线性规划问题找到最优张量放置策略
3. **Zig-zag块调度**：列式遍历计算图，实现权重复用
4. **4-bit压缩**：大幅降低内存需求，精度损失<0.5%
5. **计算-传输重叠**：隐藏PCIe传输延迟

**性能提升**：
- 与现有系统相比，在大模型上吞吐量提升100倍
- 首次在单16GB GPU上实现1 token/s的OPT-175B推理
- 使消费级硬件能够运行百亿级参数模型


**参考材料**
- https://github.com/FMInference/FlexLLMGen
- https://arxiv.org/abs/2303.06865



## 7 多设备协同延申（有兴趣同学可以尝试）
上述FlexGen本质上事单设备上参数的卸载协同，真实边缘场景下还可以通过设备间协同来实现高效推理。例如利用PP模式，在多设备间流水执行，参考资料如下：
![这是图片](/Co-Inference_Tutorial/images/prima.png "Prima.cpp")
- https://gitee.com/zonghang-li/prima.cpp
- https://arxiv.org/abs/2504.08791