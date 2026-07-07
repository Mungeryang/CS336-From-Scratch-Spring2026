# <p align="center"> 🧐50 $QA_{s}$ ON LLM - 大模型面试50问🧐 </p>

#### <p align="center"> 杨桂淼整理 2026年4月 </p>

<div align="center">
  <img src="https://www.ydylcn.com/skwx_ydyl//zpimage.zhtml?ID=10600582&SiteID=1&draft=0&type=norm" width="70%">
</div>

> ### 通过整理 `Standford CS336 Spring26` 课堂笔记，总结大模型算法经典面试**50**问

> 参考资料: CS336课堂笔记、[李博杰老师 - 大模型面试题 200 问](https://www.ituring.com.cn/book/3285)、[百面大模型](https://www.ptpress.com.cn/publishing/book/ef639cdb-d2a9-4987-8c79-14760baa4fc3)、[大模型技术30讲](https://github.com/ningg/Machine-Learning-Q-and-AI)

> 以点带面，忌贪多贪全，从单个问题出发逐步深入。 不是对八股的死记硬背，而是在实操中去总结问题。

> ### ⚠️ 持续更新中...

## Topic 1: Tokenizer and PreToken

### Byte 级的BPE分词器相比于传统的BPE分词器有哪些优势?

byte-level BPE 的核心优势可以概括为**兼顾了开放词表能力和较好的压缩能力**。

<img src="https://github.com/stanford-cs336/lectures/blob/main/images/tokenized-example.png">

- Byte-level BPE 的初始词表是全部 256 个字节值, 任意 Unicode 文本都可以先编码成 UTF-8 字节序列，因此理论上任何输入都能被表示，不会出现传统词表覆盖不到的字符或词。

- 即使是很少见的字符，也总能拆成若干字节来表示。因此它对跨语言文本、emoji、罕见符号等输入更稳定。

- 如果只按字节切分，序列会很长，训练和推理成本高。BPE 会把高频字节序列继续合并成子词，从而压缩序列长度，减少计算开销。

### BPE训练过层中,预分词(Pre-tokenization)的作用是什么？

- 防止跨界合并，避免将本应该属于不用语义单元的片段合并成一个独立的ID，提高语义一致性。

- 通过正则表达式先将文本切分成单词或者短语块，可以在统计字节对频率时减少全量扫描语料的次数。


## Topic 2: Fundamentals

### 字节与比特

`bytes()` 返回一个不可变的字节对象。  

计算机内部其实并不认识“你”或“好”这些汉字，它只认识 0 和 1。`bytes()` 方法就像是一个翻译官，它根据你提供的编码表（如 utf-8），把人类可读的文本转换为机器存储的字节序列。  

在 utf-8 编码中，一个常用的汉字通常占用 3 个字节。当你执行 bytes("你好", encoding='utf-8') 时，得到的返回结果是： b'\xe4\xbd\xa0\xe5\xa5\xbd'


### 残差的作用是什么？

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR3osQSRVUQ8k0JqVMNFi_w4xKrk1CUn5PfbQ&s">

残差连接 = y = x + F(x)

如果没有残差连接，一个神经网络在反向传播的过程中：

∂Loss/∂x = ∂Loss/∂y · ∂f₃/∂a₂ · ∂f₂/∂a₁ · ∂f₁/∂x

每一项 ∂fᵢ/∂aᵢ₋₁ 都是该层变换的雅可比矩阵。如果每层的雅可比矩阵的谱范数（最大奇异值）小于 1，连乘 100 层后梯度指数级衰减到 0。

这是链式法则的致命问题：梯度的每一个"链接"都是一个雅可比矩阵，而神经网络中大多数非线性激活函数的导数都在 [0, 1] 范围内（sigmoid 最大 0.25，tanh 最大 1）。乘多了必然消失。

作用：

- 链式法则连乘再多也有精确的 1 保底，梯度不会消失

- 原始输入走恒等路径直达深层，每层只学"增量"（残差），而不是完全重造表征

- Loss Landscape 更平坦、更凸，优化器更容易找到好的局部最小值


### LLM架构中常用的损失函数和激活函数

#### 激活函数 

1. ReLU 系列（早期标准）

ReLU(x) = max(0, x)

使用：原始 Transformer（2017）、ResNet

特点：简单、稀疏激活，问题是负半轴完全杀死梯度。

2. GELU —— GPT-3/BERT 的选择

GELU(x) = x · Φ(x)  其中 Φ 是标准正态分布的 CDF

3. SwiGLU —— 现代 LLM 的标配（LLaMA/Qwen/Mistral）

SwiGLU(x, W, V, W₂) = (Swish(xW) ⊙ xV) · W₂

Swish(x) = x · σ(x) = x · sigmoid(x)

使用：LLaMA 1/2/3、PaLM、Mistral、Qwen、DeepSeek

特点：带门控的激活函数。它其实是两个线性投影做 element-wise 乘法后再投影回去。参数量是标准 FFN 的 1.5 倍（因为有 3 个 W），但效果更好

#### 损失函数

1. 自回归语言建模损失(Causal LM Loss)

L = -Σ log p(tᵢ | t<ᵢ)

使用：所有 LLM（GPT、LLaMA、Mistral、Qwen）

本质：标准 cross-entropy，但 attention 掩码是因果的——每个 token 只能看到自己和前面的 token；这是 LLM 最核心的损失函数，所有预训练都是用这个。

2. 对比损失(InfoNCE)

$$ \mathcal{L}_i^{\text{image}} = -\log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)} $$

使用：CLIP、SigLIP、几乎所有双塔视觉-语言模型

本质：N 个正样本对 + N²-N 个负样本对的 softmax 分类

3. DPO Loss

$$ \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right] $$

使用：LLaMA 2/3、Mistral、Qwen 的偏好对齐阶段

本质：在 RLHF 之后提出的简化方案，直接用偏好对优化策略模型，不需要训练 reward model

4. KL散度

$$ \mathcal{L}_{\text{KL}} = D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) = \mathbb{E}_{x \sim \pi_\theta} \left[ \log \frac{\pi_\theta(x)}{\pi_{\text{ref}}(x)} \right] $$

使用：RLHF 的约束项、DPO 的隐式约束、知识蒸馏

本质：防止模型在优化过程中偏离原始预训练分布太远，避免"reward hacking"

### 模型参数量是怎么估算的？

### 注意力机制通俗理解



### 怎么理解CLIP模型训练过程中的对齐？与LLaVA中的阶段1的对齐机制一样吗？

CLIP对齐的本质是对比学习：让匹配的图文对在嵌入空间中靠近，不匹配的图文对距离拉远。	

图像和文本最后都要经过一层投影层到同一个维度，之后再进行相似度计算和比较。

LLaVA中 Stage1 的对齐机制属于微调一个简单的线性层，使得视觉特征的输出能够对齐语言模型的输入。直白来讲，视觉编码器的输出维度与语言解码器的输入维度并不是同一个，需要进行线性映射使得视觉特征维度转换为文本维度。


## Topic 3: Model Architecture


### 简述大模型发展历程

- 前神经网络时代：N-gram模型，用于测量英语熵值

- 神经组件期 (2010s)：神经语言模型、Seq2Seq、Attention、Transformer、Adam优化器、MoE、模型并行

- 早期基座模型：ELMo, BERT, T5

- 规模化与闭源：GPT-2/3, PaLM, Chinchilla (Compute-optimal scaling)

开源浪潮：The Pile, GPT-J, OPT, BLOOM, Llama系列, Qwen, DeepSeek, OLMo


### Qwen架构中的使用的层归一化函数是什么？其和 *LayerNorm* 的区别体现在哪些方面?

Qwen架构中使用的层归一化函数为`RMSNorm`, 相比原始 Transformer 常用的 `LayerNorm`，现代大模型更常采用 `RMSNorm`。

- LayerNorm 会先减去均值，再除以标准差：

$$ \mathrm{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta $$

- RMSNorm 不做减均值，只按均方根进行缩放：

$$ \mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon} $$

> LayerNorm 是“减均值再缩放”，而 RMSNorm 是“只按均方根缩放、不减均值”；后者更简单、更省计算(计算开销更小)。


### 使用Transformer生成文本时,在解码过程中使用的Trick有哪些?

#### Trick 1 

模型最后输出的是 `logits`，需要先经过 **softmax** 转成词表上的概率分布，才能进行采样。在 softmax 前把 logits 除以温度参数 $ \tau $ 进行**温度缩放** :

$$ \mathrm{softmax}(v, \tau)_i = \frac{\exp(v_i / \tau)}{\sum_{j=1}^{|\mathrm{vocab}|} \exp(v_j / \tau)} $$


#### Trick 2

不直接在整个词表上采样，而是先把所有 token 按概率从大到小排序，从概率最大的 token 开始往后累加，一直加到累计概率第一次达到或超过阈值 p。

然后只在这组 token 里重新归一化并采样，截断低概率噪声 token，提升生成文本质量。


### Pre-norm 与 Post-norm 的区别，为什么现代 LLM 偏好Pre-norm 架构？

<img src="https://i-blog.csdnimg.cn/blog_migrate/d4d8a8327721f8368e1bce5f0a1b2096.png">

- Post-norm: y = LayerNorm(x + Attention(x))

- Pre-norm: y = x + Attention(LayerNorm(x))

Transformer Block 包括两个主要的sub-layers: multi-head self-attn mechanism(MHA) 和 position-wise feed-forward netword(FFN)。

结构区别：Post-norm 在残差连接相加后进行 LayerNorm；Pre-norm 则是在进入自注意力或前馈网络之前进行 LayerNorm 。(区别如上图所示，重点抓住两个核心层即可)

Pre-norm 创造了一条从输入嵌入到最终输出的“清晰残差流”，实验证明它能显著提高大模型训练的稳定性，改善梯度流动 。

### 多模态大模型架构中，视觉编码器与语言解码器之间通过什么桥接起来？

> 视觉编码器与语言解码器之间通过线性投影层连接起来。

以LLaVA为例，ViT 输出 1024-dim 视觉特征，已经能识别物体了。Vicuna 语言模型输入 5120-dim 词嵌入，已经懂语言了。

线性投影 Linear(1024, 5120) 开始是随机初始化的，需要学会把视觉语义"翻译"成 LLM 能理解的 token。


### LLaVA训练的范式？其开启的重要研究路线是什么？

LLaVA主要采用两阶段训练策略：*Stage1: 对齐预训练 +  Stage2: 指令微调*

- Stage1 - 对齐预训练阶段，目的是让投影层学会把 CLIP 特征映射到 LLM 的词嵌入空间

视觉编码器 ❌冻结   投影层 ✅训练    Vicuna ❌冻结

- Stage2 - 指令微调阶段，目的是让投影层与语言模型联合微调，适配指令跟随任务

视觉编码器 ❌冻结   投影层 ✅训练    Vicuna ✅训练


当下很多开源多模态大模型的训练范式都是沿用了LLaVA的基础：

```shell
视觉编码器（CLIP/SigLIP,通常冻结）
         ↓
投影层（Projector,可训练）
         ↓
大语言模型（LLM,SFT 阶段解开）
         ↓
训练数据：LLM 生成的指令数据 + 少量人工标注

```

> Qwen3-VL、InternVL、DeepSeek-VL 等模型虽然架构上复杂，但是**数据生成范式**和**两阶段训练策略**(对齐预训练 + 指令微调)全部源自LLaVA。

## transformers 与 peft 梳理

### transformers 中不同模型加载类的区别是什么？

> AutoModel 加载“裸模型backbone”, AutoModelForxxx 加载“backbone + 某个任务头”。

```python
model = AutoModel.from_pretrained("bert-base-cased") # 只要隐藏状态，自己接head

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B") # decoder-only 大模型微调：LLaMA/Qwen/GPT 类


model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base") # encoder-decoder 文本生成：T5/BART/MT5

```

- `AutoModel` 不会自动算任务 loss，它不知道具体的任务是什么。

- `AutoModelForCausalLM` 的 labels 通常是目标token序列，用下一个token预测训练。

- `AutoModelForSeq2SeqLM` 的 labels 是 decoder 目标序列，比如摘要或翻译结果。

做大模型 SFT/LoRA 微调，绝大多数 decoder-only 模型用 AutoModelForCausalLM；T5/BART 这类 encoder-decoder 才用 AutoModelForSeq2SeqLM。

> 加载 视觉语言模型 与加载 语言模型 的区别？

视觉语言模型的输入不止有文本token，还有图片/视频/音频经过预处理后的视觉tokens，以及图像token和文本token对齐的信息。transformers 库中AutoTokenizer 只负责文本，AutoProcessor 负责“文本 + 图片 + 视频”的整体打包。

以Qwen3-VL为例，官方Qwen3VLProcessor本质包装了：

- tokenizer: 处理文本

- image_processor: 处理图片resize、normalize、patch

- video_processor: 处理视频帧、时序信息

- chat_template: 多模态message转化为输入格式

### from_pretrained()方法中的关键参数有哪些？

`AutoModel.from_pretrained()` 的核心作用是：根据模型目录或 Hub 上的 **config.json** 自动判断架构，然后加载对应模型权重。

| 参数                          | 作用                          | 实战中怎么用                                                 |
| :---------------------------- | :---------------------------- | :----------------------------------------------------------- |
| pretrained_model_name_or_path | 模型来源                      | Hub 模型名，如 "bert-base-chinese"；或**本地路径** |
| dtype / torch_dtype           | 加载权重的数据类型            | 大模型常用 torch.float16、torch.bfloat16 或 "auto"           |
| device_map                    | 自动/手动分配设备             | 大模型常用 device_map="auto"                                 |
| attn_implementation           | 注意力实现                    | 可选 "eager"、"sdpa"、"flash_attention_2"                 |
| config                        | 指定模型配置                  | 想手动改结构、标签数、输出行为时用                           |
| local_files_only              | 只读本地缓存，不联网          | 离线环境常用                                                 |
| trust_remote_code             | 是否执行模型仓库自定义代码    | Qwen、ChatGLM、部分多模态模型可能需要；只对可信仓库开启      |


- from_pretrained() 默认会把模型设成 eval() 模式；训练时需要进入 train()，不过 Trainer 通常会处理。

- 大模型加载最常用组合是：dtype/torch_dtype + device_map + quantization_config + trust_remote_code。

### peft 微调流程

> 基本流程: transformers 加载模型backbone -> 创建LoRAConfig -> 调用get_peft_model() 包装成 PeftModel() 训练
















