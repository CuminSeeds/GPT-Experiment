# 人工智能基础作业三

本项目搭建一个 628,226 参数的字符级 GPT 类模型，从随机初始化完成预训练，再对英文短信分类进行监督微调。程序没有加载外部预训练权重。

## 实际结果

预训练执行 1,800 次更新，验证损失最低的权重位于第 1,800 次更新。测试字符预测损失为 1.8156，准确率为 46.34%。

监督微调执行 20 轮，选择验证损失最低的第 10 轮。短信测试集 517 条，准确率为 98.45%，垃圾短信查准率 95.24%，查全率 92.31%，F1 为 0.9375。选定模型的训练准确率为 99.44%，验证准确率为 98.45%。这是一次固定划分和种子的结果。

重新加载权重后，CPU 上的 517 条测试预测与训练结束时一致。6 条固定样例通过本地 HTTP 接口测试，其中包含一条实际误判。完整结果保存在 results 中。

## 文件说明

- model.py：字符编号、位置编码、因果多头注意力、前馈网络、模型保存与生成。
- prepare_data.py：下载与清洗数据，固定训练、验证、测试划分。
- train.py：预训练和监督微调，保存全部曲线数据及最终指标。
- plot_results.py：使用 Matplotlib 默认风格绘图。
- infer.py：加载权重，在 CPU 上进行文本续写或短信分类。
- serve.py：本机网页和 HTTP 分类接口。
- verify_inference.py：检查 CPU 加载结果并实际请求本地 HTTP 服务。
- data：原始公开数据、固定划分及数据统计。
- weights：随机初始化、预训练、微调后的模型权重。
- results：训练记录、最终指标、全部测试预测、固定样例和部署验证记录。
- figures：预训练和微调曲线。

## 环境与运行

实际训练使用 Windows、Python 3.11.15、PyTorch 2.7.1+cu118、NumPy 2.2.6，显卡为 GTX 1050 Ti 4GB。绘图使用独立的文档运行环境，Matplotlib 3.11.2。模型推理不要求 GPU；完整训练在 CPU 上也可执行，但速度较慢。

在本目录打开终端。普通 Python 环境安装 torch、numpy；需要画图再安装 matplotlib。CUDA 安装方式以 PyTorch 官方安装说明为准。不要把 CUDA 版 torch 的版本后缀直接当成通用 pip 安装命令。

```powershell
python -m pip install torch==2.7.1 numpy==2.2.6 matplotlib
```

### 保存模型与 CPU 推理

程序分别保存随机初始化、预训练和监督微调后的权重，位于 `weights/initial.pt`、`weights/pretrained.pt` 和 `weights/finetuned.pt`。推理时重新加载权重，保持训练时的短信清洗、截断和提示格式，并切换到评估模式以关闭 Dropout。

压缩包已经带有数据与权重，无需重新训练即可在 CPU 上运行。第一条命令判断短信类别，第二条命令使用预训练模型续写：

```powershell
python infer.py --mode spam --text "Are we meeting tonight?"
python infer.py --mode generate --prompt "The king" --length 240
```

分类默认加载 `weights/finetuned.pt`，续写默认加载 `weights/pretrained.pt`；也可用 `--checkpoint` 指定其他权重。重新加载微调模型后，已在 CPU 上核对全部 517 条测试短信，预测与训练结束时的记录全部一致，记录见 `results/deployment_check.json`。

### 本地服务

使用 Python 自带的 HTTP 服务封装分类函数，在项目目录执行：

```powershell
python serve.py
```

服务默认加载 `weights/finetuned.pt`，在本机 CPU 上运行。浏览器打开 http://127.0.0.1:8765 ，输入英文短信后即可查看分类。`GET /health` 返回服务状态，`POST /predict` 接收包含 `text` 字段的 JSON 并返回分类结果。可用 `--port` 更改端口，按 Ctrl+C 停止服务。

已实际启动服务并核对 6 条固定测试样例，HTTP 返回的类别与直接加载模型时一致，其中包含一条实际误判。验证时使用临时端口 18765，验证结束后已关闭；响应记录保存在 `results/deployment_check.json`。

### 从头复现实验


```powershell
python prepare_data.py
python train.py --stage all
python plot_results.py
python verify_inference.py
```

也可以分别运行 `python train.py --stage pretrain` 和 `python train.py --stage finetune`。重新训练会覆盖对应权重和结果，建议先复制本项目目录留存当前结果。不同设备或软件版本可能导致浮点计算和训练结果略有差别；CPU 与 GPU 的随机采样序列也可能不同，因此相同提示与种子在不同设备上续写的文字不一定完全相同。报告使用的实际生成文本已完整保存在 results/generation_samples.json。

## 实验细节

模型配置：3 个块，128 维，4 个注意力头，前馈层 512 维，ReLU，层后 LayerNorm，正弦余弦位置编码，上下文 192，Dropout 0.1，词表 130。两个阶段均使用 Adam，batch 32，权重衰减 0。预训练学习率 0.001，微调学习率 0.0003。

预训练数据按原始字符顺序分为 90%、5%、5%。训练每次随机抽取 32 个 192 字符片段。曲线训练指标固定在前 49,152 个训练字符上计算；验证和测试评估全部字符，按最多 192 字符分段。预训练随机种子为 42。

短信先做 Unicode NFKD、转 ASCII 并合并空白，再按模型实际可见的前 176 字符去重。原始 5,574 条，保留 5,158 条。分层划分后训练 4,125 条、验证 516 条、测试 517 条，集合之间没有相同提示。划分种子为 2026，微调种子为 43。短信大小写保留，较长短信截断。相似模板仍可能存在，本实验只保证实际输入不完全相同。

微调输入为 `Message: 短信正文\nLabel:`，目标为字符 0 或 1，损失只计算答案位置。训练仍采用完整的 130 字符输出层。分类比较字符 0、1 的分数，同时记录完整词表直接取最大值时的输出；微调后的测试输出全部为合法类别字符。

两个阶段均按验证损失选择权重，然后评估测试集，未用测试结果调整学习率或轮次。微调固定训练 20 轮，并不保证最后一轮最好。曲线从第 1 轮开始展示以看清后期变化；第 0 轮大损失保存在 finetune_history.json 中。

## 来源与使用说明

1. 周志强课程讲义《基于Transformer的大语言模型技术（一）》与《神经网络与深度学习》：理论、位置编码、注意力、Adam 等。
2. https://github.com/rasbt/LLMs-from-scratch ：学习预训练和微调的整体流程。
3. https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py ：参考字符级模型及因果注意力程序组织。本项目按课件采用正弦余弦位置编码、ReLU 和层后归一化，自行编写数据、训练、微调与推理程序。
4. https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt ：Tiny Shakespeare 原始文本。数据 SHA-256 为 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed。
5. Almeida T and Hidalgo J. SMS Spam Collection. UCI Machine Learning Repository, 2011. https://doi.org/10.24432/C5CC84 ，CC BY 4.0。下载页 https://archive.ics.uci.edu/dataset/228/sms+spam+collection ，原始压缩包 SHA-256 为 1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3。本项目对短信进行了清洗、去重、截断与划分，原始说明保存在 data/sms_readme.txt。

数据用于本课程实验，英文短信分类结果不代表通用语言理解或中文分类能力。

## 位置编码与注意力头数改进实验

原来的正弦余弦 4 头模型、默认推理路径和全部实验记录继续保留。新增实验仅比较 RoPE 和 2、4、8 个注意力头；没有改动其他网络结构或训练超参数。

对照一：正弦余弦 4 头与 RoPE 4 头。对照二：固定 RoPE，比较 2、4、8 头。隐藏维度均为 128，参数均为 628,226；三组新增模型都从头预训练 1,800 步，再微调 20 轮，使用原数据划分及随机种子。原基准直接复用前文完成的实验。

| 配置 | 预训练测试损失 | 字符准确率 | 短信准确率 | 垃圾短信 F1 | 微调验证损失 |
| --- | --- | --- | --- | --- | --- |
| sinusoidal_h4 | 1.8156 | 46.34% | 98.45% | 0.9375 | 0.0580 |
| rope_h4 | 1.7057 | 50.29% | 98.84% | 0.9531 | 0.0448 |
| rope_h2 | 1.7038 | 50.00% | 99.03% | 0.9606 | 0.0605 |
| rope_h8 | 1.7039 | 50.29% | 98.45% | 0.9403 | 0.0415 |

以短信验证损失选择的最终配置是 `rope_h8`。所有配置均在各自验证集选定权重后评估测试集，测试指标不用于选择配置。本次只有一组随机种子，没有测试超过 192 字符的上下文，因此不宣称统计显著性或长文本外推能力。

新增文件：

- run_improvements.py：顺序训练三组改进配置，复用原 train.py 中的数据与训练流程。
- plot_improvements.py：按 Matplotlib 默认风格绘制四张对比图，虚线表示训练集，实线表示验证集。
- verify_improvements.py：检查数据哈希、验证集选择和 CPU 权重重载预测。
- improvements/settings.json、comparison.json：统一设置、数据哈希和全部汇总指标。
- improvements/rope_h2、rope_h4、rope_h8：各自的 results 与 weights，包含完整训练历史、指标、生成样例和 517 条测试预测。
- improvements/model_checks.json、verification.json：RoPE 数学性质与因果掩码检查，以及 CPU 全测试集重载核对。

复现实验时，在本目录执行：

```powershell
python run_improvements.py --variant all
python plot_improvements.py
python verify_improvements.py
```

已有完整指标的阶段会自动跳过。如需完整重训，先复制留存整个项目，再删除待重训配置对应的 improvements 子目录；基础模型不受影响。也可用 `--variant rope_h4`、`rope_h2` 或 `rope_h8` 单独运行某一组。

加载改进权重可使用 `load_model("improvements/rope_h4/weights/finetuned.pt", "cpu")`，其他头数替换对应目录名。原 infer.py 和网页服务继续使用 weights/finetuned.pt 中的基础模型。

## 分词与词嵌入对比实验

新增两组模型固定为 RoPE、4 头、128 维、3 个网络块、192 词元上下文，预训练 1,800 次更新、微调 20 轮。对照使用已经完成的字符级 RoPE 4 头模型。预训练学习率为 0.001，微调学习率为 0.0003，batch 为 32，随机种子和数据划分沿用原实验。

混合分词在 130 个基础字符编号之外，加入仅从 `lm_train.txt` 统计出的 382 个高频完整单词，词表大小为 512。词表外单词逐字符编码，保留原文、空格和标点，不引入新的未知词。预训练训练文本从 1,003,854 个字符变为 721,635 个词元，减少 28.11%。短信仍先取同样的前 176 个字符，再分词。两组新模型均为 726,400 个参数，原字符模型为 628,226 个参数。

两组新模型分别使用随机初始化和 Word2Vec CBOW 初始化嵌入矩阵。CBOW 根据前后各 2 个词元的平均向量预测中心词元，仅使用预训练训练集，128 维向量、5 轮、Adam 学习率 0.001、batch 512、随机种子 44。这里使用小词表的完整 Softmax。CBOW 向量统一缩放至标准差 0.02，再复制到 GPT 嵌入矩阵；后续预训练和微调继续更新嵌入矩阵。两组模型的其他初始权重完全相同。

| 配置 | 选中轮次 | 验证损失 | 测试准确率 | 垃圾查准率 | 垃圾查全率 | 垃圾 F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 字符随机初始化 | 13 | 0.0448 | 98.84% | 96.83% | 93.85% | 0.9531 |
| 混合分词随机初始化 | 6 | 0.0527 | 99.23% | 95.52% | 98.46% | 0.9697 |
| 混合分词 CBOW 初始化 | 5 | 0.0505 | 99.61% | 97.01% | 100.00% | 0.9848 |

各组按验证损失选定权重后，才评估同一测试集。混合分词 CBOW 组在 517 条短信上错判 2 条普通短信、漏判 0 条垃圾短信。两个新模型的 CPU 重载预测均与训练保存的 517 条预测完全一致。原始数据和既有模型权重保持不变。

新增文件与运行方式：

- `token_embedding_experiment.py`：分词器、CBOW 训练、两组 GPT 预训练与微调。
- `infer_token_embedding.py`：使用匹配分词器加载新权重，进行 CPU 分类或续写。
- `verify_token_embedding.py`：检查训练集词表、全文分词可逆、数据哈希、验证集选择、初始化权重对照和 CPU 预测。
- `plot_token_embedding.py`：预训练和短信微调对比曲线。
- `token_embedding/`：词表、设置、CBOW 向量、两组训练记录、权重、测试预测与核对结果。

```powershell
python token_embedding_experiment.py
python verify_token_embedding.py
python plot_token_embedding.py
python infer_token_embedding.py --variant mixed_cbow --text "Are we meeting tonight?"
python infer_token_embedding.py --variant mixed_cbow --mode generate --prompt "The king" --length 120
```

已完成的阶段自动跳过；重训前应先复制留存实验目录。新模型必须搭配 `token_embedding/tokenizer.json`，不能沿用基础模型的字符编码函数。基础模型与网页服务的默认权重路径保持不变。预训练的损失和准确率按各自词元计算，字符级与混合分词的这些指标不直接横向比较；两组混合分词使用相同词表，可比较预训练曲线。

CBOW 参考：Mikolov 等，Efficient Estimation of Word Representations in Vector Space (2013)，https://arxiv.org/abs/1301.3781 。

## 报告字体

Word 报告与全部图形中的文字统一使用宋体、黑色。图形保留 Matplotlib 默认配色和线条样式，只调整文字字体与字色。plot_font.py 提供统一设置，Windows 下使用系统自带的 C:/Windows/Fonts/simsun.ttc。字体调整不改变数据、模型权重和实验指标。
