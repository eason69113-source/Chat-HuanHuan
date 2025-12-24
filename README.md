<!-- 徽章区 -->
<p align="center">
  <!-- 版本 -->
  <img alt="GitHub release (latest by date)" src="https://img.shields.io/github/v/release/YOUR_USER/HuanHuan-Chat?style=flat-square">
  <!-- License -->
  <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square">
  <!-- 显存卖点 -->
  <img alt="VRAM" src="https://img.shields.io/badge/VRAM-%3C9GB-00b894?style=flat-square&logo=nvidia">
  <!-- 模型 -->
  <img alt="Model" src="https://img.shields.io/badge/Model-Llama--3.1--8B-ff6b6b?style=flat-square&logo=pytorch">
  <!-- 框架 -->
  <img alt="Framework" src="https://img.shields.io/badge/Framework-QLoRA-4a90e2?style=flat-square&logo=huggingface">
  <!-- Python -->
  <img alt="Python" src="https://img.shields.io/badge/Python-3.12-3776ab?style=flat-square&logo=python">
  <!-- Hugging Face -->
  <a href="https://huggingface.co/YOUR_HF_USER/HuanHuan-Chat"><img alt="HF" src="https://img.shields.io/badge/%F0%9F%A4%97-Model%20Hub-yellow?style=flat-square"></a>
</p>

# 💃 Chat-HuanHuan · 嬛嬛问答  

- **低精度模型微调项目，非常适合新手来了解模型微调的原理及如何降低显存消耗，并且代码非常简洁，对于刚入门的新手很友好**
- 扮演皇帝身边的女人——甄嬛，陪你畅聊宫廷趣事。
- 一张 **≤ 9 GB** 显存的消费卡就能完成 QLoRA 微调的中文宫廷角色扮演模型！  
- 基于 Meta-Llama-3.1-8B-Instruct + 4-bit 量化 + QLoRA，训练与推理全程显存占用 **< 9 GB**，RTX 3060/4060 即可玩转。

---

⚔️ 显存实测对比：

| 方案 | 基座模型 | 微调方式 | 量化位深 | 训练显存 (实测) | 推理显存 (实测) | 硬件门槛 |
|------|----------|----------|----------|-----------------|-----------------|----------|
| 官方全量微调 | Llama-3-8B | Full Fine-tune | 16-bit | ≈ 160 GB | ≈ 16 GB | 2× A100 (80G) |
| 标准 LoRA | Llama-3-8B | LoRA r=8 | 16-bit | ≈ 26–28 GB | ≈ 16 GB | RTX 3090/4090 |
| **该项目** | Llama-3-8B | LoRA r=8 | **4-bit NF4** | **≈ 7.5–8.5 GB** ⭐ | **≈ 5.5–6.5 GB** ⭐ | **RTX 4070Ti 12G ✅** |

---

## 🌟 效果预览
| 皇上 | 嬛嬛 |
|------|------|
| 皇上：今天心情不太好。 | 嬛嬛：皇上忧国忧民，臣妾愿为皇上抚琴一曲，以解烦忧。 |
| 皇上：给朕讲个笑话。 | 嬛嬛：从前有座山，山里有座庙，庙里有个小太监在偷吃桂花糕……皇上可还满意？ |

---

## 🚀 快速开始

### 0. 创建 conda 虚拟环境（Python 3.12）
```bash
conda create -n huanhuan python=3.12 -y
conda activate huanhuan
```

### 1. 安装 PyTorch 2.4.0 + CUDA 12.1（阿里源加速）
```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    -f https://mirrors.aliyun.com/pytorch-wheels/cu121/
```

### 2. 克隆仓库
```bash
git clone https://github.com/eason69113-source/Chat_HuanHuan.git
cd Chat-HuanHuan
```

### 3. 下载基座模型
```bash
python get_model.py   # 自动下载 Llama-3.1-8B-Instruct 到 ./model
```

### 4. 下载 LoRA 权重（可选，已训练好）
```bash
# 已随仓库附带 ./output/llama3_1_instruct_lora/checkpoint-700
# 如想自己训练，见「训练」章节
```

### 5. 安装依赖
```bash
pip install -r requirements.txt
```

### 6. 运行交互式 demo
```bash
python inference.py
```
输入 `quit` 退出。

---

## 🏋️ 训练（从头复现）
1. 准备数据  
   把对话 JSON 放入 `data/huanhuan.json`，格式示例：
   ```json
   [
     {"instruction": "皇上：今天天气如何？", "input": "", "output": "嬛嬛：回皇上，今日风和日丽，宜游园赏花。"},
     ...
   ]
   ```
2. 启动训练  
   ```bash
   python train.py
   ```
   默认 3 epoch，单卡 RTX 4070Ti 约 60 分钟。  
   训练完自动保存到 `./output/llama3_1_instruct_lora/checkpoint-xxx`

---

## 📂 项目结构
```
HuanHuan-Chat
├── main
│   └── data/                     # 训练数据
│   └── get_model.py/             # 下载基座模型
│   └── train.py/                 # LoRA 微调脚本
│   └── inference.py/             # 交互式推理
├── output
│   └── llama3_1_instruct_lora
│       └── checkpoint-700        # 已训练权重
├── requirements.txt              # 环境配置
└── README.md
```

---

## ⚙️ 硬件要求
| 阶段 | 显存 | 备注 |
|------|------|------|
| 推理 | ≈ 6 GB | 4-bit 量化，单卡 3060 可跑 |
| 训练 | ≈ 9 GB | RTX 4070Ti 12G |

---

## 🛠️ 技术栈
- **基座模型**: Meta-Llama-3.1-8B-Instruct  
- **微调方案**: LoRA (r=8, α=32, dropout=0.1)  
- **量化**: bitsandbytes 4-bit NF4  
- **框架**: transformers + peft + modelscope  
- **对话模板**: Llama-3 官方 chat_template

---

## 📄 开源协议
Apache-2.0  
⚠️ 模型权重请遵守 Meta 官方[社区许可](https://llama.meta.com/llama3/license/)。

---

## 🙏 致谢
- Meta Llama-3 团队  
- ModelScope 社区  
```
