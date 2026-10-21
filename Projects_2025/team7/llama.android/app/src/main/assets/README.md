# Vosk 模型目录

请将下载的 Vosk 模型文件夹放在此目录下。

## 下载模型

### 中文模型（推荐）
小型模型（42 MB）：https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip

### 英文模型
小型模型（40 MB）：https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip

## 安装步骤

1. 下载上述模型之一
2. 解压缩下载的文件
3. 将解压后的文件夹重命名为 `model-cn`（中文）或 `model-en`（英文）
4. 将该文件夹复制到此目录（assets）下

最终目录结构应该是：
```
assets/
└── model-cn/          # 或 model-en
    ├── am/
    ├── conf/
    ├── graph/
    └── ivector/
```

更多详细信息，请参考项目根目录下的 VOSK_SETUP.md 文件。
