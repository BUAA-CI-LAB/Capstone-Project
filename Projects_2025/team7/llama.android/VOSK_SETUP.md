# Vosk 离线语音识别设置指南

本应用已修改为使用 Vosk 进行离线语音识别，不再依赖系统的语音识别服务。

## 准备工作

### 1. 下载 Vosk 模型

您需要下载一个 Vosk 语音识别模型。根据您的需求选择：

#### 中文模型（推荐）
- **小型模型**（约 42 MB）：
  - 下载地址：https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip
  - 适合资源受限的设备
  
- **大型模型**（约 1.8 GB）：
  - 下载地址：https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip
  - 识别精度更高

#### 英文模型
- **小型模型**（约 40 MB）：
  - 下载地址：https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
  
- **大型模型**（约 1.8 GB）：
  - 下载地址：https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip

### 2. 安装模型到项目

1. 下载并解压您选择的模型文件
2. 将解压后的模型文件夹重命名为 `model-cn`（如果使用中文）或 `model-en`（如果使用英文）
3. 将该文件夹复制到项目的 `app/src/main/assets/` 目录下

文件结构应该如下：
```
llama.android/
└── app/
    └── src/
        └── main/
            └── assets/
                └── model-cn/          # 或 model-en
                    ├── am/
                    ├── conf/
                    ├── graph/
                    └── ivector/
```

### 3. 修改代码（如果使用英文模型）

如果您使用的是英文模型，需要在 `MainActivity.kt` 中修改模型名称：

找到 `initVoskModel()` 函数中的这一行：
```kotlin
val modelPath = StorageService.sync(this, "model-cn")
```

改为：
```kotlin
val modelPath = StorageService.sync(this, "model-en")
```

## 使用说明

1. 构建并运行应用
2. 应用启动时会自动加载 Vosk 模型（首次加载可能需要几秒钟）
3. 查看日志确认模型是否加载成功："Vosk模型加载成功"
4. 长按麦克风按钮开始录音，松开按钮结束录音
5. 录音过程中会实时显示识别的部分结果
6. 录音结束后显示最终识别结果

## 优势

- ✅ 完全离线工作，不需要网络连接
- ✅ 不依赖系统语音识别服务
- ✅ 隐私保护，语音数据不会上传到服务器
- ✅ 支持多种语言（中文、英文等）
- ✅ 实时显示部分识别结果

## 故障排除

### 模型加载失败
- 确认模型文件夹在正确的位置（`app/src/main/assets/model-cn`）
- 确认模型文件夹内包含必要的子文件夹（am, conf, graph, ivector）
- 检查应用日志中的错误信息

### 识别效果不佳
- 尝试使用大型模型以获得更好的识别精度
- 确保录音环境安静
- 对着麦克风清晰地说话

### 应用崩溃
- 检查是否有足够的存储空间
- 确认已授予麦克风权限
- 查看崩溃日志获取详细错误信息

## 更多信息

- Vosk 官方网站：https://alphacephei.com/vosk/
- Vosk Android 文档：https://github.com/alphacep/vosk-android-demo
- 更多语言模型：https://alphacephei.com/vosk/models
