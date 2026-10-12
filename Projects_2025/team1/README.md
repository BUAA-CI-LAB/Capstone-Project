# 1 主库编译

```
cd /MNN/project/android
mkdir build_64
../build_64.sh "-DMNN_LOW_MEMORY=true -DMNN_CPU_WEIGHT_DEQUANT_GEMM=true -DMNN_BUILD_LLM=true -DMNN_SUPPORT_TRANSFORMER_FUSE=true -DMNN_ARM82=true -DMNN_USE_LOGCAT=true -DMNN_OPENCL=true -DLLM_SUPPORT_VISION=true -DMNN_BUILD_OPENCV=true -DMNN_IMGCODECS=true -DLLM_SUPPORT_AUDIO=true -DMNN_BUILD_AUDIO=true -DMNN_BUILD_DIFFUSION=ON -DMNN_SEP_BUILD=OFF -DCMAKE_INSTALL_PREFIX=."
make install
```

# 2 ARS模块下载

我们的项目运行有影响软件正常运行的必要的开源ARS模块。下载：[VOSK Models](https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip)（文件下载链接），或者在模型网站[VOSK Models](https://alphacephei.com/vosk/models)下载vosk-model-cn-0.22模型

下载该模型过后，将文件夹解压在`/MNN/apps/Android/MnnLlmChat/app/src/main/assets/vosk-model-cn-0.22/`下，并保证项目结构为：

```
vosk-model-cn-0.22
│  README
│  uuid
├─am
├─conf
├─graph
├─ivector
├─rescore
└─rnnlm
```

其中，uuid文件需要手动创建，内容为

```
8f3c1a3e-7e5c-4c0b-b2f0-1e1c6a9d9f21
```

