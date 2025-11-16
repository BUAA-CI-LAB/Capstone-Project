- 环境配置：

    建议python>=3.9

    安装量化工具 `pip install llmcompressor`

    量化工具源码：[llm-compressor](https://github.com/vllm-project/llm-compressor)

- 示例代码：

    - FP8动态量化：
    
        ```shell
        MODEL_DIR=YOUR_PATH python3 fp8_dynamic.py
        ```

    - GPTQ W4A8量化：
    
        ```shell
        MODEL_DIR=YOUR_PATH python3 w4a8_gptq.py
        ```

        注意INT量化常搭配SmoothQuant使用

    - transformers离线伪量化推理：
    
        ```shell
        python3 transformers_offline.py $MODEL_PATH
        ```

    - vllm在线推理：
    
        ```shell
        vllm serve $MODEL_PATH
        bash openai.sh
        ```

