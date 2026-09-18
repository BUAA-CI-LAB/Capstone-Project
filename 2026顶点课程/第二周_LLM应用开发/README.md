# 第二周：LLM 应用开发

- [LLM 应用开发基础课件](LLM应用开发基础.pptx)
- [结构化输出与结果处理课件](结构化输出与结果处理.pptx)
- [示例脚本](llm_app.py)
- [课后作业](作业.md)

本周使用 **Qwen2.5-1.5B-Instruct** 做一个简单代码助手，练习 Prompt、API 调用、流式输出、多轮修改和 JSON 输出校验。CPU 或 GPU 均可。

## 准备

复用第一周的 Python 环境和模型，在本目录安装依赖：

```bash
python -m pip install -r requirements.txt
```

尚未准备环境或模型的同学，参照[第一周说明](https://github.com/BUAA-CI-LAB/Capstone-Project/blob/master/2026%E9%A1%B6%E7%82%B9%E8%AF%BE%E7%A8%8B/%E7%AC%AC%E4%B8%80%E5%91%A8_LLM%E6%8E%A8%E7%90%86/README.md)，选择 1.5B 模型。

以下命令中的 `--model` 请替换为完整模型的实际本地目录。脚本自动启动本地服务，通过 OpenAI Python SDK 调用，无需云端 API Key，也无需另开服务终端。使用 GPU 时将 `--device cpu` 改为 `--device cuda`。

## 运行

先运行一次普通回答：

```bash
python llm_app.py --model ./models/Qwen2.5-1.5B-Instruct --device cpu --output results/plain.json
```

再运行内置参考示例，观察流式输出和格式校验：

```bash
python llm_app.py --model ./models/Qwen2.5-1.5B-Instruct --device cpu --stream --example --output results/example.json
```

示例要求生成 `count_positive(nums)`，返回列表中严格大于 0 的元素个数，并输出包含 `function_name`、`code` 两个字符串字段的 JSON。

修改脚本中的 Prompt 和 `CodeResult` 后，用下面的命令测试自己的版本，并继续多轮修改：

```bash
python llm_app.py --model ./models/Qwen2.5-1.5B-Instruct --device cpu --stream --structured --chat --output results/my_chat.json
```

第一轮后可以输入：“请改用普通 for 循环，并增加注释说明零不算正数。继续返回相同结构的 JSON。”输入 `/exit` 退出，`/reset` 清空历史。

测试自己的 Prompt 时不要带 `--example`，该参数会选择内置参考请求。`--structured` 使用 **Prompt 约定格式、客户端校验结果**，没有启用服务端约束解码。

## 查看结果

结果保存到 `--output` 指定的位置；多轮记录为 `my_chat.turn01.json`、`my_chat.turn02.json` 等。不同实验使用不同文件名，避免覆盖。

重点查看 `request.messages`（请求及历史）、`raw_text`（模型原文）和 `result`（校验结果）。`result.status` 为 `schema_valid` 表示字段通过，`result.python_syntax.status` 为 `syntax_valid` 表示 Python 语法通过。脚本不执行生成代码，功能是否正确仍需阅读判断。

如果输出截断，可缩短要求或加 `--max-new-tokens 384`；格式失败时，按错误提示修改 Prompt 或追加修正要求。具体任务和提交格式见[作业.md](作业.md)。
