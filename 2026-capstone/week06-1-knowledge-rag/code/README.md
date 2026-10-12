# Agent v1 参考代码

这个示例是一个项目文档助手：**记住回答偏好，读取启动报错，查阅项目资料，再给出排查建议。**

**这里提供的只是一个非常简单的参考示例。** 可以在此基础上按自己的想法自由修改、拓展，也可以采用其他实现方式，独立设计自己的项目，不必依托这份代码。

它延续第三周的 Agent Loop，加入文件 Memory 和一个小型向量知识库。主代码在 [agent_v1.py](agent_v1.py)，开放式实践安排见[作业与提交说明](../assignment.md)。

以下命令均在本 `code/` 目录内执行。可先复用[第一周](../../week01-llm-inference/README.md)准备的 Python 与 PyTorch 环境。

## 文件与流程

- `workspace/task.txt`：启动提示端口占用，希望改用 8001，并了解重启步骤与日志位置。
- `workspace/onboarding.txt`：可选的综合求助，包含首次安装、配置、端口排查、日志保留时间等问题。
- `knowledge/`：六份简短的项目资料，覆盖启动、配置、端口报错、日志、示例数据与结果导出，均为教学虚构内容。
- `data/preferences.json`：用户主动保存的回答偏好，运行后生成。
- `results/`：模型输出、工具调用、检索片段和最终回答，运行后生成。

Agent 可调用两个工具：`read_file(path)` 读取任务文件，`search_knowledge(query)` 检索资料。工具次序由模型决定，回答依据来自真实的文件读取和检索结果，没有预设最终答案。

文档中的 `app.py`、`.env` 等属于被查询的虚构项目，当前目录不提供该应用。文档助手会解释资料中的命令，不会安装或启动另一个项目。

默认任务围绕一次具体报错组织处理步骤，比两个独立事实问答多一点上下文。知识库还包含与这次报错无关的资料，可观察检索如何选择片段。综合求助文件留作自由尝试：其中日志保留天数没有在资料中规定，其他问题还需要组合不同文档；当前 3B 在这类综合任务中可能跳过检索或遗漏信息。

## 环境准备

复用前几周的 Python、PyTorch 和 Transformers 环境。这个版本使用 Qwen2.5-3B-Instruct 作为回答模型，BGE-small-zh-v1.5 作为中文 Embedding 模型。缺少依赖时再执行 `python -m pip install -r requirements.txt`，无需升级或重装现有环境。

已有 ModelScope 时，可将两个模型下载到本目录：

```bash
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-3B-Instruct', local_dir='models/Qwen2.5-3B-Instruct', allow_patterns=['*.json','*.safetensors','*.txt'])"
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5', local_dir='models/bge-small-zh-v1.5', allow_patterns=['config.json','pytorch_model.bin','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt'])"
```

也可以从 [Qwen 官方页面](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)和 [BGE 官方页面](https://huggingface.co/BAAI/bge-small-zh-v1.5)获取。脚本只读取本地模型，不在运行时自动下载。BGE 在 CPU 上执行，无需额外部署向量数据库。

## 运行示例

先保存一条偏好。这一步不需要加载模型：

```bash
python agent_v1.py --remember "我刚接触 Python，请用中文简洁说明操作步骤，保留必要的命令。"
python agent_v1.py --show-memory
```

单独看一下检索结果：

```bash
python agent_v1.py --search "Address already in use 应该修改什么配置，如何生效？"
```

再运行完整示例。下面用 GPU 加载回答模型；没有 GPU 时可改为 `--device cpu`，速度会慢一些：

```bash
python agent_v1.py --device cuda --example
```

模型放在其他位置时，用 `--model` 指定实际目录：

```bash
python agent_v1.py --model /实际路径/Qwen2.5-3B-Instruct --device cuda --example
```

示例只保留本地 Transformers 接入，不需要启动模型服务。工具定义通过 `tools` 传给模型自带的聊天模板；程序执行模型提出的工具调用并回传结果，最终回答使用普通文字。

程序会重新加载已保存的偏好，并用真实模型处理 `task.txt` 中的启动报错。终端显示调用过程，完整记录保存在 `results/` 的时间戳文件中。

## 自由修改

可以修改 `task.txt`，替换 `knowledge/` 中的 Markdown/TXT，或在 `make_tools` 和 `TOOL_SCHEMAS` 中注册、说明自己的工具。也可以把 `--example` 换成 `--task "你的任务"`。

例如，可以另问“结果导出到哪里，再次导出会怎样”，观察它是否检索到与首次启动不同的资料；也可以加入自己的文档或工具。参考问题是演示素材，不是要求同学逐项完成的验收清单。

想尝试综合求助，可执行 `python agent_v1.py --device cuda --task "请读取 onboarding.txt，检索知识库回答其中的问题，并标明来源。"`。

主文件按 Memory、知识库、工具、Agent Loop、模型接口排列。知识库采用简单字符窗口切分和 BGE 余弦相似度检索，每次返回两个片段；修改资料后重新启动即可，不存在需要同步的持久化向量缓存。

Memory 采用一个 JSON 字符串列表，可以直接编辑其中的偏好。当前只做显式保存和加载，不包含自动提取、遗忘或冲突处理。

引用形如 `[startup.md#1]`，可在运行记录的工具返回中找到对应原文。`status=ok` 表示模型给出了最终回答，不代表内容一定正确。模型出错时保留原始记录，便于自行观察和调整；示例不包含复杂的自动纠错流程。

默认任务的一次 3B 实测正确给出了端口修改、重启和日志位置，但把来源写成了虚构的网页链接。当前没有自动校验引用；核对来源应以运行记录中的真实片段 `id` 和原文为准。这个现象也可以作为自主拓展时的观察点。

工具调用的解析采用 Qwen2.5 的 `<tool_call>` 格式。更换模型系列或改接 API 时，可替换 `make_hf_llm` 与 `parse_tool_calls`；Memory、知识库和工具函数仍可复用。

BGE 的 CLS pooling、向量归一化和查询指令参照[官方模型说明](https://huggingface.co/BAAI/bge-small-zh-v1.5#using-huggingface-transformers)。这里专门采用 BGE 的编码方式，换其他 Embedding 模型时需要检查对应的输入与 pooling 要求。
