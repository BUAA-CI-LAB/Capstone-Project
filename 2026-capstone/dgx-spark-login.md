# DGX Spark 学生登录手册

课程组提供一台 NVIDIA DGX Spark（主机名 `spark-b1a6`），用于课程实验和项目开发。本手册说明如何登录、传文件、配置环境，以及使用时的注意事项。

## 基本信息

| 项目 | 内容 |
| --- | --- |
| 地址 | `10.134.43.123`，端口 **2222** |
| 账号 | 每组一个账号，用户名是 `group` 加组号（如第 3 组为 `group3`），密码由课程组发给各组 |
| 硬件 | NVIDIA GB10（Blackwell），128 GB CPU/GPU 统一内存，20 核 ARM CPU，4 TB 硬盘 |
| 系统 | Ubuntu 24.04（arm64），NVIDIA 驱动 580，CUDA 13.0 |
| 权限 | 普通用户：可以用 GPU，不能 `sudo`，不能用 `docker` |
| 网络 | 必须连北航校园网。校外需先连学校 VPN 客户端 |

## 第一次登录

建议第一次在终端里登录（Windows 用 PowerShell，macOS/Linux 用终端），以便核对主机指纹：

```bash
ssh -p 2222 你的用户名@10.134.43.123
```

1. 首次连接会提示确认主机指纹：

   ```text
   The authenticity of host '[10.134.43.123]:2222 ([10.134.43.123]:2222)' can't be established.
   ED25519 key fingerprint is SHA256:tU2OdrKz7NeDWntH1pqhwDHICBjV4mKXnpW9fqoLIAo.
   Are you sure you want to continue connecting (yes/no/[fingerprint])?
   ```

   确认指纹是 `SHA256:tU2OdrKz7NeDWntH1pqhwDHICBjV4mKXnpW9fqoLIAo` 后输入 `yes`（也可以直接粘贴这串指纹，ssh 会自动比对）。

2. 输入课程组发放的密码（输入时不显示字符），看到 `你的用户名@spark-b1a6:~$` 提示符即登录成功。

同组成员共用这个账号，**请不要修改密码**，否则组员会登录不上。每位组员可以按下一节各自配置 SSH 密钥，之后登录不用再输密码。

## 配置 SSH 密钥和简称（推荐）

配置后每次只需 `ssh spark`，不用再输密码。以下命令都在**自己的电脑**上执行，同组成员各自配置一次即可，互不影响。

1. 生成密钥（已有 `~/.ssh/id_ed25519` 可跳过），一路回车即可：

   ```bash
   ssh-keygen -t ed25519
   ```

2. 把公钥放到 Spark 上。macOS/Linux：

   ```bash
   ssh-copy-id -p 2222 你的用户名@10.134.43.123
   ```

   Windows PowerShell 没有 `ssh-copy-id`，改用：

   ```powershell
   type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh -p 2222 你的用户名@10.134.43.123 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
   ```

3. 编辑 SSH 配置文件：macOS/Linux 为 `~/.ssh/config`，Windows 为 `C:\Users\你的电脑用户名\.ssh\config`（没有扩展名）。追加：

   ```text
   Host spark
       HostName 10.134.43.123
       Port 2222
       User 你的用户名
       IdentityFile ~/.ssh/id_ed25519
       ServerAliveInterval 30
   ```

之后直接 `ssh spark` 登录；下文命令中的 `spark` 都指这个简称。

## VS Code / Cursor 远程开发

1. 先按上文在终端完成第一次登录，最好也配置好 `Host spark`。
2. 安装 **Remote - SSH** 扩展（Cursor 自带同类功能）。
3. 按 `F1`，选择 `Remote-SSH: Connect to Host...`，选 `spark`；询问系统类型时选 **Linux**。
4. 首次连接会在 Spark 上自动下载服务端，需要一两分钟。连接后 `File → Open Folder` 打开自己的目录即可。

终端、Python 解释器和 Jupyter Notebook 都在 Spark 上运行；插件需要在远程端单独安装（扩展页面的 “Install in SSH: spark”）。

## 传输文件

```bash
scp -P 2222 data.zip 你的用户名@10.134.43.123:~/        # 上传；scp 的端口参数是大写 -P
scp spark:~/results/run.json .                          # 配置了 Host spark 后可直接用简称
scp -r spark:~/results ./results                        # 下载整个目录
rsync -avP ./project/ spark:~/project/                  # 增量同步（macOS/Linux）
```

Windows 自带 `scp`，没有 `rsync`。也可以用 VS Code 资源管理器拖拽上传，或用 WinSCP / FileZilla（SFTP 协议，端口 2222）。

**Spark 上无法访问 GitHub**（`git clone https://github.com/...` 和 `pip install git+https://github.com/...` 会失败）。请在自己电脑上 clone 后用 `scp`/`rsync` 传上去；Gitee 和 PyPI 镜像可以正常访问。

## 长时间任务：使用 tmux

SSH 断开时，前台运行的程序会被终止。训练、模型服务等长任务放在 tmux 里运行：

```bash
tmux new -s exp        # 新建名为 exp 的会话，在里面启动程序
                       # 按 Ctrl+B，松开后按 D：离开会话，程序继续运行
tmux ls                # 查看会话
tmux attach -t exp     # 重新进入
exit                   # 在会话内执行，结束该会话
```

## Python 环境

系统自带 Python 3.12。不能 `sudo`，所以请在自己目录下建虚拟环境：

```bash
python3 -m venv ~/venvs/capstone
source ~/venvs/capstone/bin/activate
pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple
pip install -U pip
pip install torch
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

正常输出类似 `2.13.0+cu130 True NVIDIA GB10`。直接 `pip install torch` 得到的 arm64 版本已带 CUDA 13.0，不需要另外指定 CUDA 版本；如需官方源，可用 `--index-url https://download.pytorch.org/whl/cu130`。每次登录后先 `source ~/venvs/capstone/bin/activate` 再运行程序。需要安装系统软件包时，请联系课程组，不要尝试自行提权。

## 模型下载与共享模型

请使用 [ModelScope](https://modelscope.cn) 下载模型：

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', local_dir='models/Qwen2.5-1.5B-Instruct')"
```

课程组已下载的模型放在 `/srv/spark/models` 下，所有人可读、不可写。**可以直接使用这些模型，不用重复下载到自己目录**：

| 模型 | 本地路径 | 大小 |
| --- | --- | --- |
| Qwen3.8-27B-FP8 | `/srv/spark/models/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` | 29 GB |
| Qwen3-8B | `/srv/spark/models/Qwen3-8B` | 16 GB |
| Qwen3-1.7B | `/srv/spark/models/Qwen3-1.7B` | 4 GB |

在代码中把模型 ID 换成上面的本地路径即可，例如 `AutoTokenizer.from_pretrained("/srv/spark/models/Qwen3-8B")`。

## 使用规范

- 运行大任务前先看资源占用：

  ```bash
  free -h        # 统一内存剩余量（GPU 和 CPU 共用）
  nvidia-smi     # GPU 利用率和进程；GB10 的显存一栏显示 N/A，看内存请用 free -h
  ps -eo user,pid,rss,etime,cmd --sort=-rss | head    # 占内存最多的进程
  ```

- 自己启动 vLLM 时必须调低 `--gpu-memory-utilization`（例如 0.2）。默认值 0.9 会占掉约 110 GB 内存，导致其他人的程序失败。

- 任务结束后结束自己的进程（`ps -u $USER` 查看，`kill <PID>` 结束），退出不用的 tmux 会话。

- 硬盘大家共用。大文件用完及时删除；数据集和模型尽量用共享目录。

- 同组成员共用一个账号，家目录、进程和 tmux 会话彼此可见。建议每人在家目录下建自己的子目录，结束进程前确认是自己的。

- 账号密码只在本组内使用，不要告诉其他组的同学或借给他人。API Key 等敏感信息不要写进代码提交到仓库。

## 常见问题

**`Connection timed out`**：确认已连接北航校园网；确认端口是 `-p 2222`。

**指纹与本手册不一致**：最常见的原因是漏写了 `-p 2222`，连到了另一台机器。确认端口后仍不一致，不要输入 yes，联系课程组。

**`Permission denied (publickey,password)`**：用户名或密码错误。用户名区分大小写；密码输入时不显示字符。

**`WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!`**：只有课程组通知过 Spark 重装系统时，才在自己电脑上执行 `ssh-keygen -R "[10.134.43.123]:2222"` 后重新连接；否则不要连接，先联系课程组。

**忘记密码**：先问同组成员；密码被改掉或泄露时，联系课程组重置。

**`pip install` 很慢或超时**：确认已按上文设置北外镜像；北外也慢时，换成华为云 `https://repo.huaweicloud.com/repository/pypi/simple`。

**`CUDA out of memory` 或系统突然变慢**：用 `free -h` 查看剩余内存，减小 batch size、上下文长度或 `--gpu-memory-utilization`；必要时在群里确认是否有人占用了大量内存。

[返回课程目录](README.md)
