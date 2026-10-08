# 第五周：Agent Memory、Tool 与 MCP

- [下载课程课件](<agent-memory-tools-and-mcp.pptx>)

## 课程内容

### Agent Memory

围绕长期任务中的上下文增长、跨会话信息复用和知识更新，介绍记忆系统的主要机制与架构设计。

结合 MemGPT、Mem0、Reflexion、A-MEM 等论文，以及 OpenHands、Claude Code、AgentCore 和 Graphiti 等项目，讨论记忆如何形成、组织、读取和维护。

### File / Python / Search Tool

介绍文件访问、代码执行和外部搜索在 Agent 中的作用，结合 CodeAct 和依赖升级场景，理解工具如何配合完成任务。

### MCP

介绍 MCP 的接入背景、Host / Client / Server 架构，以及 Tools、Resources、Prompts 三类能力，结合接入示例理解其工作过程。

## 参考资料

- [Agent Memory 综述：Memory in the Age of AI Agents](https://arxiv.org/html/2512.13564v2)
  ——系统了解记忆的组织方式、机制与研究方向。
- [OpenHands Software Agent SDK](https://github.com/OpenHands/software-agent-sdk)
  ——了解 Coding Agent 的模块组成及可复用开发组件。
- [OpenHands Persistent Memory](https://docs.openhands.dev/sdk/guides/persistent-memory)
  ——了解基于文件的跨会话记忆设计。
- [Graphiti](https://github.com/getzep/graphiti)
  ——了解具有时间信息的知识图谱记忆。
- [CodeAct：Executable Code Actions Elicit Better LLM Agents（ICML 2024）](https://proceedings.mlr.press/v235/wang24h.html)
  ——理解代码作为 Agent 动作的设计思路。
- [MCP 官方架构说明](https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture)
  ——了解协议角色及能力交互方式。

[返回课程目录](../README.md)
