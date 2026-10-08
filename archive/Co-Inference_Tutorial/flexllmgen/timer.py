"""Global timer for profiling."""
from collections import namedtuple
import time
from typing import Callable, Any


class _Timer:
    """内部计时器类（用于单个计时任务）。"""

    def __init__(self, name: str):
        self.name = name  # 计时器名称
        self.started = False  # 标记是否已启动
        self.start_time = None  # 当前启动时间（单次）

        # 存储多次启动-停止的时间戳对
        self.start_times = []  # 所有启动时间戳列表
        self.stop_times = []   # 所有停止时间戳列表
        self.costs = []        # 每次计时的耗时（stop - start）列表

    def start(self, sync_func: Callable = None):
        """启动计时器。
        参数：
            sync_func: 同步函数（可选，如GPU同步操作，确保计时准确）
        """
        assert not self.started, f"计时器 {self.name} 已启动，不能重复启动。"
        if sync_func:
            sync_func()  # 执行同步（如等待GPU操作完成）

        self.start_time = time.perf_counter()  # 记录当前时间（高精度）
        self.start_times.append(self.start_time)
        self.started = True

    def stop(self, sync_func: Callable = None):
        """停止计时器。
        参数：
            sync_func: 同步函数（可选，同start）
        """
        assert self.started, f"计时器 {self.name} 未启动，无法停止。"
        if sync_func:
            sync_func()  # 执行同步

        stop_time = time.perf_counter()
        self.costs.append(stop_time - self.start_time)  # 计算本次耗时
        self.stop_times.append(stop_time)
        self.started = False

    def reset(self):
        """重置计时器（清空所有记录，恢复初始状态）。"""
        self.started = False
        self.start_time = None
        self.start_times = []
        self.stop_times = []
        self.costs = []

    def elapsed(self, mode: str = "average"):
        """计算总耗时或平均耗时。
        参数：
            mode: 计算模式，"average"返回平均值，"sum"返回总和
        返回：
            计算后的耗时（秒）
        """
        if not self.costs:  # 若没有计时记录，返回0
            return 0.0
        if mode == "average":
            return sum(self.costs) / len(self.costs)
        elif mode == "sum":
            return sum(self.costs)
        else:
            raise RuntimeError("支持的模式：average（平均） | sum（总和）")


class Timers:
    """计时器组（管理多个_Timer实例，方便批量计时）。"""

    def __init__(self):
        self.timers = {}  # 存储计时器的字典，key为名称，value为_Timer实例

    def __call__(self, name: str):
        """通过名称获取或创建计时器。
        参数：
            name: 计时器名称
        返回：
            对应的_Timer实例
        """
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def __contains__(self, name: str):
        """检查是否存在指定名称的计时器。"""
        return name in self.timers


# 全局计时器实例，供外部直接调用（如timers("prefill").start()）
timers = Timers()

# 事件命名元组，用于记录追踪事件的时间戳、名称和附加信息
Event = namedtuple("Event", ("tstamp", "name", "info"))


class Tracer:
    """活动追踪器（记录关键操作的事件日志）。"""

    def __init__(self):
        self.events = []  # 存储所有事件的列表

    def log(self, name: str, info: Any, sync_func: Callable = None):
        """记录一个事件。
        参数：
            name: 事件名称
            info: 事件的附加信息（任意类型）
            sync_func: 同步函数（可选，确保时间戳准确）
        """
        if sync_func:
            sync_func()  # 执行同步

        # 记录当前时间戳、事件名称和信息
        self.events.append(Event(time.perf_counter(), name, info))


# 全局追踪器实例，供外部记录事件（如tracer.log("load_weight", "layer_0")）
tracer = Tracer()