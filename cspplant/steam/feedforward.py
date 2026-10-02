"""吸热器功率前馈信号。

压力反馈的问题是要等压力真的偏了才动手，云一遮、降额一动，产汽量和进汽量的
差额已经形成，只能事后补。前馈把"吸热器侧功率正在怎么变"提前接进来：对功率
变化率做一阶滤波，再外推一个固定提前量，得到预判的功率缺口；超压还是欠压、
缺口多大，在压力还没动之前就已知。

信号可能缺失（仪表掉线、值不合理、长时间不刷新）。前馈信号缺失属于**部分
失败**——只丢掉"抢先"这一层能力，控制必须退回原来的压力反馈，而不是停机或
带着假信号乱动。因此 :class:`PowerSignalTracker.sample` 永远返回一个结果，
缺失与否、为什么缺失都体现在字段里，由上层据此定模式。
"""

from __future__ import annotations

from dataclasses import dataclass

# 变化率一阶滤波系数：越小越平滑，对云遮这类斜坡越稳、对阶跃越钝。
SLOPE_FILTER_ALPHA = 0.3


@dataclass(frozen=True)
class PowerSignal:
    """一个周期的吸热器功率前馈信号。

    ``available`` 为 ``False`` 时 ``predicted_delta_kw`` 强制取 0，保证缺失
    信号不会被当成"功率没变"以外的任何东西使用。
    """

    available: bool
    power_kw: float
    slope_kw_s: float
    predicted_delta_kw: float
    age_s: float
    reason: str

    def as_dict(self) -> dict:
        return {
            "available": self.available,
            "power_kw": round(self.power_kw, 2),
            "slope_kw_s": round(self.slope_kw_s, 3),
            "predicted_delta_kw": round(self.predicted_delta_kw, 2),
            "age_s": round(self.age_s, 3),
            "reason": self.reason,
        }


class PowerSignalTracker:
    """吸热器功率采样、变化率滤波与缺失判定。"""

    def __init__(self, lead_s: float, max_age_s: float) -> None:
        self._lead_s = float(lead_s)
        self._max_age_s = float(max_age_s)
        self._enabled = True
        self._slope_kw_s = 0.0
        self._last_power: float | None = None
        self._age_s = 0.0
        self._samples = 0
        self._missing_count = 0

    def set_available(self, available: bool) -> bool:
        """登记前馈仪表是否在线（物理信号缺失）。

        由缺失恢复在线时清掉历史，强制重新走一个 ``no_history`` 周期，避免拿
        掉线前后的两个采样算出一个巨大的假变化率。
        """

        becoming_available = bool(available) and not self._enabled
        self._enabled = bool(available)
        if becoming_available:
            self._last_power = None
            self._slope_kw_s = 0.0
            self._age_s = 0.0
        return self._enabled

    def available(self) -> bool:
        """当前前馈信号是否可用。"""

        return self._enabled

    def sample(self, power_kw: float, now: float, dt: float) -> PowerSignal:
        """推进一个周期，返回前馈信号（缺失时退化为零缺口信号）。

        变化率按本周期步长 ``dt`` 计算（控制循环每个周期推进一次）；``now``
        只作为信号时间戳记录。信号"多久没刷新"由连续采样累计的 ``age_s`` 体现，
        超过 ``ff_signal_max_age_s`` 即判缺失——在每个周期都调用本方法的常态下
        它不会触发，真正的物理掉线由 :meth:`set_available` 登记。
        """

        if dt <= 0:
            raise ValueError("步长必须为正")
        self._samples += 1
        self._age_s += dt
        power = float(power_kw)

        if not self._enabled:
            self._missing_count += 1
            return self._as_missing(power, 0.0, "feedforward_sensor_offline")

        if power < 0.0:
            self._missing_count += 1
            return self._as_missing(power, 0.0, "feedforward_value_implausible")

        # 首个周期没有历史，无法算变化率，视为暂不可用（而非缺口为零）。
        if self._last_power is None:
            self._last_power = power
            self._age_s = 0.0
            self._missing_count += 1
            return self._as_missing(power, 0.0, "feedforward_no_history")

        if self._age_s > self._max_age_s:
            self._missing_count += 1
            # 仍刷新历史，但本轮按缺失处理，避免用跨越大空档的假斜率。
            self._last_power = power
            self._age_s = 0.0
            return self._as_missing(power, self._slope_kw_s, "feedforward_signal_stale")

        raw_slope = (power - self._last_power) / dt
        self._slope_kw_s = (
            SLOPE_FILTER_ALPHA * raw_slope + (1.0 - SLOPE_FILTER_ALPHA) * self._slope_kw_s
        )
        self._last_power = power
        self._age_s = 0.0
        return PowerSignal(
            True, power, self._slope_kw_s, self._slope_kw_s * self._lead_s, dt, "ok"
        )

    def _as_missing(self, power_kw: float, slope_kw_s: float, reason: str) -> PowerSignal:
        return PowerSignal(False, max(0.0, power_kw), slope_kw_s, 0.0, self._age_s, reason)

    def samples(self) -> int:
        return self._samples

    def missing_count(self) -> int:
        return self._missing_count
