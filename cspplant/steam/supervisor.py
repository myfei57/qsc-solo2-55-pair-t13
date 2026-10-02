"""蒸汽压力前馈监督器。

它回答三件必须"定死"、且每轮都要能复盘的事：

1. **这一轮按前馈还是按反馈？** 由一张固定优先级的规则表（:data:`MODE_RULES`）
   判定，规则自上而下取第一条命中的，不允许按结果好不好反过来挑模式。信号缺失、
   压力已经越带、机组不在带负荷工况，都退回反馈；只有信号健康、压力尚在带内、
   预判缺口够大才进入前馈，退出用更小的阈值形成回差，防止在边界上抖动。
2. **旁路这一轮为什么开／关／不动？** 模式与方向确定后，旁路策略也是一张死表
   （:meth:`SteamSupervisor._bypass_action`）。前馈只在功率**下降**方向抢先开
   旁路（产汽要掉、先把负荷压下来，多出来的进汽能力会先把压力顶高）；功率上升
   方向不抢先开，先加负荷吸收，压力真越带再交给反馈兜底。
3. **负荷目标怎么给？** 前馈时在调度目标上叠加一个与预判缺口同号的负荷修正，
   反馈时只执行调度目标，压力残差仍由进汽量反馈项吸收。

监督器只**决策**，不直接动作执行器；它产出一份 :class:`ControlDecision`，由
:class:`~cspplant.steam.system.SteamSystem` 原样执行并原样落审计，做到"决策、
执行、审计三者同一份内容"。监督器在任何路径上都不允许关主汽门——主汽门只能
在旁路已开的前提下，由甩负荷／跳机编排去关。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import SteamLimits
from .feedforward import PowerSignal

# 控制模式。
FEEDBACK = "feedback"
FEEDFORWARD = "feedforward"

# 前馈方向：预判功率下降 / 上升 / 无显著变化。
DOWN = "down"
UP = "up"
FLAT = "flat"

# MW = 预判功率缺口(kW) / 焓差(kJ/kg) / (kg/s per MW)。
_KW_TO_STEAM_FLOW = 2600.0 * 0.33

# 负荷修正只抵消预判缺口的一部分，其余留给反馈项与压力带，避免过补偿。
FEEDFORWARD_LOAD_GAIN = 0.8

# 模式规则表：(规则名, 说明)。顺序即优先级，_select_mode 严格按此自上而下命中。
MODE_RULES: tuple[tuple[str, str], ...] = (
    ("signal_missing", "前馈信号缺失，退回压力反馈"),
    ("pressure_excursion", "压力已越出目标带，交回反馈纠偏"),
    ("plant_not_online", "机组未带负荷，不投入前馈"),
    ("ff_release_hysteresis", "已在前馈且预判缺口仍大于退出阈值，保持前馈"),
    ("ff_enter", "预判功率缺口超过进入阈值，抢先动作"),
    ("feedback_default", "扰动不显著，按压力反馈调节"),
)


@dataclass(frozen=True)
class ValveAction:
    """本轮对旁路阀的一个决策动作。"""

    action: str  # open / close / hold
    reason: str

    def as_dict(self) -> dict:
        return {"action": self.action, "reason": self.reason}


@dataclass(frozen=True)
class LoadAction:
    """本轮对汽机负荷目标的一个决策动作。"""

    target_mw: float
    base_mw: float
    trim_mw: float
    reason: str

    def as_dict(self) -> dict:
        return {
            "target_mw": round(self.target_mw, 3),
            "base_mw": round(self.base_mw, 3),
            "trim_mw": round(self.trim_mw, 3),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ControlDecision:
    """一个控制周期的完整决策，执行与审计共用这一份内容。"""

    cycle: int
    mode: str
    mode_rule: str
    mode_reason: str
    direction: str
    signal: PowerSignal
    pressure_mpa: float
    target_mpa: float
    bypass: ValveAction
    load: LoadAction
    online: bool
    rules: tuple = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "mode": self.mode,
            "mode_rule": self.mode_rule,
            "mode_reason": self.mode_reason,
            "direction": self.direction,
            "online": self.online,
            "signal": self.signal.as_dict(),
            "pressure_mpa": round(self.pressure_mpa, 3),
            "target_mpa": self.target_mpa,
            "bypass": self.bypass.as_dict(),
            "load": self.load.as_dict(),
            "rules": list(self.rules),
        }


class SteamSupervisor:
    """按固定规则表把现场信号翻译成旁路与负荷决策。"""

    def __init__(self, limits: SteamLimits) -> None:
        self._limits = limits
        self._mode = FEEDBACK
        self._cycle = 0
        self._ff_open_bypasses = 0

    @property
    def limits(self) -> SteamLimits:
        return self._limits

    @property
    def mode(self) -> str:
        return self._mode

    def ff_preemptive_opens(self) -> int:
        """返回前馈抢先开旁路的次数（复盘用）。"""

        return self._ff_open_bypasses

    def _direction(self, predicted_delta_kw: float) -> str:
        if predicted_delta_kw <= -self._limits.ff_min_delta_kw:
            return DOWN
        if predicted_delta_kw >= self._limits.ff_min_delta_kw:
            return UP
        return FLAT

    def _select_mode(
        self, signal: PowerSignal, pressure_mpa: float, online: bool
    ) -> tuple[str, str, str, list[tuple[str, bool, str]]]:
        """按 MODE_RULES 的固定顺序求模式，并记录每条规则的求值轨迹。

        每条规则都被求值并落进轨迹（复盘能看到"为什么不是别的模式"），但只有
        优先级最高的命中规则决定模式，保证"抢先还是等偏差"只有一个答案。
        """

        delta = signal.predicted_delta_kw
        in_band = self._limits.settled(pressure_mpa)

        predicates: list[tuple[str, str, bool, str]] = [
            (
                FEEDBACK,
                "signal_missing",
                not signal.available,
                signal.reason,
            ),
            (
                FEEDBACK,
                "pressure_excursion",
                signal.available and not in_band,
                "压力越出目标带" if not in_band else "压力在目标带内",
            ),
            (
                FEEDBACK,
                "plant_not_online",
                signal.available and in_band and not online,
                "未带负荷" if not online else "机组在线",
            ),
            (
                FEEDFORWARD,
                "ff_release_hysteresis",
                signal.available
                and in_band
                and online
                and self._mode == FEEDFORWARD
                and abs(delta) > self._limits.ff_release_delta_kw,
                (
                    f"缺口 {delta:.0f} kW，退出阈值 {self._limits.ff_release_delta_kw:.0f}"
                    if self._mode == FEEDFORWARD
                    else "当前不在前馈，跳过保持规则"
                ),
            ),
            (
                FEEDFORWARD,
                "ff_enter",
                signal.available and in_band and online and abs(delta) >= self._limits.ff_min_delta_kw,
                f"缺口 {delta:.0f} kW，进入阈值 {self._limits.ff_min_delta_kw:.0f}",
            ),
        ]

        trace: list[tuple[str, bool, str]] = []
        selected: tuple[str, str, str] | None = None
        for mode, name, hit, note in predicates:
            trace.append((name, hit, note))
            if hit and selected is None:
                selected = (mode, name, note)
        if selected is None:
            trace.append(("feedback_default", True, "扰动不显著"))
            return FEEDBACK, "feedback_default", "按压力反馈调节", trace
        return selected[0], selected[1], self._mode_reason(selected[1], selected[2]), trace

    @staticmethod
    def _mode_reason(rule: str, note: str) -> str:
        reasons = {
            "signal_missing": "前馈信号缺失，退回压力反馈",
            "pressure_excursion": "压力已越带，按偏差纠偏",
            "plant_not_online": "机组未带负荷，按反馈",
            "ff_release_hysteresis": "前馈保持（回差内不退出）",
            "ff_enter": "预判功率缺口超阈值，抢先动作",
        }
        return reasons.get(rule, note)

    def _bypass_action(
        self, mode: str, direction: str, pressure_mpa: float, bypass_open: bool, main_stop_open: bool
    ) -> ValveAction:
        """旁路死表：模式＋方向＋压力 → 开／关／保持，唯一确定。"""

        open_at = self._limits.pressure_target_mpa + self._limits.ff_bypass_open_band * self._limits.pressure_band_mpa
        close_at = self._limits.pressure_target_mpa + self._limits.ff_bypass_close_band * self._limits.pressure_band_mpa

        if mode == FEEDFORWARD:
            if direction == DOWN:
                # 定死：功率下降方向，前馈抢先开旁路，先于负荷动作接住蒸汽。
                if not bypass_open:
                    self._ff_open_bypasses += 1
                    return ValveAction("open", "feedforward_power_down_preempt")
                return ValveAction("hold", "feedforward_power_down_hold")
            # 上升／平坦方向不抢先开；保持现状，等压力说话。
            return ValveAction("hold", f"feedforward_{direction}_no_preempt")

        # 反馈模式：压力超过开启阈值才按偏差开（原反馈调节的安全兜底）。
        if not bypass_open and pressure_mpa >= open_at:
            return ValveAction("open", "feedback_pressure_over_threshold")
        # 回关要同时满足：压力回到关闭阈值以下，且主汽门开着（带负荷），
        # 绝不误关甩负荷／跳机后为隔离而保持开启的旁路。
        if bypass_open and pressure_mpa <= close_at and main_stop_open:
            return ValveAction("close", "feedback_pressure_recovered")
        if bypass_open:
            return ValveAction("hold", "feedback_wait_recovery")
        return ValveAction("hold", "feedback_within_band")

    def decide(
        self,
        signal: PowerSignal,
        pressure_mpa: float,
        dispatch_target_mw: float,
        bypass_open: bool,
        main_stop_open: bool,
        online: bool,
    ) -> ControlDecision:
        """给出本轮决策。纯函数式判定，不触碰执行器。"""

        self._cycle += 1
        mode, rule, reason, trace = self._select_mode(signal, pressure_mpa, online)
        self._mode = mode
        direction = self._direction(signal.predicted_delta_kw) if mode == FEEDFORWARD else FLAT

        base = max(0.0, min(self._limits.load_max_mw, float(dispatch_target_mw)))
        if mode == FEEDFORWARD:
            trim_mw = FEEDFORWARD_LOAD_GAIN * signal.predicted_delta_kw / _KW_TO_STEAM_FLOW
            trim_mw = max(-base, min(self._limits.load_max_mw - base, trim_mw))
            target_mw = max(0.0, min(self._limits.load_max_mw, base + trim_mw))
            load = LoadAction(target_mw, base, trim_mw, f"feedforward_load_{direction}")
        else:
            load = LoadAction(base, base, 0.0, "feedback_follow_dispatch")

        bypass = self._bypass_action(mode, direction, pressure_mpa, bypass_open, main_stop_open)
        return ControlDecision(
            cycle=self._cycle,
            mode=mode,
            mode_rule=rule,
            mode_reason=reason,
            direction=direction,
            signal=signal,
            pressure_mpa=float(pressure_mpa),
            target_mpa=self._limits.pressure_target_mpa,
            bypass=bypass,
            load=load,
            online=online,
            rules=tuple({"rule": name, "hit": hit, "note": note} for name, hit, note in trace),
        )
