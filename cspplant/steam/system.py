"""蒸汽系统控制器。

把热功率换算成产汽量，把汽机负荷换算成进汽量，其余走旁路，三者一起推进主
蒸汽压力。系统对外只提供三件事：建负荷、甩负荷、跳机；每一步都由
:class:`~cspplant.steam.turbine.Turbine` 保证"旁路先开、主汽门后关"。

每个常规周期还经过 :class:`~cspplant.steam.supervisor.SteamSupervisor`：按固定
规则表决定本轮是前馈（抢先）还是反馈（等压力偏差），再给出旁路与负荷动作。
决策对象只有一份：先在监督器里算定，再由这里原样执行，最后原样回传给控制台
与审计，三者口径一致、逐轮可复盘。
"""

from __future__ import annotations

import threading

from ..config import SteamLimits
from .bypass import BypassValve
from .feedforward import PowerSignal, PowerSignalTracker
from .pressure import SteamPressure
from .snapshot import SteamSnapshot
from .supervisor import FEEDBACK, FEEDFORWARD, ControlDecision, SteamSupervisor
from .turbine import Turbine, TurbineReport
from .turbine_load import LoadRamp

# 产汽焓差（kJ/kg），把吸热器热功率换算成蒸汽流量。
STEAM_ENTHALPY_KJ_KG = 2600.0

# 汽机负荷到进汽量的换算（kg/s per MW）。
TURBINE_KG_S_PER_MW = 0.33

# 旁路通流能力（kg/s）。
BYPASS_CAPACITY_KG_S = 40.0

# 主蒸汽压力偏差到汽机进汽量修正的换算（kg/s per MPa）。
PRESSURE_CONTROL_GAIN_KG_S_PER_MPA = 8.0


class SteamSystem:
    """蒸汽发生器出口到汽机的整条链。"""

    def __init__(
        self,
        limits: SteamLimits,
        pressure: SteamPressure,
        ramp: LoadRamp,
        bypass: BypassValve,
        turbine: Turbine,
    ) -> None:
        self._limits = limits
        self._pressure = pressure
        self._ramp = ramp
        self._bypass = bypass
        self._turbine = turbine
        self._supervisor = SteamSupervisor(limits)
        self._signal_tracker = PowerSignalTracker(
            limits.ff_power_lead_s, limits.ff_signal_max_age_s
        )
        self._production = 0.0
        self._last_decision: ControlDecision | None = None
        self._lock = threading.RLock()

    @property
    def limits(self) -> SteamLimits:
        return self._limits

    @property
    def pressure(self) -> SteamPressure:
        return self._pressure

    @property
    def ramp(self) -> LoadRamp:
        return self._ramp

    @property
    def bypass(self) -> BypassValve:
        return self._bypass

    @property
    def turbine(self) -> Turbine:
        return self._turbine

    @property
    def supervisor(self) -> SteamSupervisor:
        return self._supervisor

    @property
    def signal_tracker(self) -> PowerSignalTracker:
        return self._signal_tracker

    def last_decision(self) -> ControlDecision | None:
        """返回最近一个周期的前馈／反馈决策（供控制台展示）。"""

        return self._last_decision

    def production_kg_s(self) -> float:
        """返回最近一次换算得到的产汽量。"""

        return self._production

    def draw_kg_s(self, load_mw: float) -> float:
        """按汽机负荷给出进汽量。"""

        return max(0.0, float(load_mw)) * TURBINE_KG_S_PER_MW

    def bypass_draw_kg_s(self, surplus_kg_s: float) -> float:
        """按旁路通流能力给出旁路流量。"""

        if not self._bypass.is_open():
            return 0.0
        return max(0.0, min(BYPASS_CAPACITY_KG_S, float(surplus_kg_s)))

    def update(
        self,
        thermal_power_kw: float,
        dt: float,
        now: float | None = None,
        dispatch_target_mw: float | None = None,
        online: bool = True,
        feedforward_available: bool | None = None,
    ) -> dict:
        """推进一个步长：采样前馈 → 决策 → 旁路／负荷 → 压力。

        ``now`` 为前馈信号时间戳；缺省用 0 起始。``dispatch_target_mw`` 是调度
        目标负荷，缺省沿用斜率器当前目标（保持旧调用兼容）。``online`` 表示机组
        是否已带负荷；``feedforward_available`` 可显式登记前馈仪表在线与否。
        """

        if dt <= 0:
            raise ValueError("步长必须为正")
        with self._lock:
            timestamp = 0.0 if now is None else float(now)
            if feedforward_available is not None:
                self._signal_tracker.set_available(feedforward_available)
            signal = self._signal_tracker.sample(thermal_power_kw, timestamp, dt)

            production = max(0.0, float(thermal_power_kw)) / STEAM_ENTHALPY_KJ_KG
            self._production = production
            self._ramp.step(dt)

            base_target = (
                self._ramp.target_mw() if dispatch_target_mw is None else float(dispatch_target_mw)
            )
            decision = self._supervisor.decide(
                signal=signal,
                pressure_mpa=self._pressure.value_mpa(),
                dispatch_target_mw=base_target,
                bypass_open=self._bypass.is_open(),
                main_stop_open=self._turbine.main_stop_open(),
                online=online,
            )
            executed = self._execute(decision, timestamp)
            self._last_decision = decision

            # 压力偏差仍由进汽量反馈项吸收：压力偏高多抽、偏低少抽，前馈与反馈
            # 两种模式下都保留，作为前馈之后的稳态纠偏与前馈缺失时的兜底。
            error = self._pressure.value_mpa() - self._limits.pressure_target_mpa
            to_turbine = max(
                0.0,
                self.draw_kg_s(self._ramp.current_mw())
                + error * PRESSURE_CONTROL_GAIN_KG_S_PER_MPA,
            )
            surplus = production - to_turbine
            to_bypass = self.bypass_draw_kg_s(surplus)
            pressure = self._pressure.update(production, to_turbine, to_bypass, dt)
            return {
                "production_kg_s": round(production, 3),
                "to_turbine_kg_s": round(to_turbine, 3),
                "to_bypass_kg_s": round(to_bypass, 3),
                "pressure_mpa": round(pressure, 3),
                "settled": self._pressure.settled(),
                "exceedances": self._pressure.exceedances(),
                "decision": decision.as_dict(),
                "executed": executed,
            }

    def _execute(self, decision: ControlDecision, now: float) -> dict:
        """按决策对象动作旁路与负荷，返回实际执行结果（与决策同源）。

        旁路动作严格先于负荷动作，且这里永不关主汽门。
        """

        result: dict = {"steps": [], "violations": []}
        valve = decision.bypass
        before_open = self._bypass.is_open()
        if valve.action == "open" and not self._bypass.is_open():
            self._bypass.open(now)
        elif valve.action == "close" and self._bypass.is_open():
            self._bypass.close()
        result["steps"].append(
            {
                "component": "steam",
                "action": f"bypass_{valve.action}",
                "detail": {
                    "open": self._bypass.is_open(),
                    "reason": valve.reason,
                    "changed": self._bypass.is_open() != before_open,
                },
            }
        )

        self._ramp.request(decision.load.target_mw)
        result["steps"].append(
            {
                "component": "steam",
                "action": "load_target",
                "detail": {
                    "target_mw": round(decision.load.target_mw, 3),
                    "trim_mw": round(decision.load.trim_mw, 3),
                    "reason": decision.load.reason,
                },
            }
        )

        # 常规建负荷时按需开主汽门：反馈模式、旁路关闭、有正负荷目标即开（与
        # 机组是否已标记在线无关——正是这一步把机组带到在线）。前馈暂态期间
        # 不改变主汽门状态；关主汽门永远不发生在这里。
        if (
            decision.mode == FEEDBACK
            and not self._bypass.is_open()
            and not self._turbine.main_stop_open()
            and decision.load.target_mw > 0.0
        ):
            self._turbine.open_main_stop()
            result["steps"].append(
                {
                    "component": "steam",
                    "action": "open_main_stop",
                    "detail": {"open": True, "reason": "feedback_engage_load"},
                }
            )
        result["mode"] = decision.mode
        result["mode_rule"] = decision.mode_rule
        return result

    def load(self, mw: float) -> TurbineReport:
        """建负荷。"""

        with self._lock:
            return self._turbine.raise_load(mw)

    def shed(self, target_mw: float, now: float) -> TurbineReport:
        """甩负荷（先开旁路、再关主汽门）。"""

        with self._lock:
            return self._turbine.shed(target_mw, now)

    def trip(self, now: float) -> TurbineReport:
        """跳机。"""

        with self._lock:
            return self._turbine.trip(now)

    def disturb(self, delta_mpa: float) -> dict:
        """注入一次压力扰动，返回扰动后的压力。"""

        with self._lock:
            before = self._pressure.value_mpa()
            after = self._pressure.disturb(delta_mpa)
            return {
                "before_mpa": round(before, 3),
                "after_mpa": round(after, 3),
                "settled": self._pressure.settled(),
            }

    def status(self) -> dict:
        """汇总蒸汽系统状态。"""

        decision = self._last_decision
        return {
            "pressure_mpa": round(self._pressure.value_mpa(), 3),
            "target_mpa": self._limits.pressure_target_mpa,
            "band_mpa": self._limits.pressure_band_mpa,
            "settled": self._pressure.settled(),
            "peak_mpa": round(self._pressure.peak_mpa(), 3),
            "exceedances": self._pressure.exceedances(),
            "production_kg_s": round(self._production, 3),
            "load_mw": round(self._ramp.current_mw(), 3),
            "target_mw": round(self._ramp.target_mw(), 3),
            "main_stop_open": self._turbine.main_stop_open(),
            "bypass": self._bypass.as_dict(),
            "sheds": self._turbine.sheds(),
            "trips": self._turbine.trips(),
            "control_mode": self._supervisor.mode,
            "ff_preemptive_opens": self._supervisor.ff_preemptive_opens(),
            "ff_signal_available": self._signal_tracker.available(),
            "ff_missing": self._signal_tracker.missing_count(),
            "last_decision": None if decision is None else decision.as_dict(),
        }

    def snapshot(self) -> SteamSnapshot:
        """返回蒸汽系统状态快照对象。"""

        return SteamSnapshot(
            pressure_mpa=self._pressure.value_mpa(),
            settled=self._pressure.settled(),
            bypass_open=self._bypass.is_open(),
            main_stop_open=self._turbine.main_stop_open(),
            load_mw=self._ramp.current_mw(),
            target_mw=self._ramp.target_mw(),
            production_kg_s=self._production,
            sheds=self._turbine.sheds(),
        )
