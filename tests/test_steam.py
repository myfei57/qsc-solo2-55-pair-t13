"""蒸汽旁路顺序、负荷斜率与压力。"""

from __future__ import annotations

import pytest

from cspplant.config import default_config
from cspplant.errors import InterlockError
from cspplant.steam.bypass import BypassValve
from cspplant.steam.feedforward import PowerSignalTracker
from cspplant.steam.pressure import SteamPressure
from cspplant.steam.supervisor import FEEDBACK, FEEDFORWARD, MODE_RULES, SteamSupervisor
from cspplant.steam.system import SteamSystem
from cspplant.steam.turbine import Turbine
from cspplant.steam.turbine_load import LoadRamp


def build_turbine() -> Turbine:
    limits = default_config().steam
    return Turbine(limits, LoadRamp(limits), BypassValve(limits))


STEADY_POWER_KW = 18000.0
STEADY_LOAD_MW = 20.0
STEP_S = 2.0


def build_settled_system(cycles: int = 300) -> tuple[SteamSystem, float]:
    """建一个已经稳在目标压力、带负荷在线的蒸汽系统，返回系统与当前时刻。"""

    limits = default_config().steam
    ramp = LoadRamp(limits)
    bypass = BypassValve(limits)
    system = SteamSystem(
        limits, SteamPressure(limits), ramp, bypass, Turbine(limits, ramp, bypass)
    )
    now = 0.0
    for _ in range(cycles):
        system.update(
            STEADY_POWER_KW, STEP_S, now=now, dispatch_target_mw=STEADY_LOAD_MW, online=True
        )
        now += STEP_S
    return system, now



def test_bypass_rejects_double_open():
    valve = BypassValve(default_config().steam)
    valve.open(10.0)
    with pytest.raises(InterlockError):
        valve.open(11.0)
    assert valve.held_open_s(15.0) == pytest.approx(5.0)
    valve.close()
    with pytest.raises(InterlockError):
        valve.close()


def test_shed_opens_bypass_before_closing_main_stop():
    turbine = build_turbine()
    turbine.raise_load(20.0)
    assert turbine.main_stop_open() is True
    report = turbine.shed(0.0, 100.0)
    assert report.ok is True
    actions = [step["action"] for step in report.steps]
    assert actions.index("open_bypass") < actions.index("lower_load")
    assert actions.index("lower_load") < actions.index("close_main_stop")
    assert turbine.bypass.is_open() is True
    assert turbine.main_stop_open() is False
    assert turbine.sheds() == 1


def test_shed_below_current_load_is_required():
    turbine = build_turbine()
    turbine.raise_load(10.0)
    report = turbine.shed(20.0, 100.0)
    assert report.ok is False
    assert "shed_target_not_lower" in report.violations


def test_load_ramp_respects_rate_and_ceiling():
    limits = default_config().steam
    ramp = LoadRamp(limits)
    ramp.request(limits.load_max_mw * 10)
    assert ramp.target_mw() == pytest.approx(limits.load_max_mw)
    assert ramp.step(1.0) == pytest.approx(limits.load_rate_mw_s)
    ramp.settle()
    assert ramp.current_mw() == pytest.approx(limits.load_max_mw)


def test_turbine_trip_is_bypass_first():
    turbine = build_turbine()
    turbine.raise_load(12.0)
    report = turbine.trip(50.0)
    assert report.ok is True
    assert turbine.bypass.is_open() is True
    assert turbine.main_stop_open() is False
    assert turbine.ramp.current_mw() == pytest.approx(0.0)
    assert turbine.trips() == 1


def test_pressure_moves_with_net_steam():
    limits = default_config().steam
    pressure = SteamPressure(limits)
    start = pressure.value_mpa()
    pressure.update(20.0, 5.0, 0.0, 10.0)
    assert pressure.value_mpa() > start
    pressure.update(0.0, 5.0, 0.0, 10.0)
    assert pressure.value_mpa() < start + 100.0
    assert pressure.updates() == 2


def test_steam_system_settles_pressure_under_load(online):
    online.clock.advance(1.0)
    for _ in range(400):
        online.cycle(dt=2.0)
    status = online.status()["steam"]
    assert status["settled"] is True
    assert status["load_mw"] > 20.0
    assert status["main_stop_open"] is True
    assert status["bypass"]["open"] is False


def test_disturbance_is_visible_and_recoverable(online):
    disturbed = online.disturb_steam(1.5)
    assert disturbed["settled"] is False
    for _ in range(400):
        online.cycle(dt=2.0)
    assert online.status()["steam"]["settled"] is True


# ---- 吸热器功率前馈 -------------------------------------------------------

def test_close_main_stop_rejected_when_bypass_closed():
    """任何路径都不能绕过"旁路先开、主汽门后关"。"""

    turbine = build_turbine()
    turbine.raise_load(10.0)
    assert turbine.bypass.is_open() is False
    with pytest.raises(InterlockError):
        turbine.close_main_stop()
    assert turbine.main_stop_open() is True


def test_feedforward_signal_missing_is_partial_failure():
    """信号缺失返回 unavailable 信号，而不是抛异常或给出假缺口。"""

    limits = default_config().steam
    tracker = PowerSignalTracker(limits.ff_power_lead_s, limits.ff_signal_max_age_s)
    tracker.set_available(False)
    signal = tracker.sample(9000.0, 0.0, STEP_S)
    assert signal.available is False
    assert signal.predicted_delta_kw == 0.0
    assert signal.reason == "feedforward_sensor_offline"
    assert tracker.missing_count() == 1


def test_feedforward_signal_first_cycle_has_no_history():
    limits = default_config().steam
    tracker = PowerSignalTracker(limits.ff_power_lead_s, limits.ff_signal_max_age_s)
    first = tracker.sample(STEADY_POWER_KW, 0.0, STEP_S)
    assert first.available is False
    assert first.reason == "feedforward_no_history"
    second = tracker.sample(STEADY_POWER_KW, STEP_S, STEP_S)
    assert second.available is True
    assert second.slope_kw_s == pytest.approx(0.0, abs=1e-9)


def test_mode_rules_are_fixed_and_ordered():
    assert [name for name, _ in MODE_RULES] == [
        "signal_missing",
        "pressure_excursion",
        "plant_not_online",
        "ff_release_hysteresis",
        "ff_enter",
        "feedback_default",
    ]


def test_power_drop_preempts_bypass_before_pressure_deviates():
    """云遮：压力还没偏，前馈先开旁路、先压负荷。"""

    system, now = build_settled_system()
    assert system.bypass.is_open() is False
    report = system.update(
        9000.0, STEP_S, now=now, dispatch_target_mw=STEADY_LOAD_MW, online=True
    )
    decision = report["decision"]
    assert decision["mode"] == FEEDFORWARD
    assert decision["mode_rule"] == "ff_enter"
    assert decision["direction"] == "down"
    assert decision["bypass"]["action"] == "open"
    assert decision["bypass"]["reason"] == "feedforward_power_down_preempt"
    assert system.bypass.is_open() is True
    # 前馈抢先开旁路这一步，主汽门仍保持开启，且压力尚未越带。
    assert system.turbine.main_stop_open() is True
    assert report["settled"] is True
    assert system.supervisor.ff_preemptive_opens() == 1


def test_feedforward_holds_pressure_through_cloud_cover():
    """整段云遮过程压力被兜住，不产生越限，主汽门始终不关。"""

    system, now = build_settled_system()
    peak = system.pressure.value_mpa()
    modes = []
    for _ in range(40):
        report = system.update(
            9000.0, STEP_S, now=now, dispatch_target_mw=STEADY_LOAD_MW, online=True
        )
        now += STEP_S
        peak = max(peak, report["pressure_mpa"])
        modes.append(report["decision"]["mode"])
    target = default_config().steam.pressure_target_mpa
    assert peak <= target + 2.0 * default_config().steam.pressure_band_mpa
    assert system.pressure.exceedances() == 0
    assert system.turbine.main_stop_open() is True
    assert FEEDFORWARD in modes  # 确实抢先过，不是纯反馈


def test_missing_signal_falls_back_to_feedback():
    """前馈缺失时退回反馈：不抢先，按调度给负荷，旁路不按前馈开。"""

    system, now = build_settled_system()
    report = system.update(
        9000.0,
        STEP_S,
        now=now,
        dispatch_target_mw=STEADY_LOAD_MW,
        online=True,
        feedforward_available=False,
    )
    decision = report["decision"]
    assert decision["mode"] == FEEDBACK
    assert decision["mode_rule"] == "signal_missing"
    assert decision["bypass"]["action"] in ("hold",)
    assert system.bypass.is_open() is False
    assert decision["load"]["target_mw"] == pytest.approx(STEADY_LOAD_MW)
    assert system.pressure.settled() is True


def test_pressure_excursion_forces_feedback_even_with_healthy_signal():
    system, now = build_settled_system()
    system.disturb(2.0)
    report = system.update(
        9000.0, STEP_S, now=now, dispatch_target_mw=STEADY_LOAD_MW, online=True
    )
    assert report["decision"]["mode"] == FEEDBACK
    assert report["decision"]["mode_rule"] == "pressure_excursion"


def test_offline_plant_uses_feedback():
    limits = default_config().steam
    supervisor = SteamSupervisor(limits)
    signal = PowerSignalTracker(limits.ff_power_lead_s, limits.ff_signal_max_age_s)
    signal.set_available(True)
    signal.sample(STEADY_POWER_KW, 0.0, STEP_S)
    down = signal.sample(9000.0, STEP_S, STEP_S)
    decision = supervisor.decide(
        signal=down,
        pressure_mpa=limits.pressure_target_mpa,
        dispatch_target_mw=STEADY_LOAD_MW,
        bypass_open=False,
        main_stop_open=False,
        online=False,
    )
    assert decision.mode == FEEDBACK
    assert decision.mode_rule == "plant_not_online"


def test_bypass_opens_on_feedback_pressure_threshold():
    """反馈兜底：压力越过开启阈值才按偏差开旁路。"""

    limits = default_config().steam
    supervisor = SteamSupervisor(limits)
    signal = PowerSignalTracker(limits.ff_power_lead_s, limits.ff_signal_max_age_s)
    signal.set_available(False)
    missing = signal.sample(STEADY_POWER_KW, 0.0, STEP_S)
    high = limits.pressure_target_mpa + limits.ff_bypass_open_band * limits.pressure_band_mpa
    decision = supervisor.decide(
        signal=missing,
        pressure_mpa=high,
        dispatch_target_mw=STEADY_LOAD_MW,
        bypass_open=False,
        main_stop_open=True,
        online=True,
    )
    assert decision.mode == FEEDBACK
    assert decision.bypass.action == "open"
    assert decision.bypass.reason == "feedback_pressure_over_threshold"


def test_decision_executed_and_reported_are_same_source(online):
    """每周期返回的决策与执行、状态快照口径一致，并落审计。"""

    online.clock.advance(1.0)
    for _ in range(5):
        report = online.cycle(dt=2.0)
    decision = report["control_decision"]
    steam_status = online.status()["steam"]
    assert steam_status["last_decision"]["cycle"] == decision["cycle"]
    assert steam_status["last_decision"]["mode_rule"] == decision["mode_rule"]
    audit = [
        entry
        for entry in online.audit.entries(limit=50)
        if entry["action"] == "steam_control_decision"
    ]
    assert audit, "每轮控制决策必须落审计"
    latest = audit[0]["payload"]
    assert latest["mode"] == decision["mode"]
    assert latest["bypass_action"] == decision["bypass"]["action"]
    assert latest["bypass_reason"] == decision["bypass"]["reason"]
    assert latest["cycle"] == decision["cycle"]


def test_rule_trace_records_why_bypass_opened(online):
    """规则求值轨迹完整保留，可逐轮复盘为什么开旁路。"""

    online.clock.advance(1.0)
    for _ in range(300):
        online.cycle(dt=2.0)
    # 直接在蒸汽系统上制造一次云遮。
    report = online.steam.update(
        9000.0, 2.0, now=online.clock.now() + 600.0, dispatch_target_mw=20.0, online=True
    )
    trace = report["decision"]["rules"]
    rules = [item["rule"] for item in trace]
    # 命中规则之前的每条规则都按固定顺序被求值并留痕；首条命中即决定模式。
    assert rules[:5] == [name for name, _ in MODE_RULES][:5]
    assert "ff_enter" in rules
    ff_enter = next(item for item in trace if item["rule"] == "ff_enter")
    assert ff_enter["hit"] is True
    assert report["decision"]["bypass"]["reason"] == "feedforward_power_down_preempt"
