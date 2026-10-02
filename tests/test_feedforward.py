"""吸热器功率前馈：按工况抢先开旁路、信号缺失退回反馈、顺序不丢、可复盘。"""

from __future__ import annotations

import pytest

from cspplant.config import default_config
from cspplant.steam.bypass import BypassValve
from cspplant.steam.feedforward import (
    BASIS_NONE,
    BASIS_POWER_FEEDFORWARD,
    BASIS_PRESSURE_ERROR,
    REGIME_FEEDBACK,
    REGIME_FEEDFORWARD,
    FeedforwardController,
    PowerSignal,
)
from cspplant.steam.pressure import SteamPressure
from cspplant.steam.system import SteamSystem
from cspplant.steam.turbine import Turbine
from cspplant.steam.turbine_load import LoadRamp

ENTHALPY = 2600.0
KG_S_PER_MW = 0.33


def _build():
    cfg = default_config().steam
    bypass = BypassValve(cfg)
    ramp = LoadRamp(cfg)
    signal = PowerSignal(cfg.feedforward_stale_s)
    controller = FeedforwardController(
        pressure_target_mpa=cfg.pressure_target_mpa,
        pressure_band_mpa=cfg.pressure_band_mpa,
        load_max_mw=cfg.load_max_mw,
        load_rate_mw_s=cfg.load_rate_mw_s,
        bypass_lead_s=cfg.bypass_lead_s,
        min_surplus_kg_s=cfg.feedforward_min_surplus_kg_s,
    )
    system = SteamSystem(
        cfg, SteamPressure(cfg), ramp, bypass, Turbine(cfg, ramp, bypass), signal, controller
    )
    return system, cfg, signal, controller


def _warmup(system, cfg, power_kw, t=0.0, cycles=60):
    for _ in range(cycles):
        t += 1.0
        system.power_signal.ingest(power_kw, t)
        system.control_cycle(power_kw, cfg.load_max_mw, t, 1.0)
    return t


def _cycle(system, signal, power_kw, target_mw, t):
    """喂前馈采样并推进一个蒸汽周期（与平台控制循环同口径）。"""

    signal.ingest(power_kw, t)
    return system.control_cycle(power_kw, target_mw, t, 1.0)


def test_power_signal_missing_and_stale_is_partial_failure():
    cfg = default_config().steam
    signal = PowerSignal(cfg.feedforward_stale_s)
    assert signal.enabled() is True
    assert signal.available(0.0) is False  # 从无采样
    signal.ingest(1000.0, 5.0)
    assert signal.available(5.0) is True
    assert signal.available(5.0 + cfg.feedforward_stale_s + 0.1) is False  # 过期
    assert signal.set_enabled(False) is False
    assert signal.enabled() is False
    assert signal.available(6.0) is False  # 摘除后旧采样作废
    assert signal.dropouts() == 1


def test_derate_opens_bypass_on_feedforward_before_pressure_deviates():
    system, cfg, signal, controller = _build()
    power = cfg.load_max_mw * KG_S_PER_MW * ENTHALPY
    t = _warmup(system, cfg, power)
    # 降额：调度目标直接压到 0，产汽暂时不变
    opened = None
    peak = cfg.pressure_target_mpa
    for i in range(60):
        t += 1.0
        report, decision = _cycle(system, signal, power, 0.0, t)
        if opened is None and decision.opens_bypass():
            opened = (i, decision, report)
        peak = max(peak, report["pressure_mpa"])

    assert opened is not None
    _, decision, report = opened
    assert decision.regime == REGIME_FEEDFORWARD
    assert decision.bypass_basis == BASIS_POWER_FEEDFORWARD
    assert "power_surplus_ahead" in decision.reasons
    # 关键：开旁路时压力还没偏出目标带
    assert report["pressure_mpa"] <= cfg.pressure_target_mpa + cfg.pressure_band_mpa
    assert system.bypass.is_open() is True
    # 整轮压力都没被顶出带外
    assert peak <= cfg.pressure_target_mpa + cfg.pressure_band_mpa


def test_feedback_only_opens_bypass_on_pressure_error_and_overshoots_more():
    """信号缺失（部分失败）退回纯反馈：开旁路依据必须是压力偏差，且更晚。"""

    ff_system, cfg, ff_signal, _ = _build()
    fb_system, cfg2, fb_signal, _ = _build()
    power = cfg.load_max_mw * KG_S_PER_MW * ENTHALPY

    t_ff = _warmup(ff_system, cfg, power)
    t_fb = _warmup(fb_system, cfg2, power)
    fb_signal.set_enabled(False)  # 前馈测点摘掉

    ff_open_round = fb_open_round = None
    ff_peak = fb_peak = cfg.pressure_target_mpa
    for i in range(80):
        t_ff += 1.0
        t_fb += 1.0
        ff_signal.ingest(power, t_ff)
        ff_report, ff_dec = ff_system.control_cycle(power, 0.0, t_ff, 1.0)
        fb_report, fb_dec = fb_system.control_cycle(power, 0.0, t_fb, 1.0)

        assert fb_dec.regime == REGIME_FEEDBACK
        assert fb_dec.partial_failure is True
        assert fb_dec.load_preemptive is False  # 反馈不做负荷预调
        if ff_open_round is None and ff_dec.opens_bypass():
            ff_open_round = i
        if fb_open_round is None and fb_dec.opens_bypass():
            fb_open_round = (i, fb_dec, fb_report)
        ff_peak = max(ff_peak, ff_report["pressure_mpa"])
        fb_peak = max(fb_peak, fb_report["pressure_mpa"])

    assert fb_open_round is not None
    i, decision, report = fb_open_round
    assert decision.bypass_basis == BASIS_PRESSURE_ERROR
    assert report["pressure_mpa"] > cfg.pressure_target_mpa + cfg.pressure_band_mpa
    # 前馈比反馈更早开，且峰值更低
    assert ff_open_round < i
    assert ff_peak < fb_peak


def test_stale_signal_falls_back_to_feedback_without_trip():
    system, cfg, signal, controller = _build()
    power = cfg.load_max_mw * KG_S_PER_MW * ENTHALPY
    t = _warmup(system, cfg, power)
    # 这一轮吸热器没出数（不喂采样），时钟已跳过过期门限
    t += cfg.feedforward_stale_s + 1.0
    report, decision = system.control_cycle(power, 0.0, t, 1.0)
    assert decision.regime == REGIME_FEEDBACK
    assert decision.signal_ok is False
    assert decision.partial_failure is True
    assert "feedforward_signal_missing" in decision.reasons
    # 控制链仍在推进，没有甩负荷、没关主汽门
    assert system.turbine.main_stop_open() is True


def test_cloud_cover_preempts_load_without_bypass():
    system, cfg, signal, controller = _build()
    power = cfg.load_max_mw * KG_S_PER_MW * ENTHALPY
    t = _warmup(system, cfg, power)
    # 云遮：功率掉到三成，调度仍要满负荷
    cloud_power = 0.3 * power
    preempted = False
    deficit_seen = False
    for _ in range(40):
        t += 1.0
        report, decision = _cycle(system, signal, cloud_power, cfg.load_max_mw, t)
        preempted = preempted or decision.load_preemptive
        deficit_seen = deficit_seen or ("power_deficit_ahead" in decision.reasons)
    assert preempted is True
    assert deficit_seen is True
    assert system.bypass.is_open() is False  # 欠匹配靠抢先压负荷，不开旁路
    last = controller.last()
    # 执行负荷被压到产汽养得起的水平（约三成），而不是顶调度
    assert last.load_target_mw < cfg.load_max_mw


def test_controller_never_closes_main_stop():
    """前馈优化路径只开旁路，绝不碰主汽门；关主汽门只能走甩负荷编排。"""

    system, cfg, signal, _ = _build()
    power = cfg.load_max_mw * KG_S_PER_MW * ENTHALPY
    t = _warmup(system, cfg, power)
    assert system.turbine.main_stop_open() is True
    for _ in range(60):
        t += 1.0
        _cycle(system, signal, power, 0.0, t)
    # 旁路开着，但主汽门仍由甩负荷编排独占，控制器没关它
    assert system.bypass.is_open() is True
    assert system.turbine.main_stop_open() is True
    # 真正甩负荷时，先开旁路后关主汽门的顺序仍在
    report = system.shed(0.0, t + 1.0)
    actions = [step["action"] for step in report.steps]
    assert actions.index("open_bypass") < actions.index("close_main_stop")
    assert system.turbine.main_stop_open() is False


def test_bypass_closes_after_recovery_without_rejected_reopens():
    system, cfg, signal, _ = _build()
    power = cfg.load_max_mw * KG_S_PER_MW * ENTHALPY
    t = _warmup(system, cfg, power)
    for _ in range(80):
        t += 1.0
        _cycle(system, signal, power, 0.0, t)
    assert system.bypass.is_open() is True
    for _ in range(140):
        t += 1.0
        _cycle(system, signal, power, cfg.load_max_mw, t)
    assert system.bypass.is_open() is False
    assert system.pressure.settled() is True
    assert system.bypass.rejected() == 0  # 没有重复开／关被联锁挡下


def test_every_decision_is_auditable_and_identical_to_status(online):
    """每轮决策既上控制台又落审计，三处（执行/状态/审计）口径一致。"""

    online.clock.advance(1.0)
    for _ in range(5):
        online.cycle(dt=1.0)
    last = online.steam.last_decision()
    assert last is not None
    # 状态快照里的 control 与判定对象同源
    assert online.steam.status()["control"] == last.as_dict()
    assert online.status()["feedforward"]["last"] == last.as_dict()
    # 审计里能按动作检索到同一轮（seq/依据完全一致）
    entries = [
        e for e in online.audit.entries(limit=200) if e["action"] == "steam_control_decision"
    ]
    assert entries, "每轮决策必须落审计"
    audited = entries[0]  # entries 最新的在前
    assert audited["payload"]["seq"] == last.seq
    assert audited["payload"]["bypass_basis"] == last.bypass_basis
    assert audited["payload"]["regime"] == last.regime
    assert audited["payload"]["reasons"] == list(last.reasons)


def test_console_and_api_expose_decisions(online):
    online.clock.advance(1.0)
    for _ in range(3):
        online.cycle(dt=1.0)
    decisions = online.steam_decisions(limit=10)
    assert len(decisions) == 3
    # 页面与接口都拿得到逐轮判定
    from cspplant.console.api import Api
    from cspplant.console.pages import render_page

    api = Api(online)
    status, payload = api.handle("GET", "/api/steam/decisions", {}, {})
    assert status == 200
    assert len(payload["decisions"]) == 3
    html = render_page("/steam", online)
    assert "蒸汽控制台" in html and "旁路依据" in html
    # 摘掉前馈的接口
    status, payload = api.handle("POST", "/api/feedforward", {}, {"enabled": False})
    assert status == 200 and payload["enabled"] is False
    assert online.power_signal.enabled() is False


def test_basis_policy_is_deterministic():
    """同一份输入必须得到同一份判定（可逐轮重放复盘）。"""

    cfg = default_config().steam
    signal = PowerSignal(cfg.feedforward_stale_s)
    controller = FeedforwardController(
        pressure_target_mpa=cfg.pressure_target_mpa,
        pressure_band_mpa=cfg.pressure_band_mpa,
        load_max_mw=cfg.load_max_mw,
        load_rate_mw_s=cfg.load_rate_mw_s,
        bypass_lead_s=cfg.bypass_lead_s,
        min_surplus_kg_s=cfg.feedforward_min_surplus_kg_s,
    )
    kwargs = dict(
        now=1.0,
        signal=signal,
        pressure_mpa=12.0,
        production_kg_s=14.0,
        current_load_mw=30.0,
        dispatch_target_mw=0.0,
        bypass_open=False,
    )
    signal.ingest(25000.0, 1.0)
    first = controller.decide(**kwargs).as_dict()
    second = controller.decide(**kwargs).as_dict()
    # seq 会自增，其余字段必须一致
    first.pop("seq")
    second.pop("seq")
    assert first == second
    assert first["bypass_basis"] == BASIS_POWER_FEEDFORWARD
    assert first["regime"] == REGIME_FEEDFORWARD
