"""吸热器热功率前馈与旁路／负荷预调策略。

汽机负荷和压力原来只在压力偏了之后才回头调：云一遮、降额一动，产汽与进汽的
差额先砸到压力上，再由反馈慢慢拉回，压力总是冲一大截。这里把吸热器侧的热功率
变化**提前**接进来，每个控制周期先做一次确定性判定，让旁路和负荷先动起来把
压力兜住。

策略按工况定死，旁路"按前馈先开还是按偏差开"不允许临场含糊：

================== ============================================================
工况                旁路依据
================== ============================================================
前馈信号缺失／过期  退回纯反馈（部分失败）：只按压力偏差开旁路，负荷不预调
前馈有效·甩负荷富余 前馈预测产汽富余且负荷正在下压 → **抢先**开旁路
                   （``power_feedforward``）
前馈有效·云遮欠匹配 抢先把负荷压到产汽养得起的值，不开旁路
前馈有效／反馈·已超压 压力偏差越带 → 按偏差开旁路（``pressure_error``）
稳态                不动
================== ============================================================

任何一轮控制器都**只开旁路、不碰主汽门**。"先开旁路、再关主汽门"的顺序仍然
只由 :meth:`cspplant.steam.turbine.Turbine.shed` 与
:meth:`cspplant.steam.turbine.Turbine.trip` 编排，前馈优化没有资格关主汽门，
顺序因此不可能被这条优化路径破坏。

每一轮产出同一个 :class:`SteamDecision`：它既是执行器唯一的动作来源，也原样
上控制台、原样落审计，三处口径一致，事后能逐轮复盘"当时为什么开旁路"。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

# 汽机负荷到进汽量的换算（kg/s per MW），与蒸汽系统口径一致。
TURBINE_KG_S_PER_MW = 0.33

# 调节模式：前馈有效 / 退回纯反馈。
REGIME_FEEDFORWARD = "feedforward"
REGIME_FEEDBACK = "feedback"

# 旁路动作依据（定死的代码，审计与控制台都按这套代码复盘）。
BASIS_NONE = "none"                    # 本轮不需要旁路
BASIS_POWER_FEEDFORWARD = "power_feedforward"  # 前馈预测甩负荷富余，抢先开
BASIS_PRESSURE_ERROR = "pressure_error"        # 压力已偏，按偏差开

# 旁路阀本轮应执行的动作。
ACTION_OPEN = "open"
ACTION_CLOSE = "close"
ACTION_HOLD = "hold"

# 原因代码。
REASON_SIGNAL_MISSING = "feedforward_signal_missing"
REASON_SURPLUS_AHEAD = "power_surplus_ahead"
REASON_DEFICIT_AHEAD = "power_deficit_ahead"
REASON_PRESSURE_HIGH = "pressure_above_band"
REASON_STEADY = "steady"
REASON_PRESSURE_RECOVERED = "pressure_recovered"
REASON_SURPLUS_CLEARED = "power_surplus_cleared"

_BASIS_TEXT = {
    BASIS_NONE: "无需旁路",
    BASIS_POWER_FEEDFORWARD: "前馈预测甩负荷产汽富余，提前开旁路接住",
    BASIS_PRESSURE_ERROR: "压力已偏高，按偏差开旁路",
}


class PowerSignal:
    """吸热器热功率前馈信号：记录最近一次采样，判定可用与过期。

    信号可能整体缺失（测点／通信被摘掉）或只是过期（该到的周期没到）。两种
    情况都属于**部分失败**——控制链不跳闸，只是退回纯反馈调节。
    """

    def __init__(self, stale_after_s: float) -> None:
        if stale_after_s <= 0.0:
            raise ValueError("信号过期时间必须为正")
        self._stale_after_s = float(stale_after_s)
        self._enabled = True
        self._power_kw: float | None = None
        self._ts: float | None = None
        self._updates = 0
        self._dropouts = 0

    @property
    def stale_after_s(self) -> float:
        return self._stale_after_s

    def ingest(self, power_kw: float, now: float) -> float:
        """接收一次热功率采样；信号被摘掉时调用方仍会调，但此处不落任何值。"""

        if self._enabled:
            self._power_kw = max(0.0, float(power_kw))
            self._ts = float(now)
            self._updates += 1
        return self.current_kw() if self._power_kw is not None else 0.0

    def set_enabled(self, enabled: bool) -> bool:
        """挂上／摘掉前馈测点。摘掉后旧采样立即作废，按缺失处理。"""

        if self._enabled and not enabled:
            self._dropouts += 1
        self._enabled = bool(enabled)
        if not enabled:
            self._power_kw = None
            self._ts = None
        return self._enabled

    def enabled(self) -> bool:
        """测点是否在挂（未被人工／检修摘掉）。"""

        return self._enabled

    def available(self, now: float) -> bool:
        """信号当前是否可用：在挂且有采样且未过期。"""

        return self._age_s(now) is not None

    def current_kw(self) -> float:
        """最近一次可信采样功率（没有采样返回 0）。"""

        return 0.0 if self._power_kw is None else self._power_kw

    def age_s(self, now: float) -> float | None:
        """距最近一次采样的秒数；无采样返回 ``None``。"""

        return self._age_s(now)

    def updates(self) -> int:
        return self._updates

    def dropouts(self) -> int:
        return self._dropouts

    def _age_s(self, now: float) -> float | None:
        if not self._enabled or self._ts is None:
            return None
        age = max(0.0, float(now) - self._ts)
        if age > self._stale_after_s:
            return None
        return age


@dataclass(frozen=True)
class SteamDecision:
    """一个控制周期的蒸汽侧判定：执行器、控制台与审计的共同唯一来源。"""

    seq: int
    regime: str
    signal_ok: bool
    partial_failure: bool
    signal_age_s: float | None
    power_kw: float
    pressure_mpa: float
    pressure_error_mpa: float
    production_kg_s: float
    predicted_surplus_kg_s: float
    current_load_mw: float
    dispatch_target_mw: float
    load_target_mw: float
    load_preemptive: bool
    bypass_action: str
    bypass_basis: str
    bypass_open_after: bool
    reasons: tuple[str, ...]

    @property
    def basis_text(self) -> str:
        """旁路依据的中文说明，供控制台直接展示。"""

        return _BASIS_TEXT.get(self.bypass_basis, self.bypass_basis)

    def opens_bypass(self) -> bool:
        """本轮是否要求（新）开旁路。"""

        return self.bypass_action == ACTION_OPEN

    def as_dict(self) -> dict:
        """转成审计／控制台共用的扁平字典（只含标量与字符串列表）。"""

        return {
            "seq": self.seq,
            "regime": self.regime,
            "signal_ok": self.signal_ok,
            "partial_failure": self.partial_failure,
            "signal_age_s": None if self.signal_age_s is None else round(self.signal_age_s, 3),
            "power_kw": round(self.power_kw, 3),
            "pressure_mpa": round(self.pressure_mpa, 3),
            "pressure_error_mpa": round(self.pressure_error_mpa, 3),
            "production_kg_s": round(self.production_kg_s, 3),
            "predicted_surplus_kg_s": round(self.predicted_surplus_kg_s, 3),
            "current_load_mw": round(self.current_load_mw, 3),
            "dispatch_target_mw": round(self.dispatch_target_mw, 3),
            "load_target_mw": round(float(self.load_target_mw), 3),
            "load_preemptive": self.load_preemptive,
            "bypass_action": self.bypass_action,
            "bypass_basis": self.bypass_basis,
            "bypass_basis_text": self.basis_text,
            "bypass_open_after": self.bypass_open_after,
            "reasons": list(self.reasons),
        }


class FeedforwardController:
    """按工况给出每轮旁路与负荷预调判定。

    判定只依赖输入信号与本对象记住的"旁路当前为何而开"，不带其他隐状态，
    因此同一份输入永远得到同一份决策，审计可逐轮重放。
    """

    def __init__(
        self,
        *,
        pressure_target_mpa: float,
        pressure_band_mpa: float,
        load_max_mw: float,
        load_rate_mw_s: float,
        bypass_lead_s: float,
        min_surplus_kg_s: float,
        history: int = 32,
    ) -> None:
        self._target_mpa = float(pressure_target_mpa)
        self._band_mpa = float(pressure_band_mpa)
        self._load_max = float(load_max_mw)
        self._load_rate = float(load_rate_mw_s)
        # 旁路提前量内，光靠压负荷能多接走多少蒸汽；富余超过它才需要抢开旁路。
        self._ramp_absorb_kg_s = float(load_rate_mw_s) * float(bypass_lead_s) * TURBINE_KG_S_PER_MW
        self._min_surplus_kg_s = float(min_surplus_kg_s)
        self._basis = BASIS_NONE
        self._seq = 0
        self._history: deque[SteamDecision] = deque(maxlen=max(1, history))

    @property
    def basis(self) -> str:
        """旁路当前为何而开（未开为 ``none``）。"""

        return self._basis

    def history(self, limit: int | None = None) -> list[dict]:
        """最近的判定（时间升序），供控制台与自检对照。"""

        items = list(self._history)
        if limit is not None:
            items = items[-max(0, limit):]
        return [item.as_dict() for item in items]

    def last(self) -> SteamDecision | None:
        return self._history[-1] if self._history else None

    def decide(
        self,
        *,
        now: float,
        signal: PowerSignal,
        pressure_mpa: float,
        production_kg_s: float,
        current_load_mw: float,
        dispatch_target_mw: float,
        bypass_open: bool,
    ) -> SteamDecision:
        """给出本轮判定。``bypass_open`` 是执行前的实际阀位。"""

        self._seq += 1
        pressure = float(pressure_mpa)
        error = pressure - self._target_mpa
        production = max(0.0, float(production_kg_s))
        current = min(max(0.0, float(current_load_mw)), self._load_max)
        dispatch = min(max(0.0, float(dispatch_target_mw)), self._load_max)

        age = signal.age_s(now)
        signal_ok = age is not None

        # 当前汽机实际抽汽下的产汽富余：>0 表示蒸汽在往管系里憋。
        surplus = production - current * TURBINE_KG_S_PER_MW
        load_reducing = dispatch < current - 1e-9

        reasons: list[str] = []
        if signal_ok:
            regime = REGIME_FEEDFORWARD
            load_target, preemptive = self._feedforward_load_target(
                production, current, dispatch, reasons
            )
            basis, action = self._feedforward_bypass(
                surplus_kg_s=surplus,
                load_reducing=load_reducing,
                error=error,
                bypass_open=bypass_open,
                reasons=reasons,
            )
        else:
            # 部分失败：退回原来的反馈调节——负荷只跟调度，不做前馈预调；
            # 旁路只认压力偏差。
            regime = REGIME_FEEDBACK
            load_target = dispatch
            preemptive = False
            reasons.append(REASON_SIGNAL_MISSING)
            basis, action = self._feedback_bypass(error, bypass_open, reasons)

        self._basis = basis
        decision = SteamDecision(
            seq=self.seq,
            regime=regime,
            signal_ok=signal_ok,
            partial_failure=not signal_ok,
            signal_age_s=age,
            power_kw=signal.current_kw(),
            pressure_mpa=pressure,
            pressure_error_mpa=error,
            production_kg_s=production,
            predicted_surplus_kg_s=surplus,
            current_load_mw=current,
            dispatch_target_mw=dispatch,
            load_target_mw=load_target,
            load_preemptive=preemptive,
            bypass_action=action,
            bypass_basis=basis,
            bypass_open_after=basis != BASIS_NONE,
            reasons=tuple(reasons),
        )
        self._history.append(decision)
        return decision

    @property
    def seq(self) -> int:
        return self._seq

    def _feedforward_load_target(
        self,
        production_kg_s: float,
        current_mw: float,
        dispatch_mw: float,
        reasons: list[str],
    ) -> tuple[float, bool]:
        """云遮导致欠匹配时，抢先把负荷压到当前产汽养得起的值。

        判定以**调度目标**为基准：产汽养不活调度目标就压到可持续负荷并一直
        压住，直到产汽重新养得活调度才松手。若以当前负荷为基准，负荷一压到
        门槛附近就会误判"不缺了"而弹回调度，形成负荷来回抖动。
        """

        sustainable_mw = production_kg_s / TURBINE_KG_S_PER_MW
        margin_mw = self._min_surplus_kg_s / TURBINE_KG_S_PER_MW
        if sustainable_mw + margin_mw < dispatch_mw:
            reasons.append(REASON_DEFICIT_AHEAD)
            return min(dispatch_mw, sustainable_mw), True
        return dispatch_mw, False

    def _feedforward_bypass(
        self,
        *,
        surplus_kg_s: float,
        load_reducing: bool,
        error: float,
        bypass_open: bool,
        reasons: list[str],
    ) -> tuple[str, str]:
        """前馈有效时的旁路判定：富余甩负荷抢先开，否则交给偏差兜底。"""

        # 抢先开：负荷正在下压，且提前量内单靠压负荷接不走这么多富余蒸汽。
        preemptive = (
            load_reducing
            and surplus_kg_s - self._ramp_absorb_kg_s >= self._min_surplus_kg_s
        )
        # 已经抢开后继续顶住：只要富余蒸汽还接不走，哪怕负荷已不继续下压
        # （例如已压到零），也不能关，关了蒸汽立刻又憋回管系。
        still_surplus = bypass_open and surplus_kg_s >= self._min_surplus_kg_s

        if preemptive or (still_surplus and self._basis == BASIS_POWER_FEEDFORWARD):
            reasons.append(REASON_SURPLUS_AHEAD)
            if not bypass_open or self._basis != BASIS_POWER_FEEDFORWARD:
                return BASIS_POWER_FEEDFORWARD, ACTION_OPEN
            return BASIS_POWER_FEEDFORWARD, ACTION_HOLD

        if error > self._band_mpa:
            # 前馈没拦住、压力已经越带：偏差兜底，旁路依据改记为偏差。
            reasons.append(REASON_PRESSURE_HIGH)
            if not bypass_open:
                return BASIS_PRESSURE_ERROR, ACTION_OPEN
            return BASIS_PRESSURE_ERROR, ACTION_HOLD

        # 既无前馈富余、压力也在带内：按各自开阀依据决定能否收回。
        # 关阀加滞回：富余已消且压力回到目标及以下才关，避免下一轮又顶开。
        if bypass_open:
            if self._basis == BASIS_POWER_FEEDFORWARD:
                reasons.append(REASON_SURPLUS_CLEARED)
                return BASIS_NONE, ACTION_CLOSE
            if self._basis == BASIS_PRESSURE_ERROR and error <= 0.0:
                reasons.append(REASON_PRESSURE_RECOVERED)
                return BASIS_NONE, ACTION_CLOSE
            return self._basis, ACTION_HOLD

        reasons.append(REASON_STEADY)
        return BASIS_NONE, ACTION_HOLD

    def _feedback_bypass(
        self, error: float, bypass_open: bool, reasons: list[str]
    ) -> tuple[str, str]:
        """纯反馈旁路：压力偏高越带才开，回到目标及以下才关。"""

        if error > self._band_mpa:
            reasons.append(REASON_PRESSURE_HIGH)
            if not bypass_open:
                return BASIS_PRESSURE_ERROR, ACTION_OPEN
            return BASIS_PRESSURE_ERROR, ACTION_HOLD
        if bypass_open and self._basis == BASIS_PRESSURE_ERROR and error <= 0.0:
            reasons.append(REASON_PRESSURE_RECOVERED)
            return BASIS_NONE, ACTION_CLOSE
        if bypass_open:
            # 前馈开着的阀在信号丢失后：压力仍高就先挂到偏差依据下兜住，
            # 压力没偏则直接收回，绝不带着来路不明的阀位继续运行。
            if error > 0.0:
                reasons.append(REASON_PRESSURE_HIGH)
                return BASIS_PRESSURE_ERROR, ACTION_HOLD
            reasons.append(REASON_PRESSURE_RECOVERED)
            return BASIS_NONE, ACTION_CLOSE
        return BASIS_NONE, ACTION_HOLD
