"""蒸汽系统。

熔盐把热量带进蒸汽发生器，蒸汽要么进汽机做功，要么经旁路排掉。甩负荷时只有
一种安全顺序：**先开旁路、再把进汽门关小**；反过来先关主汽门会让蒸汽憋在
管系里，压力瞬间顶到安全门。这个顺序写死在 :meth:`Turbine.shed` 里。
"""

from __future__ import annotations

from .bypass import BypassValve
from .feedforward import (
    BASIS_NONE,
    BASIS_POWER_FEEDFORWARD,
    BASIS_PRESSURE_ERROR,
    REGIME_FEEDBACK,
    REGIME_FEEDFORWARD,
    FeedforwardController,
    PowerSignal,
    SteamDecision,
)
from .pressure import SteamPressure
from .snapshot import SteamSnapshot
from .system import BYPASS_CAPACITY_KG_S, STEAM_ENTHALPY_KJ_KG, SteamSystem
from .turbine import Turbine, TurbineReport
from .turbine_load import LoadRamp

__all__ = [
    "BYPASS_CAPACITY_KG_S",
    "BASIS_NONE",
    "BASIS_POWER_FEEDFORWARD",
    "BASIS_PRESSURE_ERROR",
    "REGIME_FEEDBACK",
    "REGIME_FEEDFORWARD",
    "BypassValve",
    "FeedforwardController",
    "PowerSignal",
    "SteamDecision",
    "SteamPressure",
    "SteamSnapshot",
    "SteamSystem",
    "Turbine",
    "TurbineReport",
    "LoadRamp",
    "STEAM_ENTHALPY_KJ_KG",
]
