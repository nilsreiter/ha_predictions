"""Custom types for integration_blueprint."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from custom_components.ha_predictions.entity import DeviceInfo

if TYPE_CHECKING:
    from pathlib import Path

    from homeassistant.config_entries import ConfigEntry
    from homeassistant.loader import Integration

    from .coordinator import HAPredictionUpdateCoordinator


type HAPredictionConfigEntry = ConfigEntry[HAPredictionData]


@dataclass
class HAPredictionData:
    """Data for the Blueprint integration."""

    coordinator: HAPredictionUpdateCoordinator
    integration: Integration
    datafile: Path
    device_info: DeviceInfo
    unsubscribe: list[Callable]
    # target_entity: str
    # feature_entities: list[str]
