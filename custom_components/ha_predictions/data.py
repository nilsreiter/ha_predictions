"""Custom types for integration_blueprint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    # target_entity: str
    # feature_entities: list[str]
