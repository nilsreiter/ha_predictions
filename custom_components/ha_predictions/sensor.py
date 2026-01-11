"""Sensor platform for integration_blueprint."""

from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.components.sensor import SensorEntity, SensorEntityDescription

from .const import ET_PERFORMANCE_SENSOR, MSG_TRAINING_DONE, MSG_DATASET_CHANGED
from .entity import HAPredictionEntity

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddEntitiesCallback

    from .coordinator import HAPredictionUpdateCoordinator
    from .data import HAPredictionConfigEntry


async def async_setup_entry(
    hass: HomeAssistant,  # noqa: ARG001 Unused function argument: `hass`
    entry: HAPredictionConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the sensor platform."""
    async_add_entities(
        [
            PredictionPerformanceSensor(
                coordinator=entry.runtime_data.coordinator,
                entity_description=SensorEntityDescription(
                    key="prediction_performance",
                    name="Prediction Performance",
                    icon="mdi:percent-box-outline",
                    suggested_display_precision=1,
                    native_unit_of_measurement="%",
                ),
            ),
            DatasetSensor(
                coordinator=entry.runtime_data.coordinator,
                entity_description=SensorEntityDescription(
                    key="dataset_size",
                    name="Dataset size",
                    icon="mdi:database",
                    suggested_display_precision=0,
                ),
            ),
        ]
    )


class DatasetSensor(HAPredictionEntity, SensorEntity):
    def __init__(
        self,
        coordinator: HAPredictionUpdateCoordinator,
        entity_description: SensorEntityDescription,
    ):
        super().__init__(coordinator)
        self.entity_description = entity_description
        self._attr_unique_id = self.coordinator.config_entry.entry_id + "-dataset-size"
        coordinator.register(self)

    @property
    def native_value(self) -> float | None:
        """Return the native value of the sensor."""
        if self.coordinator.dataset_size is not None:
            return self.coordinator.dataset_size
        return None

    @property
    def should_poll(self) -> bool:
        return False

    def notify(self, msg: str) -> None:
        if msg is MSG_DATASET_CHANGED:
            self.schedule_update_ha_state()


class PredictionPerformanceSensor(HAPredictionEntity, SensorEntity):
    def __init__(
        self,
        coordinator: HAPredictionUpdateCoordinator,
        entity_description: SensorEntityDescription,
    ):
        super().__init__(coordinator)
        self.entity_description = entity_description
        self._attr_unique_id = self.coordinator.config_entry.entry_id + "-accuracy"
        coordinator.register(self)

    @property
    def native_value(self) -> float | None:
        """Return the native value of the sensor."""
        if self.coordinator.accuracy is not None:
            return self.coordinator.accuracy * 100
        return None

    @property
    def available(self) -> bool:
        return self.coordinator.accuracy is not None

    @property
    def should_poll(self) -> bool:
        return False

    def notify(self, msg: str) -> None:
        if msg is MSG_TRAINING_DONE:
            self.schedule_update_ha_state()
