from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.components.button import ButtonEntity, ButtonEntityDescription
from propcache.api import cached_property

from custom_components.ha_predictions.const import MSG_DATASET_CHANGED

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
    """Set up the button platform."""
    async_add_entities(
        [
            RunTrainingButton(
                coordinator=entry.runtime_data.coordinator,
                entity_description=ButtonEntityDescription(
                    key="run_training",
                    name="Run Training",
                    icon="mdi:cog-play",
                ),
            ),
            StoreInstanceButton(
                coordinator=entry.runtime_data.coordinator,
                entity_description=ButtonEntityDescription(
                    key="store_instance",
                    name="Store Instance",
                    icon="mdi:table-plus",
                ),
            ),
        ]
    )


class StoreInstanceButton(HAPredictionEntity, ButtonEntity):
    def __init__(
        self,
        coordinator: HAPredictionUpdateCoordinator,
        entity_description: ButtonEntityDescription,
    ):
        super().__init__(coordinator)
        self.entity_description = entity_description

    async def async_press(self) -> None:
        """Handle the button press."""
        self.coordinator.collect()

    @cached_property
    def unique_id(self) -> str | None:
        return self.coordinator.config_entry.entry_id + "-store-instance"


class RunTrainingButton(HAPredictionEntity, ButtonEntity):
    def __init__(
        self,
        coordinator: HAPredictionUpdateCoordinator,
        entity_description: ButtonEntityDescription,
    ):
        super().__init__(coordinator)
        self.entity_description = entity_description
        coordinator.register(self)

    async def async_press(self) -> None:
        """Handle the button press."""
        await self.coordinator.train()

    @cached_property
    def unique_id(self) -> str | None:
        return self.coordinator.config_entry.entry_id + "-run-training"

    @property
    def available(self) -> bool:
        return self.coordinator.training_ready

    def notify(self, msg: str) -> None:
        """Handle notifications from the coordinator."""
        if msg == MSG_DATASET_CHANGED:
            self.schedule_update_ha_state()
