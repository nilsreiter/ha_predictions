from __future__ import annotations

from typing import TYPE_CHECKING

from homeassistant.components.select import SelectEntity, SelectEntityDescription

from .const import OP_MODE_PROD, OP_MODE_TRAIN
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
    """Set up the select platform."""
    async_add_entities(
        [
            SelectModeEntity(
                coordinator=entry.runtime_data.coordinator,
                entity_description=SelectEntityDescription(
                    key="operation_mode",
                    name="Mode",
                    icon="mdi:form-dropdown",
                ),
            ),
        ]
    )


class SelectModeEntity(HAPredictionEntity, SelectEntity):
    _attr_options: list[str] = ["Training", "Application"]
    _attr_current_option: str = "Training"

    def __init__(
        self,
        coordinator: HAPredictionUpdateCoordinator,
        entity_description: SelectEntityDescription,
    ):
        super().__init__(coordinator)
        self.entity_description = entity_description

    async def async_select_option(self, option: str) -> None:
        """Change the selected option."""
        if option == self._attr_options[0]:
            self.coordinator.set_operation_mode(OP_MODE_TRAIN)
        elif option == self._attr_options[1]:
            self.coordinator.set_operation_mode(OP_MODE_PROD)
