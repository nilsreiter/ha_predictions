"""Button platform for ha_predictions."""

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
    """
    Button entity for manually storing training instances.

    This button allows users to manually capture the current state of all
    configured entities and store it as a training sample in the dataset.
    """

    def __init__(
        self,
        coordinator: HAPredictionUpdateCoordinator,
        entity_description: ButtonEntityDescription,
    ) -> None:
        """
        Initialize the store instance button.

        Args:
            coordinator: The coordinator managing the prediction data and training.
            entity_description: The entity description containing button metadata.

        """
        super().__init__(coordinator)
        self.entity_description = entity_description

    async def async_press(self) -> None:
        """
        Handle the button press.

        Triggers the coordinator to collect and store the current state of all
        configured entities as a training sample.
        """
        self.coordinator.collect()

    @cached_property
    def unique_id(self) -> str | None:
        """
        Return a unique ID for this entity.

        Returns:
            A unique identifier string combining the config entry ID with a suffix.

        """
        return self.coordinator.config_entry.entry_id + "-store-instance"


class RunTrainingButton(HAPredictionEntity, ButtonEntity):
    """
    Button entity for triggering model training.

    This button initiates the training process for the prediction model using
    the collected dataset. It is only available when sufficient training data
    has been collected.
    """

    def __init__(
        self,
        coordinator: HAPredictionUpdateCoordinator,
        entity_description: ButtonEntityDescription,
    ) -> None:
        """
        Initialize the run training button.

        Args:
            coordinator: The coordinator managing the prediction data and training.
            entity_description: The entity description containing button metadata.

        """
        super().__init__(coordinator)
        self.entity_description = entity_description
        coordinator.register(self)

    async def async_press(self) -> None:
        """
        Handle the button press.

        Triggers the coordinator to train the prediction model using the
        collected dataset. This is an asynchronous operation that may take
        some time depending on the dataset size.
        """
        await self.coordinator.train()

    @cached_property
    def unique_id(self) -> str | None:
        """
        Return a unique ID for this entity.

        Returns:
            A unique identifier string combining the config entry ID with a suffix.

        """
        return self.coordinator.config_entry.entry_id + "-run-training"

    @property
    def available(self) -> bool:
        """
        Return whether the button is available.

        The button is only available when the coordinator has sufficient
        training data and is ready to perform training.

        Returns:
            True if training can be performed, False otherwise.

        """
        return self.coordinator.training_ready

    def notify(self, msg: str) -> None:
        """
        Handle notifications from the coordinator.

        This method is called when the coordinator sends notifications about
        state changes that may affect this entity.

        Args:
            msg: The notification message from the coordinator.

        """
        if msg == MSG_DATASET_CHANGED:
            self.schedule_update_ha_state()
