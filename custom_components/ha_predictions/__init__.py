"""
Custom integration to integrate integration_blueprint with Home Assistant.

For more details about this integration, please refer to
https://github.com/nilsreiter/ml
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from homeassistant.const import Platform
from homeassistant.core import callback
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.loader import async_get_loaded_integration

from .const import CONF_FEATURE_ENTITY, CONF_TARGET_ENTITY, DOMAIN, LOGGER
from .coordinator import HAPredictionUpdateCoordinator
from .data import HAPredictionData

if TYPE_CHECKING:
    from homeassistant.core import Event, EventStateChangedData, HomeAssistant

    from .data import HAPredictionConfigEntry

PLATFORMS: list[Platform] = [
    Platform.BUTTON,
    Platform.SENSOR,
    Platform.SELECT,
    #    Platform.BINARY_SENSOR,
    #    Platform.SWITCH,
]


# https://developers.home-assistant.io/docs/config_entries_index/#setting-up-an-entry
async def async_setup_entry(
    hass: HomeAssistant,
    entry: HAPredictionConfigEntry,
) -> bool:
    """Set up this integration using UI."""
    coordinator = HAPredictionUpdateCoordinator(
        hass,
        logger=LOGGER,
        name=DOMAIN,
        update_interval=timedelta(hours=1),
    )
    entry.runtime_data = HAPredictionData(
        integration=async_get_loaded_integration(hass, entry.domain),
        coordinator=coordinator,
        datafile=Path(hass.config.config_dir, DOMAIN, entry.entry_id + ".csv"),
    )
    await hass.async_add_executor_job(coordinator.read_table)
    await coordinator.async_config_entry_first_refresh()
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entities_for_subscription = [entry.data[CONF_TARGET_ENTITY]] + entry.data[
        CONF_FEATURE_ENTITY
    ]

    @callback
    def state_changed(event: Event[EventStateChangedData]) -> None:
        coordinator.state_changed(event)

    unsub = async_track_state_change_event(
        hass, entity_ids=entities_for_subscription, action=state_changed
    )

    entry.async_on_unload(entry.add_update_listener(async_reload_entry))
    return True


async def async_unload_entry(
    hass: HomeAssistant,
    entry: HAPredictionConfigEntry,
) -> bool:
    """Handle removal of an entry."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def async_reload_entry(
    hass: HomeAssistant,
    entry: HAPredictionConfigEntry,
) -> None:
    """Reload config entry."""
    await hass.config_entries.async_reload(entry.entry_id)
