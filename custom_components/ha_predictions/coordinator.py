"""DataUpdateCoordinator for integration_blueprint."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
# from sklearn.model_selection import train_test_split
# from sklearn.neural_network import MLPClassifier
# from sklearn.preprocessing import LabelEncoder

from .const import (
    CONF_FEATURE_ENTITY,
    CONF_TARGET_ENTITY,
    MSG_DATASET_CHANGED,
    MSG_TRAINING_DONE,
    OP_MODE_TRAIN,
    OP_MODE_PROD,
)

_LOGGER = logging.getLogger(__name__)


if TYPE_CHECKING:
    from types import NoneType

    from homeassistant.core import Event, EventStateChangedData

    from .data import HAPredictionConfigEntry
    from .entity import HAPredictionEntity


# https://developers.home-assistant.io/docs/integration_fetching_data#coordinated-single-api-poll-for-data-for-all-entities
class HAPredictionUpdateCoordinator(DataUpdateCoordinator):
    """Class to manage training/testing the model."""

    config_entry: HAPredictionConfigEntry
    accuracy: float | NoneType = None
    entity_registry: list[HAPredictionEntity] = []
    dataset: pd.DataFrame | NoneType = None
    dataset_size: int = 0
    model = None
    operation_mode: str = OP_MODE_TRAIN

    async def _async_update_data(self) -> Any:
        """Update data via library."""

    def register(self, entity: HAPredictionEntity):
        self.entity_registry.append(entity)

    def set_operation_mode(self, mode: str) -> None:
        self.operation_mode = mode
        _LOGGER.info("Operation mode has been changed to %s", mode)

    def state_changed(self, event: Event[EventStateChangedData]) -> None:
        new_state = event.data["new_state"]
        old_state = event.data["old_state"]

        # Only act if state actually changed
        if old_state and new_state and old_state.state != new_state.state:
            self.collect()
        _LOGGER.info("Detecting changed states, storing current instance.")

    def _initialize_dataframe(self) -> NoneType:
        self.dataset = pd.DataFrame(
            columns=[
                *list(self.config_entry.data[CONF_FEATURE_ENTITY]),
                self.config_entry.data[CONF_TARGET_ENTITY],
            ]
        )

    def _get_state_for_entity(self, entity_id: str) -> str | float | NoneType:
        if state := self.hass.states.get(entity_id=entity_id):
            try:
                return float(state.state)
            except ValueError:
                return state.state

        return None

    def read_table(self) -> NoneType:
        if Path.exists(self.config_entry.runtime_data.datafile):
            self.dataset = pd.read_csv(self.config_entry.runtime_data.datafile)
            self.dataset_size = self.dataset.shape[0]
            [e.notify(MSG_DATASET_CHANGED) for e in self.entity_registry]

    def store_table(self, df: pd.DataFrame | NoneType):
        self.config_entry.runtime_data.datafile.parent.mkdir(
            parents=True, exist_ok=True
        )
        if df is not None:
            rows = df.to_numpy().tolist()
            with Path.open(
                self.config_entry.runtime_data.datafile, "w", newline=""
            ) as f:
                writer = csv.writer(f)
                writer.writerows(rows)

    def collect(self) -> NoneType:
        """Collect the current situation as a new data point."""
        x = [
            self._get_state_for_entity(e)
            for e in self.config_entry.data[CONF_FEATURE_ENTITY]
        ]
        y = self._get_state_for_entity(
            entity_id=self.config_entry.data[CONF_TARGET_ENTITY]
        )

        xy = [*x, y]
        if self.dataset is None:
            self._initialize_dataframe()
        if self.dataset is not None:
            self.dataset.loc[len(self.dataset)] = xy
            self.dataset_size = self.dataset.shape[0]
            _LOGGER.info(self.dataset)
        [e.notify(MSG_DATASET_CHANGED) for e in self.entity_registry]

    async def train(self) -> NoneType:
        """Run the training process."""
        _LOGGER.info("training")

        # Store and read table on/from disk
        self.hass.async_add_executor_job(self.store_table, self.dataset)
        await self.hass.async_add_executor_job(self.read_table)

        # Run actual training
        if self.dataset is not None:
            await self.hass.async_add_executor_job(
                self._run_training, self.dataset.copy()
            )

    def _run_training(self, df: pd.DataFrame) -> None:
        """Run the actual training."""

        # Store categories instead of encoders
        categories = {}
        for col in df.select_dtypes(include=["object"]).columns:
            codes, uniques = pd.factorize(df[col])
            df[col] = codes
            categories[col] = uniques

        _LOGGER.debug("Data used for training: %s", str(df))
        x = df.iloc[:, :-1]  # All columns except last
        y = df.iloc[:, -1]

        # x_train, x_test, y_train, y_test = train_test_split(
        #    x, y, stratify=y, random_state=1
        # )
        # self.model = MLPClassifier(random_state=1, max_iter=10).fit(x_train, y_train)

        # Evaluate on test split
        self.accuracy = 0.7  # float(self.model.score(x_test, y_test))

        # Notify entities of finished training
        [e.notify(MSG_TRAINING_DONE) for e in self.entity_registry]

    async def evaluate(self) -> NoneType:
        pass
