"""DataUpdateCoordinator for ha_predictions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .const import (
    CONF_FEATURE_ENTITY,
    CONF_TARGET_ENTITY,
    LOGGER,
    MIN_DATASET_SIZE,
    MSG_DATASET_CHANGED,
    MSG_TRAINING_DONE,
    OP_MODE_TRAIN,
)
from .ml.LogisticRegression import LogisticRegression

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
    model: Any = None
    operation_mode: str = OP_MODE_TRAIN
    training_ready: bool = False

    async def _async_update_data(self) -> Any:
        """Update data via library."""

    def register(self, entity: HAPredictionEntity):
        self.entity_registry.append(entity)

    def set_operation_mode(self, mode: str) -> None:
        self.operation_mode = mode
        LOGGER.info("Operation mode has been changed to %s", mode)

    def state_changed(self, event: Event[EventStateChangedData]) -> None:
        new_state = event.data["new_state"]
        old_state = event.data["old_state"]

        # Only act if state actually changed
        if old_state and new_state and old_state.state != new_state.state:
            LOGGER.info("Detecting changed states, storing current instance.")
            self.collect()

    def _initialize_dataframe(self) -> NoneType:
        self.dataset = pd.DataFrame(
            columns=[
                *list(self.config_entry.data[CONF_FEATURE_ENTITY]),
                self.config_entry.data[CONF_TARGET_ENTITY],
            ]
        )
        LOGGER.debug("Initialized new dataframe: %s", str(self.dataset))

    def _get_state_for_entity(self, entity_id: str) -> str | float | NoneType:
        if state := self.hass.states.get(entity_id=entity_id):
            try:
                return float(state.state)
            except ValueError:
                return state.state

        return None

    def read_table(self) -> NoneType:
        if Path.exists(self.config_entry.runtime_data.datafile):
            self.dataset = pd.read_csv(
                self.config_entry.runtime_data.datafile, header=0
            )
            self.dataset_size = self.dataset.shape[0]
            [e.notify(MSG_DATASET_CHANGED) for e in self.entity_registry]

    def store_table(self, df: pd.DataFrame | NoneType):
        self.config_entry.runtime_data.datafile.parent.mkdir(
            parents=True, exist_ok=True
        )
        if df is not None:
            df.to_csv(self.config_entry.runtime_data.datafile, index=False)

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
            LOGGER.info(self.dataset)
        [e.notify(MSG_DATASET_CHANGED) for e in self.entity_registry]
        self.training_ready = self.dataset_size >= MIN_DATASET_SIZE

    async def train(self) -> NoneType:
        """Run the training process."""
        LOGGER.info("training")

        # Store and read table on/from disk
        self.hass.async_add_executor_job(self.store_table, self.dataset)
        await self.hass.async_add_executor_job(self.read_table)

        # Run actual training
        if self.dataset is not None and self.training_ready:
            await self.hass.async_add_executor_job(
                self._run_training, self.dataset.copy()
            )
        else:
            LOGGER.warning(
                "Not enough data points collected yet, need at least %i, have %d",
                MIN_DATASET_SIZE,
                self.dataset_size,
            )

    def _run_training(self, df: pd.DataFrame) -> None:
        """Run the actual training."""
        # Store categories instead of encoders
        categories = {}
        for col in df.select_dtypes(include=["object"]).columns:
            codes, uniques = pd.factorize(df[col])
            df[col] = codes
            categories[col] = uniques

        # train/test split in pure numpy
        # TODO: Stratify split based on last column
        dfn = df.to_numpy()
        np.random.Generator(np.random.PCG64()).shuffle(dfn)

        nrows = dfn.shape[0]
        test_size = max(int(nrows * 0.25), 1)
        train, test = dfn[:test_size, :], dfn[test_size:, :]
        LOGGER.debug("Data used for training: %s", str(train))
        LOGGER.debug("Data used for training: %s", str(test))

        # Split x and y
        x_train = train[:, :-1]
        y_train = train[:, -1]
        x_test = test[:, :-1]
        y_test = test[:, -1]

        self.model = LogisticRegression()
        LOGGER.debug("Training begins")
        self.model.fit(x_train, y_train)
        LOGGER.debug("Training ends, model: %s", str(self.model))

        # Evaluate on test split
        self.accuracy = self.model.score(x_test, y_test)

        # Notify entities of finished training
        [e.notify(MSG_TRAINING_DONE) for e in self.entity_registry]

    async def evaluate(self) -> NoneType:
        pass
