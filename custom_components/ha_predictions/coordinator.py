"""DataUpdateCoordinator for ha_predictions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator

from .const import (
    CONF_FEATURE_ENTITY,
    CONF_TARGET_ENTITY,
    MIN_DATASET_SIZE,
    MSG_DATASET_CHANGED,
    MSG_PREDICTION_MADE,
    MSG_TRAINING_DONE,
    OP_MODE_PROD,
    OP_MODE_TRAIN,
)
from .ml.model import Model

if TYPE_CHECKING:
    from types import NoneType

    from homeassistant.core import Event, EventStateChangedData

    from .data import HAPredictionConfigEntry
    from .entity import HAPredictionEntity


class HAPredictionUpdateCoordinator(DataUpdateCoordinator):
    """Class to manage training/testing the model."""

    config_entry: HAPredictionConfigEntry

    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Initialize the coordinator."""
        super().__init__(*args, **kwargs)
        # Initialize instance variables to avoid sharing between coordinator instances
        self.accuracy: float | NoneType = None
        self.entity_registry: list[HAPredictionEntity] = []
        # TODO: Use only numpy array as dataset representation
        self.dataset: pd.DataFrame | NoneType = None
        self.dataset_size: int = 0
        self.model: Model = Model(self.logger)
        self.operation_mode: str = OP_MODE_TRAIN
        self.training_ready: bool = False
        self.current_prediction: tuple[str, float] | NoneType = None

    async def _async_update_data(self) -> Any:
        """Update data via library."""

    def register(self, entity: HAPredictionEntity) -> None:
        """Register an entity to be notified on changes."""
        self.entity_registry.append(entity)

    def remove_listeners(self) -> None:
        """Remove all listeners."""
        self.entity_registry.clear()

    def set_operation_mode(self, mode: str) -> None:
        """Set the operation mode."""
        if mode != self.operation_mode:
            self.operation_mode = mode
            self.logger.info("Operation mode has been changed to %s", mode)

    def state_changed(self, event: Event[EventStateChangedData]) -> None:
        """Handle state changes of monitored entities."""
        new_state = event.data["new_state"]
        old_state = event.data["old_state"]

        # Only act if state actually changed
        if old_state and new_state and old_state.state != new_state.state:
            self.logger.info("Detecting changed states, storing current instance.")
            # Schedule data collection and prediction in executor to avoid
            # blocking event loop
            self.hass.async_create_task(self._async_collect_and_predict())

    async def _async_collect_and_predict(self) -> None:
        """
        Collect data and make prediction.

        Runs blocking operations in executor to avoid blocking event loop.
        """
        # Run blocking operations in executor
        await self.hass.async_add_executor_job(self._collect_data)

        # Notify entities on main event loop
        for entity in self.entity_registry:
            entity.notify(MSG_DATASET_CHANGED)

        # Make prediction if model is ready
        if self.model.prediction_ready and self.dataset is not None:
            self.logger.info("Making new prediction after state change.")
            await self._async_make_prediction()

    async def async_collect(self) -> None:
        """
        Collect the current state as a new data point.

        This is an async wrapper that runs blocking operations in executor.
        """
        # Run blocking operations in executor
        await self.hass.async_add_executor_job(self._collect_data)

        # Notify entities on main event loop
        for entity in self.entity_registry:
            entity.notify(MSG_DATASET_CHANGED)

    def _collect_data(self) -> None:
        """
        Collect the current state (blocking operations).

        Runs in executor to avoid blocking event loop.
        """
        xy = self._get_states_for_entities()
        if self.dataset is None:
            self._initialize_dataframe()
        if self.dataset is not None:
            self.dataset.loc[len(self.dataset)] = xy
            self.dataset_size = self.dataset.shape[0]
            self.logger.info(self.dataset)
        self.training_ready = self.dataset_size >= MIN_DATASET_SIZE

    async def _async_make_prediction(self) -> None:
        """
        Make a prediction.

        Runs blocking operations in executor to avoid blocking event loop.
        """
        # Run blocking operations in executor
        prediction = await self.hass.async_add_executor_job(self._compute_prediction)

        # Update state and notify entities on main event loop
        if prediction is not None:
            self.current_prediction = prediction
            self.logger.info("New prediction: %s", str(self.current_prediction))
            for entity in self.entity_registry:
                entity.notify(MSG_PREDICTION_MADE)

    def _factorize_with_numpy(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Factorize categorical columns using numpy.unique and convert to numpy array.
        
        Args:
            df: DataFrame to factorize
            
        Returns:
            Tuple of (numpy_array, factors_dict)
        """
        # Convert DataFrame to numpy array
        data_array = df.to_numpy()
        factors: dict[str, Any] = {}
        
        # Get column names
        col_names = df.columns.tolist()
        
        # Factorize each column that contains strings/objects
        for col_idx, col_name in enumerate(col_names):
            column_data = data_array[:, col_idx]
            
            # Check if column contains non-numeric data
            if column_data.dtype == object or not np.issubdtype(column_data.dtype, np.number):
                # Use numpy.unique to get unique values and their indices
                unique_values, inverse_indices = np.unique(column_data, return_inverse=True)
                # Store the unique values for later decoding
                factors[col_name] = unique_values
                # Replace column with encoded indices
                data_array[:, col_idx] = inverse_indices
        
        # Convert to appropriate numeric type
        return data_array.astype(float), factors

    def _encode_instance(self, instance_df: pd.DataFrame) -> np.ndarray:
        """
        Encode an instance using stored factors from the model.
        
        Args:
            instance_df: DataFrame with a single instance to encode
            
        Returns:
            Encoded numpy array
        """
        # Convert to numpy
        instance_array = instance_df.to_numpy()
        col_names = instance_df.columns.tolist()
        
        # Apply factorization using stored factors
        for col_idx, col_name in enumerate(col_names):
            if col_name in self.model.factors and col_name != self.model.target_column:
                value = instance_array[0, col_idx]
                categories = self.model.factors[col_name]
                
                # Find the index of the value in categories
                try:
                    # For numpy arrays, we need to find the index
                    idx = np.where(categories == value)[0]
                    if len(idx) > 0:
                        instance_array[0, col_idx] = idx[0]
                    else:
                        # Value not found in training data, use -1
                        instance_array[0, col_idx] = -1
                except (ValueError, TypeError):
                    instance_array[0, col_idx] = -1
        
        return instance_array.astype(float)

    def _compute_prediction(self) -> tuple[str, float] | None:
        """
        Compute prediction (blocking operations).

        Runs in executor to avoid blocking event loop.
        """
        if self.dataset is None:
            return None

        instance_data = pd.DataFrame(
            columns=self.dataset.columns[:-1],
            data=[self._get_states_for_entities(include_target=False)],
        )
        self.logger.debug("Instance data for prediction: %s", str(instance_data))
        
        # Encode the instance using model's factors
        encoded_instance = self._encode_instance(instance_data)
        return self.model.predict(encoded_instance)

    def _initialize_dataframe(self) -> NoneType:
        """Initialize empty dataframe for dataset."""
        self.dataset = pd.DataFrame(
            columns=[
                *list(self.config_entry.data[CONF_FEATURE_ENTITY]),
                self.config_entry.data[CONF_TARGET_ENTITY],
            ]
        )
        self.logger.debug("Initialized new dataframe: %s", str(self.dataset))

    def _get_state_for_entity(self, entity_id: str) -> str | float | NoneType:
        """Get the state for a given entity_id."""
        if state := self.hass.states.get(entity_id=entity_id):
            try:
                return float(state.state)
            except ValueError:
                return state.state

        return None

    def _get_states_for_entities(
        self, *, include_target: bool | NoneType = True
    ) -> list[str | float | NoneType]:
        """Get the states for all monitored entities."""
        features = [
            self._get_state_for_entity(e)
            for e in self.config_entry.data[CONF_FEATURE_ENTITY]
        ]
        if include_target:
            return [
                *features,
                self._get_state_for_entity(
                    entity_id=self.config_entry.data[CONF_TARGET_ENTITY]
                ),
            ]
        return features

    # TODO: Check that this doesn't block the event loop
    def read_table(self) -> NoneType:
        """Read dataset from file."""
        self.logger.info(
            "Reading dataset from file: %s",
            str(self.config_entry.runtime_data.datafile),
        )
        if Path.exists(self.config_entry.runtime_data.datafile):
            self.dataset = pd.read_csv(
                self.config_entry.runtime_data.datafile, header=0
            )
            self.dataset_size = self.dataset.shape[0]
            [e.notify(MSG_DATASET_CHANGED) for e in self.entity_registry]

    # TODO: handle possible IO errors
    # TODO: storing on disk should happend regularly in the background
    def store_table(self, df: pd.DataFrame | NoneType) -> None:
        """Store dataset to file."""
        self.config_entry.runtime_data.datafile.parent.mkdir(
            parents=True, exist_ok=True
        )
        if df is not None:
            df.to_csv(self.config_entry.runtime_data.datafile, index=False)

    async def train(self) -> NoneType:
        """Run the training process."""
        self.logger.info("training")

        # Store and read table on/from disk
        await self.hass.async_add_executor_job(self.store_table, self.dataset)
        await self.hass.async_add_executor_job(self.read_table)

        # Run actual training
        if self.dataset is None or not self.training_ready:
            self.logger.warning(
                "Not enough data points collected yet, need at least %i, have %d",
                MIN_DATASET_SIZE,
                self.dataset_size,
            )
            return

        # Factorize the dataset using numpy.unique
        data_numpy, factors = self._factorize_with_numpy(self.dataset.copy())

        if self.operation_mode == OP_MODE_TRAIN:
            await self.hass.async_add_executor_job(
                self.model.train_eval, data_numpy
            )
            self.accuracy = self.model.accuracy
            self.logger.info("Training complete, accuracy: %f", self.accuracy)
        elif self.operation_mode == OP_MODE_PROD:
            # Get target column name
            target_column = self.dataset.columns.tolist()[-1]
            await self.hass.async_add_executor_job(
                self.model.train_final, data_numpy, factors, target_column
            )
        else:
            self.logger.error("Unknown operation mode: %s", self.operation_mode)
            return
        [e.notify(MSG_TRAINING_DONE) for e in self.entity_registry]
