"""Constants for integration_blueprint."""

from logging import Logger, getLogger

LOGGER: Logger = getLogger(__package__)

DOMAIN = "ha_predictions"
ATTRIBUTION = "Data provided by http://jsonplaceholder.typicode.com/"

CONF_TARGET_ENTITY = "CONF_TARGET_ENTITY"
CONF_FEATURE_ENTITY = "CONF_FEATURE_ENTITY"

ET_PERFORMANCE_SENSOR = "PERFORMANCE_SENSOR"

MSG_TRAINING_DONE = "TRAINING_DONE"
MSG_DATASET_CHANGED = "DATASET_CHANGED"

OP_MODE_TRAIN = "TRAINING"
OP_MODE_PROD = "PRODUCTION"
