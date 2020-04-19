"""Support for IPX800 sensors."""
import logging

from homeassistant.components.sensor import DOMAIN
from homeassistant.helpers.entity import Entity
from homeassistant.const import STATE_OFF, STATE_ON

from . import IPX800_DEVICES, IpxDevice

_LOGGER = logging.getLogger(__name__)


def setup_platform(hass, config, add_entities, discovery_info=None):
    """Set up the IPX800 binary_sensors."""

    add_entities(
        [
            VirtualOutBinarySensor(device)
            for device in (
                item
                for item in hass.data[IPX800_DEVICES]["binary_sensor"]
                if item.get("config").get("virtualout")
            )
        ],
        True,
    )

    add_entities(
        [
            DigitalInBinarySensor(device)
            for device in (
                item
                for item in hass.data[IPX800_DEVICES]["binary_sensor"]
                if item.get("config").get("digitalin")
            )
        ],
        True,
    )


class VirtualOutBinarySensor(IpxDevice, Entity):
    """Representation of a IPX Virtual Out."""

    def __init__(self, ipx_device):
        """Initialize the IPX device."""
        super().__init__(ipx_device)
        self.virtualout = self.controller.ipx.virtualout[self.config.get("virtualout")]

    @property
    def device_class(self):
        return self._device_class

    @property
    def is_on(self):
        return bool(self._state)

    @property
    def state(self):
        return STATE_ON if self.is_on else STATE_OFF

    def update(self):
        self._state = self.virtualout.status


class DigitalInBinarySensor(IpxDevice, Entity):
    """Representation of a IPX Virtual In."""

    def __init__(self, ipx_device):
        """Initialize the IPX device."""
        super().__init__(ipx_device)
        self.digitalin = self.controller.ipx.digitalin[self.config.get("digitalin")]

    @property
    def device_class(self):
        return self._device_class

    @property
    def is_on(self):
        return bool(self._state)

    @property
    def state(self):
        return STATE_ON if self.is_on else STATE_OFF

    def update(self):
        self._state = self.digitalin.value
