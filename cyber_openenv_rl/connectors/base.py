from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..models import CyberObservation


class BaseConnector(ABC):
    """Base class for all SIEM connectors."""

    @abstractmethod
    async def fetch_telemetry(self) -> Dict[str, Any]:
        """Fetch raw telemetry from the SIEM source."""
        pass

    @abstractmethod
    def transform(self, raw_data: Dict[str, Any]) -> CyberObservation:
        """Transform raw SIEM data into a CyberObservation."""
        pass

    async def get_observation(self) -> CyberObservation:
        """Helper to fetch and transform telemetry in one go."""
        raw = await self.fetch_telemetry()
        return self.transform(raw)
