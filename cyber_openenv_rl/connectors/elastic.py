from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..models import CyberObservation, RewardSignal
from .base import BaseConnector

logger = logging.getLogger(__name__)

class ElasticConnector(BaseConnector):
    """Elasticsearch SIEM connector using elasticsearch-py."""

    def __init__(
        self,
        hosts: List[str],
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        index: str = "logs-*",
        query_body: Optional[Dict[str, Any]] = None,
    ):
        self.hosts = hosts
        self.api_key = api_key
        self.username = username
        self.password = password
        self.index = index
        self.query_body = query_body or {"query": {"match_all": {}}, "size": 20}
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from elasticsearch import Elasticsearch
                if self.api_key:
                    self._client = Elasticsearch(self.hosts, api_key=self.api_key)
                elif self.username and self.password:
                    self._client = Elasticsearch(self.hosts, basic_auth=(self.username, self.password))
                else:
                    self._client = Elasticsearch(self.hosts)
            except ImportError:
                logger.error("elasticsearch not installed. Please run 'pip install elasticsearch'")
                raise
        return self._client

    async def fetch_telemetry(self) -> Dict[str, Any]:
        client = self._get_client()
        try:
            # In a real implementation, we would use the async client
            response = client.search(index=self.index, body=self.query_body)
            return response.body
        except Exception as e:
            logger.error(f"Error fetching Elastic telemetry: {e}")
            return {"hits": {"hits": []}, "error": str(e)}

    def transform(self, raw_data: Dict[str, Any]) -> CyberObservation:
        # Map Elastic hits to CyberObservation
        hits = raw_data.get("hits", {}).get("hits", [])
        alerts = [hit.get("_source", {}).get("message", "Unknown event") for hit in hits]
        
        return CyberObservation(
            task_id="elastic_live",
            host_compromise={},
            host_isolation={},
            service_status={},
            ids_alerts=alerts[:10],
            traffic_anomaly_score=0.2,
            active_incidents=[],
            reward_signal=RewardSignal(),
            step_budget_remaining=999,
            available_defender_actions=["block_ip", "isolate_node", "scan_host", "patch_service", "ignore"],
            reward=0.0,
            done=False,
            metadata={"source": "elastic", "hosts": self.hosts},
        )
