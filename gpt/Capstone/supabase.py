from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import requests


@dataclass
class APIResponse:
    data: Any
    status_code: int


class Client:
    def __init__(self, supabase_url: str, supabase_key: str, *, timeout: int = 60) -> None:
        self.supabase_url = supabase_url.rstrip("/")
        self.supabase_key = supabase_key
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "apikey": supabase_key,
                "Authorization": f"Bearer {supabase_key}",
            }
        )

    def table(self, table_name: str) -> "TableQuery":
        return TableQuery(self, table_name)

    def rpc(self, function_name: str, payload: Optional[dict[str, Any]] = None) -> "RPCQuery":
        return RPCQuery(self, function_name, payload or {})


class TableQuery:
    def __init__(self, client: Client, table_name: str) -> None:
        self.client = client
        self.table_name = table_name
        self._method = "GET"
        self._params: dict[str, Any] = {}
        self._payload: Any = None
        self._headers: dict[str, str] = {}

    def select(self, columns: str) -> "TableQuery":
        self._method = "GET"
        self._params["select"] = columns
        return self

    def filter(self, column: str, operator: str, value: Any) -> "TableQuery":
        self._params[column] = f"{operator}.{value}"
        return self

    def eq(self, column: str, value: Any) -> "TableQuery":
        return self.filter(column, "eq", value)

    def order(self, column: str, *, desc: bool = False) -> "TableQuery":
        direction = "desc" if desc else "asc"
        self._params["order"] = f"{column}.{direction}"
        return self

    def upsert(self, payload: Any, *, on_conflict: Optional[str] = None) -> "TableQuery":
        self._method = "POST"
        self._payload = payload
        self._headers["Prefer"] = "resolution=merge-duplicates"
        if on_conflict:
            self._params["on_conflict"] = on_conflict
        return self

    def execute(self) -> APIResponse:
        response = self.client.session.request(
            method=self._method,
            url=f"{self.client.supabase_url}/rest/v1/{self.table_name}",
            params=self._params or None,
            json=self._payload,
            headers=self._headers or None,
            timeout=self.client.timeout,
        )
        return _build_response(response)


class RPCQuery:
    def __init__(self, client: Client, function_name: str, payload: dict[str, Any]) -> None:
        self.client = client
        self.function_name = function_name
        self.payload = payload

    def execute(self) -> APIResponse:
        response = self.client.session.post(
            f"{self.client.supabase_url}/rest/v1/rpc/{self.function_name}",
            json=self.payload,
            timeout=self.client.timeout,
        )
        return _build_response(response)


def _build_response(response: requests.Response) -> APIResponse:
    try:
        payload = response.json()
    except ValueError:
        payload = response.text or None

    if not response.ok:
        raise RuntimeError(
            f"Supabase request failed ({response.status_code}): {payload}"
        )

    return APIResponse(data=payload, status_code=response.status_code)


def create_client(supabase_url: str, supabase_key: str) -> Client:
    return Client(supabase_url, supabase_key)
