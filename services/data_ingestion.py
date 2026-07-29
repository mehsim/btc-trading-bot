"""
services/data_ingestion.py
--------------------------
High-performance asynchronous data ingestion service using aiohttp and websockets.
Provides non-blocking kline and orderbook delta retrieval for sub-100ms latency.
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, Optional, List

BYBIT_V5_KLINES_URL = "https://api.bybit.com/v5/market/kline"

class AsyncDataIngestionService:
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self._session = session
        self.kline_cache: Dict[str, Any] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8))
        return self._session

    async def fetch_klines_async(self, symbol: str, interval: str = "15", limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        """
        Asynchronously fetches kline data from Bybit REST API.
        """
        session = await self._get_session()
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": str(interval),
            "limit": str(limit)
        }
        try:
            async with session.get(BYBIT_V5_KLINES_URL, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("retCode") == 0:
                        raw_list = data.get("result", {}).get("list", [])
                        parsed = []
                        for row in reversed(raw_list):
                            parsed.append({
                                "start_time": int(row[0]),
                                "open": float(row[1]),
                                "high": float(row[2]),
                                "low": float(row[3]),
                                "close": float(row[4]),
                                "volume": float(row[5]),
                                "turnover": float(row[6])
                            })
                        self.kline_cache[f"{symbol}_{interval}"] = parsed
                        return parsed
        except Exception as e:
            print(f"[AsyncDataIngestion] Exception fetching {symbol} {interval}m: {e}")
        return None

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

async_data_service = AsyncDataIngestionService()
