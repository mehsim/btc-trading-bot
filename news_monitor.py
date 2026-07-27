import time
from typing import Dict, List, Tuple

class EconomicNewsMonitor:
    def __init__(self):
        self.scheduled_events: List[Dict] = []

    def set_upcoming_events(self, events: List[Dict]):
        """
        Sets scheduled news events:
        [{'title': 'CPI Release', 'timestamp': 1700000000, 'impact': 3}, ...]
        """
        self.scheduled_events = events

    def get_news_blackout_status(self) -> Tuple[bool, str]:
        """
        Rule 5: Dynamic Impact-Weighted News Blackout:
        Impact 1 (Low): 15 minutes blackout (900s)
        Impact 2 (Medium): 30 minutes blackout (1800s)
        Impact 3 / FOMC / NFP (High): 45 minutes blackout (2700s)
        Returns: (is_blackout_active, reason_message)
        """
        now = time.time()
        for event in self.scheduled_events:
            event_ts = float(event.get("timestamp", 0))
            if event_ts > 1e11:
                event_ts /= 1000.0
            impact = int(event.get("impact", 1))
            title = str(event.get("title", "Economic News"))


            if "FOMC" in title.upper() or "NFP" in title.upper() or "PAYROLL" in title.upper():
                impact = 3

            blackout_duration_sec = impact * 15 * 60  # 15m, 30m, 45m
            time_diff = abs(now - event_ts)

            if time_diff <= blackout_duration_sec:
                mins_left = round((blackout_duration_sec - time_diff) / 60.0, 1)
                return True, f"NEWS BLACKOUT ACTIVE: '{title}' (Impact Level {impact}, {mins_left}m window remaining)"

        return False, "NO_NEWS_BLACKOUT"

news_monitor = EconomicNewsMonitor()
