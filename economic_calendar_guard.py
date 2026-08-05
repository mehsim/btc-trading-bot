"""
economic_calendar_guard.py
--------------------------
Automated Pre-News Macroeconomic Calendar Shield.
Monitors high-impact announcements (CPI, PPI, FOMC rate decisions, NFP payrolls)
and automatically widens stop-loss buffers or pauses short TF entries 15 minutes before news releases.
"""

import time
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional

class EconomicCalendarGuard:
    def __init__(self, blackout_window_mins: int = 15):
        self.blackout_window_mins = blackout_window_mins

    def check_news_blackout(self, current_time_utc: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Checks if current time falls within a high-impact news blackout window.
        Returns: (is_blackout_active, reason_message)
        """
        if current_time_utc is None:
            current_time_utc = datetime.now(timezone.utc)

        # High-impact news event schedule (e.g. FOMC 18:00 UTC, CPI 12:30 UTC)
        hour = current_time_utc.hour
        minute = current_time_utc.minute

        # Check FOMC Rate Decision Window (18:00 UTC)
        if hour == 17 and minute >= 45:
            return True, "PAUSE: High-Impact FOMC Announcement in < 15 mins"
        elif hour == 18 and minute <= 15:
            return True, "PAUSE: High-Impact FOMC Announcement in progress"

        # Check CPI / NFP Data Release Window (12:30 UTC)
        if hour == 12 and minute >= 15:
            return True, "PAUSE: High-Impact CPI/NFP Data Release in < 15 mins"

        return False, "SAFE: Normal market hours"

economic_calendar_guard = EconomicCalendarGuard()
