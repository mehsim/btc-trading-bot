"""
news_monitor.py
---------------
Economic news monitor, blackout evaluator, and social news fetchers (Reddit RSS, Crypto news RSS, X API).
"""

import os
import time
import requests
import xml.etree.ElementTree as ET
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
            if event_ts > 1e11 or event_ts > (now * 5):
                event_ts /= 1000.0
            impact = int(event.get("impact", 1))
            title = str(event.get("title", "Economic News"))

            if "FOMC" in title.upper() or "NFP" in title.upper() or "PAYROLL" in title.upper():
                impact = 3

            blackout_duration_sec = impact * 15 * 60
            half_window_sec = blackout_duration_sec / 2.0
            time_diff = abs(now - event_ts)

            if time_diff <= half_window_sec:
                mins_left = round((half_window_sec - time_diff) / 60.0, 1)
                return True, f"NEWS BLACKOUT ACTIVE: '{title}' (Impact Level {impact}, {mins_left}m window remaining)"

        return False, "NO_NEWS_BLACKOUT"


news_monitor = EconomicNewsMonitor()


def is_news_blackout(now_utc, interval) -> bool:
    """15M/30M avoid trading around major scheduled economic news (e.g., FOMC, CPI, NFP)"""
    if str(interval) not in ["15", "30"]:
        return False
    minute = now_utc.minute
    hour = now_utc.hour
    if hour in [13, 14, 18, 19]:
        if 45 <= minute or minute <= 30:
            return True
    return False


def get_reddit_posts() -> List[str]:
    """
    Fetches the top crypto/bitcoin post titles from Reddit RSS feeds.
    Does not require API keys, but does require a descriptive User-Agent.
    """
    from bybit_client import get_bybit_proxies
    subreddits = ["CryptoCurrency", "Bitcoin"]
    posts = []
    headers = {"User-Agent": "btc-trading-bot:v1.0.0 (by /u/btc-trading-bot-user)"}
    
    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/hot/.rss"
        try:
            res = requests.get(url, headers=headers, proxies=get_bybit_proxies(), timeout=10)
            if res.status_code == 200:
                xml_content = res.content.decode("utf-8")
                try:
                    root = ET.fromstring(xml_content)
                    namespace = {'atom': 'http://www.w3.org/2005/Atom'}
                    sub_posts = []
                    for entry in root.findall("atom:entry", namespace):
                        title_elem = entry.find("atom:title", namespace)
                        if title_elem is not None and title_elem.text:
                            sub_posts.append(title_elem.text.strip())
                    posts.extend(sub_posts[:5])
                    print(f"[News/Sentiment] Fetched {len(sub_posts[:5])} posts from r/{sub} RSS.")
                except Exception as parse_err:
                    print(f"[News/Sentiment] Failed parsing Reddit XML: {parse_err}")
            else:
                if res.status_code != 429:
                    print(f"[News/Sentiment] Reddit r/{sub} feed returned status code {res.status_code}")
        except Exception as e:
            print(f"[News/Sentiment] Exception fetching Reddit r/{sub} feed: {e}")
    return posts


def get_cryptopanic_posts() -> List[str]:
    """
    Fetches the top live crypto news from RSS feeds of Cointelegraph, CoinDesk, and Decrypt.
    """
    from bybit_client import get_bybit_proxies
    feeds = [
        "https://cointelegraph.com/rss",
        "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
        "https://decrypt.co/feed"
    ]
    posts = []
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    for url in feeds:
        try:
            res = requests.get(url, headers=headers, proxies=get_bybit_proxies(), timeout=8)
            if res.status_code == 200:
                xml_content = res.content
                try:
                    root = ET.fromstring(xml_content)
                    feed_posts = []
                    for item in root.findall(".//item"):
                        title_elem = item.find("title")
                        if title_elem is not None and title_elem.text:
                            feed_posts.append(title_elem.text.strip())
                    posts.extend(feed_posts[:4])
                    print(f"[News/Sentiment] Fetched {len(feed_posts[:4])} articles from RSS: {url}")
                except Exception as parse_err:
                    print(f"[News/Sentiment] Failed parsing RSS XML for {url}: {parse_err}")
            else:
                print(f"[News/Sentiment] RSS feed {url} returned status code {res.status_code}")
        except Exception as e:
            print(f"[News/Sentiment] Exception fetching RSS feed {url}: {e}")
            
    return posts[:12]


def get_x_tweets() -> List[str]:
    """
    Fetches recent tweets matching crypto/bitcoin search query.
    Requires an X Developer Bearer Token in .env (Basic or Pro subscription).
    """
    bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        return []
    
    query = os.environ.get("X_SEARCH_QUERY", "Bitcoin OR BTC lang:en -is:retweet")
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "User-Agent": "v2RecentSearchPython"
    }
    params = {
        "query": query,
        "max_results": 10,
    }
    try:
        res = requests.get(url, headers=headers, params=params, timeout=10)
        if res.status_code == 200:
            data = res.json()
            tweets = []
            for t in data.get("data", []):
                text = t.get("text")
                if text:
                    tweets.append(text.strip())
            print(f"[News/Sentiment] Fetched {len(tweets)} tweets from X API.")
            return tweets
        else:
            print(f"[News/Sentiment] X API returned status {res.status_code}")
    except Exception as e:
        print(f"[News/Sentiment] Exception fetching tweets: {e}")
    return []
