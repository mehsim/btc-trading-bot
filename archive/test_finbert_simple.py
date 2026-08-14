
import sys
import os
import requests
import xml.etree.ElementTree as ET

print("1. Importing pipeline from transformers...")
from transformers import pipeline

print("2. Fetching RSS feed...")
url = "https://cointelegraph.com/rss"
headers = {"User-Agent": "Mozilla/5.0"}
try:
    res = requests.get(url, headers=headers, timeout=10)
    print(f"   Status code: {res.status_code}")
    if res.status_code == 200:
        xml_content = res.content.decode("utf-8")
        root = ET.fromstring(xml_content)
        titles = []
        for item in root.findall(".//item"):
            title_elem = item.find("title")
            if title_elem is not None and title_elem.text:
                titles.append(title_elem.text.strip())
        titles = titles[:10]
        print(f"3. Extracted {len(titles)} titles:")
        for t in titles:
            print(f"   - {t}")
        
        print("4. Initializing sentiment-analysis pipeline...")
        sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=-1)
        
        print("5. Running inference on titles...")
        results = sentiment_pipeline(titles)
        print("6. Results:")
        for t, r in zip(titles, results):
            print(f"   {t} -> {r}")
except Exception as e:
    print(f"Error occurred: {e}")
