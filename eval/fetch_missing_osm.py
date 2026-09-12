import urllib.request
import urllib.parse
import json
from pathlib import Path

def fetch_box(bbox_str: str):
    query = f"""[out:json][timeout:60];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential"]({bbox_str});
);
out body geom qt;
"""
    url = "https://overpass-api.de/api/interpreter"
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "iNAV-OSM-Downloader/2.0"})
    print(f"Querying Overpass for bbox: {bbox_str}...")
    with urllib.request.urlopen(req, timeout=60) as resp:
        content = resp.read()
        d = json.loads(content)
        ways = [e for e in d.get("elements", []) if e.get("type") == "way"]
        print(f"Downloaded {len(content)} bytes, {len(ways)} road ways for {bbox_str}")
        return ways

def main():
    existing_p = Path("data/osm_uk_test_roads.json")
    all_ways = []
    seen_ids = set()
    if existing_p.exists():
        with open(existing_p, "r", encoding="utf-8") as f:
            d = json.load(f)
            for e in d.get("elements", []):
                if e.get("type") == "way" and e.get("id") not in seen_ids:
                    seen_ids.add(e.get("id"))
                    all_ways.append(e)
        print(f"Loaded {len(all_ways)} existing ways from {existing_p}")

    boxes = [
        "52.34,-2.08,52.62,-1.60",  # Birmingham / M42 / M6 south (vw14b, vw14c)
        "52.20,-1.61,52.62,-1.10",  # Coventry / Rugby / M6 / M1 (vw16a, vw16b, vw17, vw2 north)
        "51.95,-1.30,52.20,-0.70",  # Milton Keynes / M1 south (vw2 south, vw3, vw4 south)
    ]

    for bbox in boxes:
        try:
            ways = fetch_box(bbox)
            for w in ways:
                if w.get("id") not in seen_ids:
                    seen_ids.add(w.get("id"))
                    all_ways.append(w)
        except Exception as e:
            print(f"Failed for bbox {bbox}: {e}")

    out_p = Path("data/osm_uk_all_test_roads.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({"elements": all_ways}, f)
    print(f"Successfully saved {len(all_ways)} unique road ways to {out_p}")

if __name__ == "__main__":
    main()
