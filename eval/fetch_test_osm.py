import urllib.request
import urllib.parse
import json
from pathlib import Path

def download_osm_box(bbox_str: str, out_file: Path):
    query = f"""[out:json][timeout:60];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential"]({bbox_str});
);
out body geom qt;
"""
    url = "https://overpass-api.de/api/interpreter"
    data = urllib.parse.urlencode({"data": query}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "iNAV-OSM-Downloader/1.0"})
    print(f"Querying Overpass for bbox: {bbox_str}...")
    with urllib.request.urlopen(req, timeout=60) as resp:
        content = resp.read()
        print(f"Downloaded {len(content)} bytes")
        d = json.loads(content)
        ways = [e for e in d.get("elements", []) if e.get("type") == "way"]
        print(f"Extracted {len(ways)} road ways")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "wb") as f:
            f.write(content)
        print(f"Saved to {out_file}")

if __name__ == "__main__":
    # Bounding box for test runs vw10, vw11, vw12, vw14a: [52.19, -2.22, 52.36, -2.06]
    out_p = Path("data/osm_uk_test_roads.json")
    download_osm_box("52.19,-2.22,52.36,-2.06", out_p)
