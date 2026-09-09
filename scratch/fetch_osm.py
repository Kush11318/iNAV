import urllib.request
import urllib.parse
import json

lat = 22.730227
lon = 75.833604
query = f"""
[out:json][timeout:25];
way["highway"](around:1200,{lat},{lon});
out geom;
"""

data = urllib.parse.urlencode({'data': query.strip()}).encode('utf-8')
req = urllib.request.Request(
    'https://overpass-api.de/api/interpreter',
    data=data,
    headers={'User-Agent': 'iNAV-Precache/1.0'}
)

print(f"Fetching roads around ({lat}, {lon})...")
with urllib.request.urlopen(req, timeout=25) as resp:
    raw = resp.read()
    parsed = json.loads(raw.decode('utf-8'))
    elems = parsed.get('elements', [])
    print(f"Fetched {len(elems)} road ways from OSM Overpass")
    with open('android/app/src/main/assets/osm_roads_cache.json', 'wb') as f:
        f.write(raw)
    print(f"Wrote {len(raw)} bytes to android/app/src/main/assets/osm_roads_cache.json")
