import urllib.request
import json

url = 'https://tiles.openfreemap.org/styles/positron'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read().decode('utf-8'))

print(f"Positron layers count: {len(data.get('layers', []))}")
with open('android/app/src/main/assets/voyager_light_style.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)
print("Saved android/app/src/main/assets/voyager_light_style.json successfully!")
