import json

def enrich_styles():
    # 1. Update voyager_light_style.json
    with open('android/app/src/main/assets/voyager_light_style.json', 'r', encoding='utf-8') as f:
        light = json.load(f)

    icon_match_expr = [
        'match',
        ['get', 'subclass'],
        ['fuel'], 'fuel',
        ['bakery'], 'bakery',
        ['supermarket', 'grocery'], 'grocery',
        ['clothes', 'clothing'], 'clothing_store',
        ['fast_food'], 'fast_food',
        ['cafe'], 'cafe',
        ['restaurant'], 'restaurant',
        ['bar', 'pub'], 'bar',
        ['pharmacy', 'chemist'], 'pharmacy',
        ['hospital', 'clinic'], 'hospital',
        ['bank', 'atm'], 'bank',
        ['police'], 'police',
        ['post_office', 'post'], 'post',
        ['school', 'college', 'university'], 'school',
        ['hairdresser', 'beauty'], 'hairdresser',
        ['florist'], 'florist',
        ['furniture'], 'furniture',
        ['car_repair', 'car'], 'car',
        ['hotel', 'lodging'], 'lodging',
        ['cinema', 'theatre'], 'cinema',
        ['park'], 'park',
        [
            'match',
            ['get', 'class'],
            ['fuel'], 'fuel',
            ['restaurant'], 'restaurant',
            ['fast_food'], 'fast_food',
            ['cafe'], 'cafe',
            ['bar'], 'bar',
            ['bank'], 'bank',
            ['atm'], 'bank',
            ['pharmacy'], 'pharmacy',
            ['hospital'], 'hospital',
            ['doctor'], 'doctors',
            ['police'], 'police',
            ['post'], 'post',
            ['school'], 'school',
            ['college', 'university'], 'college',
            ['lodging', 'hotel'], 'lodging',
            ['grocery'], 'grocery',
            ['cinema'], 'cinema',
            ['theatre'], 'theatre',
            ['park'], 'park',
            ['commercial', 'office'], 'commercial',
            ['place_of_worship'], 'place_of_worship',
            ['shop'], 'shop',
            ''
        ]
    ]

    text_field_expr = [
        'case',
        ['has', 'name:nonlatin'],
        ['concat', ['coalesce', ['get', 'name:latin'], ['get', 'name_en'], ['get', 'name']], '\n', ['get', 'name:nonlatin']],
        ['coalesce', ['get', 'name_en'], ['get', 'name:latin'], ['get', 'name']]
    ]

    new_poi_layers_light = [
        {
            'id': 'housenumber',
            'type': 'symbol',
            'source': 'openmaptiles',
            'source-layer': 'housenumber',
            'minzoom': 16.5,
            'layout': {
                'text-field': ['get', 'housenumber'],
                'text-font': ['Noto Sans Regular'],
                'text-size': 10
            },
            'paint': {
                'text-color': '#78716c',
                'text-halo-color': '#ffffff',
                'text-halo-width': 1.0
            }
        },
        {
            'id': 'poi_transit',
            'type': 'symbol',
            'source': 'openmaptiles',
            'source-layer': 'poi',
            'minzoom': 12,
            'filter': ['all', ['match', ['get', 'class'], ['airport', 'bus', 'rail'], True, False], ['has', 'name']],
            'layout': {
                'icon-image': ['to-string', ['get', 'class']],
                'icon-size': 0.8,
                'text-anchor': 'top',
                'text-offset': [0, 0.6],
                'text-field': text_field_expr,
                'text-font': ['Noto Sans Regular'],
                'text-max-width': 9,
                'text-size': 11
            },
            'paint': {
                'text-color': '#1e3a8a',
                'text-halo-blur': 0.5,
                'text-halo-color': '#ffffff',
                'text-halo-width': 1.5
            }
        },
        {
            'id': 'poi_fuel',
            'type': 'symbol',
            'source': 'openmaptiles',
            'source-layer': 'poi',
            'minzoom': 12.0,
            'filter': ['all', ['match', ['get', 'class'], ['fuel'], True, False], ['has', 'name']],
            'layout': {
                'icon-image': 'fuel',
                'icon-size': 0.85,
                'text-anchor': 'top',
                'text-offset': [0, 0.6],
                'text-field': text_field_expr,
                'text-font': ['Noto Sans Bold'],
                'text-max-width': 9,
                'text-size': 11.5
            },
            'paint': {
                'text-color': '#b45309',
                'text-halo-blur': 0.5,
                'text-halo-color': '#ffffff',
                'text-halo-width': 1.5
            }
        },
        {
            'id': 'poi_r1',
            'type': 'symbol',
            'source': 'openmaptiles',
            'source-layer': 'poi',
            'minzoom': 13,
            'filter': [
                'all',
                ['match', ['geometry-type'], ['MultiPoint', 'Point'], True, False],
                ['>=', ['get', 'rank'], 1],
                ['<', ['get', 'rank'], 7],
                ['!=', ['get', 'class'], 'fuel'],
                ['has', 'name']
            ],
            'layout': {
                'icon-image': icon_match_expr,
                'icon-size': 0.8,
                'text-anchor': 'top',
                'text-offset': [0, 0.6],
                'text-field': text_field_expr,
                'text-font': ['Noto Sans Regular'],
                'text-max-width': 9,
                'text-size': 11
            },
            'paint': {
                'text-color': '#334155',
                'text-halo-blur': 0.5,
                'text-halo-color': '#ffffff',
                'text-halo-width': 1.5
            }
        },
        {
            'id': 'poi_r7',
            'type': 'symbol',
            'source': 'openmaptiles',
            'source-layer': 'poi',
            'minzoom': 13.8,
            'filter': [
                'all',
                ['match', ['geometry-type'], ['MultiPoint', 'Point'], True, False],
                ['>=', ['get', 'rank'], 7],
                ['<', ['get', 'rank'], 20],
                ['!=', ['get', 'class'], 'fuel'],
                ['has', 'name']
            ],
            'layout': {
                'icon-image': icon_match_expr,
                'icon-size': 0.75,
                'text-anchor': 'top',
                'text-offset': [0, 0.6],
                'text-field': text_field_expr,
                'text-font': ['Noto Sans Regular'],
                'text-max-width': 9,
                'text-size': 10.5
            },
            'paint': {
                'text-color': '#475569',
                'text-halo-blur': 0.5,
                'text-halo-color': '#ffffff',
                'text-halo-width': 1.5
            }
        },
        {
            'id': 'poi_r20',
            'type': 'symbol',
            'source': 'openmaptiles',
            'source-layer': 'poi',
            'minzoom': 14.2,
            'filter': [
                'all',
                ['match', ['geometry-type'], ['MultiPoint', 'Point'], True, False],
                ['>=', ['get', 'rank'], 20],
                ['!=', ['get', 'class'], 'fuel'],
                ['has', 'name']
            ],
            'layout': {
                'icon-image': icon_match_expr,
                'icon-size': 0.75,
                'text-anchor': 'top',
                'text-offset': [0, 0.6],
                'text-field': text_field_expr,
                'text-font': ['Noto Sans Regular'],
                'text-max-width': 9,
                'text-size': 10
            },
            'paint': {
                'text-color': '#52525b',
                'text-halo-blur': 0.5,
                'text-halo-color': '#ffffff',
                'text-halo-width': 1.5
            }
        }
    ]

    # Adjust road name minzooms
    for l in light['layers']:
        if l['id'] == 'highway-name-minor':
            l['minzoom'] = 13.0
        elif l['id'] == 'highway-name-major':
            l['minzoom'] = 11.0

    # Remove any previous additions if re-running
    existing_ids = {layer['id'] for layer in new_poi_layers_light} | {'road_one_way_arrow', 'road_one_way_arrow_opposite'}
    light['layers'] = [l for l in light['layers'] if l['id'] not in existing_ids]

    # Insert before label_other
    insert_idx = next((i for i, l in enumerate(light['layers']) if l['id'] == 'label_other'), len(light['layers']))
    light['layers'][insert_idx:insert_idx] = new_poi_layers_light

    with open('android/app/src/main/assets/voyager_light_style.json', 'w', encoding='utf-8') as f:
        json.dump(light, f, indent=2)
    print("voyager_light_style.json enriched with", len(new_poi_layers_light), "layers. Total layers:", len(light['layers']))

    # 2. Also enrich uber_dark_style.json with dark mode colors
    with open('android/app/src/main/assets/uber_dark_style.json', 'r', encoding='utf-8') as f:
        dark = json.load(f)

    new_poi_layers_dark = []
    for l in new_poi_layers_light:
        dark_l = json.loads(json.dumps(l))
        if dark_l.get('paint'):
            if 'text-color' in dark_l['paint']:
                dark_l['paint']['text-color'] = '#e2e8f0' if dark_l['id'] != 'poi_fuel' else '#fbbf24'
            if 'text-halo-color' in dark_l['paint']:
                dark_l['paint']['text-halo-color'] = '#0f172a'
        new_poi_layers_dark.append(dark_l)

    dark['layers'] = [l for l in dark['layers'] if l['id'] not in existing_ids]
    insert_idx_dark = next((i for i, l in enumerate(dark['layers']) if l['id'] == 'label_other'), len(dark['layers']))
    dark['layers'][insert_idx_dark:insert_idx_dark] = new_poi_layers_dark

    with open('android/app/src/main/assets/uber_dark_style.json', 'w', encoding='utf-8') as f:
        json.dump(dark, f, indent=2)
    print("uber_dark_style.json enriched with", len(new_poi_layers_dark), "layers. Total layers:", len(dark['layers']))

if __name__ == '__main__':
    enrich_styles()
