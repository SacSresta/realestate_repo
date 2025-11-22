#!/usr/bin/env python3
"""Attraction finder moved into package."""

import requests
import json
import argparse
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Keep the same categories
ATTRACTION_CATEGORIES = {
    'dining': ['restaurant', 'cafe', 'bar', 'pub', 'fast_food', 'food_court', 'bakery'],
    'shopping': ['supermarket', 'mall', 'shop', 'store', 'retail', 'marketplace', 'convenience'],
    'accommodation': ['hotel', 'hostel', 'motel', 'guest_house', 'bed_and_breakfast'],
    'entertainment': ['cinema', 'theatre', 'museum', 'gallery', 'artwork', 'nightclub', 'casino'],
    'transportation': ['station', 'stop', 'terminal', 'bus_station', 'subway', 'train_station'],
    'education': ['school', 'university', 'college', 'library', 'kindergarten', 'driving_school'],
    'leisure': ['park', 'garden', 'playground', 'sport', 'gym', 'fitness', 'swimming_pool'],
    'healthcare': ['hospital', 'clinic', 'pharmacy', 'dentist', 'doctor', 'veterinary'],
    'services': ['bank', 'atm', 'post_office', 'police', 'fire_station', 'fuel'],
    'religious': ['church', 'mosque', 'synagogue', 'temple', 'place_of_worship']
}

def calculate_distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r

def find_nearby_attractions(lat, lon, radius=1000, limit=60):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:25];
    (
      node["tourism"](around:{radius},{lat},{lon});
      node["amenity"](around:{radius},{lat},{lon});
      node["leisure"](around:{radius},{lat},{lon});
      node["shop"](around:{radius},{lat},{lon});
      node["cafe"](around:{radius},{lat},{lon});
      node["hospital"](around:{radius},{lat},{lon});
      node["school"](around:{radius},{lat},{lon});
      node["university"](around:{radius},{lat},{lon});
      way["tourism"](around:{radius},{lat},{lon});
      way["amenity"](around:{radius},{lat},{lon});
      way["leisure"](around:{radius},{lat},{lon});
      way["shop"](around:{radius},{lat},{lon});
    );
    out center;
    """
    try:
        logger.info(f"Searching for attractions around ({lat}, {lon}) within {radius}m")
        response = requests.post(overpass_url, data={"data": overpass_query}, timeout=30)
        response.raise_for_status()
        data = response.json()

        attractions = []
        for element in data.get("elements", []):
            if "tags" in element:
                tags = element["tags"]
                name = tags.get("name", "Unnamed")
                if element["type"] == "node":
                    elem_lat, elem_lon = element["lat"], element["lon"]
                elif element["type"] == "way" and "center" in element:
                    elem_lat, elem_lon = element["center"]["lat"], element["center"]["lon"]
                else:
                    continue
                attraction_type = "other"
                for tag_key in ["tourism", "amenity", "leisure", "shop"]:
                    if tag_key in tags:
                        attraction_type = tags[tag_key]
                        break
                distance = calculate_distance(lat, lon, elem_lat, elem_lon)
                attractions.append({
                    "name": name,
                    "type": attraction_type,
                    "lat": elem_lat,
                    "lon": elem_lon,
                    "distance": distance,
                    "tags": tags
                })

        attractions.sort(key=lambda x: x["distance"])
        total_found = len(attractions)
        return attractions, total_found
    except Exception as e:
        logger.error(f"Error fetching nearby attractions: {e}")
        return [], 0

def categorize_attractions(attractions):
    categorized = {category: [] for category in ATTRACTION_CATEGORIES}
    uncategorized = []
    for attraction in attractions:
        attraction_name = attraction['name'].lower()
        attraction_type = attraction['type'].lower()
        categorized_flag = False
        for category, keywords in ATTRACTION_CATEGORIES.items():
            if any(keyword in attraction_type for keyword in keywords) or \
               any(keyword in attraction_name for keyword in keywords):
                categorized[category].append(attraction)
                categorized_flag = True
                break
        if not categorized_flag:
            uncategorized.append(attraction)
    if uncategorized:
        categorized['uncategorized'] = uncategorized
    return categorized

def calculate_walkability_score(attractions):
    close_amenities = len([a for a in attractions if a['distance'] < 0.5])
    medium_amenities = len([a for a in attractions if 0.5 <= a['distance'] < 1.0])
    score = min(100, (15 * close_amenities + 8 * medium_amenities) / 5)
    return round(score, 1)

def calculate_amenity_impact_score(categorized_attractions):
    value_impact = {
        'transportation': 0.25,
        'education': 0.20,
        'shopping': 0.15,
        'dining': 0.15,
        'healthcare': 0.10,
        'leisure': 0.10,
        'services': 0.05
    }
    amenity_score = 0
    for category in value_impact.keys():
        if category in categorized_attractions:
            category_score = min(1.0, len(categorized_attractions[category]) / 5)
            amenity_score += value_impact[category] * category_score
    return round(amenity_score * 100, 1)

def find_categorized_attractions(lat, lon, radius=1000, limit=60):
    attractions, total_found = find_nearby_attractions(lat, lon, radius, limit)
    if not attractions:
        logger.warning("No attractions found")
        return {
            'coordinates': {'lat': lat, 'lon': lon},
            'search_radius': radius,
            'total_attractions': 0,
            'actual_total_found': 0,
            'display_total_attractions': 0,
            'categorized_attractions': {},
            'category_counts': {},
            'walkability_score': 0,
            'amenity_impact_score': 0,
            'closest_attractions': []
        }
    categorized = categorize_attractions(attractions)
    category_counts = {category: len(attractions_list) 
                      for category, attractions_list in categorized.items() 
                      if attractions_list}
    if 'uncategorized' in category_counts:
        category_counts['unknown'] = category_counts.pop('uncategorized')
        categorized['unknown'] = categorized.pop('uncategorized', [])
    walkability = calculate_walkability_score(attractions)
    amenity_impact = calculate_amenity_impact_score(categorized)
    closest = attractions[:10]
    total_display = total_found if total_found <= 50 else ">50"
    results = {
        'coordinates': {'lat': lat, 'lon': lon},
        'search_radius': radius,
        'total_attractions': len(attractions),
        'total_attractions_display': total_display,
        'categorized_attractions': categorized,
        'category_counts': category_counts,
        'walkability_score': walkability,
        'amenity_impact_score': amenity_impact,
        'closest_attractions': closest
    }
    return results

def print_results(results):
    coords = results['coordinates']
    print(f"\n{'='*60}")
    print(f"ATTRACTION ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Location: {coords['lat']:.6f}, {coords['lon']:.6f}")
    print(f"Search Radius: {results['search_radius']}m")
    print(f"Total Attractions Found: {results['total_attractions_display']}")
    print(f"\n{'CATEGORY BREAKDOWN':<25} {'COUNT':<10}")
    print(f"{'-'*35}")
    for category, count in sorted(results['category_counts'].items(), 
                                 key=lambda x: x[1], reverse=True):
        print(f"{category.title():<25} {count:<10}")
    print(f"\n{'LOCATION SCORES'}")
    print(f"{'-'*20}")
    print(f"Walkability Score: {results['walkability_score']}/100")
    print(f"Amenity Impact Score: {results['amenity_impact_score']}/100")
    if results['closest_attractions']:
        print(f"\n{'CLOSEST ATTRACTIONS'}")
        print(f"{'-'*30}")
        for i, attraction in enumerate(results['closest_attractions'][:10], 1):
            print(f"{i}. {attraction['name']} ({attraction['type']}) - {attraction['distance']:.2f}km")

def save_results_to_csv(results, filename="attraction_analysis.csv"):
    row = {
        'latitude': results['coordinates']['lat'],
        'longitude': results['coordinates']['lon'],
        'search_radius_m': results['search_radius'],
        'total_attractions': results['total_attractions'],
        'walkability_score': results['walkability_score'],
        'amenity_impact_score': results['amenity_impact_score']
    }
    for category in ATTRACTION_CATEGORIES.keys():
        row[f'num_{category}'] = results['category_counts'].get(category, 0)
    row['num_unknown'] = results['category_counts'].get('unknown', 0)
    df = pd.DataFrame([row])
    df.to_csv(filename, index=False)
    logger.info(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Find categorized attractions near coordinates')
    parser.add_argument('--lat', type=float, required=True, help='Latitude')
    parser.add_argument('--lon', type=float, required=True, help='Longitude')
    parser.add_argument('--radius', type=int, default=1000, help='Search radius in meters (default: 1000)')
    parser.add_argument('--limit', type=int, default=60, help='Maximum number of results (default: 60)')
    parser.add_argument('--output', type=str, help='Output CSV filename (optional)')
    parser.add_argument('--quiet', action='store_true', help='Suppress detailed output')
    args = parser.parse_args()
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    results = find_categorized_attractions(args.lat, args.lon, args.radius, args.limit)
    if not args.quiet:
        print_results(results)
    if args.output:
        save_results_to_csv(results, args.output)
    if args.quiet:
        print(f"{results['total_attractions']},{results['walkability_score']},{results['amenity_impact_score']}")

if __name__ == "__main__":
    main()
