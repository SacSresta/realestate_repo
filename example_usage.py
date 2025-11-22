#!/usr/bin/env python3
"""
Example usage of the attraction finder scripts

This script demonstrates how to use the attraction finder both for single properties
and batch processing.
"""

from attraction_finder import find_categorized_attractions, print_results, ATTRACTION_CATEGORIES
import pandas as pd

def example_single_property():
    """Example of analyzing a single property."""
    print("="*60)
    print("SINGLE PROPERTY EXAMPLE")
    print("="*60)
    
    # Example coordinates (Sydney CBD)
    lat, lon = -33.8675, 151.2070
    
    print(f"Analyzing property at coordinates: {lat}, {lon}")
    
    # Find attractions within 1km
    results = find_categorized_attractions(lat, lon, radius=1000)
    
    # Print formatted results
    print_results(results)
    
    return results

def example_multiple_properties(path: str = r'E:\Realestate_Research\RealEstate_geocoded.csv', save_dir: str = r'Data Tables Step 4', save_name: str = 'Property_full_table_raw.csv', radius: int = 1000, save_every: int = 1000):
    """Simple example of analyzing multiple properties from a DataFrame with periodic saves.

    - Reads the input CSV (expects latitude/Longitude columns named 'Latitude'/'Longitude' or 'latitude'/'longitude')
    - Adds category columns and summary columns
    - Iterates over rows, calls find_categorized_attractions, stores results
    - Saves periodically every `save_every` processed properties and once at the end

    Note: For large files this will make one Overpass API request per property and may be slow or hit API limits.
    """
    import os
    import time

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} properties from {path}")

    # Ensure save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    out_path = os.path.join(save_dir, save_name)

    # Add columns for predefined categories if missing
    for category in ATTRACTION_CATEGORIES.keys():
        col = f'num_{category}'
        if col not in df.columns:
            df[col] = 0

    # Add summary columns if missing
    if 'num_unknown' not in df.columns:
        df['num_unknown'] = 0
    if 'total_attractions' not in df.columns:
        df['total_attractions'] = 0
    if 'walkability_score' not in df.columns:
        df['walkability_score'] = 0

    print(f"Analyzing {len(df)} properties (no cache, one request per property)...")

    def _get_lat_lon(row):
        # Try common column names
        for lat_key in ['Latitude', 'latitude', 'LAT', 'lat']:
            if lat_key in row.index and not pd.isna(row[lat_key]):
                lat = row[lat_key]
                break
        else:
            lat = None
        for lon_key in ['Longitude', 'longitude', 'LON', 'lon']:
            if lon_key in row.index and not pd.isna(row[lon_key]):
                lon = row[lon_key]
                break
        else:
            lon = None
        return lat, lon

    processed_since_save = 0
    total_processed = 0

    for idx, row in df.iterrows():
        lat, lon = _get_lat_lon(row)
        if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
            print(f"\nSkipping index {idx} - missing coordinates")
            continue

        print(f"\nProcessing index {idx}: {row.get('Address', '')} ({lat}, {lon})")
        try:
            results = find_categorized_attractions(lat, lon, radius=radius)

            # Update DataFrame with predefined categories
            for category in ATTRACTION_CATEGORIES.keys():
                df.at[idx, f'num_{category}'] = results['category_counts'].get(category, 0)

            df.at[idx, 'num_unknown'] = results['category_counts'].get('unknown', 0)
            df.at[idx, 'total_attractions'] = results['total_attractions']
            df.at[idx, 'walkability_score'] = results['walkability_score']

            print(f"  Found {results['total_attractions']} attractions")
            print(f"  Walkability score: {results['walkability_score']}")
            if results['category_counts'].get('unknown', 0) > 0:
                print(f"  Unknown attractions: {results['category_counts']['unknown']}")

            # Small polite delay to avoid hammering the API
            time.sleep(1)

            # Update counters for periodic save
            processed_since_save += 1
            total_processed += 1

            if processed_since_save >= save_every:
                try:
                    df.to_csv(out_path, index=True)
                    print(f"Progress saved to {out_path} (processed {total_processed})")
                except Exception as e:
                    print(f"Warning: failed saving progress to {out_path}: {e}")
                processed_since_save = 0

        except Exception as e:
            print(f"Error processing property at index {idx}: {e}")
            continue

    # Final save
    try:
        df.to_csv(out_path, index=True)
        print(f"Final results saved to {out_path}")
    except Exception as e:
        print(f"Failed to save results to {out_path}: {e}")

    return df

def example_custom_analysis():
    """Example of custom analysis with specific requirements."""
    print("\n" + "="*60)
    print("CUSTOM ANALYSIS EXAMPLE")
    print("="*60)
    
    # Property near Central Station, Sydney
    lat, lon = -33.8830, 151.2067
    
    print(f"Custom analysis for property at: {lat}, {lon}")
    print("Looking for nearby attractions...")
    
    # Find attractions within 500m
    results = find_categorized_attractions(lat, lon, radius=500, limit=30)
    
    print(f"\nAttraction Analysis:")
    print(f"Total attractions within 500m: {results['total_attractions']}")
    print(f"Walkability score: {results['walkability_score']}/100")
    
    # Show closest attractions
    print(f"\nClosest attractions:")
    for i, attraction in enumerate(results['closest_attractions'][:5], 1):
        print(f"  {i}. {attraction['name']} ({attraction['type']}) - {attraction['distance']:.2f}km")
    
    # Transportation accessibility
    transport_count = results['category_counts'].get('transportation', 0)
    print(f"Transportation Accessibility: {transport_count} transport options")
    
    # Show closest attractions
    print(f"\nClosest 5 Attractions:")
    for i, attraction in enumerate(results['closest_attractions'][:5], 1):
        print(f"  {i}. {attraction['name']} ({attraction['type']}) - {attraction['distance']:.2f}km")

if __name__ == "__main__":
    # Run all examples
    try:

        multiple_results = example_multiple_properties()

        print(f"\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have internet connection for the Overpass API calls.")