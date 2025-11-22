#!/usr/bin/env python3
"""Batch attraction finder moved into package."""

import pandas as pd
import argparse
import logging
import time
from .attraction_finder import find_categorized_attractions, ATTRACTION_CATEGORIES
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_properties_batch(input_file, output_file, radius=1000, limit=60, delay=3):
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Loaded {len(df)} properties from {input_file}")
        required_cols = ['latitude', 'longitude']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        df = df.dropna(subset=['latitude', 'longitude'])
        logger.info(f"Processing {len(df)} properties with valid coordinates")
        result_columns = list(df.columns)
        for category in ATTRACTION_CATEGORIES.keys():
            col_name = f'num_{category}'
            if col_name not in result_columns:
                result_columns.append(col_name)
                df[col_name] = 0
        if 'num_unknown' not in result_columns:
            result_columns.append('num_unknown')
            df['num_unknown'] = 0
        summary_cols = ['total_attractions', 'walkability_score', 'amenity_impact_score']
        for col in summary_cols:
            if col not in result_columns:
                result_columns.append(col)
                df[col] = 0
        processed_count = 0
        error_count = 0
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing properties"):
            try:
                lat, lon = row['latitude'], row['longitude']
                results = find_categorized_attractions(lat, lon, radius=radius, limit=limit)
                for category in ATTRACTION_CATEGORIES.keys():
                    df.at[idx, f'num_{category}'] = results['category_counts'].get(category, 0)
                df.at[idx, 'num_unknown'] = results['category_counts'].get('unknown', 0)
                df.at[idx, 'total_attractions'] = results['total_attractions']
                df.at[idx, 'walkability_score'] = results['walkability_score']
                df.at[idx, 'amenity_impact_score'] = results['amenity_impact_score']
                processed_count += 1
                if processed_count % 50 == 0:
                    df.to_csv(output_file, index=False)
                    logger.info(f"Progress saved: {processed_count}/{len(df)} properties processed")
                time.sleep(delay)
            except Exception as e:
                logger.error(f"Error processing property at index {idx}: {e}")
                error_count += 1
                continue
        df.to_csv(output_file, index=False)
        logger.info(f"Processing complete!")
        logger.info(f"Successfully processed: {processed_count} properties")
        logger.info(f"Errors encountered: {error_count} properties")
        logger.info(f"Results saved to: {output_file}")
        return df
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        raise

def analyze_batch_results(df):
    print(f"\n{'='*60}")
    print(f"BATCH ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total Properties Analyzed: {len(df)}")
    print(f"Properties with Attractions: {len(df[df['total_attractions'] > 0])}")
    print(f"\nAttraction Statistics:")
    print(f"Average Attractions per Property: {df['total_attractions'].mean():.1f}")
    print(f"Max Attractions Found: {df['total_attractions'].max()}")
    print(f"Min Attractions Found: {df['total_attractions'].min()}")
    print(f"\nScore Statistics:")
    print(f"Average Walkability Score: {df['walkability_score'].mean():.1f}")
    print(f"Average Amenity Impact Score: {df['amenity_impact_score'].mean():.1f}")
    print(f"\nTop Attraction Categories:")
    category_cols = [col for col in df.columns if col.startswith('num_')]
    category_sums = df[category_cols].sum().sort_values(ascending=False)
    for col, total in category_sums.head(10).items():
        category_name = col.replace('num_', '').replace('_', ' ').title()
        avg_per_property = total / len(df)
        print(f"{category_name}: {total} total ({avg_per_property:.1f} avg per property)")
    print(f"\nTop 5 Properties by Walkability Score:")
    top_walkability = df.nlargest(5, 'walkability_score')
    for idx, row in top_walkability.iterrows():
        address_info = ""
        if 'address' in row and pd.notna(row['address']):
            address_info = f" - {row['address']}"
        elif 'suburb' in row and pd.notna(row['suburb']):
            address_info = f" - {row['suburb']}"
        print(f"  Score: {row['walkability_score']:.1f}, Attractio ns: {row['total_attractions']}{address_info}")

def create_sample_input():
    sample_data = {
        'address': [
            '1902/2 Quay St, Haymarket',
            '298 Sussex Street, Sydney',
            '26/1 Macquarie Street, Sydney',
            '301/35 Bowman Street, Pyrmont',
            '68/500 Elizabeth Street, Surry Hills'
        ],
        'suburb': ['Haymarket', 'Sydney', 'Sydney', 'Pyrmont', 'Surry Hills'],
        'state': ['NSW', 'NSW', 'NSW', 'NSW', 'NSW'],
        'postcode': [2000, 2000, 2000, 2009, 2010],
        'latitude': [-33.8803733, -33.8743784, -33.8596899, -33.8668542, -33.8800472],
        'longitude': [151.2026973, 151.2046124, 151.2133537, 151.1911452, 151.2089896]
    }
    df = pd.DataFrame(sample_data)
    df.to_csv('sample_properties.csv', index=False)
    print("Sample input file 'sample_properties.csv' created")
    return df

def main():
    parser = argparse.ArgumentParser(description='Batch process properties for attraction analysis')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file with coordinates')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file for results')
    parser.add_argument('--radius', type=int, default=1000, help='Search radius in meters (default: 1000)')
    parser.add_argument('--limit', type=int, default=60, help='Maximum attractions per property (default: 60)')
    parser.add_argument('--delay', type=float, default=3.0, help='Delay between API calls in seconds (default: 3.0)')
    parser.add_argument('--create-sample', action='store_true', help='Create a sample input CSV file')
    parser.add_argument('--analyze', action='store_true', help='Analyze results after processing')
    args = parser.parse_args()
    if args.create_sample:
        create_sample_input()
        return
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return
    try:
        df = process_properties_batch(
            args.input,
            args.output,
            radius=args.radius,
            limit=args.limit,
            delay=args.delay
        )
        if args.analyze:
            analyze_batch_results(df)
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")

if __name__ == "__main__":
    main()
