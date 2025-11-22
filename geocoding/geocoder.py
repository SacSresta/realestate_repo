import pandas as pd
import json
import logging
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import os

logging.basicConfig(filename='geocoding.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logging.info("geocoding package loaded")

def process_geocoding(input_path: str = None, output_dir: str = None):
    """Process geocoding for a CSV file. If input_path is None, uses the original default.

    Returns the resulting DataFrame.
    """
    # Defaults (kept from original script)
    if input_path is None:
        input_path = r'E:\Realestate_Research\Remaining_geocoding.csv'

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_path), 'geocode_output')

    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, 'geocoded_realestate_housing_data_2022_remaining.csv')
    cache_path = os.path.join(output_dir, 'geocode_cache_v2_remaining.json')

    df = pd.read_csv(input_path)

    def convert_price(price_str):
        if pd.isna(price_str):
            return None
        try:
            return float(''.join(c for c in str(price_str) if c.isdigit() or c == '.'))
        except:
            return None

    if 'soldPrice' in df.columns:
        df['price_numeric'] = df['soldPrice'].apply(convert_price)

    df_sample = df.dropna(subset=['address', 'suburb', 'state', 'postcode'])
    df_unique = df_sample[['address', 'suburb', 'state', 'postcode']].drop_duplicates().reset_index(drop=True)

    try:
        with open(cache_path, "r") as f:
            geocode_cache = json.load(f)
        logging.info(f"Loaded cache with {len(geocode_cache)} entries.")
    except FileNotFoundError:
        geocode_cache = {}
        logging.info("No cache found, starting fresh.")

    geolocator = Nominatim(user_agent="real_estate_analysis", timeout=10)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=3, error_wait_seconds=5)

    def geocode_address(address, suburb, state, postcode):
        full_address = f"{address}, {suburb}, {state} {postcode}, Australia"
        if full_address in geocode_cache:
            return geocode_cache[full_address]

        try:
            location = geocode(full_address)
            if location:
                coords = (location.latitude, location.longitude)
            else:
                fallback_address = f"{suburb}, {state} {postcode}, Australia"
                location = geocode(fallback_address)
                coords = (location.latitude, location.longitude) if location else None
        except Exception as e:
            logging.warning(f"Error geocoding {full_address}: {e}")
            coords = None

        geocode_cache[full_address] = coords
        return coords

    coords_list = []
    for idx, row in df_unique.iterrows():
        print(f"Processing {idx+1}/{len(df_unique)}: {row['address']}, {row['suburb']}, {row['state']} {row['postcode']}")
        coords = geocode_address(row['address'], row['suburb'], row['state'], row['postcode'])
        coords_list.append(coords)

        if (idx + 1) % 100 == 0:
            logging.info(f"Processed {idx+1}/{len(df_unique)} unique addresses, saving progress...")
            with open(cache_path, "w") as f:
                json.dump(geocode_cache, f)

    with open(cache_path, "w") as f:
        json.dump(geocode_cache, f)
    logging.info("Cache saved after final batch.")

    df_unique['coords'] = coords_list
    df_unique['latitude'] = df_unique['coords'].apply(lambda x: x[0] if x else None)
    df_unique['longitude'] = df_unique['coords'].apply(lambda x: x[1] if x else None)

    df_result = df_sample.merge(df_unique[['address','suburb','state','postcode','latitude','longitude']],
                                on=['address','suburb','state','postcode'], how='left')

    df_result.to_csv(save_path, index=False)
    logging.info(f"Final geocoded data saved to {save_path}")
    print(df_sample[['address', 'suburb', 'state', 'postcode', 'latitude', 'longitude']].head(20))

    return df_result

if __name__ == "__main__":
    process_geocoding()
