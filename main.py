# First, let's reload our data since the kernel was restarted
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
from folium.plugins import MarkerCluster
import requests
import time

# Load the data
df = pd.read_csv(r'E:\Realestate_Research\notebook\cleaned_realestate_housing_data_2022.csv')

# Convert the price from string to numeric again
def convert_price(price_str):
    if pd.isna(price_str):
        return None
    try:
        return float(''.join(c for c in str(price_str) if c.isdigit() or c == '.'))
    except:
        return None

# Apply the conversion
df['price_numeric'] = df['soldPrice'].apply(convert_price)

# Create a geocoder with rate limiting to avoid being blocked
# Increase the delay to 2 seconds to avoid timeouts
geolocator = Nominatim(user_agent="real_estate_analysis", timeout=10)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=2, max_retries=3, error_wait_seconds=5)

# Let's show the first few addresses to understand their format
print("Sample addresses in the dataset:")
print(df['address'].head(10))
print(f"Total addresses: {df.shape[0]}")

# To avoid geocoding all 722K+ addresses, let's create a VERY small sample
# Reducing from 2000 to just 10 addresses to avoid rate limiting
#sample_size = 10
#df_sample = df.dropna(subset=['address']).sample(sample_size, random_state=42)
df_sample = df.dropna(subset=['address'])
# Geocode the sample addresses
def geocode_address(address, suburb, state, postcode):
    full_address = f"{address}, {suburb}, {state} {postcode}, Australia"
    try:
        print(f"Geocoding: {full_address}")
        location = geocode(full_address)
        if location:
            print(f"Success: {full_address}")
            return (location.latitude, location.longitude)
        else:
            # Try with just suburb, state and postcode if specific address fails
            print(f"Trying fallback for: {full_address}")
            fallback_address = f"{suburb}, {state} {postcode}, Australia"
            location = geocode(fallback_address)
            if location:
                print(f"Fallback success: {fallback_address}")
                return (location.latitude, location.longitude)
            else:
                print(f"Failed to geocode: {full_address}")
                return None
    except Exception as e:
        print(f"Error geocoding {full_address}: {e}")
        return None

# Process addresses one by one with error handling
coords_data = []
for idx, row in df_sample.iterrows():
    try:
        coords = geocode_address(row['address'], row['suburb'], row['state'], row['postcode'])
        coords_data.append((idx, coords))
        # Extra sleep between requests
        time.sleep(1)
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        coords_data.append((idx, None))
        time.sleep(3)  # Longer sleep on error
df_sample['coords'] = None
# Now update the dataframe safely
for idx, coords in coords_data:
    df_sample.at[idx, 'coords'] = coords

# Split the coordinates into separate columns
df_sample['latitude'] = df_sample['coords'].apply(lambda x: x[0] if x else None)
df_sample['longitude'] = df_sample['coords'].apply(lambda x: x[1] if x else None)

# Show the results
print("\nGeocoding Results:")
print(df_sample[['address', 'suburb', 'state', 'postcode', 'latitude', 'longitude']].head())
df_sample.to_csv(r'E:\Realestate_Research\notebook\geocoded_realestate_housing_data_2022.csv', index=False)
# Count how many addresses were successfully geocoded
successful = df_sample['latitude'].notna().sum()
print(f"\nSuccessfully geocoded {successful} out of {len(df_sample)} addresses.")