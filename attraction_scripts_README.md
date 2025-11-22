# Attraction Finder Scripts

This package provides tools to find and categorize nearby attractions for properties using OpenStreetMap data via the Overpass API.

## Files Included

1. **`attraction_finder.py`** - Core script for single property analysis
2. **`batch_attraction_finder.py`** - Batch processing for multiple properties
3. **`example_usage.py`** - Examples and demonstrations
4. **`attraction_scripts_README.md`** - This documentation

## Installation

### Prerequisites
```bash
pip install requests pandas tqdm
```

### Optional (for visualization)
```bash
pip install matplotlib seaborn folium
```

## Usage

### 1. Single Property Analysis

#### Command Line
```bash
python attraction_finder.py --lat -33.8675 --lon 151.2070 --radius 1000
```

#### Python Script
```python
from attraction_finder import find_categorized_attractions

# Find attractions within 1km
results = find_categorized_attractions(-33.8675, 151.2070, radius=1000)
print(f"Found {results['total_attractions']} attractions")
print(f"Walkability score: {results['walkability_score']}/100")
```

### 2. Batch Processing

#### Prepare Input CSV
Your CSV file should have these columns:
- `latitude` (required)
- `longitude` (required)
- `address`, `suburb`, `state`, `postcode` (optional, for identification)

#### Run Batch Processing
```bash
python batch_attraction_finder.py --input properties.csv --output results.csv --radius 1000
```

#### Create Sample File
```bash
python batch_attraction_finder.py --create-sample
```

### 3. Run Examples
```bash
python example_usage.py
```

## Features

### Attraction Categories
The script categorizes attractions into these types:
- **Dining**: Restaurants, cafes, bars, pubs, fast food
- **Shopping**: Supermarkets, malls, shops, stores
- **Accommodation**: Hotels, hostels, motels
- **Entertainment**: Cinemas, theaters, museums, galleries
- **Transportation**: Stations, stops, terminals
- **Education**: Schools, universities, libraries
- **Leisure**: Parks, gardens, playgrounds, sports facilities
- **Healthcare**: Hospitals, clinics, pharmacies
- **Services**: Banks, post offices, police stations
- **Religious**: Churches, mosques, temples

### Calculated Scores

#### Walkability Score (0-100)
Based on the number and proximity of nearby amenities:
- Attractions within 0.5km get higher weight
- Attractions within 0.5-1km get moderate weight

#### Amenity Impact Score (0-100)
Weighted score based on real estate impact:
- Transportation: 25% weight
- Education: 20% weight
- Shopping: 15% weight
- Dining: 15% weight
- Healthcare: 10% weight
- Leisure: 10% weight
- Services: 5% weight

## Output Format

### Single Property Results
```python
{
    'coordinates': {'lat': -33.8675, 'lon': 151.2070},
    'search_radius': 1000,
    'total_attractions': 45,
    'categorized_attractions': {
        'dining': [...],
        'shopping': [...],
        # ... other categories
    },
    'category_counts': {
        'dining': 12,
        'shopping': 8,
        # ... other categories
    },
    'walkability_score': 85.3,
    'amenity_impact_score': 72.1,
    'closest_attractions': [...]
}
```

### Batch Processing Output CSV
Original columns plus:
- `num_dining`, `num_shopping`, etc. (count for each category)
- `total_attractions`
- `walkability_score`
- `amenity_impact_score`

## Command Line Options

### attraction_finder.py
```
--lat LATITUDE          Latitude coordinate (required)
--lon LONGITUDE         Longitude coordinate (required)
--radius METERS         Search radius in meters (default: 1000)
--limit NUMBER          Maximum results (default: 50)
--output FILENAME       Save results to CSV file
--quiet                 Suppress detailed output
```

### batch_attraction_finder.py
```
--input FILENAME        Input CSV file (required)
--output FILENAME       Output CSV file (required)
--radius METERS         Search radius in meters (default: 1000)
--limit NUMBER          Maximum attractions per property (default: 50)
--delay SECONDS         Delay between API calls (default: 3.0)
--create-sample         Create sample input CSV
--analyze               Show analysis summary after processing
```

## Examples

### Find attractions near Sydney Opera House
```bash
python attraction_finder.py --lat -33.8568 --lon 151.2153 --radius 800
```

### Process multiple properties with custom settings
```bash
python batch_attraction_finder.py \
    --input my_properties.csv \
    --output results_with_attractions.csv \
    --radius 1500 \
    --delay 2.0 \
    --analyze
```

### Quiet mode for scripting
```bash
python attraction_finder.py --lat -33.8675 --lon 151.2070 --quiet
# Output: 45,85.3,72.1 (total_attractions,walkability_score,amenity_impact_score)
```

## Rate Limiting

The scripts include built-in rate limiting to respect the Overpass API:
- Default 3-second delay between requests
- Configurable via `--delay` parameter
- Automatic retry on failures
- Progress saving every 50 properties

## Error Handling

- Network timeouts and API errors are handled gracefully
- Invalid coordinates are skipped
- Progress is saved automatically during batch processing
- Detailed logging for troubleshooting

## Troubleshooting

### Common Issues

1. **No attractions found**: Check coordinates are valid and in populated areas
2. **API timeout**: Increase delay between requests with `--delay`
3. **Memory errors**: Reduce `--limit` parameter for large datasets

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## API Dependencies

This tool uses the [Overpass API](https://overpass-api.de/) which queries OpenStreetMap data. The service is free but has usage limits. For high-volume usage, consider:

1. Running requests during off-peak hours
2. Increasing delays between requests
3. Using a local Overpass instance
4. Implementing caching for repeated queries

## License

This script is provided as-is for educational and research purposes. Please respect the Overpass API terms of service and OpenStreetMap data usage policies.