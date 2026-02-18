# Underwater Target Detection Visualization

This Python script visualizes underwater target detection data collected from a boat with range-only sensors. The script creates both static and animated visualizations showing boat positions, range circles, and estimated target location.

## Features

- **Boat Position Plotting**: Each boat position is plotted with a unique color and marker shape
- **Range Circles**: Transparent circles centered on each boat position with radius equal to the range to target
- **Target Location Estimation**: Uses least-squares triangulation to estimate the most likely target location
- **Confidence Region**: Shows areas where multiple range circles overlap, indicating higher confidence
- **Animation**: Progressive reveal of data points showing how the target location estimate improves with more data
- **Coordinate Conversion**: Converts GPS coordinates to local meters for accurate calculations

## Data Format

The script expects a CSV file with the following columns:
- `Boat Lat`: Latitude of boat position (degrees)
- `BoatLon`: Longitude of boat position (degrees)  
- `Avg range to target (m)`: Range from boat to underwater target (meters)

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your data file as `rawdata.csv` in the same directory as the script
2. Run the script:
```bash
python target_detection_visualization.py
```

The script will:
1. Load and process your data
2. Display summary statistics
3. Show the estimated target location
4. Create a static visualization
5. Create an animated visualization

## Output

- **Static Plot**: Shows all boat positions, range circles, estimated target location, and confidence region
- **Animated Plot**: Progressively reveals data points and shows how the target estimate improves
- **Console Output**: Displays estimated target coordinates and statistics

## Algorithm Details

The target location estimation uses:
- **Least Squares Triangulation**: Minimizes the sum of squared differences between predicted and actual ranges
- **Coordinate Conversion**: Converts GPS coordinates to local meters for accurate distance calculations
- **Confidence Mapping**: Creates a heatmap showing areas where multiple range circles overlap

## Notes

- The script assumes the target is stationary during data collection
- Range measurements are assumed to be from the ocean surface to the underwater target
- The confidence region shows areas where the target is most likely located based on range circle intersections
- The animation shows how additional data points improve the target location estimate

## Customization

You can modify the script to:
- Change marker styles and colors
- Adjust animation speed
- Modify the confidence region calculation
- Add additional analysis features 