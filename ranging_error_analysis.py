import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_and_process_data(filename):
    """Load and process the CSV data"""
    data = pd.read_csv(filename)
    data.columns = ['lat', 'lon', 'range_m']
    
    # Convert to local coordinates (meters)
    lat_center = data['lat'].mean()
    lon_center = data['lon'].mean()
    
    data['x_m'] = (data['lon'] - lon_center) * 111000 * np.cos(np.radians(lat_center))
    data['y_m'] = (data['lat'] - lat_center) * 111000
    
    return data, lat_center, lon_center

def calculate_ranging_errors(data, target_lat, target_lon, target_depth):
    """Calculate ranging errors for each measurement if target is at predicted location"""
    
    # Convert target to local coordinates
    lat_center = data['lat'].mean()
    lon_center = data['lon'].mean()
    
    target_x = (target_lon - lon_center) * 111000 * np.cos(np.radians(lat_center))
    target_y = (target_lat - lat_center) * 111000
    target_z = -target_depth
    
    print("=" * 80)
    print("RANGING ERROR ANALYSIS - PREDICTED TARGET LOCATION")
    print("=" * 80)
    
    print(f"Target Location:")
    print(f"  • Latitude: {target_lat:.6f}°")
    print(f"  • Longitude: {target_lon:.6f}°")
    print(f"  • Depth: {target_depth:.1f}m")
    print(f"  • Local coordinates: ({target_x:.1f}m, {target_y:.1f}m, {target_z:.1f}m)")
    
    # Calculate errors for each measurement
    errors = []
    predicted_ranges = []
    surface_distances = []
    
    for i, (_, row) in enumerate(data.iterrows()):
        # Calculate 3D distance from boat to predicted target
        boat_x, boat_y = row['x_m'], row['y_m']
        boat_z = 0  # Boat is at surface
        
        # 3D distance calculation
        distance_3d = np.sqrt((target_x - boat_x)**2 + (target_y - boat_y)**2 + (target_z - boat_z)**2)
        
        # Surface distance (horizontal only)
        surface_distance = np.sqrt((target_x - boat_x)**2 + (target_y - boat_y)**2)
        
        # Measured range
        measured_range = row['range_m']
        
        # Error calculation
        error = measured_range - distance_3d
        error_percentage = (error / measured_range) * 100
        
        errors.append(error)
        predicted_ranges.append(distance_3d)
        surface_distances.append(surface_distance)
        
        print(f"\nMeasurement {i+1}:")
        print(f"  • Boat position: ({boat_x:.1f}m, {boat_y:.1f}m)")
        print(f"  • Measured range: {measured_range:.1f}m")
        print(f"  • Predicted range: {distance_3d:.1f}m")
        print(f"  • Surface distance: {surface_distance:.1f}m")
        print(f"  • Error: {error:+.1f}m ({error_percentage:+.1f}%)")
    
    # Statistical analysis
    errors_array = np.array(errors)
    mean_error = np.mean(errors_array)
    std_error = np.std(errors_array)
    mae = np.mean(np.abs(errors_array))
    rmse = np.sqrt(np.mean(errors_array**2))
    
    print(f"\n" + "=" * 80)
    print("ERROR STATISTICS")
    print("=" * 80)
    print(f"Mean Error: {mean_error:+.1f}m")
    print(f"Standard Deviation: {std_error:.1f}m")
    print(f"Mean Absolute Error (MAE): {mae:.1f}m")
    print(f"Root Mean Square Error (RMSE): {rmse:.1f}m")
    print(f"Error Range: {errors_array.min():+.1f}m to {errors_array.max():+.1f}m")
    
    # Identify outliers (beyond 2 standard deviations)
    outliers = np.where(np.abs(errors_array - mean_error) > 2 * std_error)[0]
    if len(outliers) > 0:
        print(f"\nOutliers (beyond 2σ):")
        for idx in outliers:
            print(f"  • Measurement {idx+1}: {errors_array[idx]:+.1f}m")
    else:
        print(f"\nNo outliers detected (all within 2σ)")
    
    return {
        'errors': errors_array,
        'predicted_ranges': predicted_ranges,
        'measured_ranges': data['range_m'].values,
        'surface_distances': surface_distances,
        'target_pos': [target_x, target_y, target_z],
        'statistics': {
            'mean_error': mean_error,
            'std_error': std_error,
            'mae': mae,
            'rmse': rmse,
            'outliers': outliers
        }
    }

def create_error_visualization(data, error_data):
    """Create comprehensive error visualization"""
    fig = plt.figure(figsize=(18, 12))
    
    # Error distribution histogram
    ax1 = fig.add_subplot(231)
    errors = error_data['errors']
    ax1.hist(errors, bins=8, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(error_data['statistics']['mean_error'], color='red', linestyle='--', 
                label=f'Mean: {error_data["statistics"]["mean_error"]:+.1f}m')
    ax1.axvline(0, color='green', linestyle='-', alpha=0.5, label='Perfect measurement')
    ax1.set_xlabel('Ranging Error (m)')
    ax1.set_ylabel('Number of Measurements')
    ax1.set_title('Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Error vs measurement number
    ax2 = fig.add_subplot(232)
    measurement_numbers = range(1, len(errors) + 1)
    colors = ['red' if i in error_data['statistics']['outliers'] else 'blue' for i in range(len(errors))]
    ax2.bar(measurement_numbers, errors, color=colors, alpha=0.7)
    ax2.axhline(0, color='green', linestyle='-', alpha=0.5, label='Perfect measurement')
    ax2.axhline(error_data['statistics']['mean_error'], color='red', linestyle='--', 
                label=f'Mean: {error_data["statistics"]["mean_error"]:+.1f}m')
    ax2.set_xlabel('Measurement Number')
    ax2.set_ylabel('Error (m)')
    ax2.set_title('Error by Measurement')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Measured vs Predicted ranges
    ax3 = fig.add_subplot(233)
    measured = error_data['measured_ranges']
    predicted = error_data['predicted_ranges']
    
    ax3.scatter(measured, predicted, c=colors, alpha=0.7, s=100)
    ax3.plot([measured.min(), measured.max()], [measured.min(), measured.max()], 
             'g--', alpha=0.5, label='Perfect prediction')
    ax3.set_xlabel('Measured Range (m)')
    ax3.set_ylabel('Predicted Range (m)')
    ax3.set_title('Measured vs Predicted Ranges')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Error vs Surface Distance
    ax4 = fig.add_subplot(234)
    surface_distances = error_data['surface_distances']
    ax4.scatter(surface_distances, errors, c=colors, alpha=0.7, s=100)
    ax4.axhline(0, color='green', linestyle='-', alpha=0.5, label='Perfect measurement')
    ax4.axhline(error_data['statistics']['mean_error'], color='red', linestyle='--', 
                label=f'Mean: {error_data["statistics"]["mean_error"]:+.1f}m')
    ax4.set_xlabel('Surface Distance (m)')
    ax4.set_ylabel('Error (m)')
    ax4.set_title('Error vs Surface Distance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Error percentage by measurement
    ax5 = fig.add_subplot(235)
    error_percentages = (errors / measured) * 100
    ax5.bar(measurement_numbers, error_percentages, color=colors, alpha=0.7)
    ax5.axhline(0, color='green', linestyle='-', alpha=0.5, label='Perfect measurement')
    ax5.set_xlabel('Measurement Number')
    ax5.set_ylabel('Error Percentage (%)')
    ax5.set_title('Error Percentage by Measurement')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Summary statistics
    ax6 = fig.add_subplot(236)
    ax6.axis('off')
    
    stats = error_data['statistics']
    summary_text = f'RANGING ERROR ANALYSIS\n\n' \
                   f'Target Location:\n' \
                   f'• Predicted position used for analysis\n\n' \
                   f'Error Statistics:\n' \
                   f'• Mean Error: {stats["mean_error"]:+.1f}m\n' \
                   f'• Standard Deviation: {stats["std_error"]:.1f}m\n' \
                   f'• Mean Absolute Error: {stats["mae"]:.1f}m\n' \
                   f'• Root Mean Square Error: {stats["rmse"]:.1f}m\n\n' \
                   f'Error Range:\n' \
                   f'• Minimum: {errors.min():+.1f}m\n' \
                   f'• Maximum: {errors.max():+.1f}m\n\n' \
                   f'Outliers:\n' \
                   f'• {len(stats["outliers"])} measurements beyond 2σ\n' \
                   f'• Outlier threshold: ±{2*stats["std_error"]:.1f}m'
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
    
    plt.tight_layout()
    return fig

def print_detailed_error_analysis(data, error_data):
    """Print detailed error analysis for each measurement"""
    print("\n" + "=" * 80)
    print("DETAILED ERROR ANALYSIS BY MEASUREMENT")
    print("=" * 80)
    
    errors = error_data['errors']
    measured = error_data['measured_ranges']
    predicted = error_data['predicted_ranges']
    surface_distances = error_data['surface_distances']
    outliers = error_data['statistics']['outliers']
    
    print(f"{'#':<3} {'Measured':<10} {'Predicted':<10} {'Error':<8} {'%Error':<8} {'Surface':<8} {'Status':<10}")
    print("-" * 70)
    
    for i in range(len(errors)):
        error_pct = (errors[i] / measured[i]) * 100
        status = "OUTLIER" if i in outliers else "Normal"
        status_color = "🔴" if i in outliers else "🟢"
        
        print(f"{i+1:<3} {measured[i]:<10.1f} {predicted[i]:<10.1f} "
              f"{errors[i]:<8.1f} {error_pct:<8.1f} {surface_distances[i]:<8.1f} {status_color} {status}")

def main():
    """Main function for ranging error analysis"""
    try:
        print("Loading underwater target detection data...")
        data, lat_center, lon_center = load_and_process_data('rawdata.csv')
        
        # Use the predicted target location from previous analysis
        # These are the coordinates from the hemisphere surface analysis
        target_lat = 19.830566
        target_lon = -156.119753
        target_depth = 171.1
        
        print("Calculating ranging errors for predicted target location...")
        error_data = calculate_ranging_errors(data, target_lat, target_lon, target_depth)
        
        print_detailed_error_analysis(data, error_data)
        
        print("\nCreating error visualization...")
        fig = create_error_visualization(data, error_data)
        plt.show()
        
        print("\nRanging error analysis complete!")
        
    except FileNotFoundError:
        print("Error: rawdata.csv not found in current directory")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 