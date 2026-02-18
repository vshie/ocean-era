import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import minimize

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

def estimate_target_location(data):
    """Estimate target location using least squares triangulation"""
    def objective_function(target_pos):
        x, y = target_pos
        predicted_ranges = np.sqrt((data['x_m'] - x)**2 + (data['y_m'] - y)**2)
        return np.sum((predicted_ranges - data['range_m'])**2)
    
    # Initial guess: centroid of boat positions
    initial_guess = [data['x_m'].mean(), data['y_m'].mean()]
    
    # Optimize to find target location
    result = minimize(objective_function, initial_guess, method='Nelder-Mead')
    
    if result.success:
        return result.x
    else:
        return initial_guess

def create_visualization(data, target_location, lat_center, lon_center):
    """Create the main visualization"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Set up the plot
    ax.set_xlabel('Longitude (relative to center, meters)', fontsize=12)
    ax.set_ylabel('Latitude (relative to center, meters)', fontsize=12)
    ax.set_title('Underwater Target Detection Visualization\nRange-Only Sensor Data Analysis', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot boat positions with different markers and colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|']
    
    for i, (_, row) in enumerate(data.iterrows()):
        # Plot boat position
        ax.scatter(row['x_m'], row['y_m'], 
                  c=[colors[i]], 
                  marker=markers[i % len(markers)],
                  s=120, 
                  edgecolors='black',
                  linewidth=1.5,
                  label=f'Boat {i+1} (Range: {row["range_m"]:.1f}m)')
        
        # Create range circle
        circle = patches.Circle((row['x_m'], row['y_m']), 
                              row['range_m'], 
                              fill=False, 
                              edgecolor=colors[i], 
                              alpha=0.4, 
                              linewidth=2.5,
                              linestyle='--')
        ax.add_patch(circle)
    
    # Plot estimated target location
    target_x, target_y = target_location
    ax.scatter(target_x, target_y, 
               c='red', 
               marker='*', 
               s=400, 
               edgecolors='black',
               linewidth=2,
               label=f'Estimated Target\n({target_x:.1f}m, {target_y:.1f}m)')
    
    # Create confidence region
    plot_confidence_region(ax, data)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add target information box
    target_lat = lat_center + target_y / 111000
    target_lon = lon_center + target_x / (111000 * np.cos(np.radians(lat_center)))
    
    info_text = f'Target Location Estimate:\n' \
                f'Latitude: {target_lat:.6f}°\n' \
                f'Longitude: {target_lon:.6f}°\n' \
                f'Distance from center: {np.sqrt(target_x**2 + target_y**2):.1f}m\n' \
                f'Data points: {len(data)}'
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
    
    plt.tight_layout()
    return fig, ax

def plot_confidence_region(ax, data):
    """Plot the confidence region where target is likely located"""
    # Create a grid of points
    x_min, x_max = data['x_m'].min() - 200, data['x_m'].max() + 200
    y_min, y_max = data['y_m'].min() - 200, data['y_m'].max() + 200
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                        np.linspace(y_min, y_max, 150))
    
    # Calculate how many range circles each point falls within
    confidence = np.zeros_like(xx)
    for _, row in data.iterrows():
        distance = np.sqrt((xx - row['x_m'])**2 + (yy - row['y_m'])**2)
        confidence += (distance <= row['range_m']).astype(float)
    
    # Normalize confidence (0 to 1)
    confidence = confidence / len(data)
    
    # Plot confidence region
    contour = ax.contourf(xx, yy, confidence, levels=15, alpha=0.3, cmap='Reds')
    ax.contour(xx, yy, confidence, levels=[0.5, 0.7, 0.9], colors='red', linewidths=2, alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
    cbar.set_label('Confidence Level\n(Fraction of circles containing point)', fontsize=10)

def print_analysis_summary(data, target_location, lat_center, lon_center):
    """Print a comprehensive analysis summary"""
    print("=" * 60)
    print("UNDERWATER TARGET DETECTION ANALYSIS")
    print("=" * 60)
    print(f"Data Summary:")
    print(f"  • Number of measurement points: {len(data)}")
    print(f"  • Range measurements: {data['range_m'].min():.1f}m to {data['range_m'].max():.1f}m")
    print(f"  • Average range: {data['range_m'].mean():.1f}m")
    print(f"  • Standard deviation: {data['range_m'].std():.1f}m")
    
    # Calculate target location in GPS coordinates
    target_x, target_y = target_location
    target_lat = lat_center + target_y / 111000
    target_lon = lon_center + target_x / (111000 * np.cos(np.radians(lat_center)))
    
    print(f"\nTarget Location Estimate:")
    print(f"  • Latitude:  {target_lat:.6f}°")
    print(f"  • Longitude: {target_lon:.6f}°")
    print(f"  • Distance from center: {np.sqrt(target_x**2 + target_y**2):.1f}m")
    
    # Calculate confidence metrics
    predicted_ranges = np.sqrt((data['x_m'] - target_x)**2 + (data['y_m'] - target_y)**2)
    range_errors = np.abs(predicted_ranges - data['range_m'])
    print(f"\nEstimation Quality:")
    print(f"  • Mean range error: {range_errors.mean():.1f}m")
    print(f"  • Max range error: {range_errors.max():.1f}m")
    print(f"  • Root mean square error: {np.sqrt(np.mean(range_errors**2)):.1f}m")
    
    print("=" * 60)

def main():
    """Main function"""
    try:
        # Load and process data
        print("Loading data...")
        data, lat_center, lon_center = load_and_process_data('rawdata.csv')
        
        # Estimate target location
        print("Estimating target location...")
        target_location = estimate_target_location(data)
        
        # Print analysis summary
        print_analysis_summary(data, target_location, lat_center, lon_center)
        
        # Create visualization
        print("\nCreating visualization...")
        fig, ax = create_visualization(data, target_location, lat_center, lon_center)
        
        # Show the plot
        plt.show()
        
        print("\nVisualization complete!")
        
    except FileNotFoundError:
        print("Error: rawdata.csv not found in current directory")
        print("Please ensure your data file is named 'rawdata.csv' and is in the same directory as this script")
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your data format and try again")

if __name__ == "__main__":
    main() 