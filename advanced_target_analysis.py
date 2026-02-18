import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

def load_and_process_data(filename):
    """Load and process the CSV data with 3D considerations"""
    data = pd.read_csv(filename)
    data.columns = ['lat', 'lon', 'range_m']
    
    # Convert to local coordinates (meters)
    lat_center = data['lat'].mean()
    lon_center = data['lon'].mean()
    
    data['x_m'] = (data['lon'] - lon_center) * 111000 * np.cos(np.radians(lat_center))
    data['y_m'] = (data['lat'] - lat_center) * 111000
    
    # Add 3D considerations - assume sensor is at surface, target is underwater
    # Range is the 3D distance through water, not just horizontal distance
    data['surface_range'] = np.sqrt(data['x_m']**2 + data['y_m']**2)  # Horizontal distance
    
    return data, lat_center, lon_center

def estimate_target_location_3d(data, depth_estimate=50):
    """Estimate target location considering 3D underwater geometry"""
    def objective_function_3d(target_pos):
        x, y, z = target_pos
        # Calculate 3D distances from each boat position to target
        predicted_ranges = np.sqrt((data['x_m'] - x)**2 + (data['y_m'] - y)**2 + z**2)
        return np.sum((predicted_ranges - data['range_m'])**2)
    
    # Initial guess: centroid of boat positions, estimated depth
    initial_guess = [data['x_m'].mean(), data['y_m'].mean(), depth_estimate]
    
    # Optimize to find target location
    result = minimize(objective_function_3d, initial_guess, method='Nelder-Mead')
    
    if result.success:
        return result.x
    else:
        return initial_guess

def estimate_target_location_2d(data):
    """Estimate target location using 2D approximation"""
    def objective_function(target_pos):
        x, y = target_pos
        predicted_ranges = np.sqrt((data['x_m'] - x)**2 + (data['y_m'] - y)**2)
        return np.sum((predicted_ranges - data['range_m'])**2)
    
    initial_guess = [data['x_m'].mean(), data['y_m'].mean()]
    result = minimize(objective_function, initial_guess, method='Nelder-Mead')
    
    if result.success:
        return result.x
    else:
        return initial_guess

def create_3d_visualization(data, target_3d, target_2d, lat_center, lon_center):
    """Create 3D visualization showing underwater geometry"""
    fig = plt.figure(figsize=(16, 12))
    
    # 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Depth (m)')
    ax1.set_title('3D Underwater Target Detection')
    
    # Plot boat positions at surface (z=0)
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    for i, (_, row) in enumerate(data.iterrows()):
        ax1.scatter(row['x_m'], row['y_m'], 0, 
                   c=[colors[i]], s=100, marker='o')
        
        # Draw lines from boat to target (3D range)
        ax1.plot([row['x_m'], target_3d[0]], 
                [row['y_m'], target_3d[1]], 
                [0, -target_3d[2]], 
                '--', alpha=0.3, color=colors[i])
    
    # Plot target locations
    ax1.scatter(target_3d[0], target_3d[1], -target_3d[2], 
               c='red', s=200, marker='*', label='3D Target Estimate')
    ax1.scatter(target_2d[0], target_2d[1], 0, 
               c='blue', s=200, marker='s', label='2D Target Estimate')
    
    ax1.legend()
    ax1.invert_zaxis()  # Make depth negative (below surface)
    
    # 2D top view
    ax2 = fig.add_subplot(222)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View - Boat Positions and Range Circles')
    ax2.grid(True, alpha=0.3)
    
    for i, (_, row) in enumerate(data.iterrows()):
        ax2.scatter(row['x_m'], row['y_m'], 
                   c=[colors[i]], s=100, marker='o')
        
        # Range circles (2D projection)
        circle = patches.Circle((row['x_m'], row['y_m']), 
                              row['range_m'], 
                              fill=False, 
                              edgecolor=colors[i], 
                              alpha=0.3, 
                              linewidth=2)
        ax2.add_patch(circle)
    
    ax2.scatter(target_2d[0], target_2d[1], 
               c='red', s=300, marker='*', label='Target Estimate')
    ax2.legend()
    ax2.set_aspect('equal')
    
    # Range vs Distance analysis
    ax3 = fig.add_subplot(223)
    ax3.scatter(data['surface_range'], data['range_m'], 
               c=colors, s=100, alpha=0.7)
    ax3.plot([0, data['surface_range'].max()], 
            [0, data['surface_range'].max()], 
            'k--', alpha=0.5, label='Surface distance = Range')
    ax3.set_xlabel('Surface Distance (m)')
    ax3.set_ylabel('Measured Range (m)')
    ax3.set_title('Range vs Surface Distance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Error analysis
    ax4 = fig.add_subplot(224)
    predicted_ranges_2d = np.sqrt((data['x_m'] - target_2d[0])**2 + (data['y_m'] - target_2d[1])**2)
    predicted_ranges_3d = np.sqrt((data['x_m'] - target_3d[0])**2 + (data['y_m'] - target_3d[1])**2 + target_3d[2]**2)
    
    errors_2d = np.abs(predicted_ranges_2d - data['range_m'])
    errors_3d = np.abs(predicted_ranges_3d - data['range_m'])
    
    x_pos = np.arange(len(data))
    width = 0.35
    
    ax4.bar(x_pos - width/2, errors_2d, width, label='2D Model Error', alpha=0.7)
    ax4.bar(x_pos + width/2, errors_3d, width, label='3D Model Error', alpha=0.7)
    ax4.set_xlabel('Measurement Point')
    ax4.set_ylabel('Range Error (m)')
    ax4.set_title('Model Comparison: Range Errors')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_advanced_analysis(data, target_3d, target_2d, lat_center, lon_center):
    """Print comprehensive analysis including 3D considerations"""
    print("=" * 70)
    print("ADVANCED UNDERWATER TARGET DETECTION ANALYSIS")
    print("=" * 70)
    
    print(f"Data Summary:")
    print(f"  • Number of measurement points: {len(data)}")
    print(f"  • Range measurements: {data['range_m'].min():.1f}m to {data['range_m'].max():.1f}m")
    print(f"  • Average range: {data['range_m'].mean():.1f}m")
    print(f"  • Surface distances: {data['surface_range'].min():.1f}m to {data['surface_range'].max():.1f}m")
    
    # Calculate target locations in GPS coordinates
    target_lat_2d = lat_center + target_2d[1] / 111000
    target_lon_2d = lon_center + target_2d[0] / (111000 * np.cos(np.radians(lat_center)))
    
    target_lat_3d = lat_center + target_3d[1] / 111000
    target_lon_3d = lon_center + target_3d[0] / (111000 * np.cos(np.radians(lat_center)))
    
    print(f"\nTarget Location Estimates:")
    print(f"  2D Model (Surface Projection):")
    print(f"    • Latitude:  {target_lat_2d:.6f}°")
    print(f"    • Longitude: {target_lon_2d:.6f}°")
    print(f"    • Distance from center: {np.sqrt(target_2d[0]**2 + target_2d[1]**2):.1f}m")
    
    print(f"  3D Model (Underwater):")
    print(f"    • Latitude:  {target_lat_3d:.6f}°")
    print(f"    • Longitude: {target_lon_3d:.6f}°")
    print(f"    • Depth: {target_3d[2]:.1f}m")
    print(f"    • Horizontal distance: {np.sqrt(target_3d[0]**2 + target_3d[1]**2):.1f}m")
    
    # Calculate model performance
    predicted_ranges_2d = np.sqrt((data['x_m'] - target_2d[0])**2 + (data['y_m'] - target_2d[1])**2)
    predicted_ranges_3d = np.sqrt((data['x_m'] - target_3d[0])**2 + (data['y_m'] - target_3d[1])**2 + target_3d[2]**2)
    
    errors_2d = np.abs(predicted_ranges_2d - data['range_m'])
    errors_3d = np.abs(predicted_ranges_3d - data['range_m'])
    
    print(f"\nModel Performance Comparison:")
    print(f"  2D Model:")
    print(f"    • Mean error: {errors_2d.mean():.1f}m")
    print(f"    • RMS error: {np.sqrt(np.mean(errors_2d**2)):.1f}m")
    print(f"    • Max error: {errors_2d.max():.1f}m")
    
    print(f"  3D Model:")
    print(f"    • Mean error: {errors_3d.mean():.1f}m")
    print(f"    • RMS error: {np.sqrt(np.mean(errors_3d**2)):.1f}m")
    print(f"    • Max error: {errors_3d.max():.1f}m")
    
    # Determine which model is better
    if errors_3d.mean() < errors_2d.mean():
        print(f"\n  → 3D model provides better fit to the data")
        recommended_target = target_3d
        recommended_model = "3D"
    else:
        print(f"\n  → 2D model provides better fit to the data")
        recommended_target = target_2d
        recommended_model = "2D"
    
    print(f"\nRecommended Target Location ({recommended_model} model):")
    if recommended_model == "3D":
        print(f"  • Latitude:  {target_lat_3d:.6f}°")
        print(f"  • Longitude: {target_lon_3d:.6f}°")
        print(f"  • Depth: {target_3d[2]:.1f}m")
    else:
        print(f"  • Latitude:  {target_lat_2d:.6f}°")
        print(f"  • Longitude: {target_lon_2d:.6f}°")
    
    print("=" * 70)

def main():
    """Main function for advanced analysis"""
    try:
        print("Loading and analyzing underwater target detection data...")
        data, lat_center, lon_center = load_and_process_data('rawdata.csv')
        
        print("Estimating target location using 2D and 3D models...")
        target_2d = estimate_target_location_2d(data)
        target_3d = estimate_target_location_3d(data)
        
        print_advanced_analysis(data, target_3d, target_2d, lat_center, lon_center)
        
        print("\nCreating advanced visualization...")
        fig = create_3d_visualization(data, target_3d, target_2d, lat_center, lon_center)
        
        plt.show()
        print("\nAnalysis complete!")
        
    except FileNotFoundError:
        print("Error: rawdata.csv not found in current directory")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 