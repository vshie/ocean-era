import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist

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

def hemisphere_intersection_objective(target_pos, boat_positions, ranges):
    """
    Objective function for hemisphere intersection.
    For each boat position, calculate how well the target fits within the hemisphere.
    A point inside a hemisphere should satisfy: sqrt((x-x0)² + (y-y0)² + z²) ≤ range AND z ≤ 0
    """
    x, y, z = target_pos
    
    # Ensure target is underwater (z ≤ 0)
    if z > 0:
        return 1e6  # Large penalty for targets above surface
    
    total_error = 0
    
    for i, (boat_x, boat_y) in enumerate(boat_positions):
        range_m = ranges[i]
        
        # Calculate 3D distance from boat to target
        distance_3d = np.sqrt((x - boat_x)**2 + (y - boat_y)**2 + z**2)
        
        # Error: how far outside the hemisphere the target is
        if distance_3d > range_m:
            # Target is outside hemisphere - penalize by distance outside
            error = (distance_3d - range_m)**2
        else:
            # Target is inside hemisphere - small penalty based on how close to edge
            error = (range_m - distance_3d)**2 * 0.1  # Prefer points closer to center of intersection
        
        total_error += error
    
    return total_error

def find_hemisphere_intersection(data):
    """Find the best intersection point of multiple hemispheres"""
    boat_positions = list(zip(data['x_m'], data['y_m']))
    ranges = data['range_m'].values
    
    # Initial guess: centroid of boat positions, estimated depth
    initial_guess = [data['x_m'].mean(), data['y_m'].mean(), -50]  # 50m depth
    
    # Try multiple starting points to avoid local minima
    best_result = None
    best_error = float('inf')
    
    # Try different depth estimates
    depth_estimates = [-25, -50, -75, -100, -150]
    
    for depth in depth_estimates:
        initial_guess = [data['x_m'].mean(), data['y_m'].mean(), depth]
        
        result = minimize(
            hemisphere_intersection_objective, 
            initial_guess, 
            args=(boat_positions, ranges),
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        
        if result.success and result.fun < best_error:
            best_result = result
            best_error = result.fun
    
    if best_result is None:
        # Fallback to simple approach
        return [data['x_m'].mean(), data['y_m'].mean(), -50]
    
    return best_result.x

def calculate_hemisphere_intersection_volume(data, target_pos, grid_resolution=10):
    """Calculate the volume of intersection of all hemispheres"""
    x, y, z = target_pos
    
    # Create a 3D grid around the target
    x_range = np.linspace(x - 100, x + 100, grid_resolution)
    y_range = np.linspace(y - 100, y + 100, grid_resolution)
    z_range = np.linspace(z - 50, 0, grid_resolution)  # Only underwater points
    
    xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
    
    # Count points that are inside ALL hemispheres
    intersection_points = 0
    total_points = 0
    
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            for k in range(grid_resolution):
                point_x, point_y, point_z = xx[i,j,k], yy[i,j,k], zz[i,j,k]
                
                # Check if point is inside all hemispheres
                inside_all = True
                for _, row in data.iterrows():
                    distance = np.sqrt((point_x - row['x_m'])**2 + 
                                     (point_y - row['y_m'])**2 + 
                                     point_z**2)
                    if distance > row['range_m'] or point_z > 0:
                        inside_all = False
                        break
                
                if inside_all:
                    intersection_points += 1
                total_points += 1
    
    return intersection_points / total_points if total_points > 0 else 0

def create_hemisphere_visualization(data, target_pos, lat_center, lon_center):
    """Create visualization showing hemisphere intersections"""
    fig = plt.figure(figsize=(18, 12))
    
    # 3D plot showing hemispheres and intersection
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Depth (m)')
    ax1.set_title('Hemisphere Intersection Analysis')
    
    # Plot boat positions at surface
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    for i, (_, row) in enumerate(data.iterrows()):
        ax1.scatter(row['x_m'], row['y_m'], 0, 
                   c=[colors[i]], s=100, marker='o', label=f'Boat {i+1}')
        
        # Draw lines from boat to target
        ax1.plot([row['x_m'], target_pos[0]], 
                [row['y_m'], target_pos[1]], 
                [0, target_pos[2]], 
                '--', alpha=0.5, color=colors[i])
    
    # Plot target location
    ax1.scatter(target_pos[0], target_pos[1], target_pos[2], 
               c='red', s=300, marker='*', label='Target Estimate')
    
    ax1.legend()
    ax1.invert_zaxis()
    
    # 2D top view with range circles
    ax2 = fig.add_subplot(232)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View - Range Circles')
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
    
    ax2.scatter(target_pos[0], target_pos[1], 
               c='red', s=300, marker='*', label='Target Estimate')
    ax2.legend()
    ax2.set_aspect('equal')
    
    # Depth vs Range analysis
    ax3 = fig.add_subplot(233)
    surface_distances = np.sqrt((data['x_m'] - target_pos[0])**2 + 
                               (data['y_m'] - target_pos[1])**2)
    actual_ranges = data['range_m']
    
    ax3.scatter(surface_distances, actual_ranges, 
               c=colors, s=100, alpha=0.7)
    ax3.plot([0, surface_distances.max()], 
            [0, surface_distances.max()], 
            'k--', alpha=0.5, label='Surface distance = Range')
    ax3.set_xlabel('Surface Distance (m)')
    ax3.set_ylabel('Measured Range (m)')
    ax3.set_title('Range vs Surface Distance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Hemisphere intersection quality
    ax4 = fig.add_subplot(234)
    boat_positions = list(zip(data['x_m'], data['y_m']))
    ranges = data['range_m'].values
    
    # Calculate how well each point fits within its hemisphere
    hemisphere_errors = []
    for i, (_, row) in enumerate(data.iterrows()):
        distance_3d = np.sqrt((target_pos[0] - row['x_m'])**2 + 
                             (target_pos[1] - row['y_m'])**2 + 
                             target_pos[2]**2)
        if distance_3d > row['range_m']:
            error = distance_3d - row['range_m']
        else:
            error = 0
        hemisphere_errors.append(error)
    
    ax4.bar(range(len(data)), hemisphere_errors, color=colors, alpha=0.7)
    ax4.set_xlabel('Measurement Point')
    ax4.set_ylabel('Distance Outside Hemisphere (m)')
    ax4.set_title('Hemisphere Fit Quality')
    ax4.grid(True, alpha=0.3)
    
    # Confidence map (2D slice at target depth)
    ax5 = fig.add_subplot(235)
    x_min, x_max = data['x_m'].min() - 100, data['x_m'].max() + 100
    y_min, y_max = data['y_m'].min() - 100, data['y_m'].max() + 100
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                        np.linspace(y_min, y_max, 50))
    
    confidence = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            point_x, point_y = xx[i,j], yy[i,j]
            point_z = target_pos[2]  # Same depth as target
            
            # Count how many hemispheres contain this point
            inside_count = 0
            for _, row in data.iterrows():
                distance = np.sqrt((point_x - row['x_m'])**2 + 
                                 (point_y - row['y_m'])**2 + 
                                 point_z**2)
                if distance <= row['range_m'] and point_z <= 0:
                    inside_count += 1
            
            confidence[i,j] = inside_count / len(data)
    
    contour = ax5.contourf(xx, yy, confidence, levels=10, alpha=0.6, cmap='Reds')
    ax5.contour(xx, yy, confidence, levels=[0.5, 0.7, 0.9], colors='red', linewidths=2)
    ax5.scatter(target_pos[0], target_pos[1], c='red', s=200, marker='*')
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_title(f'Confidence Map (Depth: {target_pos[2]:.1f}m)')
    ax5.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax5, shrink=0.8)
    cbar.set_label('Fraction of Hemispheres Containing Point')
    
    # Summary statistics
    ax6 = fig.add_subplot(236)
    ax6.axis('off')
    
    # Calculate intersection volume
    intersection_volume = calculate_hemisphere_intersection_volume(data, target_pos)
    
    # Convert target position to GPS coordinates
    target_lat = lat_center + target_pos[1] / 111000
    target_lon = lon_center + target_pos[0] / (111000 * np.cos(np.radians(lat_center)))
    
    summary_text = f'HEMISPHERE INTERSECTION RESULTS\n\n' \
                   f'Target Location:\n' \
                   f'• Latitude: {target_lat:.6f}°\n' \
                   f'• Longitude: {target_lon:.6f}°\n' \
                   f'• Depth: {target_pos[2]:.1f}m\n' \
                   f'• Horizontal distance: {np.sqrt(target_pos[0]**2 + target_pos[1]**2):.1f}m\n\n' \
                   f'Intersection Quality:\n' \
                   f'• Intersection volume: {intersection_volume:.3f}\n' \
                   f'• Max hemisphere error: {max(hemisphere_errors):.1f}m\n' \
                   f'• Points outside hemispheres: {sum(1 for e in hemisphere_errors if e > 0)}\n\n' \
                   f'Data Summary:\n' \
                   f'• Measurement points: {len(data)}\n' \
                   f'• Range: {data["range_m"].min():.1f}m - {data["range_m"].max():.1f}m\n' \
                   f'• Average range: {data["range_m"].mean():.1f}m'
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
    
    plt.tight_layout()
    return fig

def print_hemisphere_analysis(data, target_pos, lat_center, lon_center):
    """Print comprehensive hemisphere intersection analysis"""
    print("=" * 80)
    print("HEMISPHERE INTERSECTION ANALYSIS - UNDERWATER TARGET DETECTION")
    print("=" * 80)
    
    print(f"Method: Each range measurement is modeled as a hemisphere centered")
    print(f"        on the GPS coordinate at the surface, extending downward.")
    print(f"        Target location is found at the intersection of all hemispheres.\n")
    
    print(f"Data Summary:")
    print(f"  • Number of measurement points: {len(data)}")
    print(f"  • Range measurements: {data['range_m'].min():.1f}m to {data['range_m'].max():.1f}m")
    print(f"  • Average range: {data['range_m'].mean():.1f}m")
    
    # Convert target position to GPS coordinates
    target_lat = lat_center + target_pos[1] / 111000
    target_lon = lon_center + target_pos[0] / (111000 * np.cos(np.radians(lat_center)))
    
    print(f"\nTarget Location (Hemisphere Intersection):")
    print(f"  • Latitude:  {target_lat:.6f}°")
    print(f"  • Longitude: {target_lon:.6f}°")
    print(f"  • Depth: {target_pos[2]:.1f}m (below surface)")
    print(f"  • Horizontal distance from center: {np.sqrt(target_pos[0]**2 + target_pos[1]**2):.1f}m")
    
    # Calculate hemisphere fit quality
    boat_positions = list(zip(data['x_m'], data['y_m']))
    ranges = data['range_m'].values
    
    hemisphere_errors = []
    for i, (_, row) in enumerate(data.iterrows()):
        distance_3d = np.sqrt((target_pos[0] - row['x_m'])**2 + 
                             (target_pos[1] - row['y_m'])**2 + 
                             target_pos[2]**2)
        if distance_3d > row['range_m']:
            error = distance_3d - row['range_m']
        else:
            error = 0
        hemisphere_errors.append(error)
    
    print(f"\nHemisphere Intersection Quality:")
    print(f"  • Points outside hemispheres: {sum(1 for e in hemisphere_errors if e > 0)}/{len(data)}")
    print(f"  • Maximum error: {max(hemisphere_errors):.1f}m")
    print(f"  • Average error: {np.mean(hemisphere_errors):.1f}m")
    
    # Calculate intersection volume
    intersection_volume = calculate_hemisphere_intersection_volume(data, target_pos)
    print(f"  • Intersection volume ratio: {intersection_volume:.3f}")
    
    if intersection_volume > 0.1:
        print(f"  → Good intersection found - target location is well-constrained")
    elif intersection_volume > 0.01:
        print(f"  → Moderate intersection - target location has some uncertainty")
    else:
        print(f"  → Poor intersection - target location is uncertain")
    
    print("=" * 80)

def main():
    """Main function for hemisphere intersection analysis"""
    try:
        print("Loading underwater target detection data...")
        data, lat_center, lon_center = load_and_process_data('rawdata.csv')
        
        print("Finding hemisphere intersection...")
        target_pos = find_hemisphere_intersection(data)
        
        print_hemisphere_analysis(data, target_pos, lat_center, lon_center)
        
        print("\nCreating hemisphere intersection visualization...")
        fig = create_hemisphere_visualization(data, target_pos, lat_center, lon_center)
        
        plt.show()
        print("\nHemisphere intersection analysis complete!")
        
    except FileNotFoundError:
        print("Error: rawdata.csv not found in current directory")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 