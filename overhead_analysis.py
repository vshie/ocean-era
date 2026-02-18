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

def analyze_overhead_position(data, target_pos):
    """Analyze if any boat position was directly over the target"""
    
    print("=" * 70)
    print("OVERHEAD POSITION ANALYSIS")
    print("=" * 70)
    
    # Calculate horizontal distance from each boat to target
    horizontal_distances = []
    for i, (_, row) in enumerate(data.iterrows()):
        boat_x, boat_y = row['x_m'], row['y_m']
        target_x, target_y = target_pos[0], target_pos[1]
        
        horizontal_dist = np.sqrt((target_x - boat_x)**2 + (target_y - boat_y)**2)
        horizontal_distances.append(horizontal_dist)
        
        print(f"Point {i+1}: {horizontal_dist:.1f}m horizontal distance to target")
    
    # Find the closest point
    min_dist_idx = np.argmin(horizontal_distances)
    min_distance = horizontal_distances[min_dist_idx]
    
    print(f"\nClosest boat position: Point {min_dist_idx+1}")
    print(f"Horizontal distance: {min_distance:.1f}m")
    
    # Check if this is "directly overhead" (within reasonable tolerance)
    overhead_threshold = 50  # meters
    is_overhead = min_distance < overhead_threshold
    
    if is_overhead:
        print(f"✓ Point {min_dist_idx+1} was DIRECTLY OVERHEAD the target!")
        print(f"  (within {overhead_threshold}m threshold)")
    else:
        print(f"✗ No boat position was directly overhead")
        print(f"  (closest was {min_distance:.1f}m away)")
    
    return min_dist_idx, min_distance, horizontal_distances

def calculate_intersection_at_point(data, target_pos, overhead_idx):
    """Calculate how well other hemisphere surfaces intersect at the overhead point"""
    
    print(f"\n" + "=" * 70)
    print("HEMISPHERE INTERSECTION ANALYSIS AT OVERHEAD POINT")
    print("=" * 70)
    
    overhead_x, overhead_y = data.iloc[overhead_idx]['x_m'], data.iloc[overhead_idx]['y_m']
    overhead_range = data.iloc[overhead_idx]['range_m']
    
    print(f"Overhead point: ({overhead_x:.1f}m, {overhead_y:.1f}m)")
    print(f"Overhead range: {overhead_range:.1f}m")
    print(f"Target depth: {target_pos[2]:.1f}m")
    
    # Calculate how well the target fits on each hemisphere surface
    surface_errors = []
    hemisphere_intersections = []
    
    for i, (_, row) in enumerate(data.iterrows()):
        boat_x, boat_y = row['x_m'], row['y_m']
        range_m = row['range_m']
        
        # Calculate 3D distance from boat to target
        distance_3d = np.sqrt((target_pos[0] - boat_x)**2 + 
                             (target_pos[1] - boat_y)**2 + 
                             target_pos[2]**2)
        
        # Error: how far from the hemisphere surface
        surface_error = abs(distance_3d - range_m)
        surface_errors.append(surface_error)
        
        # Check if target is within a small tolerance of the hemisphere surface
        tolerance = 10  # meters
        on_surface = surface_error < tolerance
        
        if i == overhead_idx:
            print(f"Point {i+1} (OVERHEAD): {surface_error:.1f}m error - {'ON SURFACE' if on_surface else 'OFF SURFACE'}")
        else:
            print(f"Point {i+1}: {surface_error:.1f}m error - {'ON SURFACE' if on_surface else 'OFF SURFACE'}")
        
        hemisphere_intersections.append(on_surface)
    
    # Calculate intersection statistics
    points_on_surface = sum(hemisphere_intersections)
    total_points = len(data)
    intersection_percentage = (points_on_surface / total_points) * 100
    
    print(f"\nIntersection Statistics:")
    print(f"  • Points on hemisphere surfaces: {points_on_surface}/{total_points}")
    print(f"  • Intersection percentage: {intersection_percentage:.1f}%")
    
    # Calculate confidence based on intersection quality
    mean_error = np.mean(surface_errors)
    std_error = np.std(surface_errors)
    
    print(f"  • Mean surface error: {mean_error:.1f}m")
    print(f"  • Standard deviation: {std_error:.1f}m")
    
    # Confidence calculation
    if intersection_percentage >= 90:
        confidence_level = "VERY HIGH"
        confidence_score = 95
    elif intersection_percentage >= 80:
        confidence_level = "HIGH"
        confidence_score = 85
    elif intersection_percentage >= 70:
        confidence_level = "MODERATE"
        confidence_score = 75
    elif intersection_percentage >= 60:
        confidence_level = "LOW"
        confidence_score = 65
    else:
        confidence_level = "VERY LOW"
        confidence_score = 45
    
    print(f"\nConfidence Assessment:")
    print(f"  • Confidence Level: {confidence_level}")
    print(f"  • Confidence Score: {confidence_score}%")
    
    # Additional analysis for overhead point
    if overhead_idx is not None:
        overhead_error = surface_errors[overhead_idx]
        print(f"\nOverhead Point Analysis:")
        print(f"  • Overhead point surface error: {overhead_error:.1f}m")
        
        if overhead_error < 5:
            print(f"  • ✓ Overhead point provides excellent constraint")
        elif overhead_error < 15:
            print(f"  • ~ Overhead point provides good constraint")
        else:
            print(f"  • ✗ Overhead point has significant error")
    
    return {
        'surface_errors': surface_errors,
        'intersection_percentage': intersection_percentage,
        'confidence_score': confidence_score,
        'confidence_level': confidence_level,
        'points_on_surface': points_on_surface,
        'total_points': total_points
    }

def create_overhead_visualization(data, target_pos, overhead_idx, intersection_stats):
    """Create visualization showing overhead analysis"""
    fig = plt.figure(figsize=(16, 12))
    
    # 2D top view with all boats and target
    ax1 = fig.add_subplot(221)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Boat Positions and Target Location')
    ax1.grid(True, alpha=0.3)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    
    for i, (_, row) in enumerate(data.iterrows()):
        if i == overhead_idx:
            # Highlight overhead point
            ax1.scatter(row['x_m'], row['y_m'], 
                       c='red', s=200, marker='s', 
                       label=f'Point {i+1} (OVERHEAD)', alpha=0.8)
        else:
            ax1.scatter(row['x_m'], row['y_m'], 
                       c=[colors[i]], s=100, marker='o', alpha=0.7)
    
    # Plot target location
    ax1.scatter(target_pos[0], target_pos[1], 
               c='red', s=300, marker='*', label='Target Estimate')
    
    # Draw line from overhead point to target
    if overhead_idx is not None:
        overhead_x = data.iloc[overhead_idx]['x_m']
        overhead_y = data.iloc[overhead_idx]['y_m']
        ax1.plot([overhead_x, target_pos[0]], [overhead_y, target_pos[1]], 
                'r--', linewidth=2, alpha=0.7)
    
    ax1.legend()
    ax1.set_aspect('equal')
    
    # Range circles showing hemisphere projections
    ax2 = fig.add_subplot(222)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Range Circles (2D Projection)')
    ax2.grid(True, alpha=0.3)
    
    for i, (_, row) in enumerate(data.iterrows()):
        if i == overhead_idx:
            color = 'red'
            alpha = 0.8
            linewidth = 3
        else:
            color = colors[i]
            alpha = 0.4
            linewidth = 2
            
        ax2.scatter(row['x_m'], row['y_m'], 
                   c=[color], s=100, marker='o', alpha=alpha)
        
        # Range circle
        circle = patches.Circle((row['x_m'], row['y_m']), 
                              row['range_m'], 
                              fill=False, 
                              edgecolor=color, 
                              alpha=alpha, 
                              linewidth=linewidth)
        ax2.add_patch(circle)
    
    ax2.scatter(target_pos[0], target_pos[1], 
               c='red', s=300, marker='*', label='Target')
    ax2.legend()
    ax2.set_aspect('equal')
    
    # Surface error analysis
    ax3 = fig.add_subplot(223)
    measurement_points = range(len(data))
    surface_errors = intersection_stats['surface_errors']
    
    # Color bars based on whether point is overhead
    bar_colors = ['red' if i == overhead_idx else colors[i] for i in range(len(data))]
    
    bars = ax3.bar(measurement_points, surface_errors, color=bar_colors, alpha=0.7)
    
    # Add threshold line
    threshold = 10
    ax3.axhline(y=threshold, color='orange', linestyle='--', 
                label=f'Surface Threshold ({threshold}m)')
    
    ax3.set_xlabel('Measurement Point')
    ax3.set_ylabel('Surface Error (m)')
    ax3.set_title('Hemisphere Surface Fit Quality')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Summary statistics
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    
    summary_text = f'OVERHEAD ANALYSIS RESULTS\n\n' \
                   f'Overhead Point:\n' \
                   f'• Point {overhead_idx+1 if overhead_idx is not None else "None"}\n' \
                   f'• Horizontal distance: {min_distance:.1f}m\n' \
                   f'• Directly overhead: {"Yes" if is_overhead else "No"}\n\n' \
                   f'Intersection Quality:\n' \
                   f'• Points on surfaces: {intersection_stats["points_on_surface"]}/{intersection_stats["total_points"]}\n' \
                   f'• Intersection %: {intersection_stats["intersection_percentage"]:.1f}%\n' \
                   f'• Mean error: {np.mean(surface_errors):.1f}m\n\n' \
                   f'Confidence:\n' \
                   f'• Level: {intersection_stats["confidence_level"]}\n' \
                   f'• Score: {intersection_stats["confidence_score"]}%\n\n' \
                   f'Conclusion:\n' \
                   f'• {"High confidence" if intersection_stats["confidence_score"] >= 80 else "Moderate confidence" if intersection_stats["confidence_score"] >= 60 else "Low confidence"} in target location'
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
    
    plt.tight_layout()
    return fig

def main():
    """Main function for overhead analysis"""
    try:
        print("Loading data and analyzing overhead positions...")
        data, lat_center, lon_center = load_and_process_data('rawdata.csv')
        
        # Use the target position from hemisphere surface analysis
        target_pos = [-255.8, 52.1, -174.8]  # From previous analysis
        
        # Analyze overhead position
        overhead_idx, min_distance, horizontal_distances = analyze_overhead_position(data, target_pos)
        
        # Calculate intersection at that point
        intersection_stats = calculate_intersection_at_point(data, target_pos, overhead_idx)
        
        # Create visualization
        print("\nCreating overhead analysis visualization...")
        fig = create_overhead_visualization(data, target_pos, overhead_idx, intersection_stats)
        
        plt.show()
        print("\nOverhead analysis complete!")
        
    except FileNotFoundError:
        print("Error: rawdata.csv not found in current directory")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 