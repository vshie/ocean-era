import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from scipy import stats

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

def hemisphere_surface_objective(target_pos, boat_positions, ranges):
    """
    Objective function for hemisphere surface intersection.
    Target must be ON the surface of each hemisphere (at exact range distance).
    For each boat position, calculate how well the target fits on the hemisphere surface.
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
        
        # Error: how far the target is from the hemisphere surface
        # Target should be exactly at the range distance
        surface_error = abs(distance_3d - range_m)
        total_error += surface_error**2
    
    return total_error

def find_hemisphere_surface_intersection(data):
    """Find the best intersection point on hemisphere surfaces"""
    boat_positions = list(zip(data['x_m'], data['y_m']))
    ranges = data['range_m'].values
    
    # Try multiple starting points to avoid local minima
    best_result = None
    best_error = float('inf')
    
    # Try different depth estimates and positions
    depth_estimates = [-25, -50, -75, -100, -150, -200]
    
    for depth in depth_estimates:
        # Try multiple starting positions
        for x_offset in [-50, 0, 50]:
            for y_offset in [-50, 0, 50]:
                initial_guess = [data['x_m'].mean() + x_offset, 
                               data['y_m'].mean() + y_offset, 
                               depth]
                
                result = minimize(
                    hemisphere_surface_objective, 
                    initial_guess, 
                    args=(boat_positions, ranges),
                    method='Nelder-Mead',
                    options={'maxiter': 2000}
                )
                
                if result.success and result.fun < best_error:
                    best_result = result
                    best_error = result.fun
    
    if best_result is None:
        # Fallback to simple approach
        return [data['x_m'].mean(), data['y_m'].mean(), -50]
    
    return best_result.x

def analyze_measurement_quality(data, target_pos):
    """Analyze the quality of each measurement and identify outliers"""
    boat_positions = list(zip(data['x_m'], data['y_m']))
    ranges = data['range_m'].values
    
    # Calculate surface errors for each measurement
    surface_errors = []
    for i, (boat_x, boat_y) in enumerate(boat_positions):
        range_m = ranges[i]
        distance_3d = np.sqrt((target_pos[0] - boat_x)**2 + 
                             (target_pos[1] - boat_y)**2 + 
                             target_pos[2]**2)
        surface_error = abs(distance_3d - range_m)
        surface_errors.append(surface_error)
    
    # Calculate statistics for outlier detection
    errors_array = np.array(surface_errors)
    mean_error = np.mean(errors_array)
    std_error = np.std(errors_array)
    
    # Identify outliers using z-score method
    z_scores = np.abs((errors_array - mean_error) / std_error)
    outliers = z_scores > 2.0  # Points with z-score > 2 are considered outliers
    
    # Calculate contribution quality (inverse of error)
    contribution_quality = 1.0 / (1.0 + errors_array)
    
    # Rank measurements by quality
    quality_ranking = np.argsort(contribution_quality)[::-1]
    
    return {
        'surface_errors': surface_errors,
        'mean_error': mean_error,
        'std_error': std_error,
        'outliers': outliers,
        'contribution_quality': contribution_quality,
        'quality_ranking': quality_ranking,
        'z_scores': z_scores
    }

def calculate_intersection_confidence(data, target_pos, quality_analysis):
    """Calculate confidence in the intersection based on measurement quality"""
    # Weighted average of surface errors (better measurements weighted more)
    weights = quality_analysis['contribution_quality']
    weighted_error = np.average(quality_analysis['surface_errors'], weights=weights)
    
    # Calculate how many measurements are "good" (error < threshold)
    good_threshold = quality_analysis['mean_error'] + quality_analysis['std_error']
    good_measurements = sum(1 for e in quality_analysis['surface_errors'] if e < good_threshold)
    
    # Calculate intersection tightness (how close all hemispheres come together)
    boat_positions = list(zip(data['x_m'], data['y_m']))
    ranges = data['range_m'].values
    
    # Find the minimum distance between any two hemisphere surfaces near the target
    min_surface_distance = float('inf')
    for i in range(len(boat_positions)):
        for j in range(i+1, len(boat_positions)):
            # Calculate distance between hemisphere surfaces at target depth
            boat1_x, boat1_y = boat_positions[i]
            boat2_x, boat2_y = boat_positions[j]
            range1, range2 = ranges[i], ranges[j]
            
            # Distance between hemisphere centers
            center_distance = np.sqrt((boat1_x - boat2_x)**2 + (boat1_y - boat2_y)**2)
            
            # Distance between hemisphere surfaces (if they intersect)
            if abs(range1 - range2) <= center_distance <= (range1 + range2):
                surface_distance = abs(range1 - range2)
                min_surface_distance = min(min_surface_distance, surface_distance)
    
    return {
        'weighted_error': weighted_error,
        'good_measurements': good_measurements,
        'total_measurements': len(data),
        'intersection_tightness': min_surface_distance if min_surface_distance != float('inf') else None
    }

def create_hemisphere_surface_visualization(data, target_pos, lat_center, lon_center, quality_analysis, confidence_metrics):
    """Create comprehensive visualization with quality analysis"""
    fig = plt.figure(figsize=(20, 15))
    
    # 3D plot showing hemisphere surfaces and intersection
    ax1 = fig.add_subplot(331, projection='3d')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Depth (m)')
    ax1.set_title('Hemisphere Surface Intersection')
    
    # Plot boat positions at surface
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    for i, (_, row) in enumerate(data.iterrows()):
        # Color based on quality (red for outliers, green for good measurements)
        if quality_analysis['outliers'][i]:
            color = 'red'
            alpha = 0.3
        else:
            color = colors[i]
            alpha = 0.7
            
        ax1.scatter(row['x_m'], row['y_m'], 0, 
                   c=[color], s=100, marker='o', alpha=alpha)
        
        # Draw lines from boat to target
        ax1.plot([row['x_m'], target_pos[0]], 
                [row['y_m'], target_pos[1]], 
                [0, target_pos[2]], 
                '--', alpha=0.5, color=color)
    
    # Plot target location
    ax1.scatter(target_pos[0], target_pos[1], target_pos[2], 
               c='red', s=300, marker='*', label='Target Estimate')
    
    ax1.legend()
    ax1.invert_zaxis()
    
    # 2D top view with range circles
    ax2 = fig.add_subplot(332)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View - Range Circles')
    ax2.grid(True, alpha=0.3)
    
    for i, (_, row) in enumerate(data.iterrows()):
        if quality_analysis['outliers'][i]:
            color = 'red'
            alpha = 0.3
            linewidth = 1
        else:
            color = colors[i]
            alpha = 0.6
            linewidth = 2
            
        ax2.scatter(row['x_m'], row['y_m'], 
                   c=[color], s=100, marker='o', alpha=alpha)
        
        # Range circles (2D projection)
        circle = patches.Circle((row['x_m'], row['y_m']), 
                              row['range_m'], 
                              fill=False, 
                              edgecolor=color, 
                              alpha=alpha, 
                              linewidth=linewidth)
        ax2.add_patch(circle)
    
    ax2.scatter(target_pos[0], target_pos[1], 
               c='red', s=300, marker='*', label='Target Estimate')
    ax2.legend()
    ax2.set_aspect('equal')
    
    # Surface error analysis
    ax3 = fig.add_subplot(333)
    measurement_points = range(len(data))
    errors = quality_analysis['surface_errors']
    outlier_mask = quality_analysis['outliers']
    
    # Plot errors with outliers highlighted
    ax3.bar([i for i in measurement_points if not outlier_mask[i]], 
            [e for i, e in enumerate(errors) if not outlier_mask[i]], 
            color='blue', alpha=0.7, label='Good Measurements')
    ax3.bar([i for i in measurement_points if outlier_mask[i]], 
            [e for i, e in enumerate(errors) if outlier_mask[i]], 
            color='red', alpha=0.7, label='Outliers')
    
    # Add threshold line
    threshold = quality_analysis['mean_error'] + quality_analysis['std_error']
    ax3.axhline(y=threshold, color='orange', linestyle='--', 
                label=f'Outlier Threshold ({threshold:.1f}m)')
    
    ax3.set_xlabel('Measurement Point')
    ax3.set_ylabel('Surface Error (m)')
    ax3.set_title('Hemisphere Surface Fit Quality')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Quality ranking
    ax4 = fig.add_subplot(334)
    ranking = quality_analysis['quality_ranking']
    qualities = quality_analysis['contribution_quality']
    
    ax4.bar(range(len(data)), [qualities[i] for i in ranking], 
            color=[colors[i] if not quality_analysis['outliers'][i] else 'red' for i in ranking],
            alpha=0.7)
    ax4.set_xlabel('Measurement Rank (Best to Worst)')
    ax4.set_ylabel('Contribution Quality')
    ax4.set_title('Measurement Quality Ranking')
    ax4.grid(True, alpha=0.3)
    
    # Z-score distribution
    ax5 = fig.add_subplot(335)
    z_scores = quality_analysis['z_scores']
    ax5.bar(range(len(data)), z_scores, 
            color=['red' if outlier else 'blue' for outlier in quality_analysis['outliers']],
            alpha=0.7)
    ax5.axhline(y=2.0, color='orange', linestyle='--', label='Outlier Threshold (z=2)')
    ax5.set_xlabel('Measurement Point')
    ax5.set_ylabel('Z-Score')
    ax5.set_title('Error Z-Score Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Confidence map at target depth
    ax6 = fig.add_subplot(336)
    x_min, x_max = data['x_m'].min() - 100, data['x_m'].max() + 100
    y_min, y_max = data['y_m'].min() - 100, data['y_m'].max() + 100
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                        np.linspace(y_min, y_max, 50))
    
    confidence = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            point_x, point_y = xx[i,j], yy[i,j]
            point_z = target_pos[2]  # Same depth as target
            
            # Calculate weighted confidence based on distance to hemisphere surfaces
            total_confidence = 0
            total_weight = 0
            
            for k, (_, row) in enumerate(data.iterrows()):
                distance = np.sqrt((point_x - row['x_m'])**2 + 
                                 (point_y - row['y_m'])**2 + 
                                 point_z**2)
                
                # Confidence is highest when point is exactly on hemisphere surface
                surface_error = abs(distance - row['range_m'])
                point_confidence = np.exp(-surface_error / 50.0)  # Decay with error
                
                # Weight by measurement quality
                weight = quality_analysis['contribution_quality'][k]
                total_confidence += point_confidence * weight
                total_weight += weight
            
            confidence[i,j] = total_confidence / total_weight if total_weight > 0 else 0
    
    contour = ax6.contourf(xx, yy, confidence, levels=15, alpha=0.6, cmap='Reds')
    ax6.contour(xx, yy, confidence, levels=[0.5, 0.7, 0.9], colors='red', linewidths=2)
    ax6.scatter(target_pos[0], target_pos[1], c='red', s=200, marker='*')
    ax6.set_xlabel('X (m)')
    ax6.set_ylabel('Y (m)')
    ax6.set_title(f'Weighted Confidence Map (Depth: {target_pos[2]:.1f}m)')
    ax6.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax6, shrink=0.8)
    cbar.set_label('Weighted Confidence')
    
    # Summary statistics
    ax7 = fig.add_subplot(337)
    ax7.axis('off')
    
    # Convert target position to GPS coordinates
    target_lat = lat_center + target_pos[1] / 111000
    target_lon = lon_center + target_pos[0] / (111000 * np.cos(np.radians(lat_center)))
    
    summary_text = f'HEMISPHERE SURFACE INTERSECTION RESULTS\n\n' \
                   f'Target Location:\n' \
                   f'• Latitude: {target_lat:.6f}°\n' \
                   f'• Longitude: {target_lon:.6f}°\n' \
                   f'• Depth: {target_pos[2]:.1f}m\n' \
                   f'• Horizontal distance: {np.sqrt(target_pos[0]**2 + target_pos[1]**2):.1f}m\n\n' \
                   f'Surface Fit Quality:\n' \
                   f'• Mean surface error: {quality_analysis["mean_error"]:.1f}m\n' \
                   f'• Weighted error: {confidence_metrics["weighted_error"]:.1f}m\n' \
                   f'• Outliers: {sum(quality_analysis["outliers"])}/{len(data)}\n' \
                   f'• Good measurements: {confidence_metrics["good_measurements"]}/{len(data)}\n\n' \
                   f'Intersection Confidence:\n' \
                   f'• Intersection tightness: {confidence_metrics["intersection_tightness"]:.1f}m\n' \
                   f'• Best measurement: Point {quality_analysis["quality_ranking"][0]+1}\n' \
                   f'• Worst measurement: Point {quality_analysis["quality_ranking"][-1]+1}'
    
    ax7.text(0.1, 0.9, summary_text, transform=ax7.transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
    
    # Individual measurement details
    ax8 = fig.add_subplot(338)
    ax8.axis('off')
    
    details_text = f'MEASUREMENT DETAILS\n\n'
    for i in quality_analysis['quality_ranking']:
        status = "OUTLIER" if quality_analysis['outliers'][i] else "GOOD"
        color = "red" if quality_analysis['outliers'][i] else "green"
        details_text += f'Point {i+1}: {quality_analysis["surface_errors"][i]:.1f}m error ({status})\n'
    
    ax8.text(0.1, 0.9, details_text, transform=ax8.transAxes, 
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
    
    # Error distribution
    ax9 = fig.add_subplot(339)
    errors = quality_analysis['surface_errors']
    ax9.hist(errors, bins=10, alpha=0.7, color='blue', edgecolor='black')
    ax9.axvline(quality_analysis['mean_error'], color='red', linestyle='--', 
                label=f'Mean: {quality_analysis["mean_error"]:.1f}m')
    ax9.axvline(threshold, color='orange', linestyle='--', 
                label=f'Threshold: {threshold:.1f}m')
    ax9.set_xlabel('Surface Error (m)')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Error Distribution')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_hemisphere_surface_analysis(data, target_pos, lat_center, lon_center, quality_analysis, confidence_metrics):
    """Print comprehensive hemisphere surface analysis"""
    print("=" * 80)
    print("HEMISPHERE SURFACE INTERSECTION ANALYSIS - UNDERWATER TARGET DETECTION")
    print("=" * 80)
    
    print(f"Method: Each range measurement defines a hemisphere SURFACE centered")
    print(f"        on the GPS coordinate at the surface, extending downward.")
    print(f"        Target must be ON the surface of each hemisphere (exact range).")
    print(f"        Target location is found at the intersection of hemisphere surfaces.\n")
    
    print(f"Data Summary:")
    print(f"  • Number of measurement points: {len(data)}")
    print(f"  • Range measurements: {data['range_m'].min():.1f}m to {data['range_m'].max():.1f}m")
    print(f"  • Average range: {data['range_m'].mean():.1f}m")
    
    # Convert target position to GPS coordinates
    target_lat = lat_center + target_pos[1] / 111000
    target_lon = lon_center + target_pos[0] / (111000 * np.cos(np.radians(lat_center)))
    
    print(f"\nTarget Location (Hemisphere Surface Intersection):")
    print(f"  • Latitude:  {target_lat:.6f}°")
    print(f"  • Longitude: {target_lon:.6f}°")
    print(f"  • Depth: {target_pos[2]:.1f}m (below surface)")
    print(f"  • Horizontal distance from center: {np.sqrt(target_pos[0]**2 + target_pos[1]**2):.1f}m")
    
    print(f"\nSurface Fit Quality:")
    print(f"  • Mean surface error: {quality_analysis['mean_error']:.1f}m")
    print(f"  • Standard deviation: {quality_analysis['std_error']:.1f}m")
    print(f"  • Weighted error: {confidence_metrics['weighted_error']:.1f}m")
    
    print(f"\nOutlier Analysis:")
    print(f"  • Outliers detected: {sum(quality_analysis['outliers'])}/{len(data)}")
    print(f"  • Good measurements: {confidence_metrics['good_measurements']}/{len(data)}")
    
    if sum(quality_analysis['outliers']) > 0:
        outlier_indices = [i+1 for i, is_outlier in enumerate(quality_analysis['outliers']) if is_outlier]
        print(f"  • Outlier points: {outlier_indices}")
    
    print(f"\nBest Contributors (Top 5):")
    for i, rank in enumerate(quality_analysis['quality_ranking'][:5]):
        error = quality_analysis['surface_errors'][rank]
        print(f"  {i+1}. Point {rank+1}: {error:.1f}m error")
    
    print(f"\nIntersection Confidence:")
    if confidence_metrics['intersection_tightness'] is not None:
        print(f"  • Intersection tightness: {confidence_metrics['intersection_tightness']:.1f}m")
        if confidence_metrics['intersection_tightness'] < 20:
            print(f"  → Very tight intersection - high confidence")
        elif confidence_metrics['intersection_tightness'] < 50:
            print(f"  → Moderate intersection - good confidence")
        else:
            print(f"  → Loose intersection - lower confidence")
    else:
        print(f"  • Hemispheres do not intersect tightly")
    
    print("=" * 80)

def main():
    """Main function for hemisphere surface intersection analysis"""
    try:
        print("Loading underwater target detection data...")
        data, lat_center, lon_center = load_and_process_data('rawdata.csv')
        
        print("Finding hemisphere surface intersection...")
        target_pos = find_hemisphere_surface_intersection(data)
        
        print("Analyzing measurement quality and identifying outliers...")
        quality_analysis = analyze_measurement_quality(data, target_pos)
        
        print("Calculating intersection confidence...")
        confidence_metrics = calculate_intersection_confidence(data, target_pos, quality_analysis)
        
        print_hemisphere_surface_analysis(data, target_pos, lat_center, lon_center, quality_analysis, confidence_metrics)
        
        print("\nCreating comprehensive visualization...")
        fig = create_hemisphere_surface_visualization(data, target_pos, lat_center, lon_center, quality_analysis, confidence_metrics)
        
        plt.show()
        print("\nHemisphere surface intersection analysis complete!")
        
    except FileNotFoundError:
        print("Error: rawdata.csv not found in current directory")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 