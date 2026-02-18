import numpy as np
import pandas as pd

def calculate_distances():
    """Calculate distances from anchor point to target location"""
    
    # Anchor point coordinates
    anchor_lat = 19.82918
    anchor_lon = -156.12066
    anchor_depth = 400  # meters below surface
    
    # Target location from hemisphere surface analysis
    target_lat = 19.830603
    target_lon = -156.119783
    target_depth = 174.8  # meters below surface
    
    # Convert to local coordinates (meters) using the same center as the analysis
    # From the analysis, we know the center coordinates
    lat_center = 19.8301333  # This is the mean of the data points
    lon_center = -156.1173333  # This is the mean of the data points
    
    # Convert anchor point to local coordinates
    anchor_x = (anchor_lon - lon_center) * 111000 * np.cos(np.radians(lat_center))
    anchor_y = (anchor_lat - lat_center) * 111000
    anchor_z = -anchor_depth  # Negative for below surface
    
    # Convert target point to local coordinates
    target_x = (target_lon - lon_center) * 111000 * np.cos(np.radians(lat_center))
    target_y = (target_lat - lat_center) * 111000
    target_z = -target_depth  # Negative for below surface
    
    # Calculate distances
    # Horizontal distance (ignoring depth)
    horizontal_distance = np.sqrt((target_x - anchor_x)**2 + (target_y - anchor_y)**2)
    
    # Vertical distance (depth difference)
    vertical_distance = abs(target_z - anchor_z)
    
    # Shortest path distance (3D distance)
    shortest_path_distance = np.sqrt((target_x - anchor_x)**2 + (target_y - anchor_y)**2 + (target_z - anchor_z)**2)
    
    # Calculate angles
    # Angle from horizontal (elevation angle)
    elevation_angle = np.arctan2(vertical_distance, horizontal_distance) * 180 / np.pi
    
    # Azimuth angle (bearing from anchor to target)
    azimuth_angle = np.arctan2(target_y - anchor_y, target_x - anchor_x) * 180 / np.pi
    
    # Print results
    print("=" * 60)
    print("DISTANCE CALCULATION: ANCHOR POINT TO TARGET")
    print("=" * 60)
    
    print(f"Anchor Point:")
    print(f"  • Latitude:  {anchor_lat:.6f}°")
    print(f"  • Longitude: {anchor_lon:.6f}°")
    print(f"  • Depth:     {anchor_depth:.1f}m below surface")
    print(f"  • Local X:   {anchor_x:.1f}m")
    print(f"  • Local Y:   {anchor_y:.1f}m")
    print(f"  • Local Z:   {anchor_z:.1f}m")
    
    print(f"\nTarget Location:")
    print(f"  • Latitude:  {target_lat:.6f}°")
    print(f"  • Longitude: {target_lon:.6f}°")
    print(f"  • Depth:     {target_depth:.1f}m below surface")
    print(f"  • Local X:   {target_x:.1f}m")
    print(f"  • Local Y:   {target_y:.1f}m")
    print(f"  • Local Z:   {target_z:.1f}m")
    
    print(f"\nDistance Calculations:")
    print(f"  • Horizontal Distance: {horizontal_distance:.1f}m")
    print(f"  • Vertical Distance:   {vertical_distance:.1f}m")
    print(f"  • Shortest Path:       {shortest_path_distance:.1f}m")
    
    print(f"\nAngles:")
    print(f"  • Elevation Angle:     {elevation_angle:.1f}°")
    print(f"  • Azimuth Angle:       {azimuth_angle:.1f}°")
    
    # Additional analysis
    print(f"\nAnalysis:")
    print(f"  • Target is {target_depth - anchor_depth:.1f}m shallower than anchor")
    print(f"  • Target is {horizontal_distance:.1f}m horizontally from anchor")
    print(f"  • The path from anchor to target has a slope of {elevation_angle:.1f}°")
    
    # Calculate the actual range that would be measured from anchor to target
    measured_range = shortest_path_distance
    print(f"  • If anchor had a range sensor, it would measure: {measured_range:.1f}m")
    
    print("=" * 60)
    
    return {
        'horizontal_distance': horizontal_distance,
        'vertical_distance': vertical_distance,
        'shortest_path_distance': shortest_path_distance,
        'elevation_angle': elevation_angle,
        'azimuth_angle': azimuth_angle
    }

def create_distance_visualization():
    """Create a visualization of the anchor-target relationship"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Anchor point coordinates
    anchor_lat = 19.82918
    anchor_lon = -156.12066
    anchor_depth = 400
    
    # Target location
    target_lat = 19.830603
    target_lon = -156.119783
    target_depth = 174.8
    
    # Convert to local coordinates
    lat_center = 19.8301333
    lon_center = -156.1173333
    
    anchor_x = (anchor_lon - lon_center) * 111000 * np.cos(np.radians(lat_center))
    anchor_y = (anchor_lat - lat_center) * 111000
    anchor_z = -anchor_depth
    
    target_x = (target_lon - lon_center) * 111000 * np.cos(np.radians(lat_center))
    target_y = (target_lat - lat_center) * 111000
    target_z = -target_depth
    
    # Create 3D visualization
    fig = plt.figure(figsize=(15, 10))
    
    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Depth (m)')
    ax1.set_title('Anchor Point to Target Distance')
    
    # Plot anchor point
    ax1.scatter(anchor_x, anchor_y, anchor_z, 
               c='blue', s=200, marker='s', label='Anchor Point (400m depth)')
    
    # Plot target point
    ax1.scatter(target_x, target_y, target_z, 
               c='red', s=200, marker='*', label='Target (174.8m depth)')
    
    # Draw line between points
    ax1.plot([anchor_x, target_x], [anchor_y, target_y], [anchor_z, target_z], 
             'k--', linewidth=2, alpha=0.7, label='Shortest Path')
    
    # Draw horizontal projection
    ax1.plot([anchor_x, target_x], [anchor_y, target_y], [anchor_z, anchor_z], 
             'g--', linewidth=1, alpha=0.5, label='Horizontal Distance')
    
    # Draw vertical line at target
    ax1.plot([target_x, target_x], [target_y, target_y], [anchor_z, target_z], 
             'orange', linewidth=1, alpha=0.5, label='Vertical Distance')
    
    ax1.legend()
    ax1.invert_zaxis()
    
    # 2D top view
    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View - Horizontal Distance')
    ax2.grid(True, alpha=0.3)
    
    # Plot points
    ax2.scatter(anchor_x, anchor_y, c='blue', s=200, marker='s', label='Anchor Point')
    ax2.scatter(target_x, target_y, c='red', s=200, marker='*', label='Target')
    
    # Draw horizontal line
    ax2.plot([anchor_x, target_x], [anchor_y, target_y], 'g--', linewidth=2, alpha=0.7)
    
    # Add distance annotation
    horizontal_dist = np.sqrt((target_x - anchor_x)**2 + (target_y - anchor_y)**2)
    ax2.annotate(f'{horizontal_dist:.1f}m', 
                xy=((anchor_x + target_x)/2, (anchor_y + target_y)/2),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax2.legend()
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Calculate distances
    distances = calculate_distances()
    
    # Create visualization
    print("\nCreating distance visualization...")
    fig = create_distance_visualization()
    plt.show()
    
    print("\nDistance calculation complete!") 