import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_anchor_chain_path():
    """Calculate the anchor chain path from anchor to target"""
    
    # Anchor point coordinates
    anchor_lat = 19.82918
    anchor_lon = -156.12066
    anchor_depth = 400  # meters below surface
    
    # Target location
    target_lat = 19.830566
    target_lon = -156.119753
    target_depth = 171.1  # meters below surface
    
    # Convert to local coordinates (meters) using the same center as previous analyses
    lat_center = 19.8301333
    lon_center = -156.1173333
    
    # Convert anchor point to local coordinates
    anchor_x = (anchor_lon - lon_center) * 111000 * np.cos(np.radians(lat_center))
    anchor_y = (anchor_lat - lat_center) * 111000
    anchor_z = -anchor_depth  # Negative for below surface
    
    # Convert target point to local coordinates
    target_x = (target_lon - lon_center) * 111000 * np.cos(np.radians(lat_center))
    target_y = (target_lat - lat_center) * 111000
    target_z = -target_depth  # Negative for below surface
    
    # Calculate distances
    horizontal_distance = np.sqrt((target_x - anchor_x)**2 + (target_y - anchor_y)**2)
    vertical_distance = abs(target_z - anchor_z)
    total_distance = np.sqrt(horizontal_distance**2 + vertical_distance**2)
    
    # Calculate angles
    elevation_angle = np.arctan2(vertical_distance, horizontal_distance) * 180 / np.pi
    azimuth_angle = np.arctan2(target_y - anchor_y, target_x - anchor_x) * 180 / np.pi
    
    print("=" * 70)
    print("ANCHOR CHAIN PATH ANALYSIS")
    print("=" * 70)
    
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
    
    print(f"\nPath Analysis:")
    print(f"  • Horizontal distance: {horizontal_distance:.1f}m")
    print(f"  • Vertical distance:   {vertical_distance:.1f}m")
    print(f"  • Total path length:   {total_distance:.1f}m")
    print(f"  • Elevation angle:     {elevation_angle:.1f}°")
    print(f"  • Azimuth angle:       {azimuth_angle:.1f}°")
    
    return {
        'anchor_pos': [anchor_x, anchor_y, anchor_z],
        'target_pos': [target_x, target_y, target_z],
        'horizontal_distance': horizontal_distance,
        'vertical_distance': vertical_distance,
        'total_distance': total_distance,
        'elevation_angle': elevation_angle,
        'azimuth_angle': azimuth_angle
    }

def analyze_anchor_chain_geometry(path_data):
    """Analyze the anchor chain geometry assuming a catenary curve"""
    
    anchor_x, anchor_y, anchor_z = path_data['anchor_pos']
    target_x, target_y, target_z = path_data['target_pos']
    horizontal_distance = path_data['horizontal_distance']
    vertical_distance = path_data['vertical_distance']
    
    print(f"\n" + "=" * 70)
    print("ANCHOR CHAIN GEOMETRY ANALYSIS")
    print("=" * 70)
    
    # For a simplified analysis, we'll assume the chain follows a catenary curve
    # The chain will have a horizontal section on the bottom, then angle up
    
    # Calculate the point where the chain starts to angle up
    # This is where the chain would naturally lift off the bottom
    # For a typical anchor chain, this happens when the angle exceeds ~15-20 degrees
    
    # Let's calculate the chain geometry step by step
    
    # 1. Bottom section (horizontal on seabed)
    # The chain will lie on the bottom until it needs to angle up to reach the target
    # For a catenary, the horizontal force component determines the lift-off point
    
    # Simplified calculation: assume chain lifts off when angle to horizontal exceeds 15°
    lift_off_angle = 15  # degrees
    lift_off_distance = horizontal_distance - (vertical_distance / np.tan(np.radians(lift_off_angle)))
    
    if lift_off_distance > 0:
        bottom_chain_length = lift_off_distance
        angled_chain_length = np.sqrt((horizontal_distance - lift_off_distance)**2 + vertical_distance**2)
        chain_angle = np.arctan2(vertical_distance, horizontal_distance - lift_off_distance) * 180 / np.pi
    else:
        # Chain angles up immediately from anchor
        bottom_chain_length = 0
        angled_chain_length = np.sqrt(horizontal_distance**2 + vertical_distance**2)
        chain_angle = np.arctan2(vertical_distance, horizontal_distance) * 180 / np.pi
    
    print(f"Chain Geometry:")
    print(f"  • Bottom chain length (horizontal): {bottom_chain_length:.1f}m")
    print(f"  • Angled chain length: {angled_chain_length:.1f}m")
    print(f"  • Total chain length: {bottom_chain_length + angled_chain_length:.1f}m")
    print(f"  • Chain angle at target: {chain_angle:.1f}°")
    
    # Calculate lift-off point coordinates
    if bottom_chain_length > 0:
        # Calculate the direction vector from anchor to target
        direction_x = (target_x - anchor_x) / horizontal_distance
        direction_y = (target_y - anchor_y) / horizontal_distance
        
        # Lift-off point is bottom_chain_length along the horizontal direction
        lift_off_x = anchor_x + direction_x * bottom_chain_length
        lift_off_y = anchor_y + direction_y * bottom_chain_length
        lift_off_z = anchor_z  # Same depth as anchor (on bottom)
        
        print(f"\nLift-off Point:")
        print(f"  • Local X: {lift_off_x:.1f}m")
        print(f"  • Local Y: {lift_off_y:.1f}m")
        print(f"  • Local Z: {lift_off_z:.1f}m")
        print(f"  • Distance from anchor: {bottom_chain_length:.1f}m")
    else:
        lift_off_x, lift_off_y, lift_off_z = anchor_x, anchor_y, anchor_z
        print(f"\nLift-off Point: Same as anchor (chain angles up immediately)")
    
    # Calculate chain tension analysis (simplified)
    # Assume the chain is in static equilibrium
    chain_weight_per_meter = 50  # kg/m (typical for anchor chain)
    water_density = 1025  # kg/m³ (seawater)
    
    # Buoyant weight per meter
    buoyant_weight_per_meter = chain_weight_per_meter - (chain_weight_per_meter / 7850) * water_density
    
    # Simplified tension calculation
    # Tension increases with depth and chain length
    max_tension_depth = abs(anchor_z)  # Maximum depth
    estimated_tension = buoyant_weight_per_meter * angled_chain_length * 9.81  # Newtons
    
    print(f"\nChain Tension Analysis:")
    print(f"  • Chain weight per meter: {chain_weight_per_meter:.1f} kg/m")
    print(f"  • Buoyant weight per meter: {buoyant_weight_per_meter:.1f} kg/m")
    print(f"  • Estimated maximum tension: {estimated_tension/1000:.1f} kN")
    
    return {
        'bottom_chain_length': bottom_chain_length,
        'angled_chain_length': angled_chain_length,
        'total_chain_length': bottom_chain_length + angled_chain_length,
        'chain_angle': chain_angle,
        'lift_off_point': [lift_off_x, lift_off_y, lift_off_z],
        'estimated_tension': estimated_tension
    }

def create_chain_visualization(path_data, chain_data):
    """Create visualization of the anchor chain path"""
    fig = plt.figure(figsize=(16, 12))
    
    # 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Depth (m)')
    ax1.set_title('Anchor Chain Path (3D View)')
    
    anchor_x, anchor_y, anchor_z = path_data['anchor_pos']
    target_x, target_y, target_z = path_data['target_pos']
    lift_off_x, lift_off_y, lift_off_z = chain_data['lift_off_point']
    
    # Plot anchor and target
    ax1.scatter(anchor_x, anchor_y, anchor_z, 
               c='blue', s=200, marker='s', label='Anchor (400m depth)')
    ax1.scatter(target_x, target_y, target_z, 
               c='red', s=200, marker='*', label='Target (171.1m depth)')
    
    # Plot chain path
    if chain_data['bottom_chain_length'] > 0:
        # Bottom section (horizontal)
        ax1.plot([anchor_x, lift_off_x], [anchor_y, lift_off_y], [anchor_z, lift_off_z], 
                'b-', linewidth=3, label='Bottom Chain (Horizontal)')
        
        # Angled section
        ax1.plot([lift_off_x, target_x], [lift_off_y, target_y], [lift_off_z, target_z], 
                'r-', linewidth=3, label='Angled Chain')
        
        # Mark lift-off point
        ax1.scatter(lift_off_x, lift_off_y, lift_off_z, 
                   c='green', s=150, marker='o', label='Lift-off Point')
    else:
        # Direct angled path
        ax1.plot([anchor_x, target_x], [anchor_y, target_y], [anchor_z, target_z], 
                'r-', linewidth=3, label='Angled Chain')
    
    ax1.legend()
    ax1.invert_zaxis()
    
    # 2D top view
    ax2 = fig.add_subplot(222)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View - Chain Path')
    ax2.grid(True, alpha=0.3)
    
    ax2.scatter(anchor_x, anchor_y, c='blue', s=200, marker='s', label='Anchor')
    ax2.scatter(target_x, target_y, c='red', s=200, marker='*', label='Target')
    
    if chain_data['bottom_chain_length'] > 0:
        ax2.plot([anchor_x, lift_off_x], [anchor_y, lift_off_y], 
                'b-', linewidth=3, label=f'Bottom Chain ({chain_data["bottom_chain_length"]:.1f}m)')
        ax2.plot([lift_off_x, target_x], [lift_off_y, target_y], 
                'r-', linewidth=3, label=f'Angled Chain ({chain_data["angled_chain_length"]:.1f}m)')
        ax2.scatter(lift_off_x, lift_off_y, c='green', s=150, marker='o', label='Lift-off Point')
    else:
        ax2.plot([anchor_x, target_x], [anchor_y, target_y], 
                'r-', linewidth=3, label=f'Angled Chain ({chain_data["angled_chain_length"]:.1f}m)')
    
    ax2.legend()
    ax2.set_aspect('equal')
    
    # Side view (depth profile)
    ax3 = fig.add_subplot(223)
    ax3.set_xlabel('Horizontal Distance (m)')
    ax3.set_ylabel('Depth (m)')
    ax3.set_title('Depth Profile - Chain Path')
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()  # Invert so depth increases downward
    
    # Calculate horizontal distances for plotting
    anchor_horizontal = 0
    target_horizontal = path_data['horizontal_distance']
    
    if chain_data['bottom_chain_length'] > 0:
        lift_off_horizontal = chain_data['bottom_chain_length']
        
        ax3.plot([anchor_horizontal, lift_off_horizontal], 
                [abs(anchor_z), abs(lift_off_z)], 
                'b-', linewidth=3, label='Bottom Chain')
        ax3.plot([lift_off_horizontal, target_horizontal], 
                [abs(lift_off_z), abs(target_z)], 
                'r-', linewidth=3, label='Angled Chain')
        ax3.scatter(lift_off_horizontal, abs(lift_off_z), 
                   c='green', s=150, marker='o', label='Lift-off Point')
    else:
        ax3.plot([anchor_horizontal, target_horizontal], 
                [abs(anchor_z), abs(target_z)], 
                'r-', linewidth=3, label='Angled Chain')
    
    ax3.scatter(anchor_horizontal, abs(anchor_z), c='blue', s=200, marker='s', label='Anchor')
    ax3.scatter(target_horizontal, abs(target_z), c='red', s=200, marker='*', label='Target')
    ax3.legend()
    
    # Summary statistics
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    
    summary_text = f'ANCHOR CHAIN ANALYSIS RESULTS\n\n' \
                   f'Path Summary:\n' \
                   f'• Total horizontal distance: {path_data["horizontal_distance"]:.1f}m\n' \
                   f'• Vertical distance: {path_data["vertical_distance"]:.1f}m\n' \
                   f'• Total path length: {path_data["total_distance"]:.1f}m\n' \
                   f'• Elevation angle: {path_data["elevation_angle"]:.1f}°\n\n' \
                   f'Chain Geometry:\n' \
                   f'• Bottom chain length: {chain_data["bottom_chain_length"]:.1f}m\n' \
                   f'• Angled chain length: {chain_data["angled_chain_length"]:.1f}m\n' \
                   f'• Total chain length: {chain_data["total_chain_length"]:.1f}m\n' \
                   f'• Chain angle at target: {chain_data["chain_angle"]:.1f}°\n\n' \
                   f'Engineering:\n' \
                   f'• Estimated tension: {chain_data["estimated_tension"]/1000:.1f} kN\n' \
                   f'• Lift-off angle: 15°\n' \
                   f'• Chain weight: 50 kg/m'
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
    
    plt.tight_layout()
    return fig

def main():
    """Main function for anchor chain analysis"""
    try:
        print("Analyzing anchor chain path...")
        path_data = calculate_anchor_chain_path()
        
        print("Calculating chain geometry...")
        chain_data = analyze_anchor_chain_geometry(path_data)
        
        print("\nCreating chain visualization...")
        fig = create_chain_visualization(path_data, chain_data)
        
        plt.show()
        print("\nAnchor chain analysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 