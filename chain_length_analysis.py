import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_chain_deployment():
    """Calculate chain deployment for a 380m chain from anchor to target"""
    
    # Anchor point coordinates
    anchor_lat = 19.82918
    anchor_lon = -156.12066
    anchor_depth = 400  # meters below surface
    
    # Target location
    target_lat = 19.830566
    target_lon = -156.119753
    target_depth = 171.1  # meters below surface
    
    # Chain specifications
    total_chain_length = 380  # meters
    
    # Convert to local coordinates
    lat_center = 19.8301333
    lon_center = -156.1173333
    
    anchor_x = (anchor_lon - lon_center) * 111000 * np.cos(np.radians(lat_center))
    anchor_y = (anchor_lat - lat_center) * 111000
    anchor_z = -anchor_depth
    
    target_x = (target_lon - lon_center) * 111000 * np.cos(np.radians(lat_center))
    target_y = (target_lat - lat_center) * 111000
    target_z = -target_depth
    
    # Calculate direct path requirements
    horizontal_distance = np.sqrt((target_x - anchor_x)**2 + (target_y - anchor_y)**2)
    vertical_distance = abs(target_z - anchor_z)
    direct_path_length = np.sqrt(horizontal_distance**2 + vertical_distance**2)
    
    print("=" * 70)
    print("CHAIN DEPLOYMENT ANALYSIS - 380m CHAIN (CATENARY ONLY)")
    print("=" * 70)
    
    print(f"Chain Specifications:")
    print(f"  • Total chain length: {total_chain_length}m")
    print(f"  • Direct path requirement: {direct_path_length:.1f}m")
    print(f"  • Excess chain available: {total_chain_length - direct_path_length:.1f}m")
    
    print(f"\nPath Requirements:")
    print(f"  • Horizontal distance: {horizontal_distance:.1f}m")
    print(f"  • Vertical distance: {vertical_distance:.1f}m")
    print(f"  • Minimum path length: {direct_path_length:.1f}m")
    
    # Calculate chain deployment with excess chain
    excess_chain = total_chain_length - direct_path_length
    
    if excess_chain > 0:
        print(f"\n✓ Sufficient chain length available!")
        print(f"  • Excess chain: {excess_chain:.1f}m")
        
        # Catenary curve analysis only
        print(f"\nCatenary Deployment Analysis:")
        
        # Estimate bottom portion based on excess chain
        # More excess chain = more bottom chain
        if excess_chain > horizontal_distance:
            # Plenty of excess - most can go on bottom
            bottom_chain_catenary = horizontal_distance * 0.8  # 80% of horizontal distance
        else:
            # Limited excess - proportional to available excess
            bottom_chain_catenary = excess_chain * 0.6  # 60% of excess goes to bottom
        
        angled_chain_catenary = total_chain_length - bottom_chain_catenary
        
        print(f"  • Bottom chain length: {bottom_chain_catenary:.1f}m")
        print(f"  • Angled chain length: {angled_chain_catenary:.1f}m")
        print(f"  • Chain utilization: {((bottom_chain_catenary + angled_chain_catenary) / total_chain_length * 100):.1f}%")
        
        # Calculate catenary lift-off point
        if bottom_chain_catenary > 0:
            direction_x = (target_x - anchor_x) / horizontal_distance
            direction_y = (target_y - anchor_y) / horizontal_distance
            
            catenary_lift_off_x = anchor_x + direction_x * bottom_chain_catenary
            catenary_lift_off_y = anchor_y + direction_y * bottom_chain_catenary
            catenary_lift_off_z = anchor_z
            
            print(f"  • Lift-off point: ({catenary_lift_off_x:.1f}m, {catenary_lift_off_y:.1f}m, {catenary_lift_off_z:.1f}m)")
            
            # Calculate catenary angle
            remaining_horizontal_cat = horizontal_distance - bottom_chain_catenary
            if remaining_horizontal_cat > 0:
                catenary_angle = np.arctan2(vertical_distance, remaining_horizontal_cat) * 180 / np.pi
                print(f"  • Chain angle at target: {catenary_angle:.1f}°")
            else:
                print(f"  • Chain angle: Vertical (90°)")
        else:
            catenary_lift_off_x, catenary_lift_off_y, catenary_lift_off_z = anchor_x, anchor_y, anchor_z
            catenary_angle = np.arctan2(vertical_distance, horizontal_distance) * 180 / np.pi
            print(f"  • Chain angle at target: {catenary_angle:.1f}°")
        
        # Calculate chain tension
        # NOTE: This is a simplified approximation with many assumptions
        # Real tension depends on: chain size/weight, current forces, target buoyancy, 
        # wave action, anchor holding capacity, and many other factors
        print(f"\nTension Analysis (Simplified Approximation):")
        print(f"  ⚠️  WARNING: This is a basic estimate with many assumptions!")
        print(f"  ⚠️  Real tension depends on chain specifications, currents, target buoyancy, etc.")
        
        # Basic assumptions for demonstration only
        chain_weight_per_meter = 50  # kg/m (assumed - actual depends on chain size)
        water_density = 1025  # kg/m³
        buoyant_weight_per_meter = chain_weight_per_meter - (chain_weight_per_meter / 7850) * water_density
        
        # Tension calculation for catenary (simplified)
        tension_catenary = buoyant_weight_per_meter * angled_chain_catenary * 9.81 / 1000  # kN
        
        print(f"  • Estimated catenary tension: {tension_catenary:.1f} kN")
        print(f"  • Assumptions: {chain_weight_per_meter} kg/m chain, no currents, no target forces")
        print(f"  • Reality: Tension could be 2-10x higher depending on conditions")
        
        # Additional factors that affect real tension
        print(f"\nFactors Affecting Real Tension:")
        print(f"  • Chain size and weight per meter (unknown)")
        print(f"  • Current forces on chain and target")
        print(f"  • Target buoyancy and drag forces")
        print(f"  • Wave action and surface conditions")
        print(f"  • Anchor holding capacity and seabed conditions")
        print(f"  • Dynamic loads from vessel movement")
        print(f"  • Temperature effects on chain properties")
        
        return {
            'total_chain': total_chain_length,
            'direct_path': direct_path_length,
            'excess_chain': excess_chain,
            'catenary': {
                'bottom_length': bottom_chain_catenary,
                'angled_length': angled_chain_catenary,
                'lift_off_point': [catenary_lift_off_x, catenary_lift_off_y, catenary_lift_off_z],
                'angle': catenary_angle,
                'tension_estimate': tension_catenary,
                'tension_note': 'Simplified estimate with many assumptions'
            },
            'anchor_pos': [anchor_x, anchor_y, anchor_z],
            'target_pos': [target_x, target_y, target_z],
            'horizontal_distance': horizontal_distance,
            'vertical_distance': vertical_distance
        }
        
    else:
        print(f"\n✗ Insufficient chain length!")
        print(f"  • Chain shortfall: {abs(excess_chain):.1f}m")
        print(f"  • Minimum required: {direct_path_length:.1f}m")
        print(f"  • Available: {total_chain_length}m")
        return None

def create_chain_deployment_visualization(chain_data):
    """Create visualization of catenary chain deployment"""
    fig = plt.figure(figsize=(18, 12))
    
    # 3D plot of catenary deployment
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Depth (m)')
    ax1.set_title('Catenary Chain Deployment (3D View)')
    
    anchor_x, anchor_y, anchor_z = chain_data['anchor_pos']
    target_x, target_y, target_z = chain_data['target_pos']
    
    # Catenary deployment
    lift_off_x, lift_off_y, lift_off_z = chain_data['catenary']['lift_off_point']
    
    # Plot anchor and target
    ax1.scatter(anchor_x, anchor_y, anchor_z, 
               c='blue', s=200, marker='s', label='Anchor (400m)')
    ax1.scatter(target_x, target_y, target_z, 
               c='red', s=200, marker='*', label='Target (171.1m)')
    
    # Plot catenary deployment
    if chain_data['catenary']['bottom_length'] > 0:
        ax1.plot([anchor_x, lift_off_x], [anchor_y, lift_off_y], [anchor_z, lift_off_z], 
                'g-', linewidth=4, label='Bottom Chain')
        ax1.plot([lift_off_x, target_x], [lift_off_y, target_y], [lift_off_z, target_z], 
                'g--', linewidth=3, label='Angled Chain')
        ax1.scatter(lift_off_x, lift_off_y, lift_off_z, 
                   c='lightgreen', s=150, marker='o', alpha=0.8, label='Lift-off Point')
    
    ax1.legend()
    ax1.invert_zaxis()
    
    # Top view of catenary deployment
    ax2 = fig.add_subplot(232)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View - Catenary Deployment')
    ax2.grid(True, alpha=0.3)
    
    ax2.scatter(anchor_x, anchor_y, c='blue', s=200, marker='s', label='Anchor')
    ax2.scatter(target_x, target_y, c='red', s=200, marker='*', label='Target')
    
    # Plot catenary deployment
    if chain_data['catenary']['bottom_length'] > 0:
        ax2.plot([anchor_x, lift_off_x], [anchor_y, lift_off_y], 
                'g-', linewidth=4, label=f'Bottom Chain: {chain_data["catenary"]["bottom_length"]:.1f}m')
        ax2.plot([lift_off_x, target_x], [lift_off_y, target_y], 
                'g--', linewidth=3, label=f'Angled Chain: {chain_data["catenary"]["angled_length"]:.1f}m')
        ax2.scatter(lift_off_x, lift_off_y, c='lightgreen', s=150, marker='o', alpha=0.8)
    
    ax2.legend()
    ax2.set_aspect('equal')
    
    # Depth profile of catenary deployment
    ax3 = fig.add_subplot(233)
    ax3.set_xlabel('Horizontal Distance (m)')
    ax3.set_ylabel('Depth (m)')
    ax3.set_title('Depth Profile - Catenary Deployment')
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()
    
    anchor_horizontal = 0
    target_horizontal = chain_data['horizontal_distance']
    
    # Plot catenary deployment
    if chain_data['catenary']['bottom_length'] > 0:
        lift_off_horizontal = chain_data['catenary']['bottom_length']
        ax3.plot([anchor_horizontal, lift_off_horizontal], 
                [abs(anchor_z), abs(lift_off_z)], 
                'g-', linewidth=4, label='Bottom Chain')
        ax3.plot([lift_off_horizontal, target_horizontal], 
                [abs(lift_off_z), abs(target_z)], 
                'g--', linewidth=3, label='Angled Chain')
        ax3.scatter(lift_off_horizontal, abs(lift_off_z), 
                   c='lightgreen', s=150, marker='o', alpha=0.8, label='Lift-off Point')
    
    ax3.scatter(anchor_horizontal, abs(anchor_z), c='blue', s=200, marker='s')
    ax3.scatter(target_horizontal, abs(target_z), c='red', s=200, marker='*')
    ax3.legend()
    
    # Chain length distribution
    ax4 = fig.add_subplot(234)
    categories = ['Bottom Chain', 'Angled Chain']
    lengths = [chain_data['catenary']['bottom_length'], chain_data['catenary']['angled_length']]
    colors = ['lightgreen', 'lightcoral']
    
    bars = ax4.bar(categories, lengths, color=colors, alpha=0.8)
    ax4.set_ylabel('Chain Length (m)')
    ax4.set_title('Chain Length Distribution')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, length in zip(bars, lengths):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{length:.1f}m', ha='center', va='bottom', fontweight='bold')
    
    # Chain utilization percentage
    utilization = ((chain_data['catenary']['bottom_length'] + chain_data['catenary']['angled_length']) / 
                  chain_data['total_chain'] * 100)
    ax4.text(0.5, 0.95, f'Chain Utilization: {utilization:.1f}%', 
             transform=ax4.transAxes, ha='center', va='top', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    
    # Tension analysis
    ax5 = fig.add_subplot(235)
    tension = chain_data['catenary']['tension_estimate']
    
    # Create a simple tension gauge with warning
    max_tension = 200  # kN for scale
    tension_percentage = (tension / max_tension) * 100
    
    # Create a gauge-like visualization
    theta = np.linspace(0, np.pi, 100)
    r = 1
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    ax5.plot(x, y, 'k-', linewidth=2)
    ax5.fill_between(x, 0, y, alpha=0.1, color='gray')
    
    # Tension indicator
    tension_angle = (tension_percentage / 100) * np.pi
    tension_x = 0.8 * np.cos(tension_angle)
    tension_y = 0.8 * np.sin(tension_angle)
    
    ax5.plot([0, tension_x], [0, tension_y], 'r-', linewidth=3)
    ax5.scatter(tension_x, tension_y, c='red', s=100, marker='o')
    
    ax5.set_xlim(-1.2, 1.2)
    ax5.set_ylim(0, 1.2)
    ax5.set_aspect('equal')
    ax5.set_title(f'Estimated Tension: {tension:.1f} kN\n(Many assumptions!)')
    ax5.text(0, -0.3, f'{tension:.1f} kN\nESTIMATE', ha='center', va='top', fontsize=10, fontweight='bold', color='red')
    ax5.text(0, -0.6, 'Real tension depends on:\n• Chain specifications\n• Current forces\n• Target buoyancy\n• Environmental conditions', 
             ha='center', va='top', fontsize=8, style='italic')
    ax5.axis('off')
    
    # Summary statistics
    ax6 = fig.add_subplot(236)
    ax6.axis('off')
    
    summary_text = f'380m CHAIN CATENARY ANALYSIS\n\n' \
                   f'Chain Specifications:\n' \
                   f'• Total chain length: {chain_data["total_chain"]}m\n' \
                   f'• Direct path required: {chain_data["direct_path"]:.1f}m\n' \
                   f'• Excess chain available: {chain_data["excess_chain"]:.1f}m\n\n' \
                   f'Catenary Deployment:\n' \
                   f'• Bottom chain: {chain_data["catenary"]["bottom_length"]:.1f}m\n' \
                   f'• Angled chain: {chain_data["catenary"]["angled_length"]:.1f}m\n' \
                   f'• Chain utilization: {utilization:.1f}%\n' \
                   f'• Chain angle at target: {chain_data["catenary"]["angle"]:.1f}°\n\n' \
                   f'Tension Analysis:\n' \
                   f'• Catenary tension: {chain_data["catenary"]["tension_estimate"]:.1f} kN\n' \
                   f'• Note: {chain_data["catenary"]["tension_note"]}\n\n' \
                   f'Key Findings:\n' \
                   f'• {chain_data["catenary"]["bottom_length"]/chain_data["total_chain"]*100:.1f}% of chain on bottom\n' \
                   f'• {chain_data["catenary"]["angled_length"]/chain_data["total_chain"]*100:.1f}% of chain angled\n' \
                   f'• Sufficient excess for good anchoring'
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.9))
    
    plt.tight_layout()
    return fig

def main():
    """Main function for chain deployment analysis"""
    try:
        print("Analyzing 500m chain deployment...")
        chain_data = calculate_chain_deployment()
        
        if chain_data:
            print("\nCreating deployment visualization...")
            fig = create_chain_deployment_visualization(chain_data)
            plt.show()
            print("\nChain deployment analysis complete!")
        else:
            print("\nAnalysis complete - insufficient chain length.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 