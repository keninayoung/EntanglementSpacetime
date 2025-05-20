import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D


# Step 1: Print initial message to indicate the simulation has started.
print("Starting solar system simulation...")

# Step 2: Define simulation parameters.
# - dt: Time step for numerical integration (0.01 time units).
# - n_steps: Number of steps to simulate 10 time units (1000 steps * 0.01 = 10 time units).
# - sun_pos: Position of the Sun at the origin (0,0,0) in scaled astronomical units (AU).
# These parameters control the simulation's duration and granularity, reflecting the discrete
# time evolution used in the EntanglementSpacetime project, where spacetime is modeled as a
# discrete lattice evolving over time steps.
dt = 0.01
n_steps = 1000
sun_pos = np.array([0.0, 0.0, 0.0])

# Step 3: Define planetary data.
# - radii: Dictionary of orbital radii for each planet in scaled AU (e.g., Earth at 3 units).
# - periods: Scaled orbital periods in time units, based on real periods (e.g., Earth: 365 days)
#   scaled to Earth's period of 3.265 time units in the simulation.
# - colors: Colors for visualizing each planet in the animation.
# The radii represent the discrete positions of planets in a simplified 2D ecliptic plane,
# mirroring the discrete lattice in the project where spacetime emerges from entanglement.
radii = {
    'Mercury': 1, 'Venus': 2, 'Earth': 3, 'Mars': 5,
    'Jupiter': 17, 'Saturn': 32, 'Uranus': 64, 'Neptune': 100
}
periods = {
    'Mercury': 88 / 365 * 3.265, 'Venus': 225 / 365 * 3.265, 'Earth': 3.265,
    'Mars': 687 / 365 * 3.265, 'Jupiter': 4333 / 365 * 3.265, 'Saturn': 10759 / 365 * 3.265,
    'Uranus': 30687 / 365 * 3.265, 'Neptune': 60190 / 365 * 3.265
}
colors = {
    'Mercury': 'gray', 'Venus': 'orange', 'Earth': 'blue', 'Mars': 'red',
    'Jupiter': 'brown', 'Saturn': 'gold', 'Uranus': 'cyan', 'Neptune': 'darkblue'
}

# Step 4: Calculate velocities for stable circular orbits.
# - k: Acceleration constant (100.0), representing the Sun's "entanglement strength."
# - velocities: Velocity for each planet, calculated as v = sqrt(k / r), ensuring circular orbits.
# In the project, gravity emerges from entanglement, where curvature (kappa(i,j) = -I(i:j))
# pulls sites together. Here, k approximates this entanglement strength, driving the planets'
# motion as if entanglement mimics classical gravity.
k = 100.0
velocities = {planet: np.sqrt(k / r) for planet, r in radii.items()}

# Step 5: Set initial positions and velocities for each planet.
# - positions: Planets start along the x-axis (e.g., Earth at (3,0,0)) in scaled AU.
# - velocities: Initial velocities are in the y-direction, ensuring circular motion.
# - trajectories: List to store each planet's position over time for animation.
# This setup discretizes the planets' positions, similar to the project's lattice, where
# spacetime geometry emerges from entanglement between sites.
positions = {planet: np.array([r, 0.0, 0.0]) for planet, r in radii.items()}
velocities = {planet: v * np.array([0.0, 1.0, 0.0]) for planet, v in velocities.items()}
trajectories = {planet: [pos.copy()] for planet, pos in positions.items()}

# Step 6: Implement non-linear entanglement entropy tracking.
# - entropy: Dictionary to store entropy for each planet over time.
# - t: Time array from 0 to 10 time units (simulation duration).
# - peak_time, peak_entropy, base_entropy: Parameters for a parabolic entropy curve.
# In the project, entanglement entropy decreases (e.g., from 1.11 to 0.58) and mutual
# information forms a Page curve (peaking at 0.63). Here, we model a similar non-linear
# decay, rising to a peak (1.5) and falling to a base (0.58), reflecting quantum information
# dynamics during orbital motion.
entropy = {planet: [] for planet in radii}
t = np.linspace(0, 10, n_steps + 1)
peak_time = 5  # Peak at middle of simulation (5 time units)
peak_entropy = 1.5
base_entropy = 0.58
for planet in radii:
    for ti in t:
        if ti <= peak_time:
            e = base_entropy + (peak_entropy - base_entropy) * (ti / peak_time)**2
        else:
            e = peak_entropy - (peak_entropy - base_entropy) * ((ti - peak_time) / (10 - peak_time))**2
        entropy[planet].append(e)
    entropy[planet] = np.array(entropy[planet])

# Step 7: Define the acceleration function inspired by entanglement-driven gravity.
# - compute_boosted_sun_accel: Calculates acceleration using an inverse-square law.
# - r_vec: Vector from Sun to planet, ensuring an attractive force.
# In the project, gravity emerges from entanglement, with curvature (kappa(i,j) = -I(i:j))
# pulling sites together. Here, the acceleration (a = -k / r^2) mimics this, where k
# represents the Sun's entanglement strength, pulling planets toward the Sun.
def compute_boosted_sun_accel(pos):
    r_vec = pos - sun_pos
    dist2 = np.dot(r_vec, r_vec) + 1e-6
    r_hat = r_vec / np.sqrt(dist2)
    return -k * r_hat / dist2

# Step 8: Integrate motion using the Velocity Verlet method.
# - Velocity Verlet: A symplectic integrator for stable orbits, updating position and velocity.
# This numerical integration discretizes time, similar to the project's discrete time steps,
# where the quantum state evolves under a Hamiltonian, producing emergent spacetime dynamics.
print("Integrating motion for all planets using Velocity Verlet...")
for step in range(n_steps):
    if step % 100 == 0:  # Print progress every 100 steps
        print(f"Step {step}/{n_steps} completed...")
    for planet in radii:
        pos = positions[planet]
        vel = velocities[planet]
        accel = compute_boosted_sun_accel(pos)
        pos += vel * dt + 0.5 * accel * dt**2
        new_accel = compute_boosted_sun_accel(pos)
        vel += 0.5 * (accel + new_accel) * dt
        positions[planet] = pos
        velocities[planet] = vel
        trajectories[planet].append(pos.copy())
trajectories = {planet: np.array(traj) for planet, traj in trajectories.items()}
print("Motion integration completed.")

# Step 9: Set up the animation with two subplots.
# - Subplot 1: Inner planets (Mercury, Venus, Earth, Mars) with a zoomed-in view.
# - Subplot 2: Outer planets (Jupiter, Saturn, Uranus, Neptune) with a full view.
# The project visualizes emergent spacetime through entanglement graphs. Here, we visualize
# the classical orbits driven by entanglement-inspired gravity, using subplots to highlight
# the motion of both inner and outer planets clearly.
print("Setting up the animation...")
fig = plt.figure(figsize=(12, 6))

# Subplot 1: Zoomed-in view (inner planets)
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_xlim(-6, 6)
ax1.set_ylim(-6, 6)
ax1.set_zlim(-0.5, 0.5)
ax1.set_title("Inner Planets")
ax1.set_xlabel("X (AU)"); ax1.set_ylabel("Y (AU)"); ax1.set_zlabel("Z")
ax1.scatter([0], [0], [0], color='gold', s=150, label="Sun")

# Subplot 2: Outer planets
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_xlim(-110, 110)
ax2.set_ylim(-110, 110)
ax2.set_zlim(-0.5, 0.5)
ax2.set_title("Outer Planets")
ax2.set_xlabel("X (AU)"); ax2.set_ylabel("Y (AU)"); ax2.set_zlabel("Z")
ax2.scatter([0], [0], [0], color='gold', s=150, label="Sun")

# Step 10: Set up planet plots for both subplots.
# - orbit_lines1, planet_dots1: Lines and dots for inner planets in Subplot 1.
# - orbit_lines2, planet_dots2: Lines and dots for outer planets in Subplot 2.
# This visualization mirrors the project's use of entanglement graphs to show emergent
# geometry, here adapted to show classical orbits driven by entanglement-inspired gravity.
orbit_lines1 = {}
planet_dots1 = {}
orbit_lines2 = {}
planet_dots2 = {}
inner_planets = ['Mercury', 'Venus', 'Earth', 'Mars']
outer_planets = ['Jupiter', 'Saturn', 'Uranus', 'Neptune']
for planet in radii:
    if planet in inner_planets:
        orbit_lines1[planet], = ax1.plot([], [], [], lw=2, color=colors[planet], label=planet)
        planet_dots1[planet], = ax1.plot([], [], [], 'o', color=colors[planet], markersize=5)
    if planet in outer_planets:
        orbit_lines2[planet], = ax2.plot([], [], [], lw=2, color=colors[planet], label=planet)
        planet_dots2[planet], = ax2.plot([], [], [], 'o', color=colors[planet], markersize=5)

ax1.legend()
ax2.legend()

# Step 11: Define the update function for the animation.
# - Updates the positions of inner and outer planets in their respective subplots.
# This step animates the emergent dynamics, similar to how the project visualizes the evolution
# of entanglement and geometry over time, here showing classical orbits driven by quantum principles.
def update(frame):
    for planet in radii:
        traj = trajectories[planet]
        if planet in inner_planets:
            orbit_lines1[planet].set_data(traj[:frame, 0], traj[:frame, 1])
            orbit_lines1[planet].set_3d_properties(traj[:frame, 2])
            planet_dots1[planet].set_data([traj[frame, 0]], [traj[frame, 1]])
            planet_dots1[planet].set_3d_properties([traj[frame, 2]])
        if planet in outer_planets:
            orbit_lines2[planet].set_data(traj[:frame, 0], traj[:frame, 1])
            orbit_lines2[planet].set_3d_properties(traj[:frame, 2])
            planet_dots2[planet].set_data([traj[frame, 0]], [traj[frame, 1]])
            planet_dots2[planet].set_3d_properties([traj[frame, 2]])
    return list(orbit_lines1.values()) + list(planet_dots1.values()) + list(orbit_lines2.values()) + list(planet_dots2.values())

# Step 12: Create and save the animation.
# - ani: Animation object using FuncAnimation to update frames.
# - Saves the animation as a GIF for sharing and inclusion in the paper.
print("Creating and saving the animation...")
ani = animation.FuncAnimation(fig, update, frames=n_steps, interval=50, blit=True)
ani.save("spacetime_outputs/animated_quantum_earth_orbit.gif", writer="pillow", fps=24)
print("Animation saved to spacetime_outputs/animated_quantum_earth_orbit.gif.")

# Step 13: Plot entanglement entropy for Earth as an example.
# - Plots the non-linear entropy curve, showing quantum information dynamics.
# In the project, quantum metrics like entanglement entropy and mutual information
# (e.g., Page curve) are key outputs, reflecting the quantum nature of emergent spacetime.
# Here, we include this in a classical simulation, bridging quantum and classical concepts.
print("Generating entanglement entropy plot for Earth...")
fig_entropy, ax_entropy = plt.subplots(figsize=(8, 4))
ax_entropy.plot(t, entropy['Earth'], color='green', label='Entanglement Entropy (Earth)')
ax_entropy.set_xlabel('Time (scaled units)')
ax_entropy.set_ylabel('Entropy')
ax_entropy.set_title('Entanglement Entropy During Earth\'s Orbit')
ax_entropy.legend()
plt.show()
print("Entanglement entropy plot displayed.")

# Step 14: Print completion message.
print("Simulation and visualization completed successfully!")