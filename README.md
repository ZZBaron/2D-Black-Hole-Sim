# 2D Black Hole Light Ray Simulation

This Python simulation visualizes the behavior of light rays around a black hole in 2D space using the Schwarzschild metric. The user can observe gravitational lensing effects in response to a strong gravitational field.

## Features

- Simulates light ray trajectories in Schwarzschild geometry
- Visualizes multiple light rays with different initial positions
- Animated visualization showing the progressive movement of light rays
- Uses proper relativistic calculations for null geodesics
- Includes numerical stability improvements for accurate simulation

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- SciPy

## Math

# Schwarzschild Metric
The Schwarzschild metric describes the geometry of spacetime around a non-rotating, chargeless, and spherically symmetric mass:

```math
 ds^2 = -\left(1-\frac{r_s}{r}\right)c^2dt^2 + \left(1-\frac{r_s}{r}\right)^{-1}dr^2 + r^2d\Omega^2 
```

where $r_s = \frac{2GM}{c^2}$ is the Schwarzschild radius which defines the event horizon of a black hole, $M$ is the mass of the black hole, $G$ is tvitational constant, and $c$ is the speed of light.
In the simulation, normalized units $G = c = 1$ are used, so the Schwarzschild radius becomes $r_s = 2M$.

# Geodesic Equations
Light rays follow null geodesics, which are paths where the spacetime interval is zero ($ds^2 = 0$). 
For motion in the equatorial plane ($\theta = \pi/2$), the geodesic equations reduce to:

```math
 \frac{dr}{d\lambda} = f(r) \cdot p_r 
 ```

```math
 \frac{d\phi}{d\lambda} = \frac{L}{r^2} 
```

```math
\frac{dp_r}{d\lambda} = \frac{L^2}{r^3}f(r) - \frac{r_s}{2r^2}f(r) 
```

```math
\frac{dL}{d\lambda} = 0 
```

where $f(r) = 1 - \frac{r_s}{r}$, $p_r$ is the radial momentum component, and $L$ is the angular momentum (conserved).

# Effective Potential
The motion of light rays can be understood through an effective potential:

```math
V_{eff}(r) = \frac{L^2}{2r^2}\left(1 - \frac{r_s}{r}\right) 
```

The closest approach of a light ray to the black hole is determined by the impact parameter $b = \frac{L}{E}$, where $E$ is the energy of the photon.

# Critical Radius
The photon sphere occurs at $r = \frac{3r_s}{2}$, where light rays can orbit the black hole in unstable circular orbits.
Light rays with impact parameters less than the critical value $b_{crit} = \frac{3\sqrt{3}r_s}{2}$ will be captured by the black hole.
            

## Installation

1. Clone this repository:
```bash
git clone https://github.com/ZZBaron/2D-Black-Hole-Sim.git
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulation by executing:
```bash
python "2D Black Hole Light Sim.py"
```


## License

MIT License