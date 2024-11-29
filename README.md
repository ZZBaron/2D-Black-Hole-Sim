# 2D Black Hole Light Ray Simulation

This Python simulation visualizes the behavior of light rays around a black hole in 2D space using the Schwarzschild metric. The simulation demonstrates gravitational lensing effects and how light rays are bent by the strong gravitational field of a black hole.

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

## Installation

1. Clone this repository:
```bash
git clone [your-repository-url]
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

The animation will show multiple light rays being bent around a central black hole, demonstrating the effects of gravitational lensing.

## Physics Background

The simulation uses the Schwarzschild metric to calculate the paths of light rays (null geodesics) in the presence of a black hole. The calculations take into account:
- Gravitational time dilation
- Space curvature effects
- Conservation of angular momentum
- Proper metric components for null geodesics

## License

MIT License