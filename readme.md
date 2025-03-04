# ArUco Interactive Table

Real-time marker detection system for interactive table applications using ArUco markers and computer vision.

## Overview

This project creates an interactive table surface that:
- Detects ArUco markers in real-time
- Calculates distances between markers
- Visualizes connections between markers
- Streams data to web applications

## Components

- **Camera Interface**: Supports both OAK-D and standard webcams
- **Marker Detection**: Uses OpenCV ArUco library
- **Data Processing**: Real-time position and distance calculation
- **Visualization**: OpenCV-based display with marker tracking

## Applications

- **VR/AR**: Data streaming to Three.js for 3D visualization
- **Interactive Surfaces**: Real-time marker tracking
- **Educational Tools**: Physical-digital interface
- **Prototyping**: Rapid spatial interface development

## Requirements

- Python 3.x
- OpenCV 4.x
- NumPy
- Web browser with WebGL support

## Installation

```bash
git clone [repository-url]
cd mesa_py
pip install -r requirements.txt
