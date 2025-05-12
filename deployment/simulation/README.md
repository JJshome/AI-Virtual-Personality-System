# Simulation Environment

This directory contains a simulation environment for testing and demonstrating the AI-based Virtual Personality System.

## Overview

The simulation environment provides a web-based interface for interacting with virtual personalities in different contexts. It allows users to test the core functionality of the system without requiring the full hardware setup.

## Features

- Virtual personality selection from predefined templates
- Multimodal interaction (text, voice, video) simulation
- Real-time emotion analysis visualization
- Domain-specific scenario simulations (entertainment, education, healthcare, etc.)
- Performance metrics and analytics

## Getting Started

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the simulation server:
   ```bash
   python simulation_server.py
   ```

3. Access the simulation interface in your browser:
   ```
   http://localhost:8000
   ```

## Simulation Scenarios

The simulation environment includes the following predefined scenarios:

- **Entertainment**: Interact with a virtual celebrity or game character
- **Education**: Learn from a virtual teacher or historical figure
- **Healthcare**: Consult with a virtual doctor or mental health counselor
- **Customer Service**: Get assistance from a virtual customer support agent
- **Custom**: Create and test your own virtual personality

## Configuration

You can customize the simulation environment by editing the `config.json` file to modify parameters such as:

- Response latency
- Emotion sensitivity
- Memory capacity
- Language model parameters
- Interaction style preferences

## Examples

See the [examples](examples/) directory for sample scripts and usage scenarios.
