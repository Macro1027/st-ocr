#!/bin/bash

# Update package list
sudo apt-get update

# Install OpenGL library
sudo apt-get install -y libgl1-mesa-glx

# Install other dependencies
pip install -r requirements.txt
