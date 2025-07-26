import numpy as np
import json

# Define the corner points
upper_left_lat = 30.376052
upper_left_lon = -97.993006
lower_right_lat = 30.134890
lower_right_lon = -97.570954

# Define the grid resolution (adjust as needed)
lat_steps = 1000  # Number of steps between the upper and lower latitude
lon_steps = 1000  # Number of steps between the left and right longitude

# Generate the latitude and longitude values
latitudes = np.linspace(upper_left_lat, lower_right_lat, lat_steps)
longitudes = np.linspace(upper_left_lon, lower_right_lon, lon_steps)

# Create a meshgrid
lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

# Flatten the grid for easier handling
grid_points = np.vstack([lat_grid.ravel(), lon_grid.ravel()]).T

# Convert grid points to a list of dictionaries
grid_points_list = [{'latitude': lat, 'longitude': lon} for lat, lon in grid_points]

# Save to a .json file
with open('grid_points.json', 'w') as json_file:
    json.dump(grid_points_list, json_file, indent=4)

print("Grid points saved to grid_points.json")
