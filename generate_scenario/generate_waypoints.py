import json
import numpy as np
from pykml import parser
import pickle
import yaml
import pandas as pd

def generate_nav_data(waypoints):
    nav_data = ""
    for i, (name, lat, lon) in enumerate(waypoints, start=1):
        nav_data += f"{i} {name} {lat} {lon} 190\n"  # Assuming 190 as altitude for VORs
        nav_data += f"{i} {name} {lat} {lon} 190 0 0\n"  # Assuming 190 as altitude for DMEs
    return nav_data

with open('/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/Austin_All_Waypoints.kml', 'r', encoding="utf-8") as f:
   root = parser.parse(f).getroot()
   

waypoints = {}
intersections = {}
route_data = {}
places = []
nav_data = ""
for place in root.Document.Folder.Placemark:
    separate_strings = str(place.Point.coordinates)
    separate_strings = separate_strings.split(",")
    waypoint_name = str(place.name)
    section_name = waypoint_name[0:4]
    inbound_intersection_name = waypoint_name[2:4]
    outbound_intersection_name = waypoint_name[0:2]
    if section_name not in route_data.keys():
        route_data[section_name] = []
    route_data[section_name].append([float(separate_strings[1]), float(separate_strings[0])])
    if section_name not in waypoints.keys():
        waypoints[section_name] = {}
        waypoints[section_name]['wpLat'] = []
        waypoints[section_name]['wpLat'].append(float(separate_strings[1]))
        waypoints[section_name]['wpLong'] = []
        waypoints[section_name]['wpLong'].append(float(separate_strings[0]))
    else:
        waypoints[section_name]['wpLat'].append(float(separate_strings[1]))
        waypoints[section_name]['wpLong'].append(float(separate_strings[0]))
    if inbound_intersection_name not in intersections.keys():
        intersections[inbound_intersection_name] = {}
        intersections[inbound_intersection_name]['inbound'] = []
        intersections[inbound_intersection_name]['inbound'].append(section_name)
    else:
        if 'inbound' not in intersections[inbound_intersection_name].keys():
            intersections[inbound_intersection_name]['inbound'] = []
        intersections[inbound_intersection_name]['inbound'].append(section_name)
    if outbound_intersection_name not in intersections.keys():
        intersections[outbound_intersection_name] = {}
        intersections[outbound_intersection_name]['outbound'] = []
        intersections[outbound_intersection_name]['outbound'].append(section_name)
    else:
        if 'outbound' not in intersections[outbound_intersection_name].keys():
            intersections[outbound_intersection_name]['outbound'] = []
        intersections[outbound_intersection_name]['outbound'].append(section_name)
    nav_data += f"2	{separate_strings[1]}	{separate_strings[0]}	0	277	25	0	{place.name}	N/A	NDB\n"

for place in root.Document.Folder.Folder.Placemark:
    separate_strings = str(place.Point.coordinates)
    separate_strings = separate_strings.split(",")
    intersections[place.name]['center'] = [float(separate_strings[1]), float(separate_strings[0])]
    nav_data += f"2	{separate_strings[1]}	{separate_strings[0]}	0	277	25	0	{place.name}	N/A	NDB\n"

yaml_data = {}
num_routes = 3

# Data to be written to YAML file
data = {'Routes': [], 'Intersections': []}

# Generate routes
for route in waypoints.keys():
    route = {
        'identifier': route,
        'sections': [route],
        'sectionWPs': [
            {
                'id': route,
                'max_slots': 1000,
                'wpLat': [waypoints[route]['wpLat'][0], waypoints[route]['wpLat'][1]],
                'wpLong': [waypoints[route]['wpLong'][0], waypoints[route]['wpLong'][1]]
            }
            ]
    }
    data['Routes'].append(route)

# File path to write YAML file
file_path = 'routes.yaml'

# Generate routes
for intersection in intersections.keys():
    intersection = {
        'identifier': intersection,
        'max_slots': 1000,
        'location': intersections[intersection]['center'],
        'radius': [waypoints[intersections[intersection]['outbound'][0]]['wpLat'][0], waypoints[intersections[intersection]['outbound'][0]]['wpLong'][1]],
        'inbound': intersections[intersection]['inbound'],
        'outbound': intersections[intersection]['outbound']
    }
    data['Intersections'].append(intersection)

# Writing data to YAML file
with open(file_path, 'w') as file:
    yaml.dump(data, file, default_flow_style=None)

file_path = 'new_route_data.pkl'

# Writing data to pickle file
with open(file_path, 'wb') as file:
    pickle.dump(route_data, file)

print("YAML file created successfully.")

with open("/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/bluesky/resources/navdata/nav_austin.dat", "w") as file:
    file.write(nav_data)
