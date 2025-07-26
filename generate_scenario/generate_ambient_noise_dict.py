import json
import yaml
import random


file_path = '/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/D2MAV_A/Austin_towers.yaml'

# Reading data from the YAML file
with open(file_path, 'r') as file:
    data = yaml.safe_load(file)
# print(data)
route_names = [route_vals['identifier'] for route_vals in data['Routes']]
intersection_names = [int_vals['identifier'] for int_vals in data['Intersections']]
name_list = route_names + intersection_names
ambient_noise_dict = {}
for name in name_list:
    ambient_noise_dict[name] = random.randint(40, 60)

# File path to save the dictionary
file_path = '/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/D2MAV_A/ambient_noise_dict.json'

# Writing the dictionary to the file
with open(file_path, 'w') as file:
    json.dump(ambient_noise_dict, file)

print("Dictionary saved to", file_path)
