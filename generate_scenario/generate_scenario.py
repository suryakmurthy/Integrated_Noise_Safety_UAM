import ray
import os
# import gin
# import argparse
# from D2MAV_A.agent import Agent
# from D2MAV_A.runner import Runner
# from bluesky.tools import geo
from copy import deepcopy
import pandas as pd
import random
# import time
# import platform
import json
# import numpy as np
# import logging

class Communication_Node():
    import tensorflow as tf
    import bluesky as bs
    def __init__(self, scenario_file):
        self.bs.init(mode="sim", configfile=f'/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/settings.cfg')
        self.bs.net.connect()
        self.reset(scenario_file)
    def reset(self, scenario_file):
        self.bs.stack.stack(r'IC ' + scenario_file)
        self.bs.stack.stack("FF")
        self.bs.sim.step()  # bs.sim.simt = 0.0 AFTER the call to bs.sim.step()
        self.bs.stack.stack("FF")
    def send_command(self, cmd):
        self.bs.stack.stack(cmd)
        self.bs.net.update()
    def update(self):
        self.bs.sim.step()
        self.bs.net.update()


def generate_scenario_austin_2(out_path, demand_dict_path, route_dict_path, dep_interval):
    print(out_path)
    folder_path = out_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    entries = []
    
    f = open(out_path + f"/austin_env_full_single_intersection.scn", "w")

    f.write("00:00:00.00>TRAILS ON \n")
    f.write("\n")
    f.write("00:00:00.00>PAN 30.29828195311632 -97.92645392342473 \n")
    f.write("\n")

    # Load Route Names for Generation
    with open(demand_dict_path, 'r') as file_1:
        demand_dict = json.load(file_1)
    with open(route_dict_path, 'r') as file_2:
        route_dict = json.load(file_2)
    print("Demand Dict: ", demand_dict)
    # Initialize a dictionary to track the next available time for each starting waypoint
    next_available_time = {}
    for route_name, waypoints in route_dict.items():
        starting_wpt = waypoints[0] + waypoints[1] + "1"
        if starting_wpt not in next_available_time:
            next_available_time[starting_wpt] = 0

    total_aircraft_num = 0

    # Keep adding aircraft until all demands are met
    while any(demand_dict[route_name] > 0 for route_name in route_dict.keys()):
        for route_name in route_dict.keys():
            if demand_dict[route_name] > 0:
                plane = "P" + route_name + str(total_aircraft_num)
                starting_wpt = route_dict[route_name][0] + route_dict[route_name][1] + "1"
                # Calculate time based on the next available time for the starting waypoint
                time_seconds = next_available_time[starting_wpt]
                time_minutes = time_seconds // 60
                time_seconds = time_seconds % 60
                time = f"00:{time_minutes:02}:{time_seconds:02}.00"
                
                first_wpt = route_dict[route_name][0] + route_dict[route_name][1] + "1"
                last_wpt = route_dict[route_name][-2] + route_dict[route_name][-1] + "2"
                entries.append((time, f">CRE {plane},EC35,{first_wpt},0,0\n"))
                entries.append((time, f">ORIG {plane} {first_wpt}\n"))
                entries.append((time, f">DEST {plane} {last_wpt}\n"))
                entries.append((time, f">SPD {plane} 40\n"))
                entries.append((time, f">ALT {plane} 800\n"))
                for index in range(0, len(route_dict[route_name]) - 1):
                    waypoint_1 = route_dict[route_name][index] + route_dict[route_name][index+1] + "1"
                    waypoint_2 = route_dict[route_name][index] + route_dict[route_name][index+1] + "2"
                    entries.append((time, f">ADDWPT {plane} {waypoint_1} 800 40\n"))
                    entries.append((time, f">ADDWPT {plane} {waypoint_2} 800 40\n"))
                entries.append((time, f">{plane} VNAV on \n"))
                # Update the next available time for the starting waypoint
                next_available_time[starting_wpt] += dep_interval
                
                # Decrease the demand for this route
                demand_dict[route_name] -= 1
                
                total_aircraft_num += 1

    # Sort entries by time
    entries.sort(key=lambda x: x[0])

    # Write sorted entries to file
    for entry in entries:
        f.write(entry[0] + entry[1])
        
    f.close()



# Test Scenario Generation:
intervals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
generate_scenario_austin_2(f'/home/suryamurthy/UT_Autonomous_Group/ULI_noise_aware_agent/scenarios/generated_scenarios', '/home/suryamurthy/UT_Autonomous_Group/ULI_noise_aware_agent/D2MAV_A/route_demand_single_intersection.json', '/home/suryamurthy/UT_Autonomous_Group/ULI_noise_aware_agent/D2MAV_A/route_info_dict.json', 10)

# scenario_file = f'C:\Users\surya\PycharmProjects\ISMS_39\ILASMS_func3a-update-routes\scenarios\generated_scenarios\test_case_0.scn'
# # scenario_file = r'C:\Users\surya\PycharmProjects\ISMS_39\ILASMS_func3a-update-routes\scenarios\basic_env.scn'
# # https://github.com/TUDelft-CNS-ATM/bluesky/wiki/navdb
# node_1 = Communication_Node(scenario_file)

# interval_1 = 1000
# interval_2 = 100000
# counter = 0
# counter_2 = 0
# counter_3 = 0
# # Simulation Update Loop: reset and load a new scenario once all vehicles have exited the simulation.
# while 1:
#     # time.sleep(0.01)
#     node_1.update()
























# counter += 1
#     if counter % interval_1 == 0:
#         counter_2 +=1
#         if counter_2 % 2 == 0:
#             print("setting speed to 20")
#             for id in node_1.bs.traf.id:
#                 node_1.send_command(r'SPD ' + id + ' 20')
#                 # node_1.send_command(r'PAN '+ id)
#         else:
#             print("setting speed to 30")
#             for id in node_1.bs.traf.id:
#                 node_1.send_command(r'SPD ' + id + ' 30')
#                 # node_1.send_command(r'PAN ' + id)