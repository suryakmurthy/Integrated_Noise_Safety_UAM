# This file is based on the original D2MAV-A codebase by Brittain et al.
# Modified by Surya Murthy for the NASA ULI project (2025).
# Major changes include:
#   - Reward shaping with noise, energy, and separation costs
#   - Altitude-aware action space and state representation
#   - Scenario support for non-cooperative agents
# All modifications are marked with "MODIFIED" or high-level function docstrings.
# Licensed under the GNU General Public License v3.0.
# Original source: https://arxiv.org/abs/2003.08353

import numpy as np
import ray
import numba as nb
from bluesky.tools import geo
from shapely.geometry import LineString, Point
from shapely.geometry.multilinestring import MultiLineString
from shapely.ops import nearest_points
import math
from gym.spaces import Discrete, Box
import os
import random
import utm
from pyproj import Transformer
import glob
import gin
from copy import copy
import yaml
from itertools import groupby
import time
import json
import torch

from inspect import currentframe
from timeit import default_timer as timer

# # Auto ATC Setup items
from D2MAV_A.qatc import TrafficManager, VehicleHelper, load_routes, BadLogic

# Load traffic manager configuration file
FILE_PREFIX = str(os.path.dirname(__file__))
TOWER_CONFIG_FILE = FILE_PREFIX + "/Austin_towers.yaml"
with open(TOWER_CONFIG_FILE, "r") as f:
    tower_config = yaml.load(f, Loader=yaml.Loader)
# Load some route data
import pickle


with open("new_route_data.pkl", "rb") as file:
    route_data = pickle.load(file)

## Limit GPU usage
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


@nb.njit()
def discount(r, discounted_r, v, done, gae):
    """Compute the gamma-discounted rewards over an episode."""
    for t in range(len(r) - 1, -1, -1):
        if done[t] or t == (len(r) - 1):
            delta = r[t] - v[t][0]
            gae[t] = delta

        else:
            delta = r[t] + 0.95 * v[t + 1][0] - v[t][0]
            gae[t] = delta + 0.95 * 0.95 * gae[t + 1]

        discounted_r[t] = gae[t] + v[t][0]

    return discounted_r


## Checks the feasibility of the generated route based on
## specified threshold
def checkPoint(x1, y1, x2, y2, ls, threshold):
    for j in range(len(ls)):
        old_ls = ls[j]
        start, end = list(old_ls.coords)

        x_1, y_1 = start
        xe_1, ye_1 = end

        # distance start to old LS start
        dx = np.sqrt((x_1 - x1) ** 2 + (y_1 - y1) ** 2) / geo.nm

        # distance start to old LS end
        dx_1 = np.sqrt((xe_1 - x1) ** 2 + (ye_1 - y1) ** 2) / geo.nm

        # distance end to old LS end
        dx_2 = np.sqrt((xe_1 - x2) ** 2 + (ye_1 - y2) ** 2) / geo.nm

        # distance end to old LS start
        dx_3 = np.sqrt((x_1 - x2) ** 2 + (y_1 - y2) ** 2) / geo.nm

        dist = np.array([dx, dx_1, dx_2, dx_3])

        # feasible route
        if any(dist <= threshold):
            return False
    return True


@ray.remote
class Runner(object):
    import tensorflow as tf
    import bluesky as bs

    """
        Worker agent. Runs the BlueSky sim within its own process. This agent
        collects the experience and sends to the scheduler/trainer Worker

    """

    def __init__(
        self,
        actor_id,
        agent,
        max_steps=1024,
        speeds=[5, 0, 220],
        simdt=1,
        bsperf="openap",
        scenario_file=None,
        working_directory=None,
        LOS=10,
        dGoal=100,
        intruderThreshold=750,
        altChangePenalty=[0],
        stepPenalty=[0],
        clearancePenalty=0.005,
        config_file=None,
        gui=False,
        non_coop_tag=0,
        traffic_manager_active=True,
        run_type="train",
        weighting_factor_noise=0.1,
        weighting_factor_energy=0.001,
        weighting_factor_separation=0.9,
        max_alt = 3000,
        min_alt = 1000,
        alt_level_separation = 500,
        n_waypoints=2
    ):
        self.id = actor_id

        self.tf.config.threading.set_intra_op_parallelism_threads(1)
        self.tf.config.threading.set_inter_op_parallelism_threads(1)
        self.tf.compat.v1.logging.set_verbosity(self.tf.compat.v1.logging.ERROR)

        self.agents = {}
        self.agent = agent
        self.scen_file = scenario_file
        self.working_directory = working_directory
        self.speeds = np.array(speeds)
        self.altitudes = np.array([-alt_level_separation,0,alt_level_separation])
        self.max_steps = max_steps
        self.simdt = simdt
        self.bsperf = bsperf
        self.step_counter = 0
        self.LOS = LOS
        self.dGoal = dGoal
        self.intruderThreshold = intruderThreshold
        self.altChangePenalty = altChangePenalty
        self.stepPenalty = stepPenalty
        self.clearancePenalty = clearancePenalty
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ### Weighting Factor
        self.weighting_factor_noise = weighting_factor_noise
        self.weighting_factor_separation = weighting_factor_separation
        self.weighting_factor_energy = weighting_factor_energy
        self.fixed_energy_cost = -0.05

        with open('grid_points.json', 'r') as json_file:
            self.grid_points_list = json.load(json_file)
        self.noise_levels = {}

        
        ### Altitude and Noise Reward Parameters
        self.min_alt = min_alt                    # feet
        self.max_alt = max_alt                   # feet

        self.alt_changing = {}
        self.alt_changing_time_steps = {}
        self.max_noise_increase = 0
        self.average_noise_increase = {}
        
        ### Aircraft Noise Impact Parameters
        self.a_0 = 88.09
        self.a_1 = 3.21
        self.a_2 = -2.62

        with open('D2MAV_A/ambient_noise_dict.json', 'r') as file:
            self.ambient_noise_level = json.load(file)


        self.gui = gui
        self.traffic_manager_active = traffic_manager_active
        self.run_type = run_type
        self.n_waypoints = n_waypoints
        self.intersection_radius = 2700

        if not "SIMDT" in os.environ.keys():
            os.environ["SIMDT"] = "{}".format(self.simdt)

        ## building episode specific parameters not configured by config.gin
        self.dones = []

        self.episode_done = True  ## initialization

        self.epsg_proj = "epsg:2163"
        self.epsg_from = "epsg:4326"
        self.transformer = Transformer.from_crs(
            self.epsg_from, self.epsg_proj, always_xy=True
        )

        self.timer = 0
        self.num_ac = 0
        self.counter = 0

        self.action_key = {}

        self.min_x = 281134.8350222109  # 686785.5111184405 #np.inf
        self.max_x = 332359.3446274982  # 737690.7518448773 #-np.inf
        self.min_y = -1352500.1522055818  # 3627144.8191298996 #np.inf
        self.max_y = -1306410.5905290868  # 3673125.271272543 #-np.inf
        self.tas_min = np.round(
            self.speeds[0] * geo.nm / 3600, 4
        )  # converting knots to m/s
        self.tas_max = np.round(
            self.speeds[2] * geo.nm / 3600, 4
        )  # converting knots to m/s
        self.ax_min = -3.5
        self.ax_max = 3.5
        self.max_d = 46726.453433800954  # 0

        # Non-cooperative
        self.non_coop_tag = non_coop_tag  # 0 for cooperative. 1 for Loos of Control. 2 for loss of communication.
        self.LControl_lst = ["PNHWL0"]
        self.LComm_lst = ["PI30L0"]

        # === MODIFIED: Adjusted State and Action Dimension ===
        self.action_dim = 3
        self.speed_dim = 0 # speed up, maintain, slow down
        self.alt_dim = 3 # ascend, maintain, descend
        self.ownship_obs_dim = 5
        self.intruder_obs_dim = 3
        # === END OF MODIFIED SEGMENT ===


        self.action_space = Discrete(self.action_dim)

        ### Travel Time
        self.halting_times = []
        self.halt_start = {}
        self.full_travel = {}
        self.travel_start = {}
        self.wait_time = {}

        # Initialize Traffic Manager
        self.create_traffic_manager()

        if self.gui:
            self.bs.init(
                mode="sim", configfile=self.working_directory + "/" + config_file
            )
            self.bs.net.connect()

        else:
            self.bs.init(
                mode="sim",
                detached=True,
                configfile=self.working_directory + "/" + config_file,
            )
        self.agent.initialize(
            tf, self.ownship_obs_dim, self.intruder_obs_dim, self.action_dim
        )


    def create_traffic_manager(self):
        """
        Create a traffic manager object. In previous work, this code implemented centralized controllers that enforced traffic congestion contraints on different areas in the environment.
        In this work, the traffic manager object is only used to store information about aircraft's current routes and progress. This information is used for determining intruder aircraft.
        """
        route_linestrings = {}
        for route_id, gps_wp_list in route_data.items():
            rtemp = []
            for item in gps_wp_list:  # item is a tuple of (lon, lat)
                x, y = self.transformer.transform(item[0], item[1])
                rtemp.append((x, y))
            route_linestrings[route_id] = LineString(rtemp)
        self.traffic_manager = TrafficManager(tower_config)
        self.vehicle_helpers = {}  # Store vehicle helpers
        self.routes_loaded = load_routes(
            tower_config, self.traffic_manager, route_linestrings
        )  # Odd name to make sure it doesn't clash
        self.pending_requests = (
            []
        )
        self.pending_initial_requests = (
            []
        ) 
        self.exiting_vehicles = []

    def reset(self):
        """
        Beginning of the episode. In this function, all variables need to be reset to default.
        """
        print("Resetting")
        self.agent.reset()

        self.timer = 0
        self.num_ac = 0
        self.counter = 0
        self.episode_done = False
        self.step_counter = 0
        self.time_without_traffic = 0
        self.dones = []
        self.acInfo = {}
        self.file_keeper = []
        collected_responses = {}
        self.congestion_distribution = {}

        ### Travel Time Metrics
        self.current_movement = {}
        self.action_override = []
        self.alt_adjustments = {}
        self.alt_override = []
        self.full_travel = {}
        self.full_halting_times = {}
        self.travel_start = {}

        # randomly sample
        if ".scn" not in self.scen_file:
            scenario_files = glob.glob(f"{self.scen_file}" + "/*.scn")
            scenario_file = np.random.choice(scenario_files, 1)[0]

        else:
            scenario_file = self.scen_file

        # Reset Traffic Manager
        self.create_traffic_manager()  # easier to just create a new one for now

        # Starting the bluesky and sim
        self.bs.stack.stack("IC " + self.working_directory + "/" + scenario_file)
        self.bs.stack.stack("FF")
        self.bs.sim.step()  # bs.sim.simt = 0.0 AFTER the call to bs.sim.step()
        self.bs.stack.stack("FF")
        # === MODIFIED: Define the maximum and minimum values for the ownship state and intruder state ===
        # Adding Altitude State information (Current Alt, Ambient Noise) to end of state
        ownship_min_state = [0, 0, 0, 0, 0]
        ownship_max_state = [2, self.max_alt, 1, self.max_alt, 50]

        intruder_min_state = [
            0,
            0,
            -self.max_alt
        ]

        intruder_max_state = [
            self.max_d,
            2,
            self.max_alt
        ]
        # === END OF MODIFIED SEGEMENT ===

        self.context_space = Box(
            np.array(intruder_min_state),
            np.array(intruder_max_state),
            dtype=np.float64,
        )

        self.observation_space = Box(
            np.array(ownship_min_state), np.array(ownship_max_state), dtype=np.float64
        )

        ## This is a catch to make sure the time between bluesky sim steps is 1 seconds
        # Should 1 second be changed to something smaller like 0.1?
        before = self.bs.sim.simt
        self.bs.sim.step()
        after = self.bs.sim.simt
        if (after - before) == 0:
            return self.reset()

        assert (after - before) == self.simdt

        ## The first self.bs.sim.step() spawns in the initial aircraft. Sim time should be at t = 12 seconds now
        self.step_counter += 1

        response = {}

        ## START OF TRAFFIC MANAGER CODE: This code keeps track of agent routes when aircraft take off.
        if self.traffic_manager_active:
            # First time the auto atc system is pinged
            initial_requests = []
            for i in range(self.bs.traf.lat.shape[0]):
                id_ = self.bs.traf.id[i]  # ownship ID

                # Check if the current ID exists. If not then create a new vehicle helper
                if not id_ in self.vehicle_helpers.keys():
                    # Get and reformat the route name coming from Bluesky
                    route_name = self.bs.traf.ap.route[i].wpname[0][0:-1]
                    self.vehicle_helpers[id_] = VehicleHelper(
                        id_, self.routes_loaded[route_name]
                    )
                    # Add initial request to enter the system
                    initial_requests.append(id_)

            # Pass collected requests to the Traffic Manager and process them
            for id_ in initial_requests:
                formatted_request = self.vehicle_helpers[id_].format_request()  # tuple
                self.traffic_manager.add_request(id_, formatted_request)

            if initial_requests:
                initial_request_response = self.traffic_manager.process_requests()
            else:
                response = None
                initial_request_response = {}
                print("No initial requests")

            collected_responses = {}
            for id_, response in initial_request_response.items():
                k_idx = self.bs.traf.id2idx(id_)
                collected_responses[id_] = [
                    response,
                    self.vehicle_helpers[id_].distance_to_next_boundary(
                        [self.bs.traf.lon[k_idx], self.bs.traf.lat[k_idx]]
                    ),
                ]
                if response and not self.within_LOS(id_):
                    self.vehicle_helpers[id_].enter_request_status = True
                    if self.vehicle_helpers[id_].next_intersection == None:
                        self.vehicle_helpers[id_].next_intersection = self.vehicle_helpers[id_].route.route_id[2:4]
                    self.travel_start[id_] = self.bs.sim.simt
                    self.full_halting_times[id_] = []
                    self.alt_adjustments[id_] = 0
                    self.wait_time[id_] = 0
                else:
                    self.vehicle_helpers[id_].enter_request_status = False
                    self.pending_initial_requests.append(id_)
                    k_idx = self.bs.traf.id2idx(id_)
                    self.bs.traf.ap.setclrcmd(
                        k_idx, False
                    )  # set the clearance to False (i.e., denied and hold on ground)

        state, _, _, _ = self.state_update(
            self.bs.traf, init=True, tm_response=collected_responses
        )

        if self.gui:
            self.bs.net.update()

        return state

    def step(self, actions, policy, value):
        """
        Update the environment with the actions from the agents. This function was modified from the original D2MAV-A code base.
        """

        collected_responses = {}
    
        coord_transform = self.transformer.transform(self.bs.traf.lon, self.bs.traf.lat)


        geometries = MultiLineString(
            [
                [(coord_transform[0][i], coord_transform[1][i])]
                + [
                    self.transformer.transform(
                        self.bs.traf.ap.route[i].wplon[j],
                        self.bs.traf.ap.route[i].wplat[j],
                    )
                    for j in range(
                        self.bs.traf.ap.route[i].iactwp,
                        len(self.bs.traf.ap.route[i].wplon),
                    )
                ]
                for i in range(self.bs.traf.lat.shape[0])
            ]
        ) 

        for ac_id in list(self.alt_changing.keys()):
            idx = self.bs.traf.id2idx(ac_id)
            if idx != -1:
                if self.alt_changing[ac_id] == round(self.meters_to_feet(self.bs.traf.alt[idx])):
                    del self.alt_changing[ac_id]
                    del self.alt_changing_time_steps[ac_id]
        for ac_id in actions.keys():
            if actions[ac_id] == -1:
                """Uncomment if running unequiped"""
                if ac_id in self.action_override:
                    speed = 0
                else:
                    speed = 40
                self.bs.stack.stack("{} SPD {}".format(ac_id, speed))
                continue
            
            ## Since we do not consider speed changes as a possible aircraft action, we set the speeds to a constant value.
            speed = self.speeds[2]            
            
            ### START OF SEGMENT ###
            ### The following segement of code considers cases where aircraft halt as a result of the centralized controllers defined in traffic manager. This code does not impact our experiments.
            # Halt Vehicles with overridden actions. There are no mechanisms for this in the current environment.  
            if ac_id in self.action_override:
                speed = 0
                if ac_id not in self.halt_start:
                    self.halt_start[ac_id] = self.bs.sim.simt
            
            if ac_id in self.alt_override:
                actions[ac_id] = 1 

            # Record Average Halting Time. This should be 
            if ac_id not in self.action_override:
                if ac_id in self.halt_start.keys():
                    halting_time = self.bs.sim.simt - self.halt_start[ac_id]
                    del self.halt_start[ac_id]
                    self.halting_times.append(halting_time)
                    self.full_halting_times[ac_id].append(halting_time)
            ### END OF SEGMENT ###
            self.bs.stack.stack("{} SPD {}".format(ac_id, speed))

            ### Update aircraft altitude levels based on actions.
            alt_change = self.altitudes[actions[ac_id]]
            current_alt = self.meters_to_feet(self.bs.traf.alt[self.bs.traf.id2idx(ac_id)])
            new_alt = current_alt + alt_change

            
            if new_alt < self.min_alt:
                new_alt = self.min_alt
            if new_alt > self.max_alt:
                new_alt = self.max_alt

            # Allow altitude adjustment actions to persist if aircraft are between altitude levels.
            if ac_id in self.alt_changing.keys():
                new_alt = self.alt_changing[ac_id]
                self.alt_changing_time_steps[ac_id] += 1
            else:
                if current_alt < self.min_alt:
                    self.alt_changing[ac_id] = self.min_alt
                    self.alt_changing_time_steps[ac_id] = 0
                else:
                    self.alt_changing[ac_id] = round(new_alt)
                    self.alt_changing_time_steps[ac_id] = 0
                        
            self.bs.stack.stack("{} ALT {}".format(ac_id, new_alt))

        # updates the bluesky environment by 1 simulation timestep (1 seconds)
        self.bs.sim.step()
        self.step_counter += 1

        if self.gui:
            self.bs.net.update()
        
        ### Start of traffic manager code. This updates aircraft flight corridors. ###
        if self.traffic_manager_active:
            new_requests = []  # requests from vehicles that are in the system
            initial_requests = []  # requests from vehicles that are not in the system

            """ Use the traffic object to determine the state of all intersections and route sections.
                1) Determine which vehicles are currently inside which intersections
                2) Determine which vehicles have transitioned:
                    a) into an intersection
                    b) out of an intersection  """

            for i in range(self.bs.traf.lat.shape[0]):
                id_ = self.bs.traf.id[i]  # ownship ID

                # Skip GA aircraft
                if id_[0:2] == "GA":
                    continue

                # Skip vehicles who are waiting outside the system
                if id_ in self.pending_initial_requests:
                    continue

                """ Vehicles that have spawned during the current step might be removed if space is not available,
                    therefore we need to skip them until they are authorized to enter the system.
                    Vehicles who were in self.vehicle_helpers at the beginning of the step have already entered the system """
                if id_ not in self.vehicle_helpers.keys():
                    continue

                curr_gps = [self.bs.traf.lon[i], self.bs.traf.lat[i]]
                # Determine which vehicles are inside which intersections
                for intersection in self.traffic_manager.intersections.values():
                    if self.traffic_manager.check_if_within_intersection(
                        curr_gps, intersection.tower_ID
                    ):
                        # Determine if a vehicle is already accepted into the intersection
                        if (
                            id_ in intersection.accepted
                            or id_ in intersection.recently_left
                        ):
                            break
                        # Determine if authorized vehicle has transitioned into intersection
                        elif id_ in intersection.authorized:
                            intersection.accepted.append(id_)
                            intersection.authorized.remove(id_)
                            # Update vehicle helper
                            self.vehicle_helpers[id_].within_intersection = True
                            self.vehicle_helpers[
                                id_
                            ].current_intersection = intersection.tower_ID
                            # Update the route section that the vehicle left if not an initial request
                            crs = self.vehicle_helpers[
                                id_
                            ].current_route_section  # Will be None if initial request
                            if crs:
                                self.traffic_manager.towers[crs].accepted.remove(id_)
                            break
                        # Determine if vehicle has entered illegally
                        elif (
                            id_ not in intersection.authorized
                            and id_ not in intersection.accepted
                            and id_ not in intersection.recently_left
                        ):
                            if id_ not in intersection.illegal:
                                intersection.illegal.append(id_)
                                print(
                                    id_,
                                    "has entered",
                                    intersection.tower_ID,
                                    "illegally",
                                    self.vehicle_helpers[id_].current_route_section,
                                    self.vehicle_helpers[id_].next_intersection,
                                )
                                
                                time.sleep(100)
                                # Update vehicle helper
                                self.vehicle_helpers[id_].within_intersection = True
                                self.vehicle_helpers[
                                    id_
                                ].current_intersection = intersection.tower_ID
                                break
                            else:
                                break
                        else:  # Raise error because it should not be possible to pass through all of the above checks
                            raise BadLogic(
                                "Bad logic when checking if vehicle is within intersection. id_: ",
                                id_,
                                "intersection: ",
                                intersection.tower_ID,
                            )
                    else:
                        # Determine if accepted vehicle has transitioned out of intersection
                        if id_ in intersection.accepted:
                            intersection.accepted.remove(id_)
                            intersection.recently_left.append(id_)
                            # Update vehicle helper to reflect that this vehicle is no longer in the intersection
                            self.vehicle_helpers[id_].within_intersection = False
                            self.vehicle_helpers[id_].current_intersection = None
                            self.vehicle_helpers[
                                id_
                            ].enter_request_status = False  # reset
                            # Update the route section that the vehicle entered if not exiting the system
                            idx = self.bs.traf.id2idx(id_)
                            autopilot_route = self.bs.traf.ap.route[idx]
                            route_counter = 0
                            found_next_wp = False
                            current_route_section = None
                            next_route_section = None
                            while not found_next_wp and autopilot_route.iactwp + route_counter < len(autopilot_route.wpname):
                                current_waypoint_string = autopilot_route.wpname[autopilot_route.iactwp + route_counter]
                                current_route_section_temp =  current_waypoint_string[0:4]
                                if current_route_section_temp in self.routes_loaded.keys():
                                    current_route_section = current_route_section_temp
                                    found_next_wp = True 
                                else:
                                    route_counter += 1
                            found_next_wp = False
                            while not found_next_wp and autopilot_route.iactwp + route_counter < len(autopilot_route.wpname):
                                next_waypoint_string = autopilot_route.wpname[autopilot_route.iactwp + route_counter]
                                next_route_section_temp =  next_waypoint_string[0:4]
                                if next_route_section_temp in self.routes_loaded.keys() and current_route_section != next_route_section_temp:
                                    next_route_section = next_route_section_temp
                                    found_next_wp = True 
                                else:
                                    route_counter += 1

                            self.vehicle_helpers[id_].within_intersection = False
                            self.vehicle_helpers[id_].current_intersection = None
                            if current_route_section != None:
                                self.vehicle_helpers[id_].route = self.routes_loaded[current_route_section]
                            self.vehicle_helpers[id_].next_route_section = next_route_section
                            self.vehicle_helpers[
                                id_
                            ].enter_request_status = False  # reset
                            if current_route_section:
                                self.traffic_manager.towers[current_route_section].accepted.append(id_)
                                self.traffic_manager.towers[current_route_section].authorized.remove(id_)
                                self.vehicle_helpers[id_].current_route_section = self.vehicle_helpers[id_].route.route_id
                                self.vehicle_helpers[id_].next_intersection = self.vehicle_helpers[id_].current_route_section[2:4] # self.traffic_manager.search_for_intersection(self.vehicle_helpers[id_].current_route_section, "inbound").tower_ID
                            break
                        # Determine if illegal vehicle has transitioned out of intersection
                        elif id_ in intersection.illegal:
                            intersection.illegal.remove(id_)
                            # Update vehicle helper to reflect that this vehicle is no longer in the intersection
                            self.vehicle_helpers[id_].within_intersection = False
                            self.vehicle_helpers[id_].current_intersection = None
                            self.vehicle_helpers[id_].enter_request_status = False
                            print(
                                "Illegal vehicle",
                                id_,
                                "has exited",
                                intersection.tower_ID,
                            )
                else:
                    # Set vehicle helper to reflect that this vehicle is not in an intersection
                    self.vehicle_helpers[id_].within_intersection = False
                    self.vehicle_helpers[id_].current_intersection = None

            # Update the volume of each intersection
            for intersection in self.traffic_manager.intersections.values():
                intersection.set_volume()
            # Update the volume of each route section
            for tower in self.traffic_manager.towers.values():
                tower.set_volume()

            """ Collect new vehicle requests for processing """
            for i in range(self.bs.traf.lat.shape[0]):
                id_ = self.bs.traf.id[i]  # ownship ID
                if id_[0:2] == "GA":
                    continue

                # Check if the current ID exists. If not then create a new vehicle helper
                if not id_ in self.vehicle_helpers.keys():
                    # Get and reformat the route name coming from Bluesky
                    route_name = self.bs.traf.ap.route[i].wpname[0][0:-1]
                    self.vehicle_helpers[id_] = VehicleHelper(
                        id_, self.routes_loaded[route_name]
                    )
                    # Add initial request to enter the system
                    initial_requests.append(id_)
                else:
                    a = id_ not in self.pending_requests
                    b = id_ not in self.pending_initial_requests
                    c = (
                        id_ not in self.exiting_vehicles
                    )
                    d = not self.vehicle_helpers[id_].enter_request_status
                    if a and b and c and d:
                        request_eligibility = self.vehicle_helpers[
                            id_
                        ].check_if_request_eligible(
                            [self.bs.traf.lon[i], self.bs.traf.lat[i]]
                        )
                        if request_eligibility:
                            new_requests.append(id_)

            """ Process requests """
            # First process pending requests
            for id_ in self.pending_requests:
                formatted_request = self.vehicle_helpers[id_].format_request()  # tuple
                self.traffic_manager.add_request(id_, formatted_request)
            pending_request_response = self.traffic_manager.process_requests()

            # Second process new requests
            for id_ in new_requests:
                formatted_request = self.vehicle_helpers[id_].format_request()
                self.traffic_manager.add_request(id_, formatted_request)
            new_request_response = self.traffic_manager.process_requests()

            # Third process pending initial requests
            for id_ in self.pending_initial_requests:
                formatted_request = self.vehicle_helpers[id_].format_request()
                self.traffic_manager.add_request(id_, formatted_request)
            pending_initial_request_response = self.traffic_manager.process_requests()

            # Fourth process initial requests
            for id_ in initial_requests:
                formatted_request = self.vehicle_helpers[id_].format_request()
                self.traffic_manager.add_request(id_, formatted_request)
            initial_request_response = self.traffic_manager.process_requests()

            """ Collect responses and update request lists based on the response """
            collected_responses = {}
            # Pending in system responses
            for id_, response in pending_request_response.items():
                k_idx = self.bs.traf.id2idx(id_)
                collected_responses[id_] = [
                    response,
                    self.vehicle_helpers[id_].distance_to_next_boundary(
                        [self.bs.traf.lon[k_idx], self.bs.traf.lat[k_idx]]
                    ),
                ]
                if response:
                    self.vehicle_helpers[id_].enter_request_status = True
                    self.pending_requests.remove(id_)
                    if self.vehicle_helpers[id_].final_route_segment:
                        self.exiting_vehicles.append(id_)
                else:
                    self.vehicle_helpers[id_].enter_request_status = False
            # New in system responses
            for id_, response in new_request_response.items():
                k_idx = self.bs.traf.id2idx(id_)
                collected_responses[id_] = [
                    response,
                    self.vehicle_helpers[id_].distance_to_next_boundary(
                        [self.bs.traf.lon[k_idx], self.bs.traf.lat[k_idx]]
                    ),
                ]
                if response:
                    self.vehicle_helpers[id_].enter_request_status = True
                    if self.vehicle_helpers[id_].final_route_segment:
                        self.exiting_vehicles.append(id_)
                else:
                    self.pending_requests.append(id_)
                    self.vehicle_helpers[id_].enter_request_status = False
            # Pending initial request responses
            for id_, response in pending_initial_request_response.items():
                k_idx = self.bs.traf.id2idx(id_)
                collected_responses[id_] = [
                    response,
                    self.vehicle_helpers[id_].distance_to_next_boundary(
                        [self.bs.traf.lon[k_idx], self.bs.traf.lat[k_idx]]
                    ),
                ]
                if response and not self.within_LOS(id_):
                    self.vehicle_helpers[id_].enter_request_status = True
                    self.pending_initial_requests.remove(id_)
                    if self.vehicle_helpers[id_].next_intersection == None:
                        self.vehicle_helpers[id_].next_intersection = self.vehicle_helpers[id_].route.route_id[2:4]
                    # set the clearance to True. Optional fields are ALT, SPD
                    k_idx = self.bs.traf.id2idx(id_)
                    self.bs.traf.ap.setclrcmd(k_idx, True, 400, 30)
                    self.travel_start[id_] = self.bs.sim.simt
                    self.full_halting_times[id_] = []
                    self.alt_adjustments[id_] = 0
                    self.wait_time[id_] = 0
                else:
                    self.vehicle_helpers[id_].enter_request_status = False
            # Initial request responses
            for id_, response in initial_request_response.items():
                k_idx = self.bs.traf.id2idx(id_)
                collected_responses[id_] = [
                    response,
                    self.vehicle_helpers[id_].distance_to_next_boundary(
                        [self.bs.traf.lon[k_idx], self.bs.traf.lat[k_idx]]
                    ),
                ]
                if response and not self.within_LOS(id_):
                    self.vehicle_helpers[id_].enter_request_status = True
                    if self.vehicle_helpers[id_].next_intersection == None:
                        self.vehicle_helpers[id_].next_intersection = self.vehicle_helpers[id_].route.route_id[2:4]
                    self.travel_start[id_] = self.bs.sim.simt
                    self.full_halting_times[id_] = []
                    self.alt_adjustments[id_] = 0
                    self.wait_time[id_] = 0
                else:
                    self.vehicle_helpers[id_].enter_request_status = False
                    self.pending_initial_requests.append(id_)
                    k_idx = self.bs.traf.id2idx(id_)
                    # set the clearance to False (i.e., denied and hold on ground)
                    self.bs.traf.ap.setclrcmd(k_idx, False)

        ### End of traffic manager code ###

        obs, reward, done, info = self.state_update(
            self.bs.traf,
            a=actions,
            policy=policy,
            value=value,
            init=False,
            tm_response=collected_responses,
        )

        if len(self.bs.traf.id) == 0:
            self.time_without_traffic += self.bs.sim.simdt
        else:
            self.time_without_traffic = 0

        if self.time_without_traffic > 1800:  # 0.5 hours
            done["__all__"] = True

        else:
            done["__all__"] = False

        return obs, reward, done, info

    def state_update(
        self,
        traf,
        a=None,
        policy=None,
        value=None,
        pp=None,
        tp=None,
        init=False,
        tm_response: dict = None,
    ):
        """
        Update the environment with the actions from the agents. This function also stores relevent metrics at each state like noise impact, altitude adjustments, and LOS events. 
        The function also implements the reward funtion used to shape agent policy. This function was modified from the original D2MAV-A code base.
        """
        # current number of a/c in bluesky sim
        n_ac = traf.lat.shape[0]
        rew = {}
        state = {}
        done = {}
        info = {}

        #
        self.action_override = []
        self.alt_override = []

        ## creating an index for the unique aircraft
        index = np.arange(n_ac).reshape(-1, 1)

        d = (
            geo.kwikdist_matrix(
                np.repeat(traf.lat, n_ac),
                np.repeat(traf.lon, n_ac),
                np.tile(traf.lat, n_ac),
                np.tile(traf.lon, n_ac),
            ).reshape(n_ac, n_ac)
            * geo.nm
        )
        argsort = np.array(np.argsort(d, axis=1))

        # transform all aircraft lon/lat positions
        coord_transform = self.transformer.transform(traf.lon, traf.lat)

        geometries = MultiLineString(
            [
                [(coord_transform[0][i], coord_transform[1][i])]
                + [
                    self.transformer.transform(
                        traf.ap.route[i].wplon[j],
                        traf.ap.route[i].wplat[j],
                    )
                    for j in range(
                        traf.ap.route[i].iactwp,
                        len(traf.ap.route[i].wplon),
                    )
                ]
                for i in range(traf.lat.shape[0])
            ]
        )
        
        ### Record individual noise impacts caused by aircraft as well as cumulative impacts in different flight corridors
        avg_noise_vals = {}
        noise_vals = {}
        for id_val in self.bs.traf.id:
            idx_val = self.bs.traf.id2idx(id_val)
            ac_alt = self.meters_to_feet(self.bs.traf.alt[idx_val])
            if ac_alt <= self.min_alt:
                ac_noise = 0
            else:
                ac_noise = self.a_0 +  (self.a_1 * math.log10(ac_alt)) + (self.a_2 * ((math.log10(ac_alt))**2))
            if self.vehicle_helpers[id_val].current_route_section != None:
                if self.vehicle_helpers[id_val].current_route_section not in noise_vals.keys():
                    noise_vals[self.vehicle_helpers[id_val].current_route_section] = []
                noise_vals[self.vehicle_helpers[id_val].current_route_section].append(10**(ac_noise / 10))
            if self.vehicle_helpers[id_val].current_intersection != None:
                if self.vehicle_helpers[id_val].current_intersection not in noise_vals.keys():
                    noise_vals[self.vehicle_helpers[id_val].current_intersection] = []
                noise_vals[self.vehicle_helpers[id_val].current_intersection].append(10**(ac_noise / 10))
        
        ### Record Cumulative Noise Increases
        for route_id in noise_vals.keys():
            if np.sum(noise_vals[route_id]) == 0:
                total_noise_impact = 0
                noise_increase = 0
            else:
                total_noise_impact = 10 * math.log10(np.sum(noise_vals[route_id]))
                if route_id not in self.ambient_noise_level.keys():
                    ambient_noise_val = 40
                else:
                    ambient_noise_val = self.ambient_noise_level[route_id]
                noise_increase = total_noise_impact - ambient_noise_val
            if noise_increase < 0:
                noise_increase = 0
            if noise_increase != 0:
                if route_id not in self.average_noise_increase.keys():
                    self.average_noise_increase[route_id] = []
                self.average_noise_increase[route_id].append(noise_increase)
                avg_noise_vals[route_id] = noise_increase
            if noise_increase >= self.max_noise_increase:
                self.max_noise_increase = noise_increase

        ### Record aircraft altitude levels
        for id_val in self.bs.traf.id:
            idx_val = self.bs.traf.id2idx(id_val)
            if id_val in self.alt_changing:
                ac_alt = self.alt_changing[id_val]
            else:
                ac_alt = round(self.meters_to_feet(self.bs.traf.alt[idx_val]))
            if ac_alt not in self.congestion_distribution.keys():
                self.congestion_distribution[ac_alt] = 0
            else:
                self.congestion_distribution[ac_alt] += 1
                

        # looping over the ownships
        for i in range(d.shape[0]):
            # ownship ID
            id_ = traf.id[i]

            if id_ not in self.acInfo:
                self.acInfo[id_] = {
                    "NMAC": [],
                    "Lat": [],
                    "Lon": [],
                    "Spd": [],
                    "Action": [],
                    "time": [],
                }

            # if the aircraft has not taken off, skip
            if not traf.active[i]:
                continue

            ownship_obj = geometries.geoms[i]

            ownship_waypoints = list(ownship_obj.coords)
            # first entry is current location
            current_position_ownship = ownship_waypoints[0]
            remaining_waypoints = ownship_waypoints[1 : 1 + self.n_waypoints]

            rel_waypoints = []
            for waypoint in remaining_waypoints:

                relx = waypoint[0] - current_position_ownship[0]
                rely = waypoint[1] - current_position_ownship[1]

                bearing = math.degrees(math.atan2(rely, relx))

                dist_waypoint = np.sqrt(relx**2 + rely**2)
                rel_waypoints += [
                    bearing,
                    dist_waypoint,
                ]

            if len(remaining_waypoints) < self.n_waypoints:
                rel_waypoints += [0, 0] * (self.n_waypoints - len(remaining_waypoints))
            dist = ownship_obj.length

            ## Converting ownship lat/lon to UTM coords
            xEast_own, yNorth_own = (
                coord_transform[0][i],
                coord_transform[1][i],
            )  # self.transformer.transform(traf.lon[i], traf.lat[i])

            prev_action_own = 1  # maintain
            if a is not None:
                if id_ in a:
                    prev_action_own = a[id_]
            rew[id_] = 0

            # === MODIFIED: Define the individual agent ownstate and intruder states ===
            if self.traffic_manager_active:
                ### This is code for centralized tower controllers and is not used in our experiments
                if id_ in tm_response:
                    response, distance = tm_response[id_]  # 0 = denied, 1 = approved
                    response = int(response)

                    if response == 0 and distance != -1:
                        rew[id_] += -self.clearancePenalty

                    if distance != -1 and distance < 250 and response == 0:
                        rew[id_] += -10 * self.clearancePenalty
                        self.action_override.append(id_)
                else:
                    response = 2  # no request
                    distance = 0

                ### Store relevant state information
                ambient_noise = 40
                current_route_section_name = self.vehicle_helpers[id_].current_route_section
                current_intersection_name = self.vehicle_helpers[id_].current_intersection
                if current_route_section_name != None and current_route_section_name in self.ambient_noise_level.keys():
                    ambient_noise = self.ambient_noise_level[current_route_section_name]
                else:
                    if current_intersection_name != None and current_intersection_name in self.ambient_noise_level.keys():
                        ambient_noise = self.ambient_noise_level[current_intersection_name]
                if id_ in self.alt_changing.keys():
                    alt_changing_bool = 1
                    target_alt = self.alt_changing[id_]
                else:
                    alt_changing_bool = 0
                    target_alt = self.meters_to_feet(traf.alt[i])
                alt_adj = self.alt_adjustments[id_]

                own_state = [
                    prev_action_own,
                    self.meters_to_feet(traf.alt[i]),
                    alt_changing_bool,
                    target_alt,
                    alt_adj
                ]            
            else:
                ambient_noise = 40
                current_route_section_name = self.vehicle_helpers[id_].current_route_section
                current_intersection_name = self.vehicle_helpers[id_].current_intersection
                if current_route_section_name != None and current_route_section_name in self.ambient_noise_level.keys():
                    ambient_noise = self.ambient_noise_level[current_route_section_name]
                else:
                    if current_intersection_name != None and current_intersection_name in self.ambient_noise_level.keys():
                        ambient_noise = self.ambient_noise_level[current_intersection_name]
                if id_ in self.alt_changing.keys():
                    alt_changing_bool = 1
                    target_alt = self.alt_changing[id_]
                else:
                    alt_changing_bool = 0
                    target_alt = self.meters_to_feet(traf.alt[i])

                
                own_state = [
                    prev_action_own,
                    self.meters_to_feet(traf.alt[i]),
                    alt_changing_bool,
                    target_alt,
                    alt_adj
                ]
            # === END OF MODIFIED SEGEMENT === 

            own_state = np.array(own_state).reshape(1, self.observation_space.shape[0])
            ## check normalization values
            self.normalization_check(x=xEast_own, y=yNorth_own, d=dist)

            own_state = (own_state - self.observation_space.low) / (
                self.observation_space.high - self.observation_space.low
            )

            self.acInfo[id_]["Lat"].append(traf.lat[i])
            self.acInfo[id_]["Lon"].append(traf.lon[i])
            self.acInfo[id_]["Spd"].append(traf.cas[i])
            self.acInfo[id_]["NMAC"].append(
                0
            )  # place holder 0 that is overwritten later if NMAC occurred
            self.acInfo[id_]["time"].append(self.bs.sim.simt)
            if a is not None and id_ in a:
                self.acInfo[id_]["Action"].append(a[id_])

            else:
                self.acInfo[id_]["Action"].append(1)  # "hold"

            done[id_] = False
            info[id_] = None

            ## made it to the goal
            if dist < self.dGoal:

                if self.traffic_manager_active:
                    intersection = self.vehicle_helpers[id_].current_intersection
                    if intersection != None:
                        self.traffic_manager.intersections[
                            intersection
                        ].accepted.remove(id_)
                        self.traffic_manager.intersections[intersection].set_volume()
                done[id_] = True

            # is this a GA aircraft?
            if id_[0:2] == "GA":
                if done[id_]:
                    self.bs.stack.stack("DEL {}".format(id_))
                # should prevent non-coop from formining a state
                # but state info of non-coop will still be available
                # to the remaining coop aircraft
                continue
            
            # === MODIFIED: Implements intruder detection.  === 
            reward_count = False
            closest_count = self.agent.max_agents
            intruder_state = None
            # Noise Information
            alt_own = self.bs.traf.alt[i]
            alt_own_rew = self.bs.traf.alt[i]
            alt_own_ft = self.meters_to_feet(alt_own)
            intruder_noise_vals = {}
            same_alt_level = 0
            same_alt_list = []
            diff_alt_level = 0
            diff_alt_list = []
            for j in range(len(argsort[i])):
                index = int(argsort[i][j])
                id_j = self.bs.traf.id[index]
                # The first entry will be: intruder == ownship so we need to skip
                if i == index:
                    continue

                if not traf.active[index]:
                    continue

                alt_intruder = self.bs.traf.alt[index]
                alt_intruder_rew = self.bs.traf.alt[index]
                alt_intruder_ft = self.meters_to_feet(alt_intruder)
                id_j = self.bs.traf.id[index]
                dist_with_alt = np.sqrt((d[i, index]**2) + ((alt_own - alt_intruder)**2)) # Replace d[i, index] with this value
                

                intruder_obj = geometries.geoms[index]
                dist = intruder_obj.length
                # intruder to be removed
                if dist < self.dGoal and d[i, index] > self.LOS:
                    continue

                # if the intruder is > 750 meters (0.5 nm) away, skip it.
                if (
                    d[i, index] > self.intruderThreshold
                ):
                    continue

                    
                # Check if Aircraft are on Parallel Routes
                if not ownship_obj.intersects(intruder_obj):
                    continue
                

                if self.vehicle_helpers[id_].current_route_section != None and self.vehicle_helpers[id_j].current_route_section != None:
                    swapped_route = self.vehicle_helpers[id_].current_route_section[2:4] + self.vehicle_helpers[id_].current_route_section[0:2]
                    if swapped_route == self.vehicle_helpers[id_j].current_route_section:
                        continue
                
                # Check if aircraft are on the same route
                if self.vehicle_helpers[id_].current_route_section != None and self.vehicle_helpers[id_j].current_route_section != None:
                    if self.vehicle_helpers[id_].current_route_section == self.vehicle_helpers[id_j].current_route_section:
                        continue

                if self.meters_to_feet(alt_own) >= self.min_alt:
                    dist_between_ac = d[i, index]
                    if np.abs(alt_own_rew - alt_intruder_rew) < self.LOS:
                        same_alt_level += 1
                        same_alt_list.append(id_j)
                    else:
                        if dist_between_ac < self.LOS:
                            diff_alt_level += 1
                        else:
                            diff_alt_level += 1 - ((dist_between_ac - self.LOS) / (self.intruderThreshold - self.LOS))
                        diff_alt_list.append(id_j)

                if alt_intruder > 0:
                    intruder_noise_vals[index] = self.a_0 +  (self.a_1 * math.log10(alt_intruder_ft)) + (self.a_2 * ((math.log10(alt_intruder_ft))**2))
                else:
                    intruder_noise_vals[index] = 0

                if dist_with_alt < self.LOS:
                    if self.vehicle_helpers[id_].current_route_section != None and self.vehicle_helpers[id_j].current_route_section != None:
                        if self.vehicle_helpers[id_j].route.route_id[0:3] != self.vehicle_helpers[id_].route.route_id[0:3]:
                            if self.bs.traf.gs[i] != 0 and self.bs.traf.gs[index] != 0:
                                info[id_] = 1
                                if id_ in a:
                                    self.acInfo[id_]["NMAC"][-1] = 1

                ## Converting intruder lat/lon to UTM coords
                xEast_int, yNorth_int = (
                    coord_transform[0][index],
                    coord_transform[1][index],
                )

                distIntGoal = dist

                relX = xEast_int - xEast_own
                relY = yNorth_int - yNorth_own
                prev_action_int = 1
                if a is not None:
                    if traf.id[index] in a:
                        prev_action_int = a[traf.id[index]]

                intruder_waypoints = list(intruder_obj.coords)
                current_position_intruder = intruder_waypoints[
                    0
                ]  # first entry is current location
                remaining_waypoints = intruder_waypoints[
                    1 : 1 + self.n_waypoints
                ]  # first entry is current location

                rel_waypoints = []
                for waypoint in remaining_waypoints:
                    rel_x = waypoint[0] - current_position_ownship[0]
                    rel_y = waypoint[1] - current_position_ownship[1]

                    bearing = math.degrees(math.atan2(rel_y, rel_x))

                    dist_waypoint = np.sqrt(rel_x**2 + rel_y**2)
                    rel_waypoints += [
                        bearing,
                        dist_waypoint,
                    ]
                if len(remaining_waypoints) < self.n_waypoints:
                    rel_waypoints += [0, 0] * (
                        self.n_waypoints - len(remaining_waypoints)
                    )
                rel_altitude = alt_own_ft - alt_intruder_ft

                ### MODIFIED: This is where we define the intruder state. This is different from the original D2MAV-A ###
                int_state = np.array(
                    [
                        d[i, index],
                        prev_action_int,
                        rel_altitude, # Changing from actual altitude to relavtive altitude
                    ]
                    # + rel_waypoints
                ).reshape(1, self.context_space.shape[0])
                self.normalization_check(x=xEast_int, y=yNorth_int, d=distIntGoal)

                int_state = (int_state - self.context_space.low) / (
                    self.context_space.high - self.context_space.low
                )

                if intruder_state is None:
                    intruder_state = int_state

                else:
                    intruder_state = np.append(intruder_state, int_state, axis=0)

                closest_count -= 1

                if closest_count == 0:
                    break
            
            if closest_count != 0:
                remaining = np.zeros((closest_count, self.intruder_obs_dim))
                if intruder_state is None:
                    intruder_state = remaining

                else:
                    intruder_state = np.append(intruder_state, remaining, axis=0)

                state[id_] = {
                    "ownship_obs": own_state.reshape(1, self.ownship_obs_dim),
                    "intruder_obs": intruder_state.reshape(
                        1, self.agent.max_agents, self.intruder_obs_dim
                    ),
                }

            else:
                state[id_] = {
                    "ownship_obs": own_state.reshape(1, self.ownship_obs_dim),
                    "intruder_obs": intruder_state.reshape(
                        1, self.agent.max_agents, self.intruder_obs_dim
                    ),
                }
            # === END OF MODIFIED SEGEMENT === 

            # === MODIFIED: Reward implmentation ===
            ### NOISE REWARD IMPLEMENTATION ###
            ## 1: Single Event Noise Calculation
            next_route_section_name = self.vehicle_helpers[id_].next_route_section
            current_route_section_name = self.vehicle_helpers[id_].current_route_section
            current_intersection_name = self.vehicle_helpers[id_].current_intersection
            ambient_noise = 40
            if current_route_section_name != None and current_route_section_name in self.ambient_noise_level.keys():
                ambient_noise = self.ambient_noise_level[current_route_section_name]
            else:
                if current_intersection_name != None and current_intersection_name in self.ambient_noise_level.keys():
                    ambient_noise = self.ambient_noise_level[current_intersection_name]
            
            if alt_own > 0:
                current_aircraft_noise = self.a_0 +  (self.a_1 * math.log10(alt_own_ft)) + (self.a_2 * ((math.log10(alt_own_ft))**2))
            else:
                current_aircraft_noise = 0

            ## 2: Group Event Noise
            inner_log_val = 0 # (10**(current_aircraft_noise / 10))
            for single_noise_val in intruder_noise_vals.values():
                inner_log_val += (10**(single_noise_val / 10))
            if inner_log_val != 0:
                total_noise_impact = 10 * math.log10(inner_log_val)
            else:
                total_noise_impact = 0

            ## Noise Reward Function
            total_noise_factored_own = current_aircraft_noise # - ambient_noise
            if total_noise_factored_own < 0:
                total_noise_factored_own = 0
            best_case_noise = self.a_0 +  (self.a_1 * math.log10(self.max_alt)) + (self.a_2 * ((math.log10(self.max_alt))**2))
            worst_case_noise = self.a_0 +  (self.a_1 * math.log10(0.1)) + (self.a_2 * ((math.log10(0.1))**2))

            best_case_noise_increase = best_case_noise # - ambient_noise
            
            if best_case_noise < 0:
                best_case_noise_increase = 0

            worst_case_noise_increase = worst_case_noise # - ambient_noise
            
            noise_increase_over_best_case = total_noise_factored_own # - best_case_noise_increase
            noise_increase_normalized = (total_noise_factored_own - best_case_noise_increase) / (worst_case_noise_increase - best_case_noise_increase)
            # Ambient noise needs to be included

            # Total noise is around ~ -26per time step. Max negative reward per time step is 
            total_noise_factored = noise_increase_over_best_case
            if total_noise_factored >= 0:
                total_noise_factored = 0

            rew[id_] -= self.weighting_factor_noise * noise_increase_normalized

            # Scale reward based on closeness of aircraft
            ### SEPARATION REWARD IMPLEMENTATION
            weighted_congestion = 0
            max_dist = self.intruderThreshold
            min_dist = self.LOS
            for id_j in same_alt_list:
                idx_j = self.bs.traf.id2idx(id_j)
                dist_ij = d[i, idx_j]
                
                if dist_ij >= min_dist and dist_ij <= max_dist:
                    proximity_weight = (max_dist - dist_ij) / (max_dist - min_dist)
                    weighted_congestion += proximity_weight
                elif dist_ij < min_dist:
                    weighted_congestion += 1
            
            max_aircraft_at_same_altitude = 10
            max_weighted_congestion = max_aircraft_at_same_altitude * 1
            normalized_congestion = min(weighted_congestion / max_weighted_congestion, 1)

            rew[id_] += -1 * (self.weighting_factor_separation) * (normalized_congestion)

            ### ENERGY REWARD IMPLEMENTATION
            if not init and not done[id_]:
                if a is not None:
                    if id_ in a.keys():
                        energy_reward = 0

                        action_taken = a[id_].copy()
                        if np.abs(alt_own_ft - 1000) < 1:
                            if action_taken == 0:
                                action_taken = 1

                        
                        if id_ in self.alt_changing.keys() and self.alt_changing_time_steps[id_] != 0:
                            action_taken = 1

                        if action_taken == 2 and alt_own_ft >= 1000 and alt_own_ft <= 3000: # or action_taken == 0:
                            self.alt_adjustments[id_] += 1
                            energy_reward = self.fixed_energy_cost * self.alt_adjustments[id_]
                        
                        rew[id_] += self.weighting_factor_energy * energy_reward
        # === END OF MODIFIED SEGEMENT === 

        return state, rew, done, info

    def run_one_iteration(self, weights):
        """
        2022/11/1 modify the policy implementation and introduce the non-cooperative behaviors
        """
        if self.agent.equipped:
            self.agent.model.set_weights(weights)
        # self.agent.data = {}
        self.agent.reset()
        self.step_counter = 0

        if self.episode_done:
            obs = self.reset()
            self.nmacs = 0
            self.nmac_time = 0
            self.total_ac = 0
        else:
            obs = self.last_obs

        while True:
            if len(obs) > 0:
                action, policy, value = self.agent.predict(
                    obs, self.non_coop_tag, self.LControl_lst, self.LComm_lst
                )

            else:
                action, policy, value = {}, {}, {}

            next_obs, rew, term, term_type = self.step(action, policy, value)

            next_obs = self.store_data(
                obs, action, rew, next_obs, term, term_type, policy, value
            )

            obs = next_obs

            if term["__all__"] or self.step_counter >= self.max_steps:
                self.last_obs = next_obs

                if term["__all__"]:
                    self.episode_done = True

                if self.step_counter >= self.max_steps:
                    self.step_counter = 0

                for id_ in self.agent.memory.keys():
                    # if the id_ has already been processed then skip it
                    if id_ in self.agent.data.keys():
                        continue
                    self.agent.process_memory(id_)
                for key in self.agent.data.keys():
                    if type(self.agent.data[key]) == list:
                        self.agent.data[key] = np.concatenate(
                            self.agent.data[key], axis=0
                        )
                self.agent.data["nmacs"] = self.nmacs
                self.agent.data["nmac_time"] = self.nmac_time
                self.agent.data["total_ac"] = self.total_ac

                if self.episode_done:  # self.run_type == "eval":
                    self.agent.data["aircraft"] = self.acInfo

                if not "raw_reward" in self.agent.data:
                    self.agent.data["raw_reward"] = np.array([0.0])
                self.agent.data["max_noise_increase"] = self.max_noise_increase
                self.agent.data["avg_noise_increase"] = self.average_noise_increase
                self.agent.data["congestion_distribution"] = self.congestion_distribution
                self.agent.data["environment_done"] = self.episode_done

                # Add this in
                self.agent.data["num_alt_adjustments"] =  self.alt_adjustments

                data_ID = ray.put([self.agent.data, self.id])
                return data_ID

    def store_data(self, obs, action, rew, next_obs, term, term_type, policy, value):
        obs_updated = copy(next_obs)
        for ac_id in obs.keys():
            self.agent.store_step(
                ac_id, obs, action, rew, next_obs, term, policy, value
            )

            if term[ac_id]:
                self.total_ac += 1

                # did an NMAC occur
                if 1 in self.acInfo[ac_id]["NMAC"]:
                    # if term_type[ac_id] == 1:
                    group = groupby(self.acInfo[ac_id]["NMAC"])
                    group = np.array([x[0] for x in group])

                    self.nmac_time += (
                        sum(self.acInfo[ac_id]["NMAC"]) * self.bs.sim.simdt
                    )
                    self.nmacs += sum(group)
                self.bs.stack.stack("DEL {}".format(ac_id))
                del obs_updated[ac_id]

        return obs_updated

    def normalization_check(self, x=None, y=None, d=None):
        traf = self.bs.traf

        if len(traf.lat) == 0:
            return

        if traf.cas.max() > self.tas_max:
            self.tas_max = traf.cas.max()

        if traf.cas.min() < self.tas_min:
            self.tas_min = traf.cas.min()

        if traf.ax.max() > self.ax_max:
            self.ax_max = traf.ax.max()
        if traf.ax.min() < self.ax_min:
            self.ax_min = traf.ax.min()

        if x is not None:
            if x < self.min_x:
                self.min_x = x

            elif x > self.max_x:
                self.max_x = x

            # prevent numerical normalization errors
            if self.max_x == self.min_x:
                self.max_x += 1e-4

        if y is not None:
            if y < self.min_y:
                self.min_y = y

            elif y > self.max_y:
                self.max_y = y

            # prevent numerical normalization errors
            if self.max_y == self.min_y:
                self.max_y += 1e-4

        if d is not None:
            if d > self.max_d:
                self.max_d = d

        # === MODIFIED: This is where we define the maximum and minimum values for the ownship state and intruder state. ===
        ownship_min_state = [0, 0, 0, 0, 0]
        ownship_max_state = [2, self.max_alt, 1, self.max_alt, 50]

        intruder_min_state = [
            0,
            0,
            -self.max_alt
        ]

        intruder_max_state = [
            self.max_d,
            2,
            self.max_alt
        ]
        # === END OF MODIFIED SEGEMENT === 

        self.context_space = Box(
            np.array(intruder_min_state),
            np.array(intruder_max_state),
            dtype=np.float64,
        )

        self.observation_space = Box(
            np.array(ownship_min_state),
            np.array(ownship_max_state),
            dtype=np.float64,
        )
        ### End of Adjusted Code

    def within_LOS(self, id_i):
        '''
        This function is used to ensure safe separation between aircraft on takeoff. It checks to see if the aircraft are within the LOS distance on takeoff.
        '''
        n_ac = self.bs.traf.lat.shape[0]
        d = (
            geo.kwikdist_matrix(
                np.repeat(self.bs.traf.lat, n_ac),
                np.repeat(self.bs.traf.lon, n_ac),
                np.tile(self.bs.traf.lat, n_ac),
                np.tile(self.bs.traf.lon, n_ac),
            ).reshape(n_ac, n_ac)
            * geo.nm
        )
        i = self.bs.traf.id2idx(id_i)
        for j in range(self.bs.traf.lat.shape[0]):
            id_j = self.bs.traf.id[j]
            if id_i == id_j:
                continue
            if not self.bs.traf.active[
                j
            ]:
                continue
            dist = d[i, j]
            if dist <= 1.5 * self.LOS:
                return True
        return False
    
    def meters_to_feet(self, meters):
        feet = meters * 3.28084
        return feet
