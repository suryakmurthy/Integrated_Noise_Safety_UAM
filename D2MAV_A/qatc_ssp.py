#!/usr/bin/env python3
"""
Author: Jesse Quattrociocchi
"""
import yaml
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.collections as collections
from PIL import Image
from shapely.geometry import Point, MultiPoint, LinearRing, LineString, MultiLineString, Polygon
from bluesky.tools import geo
from pyproj import Transformer
from collections import deque

FILE_PREFIX = ''
TOWER_CONFIG_FILE = FILE_PREFIX + 'Austin_towers.yaml'
BACKGROUND_IMAGE = FILE_PREFIX + 'DFW_intersections.png'


# Create custom errors
class NoIntersection(Exception):
    pass


class BadLogic(Exception):
    pass


class Tower:
    def __init__(self, tower_ID: str, max_slots: int):
        self.max_slots = max_slots
        # Number of slots available regardless of who is authorized or accepted
        # TODO: Swap from open_slots to current volume
        self.open_slots = deepcopy(max_slots)
        self.tower_ID = tower_ID
        self.inbound = None
        self.outbound = None
        self.req_queue = deque()  # Queue of requests
        self.accepted = []  # List of vid's that are known to be inside
        self.authorized = []  # List of vid's that are authorized to enter
        self.illegal = []  # List of vid's that were not accepted but are inside
        left_memory_limit = 5  # Number of vid's that have recently left that will be retained by the tower
        self.recently_left = deque(maxlen=left_memory_limit)  # List of vid's that have recently left the intersection.

        ### Amient Noise Value for area the tower covers
        self.ambient_noise_level = 0

    def set_volume(self):
        self.open_slots = self.max_slots - len(self.accepted) - len(self.authorized) - len(self.illegal)


class Intersection(Tower):
    def __init__(
            self,
            *args,
            towers,
            inbound,
            outbound,
            location,
            radius,
            **kwargs,
    ):
        # Base class init
        super().__init__(*args, **kwargs)
        self.location = location
        self.inbound_route_section_towers = {}
        self.max_altitude = 3000
        self.min_altitude = 1000
        alt_idx = self.max_altitude
        self.alt_levels = {}
        while alt_idx >= self.min_altitude:
            self.alt_levels[alt_idx] = []
            alt_idx -= 500
        for rs in inbound:
            # self.tower_ID in this context is the ID of the intersection that
            # gets created during the super().__init__ call
            # This means that in order to find the intersections connected to a route section,
            # you can ask the route section object directly which intersections it is connected to.
            towers[rs].inbound = self.tower_ID
            self.inbound_route_section_towers[rs] = towers[rs]

        self.outbound_route_section_towers = {}
        for rs in outbound:
            towers[rs].outbound = self.tower_ID
            self.outbound_route_section_towers[rs] = towers[rs]

        self.transformer = Transformer.from_crs("epsg:4326", "epsg:2163", always_xy=True)
        self.create_shapely_objects(location, radius)

    def create_shapely_objects(self, location, radius):
        # Input is in lat,lon degrees
        # Determine the type of shape to create
        if len(location) == 2 and len(radius) == 2:
            # Circle. radius is actually a point used to calculate the radius
            circle_radius = (geo.kwikdist(*location, *radius) * 1852) + 200  # meters
            x, y = self.transformer.transform(location[1], location[0])
            self.region_shape = Point(x, y).buffer(circle_radius)
            self.region_ring = LinearRing(self.region_shape.boundary)
        else:
            # Polygon
            temp_polygon = []
            for point in location:
                x, y = self.transformer.transform(point[1], point[0])
                temp_polygon.append((x, y))
            self.region_shape = Polygon(temp_polygon)  # Has area
            self.region_ring = LinearRing(temp_polygon)  # Shell/ring of polygon

    # # TODO: This function may be deprecated by newer functionality
    def route_section_check(self, route_section):
        '''Checks if a route section is managed by this intersection'''
        if route_section in list(self.inbound_route_section_towers.keys()):
            return 'inbound'
        elif route_section in list(self.outbound_route_section_towers.keys()):
            return 'outbound'
        else:
            return False

    def enter_request(self, id_, from_section, to_section) -> bool:
        # TODO: This request could also be extended to consider vehicles that are already authorized to leave
        # TODO: Need to add another Tower attribute that tracks the number of vehicles that are authorized to leave
        # if id_ == "PSAHL0":
        #     print("Entry Request: ", id_, from_section, to_section, self.tower_ID)
        if id_ == 'PPPDT12':
            print("Checking Entry Request: ", self.tower_ID, to_section)
        vls_active = False
        if not vls_active:
            out_flight_level = 3000
        else:
            out_flight_level = None
            for flight_level in self.alt_levels.keys():
                if id_ in self.alt_levels[flight_level]:
                    out_flight_level = flight_level
                    break
                if len(self.alt_levels[flight_level]) == 0 or all(aircraft[1:5] == id_[1:5] for aircraft in self.alt_levels[flight_level]):
                    out_flight_level = flight_level
                    if id_ not in self.alt_levels[flight_level]:
                        self.alt_levels[flight_level].append(id_)
                    break
        if to_section:
            a = self.outbound_route_section_towers[to_section].open_slots > 0
            b = self.open_slots > 0
            if a and b and out_flight_level != None:
                # Update the route section
                if id_ not in self.outbound_route_section_towers[to_section].authorized:
                    # print("Updating Route section: ", )
                    self.outbound_route_section_towers[to_section].authorized.append(id_)
                    self.outbound_route_section_towers[to_section].set_volume()
                # Update the intersection
                if id_ not in self.authorized:
                    self.authorized.append(id_)
                    self.set_volume()
                # TODO: The following is an example of how to update the inbound route section to account for the vehicle leaving
                # if from_section:
                #     self.inbound_route_section_towers[from_section].leaving.append(id_)
                return True  # Authorized to enter
            else:
                if id_ in self.outbound_route_section_towers[to_section].authorized and id_ in self.authorized:
                    return True
                else:
                    # print("rejecting 3: ", id_, self.tower_ID)
                    return False  # Not authorized to enter
        else:  # i.e. vehicle is exiting
            if self.open_slots > 0 and out_flight_level != None:
                # Update the intersection
                if id_ not in self.authorized:
                    self.authorized.append(id_)
                    self.set_volume()
                return True
            else:
                if id_ in self.authorized:
                    return True
                else:
                    # print("rejecting 4: ", id_, self.tower_ID)
                    return False  # Not authorized to enter


class TrafficManager:
    def __init__(self, tower_config, transformer: Transformer = None, section_patches=None):
        self.towers = self.create_towers(tower_config)
        # print("Creating Intersections")
        self.intersections = self.create_intersections(tower_config)
        self.current_requests = {}
        # Only used for plotting. Remains None if not plotting
        self.section_patches = section_patches

        if not transformer:
            self.transformer = Transformer.from_crs("epsg:4326", "epsg:2163", always_xy=True)
        else:
            self.transformer = transformer

    def reset(self):
        for tower in self.towers.values():
            tower.open_slots = deepcopy(tower.max_slots)
        for intersection in self.intersections.values():
            intersection.open_slots = deepcopy(intersection.max_slots)
        self.current_requests = {}

    def create_towers(self, tower_config) -> dict:
        out = {}
        for route in tower_config['Routes']:
            # route is type dict
            for section in route['sections']:
                out[section] = Tower(section, 1000)
        return out

    def create_intersections(self, tower_config) -> dict:
        I = {}
        # print("Tower config: ", tower_config['Intersections'])
        for intersection in tower_config['Intersections']:
            I[intersection['identifier']] = Intersection(
                intersection['identifier'],
                intersection['max_slots'],
                towers=self.towers,
                inbound=intersection['inbound'],
                outbound=intersection['outbound'],
                location=intersection['location'],
                radius=intersection['radius'],
            )
        # print("Intersections Created")
        return I

    def search_for_intersection(self, section, in_or_out) -> Intersection:
        # in_or_out = ['inbound','outbound']
        for k, intersection in self.intersections.items():
            # print("Checking for next intersection: ", k, intersection)
            result = intersection.route_section_check(section)
            # print(section)
            if result == in_or_out:
                return intersection
        # If no intersection was returned, throw and error
        raise NoIntersection("No intersection was found or in_or_out is not in the correct form")

    def check_if_within_intersection(self, position: list, intersection: str) -> bool:
        ''' position: list of [lon,lat] coordinates
            intersection: string of intersection ID '''
        x, y = self.transformer.transform(position[0], position[1])
        return self.intersections[intersection].region_shape.contains(Point(x, y))

    def add_request(self, vehicle_ID: str, formatted_request: tuple) -> None:
        self.current_requests[vehicle_ID] = formatted_request

    def process_requests(self, round_robin=False) -> dict:
        response = {}
        for id_, request in self.current_requests.items():
            # print("request: ", request, request[1])
            if request[2]:
                outbound_intersection = self.search_for_intersection(request[2], 'outbound')
                # print("Checking Request: ", id_, outbound_intersection)
                if outbound_intersection:
                    if id_[1:4] not in outbound_intersection.requesting_routes:
                        outbound_intersection.requesting_routes.append(id_[1:4])
                        outbound_intersection.requesting_routes = sorted(outbound_intersection.requesting_routes, key=lambda x:outbound_intersection.route_names.index(x))
                
            else:
                inbound_intersection = self.search_for_intersection(request[1], 'inbound')
                # print("request: ", request[0], inbound_intersection)
                if inbound_intersection:
                    if id_[1:4] not in inbound_intersection.requesting_routes:
                        inbound_intersection.requesting_routes.append(id_[1:4])
                        inbound_intersection.requesting_routes = sorted(inbound_intersection.requesting_routes, key=lambda x:inbound_intersection.route_names.index(x))
        
        for id_, request in self.current_requests.items():
            response[id_] = self.request(*request, round_robin=round_robin)
        
        for intersection in self.intersections.values():
            intersection.requesting_routes = []
            for id_ in intersection.rejected:
                if id_[1:4] not in intersection.requesting_routes:
                    intersection.requesting_routes.append(id_[1:4])
                    intersection.requesting_routes = sorted(intersection.requesting_routes, key=lambda x:intersection.route_names.index(x))
        # print("End of Request Processing \n")
        self.current_requests = {}
        return response
   
    def request(self, id_, from_section, to_section, srtf=False, round_robin=False) -> bool:
        if to_section:
            outbound_intersection = self.search_for_intersection(to_section, 'outbound')
            if srtf == True:
                return outbound_intersection.enter_request_srtf(id_, from_section, to_section)
            elif round_robin == True:
                return outbound_intersection.enter_request_round_robin(id_, from_section, to_section)
            else:
                return outbound_intersection.enter_request(id_, from_section, to_section)
        else:
            inbound_intersection = self.search_for_intersection(from_section, 'inbound')
            if srtf == True:
                return inbound_intersection.enter_request_srtf(id_, from_section, to_section)
            elif round_robin == True:
                return inbound_intersection.enter_request_round_robin(id_, from_section, to_section)
            else:
                return inbound_intersection.enter_request(id_, from_section, to_section)

    def set_patch_colors(self):
        for k in self.towers.keys():
            volume_used = float(self.towers[k].open_slots) / float(self.towers[k].max_slots)
            if volume_used < 0:
                self.section_patches[k].set_array(None)
            else:
                self.section_patches[k].set_array(np.array([volume_used]))


class Route:
    def __init__(self, route_id: str, sections: list, sectionWPs: dict):
        self.route_id = route_id
        self.route_sections = sections
        self.sectionWPs = sectionWPs
        self.sectionIntersectionInfo = {}
        self.shape = None
        self.section_patches = {}

    def create_route_generator(self):
        return iter(self.route_sections)


def load_routes(tower_config: dict, tm_object: TrafficManager, route_linestrings: dict) -> dict:
    # Load routes from: tower_config = yaml.load(file, Loader=yaml.Loader)
    routes = {}
    # print(route_linestrings)
    for route in tower_config['Routes']:
        routes[route['identifier']] = Route(
            route['identifier'],
            route['sections'],
            route['sectionWPs'],
        )
        routes[route['identifier']].shape = route_linestrings[route['identifier']]
        for section in routes[route['identifier']].route_sections:
            routes[route['identifier']].sectionIntersectionInfo[section] = {}
            inbound_id = tm_object.search_for_intersection(section, 'inbound').tower_ID
            routes[route['identifier']].sectionIntersectionInfo[section]['inbound'] = inbound_id
            outbound_id = tm_object.search_for_intersection(section, 'outbound').tower_ID
            routes[route['identifier']].sectionIntersectionInfo[section]['outbound'] = outbound_id
            # Replace This Function with the intersection point you have calculated.
            # print("Section Wps: ", routes[route['identifier']].sectionWPs)
            out = routes[route['identifier']].sectionWPs[0]
            # out = find_section_intersection_point(
            #     routes[route['identifier']].shape,
            #     tm_object.intersections[inbound_id].region_ring
            # )
            # TODO: Add diameter in meters of the intersection to sectionIntersectionInfo
            if out == None:
                print('Error: Intersection point is None')
            routes[route['identifier']].sectionIntersectionInfo[section]['inbound_wp'] = out  # x,y coordinate system
    return routes


def find_section_intersection_point(route_shape, intersection_shape):
    if intersection_shape.intersects(route_shape):
        intersected_shape = intersection_shape.intersection(route_shape)
        # grab the first point of the route shape
        # TODO: Confirm this will work for all intersections
        ref_point = Point(route_shape.xy[0][0], route_shape.xy[1][0])
        last_dist = 99999999.0
        if isinstance(intersected_shape, Point):
            pt_list = [intersected_shape]
        elif isinstance(intersected_shape, MultiPoint):
            pt_list = intersected_shape.geoms

        for pt in pt_list:
            dist = ref_point.distance(pt)  # Is this always positive?
            if dist < last_dist:
                closest_pt = pt
                last_dist = dist
        return closest_pt
    else:
        raise NoIntersection("No the given route and intersection do not have an intersecting shape")


class VehicleHelper:
    def __init__(self, vehicle_ID: str, route: Route, transformer=None):
        self.vehicle_ID = vehicle_ID
        self.route = route
        self.route_data = route.create_route_generator()
        # Take the first route section from route
        self.previous_route_section = None
        self.current_route_section = None
        self.next_route_section = None
        self.final_route_segment = False
        self.enter_request_status = False
        self.current_intersection = None
        self.next_intersection = None
        self.within_intersection = False  # Set at same time I inform TM that the vehicle has entered/exit
        self.get_next_route_section()
        self.initial_request_granted = False
        if not transformer:
            self.transformer = Transformer.from_crs("epsg:4326", "epsg:2163", always_xy=True)
        else:
            self.transformer = transformer

    def change_route_section(self) -> None:
        self.previous_route_section = deepcopy(self.current_route_section)
        self.current_route_section = deepcopy(self.next_route_section)
        # print("Changing route section: ", self.vehicle_ID, self.current_route_section)
        # TODO: set self.next_intersection to be the intersection that has self.current_route_section as an inbound route section
        # TODO: Try using the stored route object to find and save the next intersection.
        # print("Changing Route section: ", self.vehicle_ID, self.previous_route_section, self.current_intersection, self.current_route_section)
        self.get_next_route_section()

    def get_next_route_section(self) -> None:
        try:
            # Only increment for now
            self.next_route_section = next(self.route_data)
        except StopIteration:
            self.next_route_section = None
            self.final_route_segment = True

    def format_request(self) -> tuple:
        # Format the request as a tuple of (id_, from_section, to_section)
        # if self.vehicle_ID == "PPPDT12":
        #     print("Formatting Request: ", (self.vehicle_ID, self.current_route_section, self.next_route_section))
        return (self.vehicle_ID, self.current_route_section, self.next_route_section)

    def distance_to_next_boundary(self, current_position: list) -> float:
        ''' current_position: list of [lon,lat] coordinates '''
        # TODO: Implement the next_or_current_rs functionality
        # if next_or_current_rs == 'current':
        #     x, y = self.transformer.transform(current_position[0], current_position[1]) # x,y
        #     return self.route.sectionIntersectionInfo[self.current_route_section]['inbound_wp'].distance(Point(x,y))
        # elif next_or_current_rs == 'next':
        #     x, y = self.transformer.transform(current_position[0], current_position[1]) # x,y
        #     return self.route.sectionIntersectionInfo[self.next_route_section]['inbound_wp'].distance(Point(x,y))
        # else:
        #     raise BadLogic("next_or_current_rs must be either 'next' or 'current'")

        if self.current_route_section == None:  # Vehicle is entering the system for the first time
            return -1  # Value that will trigger an automatic request
        else:
            x, y = self.transformer.transform(current_position[0], current_position[1])  # x,y
            x_2, y_2 = self.transformer.transform(self.route.sectionIntersectionInfo[self.current_route_section]['inbound_wp']['wpLong'][1], self.route.sectionIntersectionInfo[self.current_route_section]['inbound_wp']['wpLat'][1])
            return Point(x_2, y_2).distance(Point(x, y))

    def check_if_request_eligible(self, position: list) -> bool:
        ''' position: list of [lon,lat] coordinates '''

        if self.distance_to_next_boundary(position) < 1000.0:  # x,y meters
            return True
        else:
            return False

class VLS:
    def __init__(
            self,
            vehicle_list,
            mode = "intersection"
    ):
        self.vehicle_list = {}
        self.route_ordering = []
        self.route_mapping = {}
        for v in vehicle_list:
            self.vehicle_list[v.vehicle_ID] = v
            next_route_section = v.next_route_section
            if next_route_section not in self.route_mapping.keys():
                self.route_mapping[next_route_section] = [v.vehicle_ID]
            else:
                self.route_mapping[next_route_section].append(v.vehicle_ID)
        self.mode = mode
        self.current_route = list(self.route_mapping.keys())[0]
        # self.create_ordering()
    
    def create_ordering(self):
        self.ordering = []
        
        for i, vid in enumerate(self.vehicle_list.keys()):
            ## Add in priority logic here
            self.ordering.append(vid)
    
    def add_vehicle(self, new_vehicle):
        v_id = new_vehicle.vehicle_ID
        self.vehicle_list[new_vehicle.vehicle_ID] = new_vehicle
        next_route_section = new_vehicle.next_route_section
        if next_route_section not in self.route_mapping.keys():
            self.route_mapping[next_route_section] = [v_id]
            if self.current_route not in self.route_mapping:
                self.current_route = next_route_section
        else:
            self.route_mapping[next_route_section].append(v_id)
        # self.ordering.append(new_vehicle.vehicle_ID)
    
    def remove_vehicle(self, removed_vehicle):
        v_id = removed_vehicle.vehicle_ID
        del self.vehicle_list[removed_vehicle.vehicle_ID]
        # if "Final" in self.route_mapping.keys() and v_id in self.route_mapping['Final']:
        #     self.route_mapping['Final'].remove(v_id)
        #     if self.route_mapping['Final'] == []:
        #         del self.route_mapping['Final']
        #         if self.route_mapping != {}:
        #             self.current_route = list(self.route_mapping.keys())[0]
        #     return
        # print("remove logic: ", v_id, removed_vehicle.current_route_section, removed_vehicle.next_route_section, self.route_mapping)
        c_route_section = removed_vehicle.current_route_section
        n_route_section = removed_vehicle.next_route_section
        if c_route_section in self.route_mapping.keys():
            if v_id in self.route_mapping[c_route_section]:
                self.route_mapping[c_route_section].remove(v_id)
            if self.route_mapping[c_route_section] == []:
                    del self.route_mapping[c_route_section]
                    if self.route_mapping != {}:
                        self.current_route = list(self.route_mapping.keys())[0]
        if n_route_section in self.route_mapping.keys():
            if v_id in self.route_mapping[n_route_section]:
                self.route_mapping[n_route_section].remove(v_id)
            if self.route_mapping[n_route_section] == []:
                    del self.route_mapping[n_route_section]
                    if self.route_mapping != {}:
                        self.current_route = list(self.route_mapping.keys())[0]
        # self.ordering.remove(removed_vehicle.vehicle_ID)

    def obtain_rank(self, query_vehicle):
        # print("obtain_rank: ", query_vehicle, self.route_mapping)
        next_route = self.vehicle_list[query_vehicle].next_route_section
        # if next_route == None:
        #     print("Rank Final Case: ", query_vehicle)
        #     next_route = 'Final'
        route_index = list(self.route_mapping.keys()).index(next_route)
        v_offset = 0
        for i in range(0, route_index):
            route_key = list(self.route_mapping.keys())[i]
            v_offset += len(self.route_mapping[route_key])
        v_index = self.route_mapping[next_route].index(query_vehicle) + 1
        return v_index + v_offset
    
    def get_next_vehicle(self):
        # print("Get Next Vehicle: ", self.current_route, self.route_mapping)
        output = self.route_mapping[self.current_route].pop(0)
        if self.route_mapping[self.current_route] == []:
            del self.route_mapping[self.current_route]
            if self.route_mapping != {}:
                    self.current_route = list(self.route_mapping.keys())[0]
        # print("Get Next Vehicle 2: ", self.current_route, self.route_mapping, output)
        return output

class Converter:
    def __init__(
            self,
            top_left_lat: float = 33.205171,
            top_left_long: float = -97.145291,
            bottom_right_lat: float = 32.744229,
            bottom_right_long: float = -96.443459,
            px_width: float = 2388,
            px_height: float = 1870,
    ):
        self.top_left_lat = top_left_lat
        self.top_left_long = top_left_long
        self.bottom_right_lat = bottom_right_lat
        self.bottom_right_long = bottom_right_long
        self.px_width = px_width
        self.px_height = px_height
        self.dd_width = abs(top_left_long - bottom_right_long)
        self.dd_height = top_left_lat - bottom_right_lat
        self.px_per_dd_long = px_width / self.dd_width
        self.px_per_dd_lat = px_height / self.dd_height

    def dd_to_px_xy(
            self,
            lat: float,
            long: float,
    ):
        # Origin is the top left with down and to the right as positive in px frame
        dd_lat_distance = self.top_left_lat - lat
        dd_long_distance = abs(self.top_left_long - long)
        px_x = dd_long_distance * self.px_per_dd_long
        px_y = dd_lat_distance * self.px_per_dd_lat
        return px_x, px_y


# *******************************#
def get_dist_between_points(point1: list, point2: list):
    return float(np.linalg.norm(np.array(point1) - np.array(point2)))


def get_path_angle(init, final):
    # init=[x,y]. final=[x,y]
    return np.arctan2(final[1] - init[1], final[0] - init[0])


def create_background():
    # Set up image
    im = Image.open(BACKGROUND_IMAGE)
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(im)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    return fig, ax


def create_section_patches(routes, ax) -> dict:
    section_patches = {}
    for key, route in routes.items():
        for dict_item in route.sectionWPs:
            plot_route = [dict_item['wpLat'], dict_item['wpLong']]
            temp_patches = []
            for i in range(len(plot_route[0]) - 1):
                px_X_init, px_Y_init = converter.dd_to_px_xy(plot_route[0][i], plot_route[1][i])
                px_X_final, px_Y_final = converter.dd_to_px_xy(plot_route[0][i + 1], plot_route[1][i + 1])
                height = get_dist_between_points([px_X_init, px_Y_init], [px_X_final, px_Y_final])
                height = -height  # invert b/c positive down
                angle = np.degrees(get_path_angle([px_X_init, px_Y_init], [px_X_final, px_Y_final]))
                angle = angle + 90.  # Add 90 degrees
                temp_patches.append(
                    patches.Rectangle((px_X_init, px_Y_init), 6, height, angle=angle, linewidth=6)
                )
            p = collections.PatchCollection(temp_patches, cmap=mpl.colormaps['RdYlGn'])
            p.set_clim([0, 1])
            p.set_array(np.array([1]))
            # Save the section patch to give to UTM()
            section_patches[dict_item['id']] = p
            # Add section patch to the figure ax object
            ax.add_collection(p)
    return section_patches


if __name__ == '__main__':
    print('No main function')
    '''
    load environment config data
    load route data
    Create traffic manager(config data)
    create route objects(config data, route data, traffic manager)
    create transformer
    Start simulation
    at every time step:
        create vehicles helpers for new vehicles
        determine if a vehicle is eligible for a request
        if eligible:
            collect request info
        once all requests are collected:
            send requests to traffic manager for processing
            get responses from traffic manager
            if response is True:
                update vehicle helpers
            else:
                place vehicle in queue # Different queue for vehicles currently in the system and vehicles entering for the first time

        feed state and request results info to external sources
    '''