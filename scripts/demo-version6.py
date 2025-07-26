'''
#@Time      :7/26/22 14:49
#@Author    : Chelsea Li
#@File      :demo-version5.py
#@Software  :PyCharm
'''

import numpy as np
from random import expovariate
from geopy.distance import geodesic
from tqdm import tqdm
import random
import time

class Simulator(object):
    import bluesky as bs
    def __init__(
            self,
            filename,
            AC_nums,
            Route_list,
            ROUTE,
            aclat,
            aclon,
            sec=12 * 3600,  # seconds per day
            n_steps=8000,  # simulation time
            MAC_dist=10,  # meter
            NMAC_dist = 25,  # meter
            LOS_dist = 100, # meter
    ):

        self.sec = sec
        self.AC_nums = AC_nums
        self.MAC_dist = MAC_dist
        self.n_steps = n_steps
        self.NMAC_dist = NMAC_dist
        self.LOS_dist = LOS_dist

        # route information
        self.Route_list=Route_list
        self.ROUTE=ROUTE
        self.aclat=aclat
        self.aclon=aclon

        self.NMAC = 0
        self.MAC = 0
        self.LOS = 0

        self.ac_list=[]
        self.alt_list=[]
        self.lat_list=[]
        self.lon_list=[]
        self.plane_list=[]
        self.los_sep=[]
        self.nmac_dis=[]
        self.mac_dis=[]
        self.bs.init('sim-detached')
        self.f=open(filename,"w")


    def initial(self):
        # initialize bluesky as non-networked simulation node
        self.bs.stack.stack('CRELOG demo 1')
        self.bs.stack.stack('demo ADD id,lat, lon, alt, tas, vs ')
        self.bs.stack.stack('demo ON 1  ')
        self.bs.stack.stack('TAXI OFF 4')
        self.f.write("00:00:00.00>TRAILS ON \n")
        self.f.write("\n")
        # f.write("00:00:00.00>CIRCLE a, 32.7783,-96.8046 0.2 \n")
        # f.write("00:00:00.00>CIRCLE b, 33.1559,-96.872 0.2 \n")
        self.f.write("0:00:00.00>PAN 32.77173371	-96.83678249 \n")
        self.f.write("0:00:00.00>ZOOM 2 \n")
        self.f.write("00:00:00.00>TAXI OFF 4\n")
        # f.write("0:00:00.00>FF\n")
        self.f.write("\n")


    # compute the distance of two aircraft, meters
    def get_distance(self,location1, location2):
        lat1 = location1[0]
        lon1 = location1[1]
        alt1 = location1[2]
        lat2 = location2[0]
        lon2 = location2[1]
        alt2 = location2[2]
        horizon_dist = geodesic((lat1, lon1), (lat2, lon2)).m
        # dist = sqrt(horizon_dist ** 2 + (alt1 - alt2) ** 2)
        return horizon_dist


    # Generate the demand based on poisson distribution
    # simulate the 'number' events in a poisson process, lambda-number of flight per second
    # lambda=0.1--flight interval=10s
    def departure_time(self):
        random.seed(100)
        # calculate arrival rate
        AC_intervals = []
        for i in range(len(self.AC_nums)):
            AC_intervals.append(self.sec/(self.AC_nums[i]*10))


        # generate departure time for each route
        dep_time = []
        dep_time_dict = {}
        for iters in range(len(self.AC_nums)):
            dep_time.append([])

        for i in range(len(self.AC_nums)):
            route_id = self.Route_list[i]
            lambda_x = 1 / (AC_intervals[i])
            ac_demand_interval =[int(expovariate(lambda_x)) for i in range(self.AC_nums[i])]
            dep_time[i] = np.cumsum(ac_demand_interval)
            curr_dep = dep_time[i]
            # print(dep_time[i])

            # ascend ordering departure time
            for idx in range(len(curr_dep)):
                aircraft_No = route_id + str(idx)
                dep_time_dict[aircraft_No] = curr_dep[idx]
        dep_dict_ordered = dict(sorted(dep_time_dict.items(), key=lambda item: item[1]))
        #print(dep_dict_ordered)
        return dep_dict_ordered


    def add_plane(self, plane, depart_time, route_length,route_id):
        time = "00:00:" + str(depart_time) + ".00"
        self.f.write(time + ">CRE " + plane + ",Mavic," + route_id + "1,0,0" + "\n")
        #self.f.write(time + ">LISTRTE " + plane + "\n")

        #### Commented out for now until we are sure we don't need it ####
        # self.bs.stack.stack("ORIG " + plane + " " + route_id + "1")
        # self.bs.stack.stack("DEST " + plane + " " + route_id + str(route_length))
        # self.f.write(time + ">ORIG " + plane + " " + route_id + "1\n")
        # self.f.write(time + ">DEST " + plane + " " + route_id + str(route_length) + "\n")

        self.bs.stack.stack("SPD " + plane + " 30")
        self.f.write(time + ">SPD " + plane + " 30" + "\n")

        self.bs.stack.stack("ALT " + plane + " 400")
        self.f.write(time + ">ALT " + plane + " 400" + "\n")

        for j in range(1, route_length + 1):
            wpt = route_id + str(j)
            self.bs.stack.stack("ADDWPT " + plane + " " + wpt + ", 400, 40")
            self.f.write(time + ">ADDWPT " + plane + " " + wpt + ", 400, 40" + "\n")

        self.f.write(time + ">" + plane + " VNAV on \n")
        self.bs.stack.stack(f'DUMPRTE {plane}')
        self.bs.stack.stack(f'VNAV {plane} ON')

        self.f.write("\n")

    def safety_metric(self):
        for j in range(len(self.lat_list) - 1):
            for k in range(1, len(self.lat_list) - j):
                loc1 = [self.lat_list[j], self.lon_list[j], self.alt_list[j]]
                loc2 = [self.lat_list[j + k], self.lon_list[j + k], self.alt_list[j + k]]
                dist = self.get_distance(loc1, loc2)
                # print(f'distance bwt {plane_list[j]} and {plane_list[j+k]} is {dist}')
                # print("Dist is {}, LOS_dist is {}, the gap is {}".format(dist,LOS_dist,LOS_dist - dist))
                if dist <= self.LOS_dist:  ## loss separation 100m
                    if [self.plane_list[j], self.plane_list[j + k]] in self.los_sep:
                        continue
                    else:
                        self.los_sep.append([self.plane_list[j], self.plane_list[j + k]])
                        self.LOS += 1
                        # print(f'{plane_list[j]} and {plane_list[j+k]} loss separation.')

                # remove plane if they get a safe separation
                if [self.plane_list[j], self.plane_list[j + k]] in self.los_sep and dist > self.LOS_dist:
                    self.los_sep.remove([self.plane_list[j], self.plane_list[j + k]])
                    # print(f'{plane_list[j]} and {plane_list[j+k]} get a safe separation.')

                if dist <= self.NMAC_dist:  ## near mid air collision in-air collision 25m
                    if [self.plane_list[j], self.plane_list[j + k]] in self.nmac_dis:
                        continue
                    else:
                        self.nmac_dis.append([self.plane_list[j], self.plane_list[j + k]])
                        self.NMAC += 1
                        # print(f'{plane_list[j]} and {plane_list[j + k]} have NMAC.')

                # remove plane if they get a safe separation
                if [self.plane_list[j], self.plane_list[j + k]] in self.nmac_dis and dist > self.NMAC_dist:
                    self.nmac_dis.remove([self.plane_list[j], self.plane_list[j + k]])
                    # print(f'{plane_list[j]} and {plane_list[j+k]} get a safe separation.')

                if dist <= self.MAC_dist:  ## Mid air collision 10m
                    if [self.plane_list[j], self.plane_list[j + k]] in self.mac_dis:
                        continue
                    else:
                        self.mac_dis.append([self.plane_list[j], self.plane_list[j + k]])
                        self.MAC += 1
                        # print(f'{plane_list[j]} and {plane_list[j + k]} have MAC.')

                    # remove plane if they get a safe separation
                if [self.plane_list[j], self.plane_list[j + k]] in self.mac_dis and dist > self.MAC_dist:
                    self.mac_dis.remove([self.plane_list[j], self.plane_list[j + k]])
                    # print(f'{plane_list[j]} and {plane_list[j + k]} get a safe separation.')

        return self.LOS, self.NMAC, self.MAC


    def generate_demo(self):
        self.initial()
        dep_dict_ordered = self.departure_time()
        print(dep_dict_ordered)

        for i in tqdm(range(1, self.n_steps)):
            stat_time = time.time()
            self.bs.sim.step()
            step_time = time.time() - stat_time
            # print("{}th iter the step time is {}".format(i,step_time))
            # get agent coordinates from bs
            self.ac_list = self.bs.traf.id
            self.alt_list = self.bs.traf.alt
            self.lat_list = self.bs.traf.lat
            self.lon_list = self.bs.traf.lon

            for item in dep_dict_ordered.items():
                depart_time = item[1]
                if i == depart_time:
                    route_id = item[0][:-1]
                    print(route_id)
                    if route_id not in self.ROUTE:
                        route_id = item[0][:-2]

                    route_length = self.ROUTE[route_id]
                    depart_time = item[1]
                    plane = "P" + item[0]
                    self.bs.traf.cre(acid="P" + item[0], actype="Mavic", aclat=self.aclat[route_id], aclon=self.aclon[route_id],
                                acalt=0, acspd=3)
                    self.add_plane(plane, depart_time, route_length,route_id)
                    self.plane_list.append(plane)
                    ###### test
                    # for i in range(self.bs.traf.lat.shape[0]):
                    #     print(self.bs.traf.id[i])
                    # update coordinate list after add the plane
                    self.ac_list = self.bs.traf.id
                    self.alt_list = self.bs.traf.alt
                    self.lat_list = self.bs.traf.lat
                    self.lon_list = self.bs.traf.lon

            # check in-air collision
            if i % 100 == 0:
                if len(self.lat_list) <= 1:
                    continue
                else:
                    self.LOS, self.NMAC, self.MAC = self.safety_metric()

        self.bs.stack.stack('STOP')
        self.f.close()
        print(f"number of loss separation:{self.LOS}")
        print(f"number of Near mid air collision:{self.NMAC * 2}")
        print(f"number of Mid air collision:{self.MAC * 2}")
        return [self.LOS, self.NMAC, self.MAC]

if __name__ == "__main__":

    #AC_nums = [10,2,3,7,8,4,6]
    AC_nums =  random.sample(range(1,20), 18)

    Route_list = ["I30L", "I30R", "635L", "635R", "CENL", "CENR","H12L","H12R","NHWR",
             "NHWL","COLR","COLL","BUSL","BUSR","TWYR","TWYL","I35R","I35L"]
    ROUTE = {"I30L": 20, "I30R": 16, "635L": 15, "635R": 15, "CENL": 21, "CENR": 18,"H12L":6,"H12R":6,"NHWR":14,
             "NHWL":17,"COLR":9,"COLL":12,"BUSL":20,"BUSR":21,"TWYR":15,"TWYL":12,"I35R":16,"I35L":13 }

    aclat = {"I30L": 32.77173371, "I30R": 32.89472323, "635L": 32.79248353, "635R": 32.91266151, "CENL": 33.17374526,
                  "CENR": 32.76810334,"H12L":32.78784263,"H12R":32.86871134	,"NHWR":32.87094142,"NHWL":33.0205793,
             "COLR":33.17378337,"COLL":33.02200248,"BUSL":32.84906489,"BUSR":32.98687099,"TWYR":32.77835826,
             "TWYL":33.09719204,"I35R":32.80658394,"I35L":33.1870428}
    aclon = {"I30L": -96.83678249, "I30R": -96.458926, "635L": -96.62999304, "635R": -96.9061115, "CENL": -96.64011822,
                  "CENR": -96.78229252,"H12L":-96.69016006,"H12R":-96.71687049,"NHWR":-96.81892784,"NHWL":-96.5171374,
             "COLR":-96.64029631,"COLL":-96.99358243,"BUSL":-96.56115236,"BUSR":-96.93858034,"TWYR":-96.80584122,
             "TWYL":-96.82957354,"I35R":-96.8236118,"I35L":-97.12696397}


    new_simulator = Simulator("demo-version8.scn",AC_nums,Route_list,ROUTE,aclat,aclon)
    new_simulator.generate_demo()












