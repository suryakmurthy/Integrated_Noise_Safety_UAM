'''
#@Time      :5/28/22 08:17
#@Author    :Chelsea Li
#@File      :Scenario_gen.py
#@Software  :PyCharm
'''

import numpy as np
from random import expovariate
from geopy.distance import geodesic
from math import sqrt
import itertools
from tqdm import tqdm
import bluesky as bs

f=open("metric-test.scn","w")

f.write("00:00:00.00>TRAILS ON \n")
f.write("\n")
# f.write("00:00:00.00>CIRCLE a, 32.7783,-96.8046 0.2 \n")
# f.write("00:00:00.00>CIRCLE b, 33.1559,-96.872 0.2 \n")
f.write("0:00:00.00>PAN 32.77173371	-96.83678249 \n")
f.write("0:00:00.00>ZOOM 2 \n")
f.write("00:00:00.00>TAXI OFF 4\n")
# f.write("0:00:00.00>FF\n")
f.write("\n")

#conpute the distance of two aircraft, meters
def get_distance(location1,location2):
    lat1=location1[0]
    lon1=location1[1]
    alt1=location1[2]
    lat2=location2[0]
    lon2=location2[1]
    alt2=location2[2]
    horizon_dist=geodesic((lat1,lon1), (lat2,lon2)).m
    dist=sqrt(horizon_dist**2+(alt1-alt2)**2)
    return horizon_dist


# Generate the demand based on poisson distribution
# simulate the 'number' events in a poisson process with an arrival rate(lambda_x) of 'interval' arrivals per second
def generate_interval(interval,number):
    # seed(100)
    lambda_x = 1/interval
    ac_demand_interval = [int(expovariate(lambda_x)) for i in range(number)]
    depart_time = np.cumsum(ac_demand_interval)
    depart_time_ori = depart_time.copy()
    return depart_time,depart_time_ori


ORIG_list = ['K','R','Q','L','F','G']
ROUTE = {"K":20, "R":16, "Q":10,"L":16,"F":14,"G":20}

n= 10
inv = (12*3600)/58
AC_nums = [n,n,n,n,n,n]   #number of total flights each route
AC_intervals =[(12*3600)/102,(12*3600)/150,(12*3600)/90,(12*3600)/75,(12*3600)/200,inv]  # arrival rate pre second for each route

check_inv = 10
MAC_dist = 10 #meter
NMAC_dist = 25 #meter
LOS_dist = 100 #meter
NMAC=0
LOS=0
MAC = 0
### Generate departure time for six routes under exponential distribution
dep_time = []
dep_time_ori = []
dep_time_dict = {}
for iters in range(len(AC_nums)):
    dep_time.append([])
    dep_time_ori.append([])


for i in range(len(AC_nums)):
    route_id=ORIG_list[i]
    dep_time[i],dep_time_ori[i] = generate_interval(AC_intervals[i],AC_nums[i])
    curr_dep = dep_time[i]

    for idx in range(len(curr_dep)):
        aircraft_No = route_id + str(idx)
        dep_time_dict[aircraft_No] = curr_dep[idx]
dep_dict_ordered = dict(sorted(dep_time_dict.items(), key=lambda item: item[1]))
# print(dep_dict_ordered)


###  generate scn demo
# for item in dep_dict_ordered.items():
#     route_id = item[0][0]
#     route_length = ROUTE[route_id]
#     depart_time = item[1]
#     plane = "P" + item[0]
#     time = "00:00:" + str(depart_time) + ".00"
#     f.write(time + ">CRE " + plane + ",Mavic," + route_id + "1,0,0" + "\n")
#     # f.write(time+">LISTRTE "+plane+"\n")
#     f.write(time + ">ORIG " + plane + " " + route_id + "1\n")
#     f.write(time + ">DEST " + plane + " " + route_id + str(route_length) + "\n")
#     f.write(time + ">SPD " + plane + " 30" + "\n")
#     f.write(time + ">ALT " + plane + " 400" + "\n")
#     for j in range(1, route_length + 1):
#         wpt = route_id + str(j)
#         f.write(time + ">ADDWPT " + plane + " " + wpt + " 400 40" + "\n")
#     f.write(time + ">" + plane + " VNAV on \n")
#     f.write("\n")

bs.init('sim-detached')
n_steps = list(dep_dict_ordered.items())[-1][1] +1

for i in tqdm(range(1, n_steps)):
    bs.sim.step()
    ac_list = bs.traf.id
    alt_list = bs.traf.alt
    lat_list = bs.traf.lat
    lon_list = bs.traf.lon
    spd_list = bs.traf.tas

    for item in dep_dict_ordered.items():
        depart_time = item[1]
        if i == depart_time:
            route_id = item[0][0]
            route_length = ROUTE[route_id]
            depart_time = item[1]
            plane = "P" + item[0]
            time = "00:00:" + str(depart_time) + ".00"
            f.write(time + ">CRE " + plane + ",Mavic," + route_id + "1,0,0" + "\n")
            f.write(time + ">ORIG " + plane + " " + route_id + "1\n")
            f.write(time + ">DEST " + plane + " " + route_id + str(route_length) + "\n")
            f.write(time + ">SPD " + plane + " 30" + "\n")
            f.write(time + ">ALT " + plane + " 400" + "\n")
            for j in range(1, route_length + 1):
                wpt = route_id + str(j)
                f.write(time + ">ADDWPT " + plane + " " + wpt + " 400 40" + "\n")
            f.write(time + ">" + plane + " VNAV on \n")
            f.write("\n")

    if i%1==0:
    #if i%check_inv==0:
        if len(lat_list)<=1:
            continue
        else:
            for j in range(len(lat_list)-1):
                loc1=[lat_list[j],lon_list[j],alt_list[j]]
                loc2=[lat_list[j+1],lon_list[j+1],alt_list[j+1]]
                dist=get_distance(loc1,loc2)

                if dist<LOS_dist:  ## loss separation
                    LOS+=1
                if dist<NMAC_dist:  ## near mid air collision in-air collision
                    NMAC+=1
                if dist<MAC_dist:  ## Mid air collision
                    MAC+=1


print(f"number of loss separation:{LOS}")
print(f"number of Near mid air collision:{NMAC*2}")
print(f"number of Mid air collision:{MAC*2}")
print(dep_dict_ordered)
f.close()