'''
#@Time      :6/17/22 08:17
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
import random
import time



def initial():

    # initialize bluesky as non-networked simulation node
    bs.stack.stack('CRELOG demo 1')
    bs.stack.stack('demo ADD id,lat, lon, alt, tas, vs ')
    bs.stack.stack('demo ON 1  ')
    bs.stack.stack('TAXI OFF 4')
    f.write("00:00:00.00>TRAILS ON \n")
    f.write("\n")
    # f.write("00:00:00.00>CIRCLE a, 32.7783,-96.8046 0.2 \n")
    # f.write("00:00:00.00>CIRCLE b, 33.1559,-96.872 0.2 \n")
    f.write("0:00:00.00>PAN 32.77173371	-96.83678249 \n")
    f.write("0:00:00.00>ZOOM 2 \n")
    f.write("00:00:00.00>TAXI OFF 4\n")
    # f.write("0:00:00.00>FF\n")
    f.write("\n")


#compute the distance of two aircraft, meters
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
# simulate the 'number' events in a poisson process, lambda-number of flight per second
# lambda=0.1--flight interval=10s
def generate_interval(interval,number):
    random.seed(100)
    lambda_x = 1/interval
    ac_demand_interval = [int(expovariate(lambda_x)) for i in range(number)]
    depart_time = np.cumsum(ac_demand_interval)
    # depart_time_ori = depart_time.copy()
    return depart_time


def add_plane(plane,depart_time,route_length):
    time = "00:00:" + str(depart_time) + ".00"
    f.write(time + ">CRE " + plane + ",Mavic," + route_id + "1,0,0" + "\n")
    f.write(time+">LISTRTE "+plane+"\n")

    bs.stack.stack("ORIG " + plane + " " + route_id + "1")
    bs.stack.stack("DEST " + plane + " " + route_id + str(route_length))
    f.write(time + ">ORIG " + plane + " " + route_id + "1\n")
    f.write(time + ">DEST " + plane + " " + route_id + str(route_length) + "\n")

    bs.stack.stack("SPD " + plane + " 30")
    f.write(time + ">SPD " + plane + " 30" + "\n")

    bs.stack.stack("ALT " + plane + " 400")
    f.write(time + ">ALT " + plane + " 400" + "\n")

    for j in range(1, route_length + 1):
        wpt = route_id + str(j)
        bs.stack.stack( "ADDWPT " + plane + " " + wpt + " 400 40")
        f.write(time + ">ADDWPT " + plane + " " + wpt + " 400 40" + "\n")

    f.write(time + ">" + plane + " VNAV on \n")
    bs.stack.stack(f'DUMPRTE {plane}')
    bs.stack.stack(f'VNAV {plane} ON')

    f.write("\n")



ORIG_list = ['K','R','Q','L','F','G']
ROUTE = {"K":20, "R":16, "Q":10,"L":16,"F":14,"G":20}
# route start point lat and long
aclat = {"K":32.90472277, "R":32.76326513, "Q":32.79248353,"L":33.15157984,"F":32.92829639,"G":32.76982652}
aclon = {"K":-96.45960051, "R":-96.83494346, "Q":-96.62999304,"L":-96.65206987,"F":-96.7858238,"G":-96.77212655}

n= 10
inv = (12*3600)/300
AC_nums = [10,15,9,7,20,30]   #number of total flights each route
#AC_nums = [n]
# AC_intervals =[inv,inv,inv,inv,inv,inv]
AC_intervals =[(12*3600)/102,(12*3600)/150,(12*3600)/90,(12*3600)/75,(12*3600)/200,inv] # 12h*3600/numbers
#AC_intervals = [(12*3600)/102]
# arrival rate pre second for each route
MAC_dist = 10 #meter
NMAC_dist = 25 #meter
LOS_dist = 100 #meter
NMAC=0
MAC=0
LOS=0
plane_list = []
los_sep = []
nmac_dis = []
mac_dis = []
n_steps = 8000  # simulation time


### Generate departure time for six routes under exponential distribution
def departure_time(ORIG_list,AC_nums,AC_intervals):

    dep_time = []
    dep_time_dict = {}
    for iters in range(len(AC_nums)):
        dep_time.append([])

    for i in range(len(AC_nums)):
        route_id=ORIG_list[i]
        dep_time[i] = generate_interval(AC_intervals[i],AC_nums[i])
        curr_dep = dep_time[i]
        #print(dep_time[i])

        for idx in range(len(curr_dep)):
            aircraft_No = route_id + str(idx)
            dep_time_dict[aircraft_No] = curr_dep[idx]
    dep_dict_ordered = dict(sorted(dep_time_dict.items(), key=lambda item: item[1]))
    print(dep_dict_ordered)
    return dep_dict_ordered


# generate scn demo
f=open("demo_version4.scn","w")

bs.init('sim-detached')
initial()
dep_dict_ordered = departure_time(ORIG_list,AC_nums,AC_intervals)

for i in tqdm(range(1, n_steps)):
    # bs.stack.stack('DT 1;FF')
    stat_time = time.time()
    bs.sim.step()
    step_time = time.time() - stat_time
    #print("{}th iter the step time is {}".format(i,step_time))

    ac_list = bs.traf.id
    alt_list = bs.traf.alt
    lat_list = bs.traf.lat
    lon_list = bs.traf.lon

    for item in dep_dict_ordered.items():
        depart_time = item[1]
        if i == depart_time:
            route_id = item[0][0]
            route_length = ROUTE[route_id]
            depart_time = item[1]
            plane = "P" + item[0]
            bs.traf.cre(acid="P" + item[0], actype="Mavic", aclat=aclat[route_id], aclon=aclon[route_id], acalt=0, acspd=3)
            add_plane(plane, depart_time, route_length)
            plane_list.append(plane)

            # update coordinate list after add the plane
            ac_list = bs.traf.id
            alt_list = bs.traf.alt
            lat_list = bs.traf.lat
            lon_list = bs.traf.lon
            # print(plane)
            # print(plane_list)
            # print(lat_list)
            # print(lon_list)


    # in-air collision
    if i%10==0:
        if len(lat_list)<=1:
            continue
        else:
            for j in range(len(lat_list)-1):
                for k in range(1,len(lat_list)-j):
                    loc1=[lat_list[j],lon_list[j],alt_list[j]]
                    loc2=[lat_list[j+k],lon_list[j+k],alt_list[j+k]]
                    dist=get_distance(loc1,loc2)
                    #print(f'distance bwt {plane_list[j]} and {plane_list[j+k]} is {dist}')
                    #print("Dist is {}, LOS_dist is {}, the gap is {}".format(dist,LOS_dist,LOS_dist - dist))
                    if dist<=LOS_dist:  ## loss separation 100m
                        if [plane_list[j],plane_list[j+k]] in los_sep:
                            continue
                        else:
                            los_sep.append([plane_list[j], plane_list[j + k]])
                            LOS+=1
                            #print(f'{plane_list[j]} and {plane_list[j+k]} loss separation.')

                    # remove plane if they get a safe separation
                    if [plane_list[j],plane_list[j+k]] in los_sep and dist>LOS_dist:
                        los_sep.remove([plane_list[j],plane_list[j+k]])
                        #print(f'{plane_list[j]} and {plane_list[j+k]} get a safe separation.')

                    if dist<=NMAC_dist:  ## near mid air collision in-air collision 25m
                        if [plane_list[j],plane_list[j+k]] in nmac_dis:
                            continue
                        else:
                            nmac_dis.append([plane_list[j], plane_list[j + k]])
                            NMAC += 1
                            #print(f'{plane_list[j]} and {plane_list[j + k]} have NMAC.')

                    # remove plane if they get a safe separation
                    if [plane_list[j],plane_list[j+k]] in nmac_dis and dist>NMAC_dist:
                        nmac_dis.remove([plane_list[j],plane_list[j+k]])
                        #print(f'{plane_list[j]} and {plane_list[j+k]} get a safe separation.')


                    if dist<=MAC_dist:  ## Mid air collision 10m
                        if [plane_list[j], plane_list[j + k]] in mac_dis:
                            continue
                        else:
                            mac_dis.append([plane_list[j], plane_list[j + k]])
                            MAC += 1
                            #print(f'{plane_list[j]} and {plane_list[j + k]} have MAC.')

                        # remove plane if they get a safe separation
                    if [plane_list[j], plane_list[j + k]] in mac_dis and dist > MAC_dist:
                        mac_dis.remove([plane_list[j], plane_list[j + k]])
                        #print(f'{plane_list[j]} and {plane_list[j + k]} get a safe separation.')

bs.stack.stack('STOP')
f.close()

print(f"number of loss separation:{LOS}")
print(f"number of Near mid air collision:{NMAC*2}")
print(f"number of Mid air collision:{MAC*2}")



