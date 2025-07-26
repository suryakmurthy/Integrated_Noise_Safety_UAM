"""
This script generates the scenario file for K1 ~ K20.
Nav.dat needs to be updated to provide the waypoints files.
"""

from random import choice
f=open("i30l.scn","w")

f.write("00:00:00.00>TRAILS ON \n")
f.write("\n")
f.write("00:00:00.00>CIRCLE a, 32.7783,-96.8046 0.2 \n")
f.write("00:00:00.00>CIRCLE b, 33.1559,-96.872 0.2 \n")
# f.write("00:00:00.00>CIRCLE c, 40.678505,-74.029101 0.2 \n")
# f.write("00:00:00.00>CIRCLE d, 40.712367,-73.976248 0.2 \n")
# f.write("00:00:00.00>CIRCLE e, 40.743822,-73.970742 0.2 \n")
# f.write("00:00:00.00>CIRCLE f, 40.7779659,-73.8926095 0.2 \n")
# f.write("00:00:00.00>CIRCLE g, 40.6668873,-73.8055968 0.2 \n")
# f.write("00:00:00.00>CIRCLE h, 40.775976,-73.94206 0.2 \n")
# f.write("00:00:00.00>CIRCLE i, 40.78281,-73.935198 0.2 \n")
f.write("\n")
n=1        #number of total flights
dep_inte=10  #departure interval for each resource/second



ORIG_list = ['K']
ROUTE = {"K":20}

for i in range(n):
    plane="A"+str(i)
    route_id=choice(ORIG_list)
    route_length=ROUTE[route_id]
    time="00:00:"+str(i*dep_inte)+".00"
    f.write(time+">CRE "+plane+",Mavic,"+route_id+"1,0,0"+"\n")
    # f.write(time+">LISTRTE "+plane+"\n")
    f.write(time+">ORIG "+plane+" "+route_id+"1\n")
    f.write(time+">DEST "+plane+" "+route_id+str(route_length)+"\n")
    f.write(time+">SPD "+plane+" 30"+"\n")
    f.write(time+">ALT "+plane+" 400"+"\n")
    for j in range(1, route_length+1):
        wpt=route_id+str(j)
        f.write(time+">ADDWPT "+plane+" "+wpt+" 400 40"+"\n")
    f.write(time+">"+plane+" VNAV on \n")
    f.write("\n")

f.close()