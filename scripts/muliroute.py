'''
#@Time      :5/8/22 23:02
#@Author    : Chelsea Li
#@File      :muliroute.py
#@Software  :PyCharm
'''

"""
use this script to generate scenario file which contains multiple intersected routes
"""

from scipy.stats import poisson
from random import choice
f=open("multiroute.scn","w")

f.write("00:00:00.00>TRAILS ON \n")
f.write("\n")
f.write("00:00:00.00>CIRCLE a, 32.7783,-96.8046 0.2 \n")
f.write("00:00:00.00>CIRCLE b, 33.1559,-96.872 0.2 \n")

f.write("\n")


n = poisson.rvs(mu=58, size=1)  #number of total flights

#dep_inte=10
dep_inte=30

ORIG_list = ['K', 'Q','L']
ROUTE = {"K":20, "Q":10,"L":20}

for i in range(n):
    plane="A"+str(i)
    route_id=choice(ORIG_list)

    if i == 0:
        route_id = 'K'
    elif i == 1:
        route_id = 'L'
    else:
        route_id = 'Q'
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