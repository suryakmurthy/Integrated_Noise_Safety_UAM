"""
2022/5/11: add randomness to the aircraft generation in three routes
"""


from scipy.stats import poisson
from random import choice
f=open("multiroute.scn","w")

f.write("00:00:00.00>TRAILS ON \n")
f.write("\n")
f.write("00:00:00.00>CIRCLE a, 32.7783,-96.8046 0.2 \n")
f.write("00:00:00.00>CIRCLE b, 33.1559,-96.872 0.2 \n")

f.write("\n")


# n_k = poisson.rvs(mu=58, size=1)  #number of total flights

# this control the max number of aircraft in each route
nk = poisson.rvs(mu=20, size=1)[0]
nq = poisson.rvs(mu=30, size=1)[0]
nl = poisson.rvs(mu=50, size=1)[0]

# counter
ck = 0
cq = 0
cl = 0

n = nk + nq + nl #100

#dep_inte=10
dep_inte=30

ORIG_list = ['K', 'Q','L']
ROUTE = {"K":20, "Q":10,"L":20}

for i in range(n):
    plane="A"+str(i)

    if (ck < nk) and (cq < nq) and (cl < nl):
        route_id=choice(['K', 'Q','L'])

    # ck = 20
    if (ck = nk) and (cq < nq) and (cl < nl):
        route_id = choice(['Q','L'])
    
    # cq = 30
    if (ck < nk) and (cq = nq) and (cl < nl):
        route_id=choice(['K', 'L'])  

    # cl = 50
    if (ck < nk) and (cq < nq) and (cl = nl):
        route_id=choice(['K', 'Q'])
    
    # only ck
    if (ck < nk) and (cq = nq) and (cl = nl):
        route_id=choice(['K'])   

    # only cq
    if (ck = nk) and (cq < nq) and (cl = nl):
        route_id=choice([ 'Q' ]) 

    # only cl
    if (ck = nk) and (cq = nq) and (cl < nl):
        route_id=choice(['L'])

    # counter add
    if route_id == 'K':
        ck += 1
    elif route_id == 'Q':
        cq += 1
    else:
        cl += 1

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