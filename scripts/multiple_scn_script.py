import os
import json


for i in range(100):
    jsonString = json.dumps({'iidx':i})
    jsonFile = open('pass_to_casestudy.json','w')
    jsonFile.write(jsonString)
    jsonFile.close()
    os.system('python demo-multiple.py')
