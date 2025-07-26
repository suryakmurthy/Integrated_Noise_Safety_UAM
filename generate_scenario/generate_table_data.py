import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data from JSON file
with open('/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/log/no_training/vls_version_10.json', 'r') as file:
    data_10 = json.load(file)

with open('/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/log/no_training/vls_version_20.json', 'r') as file:
    data_20 = json.load(file)

with open('/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/log/no_training/vls_version_30.json', 'r') as file:
    data_30 = json.load(file)

with open('/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/log/no_training/vls_version_40.json', 'r') as file:
    data_40 = json.load(file)

with open('/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/log/no_training/vls_version_50.json', 'r') as file:
    data_50 = json.load(file)

with open('/home/suryamurthy/UT_Autonomous_Group/vls_mod_alt/log/no_training/vls_version_100.json', 'r') as file:
    data_100 = json.load(file)



