import re

with open("simulation/satellite_env.py", "r") as f:
    content = f.read()

# Fix eclipse_mode -> eclipse_fraction in _get_obs_list
content = content.replace("float(self.eclipse_mode[i]),                                   # 2: eclipse_fraction", 
                          "float(self.eclipse_fraction[i]),                               # 2: eclipse_fraction")
content = content.replace("float(self.eclipse_mode[i]),   # 2: eclipse_fraction", 
                          "float(self.eclipse_fraction[i]),   # 2: eclipse_fraction")
content = content.replace("float(self.eclipse_mode[i]),", "float(self.eclipse_fraction[i]),")

# Where is eclipse_mode calculated? Let's find it and add eclipse_fraction calculation
# In satellite_env.py, there is a thermal delta calc.
# Let's search for eclipse_mode[i] assignment:
eclipse_find = r'self\.eclipse_mode\[i\] = (.*?)'
# Wait, let's just insert the fraction logic where eclipse_mode is used.
# Or better, let's write a python block to find where it's assigned.

