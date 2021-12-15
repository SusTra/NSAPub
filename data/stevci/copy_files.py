"""
Vzame števce gledena datoteko counter_ids in jih prekopira
v mapo, od kjer berejo ipynb 04_promet_stevci_preprocess
"""


from os import listdir
from shutil import copyfile

s = []

f = open("counter_ids.txt", encoding="utf8")
for x in f:
    x=x.strip()
    s.extend(x.split(";"))
f.close()
print(s)

dirs = listdir()
for d in dirs:
    if d.startswith("OBDELANI_PODATKI"):
        files = listdir(d)
        for f in files:
            for x in s:               
                if x and ((f.startswith(x) or f.startswith("STEP_MOL_"+x)) and f.endswith('U')):
                    copyfile(d+"\\"+f, f)

