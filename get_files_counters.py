import os

def get_ids(folder = ""):
    s = {}
    if folder:
        l = os.listdir(folder)
    else:
        l = os.listdir()


    for file_name in l:
        if file_name.endswith('U'):
            file_id = file_name.split("_")[2]
            
            f = open(folder + file_name, encoding="cp1250")

       
            name = f.readlines()[2].split("STM")[1].split(": ")[1].split("  /  ")[0]
            name = " ".join(name.split(" ")[1:])
            s[name] = file_id
            f.close()  
    """
    for file_name in l:
        if file_name.endswith('xlsx'):
            file_id = file_name.split(" ")[0]
            name = " ".join(file_name.split(" ")[1:]).split("-")[0].strip()
            if file_id not in s.values():
                s[name] = file_id  
    """        
    return s
    
def get_file_names(my_name, s, folder=""):
    names = set()
    
    my_id = s[my_name]
    if folder:
        l = [x for x in os.listdir(folder) if x.startswith('STEP') and x.endswith('U')]
    else:
        l = [x for x in os.listdir() if x.startswith('STEP') and x.endswith('U')]
    
    for file_name in l:
        if file_name.startswith("STEP_MOL_"+my_id) and file_name.endswith("U"):
            names.add(file_name)
    
    return names
        