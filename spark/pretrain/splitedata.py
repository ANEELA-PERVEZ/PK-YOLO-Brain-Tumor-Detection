
import os,glob,shutil
import numpy as np
rootpath = "/media/dzy/deep2/detr/archive"
cls = ['yes', 'no']
for c in cls:
    allfi = glob.glob(os.path.join(rootpath,"train",c,"*.jpg"))
    np.random.shuffle(allfi)
    targets = os.path.join(rootpath,"val",c)
    if not os.path.exists(targets):
        os.makedirs(targets)
    k = 0
    for file in allfi:
        if k < len(allfi) *0.15:
            target = file.replace("train","val")
            shutil.move(file,target)
        k+=1
