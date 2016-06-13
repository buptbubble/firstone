import pickle
import os



def deco_pickle(func):
    pklpath = 'pklfiles'
    curpath = os.getcwd()
    pklpath = os.path.join(curpath,pklpath)
    if not os.path.exists(pklpath):
        os.mkdir(pklpath)


    object,filename = func()
    filepath = os.path.join(pklpath,filename)
    with open(filepath,'wb') as f:
        pickle.dump(object,f)
