

def con():
    import os
    rtdir,_='./',''
    rtdir,_=os.path.split(os.path.realpath(__file__))
    from ipyparallel import Client
    c=Client(
        url_file=None
        #url_file=rtdir+'/../files/.ipython/profile_default/security/ipcontroller-client.json'
#             ,sshserver='core@init'
#             ,sshkey='c:/Users/Majid/.vagrant.d/insecure_private_key'
             #,paramiko=False
             #,debug=True
    )
    return c

def connect():
    import time
    #try a few times to connect
    for atry in range(5): 
        try:
            c=con();
            while True: #wait indef for an engine
                if len(c.ids)>0: break
                else:            time.sleep(5); c=con()
        except:
            time.sleep(.5)

    global lbv
    lbv=c.load_balanced_view()
    return c


#init()
#@lbv.remote(block=True)
def main(ts_id,job_id,params):

    import os
    # this works b/c the engines were started
    # from the directory with the analysis stuff in it
    basedir=os.path.realpath(os.curdir)
    os.chdir(os.path.join(basedir,'experiments',ts_id))

    try:
        import o; 
        ret=o.main(job_id,params)
    except:
        raise
    finally:
        os.chdir(basedir)

    return ret


# if __name__=='__main__':

#     c=connect()
    
#     import sys
#     import pickle
#     import sys
#     params=pickle.load(open(sys.argv[2])) #fn is job id
#     print params
#     ret=lbv.apply_sync(main,sys.argv[1]
#                        ,sys.argv[2],params)
#     print ret
    #ret=lbv.apply_sync(engine.ct)

    # of=open(sys.argv[2]+'.o','w')
    # try:
    #     ret=lbv.apply_sync(main,argv[1],arg[2],params)
    #     #ret=main(argv[1],arg[2],params)
    #     import pickle
    #     pickle.dump(ret,of)
    # except:
    #     of.write('error')
    #     exit(0)
    # finally:
    #     of.close()

    # exit(1)

#cant write my remote funcs here
#@lbv.remote(block=True)
#def myf(): return 3 #import math ; return math.e #np.array([2,33,2.3])



