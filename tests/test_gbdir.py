import os
import sys
import time

import numpy as np

sys.path.insert(0, '..')
from gbpipe import gbdir 

#modules for tests

def test_rotation():
    v_arr=[]

    rmat = gbdir.Rot_matrix(EL=90, coord='G')
    np.random.seed(0)
    for i in range(70000):
        v_arr.append([0,0,1])
        v_arr.append(np.array(np.random.random(3)))

    v_arr = np.array(v_arr)
    k = gbdir.dir_sky(v_arr)
    print(k[0])
    print(k.shape)

    return k


def test_rotation_loop():
    v_arr=[]

    rmat = gbdir.Rot_matrix(EL=90, coord='G')
    np.random.seed(0)
    for i in range(70000):
        v_arr.append([0,0,1])
        v_arr.append(np.array(np.random.random(3)))

    v_arr = np.array(v_arr)
    print((len(v_arr)))
    l=[]
    for v in v_arr:
        l.append(gbdir.dir_sky(v))

    l=np.array(l)
    print (l[0])
    print (l.shape)

    return l


def test_speed():
    import timeit
    print('time for pythonic:', timeit.timeit(test_rotation, number=1))
    print('itme for looping :', timeit.timeit(test_rotation_loop, number=1))

    k = test_rotation()
    l = test_rotation_loop()

    print(np.average(k - l))


def test_rotation_loop2():
    v_arr = []
    rmat = []
    np.random.seed(0)
    start_time = time.time()
    for i in range(50000):
        tmp = gbdir.Rot_matrix(EL=i, coord='C')
        rmat.append(tmp)
    print('elapsed time for rotation_matrix = ', time.time()-start_time)

    for j in range(1000):
        v_arr.append([0,0,1])
        #v_arr.append(np.array(np.random.random(3)))

    rmat = np.array(rmat)
    v_arr = np.array(v_arr)

    start_time = time.time()
    k = gbdir.Rotate(v_arr, rmat=rmat)
    print('elapsed time for matrix rotation = ', time.time()-start_time)

    print(k.shape)


def test_speed2(): # testing speed when the rotation matrix varies at each step
    import timeit
    print('time for pythonic:', timeit.timeit(test_rotation_loop2, number=1))
#    print('time elapsed:', timeit.timeit(test_rotation2, number=1))


def test_rotation_matrices():
    print("single test")
    print(gbdir.Rot_matrix(EL=90, AZ=150, coord='H'))
    print(gbdir.Rot_matrix_healpix(EL=90, AZ=150, coord='H'))

    Ntest = 10
    print("test with list of lenth %d" % Ntest)
    ELs = np.array([i for i in range(Ntest)])
    AZs = np.array([i for i in range(Ntest)])

    print("my routine")
    st = time.time()
    k = gbdir.Rot_matrix(EL=ELs, AZ=AZs, coord='H')
    print("elapsed time = %f" % (time.time() - st))

    print("with healpy routine")
    st = time.time()
    l = []

    for EL, AZ in zip(ELs, AZs):
        l.append(gbdir.Rot_matrix_healpix(EL=EL, AZ=AZ, coord='H'))

    print("elapsed time = %f" % (time.time() - st))

    l = np.array(l)
    for i in range(len(l)):
        if (np.average(k[i]-l[i]) != 0):
            print (i)
            print (k[i]-l[i])


def test_parangle():
    N = 1000000
    a = np.array([np.random.random(N), np.random.random(N), np.random.random(N)]).T
    st=time.time()
    psi = gbdir.parallactic_angle(a)
    print('elapsed time for parallactic angle for %d samples = %f' % (N, time.time()-st))
    
    psi1 = gbdir.parallactic_angle(a[0])
    print(psi[0])
    print(psi1)


if __name__=='__main__':
    test_parangle() 

