import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from numpy.linalg import inv    
from scipy.linalg import schur, sqrtm
import numpy as np


def invSqrt(a,b,c):
    eps = 1e-12 
    mask = (b !=  0)
    r1 = mask * (c - a) / (2. * b + eps)
    t1 = np.sign(r1) / (np.abs(r1) + np.sqrt(1. + r1*r1));
    r = 1.0 / np.sqrt( 1. + t1*t1)
    t = t1*r;
    
    r = r * mask + 1.0 * (1.0 - mask);
    t = t * mask;
    
    x = 1. / np.sqrt( r*r*a - 2*r*t*b + t*t*c)
    z = 1. / np.sqrt( t*t*a + 2*r*t*b + r*r*c)
    
    d = np.sqrt( x * z)
    
    x = x / d
    z = z / d
       
    new_a = r*r*x + t*t*z
    new_b = -r*t*x + t*r*z
    new_c = t*t*x + r*r *z

    return new_a, new_b, new_c

def Ell2LAF(ell):
    A23 = np.zeros((2,3))
    A23[0,2] = ell[0]
    A23[1,2] = ell[1]
    a = ell[2]
    b = ell[3]
    c = ell[4]
    sc = np.sqrt(np.sqrt(a*c - b*b))
    ia,ib,ic = invSqrt(a,b,c) 
    A = np.array([[ia, ib], [ib, ic]]) / sc
    sc = np.sqrt(A[0,0] * A[1,1] - A[1,0] * A[0,1])
    A23[0:2,0:2] = rectifyAffineTransformationUpIsUp(A / sc) * sc
    return A23

def rectifyAffineTransformationUpIsUp(A):
    det = np.sqrt(np.abs(A[0,0]*A[1,1] - A[1,0]*A[0,1] + 1e-10))
    b2a2 = np.sqrt(A[0,1] * A[0,1] + A[0,0] * A[0,0])
    A_new = np.zeros((2,2))
    A_new[0,0] = b2a2 / det
    A_new[0,1] = 0
    A_new[1,0] = (A[1,1]*A[0,1]+A[1,0]*A[0,0])/(b2a2*det)
    A_new[1,1] = det / b2a2
    return A_new

def ells2LAFs(ells):
    LAFs = np.zeros((len(ells), 2,3))
    for i in range(len(ells)):
        LAFs[i,:,:] = Ell2LAF(ells[i,:])
    return LAFs

def LAF2pts(LAF, n_pts = 50):
    a = np.linspace(0, 2*np.pi, n_pts);
    x = [0]
    x.extend(list(np.sin(a)))
    x = np.array(x).reshape(1,-1)
    y = [0]
    y.extend(list(np.cos(a)))
    y = np.array(y).reshape(1,-1)
    HLAF = np.concatenate([LAF, np.array([0,0,1]).reshape(1,3)])
    H_pts =np.concatenate([x,y,np.ones(x.shape)])
    H_pts_out = np.transpose(np.matmul(HLAF, H_pts))
    H_pts_out[:,0] = H_pts_out[:,0] / H_pts_out[:, 2]
    H_pts_out[:,1] = H_pts_out[:,1] / H_pts_out[:, 2]
    return H_pts_out[:,0:2]

def convertLAFs_to_A23format(LAFs):
    sh = LAFs.shape
    if (len(sh) == 3) and (sh[1]  == 2) and (sh[2] == 3): # n x 2 x 3 classical [A, (x;y)] matrix
        work_LAFs = deepcopy(LAFs)
    elif (len(sh) == 2) and (sh[1]  == 7): #flat format, x y scale a11 a12 a21 a22
        work_LAFs = np.zeros((sh[0], 2,3))
        work_LAFs[:,0,2] = LAFs[:,0]
        work_LAFs[:,1,2] = LAFs[:,1]
        work_LAFs[:,0,0] = LAFs[:,2] * LAFs[:,3] 
        work_LAFs[:,0,1] = LAFs[:,2] * LAFs[:,4]
        work_LAFs[:,1,0] = LAFs[:,2] * LAFs[:,5]
        work_LAFs[:,1,1] = LAFs[:,2] * LAFs[:,6]
    elif (len(sh) == 2) and (sh[1]  == 6): #flat format, x y s*a11 s*a12 s*a21 s*a22
        work_LAFs = np.zeros((sh[0], 2,3))
        work_LAFs[:,0,2] = LAFs[:,0]
        work_LAFs[:,1,2] = LAFs[:,1]
        work_LAFs[:,0,0] = LAFs[:,2] 
        work_LAFs[:,0,1] = LAFs[:,3]
        work_LAFs[:,1,0] = LAFs[:,4]
        work_LAFs[:,1,1] = LAFs[:,5]
    else:
        print 'Unknown LAF format'
        return None
    return work_LAFs

def LAFs2ell(in_LAFs):
    LAFs = convertLAFs_to_A23format(in_LAFs)
    ellipses = np.zeros((len(LAFs),5))
    for i in range(len(LAFs)):
        LAF = deepcopy(LAFs[i,:,:])
        scale = np.sqrt(LAF[0,0]*LAF[1,1]  - LAF[0,1]*LAF[1, 0] + 1e-10)
        u, W, v = np.linalg.svd(LAF[0:2,0:2] / scale, full_matrices=True)
        W[0] = 1. / (W[0]*W[0]*scale*scale)
        W[1] = 1. / (W[1]*W[1]*scale*scale)
        A =  np.matmul(np.matmul(u, np.diag(W)), u.transpose())
        ellipses[i,0] = LAF[0,2]
        ellipses[i,1] = LAF[1,2]
        ellipses[i,2] = A[0,0]
        ellipses[i,3] = A[0,1]
        ellipses[i,4] = A[1,1]
    return ellipses

def visualize_LAFs(img, LAFs):
    work_LAFs = convertLAFs_to_A23format(LAFs)
    plt.figure()
    plt.imshow(255 - img)
    for i in range(len(work_LAFs)):
        ell = LAF2pts(work_LAFs[i,:,:])
        plt.plot( ell[:,0], ell[:,1], 'r')
    plt.show()
    return 

def readMODS_ExtractFeaturesFile(fname):
    mrSize = 3.0 * np.sqrt(3.0)
    features_dict = {}
    with open(fname, 'rb') as f:
        lines = f.readlines()
        det_num = int(lines[0])
        current_pos = 1
        for det_idx in range(det_num):
            dd = lines[current_pos]
            dd = dd.strip().split(' ')
            det_name = dd[0]
            desc_num = int(dd[1])
            features_dict[det_name] = {}
            current_pos +=1
            print det_name, desc_num
            for desc_idx in range(desc_num):
                dd2 = lines[current_pos]
                dd2 = dd2.strip().split(' ')
                desc_name = dd2[0]
                features_num = int(dd2[1])
                print desc_name, features_num
                current_pos+=1
                desc_len =  int(lines[current_pos])
                print desc_len
                LAFs = np.zeros((features_num, 7))
                if desc_len > 0:
                    descriptors = np.zeros((features_num, desc_len))
                else:
                    descriptors = None
                for feat_idx in range(features_num):
                    current_pos+=1
                    l = lines[current_pos].strip().split(' ')
                    LAFs[feat_idx,0:2] = np.array(l[4:6]) 
                    LAFs[feat_idx,2] = mrSize * np.array(float(l[12])) 
                    LAFs[feat_idx,3:] = np.array(l[6:10]) 
                    if desc_len > 0:
                        descriptors[feat_idx,:] = np.array(l[26:])
                features_dict[det_name][desc_name] = (LAFs, descriptors)
                current_pos+=1
    return features_dict