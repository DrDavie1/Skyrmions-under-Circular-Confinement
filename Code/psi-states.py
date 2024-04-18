
import numpy as np

# ---------------pi twisted skyrmion---------------

def init_psi_pi(x,y,R,alpha=0.055,theta=np.pi):
    
    R1 = R

    w = complex(x,y)

    w_conj = np.conj(w)

    w_mag = np.linalg.norm(w)

    sigma = alpha * complex(np.cos(theta),np.sin(theta))
       
    psi = ((w_mag**2 - R1**2) ) * sigma * 1 / w_conj 

    return psi

#------------3 pi------------


def init_psi_3pi(x,y,R,alpha=0.01,theta=np.pi):
    
    R1 = R
    R2 = R/3
    R3 = 2*R/3

    w = complex(x,y)

    w_conj = np.conj(w)

    w_mag = np.linalg.norm(w)

    sigma = alpha * complex(np.cos(theta),np.sin(theta))
       
    psi = ((w_mag**2 - R1**2) * 1/( w_mag**2 - R3**2 ) * (w_mag**2 - R2**2) ) * sigma * 1 / w_conj 

    return psi



#------------5 pi------------


def init_psi_5pi(x,y,R,alpha=0.02,theta=np.pi):
    
    R1 = R
    R2 = R/5
    R3 = 2*R/5
    R4 = 3*R/5
    R5 = 4*R/5

    w = complex(x,y)

    w_conj = np.conj(w)

    w_mag = np.linalg.norm(w)

    sigma = alpha * complex(np.cos(theta),np.sin(theta))
       
    psi = ((w_mag**2 - R1**2) * 1/( w_mag**2 - R3**2 ) * (w_mag**2 - R2**2) * 1/( w_mag**2 - R5**2 ) * (w_mag**2 - R4**2)) * sigma * 1 / w_conj 

    return psi


#------------7 pi------------

def init_psi_7pi(x,y,R,alpha=0.02,theta=np.pi):
    
    R1 = R 
    R2 = R/7
    R3 = 2*R/7 
    R4 = 3*R/7 
    R5 = 4*R/7 
    R6 = 5*R/7 
    R7 = 6*R/7 

    w = complex(x,y)

    w_conj = np.conj(w)

    w_mag = np.linalg.norm(w)

    sigma = alpha * complex(np.cos(theta),np.sin(theta))
       
    psi = ((w_mag**2 - R1**2) * 1/( w_mag**2 - R3**2 ) * (w_mag**2 - R2**2) * 1/( w_mag**2 - R5**2 ) * (w_mag**2 - R4**2) * (w_mag**2 - R6**2)* 1/( w_mag**2 - R7**2 )) * sigma * 1 / w_conj 

    return psi