#---------------------------------------------------
# William Davie - 14/05/2024

# Main supplementory code for BSc Final Project:

#'Skyrmions-under-Circular-Confinement'
#---------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from typing import Callable

# Example simulation parameters - High B low N is a good example to see behaviour

# Lattice size and spacing

N = 60

radius = 5/12 * N

a = 1

# time step, learning rate stopping condition

timestep = 0.1
gamma = 0.1

sigma = 0.1

# effective K and B value

K = 1
B = 1.2

# psi function for  Q = +1 pi twisted skyrmion


def init_psi(x: float, y: float,R: float,alpha: float=0.02, theta: float=np.pi):

    w = complex(x,y)

    w_conj = np.conj(w)

    w_mag = np.linalg.norm(w)

    sigma = alpha * complex(np.cos(theta),np.sin(theta))
       
    psi = ((w_mag**2 - R**2) ) * sigma * 1 / w_conj 

    return psi


# director class 

class director():
    
    # ----------------------init director params----------------------------------

    def __init__(self, N_x: int, N_y: int, a_x: float, a_y: float, R: float) -> None:
      
      self.N_x = N_x
      self.N_y = N_y

      self.a_x = a_x
      self.a_y = a_y

      self.radius = R

      self.x_correction = (N_x-1) / 2
      self.y_correction = (N_x-1) / 2
      
      self.n = np.zeros([N_x,N_y,3])

    # ----------------------init director field based on psi definition ----------

    def init_n(self, psi: Callable, *args):
      
      for i in range(self.N_x):
         
         for j in range(self.N_y):
          
          
            x = (i - self.x_correction)*self.a_x
            y = (j - self.y_correction)*self.a_y

            if np.sqrt((x)**2 + (y)**2) <= self.radius:
              
              _psi = psi(x,y,self.radius)

              # using psi to initalise each component of n

              self.n[i][j][0] = 2*np.real(_psi) / ( 1 + np.linalg.norm(_psi) )
              self.n[i][j][1] = 2*np.imag(_psi) / ( 1 + np.linalg.norm(_psi) )
              self.n[i][j][2] = ( np.linalg.norm(_psi) - 1) / ( 1 + np.linalg.norm(_psi) )

              # ensuring n . n = 1

              self.n[i][j] = self.n[i][j] / np.linalg.norm(self.n[i][j])

            else:

              self.n[i][j][0] = 0
              self.n[i][j][1] = 0
              self.n[i][j][2] = -1


    # ------------Compute derivatives using finite difference methods ----------

    def calculate_derivatives(self):

      #Energy minimisation and cacluation:

      self.n_grad = np.zeros([self.N_x,self.N_y,3])

      self.n_curl = np.zeros([self.N_x,self.N_y,3])

      self.n_laplace = np.zeros([self.N_x,self.N_y,3])

      self.dn_dx = np.zeros([self.N_x,self.N_y,3])

      self.dn_dy = np.zeros([self.N_x,self.N_y,3])

      for i in range(self.N_x):

        for j in range(self.N_y):

          current_x = (i - self.x_correction)*self.a_x
          current_y = (j - self.y_correction)*self.a_y

          if np.sqrt((current_x)**2 + (current_y)**2) <= self.radius:

              #First Derivatives

              # for grad:
              
            dnx_dx = (self.n[i+1][j][0] - self.n[i-1][j][0]) / 2*self.a_x
            dny_dy = (self.n[i][j+1][1] - self.n[i][j-1][1]) / 2*self.a_y

              #for curl:

            dnz_dy = (self.n[i][j+1][2] - self.n[i][j-1][2]) / 2*self.a_y
            dnz_dx = (self.n[i+1][j][2] - self.n[i-1][j][2]) / 2*self.a_x

            dny_dx = (self.n[i+1][j][1] - self.n[i-1][j][1]) / 2*self.a_x
            dnx_dy = (self.n[i][j+1][0] - self.n[i][j-1][0]) / 2*self.a_y


            self.n_grad[i][j] = [dnx_dx,dny_dy,0]

            self.n_curl[i][j] = [dnz_dy,-dnz_dx,(dny_dx - dnx_dy)]

            self.dn_dx[i][j] = [dnx_dx,dny_dx,dnz_dx]

            self.dn_dy[i][j] = [dnx_dy,dny_dy,dnz_dy]


              #Second derivatives 

            dnx_dx_2 = ( self.n[i+1][j][0] - 2*self.n[i][j][0] + self.n[i-1][j][0] ) / self.a_x**2

            dnx_dy_2 = ( self.n[i][j+1][0] - 2*self.n[i][j][0] + self.n[i][j-1][0] ) / self.a_y**2

            dny_dx_2 = ( self.n[i+1][j][1] - 2*self.n[i][j][1] + self.n[i-1][j][1] ) / self.a_x**2

            dny_dy_2 = ( self.n[i][j+1][1] - 2*self.n[i][j][1] + self.n[i][j-1][1] ) / self.a_y**2

            dnz_dx_2 = ( self.n[i+1][j][2] - 2*self.n[i][j][2] + self.n[i-1][j][2] ) / self.a_x**2

            dnz_dy_2 = ( self.n[i][j+1][2] - 2*self.n[i][j][2] + self.n[i][j-1][2] ) / self.a_y**2


            self.n_laplace[i][j] = [ (dnx_dx_2 + dnx_dy_2), (dny_dx_2 + dny_dy_2), (dnz_dx_2 + dnz_dy_2)]

      return self.n
    
    # ------------Calculate effective energy ----------
    
    def calculate_energy(self, K: float, B: float):

      # E = sum  ( K/2 * mod_grad_n_squared + B (n . n_curl) )

      self.mod_grad_squared = np.zeros([self.N_x,self.N_y])

      self.n_dot_curl = np.zeros([self.N_x,self.N_y])

      for i in range(self.N_x):
         
         for j in range(self.N_y):
            
            # sum (delta{}_i n . delta{}_i n)

            self.mod_grad_squared[i][j] = np.dot(self.dn_dx[i][j],self.dn_dx[i][j]) +  np.dot(self.dn_dy[i][j],self.dn_dy[i][j])

            self.n_dot_curl[i][j] = np.dot(self.n[i][j],self.n_curl[i][j])

      self.E_total = K/2 * self.mod_grad_squared + B * self.n_dot_curl

      self.E = np.sum(self.E_total) 

      return self.E
    

    # ------------Calculate Toplogical charge ----------


    def calculate_Q(self):

      self.q_sum = np.zeros([self.N_x,self.N_y])

      for i in range(self.N_x):
         
         for j in range(self.N_y):
            
            cross_prod = np.cross( self.dn_dx[i][j] , self.dn_dy[i][j] )
            
            self.q_sum[i][j] = np.dot(self.n[i][j],cross_prod)

      self.Q = 1/(4*np.pi) * np.sum(self.q_sum)

      return self.Q
  



# ------------run gradient descent----------


def run_simualtion(director_field: director, K: float, B: float, time_step: float, learning_rate: float, tolerance: float, save_director: bool) -> tuple:
   
   #stores values for animation
   
   n_list = []
   E_list = []
   t_list = []

   # initially set delta E to a arbitrary large number.

   deltaE = 1000

   # begin stopping condition loop

   time = 0
   
   while abs(deltaE) > tolerance:
      
      director_field.calculate_derivatives()

      E = director_field.calculate_energy(K,B)

      #Q = director1.calculate_Q() - option
      #print(Q)

      # decaying gamma (optional) :

      #initial_gamma = 5*learning_rate
      #decay_rate = 0.01 #(tau)
      #base_rate = learning_rate
      #learning_rate = base_rate + initial_gamma / (1 + decay_rate * time)

      # gradient descent:

      new_n = director_field.n + timestep*learning_rate*( K*director_field.n_laplace - 2*B*director_field.n_curl )

      # normalisation

      norm_array = np.linalg.norm(new_n, axis=-1, keepdims=True)

      new_n = new_n/norm_array


      # update lists

      n_list.append(new_n)
      E_list.append(E)
      t_list.append(time)

      # update delta E for stopping criteria

      if len(E_list) > 1:
         deltaE = E_list[int(time/timestep)] - E_list[int(time/timestep) - 1]

      # optionally print energy every 10 steps:

      if round(time, 2) % 10 == 0:
         print('Step: ',int(time/time_step))
         print('Energy: ',E)
         print('Change in Energy: ', deltaE)
         
      
      # update director:

      director_field.n = new_n

      # update time

      time += time_step

   # save director if specified.

   if save_director == True:

      np.save('director_file.npy',director_field.n)

   return n_list, E_list, t_list


# ------------animate the director----------

def animate_director(director_list: list, N_x: int, N_y: float) -> None:
    
   # grid points
    x, y = np.meshgrid(np.linspace(-N_x/2, N_x/2, N_x), np.linspace(-N_y/2, N_y/2, N_y))

    # figure setup

    fig = plt.figure(figsize=(7.5,7.5))

    ax = plt.axes()

    ax.set_facecolor('black')

    u = np.zeros(N**2)
    v = np.zeros(N**2)
    w = np.zeros(N**2)

    # define quivers on each grid point

    new_quivers = ax.quiver(x, y, u, v, w,cmap='twilight_shifted')

    # update quivers based on director list

    def update_frame(t):

        for j in range(N):
           for i in range(N):
              u[i+N*j] = director_list[t][i, j, 0]
              v[i+N*j] = director_list[t][i, j, 1]
              w[i+N*j] = director_list[t][i, j, 2]

        new_quivers.set_UVC(u, v, w)

        return new_quivers,

   # animate

    anim = animation.FuncAnimation(fig,func=update_frame,frames=len(director_list),interval=1,blit=True)

    plt.show()


# ------------plot energy profile----------

def plot_energy_profile(time,energy):
   
   fig = plt.figure()

   ax = plt.axes()

   ax.set_ylabel('E(t)')

   ax.set_xlabel('t')

   ax.plot(time,energy/energy[0],c='red')

   plt.show()


# ------------Example Case---------

director_example = director(N,N,a,a,radius)
director_example.init_n(init_psi)

director_list, energy_list, time_list = run_simualtion(director_example, K, B, timestep, gamma, sigma, False)

animate_director(director_list, N, N)
plot_energy_profile(time_list,energy_list)



