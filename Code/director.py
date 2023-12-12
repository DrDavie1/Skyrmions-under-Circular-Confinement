#------------------------------------------------------

#Class to define a director field. 

#------------------------------------------------------


import numpy as np

class Director():
    
    #initialize field based on initial conditions

    def __init__(self,L_x: float,L_y: float,a_x: float,a_y: float,radius: float) -> None:

      self.N_x = int(L_x/a_x)
      self.N_y = int(L_y/a_x)

      self.a_x = a_x
      self.a_y = a_y

      self.radius = radius

      self.x_correction = (self.N_x-1) / 2
      self.y_correction = (self.N_y-1) / 2

      self.n = np.zeros([self.N_x,self.N_y,3])

    # defining n: 

      for i in range(self.N_x):
        for j in range(self.N_y):
           
          current_x = (i - self.x_correction)*a_x
          current_y = (j - self.y_correction)*a_y

          if np.sqrt((current_x)**2 + (current_y)**2) <= self.radius:
              
            new_ni,new_nj,new_nk = n_init(current_x,current_y,self.radius,alpha,twist)

            self.n[i][j][0] = new_ni
            self.n[i][j][1] = new_nj
            self.n[i][j][2] = new_nk

            #if outside skyrmion boundry

          else:
              
            self.n[i][j][0] = 0
            self.n[i][j][1] = 0
            self.n[i][j][2] = -1

    #method to calculate grad n, curl n and laplace n 

    def calculate_derivatives(self) -> np.ndarray:

      #Energy minimisation and cacluation:

      self.n_grad = np.zeros([self.N_x,self.N_y,3])

      self.n_curl = np.zeros([self.N_x,self.N_y,3])

      self.n_laplace = np.zeros([self.N_x,self.N_y,3])

      # Q calculation:

      self.dn_dx = np.zeros([self.N_x,self.N_y,3])

      self.dn_dy = np.zeros([self.N_x,self.N_y,3])

      for i in range(self.N_x):
        for j in range(self.N_y):

          current_x = (i - self.x_correction)*self.a_x
          current_y = (j - self.y_correction)*self.a_y

            # Ignore outside boundry:

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
    
    def calculate_energy(self,K: float,B: float) -> np.ndarray:

      # Energy functional caclulation found using:

      # E = sum  ( K/2 * mod_grad_n_squared + B (n . n_curl) )

      self.mod_grad_squared = np.zeros([self.N_x,self.N_y])

      self.n_dot_curl = np.zeros([self.N_x,self.N_y])

      for i in range(self.N_x):
         for j in range(self.N_y):
            
            self.mod_grad_squared[i][j] = (self.n_grad[i][j][0])**2 + (self.n_grad[i][j][1])**2 + (self.n_grad[i][j][2])**2

            self.n_dot_curl[i][j] = np.dot(self.n[i][j],self.n_curl[i][j])

      E_total = K/2 * self.mod_grad_squared + B * self.n_dot_curl

      self.E = np.sum(E_total)

      return self.E

    # topological charge calculation
    
    def calculate_Q(self) -> np.ndarray:

      self.q_sum = np.zeros([self.N_x,self.N_y])

      for i in range(self.N_x):
         for j in range(self.N_y):
            
            cross_prod = np.cross( self.dn_dx[i][j] , self.dn_dy[i][j] )
            
            self.q_sum[i][j] = np.dot(self.n[i][j],cross_prod)

      self.Q = 1/(4*np.pi) * np.sum(self.q_sum)

      return self.Q
    

# Example n_init function:

alpha = 0
twist = 1

def n_init(x: float,y: float,R: float,alpha: float,twist: float) -> tuple:
    
    r = np.sqrt(x**2 + y**2) 
   
    phi = np.arctan2(y,x)

    n_i = np.sin( twist*np.pi * r  / R ) * np.cos(phi+alpha)
    n_j = np.sin( twist*np.pi * r  / R ) * np.sin(phi+alpha)
    n_k = np.cos( twist*np.pi * r  / R )

    return n_i,n_j,n_k
