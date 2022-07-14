# reference: https://github.com/braintimeException/vi-with-normalizing-flows


import numpy as np
import torch 
from torch import nn


class Transformation:
    def __init__(self):
        self.training = None
        self.logdet = None

    def num_params(self):
        return 0

    def forward(self, z, params):
        if self.training:
            self.logdet = np.log(self.det(z, params))
        return self.f(z, params)


class PlanarTransfromation(Transformation):

    """
    Special Transformations of form f(z) = z + u * h(w^T * z + b)

        D: int, dimension of latent variable z
        parameters: {w: numpy array \in R^D, u: numpy array \in R^D, b: numpy array \in R^D}
    """

    def __init__(self, D:int, training:bool=True):
        super().__init__()
        self.D = D
        self.training = training 
        self.h = nn.Tanh()
    
    #num_params = |w| + |u| + |b|
    def num_params(self):
        return 2 * self.D + 1

    #f(z) = z + u * h(w^T * z + b)
    def f(self, z, params):
        [w, u, b] = params
        return z + self.u @ self.h(self.w.reshape(1, self.D) @ z + self.b)

    #tanh'(x) = 1 - tanh^2(x)
    def h_prime(self, x):
        return 1- np.power(self.h(x),2)
    
    #psi(z) = h'(w^T * z + b) * w
    def psi(self, z, params):
        [w, _, b] = params
        return self.h_prime(w.reshape(1, self.D) @ z + b) @ w
    
    #|det df/dz| = |det(I + u * psi(z)^T| = |1 + u^T * psi(z)|
    def det(self, z, params):
        u = params[1]
        return (1 + u.reshape(1, self.D) @ self.psi(z, params))


class RadialTransfromation(Transformation):

    """
    Special Transformations of form f(z) = z + beta * h(alpha , r) * (z - z_0)

        D: int, D of input z
        parameters: {z_0: numpy array \in R^D, alpha: int \in R^+, beta: int \in R}
    """

    def __init__(self, D:int, training:bool=True):
        super().__init__()
        self.D = D
        self.training = training

    #num_params = |z_0| + |alpha| + |beta|
    def num_params(self):
        return self.D + 2

    #f(z) = z + beta * h(alpha , r) * (z - z_0)
    def f(self, z, params):
        [z_0 , alpha, beta] = params
        r = np.norm(z - z_0)
        return z + beta * self.h(alpha, r) * (z - z_0)

    #h = 1/(alpha + r) \in R
    def h(self, alpha, r):      
        return 1/(alpha + r)
       
    #h'(x) = (1/(alpha+|z-z_0|))' = - 1/(alpha+|z-z_0|)^2 = - h(x)^2
    def h_prime(self, alpha, r):
        return -np.power(self.h(alpha, r),2)
    
    #|det df/dz| = [1 + βh(α, r)]^(D − 1) * [1 + βh(α, r) + βh'(α, r)r)]
    def det(self, z, params):
        [z_0 , alpha, beta] = params
        r = np.norm(z - z_0)
        return np.power((1+beta * self.h(alpha, r)), (self.D-1)) * (1+ beta*self.h(alpha, r)+ beta*self.h_prime(alpha, r)*r)

class NormalizingFlow():

    """
        Normalizing flow with flow length K and latent variable of dimension D. 

        Complexity: O(KD)
    """

    def __init__(self, D: int, K: int, transformation):
        self.D = D
        self.K = K 
        self.flow = [transformation(self.D) for _ in range(K)]
    
    def forward(self, params):
        for i, trans in enumerate(self.flow):
            z = trans.foward(params[i])
        return z

class VAE(nn.Module):
    def __init__(self, transformation, latent_size, flow_length):
        super().__init__()


    


    





