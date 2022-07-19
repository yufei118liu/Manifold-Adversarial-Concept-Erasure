# reference: https://github.com/braintimeException/vi-with-normalizing-flows


import numpy as np
import torch 
from torch import nn
from torch.nn import functional as F


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
        self.num_params = self.flow[0].num_params()
    
    def forward(self, params):
        for i, trans in enumerate(self.flow):
            z = trans.foward(params[i])
        return z

    def sum_logdet(self):
        return sum([trans.logdet for trans in self.flow])

class VAE(nn.Module):
    def __init__(self, input_dim, layer_size, transformation, latent_size, flow_length):
        super().__init__()

        #encoder
        self.fc1 = nn.Linear(input_dim, 400)
        # encode mean
        self.fc21_mean = nn.Linear(400, layer_size)
        # encode variance
        self.fc22_var = nn.Linear(400, layer_size)

        #decoder
        self.fc3 = nn.Linear(layer_size, 400)
        self.fc4 = nn.Linear(400, input_dim)

        # normalizing flow
        self.flow = NormalizingFlow(transformation, latent_size, flow_length)

        # encode flow parameters ( parallel to mean and var )
        self.fc23_flow = nn.Linear(400, self.flow.nParams * flow_length)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        # returns mean, logvar and flow params
        return (self.fc21_mean(h), self.fc22_var(h),
                self.fc23_flow(h).mean(dim=0).chunk(self.flow.K, dim=0))

    #generate latent variable z = mu + std * eps
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like()
        return torch.add(mu, eps, alpha=std)

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        """Forward pass
        Transforms the input into latent variables and reconstructs the input

        Args
        x  -- input tensor

        Returns recontructed input along with mean and variance of the latent variables
        """
        mu, logvar, params = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        z = self.flow.forward(z, params)
        return self.decode(z), mu, logvar

