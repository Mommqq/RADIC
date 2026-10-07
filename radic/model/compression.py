# import torch
# import torch.nn as nn
# from compressai.models import CompressionModel
# from compressai.entropy_models import GaussianConditional
# from compressai.ops import quantize_ste
# from compressai.ans import BufferedRansEncoder, RansDecoder
# from utils1.func import get_scale_table
# from model.compression_modules import *
# import math
# from compressai.ops import LowerBound
# from compressai.models import CompressionModel


# import torch.nn as nn 
# import torch 
# import numpy as np

# from typing import Any, Callable, List, Optional, Tuple, Union 

# import scipy.stats
# from torch import Tensor
# from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
# from compressai.ops import LowerBound
# import torch.nn.functional as F
# # from compress.quantization.activation import SumOfTanh  , ActualQuantizer,  NonLinearStanh, DeltaQuantized
# import torchac
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import sys 
# import copy
# import math
# import time 
# import numpy as np
# import pandas as pd
# from typing import Any

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from torch import Tensor

# from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf

# class EncoderPredictor(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(channels, channels // 4, 3, 1, 1),
#             nn.SiLU(),
#             nn.Conv2d(channels // 4, 1, 3, 1, 1),
#             nn.Sigmoid() 
#         )

#     def forward(self, y):
#         return self.net(y)

# class _EntropyCoder:
#     """Proxy class to an actual entropy coder class."""

#     def __init__(self, method):
#         if not isinstance(method, str):
#             raise ValueError(f'Invalid method type "{type(method)}"')

#         from compressai import available_entropy_coders

#         if method not in available_entropy_coders():
#             methods = ", ".join(available_entropy_coders())
#             raise ValueError(
#                 f'Unknown entropy coder "{method}"' f" (available: {methods})"
#             )

#         if method == "ans":
#             from compressai import ans

#             encoder = ans.RansEncoder()
#             decoder = ans.RansDecoder()
#         elif method == "rangecoder":
#             import range_coder

#             encoder = range_coder.RangeEncoder()
#             decoder = range_coder.RangeDecoder()

#         self.name = method
#         self._encoder = encoder
#         self._decoder = decoder

#     def encode_with_indexes(self, *args, **kwargs):
#         return self._encoder.encode_with_indexes(*args, **kwargs)

#     def decode_with_indexes(self, *args, **kwargs):
#         return self._decoder.decode_with_indexes(*args, **kwargs)


# def default_entropy_coder():
#     from compressai import get_entropy_coder

#     return get_entropy_coder()


# def pmf_to_quantized_cdf(pmf: Tensor, precision: int = 16) -> Tensor:
#     cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
#     cdf = torch.IntTensor(cdf)
#     return cdf


# def _forward(self, *args: Any) -> Any:
#     raise NotImplementedError()
    
        
# class SumOfTanh(nn.Module):
#     def __init__(self, beta,  num_sigmoids, extrema = 5, symmetry =True):
#         super(SumOfTanh, self).__init__()
#         #print("SOMMA NON LINEARE!!!!")

#         self.num_sigmoids = int(num_sigmoids)
#         self.beta = beta
#         self.symmetry = symmetry

   
#         self.minimo = - extrema 
#         self.massimo = extrema
            
        
#         self.range_num = torch.arange(0.5 ,self.massimo ).type(torch.FloatTensor)
#         if self.num_sigmoids > 0:
#             self.jump = len(self.range_num)/self.num_sigmoids
#             self.levels = num_sigmoids + 1
        
#         else:
#             self.levels = extrema*2 + 1 





        
#         if self.num_sigmoids == 0:
#             if self.symmetry:
#                 self.b = torch.nn.Parameter(self.range_num.type(torch.FloatTensor))   + torch.relu( torch.randn(len(self.range_num)))
#             else:
#                 self.b =  torch.nn.Parameter(torch.arange(self.minimo + 0.5 ,self.massimo ).type(torch.FloatTensor))  #+ torch.relu( torch.randn(torch.arange(self.minimo + 0.5 ,self.massimo ).type(torch.FloatTensor).shape))

#             self.w = torch.nn.Parameter(torch.ones(len(self.range_num)) )  # + torch.relu( torch.randn(len(self.range_num)))  #torch.relu( torch.randn(len(self.range_num))) +  torch.relu( torch.randn(len(self.range_num))) +   torch.relu( torch.randn(len(self.range_num)))  +  torch.relu( torch.randn(len(self.range_num))) + torch.relu( torch.randn(len(self.range_num)))
#         else:
#                 #self.b = torch.nn.Parameter(torch.FloatTensor(num_sigmoids).normal_().sort()[0]) # punti a caso
            
#             if self.simmetry:
#                 c = len(self.range_num)/self.num_sigmoids
#                 self.b = torch.nn.Parameter(torch.arange( self.jump/2   ,self.massimo + self.jump/2 , c))
#             else:
#                 tmp = torch.arange(self.minimo + 0.5 ,self.massimo ).type(torch.FloatTensor)
#                 c = len(tmp)/self.num_sigmoids
#                 self.b = torch.nn.Parameter(torch.arange(self.minimo + self.jump/2   ,self.massimo + self.jump/2 , c)) 
            
#             self.w = torch.nn.Parameter(torch.zeros(self.num_sigmoids) + self.jump  ) 

    

    
#         self.tr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         print("count number of parameters for the quantizer: ",self.tr_parameters )


#         self.length = len(self.range_num) if self.num_sigmoids ==0 else self.num_sigmoids 
#         print("lunghezza---> ",self.b)

#         self.map_sos_cdf = {}
#         self.map_cdf_sos = {}

#         self.update_state()


#     def update_weights(self):
#         self.sym_w =  torch.cat((torch.flip(self.w,[0]),self.w),0)
#         if self.symmetry:
#             self.sym_b = torch.cat((torch.flip(-self.b,[0]),self.b),0) 
#         else:
#             self.sym_b = self.b

#     def update_state(self, device = torch.device("cuda")):
#         self.update_weights()
#         self.update_cumulative_weights( )
#         self.cum_w = self.cum_w.to(device)
#         self.calculate_average_points( ) #self.average_points
#         self.average_points = self.average_points.to(device)
#         self.calculate_distance_points() #self.distance_points
#         self.distance_points = self.distance_points.to(device)

        



#     def update_cumulative_weights(self):
#         self.cum_w = torch.zeros(self.length + 1)
#         self.cum_w[1:] = torch.cumsum(self.w,dim = 0)  
#         self.cum_w = torch.cat((-torch.flip(self.cum_w[1:], dims = [0]),self.cum_w),dim = 0)






#     def reinitialize_weights_and_bias(self):
#         if self.num_sigmoids == 0:
#             self.w = torch.nn.Parameter(torch.ones(len(self.range_num)) )
#             self.b = torch.nn.Parameter(self.range_num.type(torch.FloatTensor))
#         else:
#             self.w = torch.nn.Parameter(torch.zeros(self.num_sigmoids) + self.jump  ) 
#             c = len(self.range_num)/self.num_sigmoids
#             self.b = torch.nn.Parameter(torch.arange(self.minimo + self.jump/2   ,self.massimo + self.jump/2 , c))
        


#     def define_channels_map(self ):

#         mapping = torch.arange(0, int(self.cum_w.shape[0]), 1).numpy()
#         map_float_to_int = dict(zip(list(self.cum_w.detach().cpu().numpy()),list(mapping)))
#         map_int_to_float = dict(zip(list(mapping),list(self.cum_w.detach().cpu().numpy())))            
#         self.map_sos_cdf = map_float_to_int
#         self.map_cdf_sos = map_int_to_float
#         #print("----------------------------------------------------------------------------------------------------")
#         #print("sos cdf: ", self.map_sos_cdf)
#         #print("----------------------------------------------------------------------------------------------------")      
#         #print("cdf sos: ", self.map_cdf_sos)       
    
#     def calculate_average_points(self):
#         self.average_points =  torch.add(self.cum_w[1:], self.cum_w[:-1])/2


#     def calculate_distance_points(self):
#         self.distance_points =   torch.sub(self.cum_w[1:], self.cum_w[:-1])/2
       

#     def f(self,x):
#         return 2*torch.sigmoid(2*x) - 1

#     def forward(self, x, beta=None):
#         b = torch.sort(self.sym_b)[0]
#         if beta is not None:
#             if beta == -1:
#                 return torch.sum((self.sym_w[:,None].to(x.device)/2)*(torch.sign(x - b[:,None].to(x.device))),dim = 1).unsqueeze(1).to(x.device)               
#                 #return torch.stack([w[i]*(torch.relu(torch.sign(x-b[i]))) - w[i]/2 for i in range(self.length)], dim=0).sum(dim=0) 
#             else:
#                 return torch.sum((self.sym_w[:,None].to(x.device)/2)*self.f(beta*(x - b[:,None].to(x.device))),dim = 1).unsqueeze(1).to(x.device)
#                 #return torch.stack([(self.w[i]/2)*self.f(beta*(x-b[i])) for i in range(self.length)], dim=0).sum(dim=0) 
#         else:
#             return torch.sum((self.sym_w[:,None]/2)* self.f(self.beta*(x - b[:,None]))  ,dim = 1).unsqueeze(1).to(x.device)
        


# class NonLinearStanh(nn.Module):
#     def __init__(self, beta,
#                 num_sigmoids, 
#                 extrema = 5,
#                 custom_w = None,
#                 custom_b = None,
#                 trainable =True):
#         super(NonLinearStanh, self).__init__()
#         #print("non-linear-sum")
#         self.num_sigmoids = int(num_sigmoids)
#         self.beta = beta
#         self.extrema = extrema     
#         self.minimo = - extrema 
#         self.massimo = extrema
            
        
#         self.range_num = torch.arange(self.minimo  + 0.5 ,self.massimo ).type(torch.FloatTensor)
#         if self.num_sigmoids > 0:
#             self.jump = len(self.range_num)/self.num_sigmoids
#             self.levels = num_sigmoids + 1
        
#         else:
#             self.levels = extrema*2 + 1 


#         # bias 
#         if self.num_sigmoids == 0:
#             if custom_b is None:
#                 self.b = torch.nn.Parameter(self.range_num.type(torch.FloatTensor), requires_grad= trainable)
#             else:
#                 self.b = torch.nn.Parameter(custom_b, requires_grad= trainable)

#         else:
               
#             c = len(self.range_num)/self.num_sigmoids
#             self.b = torch.nn.Parameter(torch.arange(self.minimo + self.jump/2   ,self.massimo + self.jump/2 , c))

#         # w
#         if self.num_sigmoids == 0:
#             if custom_w is None:
#                 self.w = torch.nn.Parameter(torch.ones(len(self.range_num)), requires_grad= trainable )
#             else:
#                 self.w = torch.nn.Parameter(custom_w,requires_grad= trainable)
#         else:
#             self.w = torch.nn.Parameter(torch.zeros(self.num_sigmoids) + self.jump  )      


    
#         self.tr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)

#         #print("trainable parameters for the quantizer: ",self.tr_parameters)


#         self.length = len(self.range_num) if self.num_sigmoids ==0 else self.num_sigmoids 
#         #print("lunghezza---> ",self.length)

#         self.map_sos_cdf = {}
#         self.map_cdf_sos = {}




#         n = (torch.sum(self.w)/2).item()
#         self.cum_w = torch.zeros(self.length + 1)
#         self.cum_w[1:] = torch.cumsum(self.w,dim = 0)  
#         self.cum_w = torch.sub(self.cum_w,n)
#         #print("CUMULATIVE WEIGHTS ARE: ",self.cum_w)


#         self.calculate_average_points()
#         self.calculate_distance_points()

#         self.update_state()

        


#     def update_state(self, device = torch.device("cuda")):
#         self.update_cumulative_weights(device = device )
#         self.calculate_average_points( ) #self.average_points
#         self.average_points = self.average_points.to(device)
#         self.calculate_distance_points() #self.distance_points
#         self.distance_points = self.distance_points.to(device)
#         self.define_channels_map()



#     def calculate_average_points(self):
#         self.average_points = torch.add(self.cum_w[1:], self.cum_w[:-1])/2
        

#     def calculate_distance_points(self):
#         self.distance_points = torch.sub(self.cum_w[1:], self.cum_w[:-1])/2
        


#     def update_cumulative_weights(self,device = torch.device("cuda")):
#         #if self.num_sigmoids == 0:
#         n = (torch.sum(self.w)/2).item()
#         self.cum_w = torch.zeros(self.length + 1).to(self.w.device)
#         self.cum_w[0] = 0.0
#         self.cum_w[1:] = torch.cumsum(self.w,dim = 0)
#         self.cum_w = torch.sub(self.cum_w,n) # -  self.extrema 
#         self.cum_w = self.cum_w.to(device)


#     def reinitialize_weights_and_bias(self):
#         if self.num_sigmoids == 0:
#             self.w = torch.nn.Parameter(torch.ones(len(self.range_num)) )
#             self.b = torch.nn.Parameter(self.range_num.type(torch.FloatTensor))
#         else:
#             self.w = torch.nn.Parameter(torch.zeros(self.num_sigmoids) + self.jump  ) 
#             c = len(self.range_num)/self.num_sigmoids
#             self.b = torch.nn.Parameter(torch.arange(self.minimo + self.jump/2   ,self.massimo + self.jump/2 , c))

#     def define_channels_map(self):
#         mapping = torch.arange(0, int(self.cum_w.shape[0]), 1).numpy()
#         map_float_to_int = dict(zip(list(self.cum_w.detach().cpu().numpy()),list(mapping)))
#         map_int_to_float = dict(zip(list(mapping),list(self.cum_w.detach().cpu().numpy())))            
#         self.map_sos_cdf = map_float_to_int
#         self.map_cdf_sos = map_int_to_float

#         #print("maps: ",self.map_sos_cdf)

#         #self.mapping_decoding = pd.DataFrame(list(self.map_cdf_sos.items()), columns=['key', 'value'])
#         #self.mapping_decoding.set_index('key', inplace=True)
#         #print("***************************** mapping fatto")
    
    


#     def f(self,x):
#         return 2*torch.sigmoid(2*x) - 1

#     def forward(self, x, beta=None):
#         #if self.trainable_bias:
#         b = torch.sort(self.b)[0] # non serve ?
#         #else:
#         #    b = self.b   
         
#         if beta is not None:
#             if beta == -1:
#                 return torch.sum(self.w[:,None]*torch.relu(torch.sign(x - b[:,None])) - self.w[:,None]/2,dim = 1).unsqueeze(1)             
#                 #return torch.stack([self.w[i]*(torch.relu(torch.sign(x-b[i]))) - self.w[i]/2 for i in range(self.length)], dim=0).sum(dim=0) 
#             else:
#                 return torch.sum((self.w[:,None]/2)*self.f(beta*(x - b[:,None])),dim = 1).unsqueeze(1)
#                 #return torch.stack([(self.w[i]/2)*self.f(beta*(x-b[i])) for i in range(self.length)], dim=0).sum(dim=0) 
#         else:
#             return torch.sum( (self.w[:,None]/2)* self.f(self.beta*(x - b[:,None]))  ,dim = 1).unsqueeze(1)
#             #return torch.stack([(self.w[i]/2)*self.f(self.beta*(x-b[i])) for i in range(self.length)], dim=0).sum(dim=0)






# class DeltaQuantized(nn.Module):
#     def __init__(self, beta, extrema =30, device = torch.device("cuda")):
#         super(DeltaQuantized, self).__init__()




#         self.beta = beta
#         self.device = device
#         self.extrema = extrema     
#         self.minimo = - extrema 
#         self.massimo = extrema + 1
        

#         self.delta = torch.nn.Parameter(torch.tensor(1.0))
        






#         #self.b = self.calculate_average_points()
#         self.update_state(device = self.device) #self.b , self.sym_w


    



#         self.map_sos_cdf = {}
#         self.map_cdf_sos = {}

#         self.tr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         #print("count number of parameters for the quantizer: ",self.tr_parameters )

    
#     def update_state(self, device = torch.device("cuda")):

#         self.cum_w = torch.arange(self.minimo ,self.massimo, self.delta.data.item() ).type(torch.FloatTensor).to(device)
#         self.range_num = torch.arange(self.delta.data.item() ,self.massimo + self.delta.data.item(), self.delta.data.item() ).type(torch.FloatTensor).to(device)
#         self.cum_w = torch.cat((torch.flip(-self.range_num,[0]),torch.Tensor([0]).to(device),self.range_num),0).to(device)
#         self.length = len(self.cum_w) - 1
#         self.w = torch.full((self.length,),self.delta.data.item()).to(device)
#         self.calculate_average_points() 
#         self.average_points.to(device)
#         self.calculate_distance_points()
#         self.distance_points.to(device)
#         self.b = self.average_points.to(device)


#     def calculate_distance_points(self):
#         self.distance_points =   torch.sub(self.cum_w[1:], self.cum_w[:-1])/2



#     def define_channels_map(self, ):
#         mapping = torch.arange(0, int(self.cum_w.shape[0]), 1).numpy()
#         map_float_to_int = dict(zip(list(self.cum_w.detach().cpu().numpy()),list(mapping)))
#         map_int_to_float = dict(zip(list(mapping),list(self.cum_w.detach().cpu().numpy())))            

#         self.map_sos_cdf = map_float_to_int
#         self.map_cdf_sos = map_int_to_float

   

    
    
#     def calculate_average_points(self):
#         self.average_points = torch.add(self.cum_w[1:], self.cum_w[:-1])/2
#         #return  self.average_points
#         #return res



#     def f(self,x):
#         return 2*torch.sigmoid(2*x) - 1
    


#     def forward(self, x,  beta=None):
#         self.update_state()
#         b = self.b
#         if beta is not None:
#             if beta == -1:
#                 #return torch.sum((self.w[:,None].to(x.device)/2)*(torch.sign(x - self.b[:,None].to(x.device))),dim = 1).unsqueeze(1).to(x.device)               
#                 return torch.stack([self.delta*(torch.relu(torch.sign(x-b[i]))) - self.delta/2 for i in range(self.length)], dim=0).sum(dim=0) 
#             else:
#                 #return torch.sum((self.w[:,None].to(x.device)/2)*self.f(beta*(x - self.b[:,None].to(x.device))),dim = 1).unsqueeze(1).to(x.device)
#                 return torch.stack([(self.delta/2)*self.f(beta*(x-b[i])) for i in range(self.length)], dim=0).sum(dim=0) 
#         else:
#             #return torch.sum((self.w[:,None].to(x.device)/2)* self.f(self.beta*(x - self.b[:,None].to(x.device)))  ,dim = 1).unsqueeze(1).to(x.device)
#             return torch.stack([(self.delta/2)*self.f(self.beta*(x-b[i])) for i in range(self.length)], dim=0).sum(dim=0) 



# class ActualQuantizer(nn.Module):
#     def __init__(self, beta, M,num_sigmoids, extrema = 5):
#         super(ActualQuantizer, self).__init__()


#         #print("entro qua!!!!!!")
#         self.M = M
#         self.num_sigmoids = int(num_sigmoids)
#         self.beta = beta

#         self.extrema = extrema     
#         self.minimo = - extrema 
#         self.massimo = extrema
        
        
#         #self.range_num = torch.arange(self.minimo  + 0.5 ,self.massimo ).type(torch.FloatTensor)
#         self.range_num = torch.arange(0 ,self.massimo ).type(torch.FloatTensor)
#         if self.num_sigmoids > 0:
#             self.jump = len(self.range_num)/self.num_sigmoids
#             self.levels = num_sigmoids + 1
        
#         else:
#             self.levels = extrema*2 + 1 
        
    
#         self.length = len(self.range_num) if self.num_sigmoids ==0 else self.num_sigmoids 
#         if self.num_sigmoids == 0:           
#             self.w = torch.nn.Parameter(torch.ones(len(self.range_num)) ) # + torch.relu( torch.randn(len(self.range_num)))
#         else:
#             self.w = torch.nn.Parameter(torch.zeros(self.num_sigmoids) + self.jump  )      


#         self.cum_w = torch.zeros(self.length + 1)
#         self.cum_w[1:] = torch.cumsum(self.w,dim = 0)  
#         self.cum_w = torch.cat((-torch.flip(self.cum_w[1:], dims = [0]),self.cum_w),dim = 0)
#         #print("CUMULATIVE WEIGHTS ARE: ",self.cum_w)


#         self.calculate_average_points()
#         self.calculate_distance_points()

#         #self.b = self.calculate_average_points()
#         self.update_weights() #self.b , self.sym_w


    
#         self.tr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         #print("count number of parameters for the quantizer: ",self.tr_parameters )



#         self.map_sos_cdf = {}
#         self.map_cdf_sos = {}


    


#     def update_cumulative_weights(self, device = torch.device("cuda")):

#         self.cum_w = torch.zeros(self.length + 1)
#         self.cum_w[1:] = torch.cumsum(self.w,dim = 0)  
#         self.cum_w = torch.cat((-torch.flip(self.cum_w[1:], dims = [0]),self.cum_w),dim = 0)
#         self.cum_w = self.cum_w.to(device)

#     def reinitialize_weights_and_bias(self):
#         if self.num_sigmoids == 0:
#             self.w = torch.nn.Parameter(torch.ones(len(self.range_num)) )
#             self.b = torch.nn.Parameter(self.range_num.type(torch.FloatTensor))
#         else:
#             self.w = torch.nn.Parameter(torch.zeros(self.num_sigmoids) + self.jump  ) 
#             c = len(self.range_num)/self.num_sigmoids
#             self.b = torch.nn.Parameter(torch.arange(self.minimo + self.jump/2   ,self.massimo + self.jump/2 , c))
#         #self.update_cumulative_weights()
        






#     def define_channels_map(self, ):

#         mapping = torch.arange(0, int(self.cum_w.shape[0]), 1).numpy()

#         map_float_to_int = dict(zip(list(self.cum_w.detach().cpu().numpy()),list(mapping)))

#         map_int_to_float = dict(zip(list(mapping),list(self.cum_w.detach().cpu().numpy())))            

#         self.map_sos_cdf = map_float_to_int
#         self.map_cdf_sos = map_int_to_float

   

    
    
#     def calculate_average_points(self):
#         self.average_points = torch.add(self.cum_w[1:], self.cum_w[:-1])/2
#         #return  self.average_points
#         #return res

#     def calculate_distance_points(self):
#         self.distance_points = torch.sub(self.cum_w[1:], self.cum_w[:-1])/2
#         #return  self.ditance_points
       

#     def f(self,x):
#         return 2*torch.sigmoid(2*x) - 1
    

#     def update_state(self, device = torch.device("cuda")):
#         self.update_cumulative_weights(device = device )
#         self.calculate_average_points( ) #self.average_points
#         self.average_points = self.average_points.to(device)
#         self.calculate_distance_points() #self.distance_points
#         self.distance_points = self.distance_points.to(device)
#         self.update_weights(device = device) # self.b , self.sym_w

#     def update_weights(self, device = torch.device("cuda")):
#         self.sym_w =  torch.cat((torch.flip(self.w,[0]),self.w),0).to(device)
#         self.b = self.average_points.to(device)


#     def forward(self, x,  beta=None):
#         #w =  torch.cat((torch.flip(self.w,[0]),self.w),0).to(x.device)
#         #self.b = self.calculate_average_points()
#         #b =self.calculate_average_points().to(x.device) # torch.sort(self.b)[0] # non serve ?
#         #self.update_weights(device = x.device)
#         if beta is not None:
#             if beta == -1:
#                 return torch.sum((self.sym_w[:,None].to(x.device)/2)*(torch.sign(x - self.b[:,None].to(x.device))),dim = 1).unsqueeze(1).to(x.device)               
#                 #return torch.stack([w[i]*(torch.relu(torch.sign(x-b[i]))) - w[i]/2 for i in range(self.length)], dim=0).sum(dim=0) 
#             else:
#                 return torch.sum((self.sym_w[:,None].to(x.device)/2)*self.f(beta*(x - self.b[:,None].to(x.device))),dim = 1).unsqueeze(1).to(x.device)
#                 #return torch.stack([(self.w[i]/2)*self.f(beta*(x-b[i])) for i in range(self.length)], dim=0).sum(dim=0) 
#         else:
#             return torch.sum((self.sym_w[:,None].to(x.device)/2)* self.f(self.beta*(x - self.b[:,None].to(x.device)))  ,dim = 1).unsqueeze(1).to(x.device)
#             #return torch.stack([(self.w[i]/2)*self.f(self.beta*(x-b[i])) for i in range(self.length)], dim=0).sum(dim=0) 

# class HypeEntropyModelSoS(nn.Module):
#     r"""Entropy model base class.
#     Args:
#         likelihood_bound (float): minimum likelihood bound
#         entropy_coder (str, optional): set the entropy coder to use, use default
#             one if None
#         entropy_coder_precision (int): set the entropy coder precision
#     """

#     def __init__(
#         self,
#         likelihood_bound: float = 1e-9,
#         entropy_coder: Optional[str] = None,
#         entropy_coder_precision: int = 16,
#     ):
#         super().__init__()

#         if entropy_coder is None:
#             entropy_coder = default_entropy_coder()
#         self.entropy_coder = _EntropyCoder(entropy_coder)
#         self.entropy_coder_precision = int(entropy_coder_precision)

#         self.use_likelihood_bound = likelihood_bound > 0
#         if self.use_likelihood_bound:
#             self.likelihood_lower_bound = LowerBound(likelihood_bound)

#         # to be filled on update()
#         self.register_buffer("_offset", torch.IntTensor())
#         self.register_buffer("_quantized_cdf", torch.IntTensor())
#         self.register_buffer("_cdf_length", torch.IntTensor())

#     def __getstate__(self):
#         attributes = self.__dict__.copy()
#         attributes["entropy_coder"] = self.entropy_coder.name
#         return attributes

#     def __setstate__(self, state):
#         self.__dict__ = state
#         self.entropy_coder = _EntropyCoder(self.__dict__.pop("entropy_coder"))

#     @property
#     def offset(self):
#         return self._offset

#     @property
#     def quantized_cdf(self):
#         return self._quantized_cdf

#     @property
#     def cdf_length(self):
#         return self._cdf_length

#     # See: https://github.com/python/mypy/issues/8795
#     forward: Callable[..., Any] = _forward

#     def transform_float_to_int(self,x):
#         if x not in self.sos.unique_values:
#             raise ValueError("the actual values ",x," is not present in ",self.sos.cum_w)
#         return int((self.sos.unique_values ==x).nonzero(as_tuple=True)[0].item())
    

#     def transform_int_to_float(self,x):
#         return self.sos.unique_values[x].item()



#     def transform_map(self,x,map_float_to_int):
#         if x in map_float_to_int.keys():
#             return map_float_to_int[x]
#         else:
#             # find the closest key and use this
#             keys = np.asarray(list(map_float_to_int.keys()))
#             keys = torch.from_numpy(keys).to(x.device)
#             i = (torch.abs(keys - x)).argmin()
#             key = keys[i].item()
#             return map_float_to_int[key]





#     def quantize(self, inputs, mode,  means = None, perms = None):
#         #print("si parte da qua: ",inputs.shape)
#         if perms is not None:
#             inputs =  inputs.permute(*perms[0]).contiguous() # flatten y and call it values
#             shape = inputs.size() 
#             inputs = inputs.reshape(1, 1, -1) # reshape values
#             if means is not None:
#                 means = means.permute(*perms[0]).contiguous()
#                 means = means.reshape(1, 1, -1).to(inputs.device)     


#         #print("secodno step da qua: ",inputs.shape)
#         if mode == "training":
#             #if means is not None: # ricordarsi di toglierlo!!!!!!!!!
#             #    inputs -= means
#             outputs = self.sos(inputs)
#             #if means is not None: # ricordarsi di toglierlo!!!!!!!!!
#             #    outputs += means
#             if perms is not None:
#                 outputs =outputs.reshape(shape)
#                 outputs = outputs.permute(*perms[1]).contiguous()
#             return outputs
#         outputs = inputs.clone()


#         if means is not None:
#             outputs -= means

#         #if outputs.shape[0] == 1:
#         outputs = self.sos( outputs, -1)  
#         #else:
#         #outputs = self.sos( outputs.unsqueeze(0).unsqueeze(0), -1)

#         if mode == "dequantize":
#             if means is not None:
#                 outputs += means

#             if perms is not None:
#                 outputs =outputs.reshape(shape)
#                 outputs = outputs.permute(*perms[1]).contiguous()
#             return outputs

#         if perms is not None:
#             outputs =outputs.reshape(shape)
#             outputs = outputs.permute(*perms[1]).contiguous()


#         assert mode == "symbols", mode
#         shape_out = outputs.shape
#         outputs = outputs.ravel()
#         map_float_to_int = self.sos.map_sos_cdf 
        
#         for i in range(outputs.shape[0]):
#             outputs[i] =  self.transform_map(outputs[i], map_float_to_int)

#         outputs = outputs.reshape(shape_out)    
#         return outputs



#     def map_to_level(self, inputs, maps, dequantize = False):
#         shape_out = inputs.shape
#         outputs = inputs.ravel()
#         for i in range(outputs.shape[0]):
#             if dequantize is False:
#                 outputs[i] =  self.transform_map(outputs[i], maps)
#             else: 
#                 outputs[i] =   torch.from_numpy(np.asarray(self.transform_map(outputs[i], maps))).to(outputs.device)
#         outputs = outputs.reshape(shape_out)
#         outputs = outputs.int()   
#         return outputs     

     
#     def dequantize(self, inputs, means = None, dtype = torch.float):
#         """
#         we have to 
#         1 -map again the integer values to the real values for each channel
#         2 - ad the means  
#         """
#         inputs = inputs.to(dtype)
#         map_int_to_float = self.sos.map_cdf_sos
#         shape_inp = inputs.shape
#         inputs = inputs.ravel()
#         for i in range(inputs.shape[0]):
#             c = torch.tensor(map_int_to_float[inputs[i].item()],dtype=torch.float32)
#             inputs[i] = c.item()
#         inputs = inputs.reshape(shape_inp)

#         if means is not None:
#             inputs = inputs.type_as(means)
#             inputs += means
#         outputs = inputs.type(dtype)
#         return outputs





#     def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
#         cdf = torch.zeros(
#             (len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device
#         )
#         for i, p in enumerate(pmf):
#             prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
#             _cdf = pmf_to_quantized_cdf(prob, self.entropy_coder_precision)
#             cdf[i, : _cdf.size(0)] = _cdf
#         return cdf

#     def _check_cdf_size(self):
#         if self._quantized_cdf.numel() == 0:
#             raise ValueError("Uninitialized CDFs. Run update() first")

#         if len(self._quantized_cdf.size()) != 2:
#             raise ValueError(f"Invalid CDF size {self._quantized_cdf.size()}")

#     def _check_offsets_size(self):
#         if self._offset.numel() == 0:
#             raise ValueError("Uninitialized offsets. Run update() first")

#         if len(self._offset.size()) != 1:
#             raise ValueError(f"Invalid offsets size {self._offset.size()}")

#     def _check_cdf_length(self):
#         if self._cdf_length.numel() == 0:
#             raise ValueError("Uninitialized CDF lengths. Run update() first")

#         if len(self._cdf_length.size()) != 1:
#             raise ValueError(f"Invalid offsets size {self._cdf_length.size()}")

    

#     def retrieve_cdf_from_indexes(self, shapes,indexes): 
#         output_cdf = torch.zeros(shapes)
#         output_cdf = output_cdf[:,None] + torch.zeros(self.cdf.shape[1])
#         output_cdf = output_cdf.to("cpu")
#         for i in range(shapes):
#             output_cdf[i,:] = self.cdf[indexes[i].item(),:]  
#         return output_cdf 



    
#     def compress(self, inputs, indexes):


#         symbols = inputs #[1,128,32,48]
#         shape_symbols = symbols.shape


#         symbols = symbols.ravel().to(torch.int16)
#         indexes = indexes.ravel().to(torch.int16)

        
#         symbols = symbols.to("cpu")  

#         output_cdf = torch.zeros_like(symbols)
#         output_cdf = output_cdf[:,None] + torch.zeros(self.cdf.shape[1])
#         output_cdf = output_cdf.to("cpu")
#         for i in range(symbols.shape[0]):
#             output_cdf[i,:] = self.cdf[indexes[i].item(),:]
#         byte_stream = torchac.encode_float_cdf(output_cdf, symbols, check_input_bounds=True)

#         #c = torchac.decode_float_cdf(output_cdf, byte_stream)
#         #if torchac.decode_float_cdf(output_cdf, byte_stream).equal(symbols) is False:
#         #    raise ValueError("L'output Gaussiano codificato è diverso, qualcosa non va!")
#         #else:
#         #    print("l'immagine è ok!")
#         return byte_stream, output_cdf, shape_symbols 
    

#     def compress_new(self, inputs, indexes):

#         symbols = inputs #self.quantize(inputs, "symbols", means)




#         strings = []
#         for i in range(symbols.size(0)):
#             rv = self.entropy_coder.encode_with_indexes(
#                 symbols[i].reshape(-1).int().tolist(),
#                 indexes[i].reshape(-1).int().tolist(),
#                 self._quantized_cdf.tolist(),
#                 self._cdf_length.reshape(-1).int().tolist(),
#                 self._offset.reshape(-1).int().tolist(),
#             )
#             strings.append(rv)


#         return strings 





#     def decompress(self, byte_stream, output_cdf):
#         output = torchac.decode_float_cdf(output_cdf, byte_stream)#.type(torch.FloatTensor)
#         print(output.shape,"decomp")
#         output = output.to("cuda")
#         output = self.dequantize(output)
#         return output
   
    
    



    
  


# class GaussianConditionalSoS(HypeEntropyModelSoS):
#     r"""Gaussian conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
#     S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
#     hyperprior" <https://arxiv.org/abs/1802.01436>`_.
#     This is a re-implementation of the Gaussian conditional layer in
#     *tensorflow/compression*. See the `tensorflow documentation
#     <https://tensorflow.github.io/compression/docs/api_docs/python/tfc/GaussianConditional.html>`__
#     for more information.
#     """

#     def __init__(
#         self,
#         scale_table: Optional[Union[List, Tuple]],
#         *args: Any,
#         channels: int = 128, 
#         num_sigmoids: int = 1,
#         activation = "nonlinearstanh",
#         beta: int = 1,      
#         extrema: int = 10,
#         scale_bound: float = 0.11,
#         tail_mass: float = 1e-9,
#         trainable = True,
#         device = torch.device("cuda"),
#         **kwargs: Any,
#     ):
#         super().__init__(*args, **kwargs)

#         if not isinstance(scale_table, (type(None), list, tuple)):
#             raise ValueError(f'Invalid type for scale_table "{type(scale_table)}"')

#         if isinstance(scale_table, (list, tuple)) and len(scale_table) < 1:
#             raise ValueError(f'Invalid scale_table length "{len(scale_table)}"')

#         if scale_table and (
#             scale_table != sorted(scale_table) or any(s <= 0 for s in scale_table)
#         ):
#             raise ValueError(f'Invalid scale_table "({scale_table})"')

#         self.tail_mass = float(tail_mass)
#         if scale_bound is None and scale_table:
#             scale_bound = self.scale_table[0]
#         if scale_bound <= 0:
#             raise ValueError("Invalid parameters")
#         self.lower_bound_scale = LowerBound(scale_bound)

#         self.register_buffer(
#             "scale_table",
#             self._prepare_scale_table(scale_table) if scale_table else torch.Tensor(),
#         )

#         self.register_buffer(
#             "scale_bound",
#             torch.Tensor([float(scale_bound)]) if scale_bound is not None else None,
#         )

#         self.channels = int(channels)
#         self.M = int(channels)
#         self.tail_mass = float(tail_mass)
#         self.num_sigmoids = int(num_sigmoids)

#         self.extrema = extrema
#         self.activation = activation



#         if self.activation == "aq":
#             self.sos = ActualQuantizer(beta, self.M,self.num_sigmoids,extrema = self.extrema)
#         elif self.activation == "delta":
#             self.sos = DeltaQuantized(beta,extrema = self.extrema, device = device)
#         elif self.activation == "nonlinearstanh": 
#             self.sos = NonLinearStanh(beta,self.num_sigmoids, extrema = self.extrema, trainable= trainable)
#         elif self.activation == "tanh":
#             print("oppure qua")
#             self.sos = SumOfTanh(beta, self.M,self.num_sigmoids, extrema = self.extrema)
#         else: 
#             raise ValueError(f'insert a valid activation function ')
          
#     @staticmethod
#     def _prepare_scale_table(scale_table):
#         return torch.Tensor(tuple(float(s) for s in scale_table))

#     def _standardized_cumulative(self, inputs):
#         half = float(0.5)
#         const = float(-(2**-0.5))
#         # Using the complementary error function maximizes numerical precision.
#         return half * torch.erfc(const * inputs)

#     @staticmethod
#     def _standardized_quantile(quantile):
#         return scipy.stats.norm.ppf(quantile)


#     def update_scale_table(self, scale_table):
#         # Check if we need to update the gaussian conditional parameters, the
#         # offsets are only computed and stored when the conditonal model is
#         # updated.
#         device = self.scale_table.device
#         self.scale_table = self._prepare_scale_table(scale_table).to(device)
#         #self.update()
#         return True







    
#     def update(self, device = torch.device("cuda")):


#         self.sos.update_state(device)
#         max_length = self.sos.cum_w.shape[0]
            


#         pmf_length = torch.zeros(self.scale_table.shape[0]).int().to(device) + max_length
#         pmf_length = pmf_length.unsqueeze(1)

#         self.sos.define_channels_map()


#         average_points = self.sos.average_points # punti-medi per ogni livello di quantizzazione 
#         distance_points = self.sos.distance_points

#         samples = self.sos.cum_w
#         samples = samples.repeat(self.scale_table.shape[0],1)
#         samples = samples.to(device)

#         self._offset = -self.sos.cum_w[0]


#         low,up = self.define_v0_and_v1(samples, average_points, distance_points)
#         low = low.to(samples.device)
#         up = up.to(samples.device)


#         samples_scale = self.scale_table.unsqueeze(1)  #[64,1]
#         samples = samples.float()
#         #samples = torch.abs(samples) # da correggerre 
#         samples_scale = samples_scale.float()
        

#             # adapt to non simmetric quantization steps 
#         upper_pos = self._standardized_cumulative((low - samples) / samples_scale)*(samples >= 0)
#         upper_neg = self._standardized_cumulative((samples + up) / samples_scale)*(samples < 0)
#         lower_pos = self._standardized_cumulative((-up  - samples) / samples_scale)*(samples >= 0)
#         lower_neg = self._standardized_cumulative(( samples - low) / samples_scale)*(samples < 0)
            
#         upper = upper_pos + upper_neg
#         lower = lower_pos + lower_neg
            
#         pmf = upper - lower

#         self.pmf = pmf
#         self.cdf =  self.pmf_to_cdf()

#         # loro 
#         tail_mass = 2 * lower[:, :1]
#         quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
#         quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
#         self._quantized_cdf = quantized_cdf
        
#         self._cdf_length = pmf_length + 2
#         self._cdf_length = self._cdf_length.ravel()




#     def pmf_to_cdf(self):
#         cdf = self.pmf.cumsum(dim=-1)
#         spatial_dimensions = self.pmf.shape[:-1] + (1,)
#         zeros = torch.zeros(spatial_dimensions, dtype=self.pmf.dtype, device=self.pmf.device)
#         cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
#         cdf_with_0 = cdf_with_0.clamp(max=1.)
#         return cdf_with_0
         

    
#     def define_v0_and_v1(self, inputs, average_points, distance_points): 


#         inputs_shape = inputs.shape
#         inputs = inputs.reshape(-1) #.to(inputs.device) # perform reshaping 
#         inputs = inputs.unsqueeze(1)#.to(inputs.device) # add a dimension
       
#         average_points = average_points.to(inputs.device)
#         distance_points = distance_points.to(inputs.device)
       
        
#         average_points_left = torch.zeros(average_points.shape[0] + 1 ).to(inputs.device) - 1000 # 1000 è messo a caso al momento 
#         average_points_left[1:] = average_points
#         average_points_left = average_points_left.unsqueeze(0).to(inputs.device)
        

#         average_points_right = torch.zeros(average_points.shape[0] + 1 ).to(inputs.device) + 1000 # 1000 è messo a caso al momento 
#         average_points_right[:-1] = average_points
#         average_points_right = average_points_right.unsqueeze(0).to(inputs.device)       
               
               
#         distance_points_left = torch.cat((torch.tensor([0]).to(inputs.device),distance_points),dim = -1).to(inputs.device)
#         distance_points_left = distance_points_left.unsqueeze(0).to(inputs.device)
        
#         distance_points_right = torch.cat((distance_points, torch.tensor([0]).to(inputs.device)),dim = -1).to(inputs.device)
#         distance_points_right = distance_points_right.unsqueeze(0).to(inputs.device)
        
#         li_matrix = inputs > average_points_left # 1 if x in inputs is greater that average point, 0 otherwise. shape [__,15]
#         ri_matrix = inputs <= average_points_right # 1 if x in inputs is smaller or equal that average point, 0 otherwise. shape [__,15]
        
#         li_matrix = li_matrix.to(inputs.device)
#         ri_matrix = ri_matrix.to(inputs.device)

#         one_hot_inputs = torch.logical_and(li_matrix, ri_matrix).to(inputs.device) # tensr that represents onehot encoding of inouts tensor (1 if in the interval, 0 otherwise)
              
#         one_hot_inputs_left = torch.sum(distance_points_left*one_hot_inputs, dim = 1).unsqueeze(1).to(inputs.device) #[1200,1]
#         one_hot_inputs_right = torch.sum(distance_points_right*one_hot_inputs, dim = 1).unsqueeze(1).to(inputs.device) #[1200,1]
        
        
#         v0 = one_hot_inputs_left.reshape(inputs_shape)#.to(inputs.device) #  in ogni punto c'è la distanza con il livello a sinistra       
#         v1 = one_hot_inputs_right.reshape(inputs_shape)#.to(inputs.device) # in ogni punto c'è la distanza con il livello di destra

#         return v0 , v1


#     #add something
#     def _likelihood(self, inputs: Tensor, scales: Tensor, means: Optional[Tensor] = None):


#         average_points = self.sos.average_points.to(inputs.device)
#         distance_points = self.sos.distance_points.to(inputs.device)

#         if means is not None:
#             values = inputs - means
#         else:
#             values = inputs
        
#         #values = torch.abs(values)
#         low,up = self.define_v0_and_v1(values, average_points, distance_points)
#         low = low.to(inputs.device)
#         up = up.to(inputs.device)


#         values = values.to(inputs.device)
#         #values = torch.abs(values).to(inputs.device)


#         scales = self.lower_bound_scale(scales)

#         upper_pos = self._standardized_cumulative((low - values) / scales)*(values >= 0)
#         upper_neg = self._standardized_cumulative((values + up) / scales)*(values < 0)
#         lower_pos = self._standardized_cumulative((-up  - values) / scales)*(values >= 0)
#         lower_neg = self._standardized_cumulative(( values - low) / scales)*(values < 0)
        

#         upper = upper_pos  + upper_neg
#         lower = lower_pos + lower_neg
        
#         #lower = lower_pos + lower_neg
#         likelihood = upper - lower

#         #upper =self._standardized_cumulative((low - values) / scales)
#         #lower = self._standardized_cumulative((-up - values) / scales)

#         #likelihood = upper - lower
#         return likelihood



#     def define_permutation(self, x):
#         perm = np.arange(len(x.shape)) 
#         perm[0], perm[1] = perm[1], perm[0]
#         inv_perm = np.arange(len(x.shape))[np.argsort(perm)] # perm and inv perm
#         return perm, inv_perm   



#     def forward(self, x ,scales , perms, training = True, means = None):



#         values =  x.permute(*perms[0]).contiguous() # flatten y and call it values
#         shape = values.size() 
#         values = values.reshape(1, 1, -1) # reshape values
#         if means is not None:
#             means = means.permute(*perms[0]).contiguous()
#             means = means.reshape(1, 1, -1)#.to(x.device)     

#         y_hat = self.quantize(values, "training" if training else "dequantize", means = means)
        
#         y_hat = y_hat.reshape(shape)
#         y_hat = y_hat.permute(*perms[1]).contiguous()

#         #values = values.reshape(shape)
#         #values = values.permute(*perms[1]).contiguous()

#         if means is not None:
#             means = means.reshape(shape)
#             means = means.permute(*perms[1]).contiguous()


#         likelihood = self._likelihood(y_hat, scales, means = means)#.to(x.device)  nuovo !!
#         #likelihood = self._likelihood(values, scales, means = means).to(x.device)
#         if self.use_likelihood_bound:
#             likelihood = self.likelihood_lower_bound(likelihood)  
#         return y_hat, likelihood 


#     def build_indexes(self, scales: Tensor):
#         """
#         Questa funzione associa ad ogni elemento output scala l'indice corrispondende alla deviazione standard 
#         one-to-one mapping tra scala e indexe
#         Non è ottimale, perché si associa la scala subito più grande da una lista fissata
#         1- la lista fissata mi sembra troppo estesa (serve?)
#         """
#         scales = self.lower_bound_scale(scales)
#         indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
#         for s in self.scale_table[:-1]:
#             indexes -= (scales <= s).int()



#         #print("--------------------------------")
#         #print("il massimo indice relativo a questa immagine è ",torch.max(indexes),"        ", self.scale_table[torch.max(indexes).item()])
#         #print("check se nella lista compaiono valori più grandi del punto medio di scale tables")
#         #if indexes.ravel() >= int(len(self.scale_table) - 1)/2:
#         #    print("le scale in questo caso servono  ")
#         #else:
#         #    print("bastano scale più piccole")
        
#         return indexes



#     def permutation_function(self,x):
#         perm = np.arange(len(x.shape))
#         perm[0], perm[1] = perm[1], perm[0]
#         inv_perm = np.arange(len(x.shape))[np.argsort(perm)]
#         return perm, inv_perm







#     def compress(self, x, indexes, perms = None, means = None ):

#         if perms is not None:
#             values =  x.permute(*perms[0]).contiguous() # flatten y and call it values
#             shape = values.size() 
#             values = values.reshape(1, 1, -1) # reshape values
#             if means is not None:
#                 means =  means.permute(*perms[0]).contiguous() # flatten y and call it values
#                 means = means.reshape(1, 1, -1) # reshape values
#         else:
#             values = x
#         #print("lo shape di values prima è: ",values.shape)
#         x = self.quantize(values, "symbols", means = means)  
#         #print("lo shape di x prima è: ",x.shape)

#         if perms is not None:
#             #print("mannaggia a satana io non devo entrare qua!!!")
#             x = x.reshape(shape)
#             x = x.permute(*perms[1]).contiguous()

#         return super().compress(x, indexes) 

#     def decompress(self, byte_stream, output_cdf, shapes = None, means = None):
#         #outputs = super().decompress(byte_stream, output_cdf)   
#         outputs =   torchac.decode_float_cdf(output_cdf, byte_stream)
#         #print("lo shape è ---> ",outputs.shape,"     ",means.shape)
#         #outputs = outputs.to("cuda")
#         #means = means.to("cuda")
#         if shapes is not None:
#             print("print inputs shape: ",shapes)
#             outputs = outputs.reshape(shapes)
#             means = means.reshape(shapes)
#         #else: 
#             #outputs = outputs.reshape(means.shape)
#         outputs = self.dequantize(outputs, means = means)
#         #outputs = outputs.to("cuda")
#         return outputs



# class EncoderPredictor(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(channels, channels // 4, 3, 1, 1),
#             nn.SiLU(),
#             nn.Conv2d(channels // 4, 1, 3, 1, 1),
#             nn.Sigmoid() 
#         )

#     def forward(self, y):
#         return self.net(y)

# class STanHQuantizer(nn.Module):
#     def __init__(self, channels, num_sigmoids=64, extrema=10, symmetry=True):
#         super().__init__()
#         self.num_sigmoids = int(num_sigmoids)
#         self.symmetry = symmetry
#         self.extrema = extrema
#         self.jump = float(extrema) / num_sigmoids if num_sigmoids > 0 else 1.0

#         if self.symmetry:
#             self.w = nn.Parameter(torch.zeros(self.num_sigmoids) + self.jump)
#             initial_b = torch.arange(self.jump/2, extrema + self.jump/2, self.jump)
#             self.b = nn.Parameter(initial_b[:self.num_sigmoids])
#         else:
#             self.w = nn.Parameter(torch.zeros(self.num_sigmoids * 2) + self.jump)
#             self.b = nn.Parameter(torch.arange(-extrema + 0.5, extrema + 0.5, self.jump))

#         self.register_buffer("cum_w", torch.Tensor())
#         self.register_buffer("average_points", torch.Tensor())
#         self.register_buffer("distance_points", torch.Tensor())

#     def _get_params(self):
#         if self.symmetry:
#             sym_w = torch.cat((torch.flip(self.w, [0]), self.w), 0)
#             sym_b = torch.cat((torch.flip(-self.b, [0]), self.b), 0)
#             return sym_w, sym_b
#         return self.w, self.b

#     def update_state(self, device=None):
#         w, b = self._get_params()
#         n = (torch.sum(w) / 2).item()
#         self.cum_w = torch.sub(torch.cumsum(torch.cat([torch.tensor([0.0]).to(w.device), w]), dim=0), n)
#         self.average_points = (self.cum_w[1:] + self.cum_w[:-1]) / 2
#         self.distance_points = (self.cum_w[1:] - self.cum_w[:-1]) / 2
#         if device:
#             self.to(device)

#     def forward(self, x, beta=1.0, spatial_offset=None):
#         """
#         优化后的显存友好型 forward
#         """
#         w, b = self._get_params()
#         b = torch.sort(b)[0]
        
#         # 初始化输出张量
#         y_out = torch.zeros_like(x)
        
#         # 🟢 核心修复：通过循环 $L$ 次来替代巨大的 5D 广播
#         # 这能极大降低显存峰值，且由于 L 较小 (64-128)，对训练速度影响忽略不计
#         for i in range(len(w)):
#             # 拿到当前阶梯的参数
#             w_i = w[i]
#             b_i = b[i]
            
#             # 应用空间偏移
#             # if spatial_offset is not None:
#             #     curr_b = b_i + spatial_offset # [B, 1, H, W]
#             # else:
#             curr_b = b_i
            
#             if beta == -1:
#                 # 硬量化累加
#                 y_out = y_out + (w_i / 2) * torch.sign(x - curr_b)
#             else:
#                 # 软量化累加
#                 y_out = y_out + (w_i / 2) * torch.tanh(beta * (x - curr_b))

#         # 计算退火误差 Et
#         et = torch.tensor(0.0).to(x.device)
#         if self.training and beta != -1:
#             with torch.no_grad():
#                 # 硬量化用于误差计算
#                 y_hard = torch.zeros_like(x)
#                 for i in range(len(w)):
#                     curr_b = b[i] + (spatial_offset if spatial_offset is not None else 0)
#                     y_hard += (w[i] / 2) * torch.sign(x - curr_b)
            
#             # 计算 MSE 差值
#             e_soft = torch.mean((y_out - x)**2)
#             e_hard = torch.mean((y_hard - x)**2)
#             et = torch.abs(e_hard - e_soft)

#         return y_out, et

# import torch
# import torch.nn as nn
# import math
# import torch.nn.functional as F
# from compressai.models import CompressionModel
# from compressai.entropy_models import EntropyBottleneck  # 🚨 替换 VQ-VAE，使用标准超先验
# from model.compression_modules import * # 引入 ckbd_split, Encoder, Decoder 等

# class Compression(CompressionModel):
#     def __init__(self, in_nc, out_nc, N, M, slice_num, slice_ch, num_stanh=3, gaussian_config=None, **kwargs):
#         super().__init__()
#         self.slice_num = slice_num
#         self.slice_ch = slice_ch
#         self.gain = 1
#         self.M = M

#         self.encoder = Encoder(in_nc, M)
#         self.encoder_y = Encoder(in_nc, M)
#         self.hyper_enc = HyperEncoder(N, M)
#         self.hyper_dec = HyperDecoder(N, M)
#         self.decoder = Decoder(M)
#         self.decoder_y = Decoder(M)
#         self.out = nn.Conv2d(M, out_nc, 3, 1, 1)
#         self.out_y = nn.Conv2d(M, out_nc, 3, 1, 1)

#         self.entropy_bottleneck = EntropyBottleneck(N)

#         self.num_stanh = num_stanh
#         if gaussian_config is None:
#             gaussian_config =[{"beta": 1.0, "num_sigmoids": 20, "activation": "nonlinearstanh", "extrema": 5.0, "trainable": True} for _ in range(num_stanh)]

#         self.gaussian_conditional = nn.ModuleList([
#             GaussianConditionalSoS(
#                 None,
#                 channels=M,
#                 beta=gaussian_config[i]["beta"],
#                 num_sigmoids=gaussian_config[i]["num_sigmoids"],
#                 activation=gaussian_config[i]["activation"],
#                 extrema=gaussian_config[i]["extrema"],
#                 trainable=gaussian_config[i]["trainable"],
#                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             ) for i in range(self.num_stanh)
#         ])

#         self.local_context = nn.ModuleList(nn.Conv2d(slice_ch[i], slice_ch[i] * 2, 5, 1, 2) for i in range(len(slice_ch)))
#         self.channel_context = nn.ModuleList(ChannelContextEX(sum(slice_ch[:i]), slice_ch[i] * 2) if i else None for i in range(slice_num))
#         self.entropy_parameters_anchor = nn.ModuleList(EntropyParametersEX(M * 2 + (slice_ch[i] * 2 if i else 0), slice_ch[i] * 2) for i in range(slice_num))
#         self.entropy_parameters_nonanchor = nn.ModuleList(EntropyParametersEX(M * 2 + slice_ch[i] * (4 if i else 2), slice_ch[i] * 2) for i in range(slice_num))

#         self.proj_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 256))

#         self.sram = nn.Sequential(
#             nn.Conv2d(N * 2 + 1, M, 3, 1, 1),
#             nn.SiLU(),
#             nn.Conv2d(M, M, 3, 1, 1),
#             nn.Softplus() 
#         )
#         self.predictor = EncoderPredictor(M)

#     def get_embedding(self, feat):
#         emb = self.proj_head(feat)
#         return torch.nn.functional.normalize(emb, p=2, dim=1)

#     def side_encode(self, h_i):
#         f_y = self.encoder_y(h_i)
#         f_y = self.decoder_y(f_y)
#         z_y = self.out_y(f_y)
#         return f_y, z_y

#     def get_floor_ceil_decimal(self, num):
#         floor_num = math.floor(num)
#         ceil_num = math.ceil(num)
#         decimal_part = num - floor_num
#         return floor_num, ceil_num, decimal_part

#     def define_gaussian_conditional(self, floor, ceil, decimal):
#         first_sos = self.gaussian_conditional[floor].sos
#         second_sos = self.gaussian_conditional[ceil].sos

#         custom_w = first_sos.w * (1 - decimal) + second_sos.w * decimal
#         custom_b = first_sos.b * (1 - decimal) + second_sos.b * decimal

#         gaussian_cond = self.gaussian_conditional[floor] if decimal <= 0.5 else self.gaussian_conditional[ceil]
#         return gaussian_cond, custom_w, custom_b

#     def get_interpolated_params(self, stanh_level):
#         if stanh_level == int(stanh_level):
#             sos = self.gaussian_conditional[int(stanh_level)].sos
#             return sos.w, sos.b
#         else:
#             floor, ceil, decimal = self.get_floor_ceil_decimal(stanh_level)
#             _, custom_w, custom_b = self.define_gaussian_conditional(floor, ceil, decimal)
#             return custom_w, custom_b

#     def forward(self, x, h_y, beta=1.0, stanh_level=0.0, training=True):
#         y = self.encoder(x)
#         h_y = self.encoder_y(h_y)
#         pred_attn = self.predictor(y) 
#         attenuation = 1.0 - 0.9 * pred_attn 
#         y_scaled = y * self.gain * attenuation
        
#         z = self.hyper_enc(y_scaled) 
#         z_hat, z_likelihoods = self.entropy_bottleneck(z)
#         hyper_params = self.hyper_dec(z_hat)

#         w_v, b_v = self.get_interpolated_params(stanh_level)
#         w_v = torch.clamp(w_v, min=1e-4)
#         w_v = w_v / torch.sum(w_v) * torch.sum(self.gaussian_conditional[0].sos.w)
#         b_v = torch.sort(b_v)[0]
#         w_v = w_v.view(-1, 1, 1, 1, 1).to(x.device)
#         b_v = b_v.view(-1, 1, 1, 1, 1).to(x.device)

#         def apply_stanh(res, is_hard=False):
#             y_out = torch.zeros_like(res)
#             for i in range(len(w_v)):
#                 if is_hard or beta < 0:  
#                     y_out += (w_v[i] / 2) * torch.sign(res - b_v[i])
#                 else:                    
#                     y_out += (w_v[i] / 2) * torch.tanh(beta * (res - b_v[i]))
#             return y_out

#         y_hat_slices, y_likelihoods = [],[]
#         total_et = 0.0
#         slice_start = 0

#         floor_idx = math.floor(stanh_level)
#         stanh_model = self.gaussian_conditional[floor_idx]
#         stanh_model.sos.update_state(x.device)

#         for idx in range(self.slice_num):
#             ch = self.slice_ch[idx]
#             y_slice = y_scaled[:, slice_start : slice_start + ch, ...]
#             slice_start += ch

#             slice_anchor, slice_nonanchor = ckbd_split(y_slice)

#             # Anchor
#             if idx == 0:
#                 params_a = self.entropy_parameters_anchor[idx](hyper_params)
#             else:
#                 channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, 1))
#                 params_a = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], 1))
            
#             mu_a, scale_a = params_a.chunk(2, 1)
#             mu_a, scale_a = ckbd_anchor(mu_a), ckbd_anchor(scale_a)

#             res_a = slice_anchor - mu_a
#             y_hat_a = apply_stanh(res_a) + mu_a
#             y_hat_a = ckbd_anchor(y_hat_a) 

#             # Non-anchor 
#             local_ctx = self.local_context[idx](y_hat_a)
#             if idx == 0:
#                 params_na = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], 1))
#             else:
#                 params_na = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], 1))
            
#             mu_na, scale_na = params_na.chunk(2, 1)
#             mu_na, scale_na = ckbd_nonanchor(mu_na), ckbd_nonanchor(scale_na)

#             res_na = slice_nonanchor - mu_na
#             y_hat_na = apply_stanh(res_na) + mu_na
#             y_hat_na = ckbd_nonanchor(y_hat_na) 

#             # Merge
#             y_hat_slice = y_hat_a + y_hat_na
#             y_hat_slices.append(y_hat_slice)

#             mu_slice = ckbd_merge(mu_a, mu_na)
#             scale_slice = ckbd_merge(scale_a, scale_na)
#             res_slice = y_slice - mu_slice
#             y_q_slice = apply_stanh(res_slice) 

#             if training:
#                 with torch.no_grad():
#                     y_hard = apply_stanh(res_slice, is_hard=True) 
#                 total_et += torch.abs(torch.mean((y_hard)**2) - torch.mean((y_q_slice)**2))

#             scales = stanh_model.lower_bound_scale(scale_slice)
#             # # half_bin = 0.25 * self.gain 
#             # delta = torch.mean(torch.abs(b_v[1:] - b_v[:-1]))
#             # half_bin = delta / 2 
#             # upper = stanh_model._standardized_cumulative((half_bin - y_q_slice) / scales)
#             # lower = stanh_model._standardized_cumulative((-half_bin - y_q_slice) / scales)

#             y_q_quantized = y_q_slice + mu_slice
#             y_lik = stanh_model._likelihood(y_q_quantized, scales=scale_slice, means=mu_slice)
            
#             y_likelihoods.append(y_lik)
#             # y_lik = torch.clamp(upper - lower, min=1e-9)
#             # y_likelihoods.append(y_lik)

#         y_hat = torch.cat(y_hat_slices, 1) / self.gain
#         guide_hint = self.decoder(y_hat)
#         output = self.out(guide_hint)
#         h_y_out = self.decoder_y(h_y)

#         avg_et = total_et / self.slice_num if self.slice_num > 0 else torch.tensor(0.0).to(x.device)

#         return output, [torch.cat(y_likelihoods, 1)],[z_likelihoods], torch.tensor(0.0).to(x.device), guide_hint, h_y_out, avg_et, pred_attn


import torch
import torch.nn as nn
from compressai.models import CompressionModel
from compressai.entropy_models import GaussianConditional
from compressai.ops import quantize_ste
from compressai.ans import BufferedRansEncoder, RansDecoder
from utils1.func import get_scale_table
from model.compression_modules import *

class EncoderPredictor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(channels // 4, 1, 3, 1, 1),
            nn.Sigmoid() 
        )

    def forward(self, y):
        return self.net(y)

class Compression(CompressionModel):
    def __init__(self, in_nc, out_nc, N, M, slice_num, slice_ch, codebook_size):
        super().__init__()

        self.slice_num = slice_num
        self.slice_ch = slice_ch

        self.encoder = Encoder(in_nc, M)
        self.encoder_y = Encoder(in_nc, M)
        self.hyper_enc = HyperEncoder(N, M)
        self.hyper_dec = HyperDecoder(N, M)
        self.decoder = Decoder(M)
        self.decoder_y = Decoder(M)
        self.out = nn.Conv2d(M, out_nc, 3, 1, 1)
        self.out_y = nn.Conv2d(M, out_nc, 3, 1, 1)
        self.proj_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256)  
        )
        self.local_context = nn.ModuleList(
            nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2)
            for i in range(len(slice_ch))
        )

        self.channel_context = nn.ModuleList(
            ChannelContextEX(in_dim=sum(slice_ch[:i]), out_dim=slice_ch[i] * 2) if i else None
            for i in range(slice_num)
        )

        # Use channel_ctx and hyper_params
        self.entropy_parameters_anchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2)
            if i else EntropyParametersEX(in_dim=M * 2, out_dim=slice_ch[i] * 2)
            for i in range(slice_num)
        )

        # Entropy parameters for non-anchors
        # Use spatial_params, channel_ctx and hyper_params
        self.entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2)
            if i else  EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2)
            for i in range(slice_num)
        )

        self.codebook_size = codebook_size
        self.quantize = VectorQuantiser(self.codebook_size, N, contras_loss=True)
        self.gaussian_conditional = GaussianConditional(None)
        self.predictor = EncoderPredictor(M)

    def get_embedding(self, feat):
        emb = self.proj_head(feat)
        return torch.nn.functional.normalize(emb, p=2, dim=1)

    def side_encode(self, h_i):
        f_y = self.encoder_y(h_i)
        f_y = self.decoder_y(f_y)
        z_y = self.out_y(f_y)
        return f_y, z_y

    def forward(self, x, h_y):
        # print("x:",x.shape) # (b,512,32,32)
        y = self.encoder(x)
        h_y = self.encoder_y(h_y)
        z = self.hyper_enc(y)
        z_q, emb_loss, _  = self.quantize(z)
        pred_attn = self.predictor(y) 
        attenuation = 1.0 - 0.7 * pred_attn
        # attenuation_d = attenuation.detach()
        y = y * attenuation
        # Hyper-parameters
        hyper_params = self.hyper_dec(z_q)

        y_slices = [y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...] for i in range(len(self.slice_ch))]
        y_hat_slices = []
        y_likelihoods = []
        q_likelihoods = []
        for idx, y_slice in enumerate(y_slices):
            """
            Split y to anchor and non-anchor
            anchor :
                0 1 0 1 0
                1 0 1 0 1
                0 1 0 1 0
                1 0 1 0 1
                0 1 0 1 0
            non-anchor:
                1 0 1 0 1
                0 1 0 1 0
                1 0 1 0 1
                0 1 0 1 0
                1 0 1 0 1
            """
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor
                
                # Non-anchor
                # local_ctx: [B, H, W, 2 * C]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
            
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                _, q_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice, False)
                # round slice_nonanchor
                slice_nonanchor = quantize_ste(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)
                q_likelihoods.append(q_slice_likelihoods)
            else:
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                # Anchor(Use channel context and hyper params)
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor
                
                # Non-anchor
                # ctx_params: [B, H, W, 2 * C]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                _, q_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice, False)
                # round slice_nonanchor
                slice_nonanchor = quantize_ste(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)
                q_likelihoods.append(q_slice_likelihoods)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihoods, dim=1)
        q_likelihoods = torch.cat(q_likelihoods, dim=1)
        
        guide_hint = self.decoder(y_hat)
        h_y = self.decoder_y(h_y)
        output = self.out(guide_hint)
        output_y = self.out(h_y)
        return output, [y_likelihoods], [q_likelihoods], emb_loss, guide_hint, h_y, attenuation
        # return output, [y_likelihoods], [q_likelihoods], emb_loss, guide_hint, h_y
    # ---------------------------------------------------------------------
    # Compress & Decompress 
    # ---------------------------------------------------------------------
    def compress(self, x):
        y = self.encoder(x)
        z = self.hyper_enc(y)
        z_q, encoding_indices = self.quantize.quant(z)
        
        torch.backends.cudnn.deterministic = True
        z_strings = compress_hyper_latent(encoding_indices, self.codebook_size)
        hyper_params = self.hyper_dec(z_q)

        y_slices = [y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...] for i in range(len(self.slice_ch))]
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # round and compress anchor
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                # Non-anchor
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                y_slice_hat = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_slice_hat)

            else:
                # Anchor
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # round and compress anchor
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                # Non-anchor
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        torch.backends.cudnn.deterministic = False
        return {
            "strings": [y_strings, [z_strings]],
            "shape": z.size()[-2:]
        }
    
    def decompress(self, strings, shape):
        torch.backends.cudnn.deterministic = True

        y_strings = strings[0][0]
        z_strings = strings[1][0]
        encoding_indices = decompress_hyper_latent(z_strings, shape, codebook_size=self.codebook_size)
        z_q = self.quantize.get_codebook_entry(encoding_indices.long())
        
        hyper_params = self.hyper_dec(z_q)

        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        for idx in range(self.slice_num):
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # decompress anchor
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                # Non-anchor
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                y_hat_slice = slice_nonanchor + slice_anchor
                y_hat_slices.append(y_hat_slice)
            else:
                # Anchor
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # decompress anchor
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                # Non-anchor
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                y_hat_slice = slice_nonanchor + slice_anchor
                y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        torch.backends.cudnn.deterministic = False

        guide_hint = self.decoder(y_hat)

        output = self.out(guide_hint)

        return output, guide_hint
    
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated