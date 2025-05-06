import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('bmh')
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import scipy
import seaborn
from tqdm import tqdm
import yfinance as yf
from scipy.stats import norm
from optimization_functions import *
from functions_multidim import *
# Defines an optimal strategy
# def a(omega,t,a_tilde):
#     a_opt = []
#     # t = 0:
#     a_opt.append(a_tilde[0])
#     # times t >0
#     for s in range(t):
#         omega_input = tf.reshape([omega[i] for i in range(s+1)],(1,s+1))
#         a_input = tf.reshape(tf.concat(a_opt,axis = 0),(1,s+2))
#         a_opt.append(a_tilde[s+1]({"omega": omega_input,"a": a_input})[0])
#     return a_opt[-1]

# Defines an optimal strategy
def a(omega, t, a_tilde):
    a_opt = [a_tilde[0]]
    omega_input = []

    for s in range(t):
        omega_input.append(omega[s])  # Collect values instead of reshaping each time
        a_input = tf.reshape(tf.concat(a_opt, axis=0), (1, s + 2))
        a_opt.append(a_tilde[s + 1]({"omega": tf.reshape(omega_input, (1, s + 1)), "a": a_input})[0])

    return a_opt[-1]

def a_ndim(omega, t, a_tilde,d):
    a_opt = [tf.reshape(a_tilde[0],(d,2))]
    omega_input = []

    for s in range(t):
        omega_input.append(omega[:,s])  # Collect values instead of reshaping each time

        a_input = tf.reshape(tf.concat(a_opt, axis=1), (1,d, s + 2))
        a_opt.append(tf.reshape(a_tilde[s + 1]({"omega": tf.reshape(omega_input, (1,d, s + 1)), "a": a_input})[0],(d,1)))
    #print(a_opt)
    return a_opt[-1]
def estimate_covariance_matrix(time_series: np.ndarray) -> np.ndarray:
    """
    Estimates the covariance matrix of a given time series.

    Parameters:
    time_series (np.ndarray): A 2D array where each row is a time step and each column is a variable.

    Returns:
    np.ndarray: The estimated covariance matrix.
    """
    if not isinstance(time_series, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    
    if time_series.ndim != 2:
        raise ValueError("Input must be a 2D array with shape (time_steps, variables).")

    return np.cov(time_series, rowvar=False)
# # Evaluates the trained neural network
# def evaluate_nn(dataset,a_tilde,a_tilde_robust1,a_tilde_robust2,T):
#   Returns_lagged_test1 = tf.concat([dataset[:,i:(-T+i)] for i in range(T)],axis = 0) #(t,N_Returns)
#   Returns_lagged_test1  = tf.cast(tf.transpose(Returns_lagged_test1),tf.float32)
#   vol = 0.22 # Improve if desired

#   Rewards = []
#   Rewards_quadratic = []
#   Rewards_robust1 = []
#   Rewards_robust_quadratic1 = []
#   Rewards_robust2 = []
#   Rewards_robust_quadratic2 = []
#   Rewards_BS = []
#   Rewards_BS_quadratic = []

#   # Create Omegas from the Data:
#   for j in tqdm(range(Returns_lagged_test1.shape[0])):
#       omega = Returns_lagged_test1[j,:]
#       input_dict = {"a": tf.reshape(list(a(omega[:0],0,a_tilde).numpy())+[a(omega[:i],i,a_tilde).numpy()[0] for i in range(1,T)],(1,T+1)),
#                     "omega": tf.reshape(omega ,(1,T))}
#       input_dict_robust1 = {"a": tf.reshape(list(a(omega[:0],0,a_tilde_robust1).numpy())+[a(omega[:i],i,a_tilde_robust1).numpy()[0] for i in range(1,T)],(1,T+1)),
#                     "omega": tf.reshape(omega ,(1,T))}
#       input_dict_robust2 = {"a": tf.reshape(list(a(omega[:0],0,a_tilde_robust2).numpy())+[a(omega[:i],i,a_tilde_robust2).numpy()[0] for i in range(1,T)],(1,T+1)),
#                     "omega": tf.reshape(omega ,(1,T))}
      
#       S = np.cumprod(omega+1)
#       Cash = blackScholes(r=0, S=1, K=1, T=T/365, sigma =vol, type="c")
#       Delta_0 = delta_calc(r=0, S=1, K=1, T=T/365, sigma =vol, type="c")
#       a_BS = [Cash,Delta_0] + [delta_calc(r=0, S=S[i-1], K=1, T=(T-i)/365, sigma =vol, type="c") for i in range(1,T)]
#       input_dict_BS = {"a": tf.reshape(tf.cast(a_BS,tf.float32),(1,T+1)),
#                     "omega": tf.reshape(omega ,(1,T))}

#       Rewards += [-Psi_non_batch(input_dict)[0].numpy()]
#       Rewards_robust1 += [-Psi_non_batch(input_dict_robust1)[0].numpy()]
#       Rewards_robust2 += [-Psi_non_batch(input_dict_robust2)[0].numpy()]
#       Rewards_BS += [-Psi_non_batch(input_dict_BS)[0].numpy()]
#       Rewards_quadratic += [-Psi_quadratic(input_dict)[0].numpy()]
#       Rewards_robust_quadratic1 += [-Psi_quadratic(input_dict_robust1)[0].numpy()]
#       Rewards_robust_quadratic2 += [-Psi_quadratic(input_dict_robust2)[0].numpy()]
#       Rewards_BS_quadratic += [-Psi_quadratic(input_dict_BS)[0].numpy()]

#   df_Rewards = pd.DataFrame(Rewards)
#   df_Rewards_robust1 = pd.DataFrame(Rewards_robust1)
#   df_Rewards_robust2 = pd.DataFrame(Rewards_robust2)
#   df_Rewards_BS = pd.DataFrame(Rewards_BS)
#   df_Rewards_quadratic = pd.DataFrame(Rewards_quadratic)
#   df_Rewards_robust_quadratic1 = pd.DataFrame(Rewards_robust_quadratic1)
#   df_Rewards_robust_quadratic2 = pd.DataFrame(Rewards_robust_quadratic2)
#   df_Rewards_BS_quadratic = pd.DataFrame(Rewards_BS_quadratic)

#   df_merged = pd.concat([df_Rewards,df_Rewards_robust1,df_Rewards_robust2,df_Rewards_BS,
#                          df_Rewards_quadratic,df_Rewards_robust_quadratic1,df_Rewards_robust_quadratic2,df_Rewards_BS_quadratic],axis = 1)
#   df_merged.columns=["Non-Robust, Prospect Utility ","Robust, eps = 0.001, Prospect Utility ","Robust eps, = 0.01, Prospect Utility ",
#                      "Black-Scholes, Prospect Utility","Non-Robust, Quadratic Error ",
#                      "Robust, eps = 0.001, Quadratic Error ","Robust, eps = 0.01, Quadratic Error ","Black-Scholes, Quadratic Error"]
#   return df_merged, Rewards,Rewards_robust1, Rewards_robust2, Rewards_BS, Rewards_quadratic, Rewards_robust_quadratic1, Rewards_robust_quadratic2,Rewards_BS_quadratic


# # Evaluates the trained neural network
# def evaluate_nn(dataset, a_tilde, a_tilde_robust1, a_tilde_robust2,a_tilde_robust_adaptive, T):
#     Returns_lagged_test1 = tf.cast(tf.transpose(tf.concat([dataset[:, i:(-T + i)] for i in range(T)], axis=0)), tf.float32)
#     vol = 0.22  # Can be optimized further if needed
    
#     N = Returns_lagged_test1.shape[0]
    
#     Rewards = np.zeros(N)
#     Rewards_robust1 = np.zeros(N)
#     Rewards_robust2 = np.zeros(N)
#     Rewards_robust_adaptive = np.zeros(N)
#     Rewards_BS = np.zeros(N)
# #       omega = Returns_lagged_test1[j,:]
# #       input_dict = {"a": tf.reshape(list(a(omega[:0],0,a_tilde).numpy())+[a(omega[:i],i,a_tilde).numpy()[0] for i in range(1,T)],(1,T+1)),
# #                     "omega": tf.reshape(omega ,(1,T))}
# #       input_dict_robust1 = {"a": tf.reshape(list(a(omega[:0],0,a_tilde_robust1).numpy())+[a(omega[:i],i,a_tilde_robust1).numpy()[0] for i in range(1,T)],(1,T+1)),
# #                     "omega": tf.reshape(omega ,(1,T))}
# #       input_dict_robust2 = {"a": tf.reshape(list(a(omega[:0],0,a_tilde_robust2).numpy())+[a(omega[:i],i,a_tilde_robust2).numpy()[0] for i in range(1,T)],(1,T+1)),
# #                     "omega": tf.reshape(omega ,(1,T))}
#     for j in tqdm(range(N)):
#         omega = Returns_lagged_test1[j, :]
        
#         # Precompute a_tilde values
#         a_values = [a(omega[:i], i, a_tilde).numpy()[0] for i in range(1,T)]
#         a_values_robust1 = [a(omega[:i], i, a_tilde_robust1).numpy()[0] for i in range(1,T)]
#         a_values_robust2 = [a(omega[:i], i, a_tilde_robust2).numpy()[0] for i in range(1,T)]
#         a_values_robust_adaptive = [a(omega[:i], i, a_tilde_robust_adaptive).numpy()[0] for i in range(1,T)]
        
#         input_dict = {"a": tf.reshape(list(a(omega[:0],0,a_tilde).numpy()) + a_values, (1, T + 1)), "omega": tf.reshape(omega, (1, T))}
#         input_dict_robust1 = {"a": tf.reshape(list(a(omega[:0],0,a_tilde_robust1).numpy()) + a_values_robust1, (1, T + 1)), "omega": tf.reshape(omega, (1, T))}
#         input_dict_robust2 = {"a": tf.reshape(list(a(omega[:0],0,a_tilde_robust2).numpy()) + a_values_robust2, (1, T + 1)), "omega": tf.reshape(omega, (1, T))}
#         input_dict_robust_adaptive = {"a": tf.reshape(list(a(omega[:0],0,a_tilde_robust_adaptive).numpy()) + a_values_robust_adaptive, (1, T + 1)), "omega": tf.reshape(omega, (1, T))}       
#         # Compute Black-Scholes delta hedging strategy
#         S = np.cumprod(omega + 1)
#         Cash = blackScholes(r=0, S=1, K=1, T=T/365, sigma=vol, type="c")
#         Delta_0 = delta_calc(r=0, S=1, K=1, T=T/365, sigma=vol, type="c")
#         a_BS = [Cash, Delta_0] + [delta_calc(r=0, S=S[i - 1], K=1, T=(T - i) / 365, sigma=vol, type="c") for i in range(1, T)]
#         input_dict_BS = {"a": tf.reshape(tf.cast(a_BS, tf.float32), (1, T + 1)), "omega": tf.reshape(omega, (1, T))}

#         # Compute rewards
#         Rewards[j] = -Psi_non_batch(input_dict)[0].numpy()
#         Rewards_robust1[j] = -Psi_non_batch(input_dict_robust1)[0].numpy()
#         Rewards_robust2[j] = -Psi_non_batch(input_dict_robust2)[0].numpy()
#         Rewards_robust_adaptive[j] = -Psi_non_batch(input_dict_robust_adaptive)[0].numpy()
#         Rewards_BS[j] = -Psi_non_batch(input_dict_BS)[0].numpy()


#     # Convert results to Pandas DataFrame efficiently
#     df_merged = pd.DataFrame(
#         np.column_stack([Rewards, Rewards_robust1, Rewards_robust2, Rewards_robust_adaptive,Rewards_BS]),
#         columns=["Non-Robust", "Robust, eps = 0.0001", "Robust, eps = 0.0005",
#                  "Robust adaptive", "Black-Scholes"]
#     )

#     return df_merged, Rewards, Rewards_robust1, Rewards_robust2, Rewards_robust_adaptive, Rewards_BS


def evaluate_nn(dataset, a_tilde, a_tilde_robust1, a_tilde_robust2,a_tilde_robust_adaptive, T,vol = 0.22):
    Returns_lagged_test1 = tf.cast(tf.transpose(tf.concat([dataset[:, i:(-T + i)] for i in range(T)], axis=0)), tf.float32)

    N = Returns_lagged_test1.shape[0]
    
    Rewards = np.zeros(N)
    Rewards_robust1 = np.zeros(N)
    Rewards_robust2 = np.zeros(N)
    Rewards_robust_adaptive = np.zeros(N)
    Rewards_BS = np.zeros(N)
#       omega = Returns_lagged_test1[j,:]
#       input_dict = {"a": tf.reshape(list(a(omega[:0],0,a_tilde).numpy())+[a(omega[:i],i,a_tilde).numpy()[0] for i in range(1,T)],(1,T+1)),
#                     "omega": tf.reshape(omega ,(1,T))}
#       input_dict_robust1 = {"a": tf.reshape(list(a(omega[:0],0,a_tilde_robust1).numpy())+[a(omega[:i],i,a_tilde_robust1).numpy()[0] for i in range(1,T)],(1,T+1)),
#                     "omega": tf.reshape(omega ,(1,T))}
#       input_dict_robust2 = {"a": tf.reshape(list(a(omega[:0],0,a_tilde_robust2).numpy())+[a(omega[:i],i,a_tilde_robust2).numpy()[0] for i in range(1,T)],(1,T+1)),
#                     "omega": tf.reshape(omega ,(1,T))}
    for j in tqdm(range(N)):
        omega = Returns_lagged_test1[j, :]
        
        # Precompute a_tilde values
        # a_values = [a(omega[:i], i, a_tilde).numpy()[0] for i in range(1,T)]
        # a_values_robust1 = [a(omega[:i], i, a_tilde_robust1).numpy()[0] for i in range(1,T)]
        # a_values_robust2 = [a(omega[:i], i, a_tilde_robust2).numpy()[0] for i in range(1,T)]
        # a_values_robust_adaptive = [a(omega[:i], i, a_tilde_robust_adaptive).numpy()[0] for i in range(1,T)]
        
        
        a_values = [a(omega[:i], i, a_tilde) for i in range(1,T)]
        a_values_robust1 = [a(omega[:i], i, a_tilde_robust1) for i in range(1,T)]
        a_values_robust2 = [a(omega[:i], i, a_tilde_robust2) for i in range(1,T)]
        a_values_robust_adaptive = [a(omega[:i], i, a_tilde_robust_adaptive) for i in range(1,T)]
        
        #input_dict = {"a": tf.reshape(tf.concat([a_ndim(omega[:,:0],0,a_tilde,d)] + a_values,1), (1,d, T + 1)), "omega": tf.reshape(omega, (1,d, T))}


        
        # input_dict = {"a": tf.reshape(list(a(omega[:0],0,a_tilde).numpy()) + a_values, (1, T + 1)), "omega": tf.reshape(omega, (1, T))}
        # input_dict_robust1 = {"a": tf.reshape(list(a(omega[:0],0,a_tilde_robust1).numpy()) + a_values_robust1, (1, T + 1)), "omega": tf.reshape(omega, (1, T))}
        # input_dict_robust2 = {"a": tf.reshape(list(a(omega[:0],0,a_tilde_robust2).numpy()) + a_values_robust2, (1, T + 1)), "omega": tf.reshape(omega, (1, T))}
        # input_dict_robust_adaptive = {"a": tf.reshape(list(a(omega[:0],0,a_tilde_robust_adaptive).numpy()) + a_values_robust_adaptive, (1, T + 1)), "omega": tf.reshape(omega, (1, T))}    
        
        
        input_dict = {"a": tf.reshape(tf.concat([a(omega[:0],0,a_tilde)]+a_values,axis = 0), (1, T + 1)), "omega": tf.reshape(omega, (1, T))}
        input_dict_robust1 = {"a": tf.reshape(tf.concat([a(omega[:0],0,a_tilde_robust1)]+a_values_robust1,axis = 0), (1, T + 1)), "omega": tf.reshape(omega, (1, T))}
        input_dict_robust2 = {"a": tf.reshape(tf.concat([a(omega[:0],0,a_tilde_robust2)]+a_values_robust2,axis = 0), (1, T + 1)), "omega": tf.reshape(omega, (1, T))}
        input_dict_robust_adaptive = {"a": tf.reshape(tf.concat([a(omega[:0],0,a_tilde_robust_adaptive)]+a_values_robust_adaptive,axis = 0), (1, T + 1)), "omega": tf.reshape(omega, (1, T))}  
        # Compute Black-Scholes delta hedging strategy
        S = np.cumprod(omega + 1)
        Cash = blackScholes(r=0, S=1, K=1, T=T/365, sigma=vol, type="c")
        Delta_0 = delta_calc(r=0, S=1, K=1, T=T/365, sigma=vol, type="c")
        a_BS = [Cash, Delta_0] + [delta_calc(r=0, S=S[i - 1], K=1, T=(T - i) / 365, sigma=vol, type="c") for i in range(1, T)]
        input_dict_BS = {"a": tf.reshape(tf.cast(a_BS, tf.float32), (1, T + 1)), "omega": tf.reshape(omega, (1, T))}

        # Compute rewards
        Rewards[j] = -Psi_non_batch(input_dict)[0].numpy()
        Rewards_robust1[j] = -Psi_non_batch(input_dict_robust1)[0].numpy()
        Rewards_robust2[j] = -Psi_non_batch(input_dict_robust2)[0].numpy()
        Rewards_robust_adaptive[j] = -Psi_non_batch(input_dict_robust_adaptive)[0].numpy()
        Rewards_BS[j] = -Psi_non_batch(input_dict_BS)[0].numpy()


    # Convert results to Pandas DataFrame efficiently
    df_merged = pd.DataFrame(
        np.column_stack([Rewards, Rewards_robust1, Rewards_robust2, Rewards_robust_adaptive,Rewards_BS]),
        columns=["Non-Robust", "Robust, eps = 0.0001", "Robust, eps = 0.0005",
                 "Robust adaptive", "Black-Scholes"]
    )

    return df_merged, Rewards, Rewards_robust1, Rewards_robust2, Rewards_robust_adaptive, Rewards_BS



def evaluate_nn_ndim(dataset, a_tilde, a_tilde_robust1, a_tilde_robust2,a_tilde_adaptive, d, T,cov = 0):
    dataset = tf.expand_dims(dataset,0) #(1,d,N)
    Returns_lagged_test1 = tf.cast(tf.transpose(tf.concat([dataset[:,:, i:(-T + i)] for i in range(T)], axis=0)), tf.float32)  #(N,d,T)
    vol = 0.22  # Can be optimized further if needed
    
    N = Returns_lagged_test1.shape[0]
    
    Rewards = np.zeros(N)
    Rewards_robust1 = np.zeros(N)
    Rewards_robust2 = np.zeros(N)
    Rewards_adaptive = np.zeros(N)
    Rewards_BS = np.zeros(N)
    
    

    for j in tqdm(range(N)):
        omega = Returns_lagged_test1[j, :,:] 
        
        # Precompute a_tilde values
        a_values = [tf.reshape(a_ndim(omega[:,:i], i, a_tilde,d),(d,1)) for i in range(1,T)]
        a_values_robust1 = [tf.reshape(a_ndim(omega[:,:i], i, a_tilde_robust1,d),(d,1)) for i in range(1,T)]
        a_values_robust2 = [tf.reshape(a_ndim(omega[:,:i], i, a_tilde_robust2,d),(d,1)) for i in range(1,T)]
        a_values_adaptive = [tf.reshape(a_ndim(omega[:,:i], i, a_tilde_adaptive,d),(d,1)) for i in range(1,T)]
        
        
        input_dict = {"a": tf.reshape(tf.concat([a_ndim(omega[:,:0],0,a_tilde,d)] + a_values,1), (1,d, T + 1)), "omega": tf.reshape(omega, (1,d, T))}
        input_dict_robust1 = {"a": tf.reshape(tf.concat([a_ndim(omega[:,:0],0,a_tilde_robust1,d)] + a_values,1), (1,d, T + 1)), "omega": tf.reshape(omega, (1,d, T))}
        input_dict_robust2 = {"a": tf.reshape(tf.concat([a_ndim(omega[:,:0],0,a_tilde_robust2,d)] + a_values,1), (1,d, T + 1)), "omega": tf.reshape(omega, (1,d, T))}
        input_dict_adaptive = {"a": tf.reshape(tf.concat([a_ndim(omega[:,:0],0,a_tilde_adaptive,d)] + a_values,1), (1,d, T + 1)), "omega": tf.reshape(omega, (1,d, T))}
        # Compute Black-Scholes delta hedging strategy
        S = np.cumprod(omega + 1,axis = 1)
        weighted_S = 1
        basket_vol = np.sqrt((1/N**2)*np.sum(cov))
        
        Cash = blackScholes(r=0, S=weighted_S, K=1, T=T/365, sigma=basket_vol, type="c")
        Delta_0 = delta_calc(r=0, S=weighted_S, K=1, T=T/365, sigma=basket_vol, type="c")
        # a_BS = [(1/d)*Cash]*d+ [(1/d)*Delta_0]*d + [(1/d)*delta_calc(r=0, S=np.mean(S[i - 1]), K=1, T=(T - i) / 365, sigma=basket_vol, type="c") for i in range(1, T)]*d
        # input_dict_BS = {"a": tf.reshape(tf.cast(a_BS, tf.float32), (1,d, T + 1)), "omega": tf.reshape(omega, (1, d, T))}
        
        # # Compute rewards
        # Rewards_BS[j] = -Psi_non_batch_ndim(input_dict_BS)[0].numpy()
        Rewards[j] = -Psi_non_batch_ndim(input_dict)[0].numpy()
        Rewards_robust1[j] = -Psi_non_batch_ndim(input_dict_robust1)[0].numpy()
        Rewards_robust2[j] = -Psi_non_batch_ndim(input_dict_robust2)[0].numpy()
        Rewards_adaptive[j] = -Psi_non_batch_ndim(input_dict_adaptive)[0].numpy()

    # Convert results to Pandas DataFrame efficiently
    df_merged = pd.DataFrame(
        np.column_stack([Rewards, Rewards_robust1, Rewards_robust2,Rewards_adaptive]),
        columns=["Non-Robust", "Robust, eps = 0.0001", "Robust, eps = 0.001","Robust, adaptive"]
    )

    return df_merged, Rewards, Rewards_robust1, Rewards_robust2, Rewards_adaptive
    