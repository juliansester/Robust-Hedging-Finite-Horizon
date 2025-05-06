# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 14:22:18 2025

@author: jul_ses
"""
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import scipy
from tqdm import tqdm
from scipy.stats import norm

import numpy as np
from optimization_functions import *

def brownian_bridge(n):
    """
    Simulate a Brownian Bridge from start to end over time 1 with n partitions.

    Parameters:
    - n: Number of partitions


    Returns:
    - t: Time points
    - B_bridge: Values of the Brownian Bridge at each time point
    """
    dt = 1 / n  # Time step
    t = np.linspace(0, 1, n+1)  # Time points

    # Generate a standard Brownian motion
    dW = np.random.normal(0, np.sqrt(dt), n)
    W = np.concatenate([[0], np.cumsum(dW)])  # Brownian motion path

    # Construct the Brownian Bridge
    B_bridge = W + ( - W[-1]) * t / 1

    return t, B_bridge

def H_alpha(returns,N= 10000,alpha = 0.9):
    integral = []
    for i in range(N):
        #order the returns
        ordered_returns = np.sort(returns)[0]
        #Simulate Brownian Bridge with N+t partitions
        _,bb=brownian_bridge( len(ordered_returns)-2)
        #compute the integral and store the result
        integral.append(np.sum(bb*np.diff(ordered_returns)))
    return np.quantile(integral,alpha)


#            indices = tf.random.uniform(shape=(N_MC,), minval=0, maxval=Returns_train_tf.shape[1],dtype=tf.int32)
#            selected_returns = tf.gather(Returns_train_tf[0,:], indices, axis=0)
#            omega_input= tf.cast(tf.reshape(selected_returns, (N_MC, 1)), tf.float32)

def sample_omega_batch_empirical(omega, t, N_MC, batch_size,Returns_train_tf):
    
    samples_batch = tf.concat([tf.cast(tf.tile(Returns_train_tf,[batch_size,1]), tf.float32),omega],1)
    # Get the number of columns in samples_batch
    num_columns = samples_batch.shape[1]
    # Generate random indices for sampling, for each batch element
    random_indices = tf.random.uniform((samples_batch.shape[0], N_MC), minval=0, maxval=num_columns, dtype=tf.int32)
    # Use tf.gather to sample elements based on the random indices
    sampled_tensor = tf.gather(samples_batch, random_indices, axis=1, batch_dims=1)
    return tf.reshape(sampled_tensor, (batch_size,N_MC, 1) )



#Function that maximizes the expected value w.r.t. a
def max_a_adaptive(Psi_tilde,a_tilde,omega, a,t,optimizer_a,optimizer_lam,lam,Returns_train_tf,N_MC =1024,N_MC_inner = 1024, epsilon=0,T=5,Batch_size=32):
    with tf.GradientTape(persistent=True) as tape: #
        loss_a = -tf.reduce_mean(min_exp_adaptive(Psi_tilde,a_tilde,omega,a,t,lam,Returns_train_tf,N_MC,N_MC_inner,training_a = True,
                                         training_Psi = False,epsilon =epsilon,T=T,Batch_size=Batch_size))
        
    if t>0:
        variables_to_optimize = a_tilde[t].trainable_variables + [lam]
        gradients = tape.gradient(loss_a, variables_to_optimize)
        # Split gradients for respective optimizers (optional, if needed)
        gradient_a = gradients[:len(a_tilde[t].trainable_variables)]
        gradients_lam = gradients[-1:]

        # Apply gradients for both optimizers

        optimizer_lam.apply_gradients([(gradients_lam[0], lam)])
        optimizer_a[t].apply_gradients(zip(gradient_a, a_tilde[t].trainable_variables))
        #optimizer_a[t].apply_gradients(zip(gradients, variables_to_optimize))

   

    if t == 0:
        variables_to_optimize = [a_tilde[t]] + [lam]
        gradients = tape.gradient(loss_a, variables_to_optimize)
        # Split gradients for respective optimizers (optional, if needed)
        gradient_a = gradients[:len([a_tilde[t]] )]
        gradients_lam = gradients[-1:]

        # Apply gradients for both optimizers

        optimizer_lam.apply_gradients([(gradients_lam[0], lam)])
        optimizer_a[t].apply_gradients(zip(gradient_a, [a_tilde[t]]))
        #optimizer_a[t].apply_gradients(zip(gradients, variables_to_optimize))            
             
    del tape
    return -loss_a, a_tilde, optimizer_a, lam, optimizer_lam, gradient_a

# def max_a(Psi_tilde, a_tilde, omega, a, t, optimizer_a, optimizer_lam, lam, Returns_train_tf, N_MC=1024, 
#           N_MC_inner=1024, epsilon=0, T=5, Batch_size=32):
#     with tf.GradientTape(persistent=True) as tape:
#         if t > 0:
#             # Compute a_tilde[t] only once
#             a_tilde_values = a_tilde[t]({"omega": omega, "a": a}, training=True)
#             a_concat = tf.concat([a, a_tilde_values], axis=1)  
#         else:
#             a_concat = a
        
#         loss_a = -tf.reduce_mean(min_exp(Psi_tilde, a_tilde, omega, a_concat, t, lam, Returns_train_tf, 
#                                          N_MC, N_MC_inner, training_a=True, training_Psi=False, epsilon=epsilon, 
#                                          T=T, Batch_size=Batch_size))

#     gradients = tape.gradient(loss_a, a_tilde[t].trainable_variables)
#     optimizer_a[t].apply_gradients(zip(gradients, a_tilde[t].trainable_variables))
#     del tape
#     return -loss_a, a_tilde, optimizer_a

# Function that computes the (Robust) expectation
@tf.function
def min_exp_adaptive(Psi_tilde,a_tilde,omega,a,t,lam,Returns_train_tf,N_MC =1024, N_MC_inner = 1024, 
            training_a = False,training_Psi = False,epsilon=0,T=5,Batch_size=32):  


    if t >0:

        #NEW
        old_omega = tf.expand_dims(omega, axis=1)  # Shape (Batch_size, 1, t)
        old_omega = tf.tile(old_omega, [1, N_MC_inner, 1]) # Shape (Batch_size, N_MC_inner, t)
        z = tf.random.normal(shape=(Batch_size,N_MC_inner,1), mean=0.0, stddev=0.01)  # Shape (Batch_size, N_MC_inner, 1) 
        omega_input = tf.concat([old_omega,z],axis = 2)  # Dimension (Batch_size,N_MC_inner,t+1)

        x = sample_omega_batch_empirical(omega,t,N_MC,Batch_size,Returns_train_tf ) # size = (Batch_size,N_MC,1)    
        x_transpose= tf.transpose(x, perm=[0, 2, 1]) # size = (Batch_size,1,N_MC)   

        x_expanded = tf.repeat(x_transpose, N_MC_inner, axis=1)  # (Batch_size,N_MC_inner,N_MC)   
        z_expanded = tf.repeat(z, N_MC, axis=2)  # (Batch_size,N_MC_inner,N_MC)

        a_concat = tf.concat([a,a_tilde[t]({"omega":omega,"a":a},training = training_a)],axis =1 ) #(Batch_size,t+1)
        a_expanded = tf.expand_dims(a_concat, axis=1) #(Batch_size,1,t+1)            
        a_input = tf.tile(a_expanded, multiples=[1, N_MC_inner, 1])  # Dimension (Batch_size,N_MC_inner,t+1)

        if t == T-1:
            psi_tilde_output = Psi_tilde[t+1]({"omega":omega_input,"a":a_input}) # Dimension (Batch_size,N_MC)
        else:
            psi_tilde_output = Psi_tilde[t+1]({"omega":tf.reshape(omega_input,(Batch_size*N_MC_inner,t+1)),
                                     "a":tf.reshape(a_input,(Batch_size*N_MC_inner,t+2))})
            psi_tilde_output = tf.reshape(psi_tilde_output,(Batch_size,N_MC_inner))  # Dimension (Batch_size,N_MC_inner) 
            
            
        inner_term = (tf.expand_dims(psi_tilde_output, 2) + lam * tf.abs(x_expanded - z_expanded))   # (Batch_size,N_MC_inner,N_MC)

        all_together =  tf.math.reduce_min(inner_term,1)  # Use TensorFlow's tf.exp         # (Batch_size,N_MC)
        outer_mean = tf.reduce_mean(all_together,axis =1)      # (Batch_size)   
        output = outer_mean-lam*epsilon # (Batch_size)



    elif t == 0:
        #NEW
        z = tf.random.normal(shape=(N_MC_inner, 1), mean=0.0, stddev=0.01)
        omega_input = z  # Dimension (N_MC,t+1)

        indices = tf.random.uniform(shape=(N_MC,), minval=0, maxval=Returns_train_tf.shape[1],dtype=tf.int32)
        selected_returns = tf.gather(Returns_train_tf[0,:],indices, axis=0)

        #x = tf.cast(tf.reshape(Returns_train[np.random.choice(Returns_train_tf.shape[1],N_MC)],(N_MC,1)),tf.float32)   
        x = tf.cast(tf.reshape(selected_returns, (N_MC, 1)), tf.float32)
        x_transpose= tf.transpose(x)

        x_expanded = tf.repeat(x_transpose, N_MC_inner, axis=0)  # (N_MC_inner, N_MC)
        z_expanded = tf.repeat(z, N_MC, axis=1)  # (N_MC_inner, N_MC)

        a_tilde_tensor = tf.convert_to_tensor(a_tilde[t], dtype=tf.float32)
        a_repeated = tf.repeat(a_tilde_tensor[tf.newaxis, :], repeats=N_MC_inner, axis=0)
        a_input = tf.reshape(a_repeated, (N_MC_inner, 2))  # Adjust the second dimension as needed


        psi_tilde_output = Psi_tilde[t+1]({"omega": omega_input, "a": a_input})
        inner_term = psi_tilde_output + lam * tf.abs(x_expanded - z_expanded) # (N_MC_inner, N_MC)

        all_together =  tf.math.reduce_min(inner_term,0)  # Use TensorFlow's tf.exp        
        outer_mean = tf.reduce_mean(all_together)        
        output = outer_mean-lam*epsilon

    return output

# Function that minimizes the difference between psi_t and the corresponding expectation
# def approx_psi(Psi_tilde,a_tilde,omega,a,t,lam,optimizer_Psi,Returns_train_tf,N_MC =128,N_MC_inner =128,epsilon=0,T=5,
#                scale_factor =1,Batch_size = 32):
#     with tf.GradientTape() as tape:
#         difference = (tf.squeeze(Psi_tilde[t]({"omega":omega,"a":a},
#                                                 training = True),axis=1)-min_exp(Psi_tilde,a_tilde,
#                                                                            omega,a,t,lam,Returns_train_tf,N_MC,N_MC_inner,epsilon=epsilon,T=T,Batch_size=Batch_size))
#         loss_psi = tf.reduce_mean((difference/scale_factor)**2)
#         # Scaling Factor is introduced to ensure same range of gradients throughout training
#     gradient_psi = tape.gradient(loss_psi,Psi_tilde[t].trainable_variables)
#     optimizer_Psi[t].apply_gradients(zip(gradient_psi, Psi_tilde[t].trainable_variables))
#     return loss_psi, Psi_tilde, optimizer_Psi
    
# Function that minimizes the difference between psi_t and the corresponding expectation
def approx_psi(Psi_tilde, a_tilde, omega, a, t, lam, optimizer_Psi, Returns_train_tf, 
               N_MC=128, N_MC_inner=128, epsilon=0, T=5, scale_factor=1, Batch_size=32):
    with tf.GradientTape() as tape:
        min_exp_output = min_exp_adaptive(Psi_tilde, a_tilde, omega, a, t, lam, Returns_train_tf, 
                                 N_MC, N_MC_inner, epsilon=epsilon, T=T, Batch_size=Batch_size)
        psi_output = tf.squeeze(Psi_tilde[t]({"omega": omega, "a": a}, training=True), axis=1)
        loss_psi = tf.reduce_mean((psi_output - min_exp_output) ** 2) / scale_factor**2

    gradients = tape.gradient(loss_psi, Psi_tilde[t].trainable_variables)
    optimizer_Psi[t].apply_gradients(zip(gradients, Psi_tilde[t].trainable_variables))
    return loss_psi, Psi_tilde, optimizer_Psi


# The training routine
def train_networks_adaptive(Returns_train_tf,
                   T= 5,
                   inner_psi = 5000,
                   inner_a = 5000, 
                   N_MC = 512,
                   N_MC_inner = 512,
                  H_alpha_const = 1,
                  tolerance_psi = 0.00001,
                   tolerance_a = 0.00001,
                   learning_rate_Psi = 0.001,
                  learning_rate_a = 0.001,
                   learning_rate_LAM = 0.001,
                  Batch_size_psi = 32,
                  Batch_size_a = 32,
                  print_intermediate_results = False,
                  print_step_size = 25):
    # Assign models and optimizers
    a_tilde, Psi_tilde, lam, optimizer_a, optimizer_Psi, optimizer_lam = create_models(T,learning_rate_Psi,learning_rate_a,learning_rate_LAM)
    N_returns = Returns_train_tf.shape[1] 
    print("Start Backwards Iterations")
    for t in range(T)[::-1]: # Backwards Iteration
        #for j in range(N_iterations):
        psi_errors = []
        a_values = []
        epsilon = tf.convert_to_tensor(H_alpha_const/np.sqrt(N_returns+t), dtype=tf.float32) 

        # # transfer learning:
        # if t <T-1 and t>0:
        #     for k in range(6,len(a_tilde[t+1].layers)-2):
        #         a_tilde[t].layers[k].set_weights(a_tilde[t+1].layers[k].get_weights())
        #         a_tilde[t].layers[k].trainable = False
        #     k = len(a_tilde[t+1].layers)-1
        #     a_tilde[t].layers[k].set_weights(a_tilde[t+1].layers[k].get_weights())
        #     a_tilde[t].layers[k].trainable = True 

        # # transfer learning:
        # if t <T-1 and t>0:
        #     for k in range(6,len(Psi_tilde[t+1].layers)-2):
        #         Psi_tilde[t].layers[k].set_weights(Psi_tilde[t+1].layers[k].get_weights())
        #         Psi_tilde[t].layers[k].trainable = False
        #     k = len(Psi_tilde[t+1].layers)-1
        #     Psi_tilde[t].layers[k].set_weights(Psi_tilde[t+1].layers[k].get_weights())
        #     Psi_tilde[t].layers[k].trainable = True 
            
        print("#########\n# t = {} #\n#########\n".format(t))

        # Maximization w.r.t. a
        for k in range(inner_a):
            if t>0:
                Batch_size_t = Batch_size_a #int(Batch_size * (1.1 ** min(k, 50)))  # Cap at `50` iterations
                #Batch_size_t = Batch_size

                
                omega_t = sample_states_batch(t, Returns_train_tf, Batch_size_t)
                a_t = sample_actions_batch(t, Batch_size_t)
                
                
            if t== 0:
                a_t = 0.
                omega_t = 0.
              #(1,2*T*t))
            a_value, a_tilde, optimizer_a, lam, optimizer_lam, gradient_a = max_a_adaptive(Psi_tilde,a_tilde,omega_t,a_t,t,
                                                  optimizer_a,optimizer_lam,lam,Returns_train_tf,
                                                   N_MC,N_MC_inner,epsilon,T,Batch_size=Batch_size_t)
            a_values.append(a_value)
            #print(np.mean([np.mean(np.abs(gradient_a[i])) for i in range(len(gradient_a))]))

            if print_intermediate_results:
                if not(k % print_step_size) and k >0:
                    print("a: {}".format(np.mean(a_values[-print_step_size:])))
                    #print("a_gradient: {}".format(np.abs(np.mean(gradient_a_reference[-25:]))))
        # Minimization of Psi
        if t >0:
            for k in range(inner_psi):
                
                
                scale_factor = np.mean(a_values[-25:])
                
                Batch_size_t = Batch_size_psi #int(1+((k+1)/inner_psi)*Batch_size) # Increase Batch Size linearly
                #Batch_size_t = Batch_size

                
                #omega_t = [sample_state(t,Returns_train_tf) for _ in range(Batch_size_t)]
                omega_t = sample_states_batch(t, Returns_train_tf, Batch_size_t)
                a_t = sample_actions_batch(t, Batch_size_t)
                
                
                
                psi_error, Psi_tilde, optimizer_Psi = approx_psi(Psi_tilde,a_tilde,omega_t,a_t,t,lam,
                                                                 optimizer_Psi,Returns_train_tf,N_MC,N_MC_inner,epsilon,T,
                                                                 scale_factor=scale_factor,Batch_size=Batch_size_t) # Fit Psi_t and J_t
                psi_errors.append(psi_error)
                if print_intermediate_results:
                    if not(k % print_step_size) and k >0:
                        print("Psi: {}".format(np.abs(scale_factor)*np.mean(psi_errors[-print_step_size:])))
                if np.abs(scale_factor)*np.mean(psi_errors[-print_step_size:]) < tolerance_psi:
                    print("Psi: {} Tolerance reached".format(np.abs(scale_factor)*np.mean(psi_errors[-print_step_size:])))
                    break
    return a_tilde, Psi_tilde