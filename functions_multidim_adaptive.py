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
   
    


def sample_omega_batch_empirical_ndim(omega, t, N_MC, batch_size,Returns_train_tf):
    
    samples_batch = tf.concat([tf.cast(tf.tile(tf.expand_dims(Returns_train_tf,0),[batch_size,1,1]), tf.float32),omega],2)
    # Get the number of columns in samples_batch
    num_columns = samples_batch.shape[2]
    # Generate random indices for sampling, for each batch element
    random_indices = tf.random.uniform((samples_batch.shape[0], N_MC), minval=0, maxval=num_columns, dtype=tf.int32)
    # Use tf.gather to sample elements based on the random indices
    sampled_tensor = tf.gather(samples_batch, random_indices, axis=2, batch_dims=1)
    ret_vector = tf.expand_dims(tf.transpose(sampled_tensor,[0,2,1]),3)
    return ret_vector



#Function that maximizes the expected value w.r.t. a
def max_a_ndim_adaptive(Psi_tilde,a_tilde,omega, a,t,optimizer_a,optimizer_lam,lam,Returns_train_tf,N_MC =1024,N_MC_inner = 1024, epsilon=0,T=5,Batch_size=32):
    with tf.GradientTape(persistent=True) as tape: #
        loss_a = -tf.reduce_mean(min_exp_ndim_adaptive(Psi_tilde,a_tilde,omega,a,t,lam,Returns_train_tf,N_MC,N_MC_inner,training_a = True,
                                         training_Psi = False,epsilon =epsilon,T=T,Batch_size=Batch_size))
        
    if epsilon == 0: # NON-ROBUST CASE
        if t>0:
            gradient_a = tape.gradient(loss_a,a_tilde[t].trainable_variables)
            optimizer_a[t].apply_gradients(zip(gradient_a, a_tilde[t].trainable_variables))      

        if t == 0:
            gradient_a = tape.gradient(loss_a,[a_tilde[t]])
            optimizer_a[t].apply_gradients(zip(gradient_a, [a_tilde[t]]))
        del tape
        
    else: #ROBUST CASE
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

# Function that computes the (Robust) expectation
@tf.function
def min_exp_ndim_adaptive(Psi_tilde,a_tilde,omega,a,t,lam,Returns_train_tf,N_MC =1024, N_MC_inner = 1024, 
            training_a = False,training_Psi = False,epsilon=0,T=5,Batch_size=32):  
    output = tf.cast(0,tf.float32)
    d = Returns_train_tf.shape[0]
    if  epsilon == 0:# NON-ROBUST CASE
        if t >0:
            # omega has Dimension (Batch_size,d,t)
            next_omega =  sample_omega_batch_empirical_ndim(omega,t,N_MC,Batch_size,Returns_train_tf ) # size = (Batch_size,N_MC,d,1)  
            omega_expanded = tf.expand_dims(omega, axis=1) # size =   (Batch_size_t,1, d,t)
            old_omega = tf.tile(omega_expanded, multiples=[1, N_MC, 1,1])  # Shape: (Batch_size, N_MC,d, t)
            omega_input = tf.concat([old_omega,next_omega],axis = 3)  # Dimension (Batch_size,N_MC,d,t+1)
            a_concat = tf.concat([a,tf.expand_dims(a_tilde[t]({"omega":omega,"a":a},training = training_a),2)],axis =2 ) #(Batch_size,d,t+1)
            a_expanded = tf.expand_dims(a_concat, axis=1) #(Batch_size,1,d,t+1)            
            a_input = tf.tile(a_expanded, multiples=[1, N_MC, 1,1])  # Dimension (Batch_size,N_MC,d,t+1)
            
            if t == T-1:
                output = Psi_tilde[t+1]({"omega":omega_input,"a":a_input})
                output = tf.reduce_mean(output,axis = 1)  # Dimension (Batch_size)
            else:
                output = Psi_tilde[t+1]({"omega":tf.reshape(omega_input,(Batch_size*N_MC,d,t+1)),
                                         "a":tf.reshape(a_input,(Batch_size*N_MC,d,t+2))})
                output = tf.reduce_mean(tf.reshape(output,(Batch_size,N_MC)),axis = 1)  # Dimension (Batch_size) 
                
        elif t == 0:
 
            #samples_batch = tf.concat([tf.cast(tf.tile(Returns_train_tf,[batch_size,1]), tf.float32),omega],1)  #(d,N))
            # Get the number of columns in samples_batch
            num_columns = Returns_train_tf.shape[1]
            # Generate random indices for sampling, for each batch element
            random_indices = tf.random.uniform([N_MC], minval=0, maxval=num_columns, dtype=tf.int32)
            # Use tf.gather to sample elements based on the random indices
            omega_input =  tf.transpose(tf.gather(Returns_train_tf, random_indices, axis=1)) #(N_MC,d)          
            

            #next_omega = tf.cast(tf.reshape(Returns_train[np.random.choice(Returns_train_tf.shape[1],N_MC)],(N_MC,1)),tf.float32)
            #omega_input = next_omega # Dimension (N_MC,t+1)

            a_input = tf.reshape(tf.repeat(a_tilde[t],N_MC),(N_MC,d,2)) # Dimension (N_MC,d,2)
            #a_input = tf.repeat(tf.expand_dims(a_input,axis =1),N_MC,axis = 1) # Dimension (B,N_MC,t+1)
            output = tf.reduce_mean(Psi_tilde[t+1]({"omega":omega_input,
                                                        "a":a_input})) 
    elif epsilon > 0:# ROBUST CASE
        if t >0:

            #NEW
            old_omega = tf.expand_dims(omega, axis=1)  # Shape (Batch_size, 1,d, t)
            old_omega = tf.tile(old_omega, [1, N_MC_inner, 1,1]) # Shape (Batch_size, N_MC_inner,d, t)
            z = tf.random.normal(shape=(Batch_size,N_MC_inner,d,1), mean=0.0, stddev=0.01)  # Shape (Batch_size, N_MC_inner, d,1) 
            omega_input = tf.concat([old_omega,z],axis = 3)  # Dimension (Batch_size,N_MC_inner,d,t+1)

            x = sample_omega_batch_empirical_ndim(omega,t,N_MC,Batch_size,Returns_train_tf ) # size = (Batch_size,N_MC,d,1)   
            x_transpose= tf.transpose(x, perm=[0, 3,2, 1]) # size = (Batch_size,1,d,N_MC)   

            x_expanded = tf.repeat(x_transpose, N_MC_inner, axis=1)  # (Batch_size,N_MC_inner,d,N_MC)   
            z_expanded = tf.repeat(z, N_MC, axis=3)  # (Batch_size,N_MC_inner,d,N_MC)

            a_concat = tf.concat([a,tf.expand_dims(a_tilde[t]({"omega":omega,"a":a},training = training_a),2)],axis =2 ) #(Batch_size,d,t+1)
            a_expanded = tf.expand_dims(a_concat, axis=1) #(Batch_size,1,d,t+1)            
            a_input = tf.tile(a_expanded, multiples=[1, N_MC_inner, 1,1])  # Dimension (Batch_size,N_MC_inner,d,t+1)

            if t == T-1:
                psi_tilde_output = Psi_tilde[t+1]({"omega":omega_input,"a":a_input}) # Dimension (Batch_size,N_MC)
            else:
                psi_tilde_output = Psi_tilde[t+1]({"omega":tf.reshape(omega_input,(Batch_size*N_MC_inner,d,t+1)),
                                         "a":tf.reshape(a_input,(Batch_size*N_MC_inner,d,t+2))})
                psi_tilde_output = tf.reshape(psi_tilde_output,(Batch_size,N_MC_inner))  # Dimension (Batch_size,N_MC_inner) 
                
                ##########################################
            inner_term = tf.expand_dims(psi_tilde_output, 2) + lam *tf.math.sqrt(tf.reduce_sum( (x_expanded - z_expanded)**2,axis = 3))   # (Batch_size,N_MC_inner,N_MC)

            all_together =  tf.math.reduce_min(inner_term,1)  # Use TensorFlow's tf.exp         # (Batch_size,N_MC)
            outer_mean = tf.reduce_mean(all_together,axis =1)      # (Batch_size)   
            output = outer_mean-lam*epsilon # (Batch_size)



        elif t == 0:
            #NEW
            z = tf.random.normal(shape=(N_MC_inner, d,1), mean=0.0, stddev=0.01) #(N_MC_inner, d,1)
            omega_input = z  # Dimension (N_MC,d,1)

        
            indices = tf.random.uniform(shape=(N_MC,), maxval=Returns_train_tf.shape[1], dtype=tf.int32)
            x_transpose = tf.gather(Returns_train_tf, indices, axis=1)  # (d,N_MC)



            x_expanded = tf.cast(tf.repeat(tf.expand_dims(x_transpose, axis=0),N_MC_inner, axis = 0),tf.float32) # (N_MC_inner, d,N_MC)
            z_expanded = tf.repeat(z, N_MC, axis=2)  # (N_MC_inner, d,N_MC)

            a_tilde_tensor = tf.convert_to_tensor(a_tilde[t], dtype=tf.float32)
            a_repeated = tf.repeat(a_tilde_tensor[tf.newaxis, :], repeats=N_MC_inner, axis=0) #
            a_input = tf.reshape(a_repeated, (N_MC_inner, d,2))  # Adjust the second dimension as needed


            psi_tilde_output = Psi_tilde[t+1]({"omega": omega_input, "a": a_input})
            inner_term = psi_tilde_output + lam *tf.math.sqrt(tf.reduce_sum( (x_expanded - z_expanded)**2,axis = 1)) # (N_MC_inner, N_MC)

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
def approx_psi_ndim_adaptive(Psi_tilde, a_tilde, omega, a, t, lam, optimizer_Psi, Returns_train_tf, 
               N_MC=128, N_MC_inner=128, epsilon=0, T=5, scale_factor=1, Batch_size=32):
    with tf.GradientTape() as tape:
        min_exp_output = min_exp_ndim_adaptive(Psi_tilde, a_tilde, omega, a, t, lam, Returns_train_tf, 
                                 N_MC, N_MC_inner, epsilon=epsilon, T=T, Batch_size=Batch_size)
        psi_output = tf.squeeze(Psi_tilde[t]({"omega": omega, "a": a}, training=True), axis=1)
        loss_psi = tf.reduce_mean((psi_output - min_exp_output) ** 2) / scale_factor**2

    gradients = tape.gradient(loss_psi, Psi_tilde[t].trainable_variables)
    optimizer_Psi[t].apply_gradients(zip(gradients, Psi_tilde[t].trainable_variables))
    return loss_psi, Psi_tilde, optimizer_Psi


# The training routine
def train_networks_ndim_adaptive(Returns_train_tf,
                   T= 5,
                   inner_psi = 5000,
                   inner_a = 5000, 
                   N_MC = 512,
                   N_MC_inner = 512,
                  epsilon = 0,
                  tolerance_psi = 0.00001,
                   tolerance_a = 0.00001,
                   learning_rate_Psi = 0.001,
                  learning_rate_a = 0.001,
                   learning_rate_LAM = 0.001,
                  Batch_size_psi = 32,
                  Batch_size_a = 32,
                  print_intermediate_results = False,
                  print_step_size = 25):
    #Determine number of stocks
    d = Returns_train_tf.shape[0]
    # Assign models and optimizers
    a_tilde, Psi_tilde, lam, optimizer_a, optimizer_Psi, optimizer_lam = create_models_ndim(T,d,learning_rate_Psi,learning_rate_a,learning_rate_LAM)

    print("Start Backwards Iterations")
    for t in range(T)[::-1]: # Backwards Iteration
        #for j in range(N_iterations):
        psi_errors = []
        a_values = []


        # transfer learning:
        if t <T-1 and t>0:
            for k in range(6,len(a_tilde[t+1].layers)-2):
                a_tilde[t].layers[k].set_weights(a_tilde[t+1].layers[k].get_weights())
                a_tilde[t].layers[k].trainable = False
            k = len(a_tilde[t+1].layers)-1
            a_tilde[t].layers[k].set_weights(a_tilde[t+1].layers[k].get_weights())
            a_tilde[t].layers[k].trainable = True 

        # transfer learning:
        if t <T-1 and t>0:
            for k in range(6,len(Psi_tilde[t+1].layers)-2):
                Psi_tilde[t].layers[k].set_weights(Psi_tilde[t+1].layers[k].get_weights())
                Psi_tilde[t].layers[k].trainable = False
            k = len(Psi_tilde[t+1].layers)-1
            Psi_tilde[t].layers[k].set_weights(Psi_tilde[t+1].layers[k].get_weights())
            Psi_tilde[t].layers[k].trainable = True 
                
        print("#########\n# t = {} #\n#########\n".format(t))

        # Maximization w.r.t. a
        for k in range(inner_a):
            if t>0:
                Batch_size_t = Batch_size_a #int(Batch_size * (1.1 ** min(k, 50)))  # Cap at `50` iterations
                #Batch_size_t = Batch_size

                
                omega_t = sample_states_batch_ndim(t, Returns_train_tf, Batch_size_t)
                a_t = sample_actions_batch_ndim(t, d,Batch_size_t)
                
                
            if t== 0:
                a_t = 0.
                omega_t = 0.
              #(1,2*T*t))
            a_value, a_tilde, optimizer_a, lam, optimizer_lam, gradient_a = max_a_ndim_adaptive(Psi_tilde,a_tilde,omega_t,a_t,t,
                                                  optimizer_a,optimizer_lam,lam,Returns_train_tf,
                                                   N_MC,N_MC_inner,tf.cast(epsilon[t],tf.float32),T,Batch_size=Batch_size_t)
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
                omega_t = sample_states_batch_ndim(t, Returns_train_tf, Batch_size_t)
                a_t = sample_actions_batch_ndim(t,d, Batch_size_t)
                
                
                
                psi_error, Psi_tilde, optimizer_Psi = approx_psi_ndim_adaptive(Psi_tilde,a_tilde,omega_t,a_t,t,lam,
                                                                 optimizer_Psi,Returns_train_tf,N_MC,N_MC_inner,tf.cast(epsilon[t],tf.float32),T,
                                                                 scale_factor=scale_factor,Batch_size=Batch_size_t) # Fit Psi_t and J_t
                psi_errors.append(psi_error)
                if print_intermediate_results:
                    if not(k % print_step_size) and k >0:
                        print("Psi: {}".format(np.abs(scale_factor)*np.mean(psi_errors[-print_step_size:])))
                if np.abs(scale_factor)*np.mean(psi_errors[-print_step_size:]) < tolerance_psi:
                    print("Psi: {} Tolerance reached".format(np.abs(scale_factor)*np.mean(psi_errors[-print_step_size:])))
                    break
    return a_tilde, Psi_tilde

