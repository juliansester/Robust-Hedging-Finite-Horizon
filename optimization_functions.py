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

def U(x,a=0.88,b=2.25):
    if x>0:
        return (x**(a))
    if x<=0:
        return b*((-x)**(a))

# Define Black-Scholes Price and Hedge
def blackScholes(r, S, K, T, sigma, type="c"):
    "Calculate BS price of call/put"
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if type == "c":
        price = S*norm.cdf(d1, 0, 1) - K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
    elif type == "p":
        price = K*np.exp(-r*T)*norm.cdf(-d2, 0, 1) - S*norm.cdf(-d1, 0, 1)
    return price

def delta_calc(r, S, K, T, sigma, type="c"):
    "Calculate delta of an option"
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    if type == "c":
        delta_calc = norm.cdf(d1, 0, 1)
    elif type == "p":
        delta_calc = -norm.cdf(-d1, 0, 1)
    return delta_calc

def delta_basket(r,weighted_S,K,T,basket_vol):
    d1 = (np.log(weighted_S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    return norm.cdf(d1, 0, 1) 

# Functions to sample omega

def sample_omega(omega, t, N_MC, Returns_train_tf, beta=5000000):
    Returns_lagged = tf.concat([Returns_train_tf[:, i:(-t + i)] for i in range(t)], axis=0)  # (t,N_Returns)
    Returns_lagged = tf.cast(tf.transpose(Returns_lagged), tf.float32)[:-1,:]  
    # Do not allow for the last return to be considered (as it should be possible to predict it)
    dist = tf.math.exp(-beta * tf.reduce_sum((Returns_lagged - omega) ** 2, axis=1))
    probs = np.array(dist) / tf.reduce_sum(dist)
    # print(pd.DataFrame(probs).describe())
    p_hat = tfp.distributions.Categorical(probs=dist)
    closest_returns_indices = np.array(p_hat.sample(N_MC) + t, dtype=int)  # (Batch)
    # Access the values directly using the indices:
    return tf.cast(tf.reshape(Returns_train_tf[0,:].iloc[closest_returns_indices].values, (N_MC, 1)), tf.float32)

# Function to sample omega batch-wise

def sample_omega_batch(omega, t, N_MC, batch_size,Returns_train_tf, beta=5000000, empirical = False):
    Returns_lagged = tf.concat([Returns_train_tf[:, i:(-t + i)] for i in range(t)], axis=0)  # (t,N_Returns)
    Returns_lagged = tf.cast(tf.transpose(Returns_lagged), tf.float32)[:-1,:]  # (N_Returns,t-1)
    Returns_lagged = tf.tile(tf.expand_dims(Returns_lagged, axis=0), [batch_size, 1, 1]) # (batch_size,N_Returns,t-1)
    # Do not allow for the last return to be considered (as it should be possible to predict it)
    
    if empirical:
        # Create uniform probabilities for N_Returns
        uniform_probs = tf.fill([batch_size, Returns_lagged.shape[1]], 1.0 / Returns_lagged.shape[1])
        p_hat = tfp.distributions.Categorical(probs=uniform_probs)
    else:
        dist = tf.math.exp(-beta * tf.reduce_sum((Returns_lagged - tf.expand_dims(omega,axis=1)) ** 2, axis=2)) # (batch_size,N_Returns)
        p_hat = tfp.distributions.Categorical(probs=dist)
    #closest_returns_indices = np.array(p_hat.sample(N_MC) + t, dtype=int)  # (N_MC,batch_size)
    closest_returns_indices = tf.cast(p_hat.sample(N_MC) + t, tf.int32)  # (N_MC,batch_size)
    # Access the values directly using the indices:
    Returns_selected = tf.gather(Returns_train_tf[0,:], closest_returns_indices) # (N_MC,batch_size)
    return tf.cast(tf.reshape(Returns_selected, (batch_size,N_MC, 1)), tf.float32)

# Tensorflow adaption of U
def U_tf(x, a=0.88, b=2.25):
    x_inp = tf.cast(x, tf.float32)
    pos = tf.pow(tf.maximum(x_inp,0), a)
    neg = b * tf.pow(tf.maximum(-x_inp, 0), a)  # Here Maximum necessary yo have gradients
    return pos + neg

def Psi(input_dict,S_0 = 1):
    omega, a = input_dict["omega"], input_dict["a"] #(Batch,N_MC,T) for omega and #(Batch,N_MC,T+1)  for a
    S = S_0*tf.math.cumprod(omega+1,axis =2)
    cash = a[:,:,0]
    profit = tf.reduce_sum((S[:,:,1:]-S[:,:,:-1])*a[:,:,2:],axis=2)+a[:,:,1]*(S[:,:,0]-S_0)
    payoff = tf.nn.relu(S[:,:,-1]-S_0) # At the money call option
    return -U_tf(cash+profit-payoff)

def Psi_non_batch(input_dict,S_0 = 1):
    omega, a = input_dict["omega"], input_dict["a"] #(N_MC,T) for omega and #(N_MC,T+1)  for a
    S = S_0*tf.math.cumprod(omega+1,axis =1)
    cash = a[:,0]
    profit = tf.reduce_sum((S[:,1:]-S[:,:-1])*a[:,2:],axis=1)+a[:,1]*(S[:,0]-S_0)
    payoff = tf.nn.relu(S[:,-1]-S_0) # At the money call option
    return -U_tf(cash+profit-payoff)

def Psi_quadratic(input_dict,S_0 = 1):
    omega, a = input_dict["omega"], input_dict["a"] #(N_MC,T) for omega and #(N_MC,T+1)  for a
    S = S_0*tf.math.cumprod(omega+1,axis =1)
    cash = a[:,0]
    profit = tf.reduce_sum((S[:,1:]-S[:,:-1])*a[:,2:],axis=1)+a[:,1]*(S[:,0]-S_0)
    payoff = tf.nn.relu(S[:,-1]-S_0) # At the money call option
    return -(cash+profit-payoff)**2

# Functions to create neural networks

def build_model_a(Input_Dim,T=5,depth =5,nr_neurons =32):
    """
    Function that creates the neural network for a
    """
    #Input Layer
    omega = keras.Input(shape=(Input_Dim,),name = "omega")
    a = keras.Input(shape=(1+Input_Dim,),name = "a")

    #in1 = tf.keras.layers.GlobalAveragePooling1D()(omega)
    #in2 = tf.keras.layers.GlobalAveragePooling1D()(a)
    in1 = tf.keras.layers.Flatten()(omega)
    in2 = tf.keras.layers.Flatten()(a)
    # concatenate
    v = layers.concatenate([in1,in2])
    # Batch Normalization applied to the input
    #v = layers.BatchNormalization()(v)

    # Create the NN
    v = layers.Dense(nr_neurons,activation = "relu")(v) #,kernel_regularizer=tf.keras.regularizers.L1(0.01))(v)

    for i in range(depth):
        #v = layers.BatchNormalization()(v)
        v= layers.Dense(nr_neurons,activation = "relu")(v) #,kernel_regularizer=tf.keras.regularizers.L1(0.01))(v)
    # Output Layers
    value_out = layers.Dense(1,activation = "tanh")(v)
    model = keras.Model(inputs=[omega,a],outputs = [value_out])
    return model

# Function to create a neural network for psi

def build_model_psi(Input_Dim,T=5,depth =5,nr_neurons =32):
    """
    Function that creates the neural network for the value function V
    """
    #Input Layer
    omega = keras.Input(shape=(Input_Dim,),name = "omega")
    a = keras.Input(shape=(1+Input_Dim,),name = "a")

    in1 = tf.keras.layers.Flatten()(omega)
    in2 = tf.keras.layers.Flatten()(a)
    # concatenate
    v = layers.concatenate([in1,in2])
    # Batch Normalization applied to the input
    #v = layers.BatchNormalization()(v)

    # Create the NN
    v = layers.Dense(nr_neurons,activation = "relu")(v) #,kernel_regularizer=tf.keras.regularizers.L1(0.01))(v)
    # Create deep layers
    for i in range(depth):
        #v = layers.BatchNormalization()(v)
        v = layers.Dense(nr_neurons,activation =  "relu")(v) #,kernel_regularizer=tf.keras.regularizers.L1(0.01))(v)
    # Output Layers
    value_out = layers.Dense(1)(v)
    model = keras.Model(inputs=[omega,a],outputs = [value_out])
    return model

# Create Models (NNs)
def create_models(T=5,learning_rate_Psi = 0.001,learning_rate_a = 0.001,learning_rate_LAM = 0.001):
    a_tilde = []
    Psi_tilde = []
    optimizer_a = []
    optimizer_Psi = []
    
    # Ensure Actions between -1 and 1
    class AbsValueConstraint(tf.keras.constraints.Constraint):
        def __call__(self, w):
            return tf.clip_by_value(w, -1.0, 1.0)
            
    # Ensure Lambda between 0.5 and 50
    class LamValueConstraint(tf.keras.constraints.Constraint):
        def __call__(self, w):
            return tf.clip_by_value(w, 0.5, 10000.0)
        
    # time 0
    a_tilde.append(tf.Variable([0.,0.5],trainable=True,dtype = "float32", 
                      constraint=AbsValueConstraint())) # inititial cash d_0 and Delta_0
    Psi_tilde.append(tf.Variable([1,],trainable=True,dtype = "float32")) # just a placeholder. actually not needed
    # Create optimizers
    optimizer_a.append(tf.keras.optimizers.Adam(learning_rate = learning_rate_a) )
    optimizer_Psi.append(tf.keras.optimizers.Adam(learning_rate = learning_rate_Psi) ) # just a placeholder. actually not needed

    # Times 1 to T-1
    for t in range(1,T):
        a_tilde.append(build_model_a(t,T=T,depth =4,nr_neurons =8*t))
        Psi_tilde.append(build_model_psi(t,T=T,depth =4,nr_neurons =8*t))
         # Create optimizers
        optimizer_a.append(tf.keras.optimizers.Adam(learning_rate = learning_rate_a) )
        optimizer_Psi.append(tf.keras.optimizers.Adam(learning_rate = learning_rate_Psi) )
    # Time T
    Psi_tilde.append(Psi)
    optimizer_Psi.append(tf.keras.optimizers.Adam(learning_rate = learning_rate_Psi) )
    
    optimizer_lam = tf.keras.optimizers.Adam(learning_rate = learning_rate_LAM)    
    lam = tf.Variable([1.,],trainable=True,dtype = "float32", constraint = LamValueConstraint())
    # Define the optimizers for a and Psi respectively
    return a_tilde, Psi_tilde, lam, optimizer_a, optimizer_Psi, optimizer_lam


#Function to sample a state
def sample_state(t, # time
                 Returns_train_tf,
                 C=5):
    Returns_lagged = tf.concat([Returns_train_tf[:,i:(-t+i)] for i in range(t)],axis = 0) #(t,N_Returns)
    Returns_lagged = tf.cast(Returns_lagged,tf.float32) # Do not allow for the last return to be considered (as it should be possible to predict it)
    index = np.random.choice(Returns_lagged.shape[1],1)[0]
    return tf.cast(tf.reshape(Returns_lagged[:,index],(1,t)),tf.float32) #(1,t)

def sample_states_batch(t, Returns_train_tf, Batch_size_t):
    Returns_lagged = tf.concat([Returns_train_tf[:, i:(-t + i)] for i in range(t)], axis=0)  # (t, N_Returns)
    Returns_lagged = tf.cast(Returns_lagged, tf.float32)

    indices = tf.random.uniform(shape=(Batch_size_t,), maxval=Returns_lagged.shape[1], dtype=tf.int32)
    sampled_states = tf.gather(Returns_lagged, indices, axis=1)  # (t, Batch_size_t)
    
    return tf.transpose(sampled_states)  # (Batch_size_t, t)

def samples_a_from_action(omega, t, a_tilde,Batch_size_t):
    a_opt = [tf.tile(tf.expand_dims(a_tilde[0],0),[Batch_size_t,1])]
    omega_input = []

    for s in range(t):
        omega_input.append(omega[:,s])  # Collect values instead of reshaping each time
        a_input = tf.reshape(tf.concat(a_opt, axis=1), (Batch_size_t, s + 2))
        a_opt.append(a_tilde[s + 1]({"omega": tf.reshape(omega_input, (Batch_size_t, s + 1)), "a": a_input}))
    noise = tf.random.normal(shape=(Batch_size_t,t+2), mean=0.0, stddev=0.01)
    return tf.concat(a_opt,1)+noise #(Batch_size_t,t+1)

#Function to sample an action
def sample_action(t,D=1):
    vals = tf.random.uniform([1+t], minval = -D,maxval=D)
    return tf.reshape(tf.cast(vals,tf.float32),(1,1+t))

# def sample_actions_batch(t, batch_size_t, omega, t,d=1,vol = 0.2):
#     s = np.cumprod(omega + 1,1)[:,:-1] #(batch_size_t,t)
#     times = tf.tile(tf.expand_dims(tf.range(t-t+2,t+1, dtype=tf.float32)[::-1], axis=0), [32, 1])
#     cash = tf.cast(blackscholes(r=0, s=1, k=1, t=t/365, sigma=vol, type="c"),tf.float32)
#     delta_0 = tf.cast(delta_calc(r=0, s=1, k=1, t=t/365, sigma=vol, type="c"),tf.float32)
#     delta = norm.cdf((np.log(s) + (vol**2/2)*times)/(vol*np.sqrt(times)))
#     vals = tf.concat([tf.expand_dims(tf.repeat(delta_0,32),1),tf.expand_dims(tf.repeat(cash,32),1),delta],1)
#     return vals  # shape: (batch_size_t, 1 + t)

def sample_actions_batch(t, Batch_size_t,D=1):
    vals = tf.random.uniform([Batch_size_t, 1 + t], minval= -D, maxval=D, dtype=tf.float32)
    return vals  # Shape: (Batch_size_t, 1 + t)

#Function that maximizes the expected value w.r.t. a
def max_a(Psi_tilde,a_tilde,omega, a,t,optimizer_a,optimizer_lam,lam,Returns_train_tf,N_MC =1024,N_MC_inner = 1024, epsilon=0,T=5,Batch_size=32):
    with tf.GradientTape(persistent=True) as tape: #
        loss_a = -tf.reduce_mean(min_exp(Psi_tilde,a_tilde,omega,a,t,lam,Returns_train_tf,N_MC,N_MC_inner,training_a = True,
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
def min_exp(Psi_tilde,a_tilde,omega,a,t,lam,Returns_train_tf,N_MC =1024, N_MC_inner = 1024, 
            training_a = False,training_Psi = False,epsilon=0,T=5,Batch_size=32):  


    if  epsilon == 0:# NON-ROBUST CASE
        if t >0:
            # omega has Dimension (Batch_size,t)
            next_omega =  sample_omega_batch(omega,t,N_MC,Batch_size,Returns_train_tf ) # size = (Batch_size,N_MC,1)  np.random.exponential(scale=scale, size = (N_MC,1))
            omega_expanded = tf.expand_dims(omega, axis=1) 
            old_omega = tf.tile(omega_expanded, multiples=[1, N_MC, 1])  # Shape: (Batch_size, N_MC, t)
            omega_input = tf.concat([old_omega,next_omega],axis = 2)  # Dimension (Batch_size,N_MC,t+1)
            a_concat = tf.concat([a,a_tilde[t]({"omega":omega,"a":a},training = training_a)],axis =1 ) #(Batch_size,t+1)
            a_expanded = tf.expand_dims(a_concat, axis=1) #(Batch_size,1,t+1)            
            a_input = tf.tile(a_expanded, multiples=[1, N_MC, 1])  # Dimension (Batch_size,N_MC,t+1)
            
            if t == T-1:
                output = Psi_tilde[t+1]({"omega":omega_input,"a":a_input})
                output = tf.reduce_mean(output,axis = 1)  # Dimension (Batch_size)
            else:
                output = Psi_tilde[t+1]({"omega":tf.reshape(omega_input,(Batch_size*N_MC,t+1)),
                                         "a":tf.reshape(a_input,(Batch_size*N_MC,t+2))})
                output = tf.reduce_mean(tf.reshape(output,(Batch_size,N_MC)),axis = 1)  # Dimension (Batch_size) 
                          
        elif t == 0:
            # omega has Dimension (t)
                # omega has Dimension t
            indices = tf.random.uniform(shape=(N_MC,), minval=0, maxval=Returns_train_tf.shape[1],dtype=tf.int32)
            selected_returns = tf.gather(Returns_train_tf[0,:], indices, axis=0)
            omega_input= tf.cast(tf.reshape(selected_returns, (N_MC, 1)), tf.float32)
    
            
            #next_omega = tf.cast(tf.reshape(Returns_train[np.random.choice(Returns_train_tf.shape[1],N_MC)],(N_MC,1)),tf.float32)
            #omega_input = next_omega # Dimension (N_MC,t+1)
    
            a_input = tf.reshape(tf.repeat(a_tilde[t],N_MC),(N_MC,2)) # Dimension (N_MC,2) , Action at time 0
            #a_input = tf.repeat(tf.expand_dims(a_input,axis =1),N_MC,axis = 1) # Dimension (B,N_MC,t+1)
            output = tf.reduce_mean(Psi_tilde[t+1]({"omega":omega_input,
                                                        "a":a_input})) 
    elif epsilon > 0:# ROBUST CASE
        if t >0:

            #NEW
            old_omega = tf.expand_dims(omega, axis=1)  # Shape (Batch_size, 1, t)
            old_omega = tf.tile(old_omega, [1, N_MC_inner, 1]) # Shape (Batch_size, N_MC_inner, t)
            z = tf.random.normal(shape=(Batch_size,N_MC_inner,1), mean=0.0, stddev=0.01)  # Shape (Batch_size, N_MC_inner, 1) 
            omega_input = tf.concat([old_omega,z],axis = 2)  # Dimension (Batch_size,N_MC_inner,t+1)

            x = sample_omega_batch(omega,t,N_MC,Batch_size,Returns_train_tf ) # size = (Batch_size,N_MC,1)    
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
        min_exp_output = min_exp(Psi_tilde, a_tilde, omega, a, t, lam, Returns_train_tf, 
                                 N_MC, N_MC_inner, epsilon=epsilon, T=T, Batch_size=Batch_size)
        psi_output = tf.squeeze(Psi_tilde[t]({"omega": omega, "a": a}, training=True), axis=1)
        loss_psi = tf.reduce_mean((psi_output - min_exp_output) ** 2) / scale_factor**2

    gradients = tape.gradient(loss_psi, Psi_tilde[t].trainable_variables)
    optimizer_Psi[t].apply_gradients(zip(gradients, Psi_tilde[t].trainable_variables))
    return loss_psi, Psi_tilde, optimizer_Psi


# The training routine
def train_networks(Returns_train_tf,
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
                  print_step_size = 25,
                  load_model = False,
                  folder_name = "to_fill",
                  file_name = "to_fill"):
    # Assign models and optimizers
    a_tilde, Psi_tilde, lam, optimizer_a, optimizer_Psi, optimizer_lam = create_models(T,learning_rate_Psi,learning_rate_a,learning_rate_LAM)
    
    if load_model == True:
        a_tilde = file_load(folder_name,file_name,T)
        Psi_tilde = file_load(folder_name,"Psi_"+file_name,T,psi = True)
    # Time T
    Psi_tilde.append(Psi)    
    print("Start Backwards Iterations")
    for t in range(T)[::-1]: # Backwards Iteration
        #for j in range(N_iterations):
        psi_errors = []
        a_values = []
        

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
                if load_model:
                    a_t = sample_actions_batch(t, Batch_size_t)
                    #a_t = samples_a_from_action(omega_t, t-1, a_tilde,Batch_size_t)  # (Batch_size_t, t+1)
                if load_model == False:
                    a_t = sample_actions_batch(t, Batch_size_t)
                
            if t== 0:
                a_t = 0.
                omega_t = 0.
              #(1,2*T*t))
            a_value, a_tilde, optimizer_a, lam, optimizer_lam, gradient_a = max_a(Psi_tilde,a_tilde,omega_t,a_t,t,
                                                  optimizer_a,optimizer_lam,lam,Returns_train_tf,
                                                   N_MC,N_MC_inner,epsilon,T,Batch_size=Batch_size_t)
            a_values.append(a_value)
            #print(np.mean([np.mean(np.abs(gradient_a[i])) for i in range(len(gradient_a))]))

            if print_intermediate_results:
                if not(k % print_step_size):
                    print("a: {}, Iteration: {}".format(np.mean(a_values[-print_step_size:]),k))
                    #print("a_gradient: {}".format(np.abs(np.mean(gradient_a_reference[-25:]))))
        # Minimization of Psi
        if t >0:
            for k in range(inner_psi):
                
                
                scale_factor = np.mean(a_values[-25:])
                
                Batch_size_t = Batch_size_psi #int(1+((k+1)/inner_psi)*Batch_size) # Increase Batch Size linearly
                #Batch_size_t = Batch_size
                omega_t = sample_states_batch(t, Returns_train_tf, Batch_size_t)
                
                if load_model:
                    a_t = sample_actions_batch(t, Batch_size_t) 
                    #a_t = samples_a_from_action(omega_t, t-1, a_tilde,Batch_size_t)
                if load_model == False:
                    a_t = sample_actions_batch(t, Batch_size_t) # (Batch_size_t, t)
                
                
                
                psi_error, Psi_tilde, optimizer_Psi = approx_psi(Psi_tilde,a_tilde,omega_t,a_t,t,lam,
                                                                 optimizer_Psi,Returns_train_tf,N_MC,N_MC_inner,epsilon,T,
                                                                 scale_factor=scale_factor,Batch_size=Batch_size_t) # Fit Psi_t and J_t
                psi_errors.append(psi_error)
                if print_intermediate_results:
                    if not(k % print_step_size):
                        print("Psi: {}, Iteration: {}".format(np.abs(scale_factor)*np.mean(psi_errors[-print_step_size:]),k))
                if np.abs(scale_factor)*np.mean(psi_errors[-print_step_size:]) < tolerance_psi:
                    print("Psi: {} Tolerance reached".format(np.abs(scale_factor)*np.mean(psi_errors[-print_step_size:])))
                    break
    return a_tilde, Psi_tilde

# Function to save the trained neural network in a file
def file_save(list,folder_name,file_name):
    # Save the variable using TensorFlow checkpointing
    checkpoint = tf.train.Checkpoint(var=list[0])
    checkpoint.write(folder_name+"/"+file_name+"_0.ckpt")
    
    # Save the model using TensorFlow's save method
    for t in range(1,len(list)):
        list[t].save(folder_name+"/"+file_name+"_"+str(t)+".keras",overwrite=True )
        
# Function to load the trained neural network from a file
def file_load(folder_name,file_name,T,psi = False):
    dummy_ll = []
    if psi:
        dummy_ll.append(tf.Variable([1,],trainable=True,dtype = "float32")) # just a placeholder. actually not needed
    if psi == False:  
        # Save the variable using TensorFlow checkpointing
        restored_variable = tf.Variable([0,  0], dtype=tf.float32)
        checkpoint = tf.train.Checkpoint(var=restored_variable)
        checkpoint.restore(folder_name+"/"+file_name+"_0.ckpt").expect_partial()
        dummy_ll.append(restored_variable)
    # load the model using TensorFlow's save method
    for t in range(1,T):
        dummy_ll.append(tf.keras.models.load_model(folder_name+"/"+file_name+"_"+str(t)+".keras",safe_mode=False))
    return dummy_ll

# Function to load the trained neural network from a file
def file_load_ndim(folder_name,file_name,T,d = 5):
    dummy_ll = []
    # Save the variable using TensorFlow checkpointing
    restored_variable = tf.Variable([0.,  0.]*d, dtype=tf.float32)
    checkpoint = tf.train.Checkpoint(var=restored_variable)
    checkpoint.restore(folder_name+"/"+file_name+"_0.ckpt").expect_partial()
    dummy_ll.append(restored_variable)
    # Save the model using TensorFlow's save method
    for t in range(1,T):
        dummy_ll.append(tf.keras.models.load_model(folder_name+"/"+file_name+"_"+str(t)+".keras",safe_mode=False))
    return dummy_ll