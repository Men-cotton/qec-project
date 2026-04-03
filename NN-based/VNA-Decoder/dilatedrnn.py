import tensorflow as tf
import tf_slim as slim
import numpy as np
import os
import time
import random
import pandas as pd
import copy
from src.rotated_surface_model import RotSurCode
def return_local_energies(samples, init_code):
    numsamples = samples.shape[0]

    local_energies = np.zeros((numsamples), dtype = np.float64)
    oldcode = RotSurCode(N_new)
    copymatrix = np.copy(init_code.qubit_matrix)
    oldcode.update_matrix(copymatrix)

    for x in range(numsamples):
        newcode = RotSurCode(N_new)
        flat = samples[x,:]
        notflat = flat.reshape(N_new,N_new)
        newcode.update_matrix(notflat)

        for i in range(N_new+1):
            for j in range(N_new+1):
                local_energies[x] += 100*np.absolute(newcode.plaquette_defects[i,j]-oldcode.plaquette_defects[i,j]) + np.sum(samples[x,:])
    return local_energies

def failrate(sols,testcode_matrix):
  num_samples = sols.shape[0]
  copymatrix=np.copy(testcode_matrix)
  newcode = RotSurCode(N_new)
  failarray = np.zeros([num_samples])
  for x in range(num_samples):
      flat = samples[x,:]
      notflat = flat.reshape(N_new,N_new)
      newcode.update_matrix(notflat)
      oldcode = RotSurCode(N_new)
      oldcode.update_matrix(copymatrix)
      fails = 0
      for i in range(N_new+1):
          for j in range(N_new+1):
              if np.absolute(newcode.plaquette_defects[i,j]-oldcode.plaquette_defects[i,j])!=0:
                  fails+=1
      failarray[x]=fails
  
  return np.count_nonzero(failarray)/num_samples

def individual_failrate(sample,init_code):
      newcode = RotSurCode(Nx)
      newcode.update_matrix(sample)
      copymatrix=np.copy(init_code.qubit_matrix)
      oldcode = RotSurCode(Nx)
      oldcode.update_matrix(copymatrix)
      size = sample.shape[0]
      for i in range(size+1):
          for j in range(size+1):
              if np.absolute(newcode.plaquette_defects[i,j]-oldcode.plaquette_defects[i,j])!=0:
                  return False
      return True

def f_join(numList):
    s = ''.join(map(str, numList.astype(int)))
    return int(s)
def code_circ(it,Nx):
    newmatrix = np.zeros([Nx,Nx])
    if it%2==0:
      newmatrix[0,0]=1
    if it%4<2:
      newmatrix[0,2]=1
    if it%8<4:
      newmatrix[2,0]=1
    if it%16<8:
      newmatrix[2,2]=1
    return newmatrix
def findplaq_place(N):
    currentx = N%3
    if N<3:
      currenty = 0
    if N<6 and N>2:
      currenty = 1
    if N>5:
      currenty = 2
    return currentx, currenty

"""## **The RNN cell**"""

class CustomRNNCell(tf.compat.v1.nn.rnn_cell.RNNCell):
    #The custom RNN cell to be able to use inputs from syndrome matrix
    def __init__(self, num_units = None, num_in = 2, activation = None,num_layers = 1, name=None, dtype = None, reuse=None):
        super(CustomRNNCell, self).__init__(_reuse=reuse, name=name)
        self._num_units = num_units
        self._activation = activation if activation is not None else tf.tanh

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        input_size = inputs_shape[-1]
        self._weights = self.add_variable("kernel", shape=[input_size, self._num_units])
        self._recurrent_weights = self.add_variable("recurrent_kernel", shape=[self._num_units, self._num_units])
        self._upleft_weight = self.add_variable("upleft", shape=[self._num_units, self._num_units])
        self._upright_weight = self.add_variable("upright", shape=[self._num_units, self._num_units])
        self._downleft_weight = self.add_variable("downleft", shape=[self._num_units, self._num_units])
        self._downright_weight = self.add_variable("downright", shape=[self._num_units, self._num_units])

    def call(self, inputs, state):
        current_x , current_y = findplaq_place(currentn)

        upleftvalue = init_code.plaquette_defects[current_x,current_y]
        upleftstate = tf.constant(upleftvalue, shape=[numsamples,self._num_units])

        uprightvalue = init_code.plaquette_defects[current_x,current_y+1]
        uprightstate = tf.constant(uprightvalue, shape=[numsamples,self._num_units])

        downleftvalue = init_code.plaquette_defects[current_x+1,current_y]
        downleftstate = tf.constant(downleftvalue, shape=[numsamples,self._num_units])

        downrightvalue = init_code.plaquette_defects[current_x+1,current_y+1]
        downrightstate = tf.constant(downrightvalue, shape=[numsamples,self._num_units])

        preact = tf.matmul(inputs, self._weights) + tf.matmul(state, self._recurrent_weights) + tf.matmul(upleftstate, self._upleft_weight)+ tf.matmul(uprightstate, self._upright_weight)+ tf.matmul(downleftstate, self._downleft_weight)+ tf.matmul(downrightstate, self._upleft_weight)
        output = self._activation(preact)
        new_state = output
        return output, new_state

"""## **RNN Wavefunction Class**"""

class DilatedRNNWavefunction(object):
    def __init__(self,systemsize,cell=None,activation=tf.nn.relu,units=[2],scope='DilatedRNNwavefunction'):

        self.graph=tf.Graph()
        self.scope=scope #Label of the RNN wavefunction
        self.N=systemsize #Number of sites of the 1D chain
        self.numlayers = len(units)
        dim_inputs = [2]+units[:-1]
        #Defining the neural network
        with self.graph.as_default():
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                
                #Define the RNN cell where units[n] corresponds to the number of memory units in each layer n
                self.rnn=[[cell(num_units = units[i], activation = activation,name="rnn_"+str(n)+str(i),dtype=tf.float64) for n in range(self.N)] for i in range(self.numlayers)]
                self.dense = [tf.compat.v1.layers.Dense(2,activation=tf.nn.softmax,name='wf_dense'+str(n)) for n in range(self.N)] #Define the Fully-Connected la

    def sample(self,numsamples,inputdim):
        with self.graph.as_default(): #Call the default graph, used if willing to create multiple graphs.
            samples = []
            probs = []
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):

                inputs=tf.zeros((numsamples,inputdim), dtype = tf.float64) #Feed the table b in tf.
                #Initial input to feed to the rnn

                self.inputdim=inputs.shape[1]
                self.outputdim=self.inputdim
                self.numsamples=inputs.shape[0]

                rnn_states = []

                for i in range(self.numlayers):
                    for n in range(self.N):
                        currentn = n
                        # rnn_states.append(1.0-self.rnn[i].zero_state(self.numsamples,dtype=tf.float64)) #Initialize the RNN hidden state
                        rnn_states.append(self.rnn[i][n].zero_state(self.numsamples,dtype=tf.float64)) #Initialize the RNN hidden state
                #zero state returns a zero filled tensor withs shape = (self.numsamples, num_units)
                for n in range(self.N):
                    currentn = n

                    rnn_output = inputs

                    for i in range(self.numlayers):
                        if (n-2**i)>=0:
                            rnn_output, rnn_states[i + n*self.numlayers] = self.rnn[i][n](rnn_output, rnn_states[i+((n-2**i)*self.numlayers)]) #Compute the next hidden states
                        else:
                            rnn_output, rnn_states[i + n*self.numlayers] = self.rnn[i][n](rnn_output, rnn_states[i])

                    output = self.dense[n](rnn_output)
                    probs.append(output)
                    sample_temp=tf.reshape(tf.compat.v1.multinomial(tf.math.log(output),num_samples=1),[-1,]) #Sample from the probability
                    samples.append(sample_temp)
                    inputs=tf.one_hot(sample_temp,depth=self.outputdim,dtype = tf.float64)

        probs=tf.transpose(tf.stack(values=probs,axis=2),perm=[0,2,1])
        self.samples=tf.stack(values=samples,axis=1) # (self.N, num_samples) to (num_samples, self.N): Generate self.numsamples vectors of size self.N spin containing 0 or 1
        one_hot_samples=tf.one_hot(self.samples,depth=self.inputdim, dtype = tf.float64)
        self.log_probs=tf.reduce_sum(tf.math.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=2)),axis=1)

        return self.samples,self.log_probs

    def log_probability(self,samples,inputdim):
        with self.graph.as_default():

            self.inputdim=inputdim
            self.outputdim=self.inputdim

            self.numsamples=tf.shape(samples)[0]

            inputs=tf.zeros((self.numsamples, self.inputdim), dtype=tf.float64)

            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                probs=[]

                rnn_states = []

                for i in range(self.numlayers):
                    for n in range(self.N):
                        currentn = n
                        rnn_states.append(self.rnn[i][n].zero_state(self.numsamples,dtype=tf.float64)) #Initialize the RNN hidden state
                #zero state returns a zero filled tensor withs shape = (self.numsamples, num_units)
                
                for n in range(self.N):
                    currentn = n

                    rnn_output = inputs

                    for i in range(self.numlayers):
                        if (n-2**i)>=0:
                            rnn_output, rnn_states[i + n*self.numlayers] = self.rnn[i][n](rnn_output, rnn_states[i+((n-2**i)*self.numlayers)]) #Compute the next hidden states
                        else:
                            rnn_output, rnn_states[i + n*self.numlayers] = self.rnn[i][n](rnn_output, rnn_states[i])

                    output = self.dense[n](rnn_output)
                    probs.append(output)
                    inputs=tf.reshape(tf.one_hot(tf.reshape(tf.slice(samples,begin=[np.int32(0),np.int32(n)],size=[np.int32(-1),np.int32(1)]),shape=[self.numsamples]),depth=self.outputdim,dtype = tf.float64),shape=[self.numsamples,self.inputdim])

            probs=tf.cast(tf.transpose(tf.stack(values=probs,axis=2),perm=[0,2,1]),tf.float64)
            one_hot_samples=tf.one_hot(samples,depth=self.inputdim, dtype = tf.float64)

            self.log_probs=tf.reduce_sum(tf.math.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=2)),axis=1)

            return self.log_probs

if __name__ == '__main__':
    """## **Helper Functions** """
    job_name = "Timecheck_dilated_size3"
    params = {'code': 'rotated',
              'method': 'VNA_1D_Dilated',
              'size': 3,
              'noise': 'alpha',
              'eta': 0.5,
              'alpha': 1,
              'p_sampling': 0.3,
              'droplets': 1,
              'mwpm_init': False,
              'fixed_errors': None,
              'Nc': None,
              'iters': 10,
              'conv_criteria': 'error_based',
              'SEQ': 2,
              'TOPS': 10,
              'eps': 0.01,
              'onlyshortest': True}

    # Steps is a function of code size
    params['steps'] = int(5 * params['size'] ** 5)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # stop displaying tensorflow warnings
    N = 25 #number of spins in the chain
    N_new = int(N**(0.5))
    currentn=0
    threshold=0.02
    num_units = 20 #number of memory units
    numlayers = 2 #number of layers
    numsamples = 200 #Batch size
    ndatapoints = 10
    lr = 1e-2 #learning rate
    T0 = 2 #Initial temperature
    num_warmup_steps = 10 #number of warmup steps
    num_annealing_steps = 5 #number of annealing steps
    num_equilibrium_steps = 2 #number of training steps after each annealing step
    activation_function = tf.nn.elu #activation of the RNN cell
    units=[num_units]*numlayers #list containing the number of hidden units for each layer of the RNN
    p_x = 0
    p_y = 0
    p_z = 0
    init_code = RotSurCode(N_new)
    init_code.generate_random_error(p_x=p_x, p_y=p_y, p_z=p_z)
    num_steps = num_annealing_steps*num_equilibrium_steps + num_warmup_steps #Total number of gradient steps
    # Intitializing the RNN-----------
    RNNWF =  DilatedRNNWavefunction(N,units=units,cell=CustomRNNCell, activation = activation_function) #contains the graph with the RNNs

    #Building the graph -------------------
    with tf.compat.v1.variable_scope(RNNWF.scope,reuse=tf.compat.v1.AUTO_REUSE):
        with RNNWF.graph.as_default():
            global_step = tf.Variable(0, trainable=False)
            learningrate_placeholder = tf.compat.v1.placeholder(dtype=tf.float64,shape=[])
            learningrate = tf.compat.v1.train.exponential_decay(learningrate_placeholder, global_step, 100, 1.0, staircase=True)

            #Defining the optimizer
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learningrate)

            #Defining Tensorflow placeholders
            Eloc=tf.compat.v1.placeholder(dtype=tf.float64,shape=[numsamples])
            sampleplaceholder_forgrad=tf.compat.v1.placeholder(dtype=tf.int32,shape=[numsamples,N])
            log_probs_forgrad = RNNWF.log_probability(sampleplaceholder_forgrad,inputdim=2)

            samples_placeholder=tf.compat.v1.placeholder(dtype=tf.int32,shape=(None,N))
            log_probs_tensor=RNNWF.log_probability(samples_placeholder,inputdim=2)
            samplesandprobs = RNNWF.sample(numsamples=numsamples,inputdim=2)

            T_placeholder = tf.compat.v1.placeholder(dtype=tf.float64,shape=())

            #Here we define a fake cost function that would allows to get the gradients of free energy using the tf.stop_gradient trick
            Floc = Eloc + T_placeholder*log_probs_forgrad
            cost = tf.reduce_mean(tf.multiply(log_probs_forgrad,tf.stop_gradient(Floc))) - tf.reduce_mean(log_probs_forgrad)*tf.reduce_mean(tf.stop_gradient(Floc))

            gradients, variables = zip(*optimizer.compute_gradients(cost))
            #Calculate Gradients---------------

            #Define the optimization step
            optstep=optimizer.apply_gradients(zip(gradients,variables), global_step = global_step)

            #Tensorflow saver to checkpoint
            saver=tf.compat.v1.train.Saver()

            #For initialization
            init=tf.compat.v1.global_variables_initializer()
            initialize_parameters = tf.compat.v1.initialize_all_variables()
    #----------------------------------------------------------------

    """Here we initialize the tensorflow session:"""

    #Starting Session------------
    #GPU management
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    #Start session
    sess=tf.compat.v1.Session(graph=RNNWF.graph, config=config)
    sess.run(init)
    n_ps = 20
    ps = np.linspace(0.01, 0.15, num=n_ps)
    for p_idx in range(n_ps):
        params['p_error'] = ps[p_idx]
        file_path = f'data/{job_name}_{p_idx}.xz'
        p_x = ps[p_idx]
        df = pd.DataFrame()
        df_list = []
        names = ['data_nr', 'type']
        index_params = pd.MultiIndex.from_product([[-1], np.arange(1)],
                                                  names=names)
        df_params = pd.DataFrame([[params]],
                                 index=index_params,
                                 columns=['data'])
        df = df.append(df_params)
        print('\nDataFrame opened at: ' + str(file_path))
        print("p_x=", p_x)
        for samp in range(ndatapoints):
            RNNWF = DilatedRNNWavefunction(N, units=units, cell=CustomRNNCell,
                                           activation=activation_function)  # contains the graph with the RNNs

            init_code = RotSurCode(N_new)
            init_code.generate_random_error(p_x=p_x, p_y=p_y, p_z=p_z)
            df_qubit = copy.deepcopy(init_code.qubit_matrix)
            with tf.compat.v1.variable_scope(RNNWF.scope,reuse=tf.compat.v1.AUTO_REUSE):
                with RNNWF.graph.as_default():

                  #To store data
                  meanEnergy=[]
                  varEnergy=[]
                  varFreeEnergy = []
                  meanFreeEnergy = []
                  samples = np.zeros((numsamples, N), dtype=np.int32)
                  queue_samples = np.zeros((N+1, numsamples, N), dtype = np.int32) #Array to store all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)
                  log_probs = np.zeros((N+1)*numsamples, dtype=np.float64) #Array to store the log_probs of all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)

                  T = T0 #initializing temperature

                  sess.run(initialize_parameters) #Reinitialize the parameters
                  start = time.time()
                  tempfailrate = 1
                  tempiter = 0
                  while tempfailrate > threshold:
                        if T>0:
                            T=T-0.1
                        if tempfailrate > 0.9:
                            tempiter +=1

                        if tempiter > 100:
                            break;

                        #Getting samples and log_probs from the RNN
                        samples, log_probabilities = sess.run(samplesandprobs)

                        # Estimating the local energies
                        local_energies = return_local_energies(samples, init_code)

                        #computing <H> and var(<H>)
                        meanE = np.mean(local_energies)
                        varE = np.var(local_energies)

                        #adding elements to be saved
                        meanEnergy.append(meanE)
                        varEnergy.append(varE)

                        #computing F and var(F)
                        meanF = np.mean(local_energies+T*log_probabilities)
                        varF = np.var(local_energies+T*log_probabilities)

                        #adding elements to be saved
                        meanFreeEnergy.append(meanF)
                        varFreeEnergy.append(varF)

                        #Run gradient descent step
                        print(failrate(samples, init_code.qubit_matrix))
                        tempfailrate = failrate(samples, init_code.qubit_matrix)
                        sess.run(optstep,feed_dict={Eloc:local_energies, sampleplaceholder_forgrad: samples, learningrate_placeholder: lr, T_placeholder:T})
                  samples, log_probabilities = sess.run(samplesandprobs)
                  df_eq_distr = np.zeros(4)
                  for i in range(numsamples):
                      temparray = samples[0, :].reshape(N_new, N_new)
                      init_code.update_matrix(temparray)
                      df_eq_distr[init_code.define_equivalence_class()] += 1

                  df_eq_distr = (np.divide(df_eq_distr, sum(df_eq_distr)) * 100)
                  print(df_eq_distr)
                  # Create indices for generated data
                  names = ['data_nr', 'type']
                  index_qubit = pd.MultiIndex.from_product([[samp], np.arange(1)],
                                                           names=names)
                  index_distr = pd.MultiIndex.from_product([[samp], np.arange(1) + 1], names=names)

                  # Add data to Dataframes
                  df_qubit = pd.DataFrame([[df_qubit.astype(np.uint8)]], index=index_qubit,
                                          columns=['data'])
                  df_distr = pd.DataFrame([[df_eq_distr]],
                                          index=index_distr, columns=['data'])

                  # Add dataframes to temporary list to shorten computation time

                  df_list.append(df_qubit)
                  df_list.append(df_distr)

                  # Every x iteration adds data to data file from temporary list
                  # and clears temporary list

                  if (samp+ 1) % 50 == 0:
                      df = df.append(df_list)
                      df_list.clear()
                      print('Intermediate save point reached (writing over)')
                      df.to_pickle(file_path)
        if len(df_list) > 0:
            df = df.append(df_list)
            print('\nSaving all generated data (writing over)')
            df.to_pickle(file_path)

        print('\nCompleted')