import tensorflow as tf
import tf_slim as slim
import numpy as np
import os
import time
import random
import pandas as pd
import copy
from src.rotated_surface_model import RotSurCode
from typing import Any, Optional, Union, Text, Sequence, Tuple, List

Tensor = Any

def tensordot(tf,
              a,
              b,
              axes,
              name: Optional[Text] = None) -> Tensor:
    r"""Tensor contraction of a and b along specified axes.
    Tensordot (also known as tensor contraction) sums the product of elements
    from `a` and `b` over the indices specified by `a_axes` and `b_axes`.
    The lists `a_axes` and `b_axes` specify those pairs of axes along which to
    contract the tensors. The axis `a_axes[i]` of `a` must have the same dimension
    as axis `b_axes[i]` of `b` for all `i` in `range(0, len(a_axes))`. The lists
    `a_axes` and `b_axes` must have identical length and consist of unique
    integers that specify valid axes for each of the tensors.
    This operation corresponds to `numpy.tensordot(a, b, axes)`.
    Example 1: When `a` and `b` are matrices (order 2), the case `axes = 1`
    is equivalent to matrix multiplication.
    Example 2: When `a` and `b` are matrices (order 2), the case
    `axes = [[1], [0]]` is equivalent to matrix multiplication.
    Example 3: Suppose that \\(a_{ijk}\\) and \\(b_{lmn}\\) represent two
    tensors of order 3. Then, `contract(a, b, [[0], [2]])` is the order 4 tensor
    \\(c_{jklm}\\) whose entry
    corresponding to the indices \\((j,k,l,m)\\) is given by:
    \\( c_{jklm} = \sum_i a_{ijk} b_{lmi} \\).
    In general, `order(c) = order(a) + order(b) - 2*len(axes[0])`.
    Args:
      tf: The TensorFlow module. This must be passed in instead of imported
        since we don't assume users have TensorFlow installed.
      a: `Tensor` of type `float32` or `float64`.
      b: `Tensor` with the same type as `a`.
      axes: Either a scalar `N`, or a list or an `int32` `Tensor` of shape [2, k].
        If axes is a scalar, sum over the last N axes of a and the first N axes of
        b in order. If axes is a list or `Tensor` the first and second row contain
        the set of unique integers specifying axes along which the contraction is
        computed, for `a` and `b`, respectively. The number of axes for `a` and
        `b` must be equal.
      name: A name for the operation (optional).
    Returns:
      A `Tensor` with the same type as `a`.
    Raises:
      ValueError: If the shapes of `a`, `b`, and `axes` are incompatible.
      IndexError: If the values in axes exceed the rank of the corresponding
        tensor.
    """

    def _tensordot_should_flip(contraction_axes: List[int],
                               free_axes: List[int]) -> bool:
        """Helper method to determine axis ordering.
        We minimize the average distance the indices would have to move under the
        transposition.
        Args:
          contraction_axes: The axes to be contracted.
          free_axes: The free axes.
        Returns:
          should_flip: `True` if `contraction_axes` should be moved to the left,
            `False` if they should be moved to the right.
        """
        # NOTE: This will fail if the arguments contain any Tensors.
        if contraction_axes and free_axes:
            return bool(np.mean(contraction_axes) < np.mean(free_axes))

        return False

    def _tranpose_if_necessary(tensor: Tensor, perm: List[int]) -> Tensor:
        """Like transpose(), but avoids creating a new tensor if possible.
        Although the graph optimizer should kill trivial transposes, it is best not
        to add them in the first place!
        """
        if perm == list(range(len(perm))):
            return tensor

        return tf.transpose(tensor, perm)

    def _reshape_if_necessary(tensor: Tensor,
                              new_shape: List[int]) -> Tensor:
        """Like reshape(), but avoids creating a new tensor if possible.
        Assumes shapes are both fully specified."""
        cur_shape = tensor.get_shape().as_list()
        if (len(new_shape) == len(cur_shape) and
                all(d0 == d1 for d0, d1 in zip(cur_shape, new_shape))):
            return tensor

        return tf.reshape(tensor, new_shape)

    def _tensordot_reshape(
            a: Tensor, axes: Union[Sequence[int], Tensor], is_right_term=False
    ) -> Tuple[Tensor, Union[List[int], Tensor], Optional[List[int]], bool]:
        """Helper method to perform transpose and reshape for contraction op.
        This method is helpful in reducing `math_ops.tensordot` to `math_ops.matmul`
        using `array_ops.transpose` and `array_ops.reshape`. The method takes a
        tensor and performs the correct transpose and reshape operation for a given
        set of indices. It returns the reshaped tensor as well as a list of indices
        necessary to reshape the tensor again after matrix multiplication.
        Args:
          a: `Tensor`.
          axes: List or `int32` `Tensor` of unique indices specifying valid axes of
           `a`.
          is_right_term: Whether `a` is the right (second) argument to `matmul`.
        Returns:
          A tuple `(reshaped_a, free_dims, free_dims_static, transpose_needed)`
          where `reshaped_a` is the tensor `a` reshaped to allow contraction via
          `matmul`, `free_dims` is either a list of integers or an `int32`
          `Tensor`, depending on whether the shape of a is fully specified, and
          free_dims_static is either a list of integers and None values, or None,
          representing the inferred static shape of the free dimensions.
          `transpose_needed` indicates whether `reshaped_a` must be transposed,
          or not, when calling `matmul`.
        """

        if a.get_shape().is_fully_defined() and isinstance(axes, (list, tuple)):
            shape_a = a.get_shape().as_list()
            # NOTE: This will fail if axes contains any tensors
            axes = [i if i >= 0 else i + len(shape_a) for i in axes]
            free = [i for i in range(len(shape_a)) if i not in axes]
            flipped = _tensordot_should_flip(axes, free)

            free_dims = [shape_a[i] for i in free]
            prod_free = int(np.prod([shape_a[i] for i in free]))
            prod_axes = int(np.prod([shape_a[i] for i in axes]))
            perm = axes + free if flipped else free + axes
            new_shape = [prod_axes, prod_free] if flipped else [prod_free, prod_axes]
            transposed_a = _tranpose_if_necessary(a, perm)
            reshaped_a = _reshape_if_necessary(transposed_a, new_shape)
            transpose_needed = (not flipped) if is_right_term else flipped
            return reshaped_a, free_dims, free_dims, transpose_needed

        if a.get_shape().ndims is not None and isinstance(axes, (list, tuple)):
            shape_a = a.get_shape().as_list()
            axes = [i if i >= 0 else i + len(shape_a) for i in axes]
            free = [i for i in range(len(shape_a)) if i not in axes]
            flipped = _tensordot_should_flip(axes, free)
            perm = axes + free if flipped else free + axes

            axes_dims = [shape_a[i] for i in axes]
            free_dims = [shape_a[i] for i in free]
            free_dims_static = free_dims
            axes = tf.convert_to_tensor(axes, dtype=tf.dtypes.int32, name="axes")
            free = tf.convert_to_tensor(free, dtype=tf.dtypes.int32, name="free")
            shape_a = tf.shape(a)
            transposed_a = _tranpose_if_necessary(a, perm)
        else:
            free_dims_static = None
            shape_a = tf.shape(a)
            rank_a = tf.rank(a)
            axes = tf.convert_to_tensor(axes, dtype=tf.dtypes.int32, name="axes")
            axes = tf.where(axes >= 0, axes, axes + rank_a)
            free, _ = tf.compat.v1.setdiff1d(tf.range(rank_a), axes)
            # Matmul does not accept tensors for its transpose arguments, so fall
            # back to the previous, fixed behavior.
            # NOTE(amilsted): With a suitable wrapper for `matmul` using e.g. `case`
            #   to match transpose arguments to tensor values, we could also avoid
            #   unneeded tranposes in this case at the expense of a somewhat more
            #   complicated graph. Unclear whether this would be beneficial overall.
            flipped = is_right_term
            perm = (
                tf.concat([axes, free], 0) if flipped else tf.concat([free, axes], 0))
            transposed_a = tf.transpose(a, perm)

        free_dims = tf.gather(shape_a, free)
        axes_dims = tf.gather(shape_a, axes)
        prod_free_dims = tf.reduce_prod(free_dims)
        prod_axes_dims = tf.reduce_prod(axes_dims)

        if flipped:
            new_shape = tf.stack([prod_axes_dims, prod_free_dims])
        else:
            new_shape = tf.stack([prod_free_dims, prod_axes_dims])
        reshaped_a = tf.reshape(transposed_a, new_shape)
        transpose_needed = (not flipped) if is_right_term else flipped

        return reshaped_a, free_dims, free_dims_static, transpose_needed

    def _tensordot_axes(a: Tensor, axes) -> Tuple[Any, Any]:
        """Generates two sets of contraction axes for the two tensor arguments."""
        a_shape = a.get_shape()
        if isinstance(axes, tf.compat.integral_types):
            if axes < 0:
                raise ValueError("'axes' must be at least 0.")
            if a_shape.ndims is not None:
                if axes > a_shape.ndims:
                    raise ValueError("'axes' must not be larger than the number of "
                                     "dimensions of tensor %s." % a)
                return (list(range(a_shape.ndims - axes,
                                   a_shape.ndims)), list(range(axes)))
            rank = tf.rank(a)
            return (tf.range(rank - axes, rank,
                             dtype=tf.int32), tf.range(axes, dtype=tf.int32))
        if isinstance(axes, (list, tuple)):
            if len(axes) != 2:
                raise ValueError("'axes' must be an integer or have length 2.")
            a_axes = axes[0]
            b_axes = axes[1]
            if isinstance(a_axes, tf.compat.integral_types) and \
                    isinstance(b_axes, tf.compat.integral_types):
                a_axes = [a_axes]
                b_axes = [b_axes]
            # NOTE: This fails if either a_axes and b_axes are Tensors.
            if len(a_axes) != len(b_axes):
                raise ValueError(
                    "Different number of contraction axes 'a' and 'b', %s != %s." %
                    (len(a_axes), len(b_axes)))

            # The contraction indices do not need to be permuted.
            # Sort axes to avoid unnecessary permutations of a.
            # NOTE: This fails if either a_axes and b_axes contain Tensors.
            # pylint: disable=len-as-condition
            if len(a_axes) > 0:
                a_axes, b_axes = list(zip(*sorted(zip(a_axes, b_axes))))

            return a_axes, b_axes

        axes = tf.convert_to_tensor(axes, name="axes", dtype=tf.int32)
        return axes[0], axes[1]

    with tf.compat.v1.name_scope(name, "Tensordot", [a, b, axes]) as _name:
        a = tf.convert_to_tensor(a, name="a")
        b = tf.convert_to_tensor(b, name="b")
        a_axes, b_axes = _tensordot_axes(a, axes)
        a_reshape, a_free_dims, a_free_dims_static, a_transp = _tensordot_reshape(
            a, a_axes)
        b_reshape, b_free_dims, b_free_dims_static, b_transp = _tensordot_reshape(
            b, b_axes, is_right_term=True)

        ab_matmul = tf.matmul(
            a_reshape, b_reshape, transpose_a=a_transp, transpose_b=b_transp)

        if isinstance(a_free_dims, list) and isinstance(b_free_dims, list):
            return tf.reshape(ab_matmul, a_free_dims + b_free_dims, name=_name)
        a_free_dims = tf.convert_to_tensor(a_free_dims, dtype=tf.dtypes.int32)
        b_free_dims = tf.convert_to_tensor(b_free_dims, dtype=tf.dtypes.int32)
        product = tf.reshape(
            ab_matmul, tf.concat([a_free_dims, b_free_dims], 0), name=_name)
        if a_free_dims_static is not None and b_free_dims_static is not None:
            product.set_shape(a_free_dims_static + b_free_dims_static)

        return product
def return_local_energies(samples, init_code):
        numsamples = samples.shape[0]
        Nx = samples.shape[1]
        Ny = samples.shape[2]

        N = Nx * Ny  # Total number of spins

        local_energies = np.zeros((numsamples), dtype=np.float64)
        oldcode = RotSurCode(Nx)
        copymatrix = np.copy(init_code.qubit_matrix)

        for x in range(numsamples):
            newcode = RotSurCode(Nx)
            newcode.update_matrix(samples[x, :, :])
            oldcode.update_matrix(copymatrix)
            array_iszero = 0
            if np.count_nonzero(samples[x, :, :]) == 0:
                array_iszero = 1
            test_all = 0
            for i in range(Ny + 1):
                for j in range(Nx + 1):
                    local_energies[x] += 40 * np.absolute(newcode.plaquette_defects[i, j] - oldcode.plaquette_defects[i, j]) + np.sum(samples[x,:,:])
        return local_energies


def failrate(sols, qubit_matrix):
    num_samples = sols.shape[0]
    size = sols.shape[1]  # Nx so presumes Nx=Ny
    copymatrix = np.copy(qubit_matrix)
    newcode = RotSurCode(Nx)
    failarray = np.zeros([num_samples])
    for x in range(num_samples):
        newcode.update_matrix(sols[x, :, :])
        oldcode = RotSurCode(Nx)
        oldcode.update_matrix(copymatrix)
        fails = 0
        for i in range(size + 1):
            for j in range(size + 1):
                if np.absolute(newcode.plaquette_defects[i, j] - oldcode.plaquette_defects[i, j]) != 0:
                    fails += 1
        failarray[x] = fails
    print(np.count_nonzero(failarray) / num_samples)

    return np.count_nonzero(failarray) / num_samples


def individual_failrate(sample, init_code):
    newcode = RotSurCode(Nx)
    newcode.update_matrix(sample)
    copymatrix = np.copy(init_code.qubit_matrix)
    oldcode = RotSurCode(Nx)
    oldcode.update_matrix(copymatrix)
    size = sample.shape[0]
    for i in range(size + 1):
        for j in range(size + 1):
            if np.absolute(newcode.plaquette_defects[i, j] - oldcode.plaquette_defects[i, j]) != 0:
                return False
    return True


def f_join(numList):
    s = ''.join(map(str, numList.astype(int)))
    return int(s)


def code_circ(it, Nx):
    newmatrix = np.zeros([Nx, Nx])
    if it % 2 == 0:
        newmatrix[0, 0] = 1
    if it % 4 < 2:
        newmatrix[0, 2] = 1
    if it % 8 < 4:
        newmatrix[2, 0] = 1
    if it % 16 < 8:
        newmatrix[2, 2] = 1
    return newmatrix


def findplaq_place(N):
    currentx = N % 3
    if N < 3:
        currenty = 0
    if N < 6 and N > 2:
        currenty = 1
    if N > 5:
        currenty = 2
    return currentx, currenty


"""## **The RNN cell**"""


class MDTensorizedRNNCell(tf.compat.v1.nn.rnn_cell.RNNCell):
    def __init__(self, num_units = None, activation = None, name=None, dtype = None, reuse=None):
        super(MDTensorizedRNNCell, self).__init__(_reuse=reuse, name=name)
        # save class variables
        self._num_in = 2
        self._num_units = num_units
        self._state_size = num_units
        self._output_size = num_units
        self.activation = activation

        # set up input -> hidden connection
        self.W0 = tf.compat.v1.get_variable("W0_"+name, shape=[num_units, 2*num_units, 2*self._num_in],
                                    initializer=slim.xavier_initializer(), dtype = dtype)
        self.W1 = tf.compat.v1.get_variable("W1_"+name, shape=[num_units, 2*num_units, 2*self._num_in],
                                    initializer=slim.xavier_initializer(), dtype = dtype)
        self.W2 = tf.compat.v1.get_variable("W2_"+name, shape=[num_units, 2*num_units, 2*self._num_in],
                                    initializer=slim.xavier_initializer(), dtype = dtype)
        self.W3 = tf.compat.v1.get_variable("W3_"+name, shape=[num_units, 2*num_units, 2*self._num_in],
                                   initializer=slim.xavier_initializer(), dtype = dtype)
        self.W4 = tf.compat.v1.get_variable("W4_"+name, shape=[num_units, 2*num_units, 2*self._num_in],
                                   initializer=slim.xavier_initializer(), dtype = dtype)

        self.b = tf.compat.v1.get_variable("b_"+name, shape=[num_units],
                                    initializer=slim.xavier_initializer(), dtype = dtype)

    # needed properties

    @property
    def input_size(self):
        return self._num_in # real

    @property
    def state_size(self):
        return self._state_size # real

    @property
    def output_size(self):
        return self._output_size # real

    def call(self, inputs, states):
        inputstate_mul_W0 = tf.einsum('ij,ik->ijk', tf.concat(states, 1), tf.concat(inputs,1))
        upleftvalue = init_code.plaquette_defects[current_nx,current_ny]
        inputstate_mul_W1 = tf.constant(upleftvalue, shape=[numsamples, 2*num_units, 2*self._num_in])

        uprightvalue = init_code.plaquette_defects[current_nx,current_ny+1]
        inputstate_mul_W2 = tf.constant(uprightvalue, shape=[numsamples, 2*num_units, 2*self._num_in])

        downleftvalue = init_code.plaquette_defects[current_nx+1,current_ny]
        inputstate_mul_W3 = tf.constant(downleftvalue, shape=[numsamples, 2*num_units, 2*self._num_in])

        downrightvalue = init_code.plaquette_defects[current_nx+1,current_ny+1]
        inputstate_mul_W4 = tf.constant(downrightvalue, shape=[numsamples, 2*num_units, 2*self._num_in])

        state_mul0 = tensordot(tf, inputstate_mul_W0, self.W0, axes=[[1,2],[1,2]]) # [batch_sz, num_units]
        state_mul1 = tensordot(tf, inputstate_mul_W1, self.W1, axes=[[1,2],[1,2]])
        state_mul2 = tensordot(tf, inputstate_mul_W2, self.W2, axes=[[1,2],[1,2]])
        state_mul3 = tensordot(tf, inputstate_mul_W3, self.W3, axes=[[1,2],[1,2]])
        state_mul4 = tensordot(tf, inputstate_mul_W4, self.W4, axes=[[1,2],[1,2]])


        #preact = state_mul0 + state_mul1 + state_mul2 + state_mul3 + state_mul4 + self.b
        preact = state_mul0 + state_mul1 + state_mul2 + state_mul3 + state_mul4 + self.b
        output = self.activation(preact) # [batch_sz, num_units] C

        new_state = output

        return output, new_state


"""## **RNN Wavefunction Class**"""

class MDRNNWavefunction(object):
    def __init__(self,systemsize_x = None, systemsize_y = None,cell=None,activation=None,num_units = None,scope='RNNWavefunction',seed = 111):
        self.graph=tf.Graph()
        self.scope=scope #Label of the RNN wavefunction
        self.Nx=systemsize_x
        self.Ny=systemsize_y
        self.current_nx = 0
        self.current_ny = 0

        random.seed(seed)  # `python` built-in pseudo-random generator
        np.random.seed(seed)  # numpy pseudo-random generator

        #Defining the neural network
        with self.graph.as_default():
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):

              tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator

              #Defining the 2D Tensorized RNN cell with non-weight sharing
              self.rnn=[cell(num_units = num_units, activation = activation, name="rnn_"+str(0)+str(i),dtype=tf.float64) for i in range(self.Nx*self.Ny)]
              self.dense = [tf.compat.v1.layers.Dense(2,activation=tf.nn.softmax,name='wf_dense'+str(i), dtype = tf.float64) for i in range(self.Nx*self.Ny)]
              #self.dense2 = [tf.compat.v1.layers.Dense(2,activation=tf.nn.softmax,name='wf_dense2'+str(i), dtype = tf.float64) for i in range(self.Nx*self.Ny)]

    def sample(self,numsamples,inputdim):
        with self.graph.as_default(): #Call the default graph, used if willing to create multiple graphs.
            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):

                #Initial input to feed to the lstm

                self.inputdim=inputdim
                self.outputdim=self.inputdim
                self.numsamples=numsamples


                samples=[[[] for nx in range(self.Nx)] for ny in range(self.Ny)]
                probs = [[[] for nx in range(self.Nx)] for ny in range(self.Ny)]
                rnn_states = {}
                inputs = {}

                for ny in range(self.Ny): #Loop over the boundaries for initialization
                    current_ny=ny
                    if ny%2==0:
                        nx = -1
                        # print(nx,ny)
                        rnn_states[str(nx)+str(ny)]=self.rnn[0].zero_state(self.numsamples,dtype=tf.float64)
                        inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) #Feed the table b in tf.

                    if ny%2==1:
                        nx = self.Nx
                        # print(nx,ny)
                        rnn_states[str(nx)+str(ny)]=self.rnn[0].zero_state(self.numsamples,dtype=tf.float64)
                        inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) #Feed the table b in tf.


                for nx in range(self.Nx): #Loop over the boundaries for initialization
                    current_nx=nx
                    ny = -1
                    rnn_states[str(nx)+str(ny)]=self.rnn[0].zero_state(self.numsamples,dtype=tf.float64)
                    inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) #Feed the table b in tf.

                #Making a loop over the sites with the 2DRNN
                for ny in range(self.Ny):
                    current_ny=ny

                    if ny%2 == 0:

                        for nx in range(self.Nx): #left to right
                            current_nx=nx

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn[ny*self.Nx+nx]((inputs[str(nx-1)+str(ny)],inputs[str(nx)+str(ny-1)]), (rnn_states[str(nx-1)+str(ny)],rnn_states[str(nx)+str(ny-1)]))
                           # middle=self.dense[ny*self.Nx+nx](rnn_output)
                            output = self.dense[ny*self.Nx+nx](rnn_output)
                            sample_temp=tf.reshape(tf.compat.v1.multinomial(tf.compat.v1.log(output),num_samples=1),[-1,])
                            samples[nx][ny] = sample_temp
                            probs[nx][ny] = output

                            inputs[str(nx)+str(ny)]=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float64)


                    if ny%2 == 1:

                        for nx in range(self.Nx-1,-1,-1): #right to left
                            current_nx=nx

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn[ny*self.Nx+nx]((inputs[str(nx+1)+str(ny)],inputs[str(nx)+str(ny-1)]), (rnn_states[str(nx+1)+str(ny)],rnn_states[str(nx)+str(ny-1)]))

                           # middle=self.dense[ny*self.Nx+nx](rnn_output)
                            output = self.dense[ny*self.Nx+nx](rnn_output)
                            sample_temp=tf.reshape(tf.compat.v1.multinomial(tf.compat.v1.log(output),num_samples=1),[-1,])
                            samples[nx][ny] = sample_temp
                            probs[nx][ny] = output
                            inputs[str(nx)+str(ny)]=tf.one_hot(sample_temp,depth=self.outputdim, dtype = tf.float64)


        self.samples=tf.transpose(tf.stack(values=samples,axis=0), perm = [2,0,1])
        probs=tf.transpose(tf.stack(values=probs,axis=0),perm=[2,0,1,3])
        one_hot_samples=tf.one_hot(self.samples,depth=self.inputdim, dtype = tf.float64)
        self.log_probs=tf.reduce_sum(tf.reduce_sum(tf.compat.v1.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=3)),axis=2),axis=1)

        return self.samples,self.log_probs

    def log_probability(self,samples,inputdim):
        with self.graph.as_default():

            self.inputdim=inputdim
            self.outputdim=self.inputdim

            self.numsamples=tf.shape(samples)[0]

            #Initial input to feed to the lstm
            self.outputdim=self.inputdim


            samples_=tf.transpose(samples, perm = [1,2,0])
            rnn_states = {}
            inputs = {}
            extra_inputs = {}

            for ny in range(self.Ny): #Loop over the boundaries for initialization
                current_ny=ny
                if ny%2==0:
                    nx = -1
                    rnn_states[str(nx)+str(ny)]=self.rnn[0].zero_state(self.numsamples,dtype=tf.float64)
                    inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) #Feed the table b in tf.
                    if ny==0:
                        extra_inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64)



                if ny%2==1:
                    nx = self.Nx
                    rnn_states[str(nx)+str(ny)]=self.rnn[0].zero_state(self.numsamples,dtype=tf.float64)
                    inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) #Feed the table b in tf.


            for nx in range(self.Nx): #Loop over the boundaries for initialization
                current_nx=nx
                ny = -1
                rnn_states[str(nx)+str(ny)]=self.rnn[0].zero_state(self.numsamples,dtype=tf.float64)
                inputs[str(nx)+str(ny)] = tf.zeros((self.numsamples,inputdim), dtype = tf.float64) #Feed the table b in tf.


            with tf.compat.v1.variable_scope(self.scope,reuse=tf.compat.v1.AUTO_REUSE):
                probs = [[[] for nx in range(self.Nx)] for ny in range(self.Ny)]

                #Making a loop over the sites with the 2DRNN
                for ny in range(self.Ny):
                    current_ny=ny

                    if ny%2 == 0:

                        for nx in range(self.Nx): #left to right
                            current_nx=nx

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn[ny*self.Nx+nx]((inputs[str(nx-1)+str(ny)],inputs[str(nx)+str(ny-1)]), (rnn_states[str(nx-1)+str(ny)],rnn_states[str(nx)+str(ny-1)]))

                           # middle=self.dense[ny*self.Nx+nx](rnn_output)
                            output = self.dense[ny*self.Nx+nx](rnn_output)
                            sample_temp=tf.reshape(tf.compat.v1.multinomial(tf.compat.v1.log(output),num_samples=1),[-1,])
                            probs[nx][ny] = output
                            inputs[str(nx)+str(ny)]=tf.one_hot(samples_[nx,ny],depth=self.outputdim,dtype = tf.float64)

                    if ny%2 == 1:

                        for nx in range(self.Nx-1,-1,-1): #right to left
                            current_nx=nx

                            rnn_output, rnn_states[str(nx)+str(ny)] = self.rnn[ny*self.Nx+nx]((inputs[str(nx+1)+str(ny)],inputs[str(nx)+str(ny-1)]), (rnn_states[str(nx+1)+str(ny)],rnn_states[str(nx)+str(ny-1)]))

                           # middle=self.dense[ny*self.Nx+nx](rnn_output)
                            output = self.dense[ny*self.Nx+nx](rnn_output)
                            sample_temp=tf.reshape(tf.compat.v1.multinomial(tf.compat.v1.log(output),num_samples=1),[-1,])
                            probs[nx][ny] = output
                            inputs[str(nx)+str(ny)]=tf.one_hot(samples_[nx,ny],depth=self.outputdim,dtype = tf.float64)

            probs=tf.transpose(tf.stack(values=probs,axis=0),perm=[2,0,1,3])
            one_hot_samples=tf.one_hot(samples,depth=self.inputdim, dtype = tf.float64)

            self.log_probs=tf.reduce_sum(tf.reduce_sum(tf.compat.v1.log(tf.reduce_sum(tf.multiply(probs,one_hot_samples),axis=3)),axis=2),axis=1)

            return self.log_probs


if __name__ == '__main__':
    """## **Helper Functions** """
    job_name = "Timecheck_2D_size5"
    #Rnn_test_d5 är inte d5 utan d3
    params = {'code': 'rotated',
              'method': 'VNA_1D_Dilated',
              'size': 5,
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
    Nx = 5
    Ny = 5
    N = 5
    current_nx = 0
    current_ny = 0
    threshold = 0.02
    num_units = 20  # number of memory units
    numlayers = 2  # number of layers
    numsamples = 200  # Batch size
    ndatapoints = 20
    lr = 1e-2  # learning rate
    T0 = 20  # Initial temperature
    num_warmup_steps = 10  # number of warmup steps
    num_annealing_steps = 5  # number of annealing steps
    num_equilibrium_steps = 2  # number of training steps after each annealing step
    activation_function = tf.nn.elu  # activation of the RNN cell
    units = [num_units] * numlayers  # list containing the number of hidden units for each layer of the RNN
    p_x = 0
    p_y = 0
    p_z = 0
    init_code = RotSurCode(N)
    init_code.generate_random_error(p_x=p_x, p_y=p_y, p_z=p_z)
    num_steps = num_annealing_steps * num_equilibrium_steps + num_warmup_steps  # Total number of gradient steps
    # Intitializing the RNN-----------
    MDRNNWF = MDRNNWavefunction(systemsize_x=Nx,systemsize_y=Ny, num_units=num_units, cell=MDTensorizedRNNCell,
                                   activation=activation_function)  # contains the graph with the RNNs

    # Building the graph -------------------
    with tf.compat.v1.variable_scope(MDRNNWF.scope, reuse=tf.compat.v1.AUTO_REUSE):
        with MDRNNWF.graph.as_default():
            global_step = tf.Variable(0, trainable=False)
            learningrate_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=[])
            learningrate = tf.compat.v1.train.exponential_decay(learningrate_placeholder, global_step, 100, 1.0,
                                                                staircase=True)

            # Defining the optimizer
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learningrate)

            # Defining Tensorflow placeholders
            Eloc = tf.compat.v1.placeholder(dtype=tf.float64, shape=[numsamples])
            sampleplaceholder_forgrad = tf.compat.v1.placeholder(dtype=tf.int32, shape=[numsamples, Nx, Ny])
            log_probs_forgrad = MDRNNWF.log_probability(sampleplaceholder_forgrad, inputdim=2)

            samples_placeholder = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None, Nx, Ny))
            log_probs_tensor = MDRNNWF.log_probability(samples_placeholder, inputdim=2)
            samplesandprobs = MDRNNWF.sample(numsamples=numsamples, inputdim=2)

            T_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=())

            # Here we define a fake cost function that would allows to get the gradients of free energy using the tf.stop_gradient trick
            Floc = Eloc + T_placeholder * log_probs_forgrad
            cost = tf.reduce_mean(tf.multiply(log_probs_forgrad, tf.stop_gradient(Floc))) - tf.reduce_mean(
                log_probs_forgrad) * tf.reduce_mean(tf.stop_gradient(Floc))

            gradients, variables = zip(*optimizer.compute_gradients(cost))
            # Calculate Gradients---------------

            # Define the optimization step
            optstep = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

            # Tensorflow saver to checkpoint
            saver = tf.compat.v1.train.Saver()

            # For initialization
            init = tf.compat.v1.global_variables_initializer()
            initialize_parameters = tf.compat.v1.initialize_all_variables()
    # ----------------------------------------------------------------
    # Starting Session------------
    # GPU management
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    # Start session
    sess = tf.compat.v1.Session(graph=MDRNNWF.graph, config=config)
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

            init_code = RotSurCode(N)
            init_code.generate_random_error(p_x=p_x, p_y=p_y, p_z=p_z)
            df_qubit = copy.deepcopy(init_code.qubit_matrix)
            with tf.compat.v1.variable_scope(MDRNNWF.scope, reuse=tf.compat.v1.AUTO_REUSE):
                with MDRNNWF.graph.as_default():

                    # To store data
                    meanEnergy = []
                    varEnergy = []
                    varFreeEnergy = []
                    meanFreeEnergy = []
                    samples = np.zeros((numsamples, N), dtype=np.int32)
                    queue_samples = np.zeros((N + 1, numsamples, N),
                                             dtype=np.int32)  # Array to store all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)
                    log_probs = np.zeros((N + 1) * numsamples,
                                         dtype=np.float64)  # Array to store the log_probs of all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)

                    T = T0  # initializing temperature

                    sess.run(initialize_parameters)  # Reinitialize the parameters
                    start = time.time()
                    tempfailrate = 1
                    tempiter = 0
                    while tempfailrate > threshold:
                        if T > 0:
                            T = T - 1
                        if T == 0 and tempfailrate > 0.9:
                            tempiter += 1

                        if tempiter > 200:
                            break;

                        # Getting samples and log_probs from the RNN
                        samples, log_probabilities = sess.run(samplesandprobs)

                        # Estimating the local energies
                        local_energies = return_local_energies(samples, init_code)

                        # computing <H> and var(<H>)
                        meanE = np.mean(local_energies)
                        varE = np.var(local_energies)

                        # adding elements to be saved
                        meanEnergy.append(meanE)
                        varEnergy.append(varE)

                        # computing F and var(F)
                        meanF = np.mean(local_energies + T * log_probabilities)
                        varF = np.var(local_energies + T * log_probabilities)

                        # adding elements to be saved
                        meanFreeEnergy.append(meanF)
                        varFreeEnergy.append(varF)

                        # Run gradient descent step
                        print(failrate(samples, init_code.qubit_matrix))
                        tempfailrate = failrate(samples, init_code.qubit_matrix)
                        sess.run(optstep, feed_dict={Eloc: local_energies, sampleplaceholder_forgrad: samples,
                                                     learningrate_placeholder: lr, T_placeholder: T})
                    samples, log_probabilities = sess.run(samplesandprobs)
                    df_eq_distr = np.zeros(4)
                    for i in range(numsamples):
                        temparray = samples[i,:,:]
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

                    if (samp + 1) % 50 == 0:
                        df = df.append(df_list)
                        df_list.clear()
                        print('Intermediate save point reached (writing over)')
                        df.to_pickle(file_path)
        if len(df_list) > 0:
            df = df.append(df_list)
            print('\nSaving all generated data (writing over)')
            df.to_pickle(file_path)

        print('\nCompleted')