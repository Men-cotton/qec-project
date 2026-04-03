import numpy as np
from random import random
from numba import njit
import seaborn as sns
import random as rand
import matplotlib.pyplot as plt
sns.set_style("darkgrid")

class RotSurCode():
    nbr_eq_classes = 4

    def __init__(self, size):
        self.system_size = size
        self.qubit_matrix = np.zeros((self.system_size, self.system_size), dtype=np.uint8)
        self.plaquette_defects = np.zeros((size + 1, size + 1))

    def generate_random_error(self, p_x, p_y, p_z):
        size = self.system_size
        for i in range(size):
            for j in range(size):
                q = 0
                r = rand.random()
                if r < p_z:
                    q = 3
                if p_z < r < (p_z + p_x):
                    q = 1
                if (p_z + p_x) < r < (p_z + p_x + p_y):
                    q = 2
                self.qubit_matrix[i, j] = q
        self.syndrome()


    def chain_lengths(self):
        nx = np.count_nonzero(self.qubit_matrix[:, :] == 1)
        ny = np.count_nonzero(self.qubit_matrix[:, :] == 2)
        nz = np.count_nonzero(self.qubit_matrix[:, :] == 3)
        return nx, ny, nz

    def count_errors(self):
        return _count_errors(self.qubit_matrix)

    def apply_logical(self, operator: int, X_pos=0, Z_pos=0):
        return _apply_logical(self.qubit_matrix, operator, X_pos, Z_pos)

    def apply_stabilizer(self, row: int, col: int, operator: int):
        return _apply_stabilizer(self.qubit_matrix, row, col, operator)

    def apply_random_logical(self):
        return _apply_random_logical(self.qubit_matrix)

    def apply_random_stabilizer(self):
        return _apply_random_stabilizer(self.qubit_matrix)

    def apply_stabilizers_uniform(self, p=0.5):
        return _apply_stabilizers_uniform(self.qubit_matrix, p)

    def define_equivalence_class(self):
        return _define_equivalence_class(self.qubit_matrix)
    
    def to_class(self, eq):
        eq_class = self.define_equivalence_class()
        op = eq_class ^ eq
        return self.apply_logical(op)[0]

    def update_matrix(self, newmatrix):
        size = self.qubit_matrix.shape[1]
        for i in range(size):
            for j in range(size):
                self.qubit_matrix[i,j] = newmatrix[i,j]
        self.syndrome()

    def syndrome(self):
        size = self.qubit_matrix.shape[1]
        qubit_matrix = self.qubit_matrix
        for i in range(size-1):
            for j in range(size-1):
                self.plaquette_defects[i+1, j+1] = _find_syndrome(qubit_matrix, i, j, 1)
        for i in range(int((size - 1)/2)):
            for j in range(4):
                row = 0
                col = 0
                if j == 0:
                    row = 0
                    col = 2 * i + 2
                elif j == 1:
                    row = 2 * i + 2
                    col = size
                elif j == 2:
                    row = size
                    col = 2 * i + 1
                elif j == 3:
                    row = 2 * i + 1
                    col = 0
                self.plaquette_defects[row, col] = _find_syndrome(qubit_matrix, i, j, 3)

    def plot(self, title):
        system_size = self.system_size
        xLine = np.linspace(0, system_size - 1, system_size)
        a = range(system_size)
        X, Y = np.meshgrid(a, a)
        XLine, YLine = np.meshgrid(a, xLine)
        plaquette_defect_coordinates = np.where(self.plaquette_defects)

        x_error = np.where(self.qubit_matrix[:, :] == 1)
        y_error = np.where(self.qubit_matrix[:, :] == 2)
        z_error = np.where(self.qubit_matrix[:, :] == 3)

        def generate_semicircle(center_x, center_y, radius, stepsize=0.1):
            x = np.arange(center_x, center_x + radius + stepsize, stepsize)
            y = np.sqrt(radius ** 2 - x ** 2)
            x = np.concatenate([x, x[::-1]])
            y = np.concatenate([y, -y[::-1]])
            return x, y + center_y

        markersize_qubit = 15
        markersize_excitation = 7
        markersize_symbols = 7
        linewidth = 2

        # Plot grid lines
        ax = plt.subplot(111)

        x, y = generate_semicircle(0, 1, 0.5, 0.01)

        for i in range(int((system_size - 1) / 2)):
            ax.plot(y + 0.5 + i * 2, x + system_size - 1, color='black', linewidth=linewidth)
            ax.plot(-y + 1.5 + 2 * i, -x, color='black', linewidth=linewidth)
            ax.plot(x + system_size - 1, y - 0.5 + i * 2, color='black', linewidth=linewidth)
            ax.plot(-x, -y + 0.5 + system_size - 1 - 2 * i, color='black', linewidth=linewidth)

        ax.plot(XLine, YLine, 'black', linewidth=linewidth)
        ax.plot(YLine, XLine, 'black', linewidth=linewidth)

        ax.plot(X, Y, 'o', color='black', markerfacecolor='white', markersize=markersize_qubit + 1)

        ax.plot(x_error[1], system_size - 1 - x_error[0], 'o', color='blue', markersize=markersize_symbols, marker=r'$X$')
        ax.plot(y_error[1], system_size - 1 - y_error[0], 'o', color='blue', markersize=markersize_symbols, marker=r'$Y$')
        ax.plot(z_error[1], system_size - 1 - z_error[0], 'o', color='blue', markersize=markersize_symbols, marker=r'$Z$')

        for i in range(len(plaquette_defect_coordinates[1])):
            if plaquette_defect_coordinates[1][i] == 0:
                ax.plot(plaquette_defect_coordinates[1][i] - 0.5 + 0.25, system_size - plaquette_defect_coordinates[0][i] - 0.5, 'o', color='red', label="flux", markersize=markersize_excitation)
            elif plaquette_defect_coordinates[0][i] == 0:
                ax.plot(plaquette_defect_coordinates[1][i] - 0.5, system_size - plaquette_defect_coordinates[0][i] - 0.5 - 0.25, 'o', color='red', label="flux", markersize=markersize_excitation)
            elif plaquette_defect_coordinates[1][i] == system_size:
                ax.plot(plaquette_defect_coordinates[1][i] - 0.5 - 0.25, system_size - plaquette_defect_coordinates[0][i] - 0.5, 'o', color='red', label="flux", markersize=markersize_excitation)
            elif plaquette_defect_coordinates[0][i] == system_size:
                ax.plot(plaquette_defect_coordinates[1][i] - 0.5, system_size - plaquette_defect_coordinates[0][i] - 0.5 + 0.25, 'o', color='red', label="flux", markersize=markersize_excitation)
            else:
                ax.plot(plaquette_defect_coordinates[1][i] - 0.5, system_size - plaquette_defect_coordinates[0][i] - 0.5, 'o', color='red', label="flux", markersize=markersize_excitation)

        # ax.plot(plaquette_defect_coordinates[1] - 0.5, system_size - plaquette_defect_coordinates[0] - 0.5, 'o', color='red', label="flux", markersize=markersize_excitation)
        ax.axis('off')

        plt.axis('equal')
        #plt.show()
        plt.savefig('plots/graph_'+str(title)+'.png')
        # plt.close()


@njit('(uint8[:,:],)')
def _count_errors(qubit_matrix):
    return np.count_nonzero(qubit_matrix)


@njit('(uint8[:,:], int64, int64, int64)')
def _find_syndrome(qubit_matrix, row: int, col: int, operator: int):

    def flip(a):
        if a == 0:
            return 1
        elif a == 1:
            return 0

    size = qubit_matrix.shape[1]
    result_qubit_matrix = np.copy(qubit_matrix)
    defect = 0
    op = 0

    if operator == 1:  # full
        qarray = [[0 + row, 0 + col], [0 + row, 1 + col], [1 + row, 0 + col], [1 + row, 1 + col]]
        if row % 2 == 0:
            if col % 2 == 0:
                op = 1
            else:
                op = 3
        else:
            if col % 2 == 0:
                op = 3
            else:
                op = 1
    elif operator == 3:  # half
        if col == 0:
            op = 1
            qarray = [[0, row*2 + 1], [0, row*2 + 2]]
        elif col == 1:
            op = 3
            qarray = [[row*2 + 1, size - 1], [row*2 + 2, size - 1]]
        elif col == 2:
            op = 1
            qarray = [[size - 1, row*2], [size - 1, row*2 + 1]]
        elif col == 3:
            op = 3
            qarray = [[row*2, 0], [row*2 + 1, 0]]

    for i in qarray:
        old_qubit = result_qubit_matrix[i[0], i[1]]
        if old_qubit != 0 and old_qubit != op:
            defect = flip(defect)

    return defect


@njit('(uint8[:,:], int64, int64, int64)')  # Z-biased noise
def _apply_logical(qubit_matrix, operator: int, X_pos=0, Z_pos=0):
    
    result_qubit_matrix = np.copy(qubit_matrix)

    # List to store how errors redestribute when logical is applied
    n_eq = [0, 0, 0, 0]

    if operator == 0:
        return result_qubit_matrix #, (0, 0, 0)
    
    size = qubit_matrix.shape[0]

    do_X = (operator == 1 or operator == 2)
    do_Z = (operator == 3 or operator == 2)

    if do_X:
        for i in range(size):
            old_qubit = result_qubit_matrix[i, X_pos]
            new_qubit = 1 ^ old_qubit
            result_qubit_matrix[i, X_pos] = new_qubit
            
            n_eq[old_qubit] -= 1
            n_eq[new_qubit] += 1
    if do_Z:
        for i in range(size):
            old_qubit = result_qubit_matrix[Z_pos, i]
            new_qubit = 3 ^ old_qubit
            result_qubit_matrix[Z_pos, i] = new_qubit
            
            n_eq[old_qubit] -= 1
            n_eq[new_qubit] += 1

    return result_qubit_matrix #, (n_eq[1], n_eq[2], n_eq[3])


@njit('(uint8[:,:],)')
def _apply_random_logical(qubit_matrix):
    size = qubit_matrix.shape[0]

    op = int(random() * 4)

    if op == 1 or op == 2:
        X_pos = int(random() * size)
    else:
        X_pos = 0
    if op == 3 or op == 2:
        Z_pos = int(random() * size)
    else:
        Z_pos = 0

    return _apply_logical(qubit_matrix, op, X_pos, Z_pos)


@njit('(uint8[:,:], int64, int64, int64)')
def _apply_stabilizer(qubit_matrix, row: int, col: int, operator: int):

    size = qubit_matrix.shape[0]
    result_qubit_matrix = np.copy(qubit_matrix)

    # List to store how errors redestribute when stabilizer is applied
    n_eq = [0, 0, 0, 0]
    
    op = 0

    if operator == 1:  # full
        qarray = [[0 + row, 0 + col], [0 + row, 1 + col], [1 + row, 0 + col], [1 + row, 1 + col]]
        if row % 2 == 0:
            if col % 2 == 0:
                op = 1
            else:
                op = 3
        else:
            if col % 2 == 0:
                op = 3
            else:
                op = 1
    elif operator == 3:  # half
        if col == 0:
            op = 1
            qarray = [[0, row*2 + 1], [0, row*2 + 2]]
        elif col == 1:
            op = 3
            qarray = [[row*2 + 1, size - 1], [row*2 + 2, size - 1]]
        elif col == 2:
            op = 1
            qarray = [[size - 1, row*2], [size - 1, row*2 + 1]]
        elif col == 3:
            op = 3
            qarray = [[row*2, 0], [row*2 + 1, 0]]

    for i in qarray:
        old_qubit = result_qubit_matrix[i[0], i[1]]
        new_qubit = op ^ old_qubit
        result_qubit_matrix[i[0], i[1]] = new_qubit
        
        n_eq[old_qubit] -= 1
        n_eq[new_qubit] += 1

    return result_qubit_matrix #, (n_eq[1], n_eq[2], n_eq[3])


@njit('(uint8[:,:],)')
def _apply_random_stabilizer(qubit_matrix):
    size = qubit_matrix.shape[0]
    rows = int((size-1)*random())
    cols = int((size-1)*random())
    rows2 = int(((size - 1)/2) * random())
    cols2 = int(4 * random())
    phalf = (size**2 - (size-1)**2 - 1)/(size**2-1)
    if rand.random() > phalf:
        # operator = 1 = full stabilizer
        return _apply_stabilizer(qubit_matrix, rows, cols, 1)
    else:
        # operator = 3 = half stabilizer
        return _apply_stabilizer(qubit_matrix, rows2, cols2, 3)


@njit('(uint8[:,:],)')
def _define_equivalence_class(qubit_matrix):

    x_errors = np.count_nonzero(qubit_matrix[0, :] == 1)
    x_errors += np.count_nonzero(qubit_matrix[0, :] == 2)

    z_errors = np.count_nonzero(qubit_matrix[:, 0] == 3)
    z_errors += np.count_nonzero(qubit_matrix[:, 0] == 2)

    if x_errors % 2 == 0:
        if z_errors % 2 == 0:
            return 0
        else:
            return 3
    else:
        if z_errors % 2 == 0:
            return 1
        else:
            return 2


def _apply_stabilizers_uniform(qubit_matrix, p=0.5):
    size = qubit_matrix.shape[0]
    result_qubit_matrix = np.copy(qubit_matrix)

    # Apply full stabilizers
    random_stabilizers = np.random.rand(size-1, size-1)
    random_stabilizers = np.less(random_stabilizers, p)

    it = np.nditer(random_stabilizers, flags=['multi_index'])
    while not it.finished:
        if it[0]:
            row, col = it.multi_index
            result_qubit_matrix = _apply_stabilizer(result_qubit_matrix, row, col, 1)
        it.iternext()

    # Apply half stabilizers
    random_stabilizers = np.random.rand(int((size - 1)/2), 4)
    random_stabilizers = np.less(random_stabilizers, p)
    it = np.nditer(random_stabilizers, flags=['multi_index'])
    while not it.finished:
        if it[0]:
            row, col = it.multi_index
            result_qubit_matrix = _apply_stabilizer(result_qubit_matrix, row, col, 3)
        it.iternext()

    return result_qubit_matrix

import tensorflow as tf
import tf_slim as slim
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #stop displaying tensorflow warnings
import numpy as np
import os
import time
import random
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['axes.labelsize']  = 20
rcParams['font.serif']      = ['Computer Modern']
rcParams['font.size']       = 10
rcParams['legend.fontsize'] = 20
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20

"""## **Seeding for reproducibility purposes**"""

#Seeding for reproducibility purposes
seed = 132 #seeding for reproducibility purposes
tf.compat.v1.reset_default_graph()
random.seed(seed)  # `python` built-in pseudo-random generator
np.random.seed(seed)  # numpy pseudo-random generator
tf.compat.v1.set_random_seed(seed)  # tensorflow pseudo-random generator

"""## **Helper Functions**"""

def canbehalf(i,j, size): #unused in this version
    allowed_i = np.arange(0,int((size-1)/2))
    allowed_j = np.arange(0,4)
    if i in allowed_i and j in allowed_j:
      return True
    else: 
      return False

def Ising2D_local_energies(samples, testcode):

    numsamples = samples.shape[0]
    Nx = samples.shape[1]
    Ny = samples.shape[2]

    N = Nx*Ny #Total number of spins

    local_energies = np.zeros((numsamples), dtype = np.float64)
    oldcode = RotSurCode(Nx)
    copymatrix = np.copy(testcode.qubit_matrix)

    for x in range(numsamples):
        newcode = RotSurCode(Nx)
        newcode.update_matrix(samples[x,:,:])
        oldcode.update_matrix(copymatrix)
        for i in range(Ny+1):
            for j in range(Nx+1):
                local_energies[x] += 100*np.absolute(newcode.plaquette_defects[i,j]-oldcode.plaquette_defects[i,j])+np.sum(samples[x,:,:])
    return local_energies


from typing import Any, Optional, Union, Text, Sequence, Tuple, List

Tensor = Any

def tensordot(tf,
              a,
              b,
              axes,
              name: Optional[Text] = None) -> Tensor:

  def _tensordot_should_flip(contraction_axes: List[int],
                              free_axes: List[int]) -> bool:
    # NOTE: This will fail if the arguments contain any Tensors.
    if contraction_axes and free_axes:
      return bool(np.mean(contraction_axes) < np.mean(free_axes))

    return False

  def _tranpose_if_necessary(tensor: Tensor, perm: List[int]) -> Tensor:
    if perm == list(range(len(perm))):
      return tensor

    return tf.transpose(tensor, perm)

  def _reshape_if_necessary(tensor: Tensor,
                            new_shape: List[int]) -> Tensor:
    cur_shape = tensor.get_shape().as_list()
    if (len(new_shape) == len(cur_shape) and
        all(d0 == d1 for d0, d1 in zip(cur_shape, new_shape))):
      return tensor

    return tf.reshape(tensor, new_shape)

  def _tensordot_reshape(
      a: Tensor, axes: Union[Sequence[int], Tensor], is_right_term=False
  ) -> Tuple[Tensor, Union[List[int], Tensor], Optional[List[int]], bool]:

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
        upleftvalue = testcode.plaquette_defects[current_nx,current_ny]
        inputstate_mul_W1 = tf.constant(upleftvalue, shape=[numsamples, 2*num_units, 2*self._num_in])

        uprightvalue = testcode.plaquette_defects[current_nx,current_ny+1]
        inputstate_mul_W2 = tf.constant(uprightvalue, shape=[numsamples, 2*num_units, 2*self._num_in])

        downleftvalue = testcode.plaquette_defects[current_nx+1,current_ny]
        inputstate_mul_W3 = tf.constant(downleftvalue, shape=[numsamples, 2*num_units, 2*self._num_in])

        downrightvalue = testcode.plaquette_defects[current_nx+1,current_ny+1]
        inputstate_mul_W4 = tf.constant(downrightvalue, shape=[numsamples, 2*num_units, 2*self._num_in])

        state_mul0 = tensordot(tf, inputstate_mul_W0, self.W0, axes=[[1,2],[1,2]]) # [batch_sz, num_units]
        state_mul1 = tensordot(tf, inputstate_mul_W1, self.W1, axes=[[1,2],[1,2]])
        state_mul2 = tensordot(tf, inputstate_mul_W2, self.W2, axes=[[1,2],[1,2]])
        state_mul3 = tensordot(tf, inputstate_mul_W3, self.W3, axes=[[1,2],[1,2]])
        state_mul4 = tensordot(tf, inputstate_mul_W4, self.W4, axes=[[1,2],[1,2]])

        preact = state_mul0 + state_mul1 + state_mul2 + state_mul3 + state_mul4 + self.b
        output = self.activation(preact)

        new_state = output

        return output, new_state

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


Nx = 3 #x-size
Ny = 3 #y-size
N = Nx*Ny #total number of sites
num_units = 100 #number of memory units
numsamples = 500 #number of samples used for training
lr = 1e-3 #learning rate
T0 = 500 #Initial temperature
num_warmup_steps = 0 #number of warmup steps
num_annealing_steps = 200 #number of annealing steps
num_equilibrium_steps = 5 #number of training steps after each annealing step
activation_function = tf.nn.elu #non-linear activation function for the 2D Tensorized RNN cell

p_x=0.10
p_y=p_z=0
testcode = RotSurCode(Nx)
testcode.generate_random_error(p_x=p_x, p_y=p_y, p_z=p_z)
print("Initial_temperature =", T0)
print('Seed = ', seed)

current_nx = 0
current_ny = 0

num_steps = num_annealing_steps*num_equilibrium_steps + num_warmup_steps

print("\nNumber of annealing steps = {0}".format(num_annealing_steps))
print("Number of training steps = {0}".format(num_steps))

# Intitializing the RNN (with only one layer)-----------
MDRNNWF = MDRNNWavefunction(systemsize_x = Nx, systemsize_y = Ny ,num_units = num_units,cell=MDTensorizedRNNCell, activation = activation_function, seed = seed) #contains the graph with the RNNs

##Building the graph -------------------
with tf.compat.v1.variable_scope(MDRNNWF.scope,reuse=tf.compat.v1.AUTO_REUSE):
    with MDRNNWF.graph.as_default():

        global_step = tf.Variable(0, trainable=False)
        learningrate_placeholder = tf.compat.v1.placeholder(dtype=tf.float64,shape=[])
        learningrate = tf.compat.v1.train.exponential_decay(learningrate_placeholder, global_step, 100, 1.0, staircase=True)
      
        #Defining the optimizer
        optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learningrate)

        #Defining Tensorflow placeholders
        Eloc=tf.compat.v1.placeholder(dtype=tf.float64,shape=[numsamples])
        sampleplaceholder_forgrad=tf.compat.v1.placeholder(dtype=tf.int32,shape=[numsamples,Nx,Ny])
        log_probs_forgrad = MDRNNWF.log_probability(sampleplaceholder_forgrad,inputdim=2)

        samples_placeholder=tf.compat.v1.placeholder(dtype=tf.int32,shape=(None,Nx, Ny))
        log_probs_tensor=MDRNNWF.log_probability(samples_placeholder,inputdim=2)
        samplesandprobs = MDRNNWF.sample(numsamples=numsamples,inputdim=2)

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

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

sess=tf.compat.v1.Session(graph=MDRNNWF.graph, config=config)
sess.run(init)


def failrate(sols,testcode):
  num_samples = sols.shape[0]
  size = sols.shape[1] #Nx so presumes Nx=Ny
  copymatrix=np.copy(testcode.qubit_matrix)
  newcode = RotSurCode(Nx)
  failarray = np.zeros([num_samples])
  for x in range(num_samples):
      newcode.update_matrix(sols[x,:,:])
      oldcode = RotSurCode(Nx)
      oldcode.update_matrix(copymatrix)
      fails = 0
      for i in range(size+1):
          for j in range(size+1):
              if np.absolute(newcode.plaquette_defects[i,j]-oldcode.plaquette_defects[i,j])!=0:
                  fails+=1
      failarray[x]=fails
  print(np.count_nonzero(failarray)/num_samples)
  
  return np.count_nonzero(failarray)/num_samples

def individual_failrate(sample,testcode):
      newcode = RotSurCode(Nx)
      newcode.update_matrix(sample)
      copymatrix=np.copy(testcode.qubit_matrix)
      oldcode = RotSurCode(Nx)
      oldcode.update_matrix(copymatrix)
      size = sample.shape[0]
      for i in range(size+1):
          for j in range(size+1):
              if np.absolute(newcode.plaquette_defects[i,j]-oldcode.plaquette_defects[i,j])!=0:
                  return False
      return True


def magic(numList):

    s = ''.join(map(str, numList.astype(int)))
    return int(s)

def code_circ(it,Nx):
    newmatrix = np.zeros([Nx,Nx])
    if it%20==0:
      newmatrix[0,0]=1
    if it%4<2:
      newmatrix[0,2]=1
    if it%8<4:
      newmatrix[2,0]=1
    if it%16<8:
      newmatrix[2,2]=1
    return newmatrix

with tf.compat.v1.variable_scope(MDRNNWF.scope,reuse=tf.compat.v1.AUTO_REUSE):
    with MDRNNWF.graph.as_default():

      #To store data
      meanEnergy=[]
      varEnergy=[]
      varFreeEnergy = []
      meanFreeEnergy = []
      fail_array_large = []
      defect_array = []
      defect_array2 = []
      samples = np.ones((numsamples, Nx, Ny), dtype=np.int32)
      queue_samples = np.zeros((N+1, numsamples, Nx, Ny), dtype = np.int32) #Array to store all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)
      log_probs = np.zeros((N+1)*numsamples, dtype=np.float64) #Array to store the log_probs of all the diagonal and non diagonal matrix elements (We create it here for memory efficiency as we do not want to allocate it at each training step)

      T = T0 #initializing temperature
      sess.run(initialize_parameters) #Reinitialize the parameters

      start = time.time()
      for it in range(len(meanEnergy),num_steps+100):

            # Create list of all defects trained on
            #testcodearray = code_circ(it,3)
            #testcode.update_matrix(testcodearray)

            #testcode = RotSurCode(Nx)
            #testcode.generate_random_error(p_x=p_x,p_y=p_y,p_z=p_z)
            tempflat = testcode.plaquette_defects.flatten()
            plac_defects_asint = magic(tempflat)
            if plac_defects_asint not in defect_array:
                print("New defect", len(defect_array))
                defect_array.append(plac_defects_asint)
                defect_array2.append(testcode.plaquette_defects)
            
            #Annealing
            if T>0:
              if it>=num_warmup_steps and  it <= num_annealing_steps*num_equilibrium_steps + num_warmup_steps and it % num_equilibrium_steps == 0:
                annealing_step = (it-num_warmup_steps)/num_equilibrium_steps
                T = T0*(1-annealing_step/num_annealing_steps)

              #Showing current status after that the annealing starts
              if it%(num_equilibrium_steps*10)==0:
                if it <= num_annealing_steps*num_equilibrium_steps + num_warmup_steps and it>=num_warmup_steps:
                    annealing_step = (it-num_warmup_steps)/num_equilibrium_steps
                    print("\nAnnealing step: {0}/{1}".format(annealing_step,num_annealing_steps))

            samples, log_probabilities = sess.run(samplesandprobs)

            # Estimating the local energies
            local_energies = Ising2D_local_energies(samples, testcode)

            meanE = np.mean(local_energies)
            varE = np.var(local_energies)

            #adding elements to be saved
            meanEnergy.append(meanE)
            varEnergy.append(varE)

            meanF = np.mean(local_energies+T*log_probabilities)
            varF = np.var(local_energies+T*log_probabilities)
            
            meanFreeEnergy.append(meanF)
            varFreeEnergy.append(varF)
            #Run gradient descent step
            failratetemp = failrate(samples,testcode)
            fail_array_large.append(failratetemp)
            sess.run(optstep,feed_dict={Eloc:local_energies, sampleplaceholder_forgrad: samples, learningrate_placeholder: lr, T_placeholder:T})
            
            if it%(num_equilibrium_steps)==0:
                print('mean(E): {0}, p_x: {1}, failrate {2}, #samples {3}, #Training step {4}'.format(meanE,p_x,failratetemp,varF,numsamples, it))
                print("Temperature: ", T)
                print("Elapsed time is =", time.time()-start, " seconds")
                print('\n\n')
#----------------------------
def calc_c(samples):
    numsamples = samples.shape[0]
    Nx = samples.shape[1]
    Ny = samples.shape[2]
    c_array = np.zeros([numsamples]) #1 if even
    for x in range(numsamples):
        c_count = 0
        for i in range(Nx):
            if samples[x,i,0]==1:
                c_count +=1
        if c_count%2==0:
            c_array[x]=1
        
    return np.sum(c_array)/numsamples

fig = plt.figure(figsize=(4, 3))

ax1 = plt.subplot()
l1, = ax1.plot(fail_array_large, color='red')
ax2 = ax1.twinx()
l2, = ax2.plot(meanEnergy, color='blue')
plt.legend([l1, l2], ["Sample failure rate", "Mean energy"], fontsize = 10)
ax1.set_ylabel("Sample failure rate, $p_f$", fontsize = 20)
ax2.set_ylabel("Mean energy, $<e>$", fontsize = 20)

plt.xlabel("Training step n", fontsize = 20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('Training progress for a 2D RNN on d=3', fontsize = 20)
plt.show()