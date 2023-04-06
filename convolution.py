import numpy as np
from layer import Layer
from scipy import signal
from utility import clip_gradient_by_norm

"""
Theory
I => Input matrix  (i1,i2)
K => Kernel matrix (k,k)
Y => Output matrix (y1,y2)

Cross correlation =>  I ★ K 
Convolution => I ❊ K 
By   I ❈ K =  I ★ rot_180(K)  => This means that convolution is the similar operation as cross correlation but need to rotate the kernel matrix for 180 degree.
Then the output Y = I ❈ K = I ★ rot_180(K)

As all of these matrices are square matrices we can calculate the shape as
y = i_1/2 -k +1   or the output has a shape of (i1-k +1 , i2-k +1)   *The shape is for "valid" cross correlation (kernel stop when it hits the border)

Note : Number of element in Y represent the number of convolution / cross-correlated process

Principle of convolutional layer
1) Input Tensor I contains 3 matrices => X_1 , X_2 , X_3   Each matrix has the same shape (square) => the deep of the tensor represents color channel
2) Kernel Tensor K contains 3 matrices => K_1 , K_2 , K_3  (Note that it has the same deep as the Input tensor)
    - In each layer, you can have #k kernel tensor <= Multiple kernel tensors
    - Each element in the kernel tensor is a trainable parameters
3) Bias matrix B => Each bias matrix associate with the each kernel tensor
4) Output tensor Y contains #k matrices => Y1 , Y2

How the computation happened in the convolution layer?
i => kernel index (Index associate with each kernel tensor)
j => input index (Index associate with each input matrices)
Then  Y_i  =  B_i + ∑ X_j ★ K_ij   (i = 0, 1, 2,..,d)    
 
(In simple words , Each kernel matrix will cross correlated with the each correspond input matrix (EX. Red matrix match with red kernel matrix)
In each convolution process (happened parallel in all three matrices) then the result will be the sum of those add with the bias.)  

Idea :
1) Loop for i (The index of kernel tensor)
    1) output matrix is initialize with B_i (y,y) 
    2) K_ij cross correlate with X_j result in output matrix (y,y) which add to the initialized output matrix, B_i (But not complete)
    3) As looping for j, K_ij cross correlate with the next X_j, which add to the latest output matrix (y,y) (Until the end of Loop)
2) Next i until the loop end 



Forward propagation
In matrix form;
Y => The vector of output matrix Y_i
B => The vector of bias matrix B_i
K => The matrix of kernel matrix K_ij (matrix of matrix)
X => The vector of input matrix X_j
then Y = B + ( K ⚙ X )  => ⚙︎ is the operation of matrix multiplication combine with cross-correlation︎

"""

"""
Backpropagation
Definition y_ij => the element in output matrix Y (i,j is the index of rows and columns)
∂E/∂Y_i = [[∂E/∂y_11, ∂E/∂y_12] ,[...]]   <= Shape : (y1,y2) = (i1-k +1 , i2-k +1)
(Each element ∂E/∂y_ij in the ∂E/∂Y will affect the rate of change of E)

then we need to find
1) ∂E/∂K = [[∂E/∂k_11, ∂E/∂k_12] ,[...]]
    Then  ∂E/∂K_ij = X_j ★ ∂E/∂Y_i   
          Note : i,j is the kernel and input index

2) ∂E/∂B = [[∂E/∂b_11, ∂E/∂b_12] ,[...]]
    Then  ∂E/∂B_i = ∂E/∂Y_i   <= Shape : (#k , i1 - k + 1 , i2 -k + 1)
    
3) ∂E/∂X = [[∂E/∂x_11, ∂E/∂x_12] ,[...]]
    Then  ∂E/∂X_j  = ∑ ∂E/∂Y_i ❊_full K_ij   <= Shape : (depth , i1 , i2) 



Eq 1 (Finding ∂E/∂K)
As ∂E/∂k_ij = ∑ ∂E/∂y_ij * x_ij 
and X_j = [[x_11 , x_12 , x_13],...] <= Shape: (i1,i2)
then X_j ★ ∂E/∂Y <= Each ∂E/∂k_ij in ∂E/∂K is the result of a cross correlation.

Eq 2 (Finding ∂E/∂B)
As ∂E/∂b_ij = ∂E/∂y_ij * ∂y_ij/∂k_ij   and ∂y_ij/∂k_ij = 1
   ∂E/∂b_ij = ∂E/∂y_ij
then ∂E/∂B = ∂E/∂Y

Eq3 (Finding ∂E/∂X)
For example, ∂E/∂x_11 = ∂E/∂y_11 * k_11
             ∂E/∂x_12 = ∂E/∂y_11 * k_12  + ∂E/∂y_12 * k_11
             ∂E/∂x_13 = ∂E/∂Y_12 * k_12
             ∂E/∂x_21 = ∂E/∂y_11 * k_21  + ∂E/∂y_21 * k_11
             ∂E/∂x_22 = ∂E/∂y_11 * k_22  + ∂E/∂y_12 * k_21 + ∂E/∂y_21 * k_12 + ∂E/∂y_22 * k_11
                .
                .
                .
You can see that this operation could be done by full convolution;
∂E/∂X = ∂E/∂Y ❊_full K 
      = ∂E/∂Y ★_full rot_180(K)   => Shape : (i1,i2) 

As X is contributed to many Y then in general;
∂E/∂X_j  = ∑ ∂E/∂Y_i * ∂Y_i/∂X_j          <= Multivariable chain rule (Think of ∂E/∂Y_i * ∂Y_i/∂X_j  = ∂E/∂X_j)
∂E/∂X_j  = ∑ ∂E/∂Y_i ★_full rot_180(K_ij) <= Multivariable contribution


"""


class Reshape(Layer):
    def __init__(self , input_shape , output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input , self.output_shape)

    def backward(self, output_gradient , learning_rate):
        return np.reshape(output_gradient , self.input_shape)




class Convolution(Layer):
    def __init__(self , input_shape , kernel_size , depth):
        # input_shape is the shape of input in tuple , kernel_size is the size of each matrix in each Kernel tensor , depth is how many kernel tensor we want (The depth of the output)
        super().__init__()
        input_depth , input_height , input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth , input_height - kernel_size + 1, input_width - kernel_size + 1) # shape: (i1-k +1 , i2-k +1)
        self.kernel_shape = (depth , input_depth , kernel_size  , kernel_size)  # we have "depth" kernel tensors , Each kernel tensor has "input_depth" matrices , Each matrix has a shape of (kernel_size , kernel_size)
        self.kernels = np.random.randn(*self.kernel_shape)
        self.biases = np.random.randn(*self.output_shape) # The size of each bias matrix is equal to the size of each output matrix
        # Numpy note:  np.random.randn(*shape_tuple) => random number based on the point that each random values will be normally distributed (Mean = 0 , S.D. = 1)
        # Python note: *tuple used to unpack tuple (in this case, shape) and pass it as an argument of the function.


    def forward(self , input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth): # Loop on each element, Y_i in Y
            for j in range(self.input_depth): # Loop for each input matrix, X_j in X
                self.output[i] += signal.correlate2d(self.input[j] , self.kernels[i,j], "valid") # i represent kernel tensor index , while j represent each matrix index in each kernel tensor
                # signal.correlate2d(self.input[j] , self.kernels[i,j], mode = "valid") will have an array (same size as each bias matrix) as the output

        return self.output

    def backward(self, output_gradient , learning_rate):
        kernel_grad = np.zeros(self.kernel_shape)
        input_grad = np.zeros(self.input_shape)

        # Update Parameters
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernel_grad[i, j] = signal.correlate2d(self.input[j] , output_gradient[i] , "valid")
                input_grad[j] += signal.convolve2d(output_gradient[i] , self.kernels[i,j] , "full")  # ∂E/∂X_j  = ∑ ∂E/∂Y_i ❊_full K_ij  # This also indicate that every matrix in the Input tenser is updated.

        kernel_grad = clip_gradient_by_norm(kernel_grad,1e+50)
        self.kernels -= learning_rate * kernel_grad  # We can see that the shape of kernel_grad is equal to the kernel in this layer (initialize)
        self.biases -= learning_rate * output_gradient
        return input_grad
