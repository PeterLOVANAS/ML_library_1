import numpy as np

def pooling2D(array: np.array , pooling_size: tuple,stride: int):
    pool_lst = []
    for i in np.arange(array.shape[0] , step = stride):
        for j in np.arange(array.shape[1]  ,step = stride):
            mat = array[i:i+pooling_size[0], j:j+pooling_size[1]]
            if mat.shape == pooling_size:
                pool_lst.append(mat)

            else:
                pass

    return np.array(pool_lst)


def max_pooling_2D(array: np.array, pooling_size: tuple , stride: int):
    pool_lst = pooling2D(array , pooling_size , stride)
    max_pool =[]
    max_shape = (round((array.shape[0] - pooling_size[0]) / stride) + 1  , round((array.shape[1] - pooling_size[0]) / stride) + 1)
    for i in pool_lst:
        max_pool.append(i.max())
    max_ans = np.reshape(np.array(max_pool) , max_shape)
    return max_ans


def mask(array: np.array , pooling_size: tuple , stride: int):
    size = pooling_size
    output_arr = np.zeros(array.shape)
    lst = []
    if array.shape[0] % 2 == 0:
        a = 0
    elif array.shape[0] % 2 != 0:
        a = -1
    for i in np.arange(array.shape[0] + a, step = stride):
        for j in np.arange(array.shape[1] + a ,step = stride):
            s_lst = []
            for k1 in range(i , i+ size[0]):
                for k2 in range(j , j + size[1]):
                    tup = (k1 , k2)
                    s_lst.append(tup)
            lst.append(s_lst)

    ans_lst = []

    # Finding the highest (max) value and collect the index.
    for i in lst:
        max_val = array[0,0]
        max_ind = (0,0)
        for j in i:
            if array[j[0], j[1]] >= max_val:
                max_val = array[j[0], j[1]]
                max_ind = j
            else:
                pass
        ans_lst.append(max_ind)

    for k in ans_lst:
        output_arr[k[0], k[1]] = 1

    return output_arr



class MaxPooling2D:
    def __init__(self , pooling_size: tuple , stride: int):
        self.pooling_size = pooling_size
        self.stride = stride
        self.array = None

    def forward(self , array: np.array ):
        self.array = array
        max_pool_tensor = []
        for p in self.array:
            max_pool_tensor.append(max_pooling_2D(p , self.pooling_size , self.stride))
        return np.array(max_pool_tensor)

    def backward(self , output_gradient):
        Mask_tensor = []
        for p, og in zip(self.array , output_gradient):
            Mask = mask(p , self.pooling_size , self.stride)
            idx = np.argwhere(Mask == 1)
            if idx.shape[0] * idx.shape[1] != og.shape[0] * idx.shape[1]:
                for i in idx:
                    random_element = np.random.choice(output_gradient.flatten())
                    Mask[i[0], i[1]] = random_element
            elif idx.shape[0] * idx.shape[1] == og.shape[0] * idx.shape[1]:
                for i, k in zip(idx, og.reshape((idx.shape[0] ,1))):
                    Mask[i[0], i[1]] = k
            Mask_tensor.append(Mask)

        return np.array(Mask_tensor)


def clip_gradient_by_norm(param_gradient, clip_norm): # clip_norm is a number value
    param_gradient_norm = np.linalg.norm(param_gradient)  # This turns out to be a just a length of a vector
    if param_gradient_norm > clip_norm:
        param_gradient = param_gradient * (clip_norm / param_gradient_norm)
    else:
        pass
    return param_gradient





