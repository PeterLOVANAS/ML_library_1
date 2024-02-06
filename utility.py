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

# dataset => [X ,Y]  OR  [X]
class dataloader:

    def __init__(self, dataset: list , batch_size: int , shuffle: bool):
        """
        :param dataset: list   EX.  [X ,Y]  OR  [X]
        :param batch_size:  int  EX. 32 , 256
        :param shuffle: bool
        """
        self.total_samples = len(dataset[0])
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size


    def shuffle_dataset(self):
        # Shuffle
        dataset_shuffle = []
        if self.shuffle == True:
            idx_shuffle = np.arange(self.total_samples)
            np.random.shuffle(idx_shuffle)
            for v in self.dataset:
                x = v[idx_shuffle.tolist()]
                dataset_shuffle.append(x)

        elif self.shuffle == False:
            dataset_shuffle = self.dataset

        return dataset_shuffle


    def create_batch(self):
        data_batch_split = []
        dataset_shuffle = self.shuffle_dataset()
        for v in dataset_shuffle:
            v_batch_split = []

            for i in range(0 , self.total_samples // self.batch_size):
                start_idx = i * self.batch_size
                end_idx = (i+1) * self.batch_size
                batch_data = v[start_idx: end_idx]
                v_batch_split.append(batch_data)

            if self.total_samples % self.batch_size != 0:
                start_idx = (self.total_samples // self.batch_size) * self.batch_size
                end_idx = self.total_samples
                batch_data = v[start_idx:end_idx]
                v_batch_split.append(batch_data)

            else:
                pass

            data_batch_split.append(np.array(v_batch_split , dtype=object))


        return np.array(data_batch_split , dtype= object)

    def __iter__(self):
        self.current_batch = 0
        self.batch_data = self.create_batch()
        return self

    def __next__(self):
        if len(self.dataset[0]) % self.batch_size == 0:
            if self.current_batch >= len(self.dataset[0]) // self.batch_size:
                raise StopIteration

        elif len(self.dataset[0]) % self.batch_size != 0:
            if self.current_batch >= (len(self.dataset[0]) // self.batch_size) + 1:
                raise StopIteration

        batch_input = self.batch_data[0, self.current_batch]

        if len(self.dataset) > 1:
            batch_label = self.batch_data[1, self.current_batch]
            self.current_batch += 1
            return batch_input, batch_label
        else:
            self.current_batch += 1
            return batch_input


def clip_gradient_by_norm(param_gradient, clip_norm): # clip_norm is a number value
    param_gradient_norm = np.linalg.norm(param_gradient)  # This turns out to be a just a length of a vector
    if param_gradient_norm > clip_norm:
        param_gradient = param_gradient * (clip_norm / param_gradient_norm)
    else:
        pass
    return param_gradient



if __name__ == "__main__":
    dataset = [np.array([np.random.randn(10,10), np.random.randn(10,10), np.random.randn(10,10), np.random.randn(10,10), np.random.randn(10,10)]), np.array([10, 20, 30, 40, 50])]
    #arr = np.array([np.random.randn(10,10), np.random.randn(10,10), np.random.randn(10,10), np.random.randn(10,10), np.random.randn(10,10)])
    #print(arr[0:3].shape)
    #dataset = [np.array([1,2,3,4,5]) , np.array([10,20,30,40,50])]
    batch_size = 2
    shuffle = True
    data_loader = dataloader(dataset, batch_size, shuffle)
    for d in data_loader:
        print(d)







