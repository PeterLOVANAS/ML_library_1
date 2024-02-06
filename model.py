import loss
import nn
from optimizer import *
from activation_func import *
from utility import *
from loss import *
import numpy as np
from metrics import *
from dense import *
from alive_progress import alive_bar
from typing import Optional
import re
import functools




class Sequential(nn.Module):
    def __init__(self , layers: list):
        super(Sequential, self).__init__()
        self.layers = layers

    def forward(self, input): # Forward propagation
        self.input = input
        output = input
        for l in self.layers:
            output = l.forward(output)

        self.output = output
        return self.output


    def compile(self, optimizer: Optimizer , loss: loss_func , metric: list = [] ):
        """

        :param optimizer: Optimizer object WITH pre-defined arguments; EX. Gradient_Descent()
        :param loss:  loss_func object WITHOUT pre-defined arguments; EX. loss_func.binary_cross_entropy
        :param metric: list of metrics' name for evaluation
        :param metric:  list of metric objects WITH pre-defined arguments;  EX. classification_metrics.accuracy_score
        :return: None

        """
        
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric


    def defined_labels(self , data: dataloader): # Classification ONLY! => Confusion Matrix
        label_set = list(set(data.dataset[1]))
        return functools.partial(classification_metrics.confusion_matrix , labels = label_set)
        # Defined confusion_matrix parameters: labels


    def defined_metric(self, data: dataloader):
        metric_obj_lst = []
        for i in self.metric:
            if i.lower() == "accuracy":
                metric_obj_lst.append(classification_metrics.accuracy_score)
            elif i.lower() == "confusion_matrix":
                metric_obj_lst.append(self.defined_labels(data))
            # Further conditions for future classification object

        self.metric = metric_obj_lst








    def fit(self, data: dataloader ,  epochs: int ,  validation_data: Optional[dataloader] = None ):

        history = {}

        # Defining metric objects
        if len(self.metric) != 0:
            self.defined_metric(data)
        else:
            pass

        history["loss_train"] = []
        if len(self.metric) != 0:
            for m in self.metric:
                history[m.__name__ + "_train"] = [] # e.g. accuracy , confusion matrix ,etc. (But for training)
        else:
            pass

        data_train = data
        if validation_data == None:
            pass
        else:
            history["loss_validation"] = []
            if len(self.metric) != 0:
                for m in self.metric:
                    history[m.__name__ + "_validation"] = [] # e.g. accuracy , confusion matrix ,etc. (But for validation)
            else:
                pass
            data_val = validation_data






        # train_loss , train_accuracy , validation_loss ,validation_accuracy
        for e in range(epochs):
            loss_train_sum = []
            y_train_pred = []
            with alive_bar(total = data.total_samples , title =f"Epoch {e+1}" , theme = "smooth" ) as bar:

                for batch in data_train:
                    avg_grad_batch = 0
                    if len(batch) == 2:  # batch = (X,Y)
                        avg_grad_batch_sum = []

                        for x, y in zip(batch[0] , batch[1]):
                            output = self.forward(x) # output is numpy array vector
                            y_train_pred.append(np.argmax(output))
                            loss_obj = self.loss(y , output) # ***WARNING*** => y needs to categorical vector
                            loss_train_sum.append(loss_obj.loss())
                            output_gradient = loss_obj.loss_prime()
                            model_grad_sample = self.backward(output_gradient) # [ [dW_1 , dB_1], [dW_2 , dB_2], [dW_3 , dB_3] , ... , [dW_l , dB_l] ]  gradients of parameters in layer l
                            avg_grad_batch_sum.append(model_grad_sample)

                        avg_grad_batch = np.sum(avg_grad_batch_sum) / len(batch[0]) # [[dW_l_avg , dB_l_avg]] <= Shape (l , #θ) <= In this case the shape is (l , 2) as #θ (weights and biases)

                        # Optimization
                        opt_terms = []  # Each element is the optimized term of each parameters (e.g. W,B)
                        for i in range(avg_grad_batch.shape[1]):
                            self.optimizer.avg_grad_model = avg_grad_batch[: , i] # i = 0 = dW , i = 1 = dB
                            opt_terms.append(self.optimizer.compute())

                        # Update parameters
                        for l in self.layers:
                            if l.get_gradients() == None:  # Activation function
                                pass
                            else:
                                new_param = [p + opt_terms for p , opt_term in zip(l.parameters() , opt_terms)]
                                l.update(new_param)




                    elif len(batch) == 1:
                        output = self.forward(batch[0])
                        pass
                        #  Required more knowledge on this area as the model only receive X not Y

            # Calculate the "average" loss of this entire training dataset
            history["loss_train"].append(np.array(loss_train_sum).mean())


            if len(self.metric) != 0:
                # Find accuracy or other metrics (train)
                pattern = r"\w+_train"
                for m in [k for k in list(history.keys()) if bool(re.fullmatch(pattern, k))]: # Find the right name ([..]_train)
                    if m == "loss_train":
                        continue

                    # Choose the metric
                    for i in self.metric:
                        if i.__name__ == m.replace("_train" , ""):
                            history[m].append(i(y_true = np.argmax(data.dataset[1]) , y_pred = y_train_pred)) # Assume that data.dataset[1] is a categorical vector (array)



            if validation_data != None:
                # Forward propagation over validation dataset
                y_val_pred = []
                loss_val_sum = []
                for batch in data_val:
                    if len(batch) == 2:
                        for x, y in zip(batch[0] , batch[1]):
                            output = self.forward(x)
                            y_val_pred.append(np.argmax(output))
                            loss_obj = self.loss(y , output)
                            loss_val_sum.append(loss_obj.loss())

                    elif len(batch) == 1:
                        pass # Don't know much here

                # Calculate the "average" loss of this entire validation dataset
                history["loss_validation"].append(np.array(loss_val_sum).mean())

                if len(self.metric) != 0:
                    # Find accuracy or other metrics (train)
                    pattern = r"\w+_validation"
                    for m in [k for k in list(history.keys()) if bool(re.fullmatch(pattern, k))]:
                        if m == "loss_validation":
                            continue

                        # Choose the metric
                        for i in self.metric:
                            if i.__name__ == m.replace("_validation" , ""):
                                history[m].append(i(y_true = np.argmax(validation_data.dataset[1]) , y_pred = y_val_pred))

        return history























if __name__ == "__main__":
    model = Sequential([Dense(10,20) , Dense(20,2)])
    opt = Gradient_Descent([] , 0.01 ,0.4 ,False)
    model.compile(optimizer= opt , loss = loss.mse , metric=classification_metrics.accuracy_score)

