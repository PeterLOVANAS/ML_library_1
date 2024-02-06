import numpy as np


class metric:
    pass

class classification_metrics(metric):

    @staticmethod
    def accuracy_score(y_true: list, y_pred: list):
        """
        :param y_true:  [1,0,1,2,3,4]  => the length of the list is the number of samples use in evaluation
        :param y_pred:  [3,1,1,1,3,4]
        :return:  float value of the accuracy score
        """

        if len(y_true) != len(y_pred):
            raise Exception("The length of y_true and y_pred should be the same for this evaluation")
        else:
            pass

        counter = 0
        for t, p in zip(y_true , y_pred):
            if t == p:
                counter += 1
            elif t != p:
                pass

        return counter / len(y_true)

    @staticmethod
    def confusion_matrix(y_true: list , y_pred:list , labels:list):
        """
        :param y_true:  [1,0,1,2,3,4]  => the length of the list is the number of samples use in evaluation
        :param y_pred:  [3,1,1,1,3,4]
        :param labels:  [0,1,2,3,4]  => All labels in order

        EX.
            0   1   2   3   4   <- Predicted class
        0 (0,0)
        1
        2                  (2,4)
        3
        4         (4,2)
        ^- True Labels
        (4,2) means the true label is 4 but the model predict it is 2
        (2,4) means the true label is 2 but the model predict it is 4


        :return:  numpy array represent the confusion matrix. (The cell represents number of frequency of that combination between the prediction and true label)
        """

        if len(y_true) != len(y_pred):
            raise Exception("The length of y_true and y_pred should be the same for this evaluation")
        else:
            pass

        matrix = np.zeros((len(labels), len(labels)))
        y_true_idx = [labels.index(x) for x in y_true]
        y_pred_idx = [labels.index(x) for x in y_pred]
        permu_lst = [(x,y) for x,y in zip(y_true_idx ,y_pred_idx) ]
        #x => True,  y => pred
        permu_set = set(permu_lst)
        for t in permu_set:
            f = permu_lst.count(t)  # count the frequency in the dataset
            matrix[t[0] , t[1]] = f

        return matrix













