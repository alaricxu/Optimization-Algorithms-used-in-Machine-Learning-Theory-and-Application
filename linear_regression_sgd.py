import matplotlib.pyplot as plt
from random import seed
from random import randrange
from csv import reader
from math import sqrt

class Linear_Regression:
    def predict(self, row, coefficients):
        yhat=coefficients[0]
        for i in range(len(row)-1):
            yhat+=coefficients[i+1]*row[i]
        return yhat

    # convert string column to float
    def str_column_to_float(self, dataset, column):
        for row in dataset:
            row[column]=float(row[column].strip().replace('\ufeff', ''))

    # load a csv file
    def load_csv(self, filename):
        dataset=list()
        with open(filename, 'r') as file:
            csv_reader=reader(file)
            # skip the first header row
            next(csv_reader, None)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
            return dataset


    # find the min and max values for each column
    def dataset_minmax(self, dataset):
        minmax=list()
        for i in range(len(dataset[0])):
            col_values=[row[i] for row in dataset]
            value_min=min(col_values)
            value_max=max(col_values)
            minmax.append([value_min, value_max])
        return minmax


    # rescale dataset columns to the range 0-1
    def normalized_dataset(self, dataset, minmax):
        for row in dataset:
            for i in range(len(row)):
                row[i]=(row[i]-minmax[i][0])/(minmax[i][1]-minmax[i][0])
    
    def data_preparation(self):
        # load and prepare data
        filename='winequality-white.csv'
        dataset=self.load_csv(filename)
        for i in range(len(dataset[0])):
            self.str_column_to_float(dataset, i)
        
        # normalization
        minmax=self.dataset_minmax(dataset)
        self.normalized_dataset(dataset, minmax)

        x=[row[0] for row in dataset]
        y=[row[1] for row in dataset]

        # display graph of x and y
        plt.plot(x, y, 'bx')
        plt.show()

        return dataset
    

    # 2 parameters of sgd: learning rate, epoches
    # 3 loops
    #   1. loop over each epoch
    #   2. loop over each row in the training data for an epoch
    #   3. loop over each coefficient and update it for a row in an epoch
    
    # estimate linear regression coefficients using stochastic gradient descent
    def coefficients_sgd(self, train, l_rate, n_epoch):
        coef=[0.0 for i in range(len(train[0]))]
        for epoch in range(n_epoch):
            for row in train:
                yhat=self.predict(row, coef)
                error=yhat-row[-1]
                coef[0]=coef[0]-l_rate*error
                for i in range(len(row)-1):
                    coef[i+1]=coef[i+1]-l_rate*error*row[i]
                # print(l_rate, n_epoch, error)
        return coef

    # linear regression algorithm with stochastic gradient descent
    def linear_regression_sgd(self, train, test, l_rate, n_epoch):
        predictions=list()
        coef=self.coefficients_sgd(train, l_rate, n_epoch)
        for row in test:
            yhat=self.predict(row, coef)
            predictions.append(yhat)
        return(predictions)

    # split a dataset into k folds
    def cross_validation_split(self, dataset, n_folds):
        dataset_split=list()
        dataset_copy=list(dataset)
        fold_size=int(len(dataset)/n_folds)
        for i in range(n_folds):
            fold=list()
            while len(fold)<fold_size:
                index=randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split
    
    # calculate root mean squared error
    def rmse_metric(self, actual, predicted):
        sum_error=0.0
        for i in range(len(actual)):
            prediction_error=predicted[i]-actual[i]
            sum_error+=(prediction_error**2)
        mean_error=sum_error/float(len(actual))
        return sqrt(mean_error)
    
    # evaluate an algorithm using a cross validation split
    def evaluate_algorithm(self, dataset, algorithm, n_folds, *args):
        folds=self.cross_validation_split(dataset, n_folds)
        scores=list()
        for fold in folds:
            train_set=list(folds)
            train_set.remove(fold)
            train_set=sum(train_set, [])
            test_set=list()
            for row in fold:
                row_copy=list(row)
                test_set.append(row_copy)
                row_copy[-1]=None
            predicted=algorithm(train_set, test_set, *args)
            actual=[row[-1] for row in fold]
            rmse=self.rmse_metric(actual, predicted)
            scores.append(rmse)
        return scores


# linear regression on wine quality dataset
seed(1)
n_folds=5
l_rate=0.01
n_epoch=50
lr=Linear_Regression()

scores=lr.evaluate_algorithm(lr.data_preparation(), lr.linear_regression_sgd, n_folds, l_rate, n_epoch)

print('Scores: %s' % scores)
print('Mean RMSE: %.3f' % (sum(scores)/float(len(scores))))
    










# dataset initialization

# dataset=[[1,1], [2,3], [4,3], [3,2], [5,5]]
# coef=[0.4, 0.8]
# for row in dataset:
#     yhat=predict(row, coef)
#     print("Expected=%.3f, Predicted=%.3f" %(row[-1], yhat))

# x=[row[0] for row in dataset]
# y=[row[1] for row in dataset]
# yhat=[predict(row, coef) for row in dataset]

# # display graph of x and y
# plt.plot(x, y, 'bx')
# plt.plot(x, yhat, 'r')
# plt.show()