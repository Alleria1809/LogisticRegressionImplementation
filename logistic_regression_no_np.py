import math
from sklearn.model_selection import train_test_split
from sklearn import datasets

# define sigmoid function
def sigmoid(X):
    """ sigmoid function

    Args:
        X (1-D prediction list): 1-D prediction list

    Returns:
        _type_: sigmoid list
    """
    tmp = []
    # if we have the overflow, we need to clip
    # my test is that -709 is the margin
    for v in X:
        if v<=-709:
            v = -709
        tmp.append((1/(1+math.exp(-v))))
    #     else:
    #         tmp.append()
    # tmp = [(1/(1+math.exp(-v))) for v in X]
    return tmp


def accuracy(y_pred, y_test):
    """_summary_ calculate the accuracy
    Args:
        y_pred (_type_): predicted value
        y_test (_type_): true value

    Returns:
        _type_: accuracy
    """
    count = 0
    for i, value in enumerate(y_pred):
        if value == y_test[i]:
            count += 1
    return count / len(y_pred)

class LogisticRegression():

    def __init__(self, lr = 0.001, n_iters = 1000) -> None:
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = [0] * n_features  # initialize weights
        self.bias = 0  # initialize bias
        # print(n_samples, n_features)

        for _ in range(self.n_iters):
            linear_predictions = []
            # calculate the value of the linear function y = bias + weights * x
            for idx, row in enumerate(X_train):
                # calculate for each sample (1 * n_samples)
                linear_predictions.append(sum([i*j for (i, j) in zip(row, self.weights)]))
            
            # make the prediction
            predictions = sigmoid(linear_predictions)
            # print(predictions, len(predictions))

            errors = []
            # calculate the errors, (1 * n_samples)
            for i, pred in enumerate(predictions):  # calculate the corresponding error for each prediction
                errors.append(pred - y[i])
            # errors = list(map(lambda x,y:x-y, predictions, y))  
            
            # transpose (n_features * n_samples)
            X_transpose = list(map(list, zip(*X)))  

            # calculate the gradients
            g = []
            for i, row in enumerate(X_transpose):
                # row: 1 * n_samples, errors: 1 * n_samples
                g.append(sum([i*j for (i, j) in zip(row, errors)]))
            dw = [(1/n_samples) * x for x in g]  # (1 * n_samples)

            error_sum = 0
            for error in errors:
                error_sum += error
            db = (1/n_samples) * error_sum # (1 * n_samples)
            # print(db)

            # update weights and bias
            # self.weights = self.weights - self.lr * dw
            self.weights = list(map(lambda x,y: x-self.lr*y, self.weights, dw))
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        linear_predictions = []
        for idx, row in enumerate(X):
            linear_predictions.append(sum([i*j for (i, j) in zip(row, self.weights)]))
        # predictions = sigmoid(linear_predictions)
        predictions = sigmoid(linear_predictions)
        class_pred = [0 if y<=0.5 else 1 for y in predictions]
        return class_pred
    

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2009)

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy(y_pred, y_test)
print(f"accuracy = {acc}")
