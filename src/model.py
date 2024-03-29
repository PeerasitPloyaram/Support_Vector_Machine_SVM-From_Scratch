import numpy as np

class SVM:
    def __init__(self,kernel='linear', learning_rate=0.001, c=1,lambda_param=0.01 ,max_itr=1000, debug=False)-> None:
        self.learningRate = learning_rate
        self.lambda_param = lambda_param
        self.c = c
        self.itr = max_itr
        self.w = None # Weight

        if debug:
            print("C: {}\nLearning Rate: {}\nLambda Param: {}\nC: {}\nN_Iters: {}".format(self.c, self.learningRate, self.lambda_param, self.c, self.itr))

    def add_bias_term(self, features):
        n_samples = features.shape[0]
        ones = np.ones((n_samples, 1))
        return np.concatenate((ones, features), axis=1)


    def compute_gradient(self, sample, label):
        slack = 1 - label * np.dot(sample, self.w)  # yi(wxi) < 1
        n_feature = sample.shape[0]

        gradient = np.zeros(n_feature) # Feature + b
        if max(0, slack) == 0:
            gradient += (self.lambda_param * self.w)   # Yi(WXi) >=1
        else:
            gradient += (self.lambda_param * self.w) - (self.c * sample * label) # 1-Yi(WXi)

        return gradient

    def fit(self, x_train, y_train):
        x_train = self.add_bias_term(x_train)
        index, x_sample = x_train.shape
        self.w = np.zeros(x_sample)

        for itr in range(self.itr):
            for index, x_sample in enumerate(x_train):
                # Compute Gradient
                gradient = self.compute_gradient(x_sample, y_train[index])

                # Update Weight
                self.w = self.w - self.learningRate * gradient

    def predict(self, test_features):
        test_features = self.add_bias_term(test_features)
        
        tt = []
        n_samples = test_features.shape[0]
        for index in range(n_samples):
            prediction = np.sign(np.dot(self.w, test_features[index]))
            tt.append(prediction)
        return tt
    
    def score(self, y_predict, y_test):
        
        size = len(y_test)
        counter = 0
        for i, x in enumerate(y_predict):
            if (y_predict[i] == y_test[i]):
                counter += 1

        return counter / size