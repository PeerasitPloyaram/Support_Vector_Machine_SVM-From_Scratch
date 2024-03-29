import numpy as np

class SVM:
    def __init__(self,kernel='linear', learning_rate=0.001, c=1,lambda_param=0.01 ,max_itr=1000, debug=False)-> None:
        self.learningRate = learning_rate
        self.lambda_param = lambda_param
        self.c = c
        self.itr = max_itr
        self.w = None           # Weight

        self.debug = debug      # True / False
        self.gradient_round = 0 # Gradient step

        if self.debug:
            print("-- Parameter --\nC: {}\nLearning Rate: {}\nLambda Param: {}\nN_Iters: {}\n---------------".format(self.c, self.learningRate, self.lambda_param, self.itr))

    def add_bias(self, features):
        n_samples = features.shape[0]
        return np.concatenate( (np.ones((n_samples, 1)), features), axis=1 ) # [1, x1, x2 , ..., xn]


    def compute_gradient(self, sample, label):
        slack = 1 - label * np.dot(sample, self.w)  # yi(wxi) < 1
        n_feature = sample.shape[0]

        gradient = np.zeros(n_feature)                 # feature + b
        if max(0, slack) == 0:
            gradient += (self.lambda_param * self.w)   # Yi(WXi) >=1
        else:
            gradient += (self.lambda_param * self.w) - (self.c * sample * label) # 1-Yi(WXi)
        
        self.gradient_round += 1
        return gradient

    def fit(self, x_train, y_train):
        x_train = self.add_bias(x_train)
        index, x_sample = x_train.shape
        self.w = np.zeros(x_sample)

        for itr in range(self.itr):
            for index, x_sample in enumerate(x_train):
                # Compute Gradient
                gradient = self.compute_gradient(x_sample, y_train[index])  # Find gradient

                # Update Weight
                self.w -= self.learningRate * gradient  # Update new weight

        if self.debug:
            print("Gradient {} steps.".format(self.gradient_round))

    def predict(self, test_features):
        test_features = self.add_bias(test_features)
        buffer = []                                    # List for Y Predict

        for x_sample in test_features:
            prediction = np.sign(np.dot(self.w, x_sample))
            buffer.append(prediction)

        return buffer
    
    def score(self, x_test, y_test):        
        counter = 0
        predict = self.predict(x_test)

        for index, sample_test in enumerate(predict):
            if sample_test == y_test[index]:    # If equal +1
                counter += 1
        return counter / len(x_test)