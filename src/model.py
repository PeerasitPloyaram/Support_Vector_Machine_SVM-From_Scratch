import numpy as np
import matplotlib.pyplot as plt

class SVM:
    def __init__(self,kernel='linear', learning_rate=0.001,lambda_param=0.01 ,max_itr=1000, debug=False, verbose=False)-> None:
        self.kernel = kernel
        self.learningRate = learning_rate
        self.lambda_param = lambda_param
        self.epoch = max_itr                # 1 epoch for n sample
        self.w = None                       # Weight

        self.debug = debug          # True / False
        self.gradient_round = 0     # Gradient step
        self.verbose = verbose

        # history for plot graph
        self.cost_function = []


        if self.debug:
            print("-- Parameter --\nLearning Rate: {}\nLambda Param: {}\nN_Iters: {}\n---------------".format(self.learningRate, self.lambda_param, self.epoch))

    def add_bias(self, features):                                               # add b to w for not compute b
        n_samples = len(features)
        return np.concatenate( (np.ones((n_samples, 1)), features), axis=1 )    # [1 , x1, x2, ..., xn]


    def gradient(self, sample, label):
        slack = label * np.dot(self.w, sample)          # Slack higeloss

        n_feature = sample.shape[0]
        gradient = np.zeros(n_feature)                  # Set array w for feature + b

        if slack >= 1:                               # if >=1 yi * (w * xi)
            gradient = (self.lambda_param * self.w)

        else:                                        # if <1 1 - yi * (w * xi)
            gradient = (self.lambda_param * self.w) - (sample * label)
        
        self.gradient_round += 1    # Count round
        return gradient

    def fit(self, x_train, y_train):
        x_train = self.add_bias(x_train)       # Add b in w
        index, x_sample = x_train.shape        # get Index of sample, sample
        self.w = np.zeros(x_sample)            # Set w size feature x

        for epoch in range(self.epoch):                             # Train Epoch round
            for index, x_sample in enumerate(x_train):
                # Compute Gradient
                gradient = self.gradient(x_sample, y_train[index])  # Find gradient

                # Update Weight
                self.w -= self.learningRate * gradient              # Update new weight

            avr_gradient = sum(gradient) / len(x_sample)            # Find average gradient of array gradient

            cost_w = 1 / 2 * np.dot(self.w, self.w) + ( self.lambda_param * avr_gradient)   # Cal Cost(w) of regularization
            self.cost_function.append(cost_w)

            if self.verbose == True:
                print(cost_w)
            
        if self.debug:
            print("Gradient {} steps.".format(self.gradient_round))



    def predict(self, test_features):
        test_features = self.add_bias(test_features)   # Add b to w
        buffer = []                                    # List for y predict

        for x_sample in test_features:                 # Predict from list
            predict = np.dot(self.w, x_sample)         # Predict
            if predict < 0:
                buffer.append(-1)
            elif predict > 0:
                buffer.append(1)
            else:
                buffer.append(0)
        return buffer



    def score(self, x_test, y_test):        
        counter = 0                             # Counter correct
        predict = self.predict(x_test)          # Predict

        for index, sample_test in enumerate(predict):
            if sample_test == y_test[index]:            # If equal +1
                counter += 1
        return counter / len(x_test)                    # Return Accuracy 0 - 1
    

    def plot_cost(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.cost_function)



    def plot_accuracy(self, x_train, y_train, x_test, y_test, epoch, verbose=False):
        l1 = []
        l2 = []
        self.debug = False
        for _ in epoch:
            self.epoch = _
            model = self.fit(x_train, y_train)
            score1 = self.score(x_test, y_test)
            score2 = self.score(x_train, y_train)
            if verbose:
                print("Epoch {}\nValidate Accruacy is: {}\nTrain Accuracy is: {}".format(_, score1, score2))

            l1.append(score1)
            l2.append(score2)


        plt.figure(figsize=(10,5))
        plt.title("Model accuracy")
        plt.plot(l1, label="Validation", marker='o')
        plt.plot(l2, label="Train", marker='x')
        plt.legend()
        plt.grid(linestyle = '--', linewidth = 0.5)
        plt.xticks(np.arange(len(epoch)), epoch)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (0 - 1)')
        plt.show()



    def plot_lambda(self, x_train, y_train, x_test, y_test, lambda_param, epoch, verbose=False):
        train = []
        validate = []

        self.debug = False

        for i in lambda_param:
            self.lambda_param = i
            self.epoch = epoch
            model = self.fit(x_train, y_train)

            validate_score = self.score(x_test, y_test)
            train_score = self.score(x_train, y_train)

            if verbose:
                print("Lambda {}\nValidate Accruacy is: {}\nTrain Accuracy is: {}".format(i, validate_score, train_score))

            train.append(train_score)
            validate.append(validate_score)

        plt.figure(figsize=(10,5))
        plt.title("Model accuracy [Epoch {}]".format(epoch))
        plt.plot(validate, label="Validation", marker='o')
        plt.plot(train, label="Train", marker='x')
        plt.legend()
        plt.grid(linestyle = '--', linewidth = 0.5)
        plt.xticks(np.arange(len(lambda_param)), lambda_param)
        plt.xlabel('λ (Lambda)')
        plt.ylabel('Accuracy (0 - 1)')
        plt.show()  