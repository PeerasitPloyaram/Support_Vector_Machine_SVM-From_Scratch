import math
import numpy as np

def train_test_split(x, y, testsize=0.2, random_state=None, debug=False):
    # Validate parameter
    if not type(testsize) == float or type(testsize) == int:
        return 0
    if testsize > 1 or testsize < 0:
        return 0

    test_size_ratio = testsize                  # test size ratio
    train_size_ratio = 1 - testsize             # train size ratio

    test_size = round(len(x) * test_size_ratio)     # test size
    train_size = round(len(x) * train_size_ratio)   # train size

    # print(len(x), len(y))
    if debug:
        print("Data  Size: {}".format(len(x)))
        print("Train Size: {}".format(train_size))
        print("Test  Size: {}".format(test_size))
        print("Train Size + Test Size: {}".format(train_size + test_size))


    if random_state == None:                    # If don't have
        seed = np.random.randint(0,1000000)     # random 0 - 1000000
    elif type(random_state) == int and random_state > 0:
        seed = random_state
    else:
        return 0    # error, exit

    np.random.seed(seed)
    bufferX = np.random.permutation(x)
    bufferX_train = bufferX[:train_size]    # x_train
    bufferX_test = bufferX[:test_size]      # x_test

    np.random.seed(seed)
    bufferY = np.random.permutation(y)
    bufferY_train = bufferY[:train_size]    # y_train
    bufferY_test = bufferY[:test_size]      # y_test

    return bufferX_train, bufferX_test, bufferY_train, bufferY_test



def train_test_validate_split(x, y, testsize=None, validatesize=None)-> None:
    train_size_ratio = 0.0
    test_size_ratio = 0.0
    validate_size_ratio = 0.0

    data_size = len(x)
    a = x
    b = y

    if testsize != None and validatesize != None:
            # if testsize < 0.1 or testsize > 0.9 or validatesize < 0.1 or validatesize > 0.9:
            #     return 0
            # else:
        test_size_ratio = testsize
        validate_size_ratio = validatesize
        train_size_ratio = "{:.2f}".format(1 - test_size_ratio - validate_size_ratio)

        print(train_size_ratio, validate_size_ratio, test_size_ratio, data_size)

    

    test_size = math.floor(data_size * float(test_size_ratio))
    validate_size = round(data_size * float(validate_size_ratio))
    train_size = math.ceil(data_size * float(train_size_ratio))

    print(train_size,validate_size,test_size)
    print(train_size + test_size + validate_size)
        
