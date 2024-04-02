import numpy as np
import pandas as pd

def train_test_split(x, y, test_size=0.2, random_state=None, debug=False):
    # Validate parameter
    if not type(test_size) == float or type(test_size) == int:  # If not number
        return 0
    if test_size > 1 or test_size < 0:              # If max
        return 0

    test_size_ratio = test_size                     # test size ratio
    train_size_ratio = 1 - test_size                # train size ratio

    test_size = round(len(x) * test_size_ratio)     # test size
    train_size = round(len(x) * train_size_ratio)   # train size

    # print(len(x), len(y))
    if debug:
        print("Data  Size: {}".format(len(x)))
        print("Train Size: {}".format(train_size))
        print("Test  Size: {}".format(test_size))
        print("Train Size + Test Size: {}".format(train_size + test_size))


    if random_state == None:                            # If don't have
        seed = np.random.randint(0,1000000)             # random 0 - 1000000
    elif type(random_state) == int and random_state > 0:
        seed = random_state     # set seed
    else:
        return 0                # error, exit

    np.random.seed(seed)
    bufferX = np.random.permutation(x)
    bufferX_train = bufferX[:train_size]    # x_train
    bufferX_test = bufferX[:test_size]      # x_test

    np.random.seed(seed)
    bufferY = np.random.permutation(y)
    bufferY_train = bufferY[:train_size]    # y_train
    bufferY_test = bufferY[:test_size]      # y_test

    return bufferX_train, bufferX_test, bufferY_train, bufferY_test



def standard_scaler(data):
    df = pd.DataFrame(data) # Gen Frame

    for _ in df:
        mean = df[_].mean() # Mean
        std = df[_].std()   # Standard deviation

        for i, sample in enumerate(df[_]):
            z = ( sample - mean ) / std     # Create new data
            df.loc[i, _] = z                # Replace at location

    return df.to_numpy()



def positive_negative_check(y):

    buff1=  y[0]        # Get first label
    for _ in y:
        if buff1 != _:  # Find another label
            buff2 = _   # If found break
            break

    if buff1 < buff2:   # Compare to correct class
        n_cl = buff1
        p_cl = buff2
    else:
        n_cl = buff2
        p_cl = buff1

    p = 0       # Counter positive class
    s = 0       # Counter negative class

    for _ in y:         # Count positve and negative class
        if _ == p_cl:
            p += 1
        else:
            s += 1
            
    print("Total {} Samples.".format(p + s))
    print("Positive Class [{}]: {} sample.".format(p_cl, p))
    print("Negative Class [{}]: {} sample.".format(n_cl, s))

    return p, s


def random_under_sampling(data, n_sample, random_state=None):
    size = len(data)                            # n_sample
    delete_size = size - n_sample               # Cal size want to delete

    print("Current Size:",size)
    print("Delete Size Target:",delete_size)

    df = pd.DataFrame(data)
    if random_state != None:            # Set random state
        np.random.seed(random_state)
        
    drop_index = np.random.choice(df.index, delete_size, replace=False) # Gen random index list
    df_subset = df.drop(drop_index)     # Drop by random index
    return df_subset


def confusion_matrix(y_predict, y_test):
    tp, tn, fp, fn = 0, 0, 0, 0

    for index, y_p in enumerate(y_predict):
        if y_p == y_test[index]:
            if y_p == 1:
                tp += 1
            else:
                tn += 1
        else:
            if y_p == -1 and y_test[index] == 1:
                fn += 1
            else:
                fp += 1

    precision = tp / (tp + fp)
    acc = (tp + tn) / (tp + tn + fp +fn)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)

    print("Precision: {:.4f}\nRecall: {:.4f}\nAccuracy: {:.4f}\nF1-score: {:.4f}".format(precision, recall, acc, f1))