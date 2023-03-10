from kNN import KNNFunctions

class LossFunctions():
    def __init__(self):
        pass

    def l01_loss_k(X_test, y_test, X_train, y_train, distance_function, k = [10], updates = 250):
        X_train_num = X_train.to_numpy()
        iter = 0
        correct_counters = dict([(x, {"correct": 0, "false": 0}) for x in k])
        for index, row in X_test.iterrows():
            preds = KNNFunctions.predict_for_multiple_k(row.to_numpy(), distance_function, X_train_num, y_train, k)
            correct = y_test.iloc[index]
            for k_val, pred in preds:
                if correct == pred:
                    correct_counters[k_val]["correct"] += + 1
                else:
                    correct_counters[k_val]["false"] += + 1
            iter = iter + 1
            if iter % updates == 0:
                print(f'iter: {index} {correct_counters}')
        return correct_counters

    def l01_loss_LOOCV(X_train, y_train, distance_function, k = [10], updates = 250):
        X_train_num = X_train.to_numpy()
        iter = 0
        correct_counters = dict([(x, {"correct": 0, "false": 0}) for x in k])
        for index, row in X_train.iterrows():
            preds = KNNFunctions.predict_for_multiple_k(row.to_numpy(), distance_function, X_train_num, y_train, k, LOOCV = True)
            correct = y_train.iloc[index]
            for k_val, pred in preds:
                if correct == pred:
                    correct_counters[k_val]["correct"] += + 1
                else:
                    correct_counters[k_val]["false"] += + 1
            iter = iter + 1
            if iter % updates == 0:
                print(f'iter: {index} {correct_counters}')
        return correct_counters

    def l01_loss_LOOCV_for_p(X_train, y_train, distance_function, k = [10], p = 1, updates = 250):
        X_train_num = X_train.to_numpy()
        iter = 0
        correct_counters = dict([(x, {"correct": 0, "false": 0}) for x in k])
        for index, row in X_train.iterrows():
            preds = KNNFunctions.predict_for_multiple_k(row.to_numpy(), distance_function, X_train_num, y_train, k, LOOCV = True, p = p)
            correct = y_train.iloc[index]
            for k_val, pred in preds:
                if correct == pred:
                    correct_counters[k_val]["correct"] += + 1
                else:
                    correct_counters[k_val]["false"] += + 1
            iter = iter + 1
            if iter % updates == 0:
                print(f'iter: {index} {correct_counters}')
        return correct_counters, p

