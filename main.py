import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import istarmap  # import to apply patch
from multiprocessing import Pool
import tqdm

# Custom classes
from loss import LossFunctions
from utils import Utils

def main():
    # Import data
    X_train = pd.read_csv("MNIST_train_small.csv", header = None)
    y_train = X_train.iloc[:,0]
    X_train = X_train.drop(X_train.columns[0], axis=1)

    X_test = pd.read_csv("MNIST_test_small.csv", header = None)
    y_test = X_test.iloc[:,0]
    X_test = X_test.drop(X_test.columns[0], axis=1)

    # Set to true to view plots
    show_plots = False

    # Question a)
    accuracies_train = LossFunctions.l01_loss_k(X_train, y_train, X_train, y_train, Utils.euclidean_distance, list(range(1,21)), 1000)
    accuracies_test = LossFunctions.l01_loss_k(X_test, y_test, X_train, y_train, Utils.euclidean_distance, list(range(1,21)), 1000)

    x_val_test = [x for x in accuracies_test]
    test_risks, test_argmin_k = Utils.calc_emp_risk(accuracies_test)
    y_val_test = list(test_risks.values())

    x_val_train = [x for x in accuracies_train]
    train_risks, train_argmin_k = Utils.calc_emp_risk(accuracies_train)
    y_val_train = list(train_risks.values())

    plt.figure(dpi=1200)
    plt.plot(x_val_test, y_val_test, label = 'Test Loss')
    plt.plot(x_val_train, y_val_train, label = "Train Loss")
    plt.vlines(test_argmin_k, 0, test_risks[test_argmin_k], label='Test Loss - Optimal K', linestyles='--', colors='r')
    plt.vlines(train_argmin_k, 0, train_risks[train_argmin_k], label='Train Loss - Optimal K', linestyles='--', colors='c')
    #plt.title('Empirical Risk for different K')
    plt.legend()
    plt.ylabel("Empirical Risk")
    plt.xlabel("K")
    plt.xticks(x_val_test)
    plt.savefig("K_comparison.jpeg")
    plt.show()

    if show_plots:
        plt.show()

    # Question b)
    accuracies_LOOCV = LossFunctions.l01_loss_LOOCV(X_train, y_train, Utils.euclidean_distance, list(range(1,21)), 1000)

    x_val_LOOCV = [x for x in accuracies_LOOCV]
    LOOCV_risks, LOOCV_argmin_k = Utils.calc_emp_risk(accuracies_LOOCV)
    y_val_LOOCV = list(LOOCV_risks.values())

    plt.figure(dpi=1200)
    plt.plot(x_val_test, y_val_test, label = 'Test Loss')
    plt.plot(x_val_train, y_val_train, label = "Train Loss")
    plt.plot(x_val_LOOCV, y_val_LOOCV, label = 'LOOCV Loss')
    plt.vlines(test_argmin_k, 0, test_risks[test_argmin_k], label='Test Loss - Optimal K', linestyles='--', colors='r')
    plt.vlines(train_argmin_k, 0, train_risks[train_argmin_k], label='Train Loss - Optimal K', linestyles='--', colors='c')
    plt.vlines(LOOCV_argmin_k, 0, LOOCV_risks[LOOCV_argmin_k], label='LOOCV Loss - Optimal K', linestyles='--', colors='m')
    #plt.title('Empirical Risk for different K')
    plt.legend()
    plt.ylabel("Empirical Risk")
    plt.xlabel("K")
    plt.xticks(x_val_test)
    plt.savefig("K_comparison_LOOCV.jpeg")
    
    if show_plots:
        plt.show()

    # Question c)
    multi_threading = False
    load_result_from_disk = True
    save_result_to_disk = False
    accuracies_LOOCV_optimized_p = {}

    if load_result_from_disk:
        with open('nested_cv.json', 'r') as fp:
            accuracies_LOOCV_optimized_p = json.load(fp)
    else:
        if multi_threading:
            with Pool(len(range(1,16))) as pool:
                args = [(X_train, y_train, Utils.minkowski_distance, list(range(1,21)), p_prime, 1000) for p_prime in range(1,16)]
                for result in tqdm.tqdm(pool.istarmap(LossFunctions.l01_loss_LOOCV_for_p, args), total=len(args)):
                    accuracies_LOOCV_optimized_p[result[1]] = result[0]
                    print(result)
        else:
            for p in range(1,16):
                print('Optimizing for p =', p)
                accuracies_LOOCV_optimized_p[p] = LossFunctions.l01_loss_LOOCV_for_p(X_train, y_train, \
                                                                                        distance_function = Utils.minkowski_distance, \
                                                                                        k = list(range(1,21)), \
                                                                                        p = p, \
                                                                                        updates = 1000)[0]
        
        if save_result_to_disk:
            with open('nested_cv.json', 'w') as fp:
                json.dump(accuracies_LOOCV_optimized_p, fp)

    # argmin k
    best_k_risks = {}
    for p, k in accuracies_LOOCV_optimized_p.items():
        k_risks_for_p, opt_k = Utils.calc_emp_risk(k)
        best_k_risks[p] = { opt_k : k_risks_for_p[opt_k] }
    print(best_k_risks.values()) # Returns k = 2 for each p as the lowest empirical risk

    # argmin p
    min_k = min(list(Utils.nested_dict_values(best_k_risks)))
    argmin_p = [p for p in best_k_risks if list(best_k_risks[p].values())[0] == min_k][0]
    argmin_k = list(best_k_risks[p].keys())[0]

    # Output
    print('Best choice for p = ', argmin_p)
    print('Best choice for k = ', argmin_k)
    accuracies_LOOCV_best_p = accuracies_LOOCV_optimized_p[argmin_p]
    x_val_LOOCV_best_p = [int(x) for x in accuracies_LOOCV_best_p]
    LOOCV_minkowski_risks, LOOCV_minkowski_argmin_k = Utils.calc_emp_risk(accuracies_LOOCV_best_p)
    y_val_LOOCV_best_p = list(LOOCV_minkowski_risks.values())

    plt.figure(dpi=1200)
    plt.plot(x_val_test, y_val_test, label = 'Test Loss', color='r')
    plt.plot(x_val_train, y_val_train, label = "Train Loss", color='b')
    plt.plot(x_val_LOOCV, y_val_LOOCV, label = 'LOOCV Loss', color='g')
    plt.plot(x_val_LOOCV_best_p, y_val_LOOCV_best_p, label = f"Minkowski Loss(p = {argmin_p})", color='y')
    plt.vlines(test_argmin_k, 0, test_risks[test_argmin_k], label='Test Loss - Optimal K', linestyles='--', colors='r')
    plt.vlines(train_argmin_k, 0, train_risks[train_argmin_k], label='Train Loss - Optimal K', linestyles='--', colors='b')
    plt.vlines(LOOCV_argmin_k, 0, LOOCV_risks[LOOCV_argmin_k], label='LOOCV Loss - Optimal K', linestyles='--', colors='g')
    plt.vlines(int(LOOCV_minkowski_argmin_k), 0, LOOCV_minkowski_risks[LOOCV_minkowski_argmin_k], label=f'Minkowski Loss(p = {argmin_p}) - Optimal K', linestyles='-.', colors='y')
    #plt.title('Empirical Risk for different K')
    plt.legend()
    plt.ylabel("Empirical Risk")
    plt.xlabel("K")
    plt.xticks(x_val_test)
    plt.savefig("K_comparison_LOOCV_minkowski_best_p.jpeg")
    
    if show_plots:
        plt.show()


def test():
    # Import data
    X_train = pd.read_csv("MNIST_train_small.csv", header = None)
    y_train = X_train.iloc[:,0]
    X_train = X_train.drop(X_train.columns[0], axis=1)

    X_test = pd.read_csv("MNIST_test_small.csv", header = None)
    y_test = X_test.iloc[:,0]
    X_test = X_test.drop(X_test.columns[0], axis=1)

    # Test code here for faster debugging


if __name__ == "__main__":
    main()
    #test()