import numpy as np

class Utils():
    def __init__(self):
        pass

    def euclidean_distance(vec1 , vec2):
        difvec = np.subtract(vec1, vec2) 
        sqvec = np.square(difvec)
        dist = np.sum(sqvec)
        return dist

    def minkowski_distance(vec1, vec2, p = 1):
        # Cast to object to avoid np.int64 overflows for p >= 7
        diffvec = np.absolute(np.subtract(vec1, vec2))
        powvec = np.power(diffvec, p, dtype=object)
        distsum = np.sum(powvec, dtype=object)
        dist = np.power(distsum, (1/p), dtype=object)
        return dist
    
    def calc_emp_risk(accuracies):
        # Get empirical risk for each k
        empirical_risks = {}
        for k, loss in accuracies.items():
            # Get size of first item (should all be equal)
            n = sum(loss.values())

            # Calc risk
            risk = (1/n)*loss['false']
            empirical_risks[k] = risk
            
        # Get best empirical risk for k
        min_val = min(empirical_risks.values())
        optimal_k = [k for k in empirical_risks if empirical_risks[k] == min_val][0] # argmin for dict
        
        return empirical_risks, optimal_k
    
    def nested_dict_values(d):
        for v in d.values():
            if isinstance(v, dict):
                yield from Utils.nested_dict_values(v)
            else:
                yield v