class KNNFunctions():
    def __init__(self):
        pass

    def predict_for_multiple_k(x, dist_function , X, y, k_values , LOOCV = False , p = None):
        dist_to_all = []

        for row, index in zip(X, y.index.values):
            if p is not None:
                dist = dist_function(x, row, p)
            else:
                dist = dist_function(x, row)
            dist_to_all.append((float(dist), int(index)))

        dist_to_all = sorted(dist_to_all, key=lambda tup: tup[0])
        
        if LOOCV:
            dist_to_all = dist_to_all[1:] + [ dist_to_all[0]]
            
        predictions = []
        for k in k_values:
            k_original = k
            neighbor_indeces = [x[1] for x in dist_to_all[:k]] 
            neighbor_classifications = y.take(neighbor_indeces) 
            closest = neighbor_classifications.mode()
            
            # In case there is a tie for a prediction , add more elements
            while len( closest ) > 1:
                k=k+1
                neighbor_indeces = [x[1] for x in dist_to_all[:k]]
                neighbor_classifications = y.take(neighbor_indeces) 
                closest = neighbor_classifications.mode()
                
            prediction = closest.iloc[0]
            predictions.append(( k_original , prediction ))
        return predictions