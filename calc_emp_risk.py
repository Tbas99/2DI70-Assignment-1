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