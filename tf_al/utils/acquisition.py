import numpy as np
from scipy.special import beta, digamma

def beta_approximated_upper_joint_entropy(a, b):
    """
        
    """
    regular_beta = beta(a, b)

    regular_beta_clipped = regular_beta*(regular_beta>0)
    # is_inf = np.any(np.isposinf(regular_beta))
    # print("Infininte",is_inf)
    # print("NaN: ", np.any(np.isnan(regular_beta)))
    # print("Negatives: ",np.any(regular_beta_clipped<0))
    # print("---------")

    sub_1 = (a/(a+b))*np.log(regular_beta_clipped, where=regular_beta_clipped!=0)
    sub_2 = (np.divide(beta(a+1, b), regular_beta_clipped, where=regular_beta_clipped!=0))

    dig_1 = digamma(a+1)
    dig_2 = digamma(a+b+1)
    dig_3 = digamma(b)

    return np.sum(sub_1-(a*sub_2*(dig_1-dig_2))-((b-1)*sub_2*(dig_3-dig_2)), axis=-1)