import W_experiment as w
import numpy as np

min_x = -0.5
max_x = 0.5
min_y = 0.5
max_y = 1.5
n_points = 51
x = np.linspace(min_x,max_x,n_points) #mean centered at 0
y = np.linspace(min_y,max_y,n_points) #std centered at 1
samples = w.Inverse_transform_sampling(1000,0)

def test_max_likelihood_est():
    anly_NLLs = w.analytic_max_likelihood_est(samples)
    (best_mu,best_std,Z,min_nll) = w.max_likelihood_est(samples,min_x,max_x,min_y,max_y,n_points,x,y)
    for i in range(51):
        mu = -0.5 + (0.02 * i)
        for j in range(51):
            std = 0.5 + (0.02 * j)
            assert(Z[j][i] == anly_NLLs[(mu,std)])
    assert(min(anly_NLLs.values()) == min_nll)
    # print(np.where(anly_NLLs == min_nll))
    for key, value in anly_NLLs.items():
        if value == min_nll:
            print("location of min_nll was found and compared")
            (mu,std) = key
            assert(best_mu == mu)
            assert(best_std == std)

test_max_likelihood_est()
