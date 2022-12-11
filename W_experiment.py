import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time
import csv

#M = experiment size
#lambda = mean of  distribution
#returns an array of random samples following a gaussian distribution
def Inverse_transform_sampling(M,lambda_):
    samples = []

    for i in range(M):
        u = np.random.uniform(0, 1)
        x = stats.cauchy.ppf(u, lambda_)
        samples.append(x)

    return(np.array(samples))

#returns mean value of distribution at which likelihood is maximized
def max_likelihood_est(samples,min_x,max_x,min_y,max_y,n_points,x,y):
    X, Y = np.meshgrid(x, y)
    Z = 0.0
    for sample in samples: # sum the NLL of each data point in sample
        Z += cauchyPDF([sample,X,Y])
    min_nll = np.min(Z) #get min NLL value
    (row,col) = np.where(Z == min_nll) #get loc of min NLL value
    assert(Z[row[0]][col[0]] == min_nll)
    best_mu = convert_index_to_val(col[0],min_x,max_x,n_points)
    best_std = convert_index_to_val(row[0],min_y,max_y,n_points)
    best_mu_loc = col[0]
    best_std_loc = row[0]
    # print("best mu ",best_mu)
    # print("best std ",best_std)
    # print("best mu and std ",best_mu," ", best_std)
    # print("smallest - nll ",min_nll)
    plt.contour(X, Y, Z, colors='black')
    plt.xlabel("mean")
    plt.ylabel("scale")
    plt.title("Contour of fixed NLL")
    plt.show()
    return (best_mu,best_std,Z,min_nll,best_mu_loc,best_std_loc)

def profile_likelihood(Z,min_nll,x):
    Z = np.transpose(Z)
    profile_ll = [np.min(row) - min_nll for row in Z]
    plt.scatter(x,profile_ll)
    plt.xlabel("mean")
    plt.ylabel(r'$\Delta Min NLL$')
    plt.title("1D Likelihood")
    plt.show()
    return (Z,profile_ll,np.max(profile_ll))

def conf_int(pll,min_x,max_x,n_points,best_mu,best_mu_loc):
    profile_ll = np.array(pll) #convert 1d profile likelihood array to np

    delta_nll = 2.0 #2 sigma significance
    # min_loc = np.where(profile_ll == 0.0)
    # print("min loc ",min_loc[0][0])
    # print(convert_index_to_val(min_loc[0][0],min_x,max_x,n_points))
    # assert(convert_index_to_val(min_loc[0][0],min_x,max_x,n_points) == best_mu)
    #calculate each value in array's distance from value of deltaNLL = 2
    diff_array = np.absolute(profile_ll - delta_nll)

    #find index in left,righthalf of diff_array of value closest to 0
    low_bound = diff_array[0:best_mu_loc].argmin()
    up_bound = diff_array[best_mu_loc+1:n_points].argmin()
    up_bound = best_mu_loc + 1 + up_bound #shift up_bound by adding center of array
    low_bound = convert_index_to_val(low_bound,min_x,max_x,n_points)
    up_bound = convert_index_to_val(up_bound,min_x,max_x,n_points)
    # print("low and up bound of conf int ",low_bound," ",up_bound)
    return(low_bound,up_bound)

#for each mu sigma pair,loop thru data points
#calculate -Nll at each point and add all NLLs for each point together
#result is a -Nll for each mu and sigma pair
def analytic_max_likelihood_est(samples):
    NLLS = {}
    #for one mu and sigma pair
    for i in range(0,51):
        mu = -0.5 + (0.02 * i)
        for j in range(0,51):
            std = 0.5 + (0.02 * j)
            nll = 0.0
            for sample in samples:
                nll += cauchyPDF([sample,mu,std])
                # if i == 5 and j ==49:
                    # print(nll)
                    # print(mu)
                    # print(std)
            NLLS[(mu,std)] = nll

    return NLLS

def convert_index_to_val(index,min,max,n_points):
    # print("convert call ",index," ",min," ",max," ",n_points)
    return min + index * (max-min)/(n_points - 1)

def cauchyPDF(params):
    # x,y,z = params
    # return x**2 + y**3 + z**3
    x,loc,scale = params
    # print(params)
    # return -1 * np.log((1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5*((x-mu)/std)**2))
    return -1 * np.log(1/((np.pi * scale)*(1 + ((x - loc)/scale)**2)))

def likelihood_ratio(Z,best_mu,best_std,min_x,max_x,min_y,max_y,n_points,best_mu_loc,best_std_loc,min_nll):
    #likelihood of observing data given the true mean and std
    L_null = Z[50][50]
    # assert(convert_index_to_val(50,-0.5,0.5,101) == 0.0)
    #likelihood of observing data given MLE for mean and std
    L_mle = Z[best_mu_loc][best_std_loc]
    # assert(L_mle == min_nll)
    # print("here ",Z[best_mu][best_std])
    #divide and take -2 ln
    modified_ratio = (2 * L_null) - (2 * L_mle)
    #value of chi2 at a significance level of 0.05 assuming 1 dof
    c_a = stats.chi2.ppf(q = 1 - 0.05,df = 1)
    #for a 95% confidence level, determine the value of the likelihood test above which we should not reject null
    #value of likelihood below which we should reject null hypothesis
    if modified_ratio > c_a:
        return 1 #reject null hypothesis
    else:
        return 0

def experiment(M):
    random_samples = Inverse_transform_sampling(M,lambda_ = 0)
    # print(random_samples)
    # analytic_max_likelihood_est(random_samples)
    min_x = -0.5
    max_x = 0.5
    min_y = 0.5
    max_y = 1.5
    n_points = 101
    x = np.linspace(min_x,max_x,n_points) #mean centered at 0
    y = np.linspace(min_y,max_y,n_points) #std centered at 1
    (best_mu,best_std,Z,min_nll,best_mu_loc,best_std_loc) = max_likelihood_est(random_samples,min_x,max_x,min_y,max_y,n_points,x,y)
    (transposed_Z,profile_ll,max_ll) = profile_likelihood(Z,min_nll,x)
    (low_bound, up_bound) = conf_int(profile_ll,min_x,max_x,n_points,best_mu,best_mu_loc)
    # print("low bound ",low_bound)
    # print("up bound ",up_bound)
    plot_estimators(x,profile_ll,n_points,best_mu,max_ll,low_bound,up_bound)
    within_CI = 0
    if low_bound <= 0.0 <= up_bound:
        within_CI = 1
    reject_null = likelihood_ratio(transposed_Z,best_mu, best_std,min_x,max_x,min_y,max_y,n_points,best_mu_loc,best_std_loc,min_nll)
    return (reject_null,within_CI)

def plot_estimators(x,profile_ll,n_points,best_mu,max_ll,low_bound,up_bound):
    plt.scatter(x,profile_ll)
    mle_x_vals = np.full(n_points,best_mu)
    truemean_x_vals = np.zeros(n_points)
    y_vals = np.linspace(0,max_ll,n_points)
    plt.plot(mle_x_vals,y_vals,color="red",label="MLE")
    plt.plot(truemean_x_vals,y_vals,'--',color="pink",label="True Mean")
    plt.xlim(-0.6,0.6)
    plt.title("1D Likelihood")
    plt.xlabel("mean")
    plt.ylabel(r'$\Delta Min NLL$')
    plt.legend()
    plt.show()

    plt.scatter(x,profile_ll)
    mle_x_vals = np.full(n_points,best_mu)
    truemean_x_vals = np.zeros(n_points)
    low_bound_x = np.full(n_points,low_bound)
    up_bound_x = np.full(n_points,up_bound)
    y_vals = np.linspace(0,max_ll,n_points)
    plt.plot(mle_x_vals,y_vals,color="red",label="MLE")
    plt.plot(truemean_x_vals,y_vals,'--',color="pink",label="True Mean")
    plt.plot(low_bound_x,y_vals,color="blue",linewidth=2,label="Lower bound of confidence interval")
    plt.plot(up_bound_x,y_vals,color="blue",linewidth=2,label="Upper bound of confidence interval")
    plt.xlim(-0.6,0.6)
    plt.xlabel("mean")
    plt.ylabel(r'$\Delta Min NLL$')
    plt.title("1D Likelihood")
    plt.legend()
    plt.show()

def plot_overall():
    with open("experiment_results.csv", 'r') as file:
        reader = csv.reader(file)
        labels = []
        hypothesis_test = []
        within_ci = []
        for row in reader:
            labels.append(row[0])
            hypothesis_test.append(float(row[2]))
            within_ci.append(float(row[3]))
        x = np.arange(len(labels))
        print(x)
        width = 0.35
        fig,ax = plt.subplots()
        rects1 = ax.bar(x - width/2, hypothesis_test,width,label = "hypothesis test",color="blue")
        rects2 = ax.bar(x + width/2, within_ci,width,label = "confidence interval test",color="orange")
        ax.set_xlabel('Data points per experiment group')
        ax.set_title('Rates of tests passed by experiment group')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Runs/Total Runs')
        ax.legend()
        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        plt.show()

# total_experiments = 500
total_experiments = 1
# for i in range(3,7):
for i in range(1):
    # M = 10**i
    M = 1000
    rejections = 0.0
    times_within_CI = 0.0
    t0 = time.time()
    for i in range(total_experiments):
        (reject_null,within_CI) = experiment(M)
        rejections += reject_null
        times_within_CI += within_CI
    t1 = time.time() - t0
    print(rejections/total_experiments)
    print(times_within_CI/total_experiments)
    print("time in seconds ",t1," time in min ",t1/60)
    row = [M,total_experiments,rejections/total_experiments,times_within_CI/total_experiments,t1,t1/60]
    with open("experiment_results.csv", 'a') as file:
        writer = csv.writer(file)
        writer.writerow(row)
    plot_overall()
