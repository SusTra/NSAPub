import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels
import statsmodels.api as sm
import statsmodels.stats.multitest as multi
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as multi
from statsmodels.sandbox.regression.predstd import wls_prediction_std

sns.set_style("white")
colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02']
#flatui = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33']
#flatui = ['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','#f0027f']
sns.set_palette(colors)


#plt.rcParams.update({'font.size': 12})


def dict_results(route_id="", counters="",model_type="", results="", R2_cross=0):
    if not results:
        return ['route_id', 
                'model_type', 
                'p', 
                'RSS', 
                #'ESS', 
                'R2', 
                'R2_train_mean', 
                'R2_train_std', 
                'R2_test_mean', 
                'R2_test_std', 
                #'R2_adj', 
                #'log-likelihood', 
                #'AIC', 
                #'BIC', 
                'params',
                'values', 
                'pvalues']
    
    return {'route_id': route_id, 
            'model_type': model_type,
            'p': results.f_pvalue, 
            'RSS': results.ssr,
            #'ESS': results.ess,
            'R2': results.rsquared,
            'R2_train_mean': R2_cross[0],
            'R2_train_std': R2_cross[1],
            'R2_test_mean': R2_cross[2],
            'R2_test_std': R2_cross[3],
            #'R2_adj': results.rsquared_adj,
            #'log-likelihood': results.llf,
            #'AIC': results.aic,
            #'BIC': results.bic,
            'params':counters,
            'values': list(results.params),
            'pvalues': list(results.pvalues)}

"""
def cross_validate(X, Y, training_size = 0.8, repeats = 100, constant = 'True'):
         
    R2s = np.zeros(repeats)
    R2s_test = np.zeros(repeats)

    if constant:
        X_fit = sm.add_constant(X,has_constant='add')
    else:
        X_fit = X
        
    for i in range(repeats):
        n = X.shape[0]
        
        idx_train = np.random.choice(np.arange(n), size=round(training_size*len(X)), replace=False)
        idx_test = np.array(list(set(np.arange(n)) - set(idx_train)))
        
        X_train = X_fit[idx_train,:]
        Y_train = Y[idx_train]
    
        model = sm.OLS(Y_train, X_train)
        results = model.fit()
        
        R2 = results.rsquared
        
        X_test = X_fit[idx_test,:]
        Y_test = Y[idx_test]
        
        Y_test_fit = results.predict(X_test)
        
        RSS_test = np.sum((Y_test - Y_test_fit)**2)
        RSS_tot_test = np.sum((Y_test - np.mean(Y_test))**2)
        R2_test = 1 - RSS_test/RSS_tot_test
        
        R2s[i] = R2
        R2s_test[i] = R2_test
        
    return np.mean(R2s), np.std(R2s), np.mean(R2s_test), np.std(R2s_test)
        
        
        #plt.plot(X, Y_fit, color=colors[route_id], alpha =0.5)
    #plt.show()
        #
        # predict
        # predict - Y_test
"""      
"""       
def fit_and_plot(X, Y, constant = True, trans="", route_id = 0, plot_on = True, save_fig = True):
    
    if constant:
        X_fit = sm.add_constant(X,has_constant='add')
    else:
        X_fit = X
    
    model = sm.OLS(Y, X_fit)
    results = model.fit()
    
    #R2_cross = cross_validate(X, Y)
    
    Y_test_fit = results.predict(X_fit)
    Y_test = Y
    
    if plot_on and X.shape[1] == 1:
        #counter_values = df2.values[:,3:]
        n_points = 10000
        #X_plot = np.linspace(0, np.max(X), n_points).reshape(n_points, 1)
        X_plot = np.linspace(0, np.max(X), n_points).transpose()
        if constant:
            X_plot_fit = sm.add_constant(X_plot, has_constant='add')
        else:
            X_plot_fit = X_plot


        sdev, lower, upper = wls_prediction_std(results, exog=X_plot_fit, alpha=0.05)
        Y_plot_fit = results.predict(X_plot_fit)

        if trans == "exp":
            Y_plot_fit = np.exp(Y_plot_fit)
            lower, upper = np.exp(lower), np.exp(upper)
            Y = np.exp(Y)
        elif trans == "log":
            Y_plot_fit = np.log(Y_plot_fit)
            lower, upper = np.log(lower), np.log(upper)
            Y = np.log(Y)
                
        plt.plot(X_plot, Y_plot_fit, color=colors[route_id], alpha =0.5)
        plt.fill_between(X_plot, lower, upper, color=colors[route_id], alpha=0.1)
        plt.plot(X, Y, '.', color=colors[route_id])
        if save_fig:
            if not trans:
                trans = "lin"
            plt.savefig("regression_results\\figs\\route_"+str(int(route_id))+"_regression_"+trans+".pdf", bbox_inches="tight")
            plt.savefig("regression_results\\figs\\route_"+str(int(route_id))+"_regression_"+trans+".png", bbox_inches="tight")
        
        plt.show()

    return results, R2_cross
"""    
    
def cross_validate(X, Y, idxs_tests, constant = 'True'):

    n = X.shape[0]
    #print(n)
    k = idxs_tests.shape[0]
    #print(k)
    
    R2s = np.zeros(k)
    R2s_test = np.zeros(k)
       
    if constant:
        X_fit = sm.add_constant(X,has_constant='add')
    else:
        X_fit = X
       
    for i in range(k):      
        idx_test = idxs_tests[i]
        idx_train = np.array(list(set(np.arange(n)) - set(idx_test)))
                
        X_train = X_fit[idx_train,:]
        Y_train = Y[idx_train]
    
        model = sm.OLS(Y_train, X_train)
        results = model.fit()
        
        #Y_fit = results.predict(X_fit)
        
        R2 = results.rsquared
        
        X_test = X_fit[idx_test,:]
        Y_test = Y[idx_test]
        
        Y_test_fit = results.predict(X_test)
        
        RSS_test = np.sum((Y_test - Y_test_fit)**2)
        RSS_tot_test = np.sum((Y_test - np.mean(Y_test))**2)
        R2_test = 1 - RSS_test/RSS_tot_test
        
        R2s[i] = R2
        R2s_test[i] = R2_test
        
    return np.mean(R2s), np.std(R2s), np.mean(R2s_test), np.std(R2s_test)

    
def fit_plot_validate(X, Y, idxs_tests, constant = True, trans="", route_id = 0, plot_on = True, save_fig = True):

    
    if constant:
        X_fit = sm.add_constant(X,has_constant='add')
    else:
        X_fit = X
    
    model = sm.OLS(Y, X_fit)
    results = model.fit()
    
    R2_cross = cross_validate(X,Y, idxs_tests)
    
    if plot_on and X.shape[1] == 1:
        #counter_values = df2.values[:,3:]
        n_points = 10000
        #X_plot = np.linspace(0, np.max(X), n_points).reshape(n_points, 1)
        X_plot = np.linspace(0, np.max(X), n_points).transpose()
        if constant:
            X_plot_fit = sm.add_constant(X_plot, has_constant='add')
        else:
            X_plot_fit = X_plot


        _, lower, upper = wls_prediction_std(results, exog=X_plot_fit, alpha=0.05)
        Y_plot_fit = results.predict(X_plot_fit)

        if trans == "exp":
            Y_plot_fit = np.exp(Y_plot_fit)
            lower, upper = np.exp(lower), np.exp(upper)
            Y = np.exp(Y)
        elif trans == "log":
            Y_plot_fit = np.log(Y_plot_fit)
            lower, upper = np.log(lower), np.log(upper)
            Y = np.log(Y)
                
        plt.plot(X_plot, Y_plot_fit, color=colors[route_id], alpha =0.5)
        plt.fill_between(X_plot, lower, upper, color=colors[route_id], alpha=0.1)
        plt.xlabel('Count [h$^{-1}$]')
        plt.ylabel('Pace [s/m]')
        plt.plot(X, Y, '.', color=colors[route_id])
        if save_fig:
            if not trans:
                trans = "lin"
            plt.savefig("regression_results\\figs\\route_"+str(int(route_id))+"_regression_"+trans+".pdf", bbox_inches="tight")
            plt.savefig("regression_results\\figs\\route_"+str(int(route_id))+"_regression_"+trans+".png", bbox_inches="tight")
        
        plt.show()

    return results, R2_cross
