class LinearModel:        # TODO: replace all 'data' by 'self'
    
    
    
    def __init__(self, df):
        
        self.df = df
        
       
    def get_lme(self, task, feature, variable, factor):

        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from data import data
        
        self.df2 = data.subset_df(self.df, task, feature, baseline = True)
        
        if task == 'CPT':           # Remove unused categories, CPT_TASK has only 11 categories of Time (lasts 210s vs. 330s)
            self.df2['Time'] = self.df2.Time.cat.remove_unused_categories()

        # Linear mixed-effects model with random effects attrubuted to participants variance
        lme = smf.mixedlm("pChange ~ Time*Sex", self.df2, groups=self.df2["Participant"], missing = 'drop')
        fit_lme = lme.fit(method = ["lbfgs"]) # lbfgs = recommended method by https://www.statsmodels.org/dev/examples/notebooks/generated/mixed_lm_example.html
        summary_lme = fit_lme.summary()

        return fit_lme, summary_lme

        # read this for residuals https://www.reneshbedre.com/blog/anova.html
        
        
    def all_taskfeat_lme(self, task_list, feature_list, factor_list, path):

        import numpy as np
        import pandas as pd
        import os
   
        import warnings
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        from pandas.core.common import SettingWithCopyWarning
        
        WarnIgnore = [ConvergenceWarning, UserWarning, SettingWithCopyWarning, RuntimeWarning]
        
        for factor in factor_list:
            for warn in WarnIgnore:
                warnings.simplefilter('ignore', warn)

            try:
                for task in task_list:
                    for feature in feature_list:
                        
                        LM = LinearModel(self.df)
                        fit_lme, summary_lme = LM.get_lme(task, feature, 'pChange', factor)
                        

                        filename = 'lme_'+task+feature+'pChange'+factor+'.txt'
                        open(os.path.join(path, 'results',filename), 'w').close()
                        f = open(os.path.join(path, 'results',filename), 'a')
                        f.writelines(str(summary_lme))
                        f.close()
                             
                        print(filename) #TODO: retirer cette ligne

            except np.linalg.LinAlgError as e:                
                if 'Singular matrix' in str(e):
                    pass
                else:
                    raise
                    
                    

    def get_limits(self, factor):

        import numpy as np
        
        if factor == 'Time': 
            start = 0
            stop = len(np.unique(self.df['Time']))
            
        elif factor == 'Sex':
            start = len(np.unique(self.df['Time']))+1
            stop = 2*len(np.unique(self.df['Time']))+1
                    
        return start, stop

    
    def show_signif_values(self, alpha, fit_lme, factor): # add fit_lme in arg
            
        LM = LinearModel(self.df)
        start, stop = LM.get_limits(factor)
        pvalues = fit_lme.pvalues[start:stop]

        signif_val = pvalues[pvalues < alpha]


        return signif_val    

