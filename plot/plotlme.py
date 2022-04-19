
class PlotLme():
    
    '''Filter data to visualise in facetgrid
    missing data has been 'drop', not imputed. 
    Original dataset can be plotted.'''
    
    
    def __init__(self, df, ls_tsk, ls_feat):
        
        self.df = df
        self.ls_feat = ls_feat
        self.ls_tsk = ls_tsk
        
        

    def plot_lme(self, path):

        import os
        import pandas as pd
        import numpy as np

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        pltlme = PlotLme(self.df, self.ls_tsk, self.ls_feat)
        hrv = pltlme.filter_df()

        # Print and save fig in path
        
        # fig = plt.figure()
        
        g = sns.relplot(
            data=hrv, x='Time', y='pChange', col='Task', row='Feature',
            hue='Sex', style='Sex', kind='line', facet_kws={'sharey': False, 'sharex': False})

        ax1, ax2, ax3 = g.axes[0]
        ax4, ax5, ax6 = g.axes[1]
        ax7, ax8, ax9 = g.axes[2]

        ls_ax = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
        for ax in ls_ax:
            ax.axhline(0, color='k', linewidth=0.2)
            
        g.set_ylabels("Percent change (%) relative to baseline", clear_inner=False)
        
        
        fig6 = plt.gcf()
        
        plt.draw()
                       
        fig6.savefig(os.path.join(path, 'results', 'ResultsLME_plot.png'))

        
        
                
        
    def filter_df(self):

        hrv = self.df[(self.df.Feature.isin(self.ls_feat)) & (self.df.Task.isin(self.ls_tsk))].reset_index()

        hrv['Feature'] = hrv.Feature.cat.remove_unused_categories()
        hrv['Task'] = hrv.Task.cat.remove_unused_categories()
        hrv['Task'] = hrv.Task.cat.reorder_categories(self.ls_tsk)
        
        return hrv     
        

    def significant_bars(self, ls_fct, alpha):

        # not yet implemented

        import numpy as np
        import pandas as pd
        import seaborn as sns
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        
        import warnings
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        from pandas.core.common import SettingWithCopyWarning
        
        BufferAx = {}
        BufferAx_Task_Feat_Factor = {}

        for factor in ls_fct:

            WarnIgnore = [ConvergenceWarning, UserWarning, SettingWithCopyWarning, RuntimeWarning]

            for warn in WarnIgnore:
                warnings.simplefilter('ignore', warn)

            try:           
                for tsk in self.ls_tsk:                
                    i=0       

                    for feat in self.ls_feat:

                        LM = LinearModel(self.df)
                        fit_lme, summary_lme = LM.get_lme(tsk, feat, 'pChange', factor)
                        pval_list = LM.show_signif_values(alpha, fit_lme, factor) 

                        task = tsk + '_TSK'

                        #TODO: Save in txt file for report

                        # Assign axis to each feat-tsk combination
                        if task == 'MENTAL_TSK':
                            ax = g.axes[i][0]
                            i = i+1
                        elif task == 'NOISE_TSK':
                            ax = g.axes[i][0]
                            i = i+1      
                        elif task == 'CPT_TSK':
                            ax = g.axes[i][0]
                            i = i+1 

                        # assign a specific position for each time category (significant levels)
                        start, stop = LM.get_limits()
                        Time_list   = fit_lme.pvalues.index[start+1:stop].to_list()
                        xpos_list   = []

                        for j in range(len(Time_list)):
                            xpos=j/len(Time_list)
                            xpos_list.append(xpos)

                        xpos_avail = pd.DataFrame(zip(Time_list, xpos_list)) # Fonction anonyme zip()
                        pvalue_ix  = pval_list.index                    
                        xpos = xpos_avail[(xpos_avail[0].isin(pvalue_ix))] #filter out position for significant values                    
                        ypos = max(hrv.pChange[(hrv.Feature==feat) & (hrv.Task==task)])+1

                        # Gather all position data in a dict

                        BufferAx_Task_Feat_Factor = {
                            "xpos_avail": xpos_avail,
                            "ax": ax,
                            "pvalue": pval_list,
                            "xpos": xpos,
                            "ypos": ypos
                        }

                        keyname = task+'_'+feat+'_'+factor
                        BufferAx[keyname] = BufferAx_Task_Feat_Factor 

            except np.linalg.LinAlgError as e:
                if 'Singular matrix' in str(e):
                    pass
                else:
                    raise

            return BufferAx



