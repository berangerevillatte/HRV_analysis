# TODO: Raccourcir le temps de traitement,
#       np au lieu de pd, stock ds une liste 
#       vide [] plutôt que dataframe

def shapiro_test(df, conn):
    
    from stats import statdesc
    import scipy
    from scipy import stats
    import numpy as np
    import pandas as pd
    from data.sql import sql 
    
    # 1. Check for normal distribution of pChange in every task x feature combination
    shapiro_data = pd.DataFrame(columns = ['Task', 'Feature', 'N', 'Value Type', 'Stat', 'p-Value'])

    i = 0
    while i < df.shape[0]:
        for t in np.unique(df.Task):
            for f in np.unique(df.Feature):
                new_df  = df[(df.Task == t) & (df.Feature == f)]
                new_df = pd.DataFrame.dropna(new_df)

                if 'BASELINE' in t:
                    val = ['Value',] # pChange is not considered for baselines, since relative to baseline
                else:
                    val = ['pChange', 'Value']

                for v in val:
                    shapiro = scipy.stats.shapiro(new_df[v])
                    headers = ['Task', 'Feature', 'N', 'Value Type', 'Stat', 'p-Value']
                    values = [t, f, new_df.shape[0], v, shapiro.statistic, shapiro.pvalue]
                    shapiro_data.loc[i, headers] = values

                    i = i+1
                
    sql.df2sql('ShapiroWilk_test', shapiro_data, conn)
                    
    return shapiro_data


def get_stat_model(df):

    from stats import statdesc
        
    # 2. Choose model according to distribution and missing data

    shapiro_data = statdesc.shapiro_test(df)
    if df.Value.isnull().any() or df.pChange.isnull().any(): # is any missing data within Value or pChange
        model = 'lme'
    else:
        if (shapiro_data['p-Value'] < 0.05).any(): # not normally distributed if p<0.05
            model = 'lme'
        else:
            model = 'mixed_anova'

    return model
