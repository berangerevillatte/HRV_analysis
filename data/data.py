def clean_df(df,conn):


    # Find and replace dataframe errors

    import numpy as np
    import pandas as pd
    from data.sql import sql 


    df.Time = np.where(df.Time != 'T.00', df.Time, 't00')

    col = (4,5,6,7,9)

    for n in col:
        for i in range(df.shape[0]):
            if type(df.loc[i, df.columns[n]]) == str:
                df.loc[i, df.columns[n]] = df.loc[i, df.columns[n]].replace(',', '.').replace("'", '')

        df[df.columns[n]] = df[df.columns[n]].astype(float)

    var_cat = ['Participant', 'Sex', 'Task', 'Feature', 'Time']

    for var in var_cat:
        if var == 'Time':
            time_lvl = ['Baseline', '-60', '-30', '0', '30', '60', '90', '120','150', '180', '210','240', '270', '300','330']  
            df.Time = pd.Categorical(list(df.Time)).rename_categories(time_lvl)
            df.Time = pd.Categorical(list(df.Time)).reorder_categories(time_lvl, ordered=True)

        df[var] = pd.Categorical(df[var])
        
    sql.df2sql('df', df, conn)   # save a clean version of df in SQLite db
        
    return df, time_lvl




def subset_df(df, task, feature, baseline = True): #task and related baseline   # Add argument 'baseline' = True

   
    if baseline:
        mask = df['Task'].str.contains(task)
        df2 = df[mask][df.Feature == feature]
    elif not baseline:
        df2 = df[(df.Feature==feature) & (df.Task == task)]

    return df2

        
