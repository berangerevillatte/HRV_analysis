## does not work because returns a dataframe without baseline

class HandleNA():   # il faut mettre les argument de __init__ quand on appelle la classe
                    # eg. impute = IMPUTE(df, task, feature, variable, strategy)
                    # impute.replace_missingdata() > va sortir df3 selon les arguments entrés dans IMPUTE    
       
    
    
    
    def __init__(data, df, task, feature, variable, strategy): # data est le nom pour appeler le contenu de cette fonction
        
        # baseline shouldn't be considered when imputing missing data
        data.df       = subset_df(df, task, feature, baseline = False) #TODO: subset_df() doit être dans une fonction f2 > f2.subset_df
        data.feature  = feature 
        data.task     = task
        data.variable = variable
        data.strategy = strategy
        data.strategy_list = ['mean', 'most_frequent', 'median']
        
    def replace_missingdata(data):  
        
        df2 = copy.deepcopy(data.df)

        if not data.strategy in data.strategy_list:
            print(f'Error: Unvailable strategy. Choose within {data.strategy_list}')
        else:
            imp = impute.SimpleImputer(missing_values = np.nan, strategy = data.strategy)
            data.df[data.variable] = imp.fit_transform(data.df[[data.variable]])
            
        return data.df

    def compare_imputer_strategies(data):

        na_ix = data.df[data.variable][data.df[data.variable].isna()].index.tolist()

        if not data.df[data.variable].isna().any():
            print(f'There isn\'t any missing value in {data.variable} for {data.task}, {data.feature}')
        else:        
            print('Initial value : ', data.df[data.variable].loc[na_ix[0]], '\n')

            for data.strategy in data.strategy_list:
                
                df2   = copy.deepcopy(data.df)
                na_ix = df2[data.variable][df2[data.variable].isna()].index.tolist()
                imp   = impute.SimpleImputer(missing_values = np.nan, strategy = data.strategy)

                df2[data.variable] = imp.fit_transform(df4[[data.variable]])
                
                print(f'Strategy: {data.strategy},', df2[data.variable].loc[na_ix[0]], '\n')
                
