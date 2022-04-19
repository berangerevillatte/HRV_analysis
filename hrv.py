if __name__ == '__main__':


    #@author: Berangere Villatte, Université de Montréal
    
    # load libraries
    print('Loading libraries...')
    
    import os
    import pandas as pd
    import numpy as np

    import sqlite3
    from sqlite3 import connect

    from data import data
    from data.sql import sql
    from data.handlena import HandleNA
    from plot.plotlme import PlotLme

    from stats import statdesc
    from stats.linearmodel import LinearModel
    from stats.machinelearning import MachineLearning

    import matplotlib
    import matplotlib.pyplot as plt

    path = os.getcwd()

    if not 'results' in os.listdir(path):
        os.mkdir(os.path.join(path, 'results'))
 
    
    # Define constants:
    ls_tsk, ls_feat, ls_fct = ['MENTAL', 'NOISE', 'CPT'],['MeanNN', 'RMSSD', 'SDNN'],['Time', 'Sex']
    
    task     = 'CPT_TSK'
    feature  = 'RMSSD'
    variable = 'pChange'
    group    = 'Sex'
    
    # Import database
    path = os.getcwd()

    # Initialize db
    db = 'hrv_sql.db'

    print('\nConnecting a SQLite database...\n')

    try:
        conn = connect(db, check_same_thread = False)
        cursor = conn.cursor()
        print("Database created and Successfully Connected to SQLite")

        cursor.execute("select sqlite_version();")
        record = cursor.fetchall()
        print("SQLite Database Version is: ", record)
        cursor.close()

    except sqlite3.Error as error:
        print("Error while connecting to sqlite", error)
    

#    conn = connect(db, check_same_thread = False)
    df = pd.read_excel('data/BerangereVillatte_base_données_psy4016-H22_20220204.xlsx')
    df, time_lvl = data.clean_df(df, conn)
    
    ls_tsk, ls_feat, ls_fct = ['MENTAL_TSK', 'NOISE_TSK', 'CPT_TSK'],['MeanNN', 'RMSSD', 'SDNN'],['Time', 'Sex']

#     Uncomment if you want to see what's the best model
#     model = statdesc.get_stat_model(df)
#     print(model)

    print('\nRunning Linear Mixed Effects model...\n')
    
    lm = LinearModel(df)   
    lm.all_taskfeat_lme(ls_tsk, ls_feat, ls_fct, path)

    print('\nThese listed files have been saved in "results" folder')
    
    pltlme = PlotLme(df, ls_tsk, ls_feat)
    pltlme.plot_lme(path)

    plt.show()
    plt.close()

    print(f'\nLinear Mixed Model plot saved in .png in "results" folder')
    
    ML = MachineLearning(df, task, feature, variable, group, time_lvl, path)
    ML.plot_data()
    plt.show()
    plt.close()
    
    ML.knn_kfold()
    plt.show()
    plt.close()
    
    ML.knn_split(test_size=0.33)
    plt.show()
    plt.close()
    
    ML.PCA()
    plt.show()
    plt.close()

    print(f'\nAutomatic Learning data and plots saved in "results" folder')

#    import zipfile
#    from zipfile import ZipFile

#    with ZipFile("results.zip", "w") as newzip:
#            newzip.write(os.path.join(path,"results"))
            

#    print(f'\nResults are archived in a "lme_hrv_results.zip" folder within {path}')

    

    
          
