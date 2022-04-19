class MachineLearning():
    
    '''1. Compute supervised and unsupervised 
    automatic learning from subsets of df
    
    Supervised:   KNN with k-fold cross-validation 
                  and a comparison with train_test_split() method
    
    Unsupervised: PCA (Principal Component Analysis)
    
    2. Plot algorithm results'''
    
    
    def __init__(self, df, task, feature, variable, group, time_lvl, path):
        
        self.df = df
        self.task = task
        self.feature = feature
        self.variable = variable
        self.group = group
        self.time_lvl = time_lvl
        self.path = path
        
        
    def filter_data(self):

        import numpy as np
        
        data = self.df[(self.df.Task==self.task) & (self.df.Feature==self.feature)].reset_index()
        data = data[data[self.variable].notna()]
                
        cat  = np.unique(self.df[self.group])
        ncat = len(cat)
        cat_bin = np.linspace(1,ncat,ncat).astype(int)
        
        target_varname = f'{self.group}_target'
        train_varname  = f'{self.group}_train'
        
        data[target_varname] = data[self.group]
        data[train_varname]  = data[self.group].replace(cat, cat_bin)
        
        if self.task == 'CPT_TSK':
            timecat = self.time_lvl[1:-4]
        else:
            timecat = self.time_lvl[1:]
            

        data['Time'] = data.Time.replace(np.unique(data.Time).tolist(), timecat)
        data['Time'] = data.Time.astype('float')       
 
        y = data[train_varname].to_numpy()
        X = data[[self.variable, 'Time']].to_numpy()
        
        return X, y, data
        
        
    def plot_data(self):

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import os
        
        ML = MachineLearning(self.df, self.task, self.feature, self.variable, self.group, self.time_lvl, self.path)
        X, y, data = ML.filter_data()
        
        # fig is saved in path
    
        # fig = plt.figure() # pour sauvegarder, il faut créer une figure    
        
        plt.scatter(X[:, 1], X[:, 0], c=y, s=50, cmap='cool')
        plt.title('Automatic Learning data\n{0} for {1} between {2} ({3} [cyan] / {4} [magenta])'
                  .format(self.feature, self.task, self.group, 
                          np.unique(self.df[self.group])[0], np.unique(self.df[self.group])[1]));
        
        if self.variable == 'pChange':
            legendname = 'Percent change (%) in {0} \n relative to baseline'.format(self.feature)
        else:
            legendname = '{0} absolute {1}'.format(self.feature, variable.lower())
        
        plt.ylabel(legendname)
        plt.xlabel(" Time (s)")
        plt.axhline(0, color='k', linewidth=0.7);
        plt.xlim(-60-10, 230+10);
        plt.xticks(np.arange(-60, 240, 30));

        
        fig1 = plt.gcf()
       # plt.show()
        plt.draw()
        fig1.savefig(os.path.join(self.path, 'results', 'DataMachineLearning_plot.png'))
       # plt.close()
        
    def knn_kfold(self):
    
        '''KNN with k-fold cross-validation
        2 models (k=2)

        For each model:
        - Returns accuracy and classification report in a .txt file in current working directory
        - Returns plot of confusion matrix

        Source: https://scikit-learn.org/stable/modules/cross_validation.html'''
        import numpy as np
        import pandas as pd
        import os
        
        import sklearn
        from sklearn.model_selection import KFold
        #from sklearn import KFold
        from sklearn.neighbors import KNeighborsClassifier as KNN
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        import seaborn as sns
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        
        ML = MachineLearning(self.df, self.task, self.feature, self.variable, self.group, self.time_lvl, self.path)
        X, y, data = ML.filter_data()
        target_varname = f'{self.group}_target'
    
        i=1
        j=1

        kf = KFold(n_splits=2)

        for train, test in kf.split(X):
            (train, test)

            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]  

            model = KNN(n_neighbors=3)
            model.fit(X_train, y_train)
            y_fit = model.predict(X_test)
            
            # Evaluate accuracy and confusion matrix

            acc = f'Model {i} accuracy is: {round(accuracy_score(y_test, y_fit)*100)}%'
            c   = classification_report(y_test, y_fit, target_names=np.unique(data[target_varname]))
            mat = confusion_matrix(y_test, y_fit)
            
            # Save data from algorithm in .txt format

            filename = f'SupervisedML_KNN_kfCV_accuracy_model{i}.txt'
            open(os.path.join(self.path, 'results', filename), 'w').close()

            f = open(os.path.join(self.path, 'results', filename), 'a')       
            f.writelines([acc, '\n\n', str(c)])
            f.writelines(['\n\nConfusion Matrix:\n', str(mat)])
            f.close() 
            
            # Save confusion matrix plot            
            
            # fig = plt.figure() # pour sauvegarder, il faut créer une figure        
           
            
            sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                    xticklabels=np.unique(data.Sex_target),
                    yticklabels=np.unique(data.Sex_target))
            plt.xlabel(f'true label ({self.group})')
            plt.ylabel(f'predicted label ({self.group})')
            plt.title('Model'+str(i)+'\n{0} for {1} between {2}'
                  .format(self.feature, self.task, self.group))

            
            fig2 = plt.gcf()
         #   plt.show()
            plt.draw()
         #   plt.close()
            
            fig2.savefig(os.path.join(self.path, 'results', f'ConfMatx_plot_model{i}.png'))     

            j=j+2
            i=i+1

            
    def knn_split(self, test_size=0.33):
        
        '''KNN with a chosen test size, default=0.33
        
        Random_state=0 for reproducibility
        
        Trained on 1-test_size
        Tested on test_size
        
        1 model:
        - Returns accuracy and classification report in a .txt file in current working directory
        - Returns plot of confusion matrix

        Source: https://scikit-learn.org/stable/modules/cross_validation.html'''
        import numpy as np
        import pandas as pd
        import os
        
        import sklearn
        from sklearn.model_selection import GridSearchCV, train_test_split
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        from sklearn.neighbors import KNeighborsClassifier as KNN
        
        import seaborn as sns
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        
        ML = MachineLearning(self.df, self.task, self.feature, self.variable, self.group, self.time_lvl, self.path)
        X, y, data = ML.filter_data()
        target_varname = f'{self.group}_target'
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
        
        model = KNN(n_neighbors=3)
        model.fit(X_train, y_train)
        y_fit = model.predict(X_test)
        acc   = f'Model accuracy is: {round(accuracy_score(y_test, y_fit)*100)}%'
        c     = classification_report(y_test, y_fit, target_names=np.unique(data[target_varname]))
        mat   = confusion_matrix(y_test, y_fit)
        
        filename = 'SupervisedML_KNN_split_accuracy.txt'
        open(os.path.join(self.path, 'results', filename), 'w').close()

        f = open(os.path.join(self.path, 'results',filename), 'a')       
        f.writelines([acc, '\n\n', str(c)])
        f.writelines(['\n\nConfusion Matrix:\n', str(mat)])
        f.close()
        
        # fig = plt.figure() # pour sauvegarder, il faut créer une figure        


        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=np.unique(data.Sex_target),
                yticklabels=np.unique(data.Sex_target))
        plt.xlabel(f'true label ({self.group})')
        plt.ylabel(f'predicted label ({self.group})')
        plt.title('KNN classifier\n{0} for {1} between {2}'
              .format(self.feature, self.task, self.group))
        
        fig3 = plt.gcf()
     #   plt.show()
        plt.draw()
     #   plt.close()
        
        fig3.savefig(os.path.join(self.path, 'results', 'ConfMatx_KNNClassifier_plot.png'))
        
    def PCA(self):
        
        '''Principal Component Analysis (PCA) 
        
        n_component=2 (must be the == ngroups)
        
        - Returns plot of the PCA decomposition, between the n components
        
        Source: https://scikit-learn.org/stable/modules/cross_validation.html'''
        import os
        import numpy as np
        import pandas as pd
        import sklearn
        from sklearn.decomposition import PCA
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        import seaborn as sns
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        
        ML = MachineLearning(self.df, self.task, self.feature, self.variable, self.group, self.time_lvl, self.path)
        X, y, data = ML.filter_data()
        
        model = PCA(n_components=2)
        model.fit(X)
        X_2D = model.transform(X)

        data['PCA1'] = X_2D[:, 0]
        data['PCA2'] = X_2D[:, 1]

        # fig = plt.figure()
        sns.lmplot(x = "PCA1", y = "PCA2", hue=self.group, data=data, fit_reg=False)
        plt.title('PCA decomposition: RMSSD for CPT_TSK')
        # fig.savefig('UnsupervisedML_PCA_plot.svg')

        fig4 = plt.gcf()
    #    plt.show()
        plt.draw()
        fig4.savefig(os.path.join(self.path, 'results', 'UnsupervisedML_PCA_plot.png'))
    #    plt.close()

