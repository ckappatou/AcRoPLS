##
# InitialExample.py
#
# Simulated spectroscopy example
#
# C.D. Kappatou, J.Odgers, S. Garcia, R.Misener: "Optimization Methods for Developing Efficient Chemometric Models", 2022.
# 
##

import pandas as pd
import numpy as np
import pyphi as phi
import matplotlib.pyplot as plt

import itertools
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from entmoot.optimizer.entmoot_minimize import entmoot_minimize
from cycler import cycler

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import momMatch as mm

def plot_spectra(X, *, xaxis=False, plot_title='Main Title', xaxis_label='X- axis', yaxis_label='Y- axis', pltName=False):
    """
    Simple way to plot Spectra with Bokeh.
    Programmed by Salvador Garcia-Munoz
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    Modified by Chryssa Kappatou

    X:      A numpy array or a pandas object with Spectra to be plotted
    xaxis:  wavenumbers or wavelengths to index the x axis of the plot
            * ignored if X is a pandas dataframe *

    optional:
    plot_title
    xaxis_label
    yaxis_label

    """

    if isinstance(X, pd.DataFrame):
        x = X.columns[1:].tolist()
        x = np.array(x)
        x = np.reshape(x, (1, -1))
        y = X.values[:, 1:].astype(float)
    elif isinstance(X, np.ndarray):
        y = X.copy()
        if isinstance(xaxis, np.ndarray):
            x = xaxis
            x = np.reshape(x, (1, -1))
        elif isinstance(xaxis, list):
            x = np.array(xaxis)
            x = np.reshape(x, (1, -1))
        elif isinstance(xaxis, np.bool):
            x = np.array(list(range(X.shape[1])))
            x = np.reshape(x, (1, -1))

    fig, ax = plt.subplots()
    ax.set_xlabel(xaxis_label, fontsize=22)
    ax.set_ylabel(yaxis_label, fontsize=22)
    ax.tick_params(labelsize=18)
    x1 = x.tolist() * y.shape[0]
    y1 = y.tolist()
    for i in range(len(x1)):
        ax.plot(np.array(x1[i]).astype(int), y1[i])
    ax.set_xlim(0,200)
    ax.hlines(y=0, xmin=min(np.array(x1[1]).astype(int)), xmax=max(np.array(x1[1]).astype(int)), color='k', linestyles=':', zorder=1000)
    plt.tight_layout()
    if pltName:
        plt.savefig(str(pltName)+".pdf")
    else:
        plt.savefig("CS1_spectra.pdf")
    plt.show()
    return

def predvsobsC(mvmobj, X, Y, *, CLASSID=False, colorby=False, x_space=False, plot_title='Main Title', xaxis_label='X- axis', yaxis_label='Y- axis', na=False, SNV=True, excludeCat=False):
    """
    Plot observed vs predicted values
    by Salvador Garcia-Munoz
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    Modified by Chryssa Kappatou

    mvmobj: A model created with phi.pca or phi.pls

    X/Y:     Data [numpy arrays or pandas dataframes]

    optional:
    CLASSID: Pandas Data Frame with classifiers per observation, each column is a class

    colorby: one of the classes in CLASSID to color by

    x_space: = 'False' will skip plotting the obs. vs pred for X *default*
               'True' will also plot obs vs pred for X

    """
    num_varX = mvmobj['P'].shape[0]

    if isinstance(X, np.ndarray):
        X_ = X.copy()
        ObsID_ = []
        for n in list(np.arange(X.shape[0]) + 1):
            ObsID_.append('Obs #' + str(n))
        XVarID_ = []
        for n in list(np.arange(X.shape[1]) + 1):
            XVarID_.append('Var #' + str(n))
    elif isinstance(X, pd.DataFrame):
        X_ = np.array(X.values[:, 1:]).astype(float)
        ObsID_ = X.values[:, 0].astype(str)
        ObsID_ = ObsID_.tolist()

    if 'varidX' in mvmobj:
        XVar = mvmobj['varidX']
    else:
        XVar = []
        for n in list(np.arange(num_varX) + 1):
            XVar.append('XVar #' + str(n))

    if 'Q' in mvmobj:
        num_varY = mvmobj['Q'].shape[0]
        if 'varidY' in mvmobj:
            YVar = mvmobj['varidY']
        else:
            YVar = []
            for n in list(np.arange(num_varY) + 1):
                YVar.append('YVar #' + str(n))

    if isinstance(Y, np.ndarray):
        Y_ = Y.copy()
    elif isinstance(Y, pd.DataFrame):
        Y_ = np.array(Y.values[:, 1:]).astype(float)

    if 'Q' in mvmobj:
        pred = phi.pls_pred(X_, mvmobj)
        yhat = pred['Yhat']
        if x_space:
            xhat = pred['Xhat']
        else:
            xhat = False
    else:
        x_space = True
        pred = phi.pca_pred(X_, mvmobj)
        xhat = pred['Xhat']
        yhat = False

    # No CLASSIDS
    if isinstance(CLASSID, bool):
        for i in list(range(Y_.shape[1])):
            x_ = Y_[:, i]
            y_ = yhat[:, i]
            min_value = np.nanmin([np.nanmin(x_), np.nanmin(y_)])
            max_value = np.nanmax([np.nanmax(x_), np.nanmax(y_)])
            plt.figure()
            plt.xlabel(xaxis_label)
            plt.ylabel(yaxis_label)
            plt.xticks([])
            plt.yticks([])
            plt.scatter(x_, y_)
            plt.plot([min_value, max_value], [min_value, max_value], color='cyan', linestyle='dashed')
        plt.show()

    else:  # YES CLASSIDS
        if isinstance(CLASSID, np.ndarray):
            Classes_ = np.unique(colorby).tolist()
            classid_ = list(colorby)
        else:
            Classes_ = np.unique(CLASSID[colorby]).tolist()
            classid_ = list(CLASSID[colorby])
        different_colors = len(Classes_)

        for i in list(range(Y_.shape[1])):
            x_ = Y_[:, i]
            y_ = yhat[:, i]
            min_value = np.nanmin([np.nanmin(x_), np.nanmin(y_)])
            max_value = np.nanmax([np.nanmax(x_), np.nanmax(y_)])
            fig, ax = plt.subplots()
            plt.xlabel(xaxis_label, fontsize=22)
            plt.ylabel(yaxis_label, fontsize=22)
            plt.xlim([0.01, 0.299])
            plt.ylim([0.01, 0.299])
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            default_cycler = (cycler(color=['r', 'g', 'b', 'y']) +
                              cycler(linestyle=['-', '--', ':', '-.']))
            ax.set_prop_cycle(default_cycler)
            marker = itertools.cycle(('o', 'x', 's'))
            for classid_in_turn in Classes_:
                x_aux = []
                y_aux = []
                obsid_aux = []
                classid_aux = []
                for i in list(range(len(ObsID_))):
                    if classid_[i] == classid_in_turn and not (np.isnan(x_[i])):
                        x_aux.append(x_[i])
                        y_aux.append(y_[i])
                        obsid_aux.append(ObsID_[i])
                        classid_aux.append(classid_in_turn)
                scatter = ax.scatter(x_aux, y_aux, alpha=1, s=22,marker=next(marker))
        legend1 = ax.legend(['1','2','3'],loc="upper left", title="Classes",fontsize=18, title_fontsize=18)
        plt.plot([0.014, 0.29], [0.014, 0.29], color='k', linestyle='dashed')
        ax.add_artist(legend1)
        plt.tight_layout()
        if SNV:
            if not(excludeCat):
                plt.savefig("problemSetting1_"+str(na)+".pdf")
            else:
                plt.savefig("problemSetting2_" + str(na) + ".pdf")
        else:
            plt.savefig("problemSetting0_" + str(na) + ".pdf")
        plt.show()
    return

def divCatNoSubCatHist(mvmobj, X, Y, rob_def, *, CLASSID=False, colorby=False, subclass=False, subclass_index=False, subclass2=False, subclass2_index=False, subclass3=False, subclass3_index=False, subclass4=False, subclass4_index=False,x_space=False):
    """
    divide categories

    mvmobj: A model created with phi.pca or phi.pls

    X/Y:     Data [numpy arrays or pandas dataframes]

    optional:
    CLASSID: Pandas Data Frame with classifiers per observation, each column is a class

    colorby: one of the classes in CLASSID to color by

    x_space: = 'False' will skip plotting the obs. vs pred for X *default*
               'True' will also plot obs vs pred for X

    """
    num_varX = mvmobj['P'].shape[0]

    if isinstance(X, np.ndarray):
        X_ = X.copy()
        ObsID_ = []
        for n in list(np.arange(X.shape[0]) + 1):
            ObsID_.append('Obs #' + str(n))
        XVarID_ = []
        for n in list(np.arange(X.shape[1]) + 1):
            XVarID_.append('Var #' + str(n))
    elif isinstance(X, pd.DataFrame):
        X_ = np.array(X.values[:, 1:]).astype(float)
        ObsID_ = X.values[:, 0].astype(str)
        ObsID_ = ObsID_.tolist()

    if 'varidX' in mvmobj:
        XVar = mvmobj['varidX']
    else:
        XVar = []
        for n in list(np.arange(num_varX) + 1):
            XVar.append('XVar #' + str(n))

    if 'Q' in mvmobj:
        num_varY = mvmobj['Q'].shape[0]
        if 'varidY' in mvmobj:
            YVar = mvmobj['varidY']
        else:
            YVar = []
            for n in list(np.arange(num_varY) + 1):
                YVar.append('YVar #' + str(n))

    if isinstance(Y, np.ndarray):
        Y_ = Y.copy()
    elif isinstance(Y, pd.DataFrame):
        Y_ = np.array(Y.values[:, 1:]).astype(float)

    if 'Q' in mvmobj:
        pred = phi.pls_pred(X_, mvmobj)
        yhat = pred['Yhat']
        if x_space:
            xhat = pred['Xhat']
        else:
            xhat = False
    else:
        x_space = True
        pred = phi.pca_pred(X_, mvmobj)
        xhat = pred['Xhat']
        yhat = False

    for i in list(range(Y_.shape[1])):
        y_mes_ = Y_[:, i]
        y_pred_ = yhat[:, i]
        class_mean_pred = []
        class_var_pred = []
        class_errorY_avg = []
        class_errorY_var = []
        class_errorYolsOfPls_avg = []
        class_errorYolsOfPls_var = []

        ols = linear_model.LinearRegression()
        reg = ols.fit(y_mes_.reshape(-1, 1), y_pred_.reshape(-1, 1))

        if not (isinstance(CLASSID, bool)):  #I have classes
            Classes_ = np.unique(CLASSID.values[:, colorby]).tolist()
            different_colors = len(Classes_)
            classid_ = list(CLASSID.values[:, colorby])
            y_pred_aux_cat = []
            for classid_in_turn in Classes_:
                y_mes_aux = []
                y_pred_aux = []
                obsid_aux = []
                classid_aux = []
                y_olsOfPls_aux = []

                for i in list(range(len(ObsID_))):
                    if classid_[i] == classid_in_turn \
                            and not (np.isnan(y_mes_[i])) :
                        y_mes_aux.append(y_mes_[i])
                        y_pred_aux.append(y_pred_[i])
                        obsid_aux.append(ObsID_[i])
                        classid_aux.append(str(classid_in_turn))
                        y_olsOfPls_aux.append(reg.predict(np.c_[y_mes_.reshape(-1, 1)[i]]))
                if y_pred_aux:
                    y_mes_aux_ = np.array(y_mes_aux[:]).astype(float)
                    y_pred_aux_= np.array(y_pred_aux[:]).astype(float)
                    errorY_ = (y_mes_aux_ - y_pred_aux_)
                    class_errorY_avg.append(np.mean(errorY_))
                    # alternatively ddof=1
                    class_errorY_var.append(np.var(errorY_, ddof=0))
                    class_var_pred.append(np.var(y_pred_aux, ddof=0))
                    class_mean_pred.append(np.mean(y_pred_aux))
                    errorYolsOfPls_= (y_mes_aux_ - y_olsOfPls_aux)
                    class_errorYolsOfPls_avg.append(np.mean(errorYolsOfPls_))
                    class_errorYolsOfPls_var.append(np.var(errorYolsOfPls_, ddof=0))
                    y_pred_aux_cat.append(y_pred_aux_)
    return y_pred_aux_cat

def predvsobsC3(mvmobj, X, Y, *, CLASSID=False, colorby=False, x_space=False,plot_title='Main Title', xaxis_label='X- axis', yaxis_label='Y- axis', na=False):
    """
    Plot observed vs predicted values
    by Salvador Garcia-Munoz
    (sgarciam@ic.ac.uk ,salvadorgarciamunoz@gmail.com)
    Modified by Chryssa Kappatou

    mvmobj: A model created with phi.pca or phi.pls

    X/Y:     Data [numpy arrays or pandas dataframes]

    optional:
    CLASSID: Pandas Data Frame with classifiers per observation, each column is a class

    colorby: one of the classes in CLASSID to color by

    x_space: = 'False' will skip plotting the obs. vs pred for X *default*
               'True' will also plot obs vs pred for X

    """
    num_varX = mvmobj['P'].shape[0]

    if isinstance(X, np.ndarray):
        X_ = X.copy()
        ObsID_ = []
        for n in list(np.arange(X.shape[0]) + 1):
            ObsID_.append('Obs #' + str(n))
        XVarID_ = []
        for n in list(np.arange(X.shape[1]) + 1):
            XVarID_.append('Var #' + str(n))
    elif isinstance(X, pd.DataFrame):
        X_ = np.array(X.values[:, 1:]).astype(float)
        ObsID_ = X.values[:, 0].astype(str)
        ObsID_ = ObsID_.tolist()

    if 'varidX' in mvmobj:
        XVar = mvmobj['varidX']
    else:
        XVar = []
        for n in list(np.arange(num_varX) + 1):
            XVar.append('XVar #' + str(n))

    if 'Q' in mvmobj:
        num_varY = mvmobj['Q'].shape[0]
        if 'varidY' in mvmobj:
            YVar = mvmobj['varidY']
        else:
            YVar = []
            for n in list(np.arange(num_varY) + 1):
                YVar.append('YVar #' + str(n))

    if isinstance(Y, np.ndarray):
        Y_ = Y.copy()
    elif isinstance(Y, pd.DataFrame):
        Y_ = np.array(Y.values[:, 1:]).astype(float)

    if 'Q' in mvmobj:
        pred = phi.pls_pred(X_, mvmobj)
        yhat = pred['Yhat']
        if x_space:
            xhat = pred['Xhat']
        else:
            xhat = False
    else:
        x_space = True
        pred = phi.pca_pred(X_, mvmobj)
        xhat = pred['Xhat']
        yhat = False

    # No CLASSIDS
    if isinstance(CLASSID, bool):
        for i in list(range(Y_.shape[1])):
            x_ = Y_[:, i]
            y_ = yhat[:, i]
            min_value = np.nanmin([np.nanmin(x_), np.nanmin(y_)])
            max_value = np.nanmax([np.nanmax(x_), np.nanmax(y_)])
            plt.figure()
            # plt.title(plot_title)
            plt.xlabel(xaxis_label)
            plt.ylabel(yaxis_label)
            plt.xticks([])
            plt.yticks([])
            plt.scatter(x_, y_)
            plt.plot([min_value, max_value], [min_value, max_value], color='cyan', linestyle='dashed')
        plt.show()

    else:  # YES CLASSIDS
        if isinstance(CLASSID, np.ndarray):
            Classes_ = np.unique(colorby).tolist()
            classid_ = list(colorby)
        else:
            Classes_ = np.unique(CLASSID[colorby]).tolist()
            classid_ = list(CLASSID[colorby])
        different_colors = len(Classes_)

        for i in list(range(Y_.shape[1])):
            x_ = Y_[:, i]
            y_ = yhat[:, i]
            min_value = np.nanmin([np.nanmin(x_), np.nanmin(y_)])  #0.03#np.nanmin([np.nanmin(x_), np.nanmin(y_)])  # 26
            max_value = np.nanmax([np.nanmax(x_), np.nanmax(y_)])  # 45
            fig, ax = plt.subplots()
            plt.xlabel(xaxis_label, fontsize=22)
            plt.ylabel(yaxis_label, fontsize=22)
            plt.xlim([0.01, 0.299])
            plt.ylim([0.01, 0.299])
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            default_cycler = (cycler(color=['b', 'r', 'g','y']) +
                              cycler(linestyle=['-', '--', ':', '-.']))
            ax.set_prop_cycle(default_cycler)
            for classid_in_turn in Classes_:
                x_aux = []
                y_aux = []
                obsid_aux = []
                classid_aux = []
                for i in list(range(len(ObsID_))):
                    if classid_[i] == classid_in_turn and not (np.isnan(x_[i])):
                        x_aux.append(x_[i])
                        y_aux.append(y_[i])
                        obsid_aux.append(ObsID_[i])
                        classid_aux.append(classid_in_turn)
                scatter = ax.scatter(x_aux, y_aux, alpha=1, s=22, marker='s')
        legend1 = ax.legend(['3'], loc="upper left", title="Classes",fontsize=18, title_fontsize=18)
        plt.plot([0.014, 0.29], [0.014, 0.29], color='k', linestyle='dashed')
        ax.add_artist(legend1)
        plt.tight_layout()
        plt.savefig("problemSetting2b_"+str(na)+".pdf")
        plt.show()
    return

class BBFunc:
    def __init__(self, Data, *, Rob=1, Dis=1, OF=1, beta=False, SNV=True, SG=True, Mcenter=True, pltHisOn=False, excludeCat=False, pltOn=False, stepWs=True, random=0, pltName=False):
        self.rob_def = Rob
        self.dis = Dis
        self.of = OF
        self.beta = beta
        self.SNV = SNV
        self.Mcenter = Mcenter
        self.SG = SG
        self.pltOn = pltOn
        self.pltHisOn = pltHisOn
        self.excludeCat = excludeCat
        self.stepWs = stepWs
        self.pltName = pltName

        # # # Data loading; The following part of reading the data will generally change by application; Action Point
        # Simulated example; Load data from csv
        X = pd.read_csv(Data, index_col=None, usecols=np.r_[0:201], na_values=np.nan)
        self.Y = pd.read_csv(Data, index_col=None, usecols=[0,201], na_values=np.nan)
        self.Z = pd.read_csv(Data, index_col=None, usecols=[0,203], na_values=np.nan)

        # # # Clean data; optional
        # X, _ = phi.clean_low_variances(X)
        # self.Y, _ = phi.clean_low_variances(self.Y)
        # self.Z, _ = phi.clean_low_variances(self.Z)

        # # # Split data set to train and test/validation
        if self.excludeCat:
            # define cat based on which you want to split the data
            catOfInterest_index = 1
            Categories_ = np.unique(self.Z.values[:, catOfInterest_index]).tolist()
            categoryid_ = list(self.Z.values[:, catOfInterest_index])
            # select one to be used for validation only
            categoryid_for_val = 2.0

            # train /test with two cat val with one more
            self.id_val = self.Z[
                self.Z['continuous interference 1'] == categoryid_for_val]
            self.y_val = self.Y[self.Z['continuous interference 1'] == categoryid_for_val]
            x_val = X[self.Z['continuous interference 1'] == categoryid_for_val]


            self.Y = self.Y[self.Z['continuous interference 1'] != categoryid_for_val]
            X = X[self.Z['continuous interference 1'] != categoryid_for_val]
            self.Z = self.Z[
                self.Z['continuous interference 1'] != categoryid_for_val]

        try:
            x_val
        except NameError:
            x_val = None
        try:
            x_val
        except NameError:
            x_val = None
        if x_val is None: # we do not have an additional independent validation set
            x_train0, x_test, y_train0, self.y_test, id_train0, self.id_test \
                = train_test_split(X, self.Y, self.Z, test_size=0.2, random_state=random)
            x_train, x_val, self.y_train, self.y_val, self.id_train, self.id_val \
                = train_test_split(x_train0, y_train0, id_train0,test_size=0.125, random_state=random)  # 0.125 x 0.8 = 0.1
        else: # we have an additional independent validation set
            x_train, x_test, self.y_train, self.y_test, self.id_train, self.id_test \
                = train_test_split(X, self.Y, self.Z, test_size=0.3, random_state=random)

        # # Test Plot 1; Raw Data
        if self.pltOn:
            plot_spectra(x_train, xaxis_label='Wavelength', yaxis_label='Intensity')

        # # # Write splitted sets to .mat file; optional
        # x_train1 = x_train.iloc[:,1:]
        # x_train1 = x_train1.to_numpy()
        # x_test1 = x_test.iloc[:, 1:]
        # x_test1 = x_test1.to_numpy()
        # self.y_train1 = self.y_train.iloc[:, 1:]
        # self.y_train1 = self.y_train1.to_numpy()
        # self.y_test1 = self.y_test.iloc[:, 1:]
        # self.y_test1 = self.y_test1.to_numpy()
        # # note the following are the samples names not the categories as indicated above by id_test and id_train
        # s_id_train1 = x_train.iloc[:, 0]
        # s_id_train1 = s_id_train1.to_numpy()
        # s_id_test1 = x_test.iloc[:, 0]
        # s_id_test1 =s_id_test1.to_numpy()
        #
        # scipy.io.savemat('cs1mat.mat', dict(x=x_train1, y=self.y_train1, z=s_id_train1, x_v=x_test1, y_v=self.y_test1,
        #                                    z_v=s_id_test1))

        # # # Preprocessing Step 1:
        if self.SNV: # turn on svn
            self.x_train_snv = phi.snv(x_train)
            self.x_test_snv = phi.snv(x_test)
            self.x_val_snv = phi.snv(x_val)
        else: # turn off svn
            self.x_train_snv = x_train
            self.x_test_snv = x_test
            self.x_val_snv = x_val
    def evaluate(self,x):
            # # # Preprocessing Step 2: SavGol
            if self.SG:  # turn on SG
                if self.stepWs:
                    ws = x[0] * 5 + 5 # plus five to start from 10 times 5 the stepsize
                else:
                    ws = x[0]
                do = x[1]
                po = x[2]
                na = x[3]
                if po < do:
                    # Todo error message po must be >= do and continue
                    obj = 1e10  # big number
                    return obj
                else:
                    x_train_snv_savgol, M = phi.savgol(ws, do, po, self.x_train_snv)
                    x_test_snv_savgol, M_t = phi.savgol(ws, do, po, self.x_test_snv)
                    x_val_snv_savgol, M_v = phi.savgol(ws, do, po, self.x_val_snv)
            else:  # turn off SG
                na = x[0]
                x_train_snv_savgol = self.x_train_snv
                x_test_snv_savgol = self.x_test_snv
                x_val_snv_savgol = self.x_val_snv

            # # Test Plot 2; Pre-processed Data
            if self.pltOn:
                plot_spectra(x_train_snv_savgol, xaxis_label='Wavelength', yaxis_label='Intensity', pltName=self.pltName)

            # # # Create the Regression Model Using the Training Set; Optional Action Point
            # Here you can change the algorithm (nipals or svd) and \
            # column-wise pre-processing (False for none, True for autoscaling & mean centering, 'center' \
            # for only mean centering and output detail shush True or False
            # pls_raman_calibration = phi.pls(x_train_snv_savgol, self.y_train, na, force_nipals=True, mcsX='center',
            #                                 mcsY='center', shush=True)
            if self.Mcenter:
                pls_raman_calibration = phi.pls(x_train_snv_savgol, self.y_train, na, force_nipals=True, mcsX='center',
                                                mcsY='center', shush=True)
            else:
                pls_raman_calibration = phi.pls(x_train_snv_savgol, self.y_train, na, force_nipals=True, mcsX=False,
                                            mcsY=False, shush=True)

            # # # Predictions Using Aboved-created Regression Model
            # Prediction Test Set
            pls_raman_calibration_predictions = phi.pls_pred(x_test_snv_savgol,
                                                             pls_raman_calibration)

            # Prediction Training/Calibration Set
            pls_raman_calibration_predictionsC = phi.pls_pred(x_train_snv_savgol,
                                                              pls_raman_calibration)

            # Prediction Accuracy Validation Set
            pls_raman_calibration_predictionsV = phi.pls_pred(x_val_snv_savgol,
                                                              pls_raman_calibration)

            # # # Diagnostics
            # Accuracy Training Set
            if isinstance(self.y_train, np.ndarray):
                y_train_ = self.y_train
            else:
                y_train_ = np.array(self.y_train.values[:, 1:]).astype(float)
            Yhatc_ = pls_raman_calibration_predictionsC['Yhat']
            errorYc_ = (y_train_ - Yhatc_)  # /y_train_
            PRESSYc = np.sum(errorYc_ ** 2)
            RMSPEYc = np.sqrt(PRESSYc / Yhatc_.size)
            print("%0.05f" % RMSPEYc)

            # Accuracy Test Set
            if isinstance(self.y_test, np.ndarray):
                y_test_ = self.y_test
            else:
                y_test_ = np.array(self.y_test.values[:, 1:]).astype(float)
            Yhat_ = pls_raman_calibration_predictions['Yhat']
            errorY_ = y_test_ - Yhat_
            PRESSY = np.sum(errorY_ ** 2)
            RMSPEY = np.sqrt(PRESSY / Yhat_.size)
            print("%0.05f" % RMSPEY)

            # Accuracy Val Set
            if isinstance(self.y_val, np.ndarray):
                y_val_ = self.y_val
            else:
                y_val_ = np.array(self.y_val.values[:, 1:]).astype(float)
            Yhatv_ = pls_raman_calibration_predictionsV['Yhat']
            errorYv_ = y_val_ - Yhatv_
            PRESSYv = np.sum(errorYv_ ** 2)
            RMSPEYv = np.sqrt(PRESSYv / Yhatv_.size)
            print("%0.05f" % RMSPEYv)

            # Moment Matching; Action Point
            # Here you need to specify if you have subcategories and call the moment matching function \
            # multiple times accordingly to the variability categories/subcategories of your choice
            # The following can be done alternative with training set (change test with train)

            # Simulated Example
            M1sum, M2sum = mm.momMatch(pls_raman_calibration, x_test_snv_savgol, self.y_test, \
                                            CLASSID=self.id_test, cat_rob_index=1, rob_def=self.rob_def, dis=self.dis)

            # # # Define Objective Function; Action Point

            MM = M1sum + self.beta * M2sum

            if self.of == 1:  # accuracy
                alphaRob = 0
                alphaAcc = 1
            elif self.of == 2:  # robustness
                alphaRob = 1
                alphaAcc = 0
            elif self.of == 3:  # both
                alphaRob = 1  # weight for robustness
                alphaAcc = 1.5  # weight for accuarcy

            obj = alphaAcc * RMSPEY + alphaRob * MM

            print(M1sum, M2sum)
            print("%0.05f" % MM)
            print(x, obj)

            # # # PredvsObs Plots; Optional Action Point
            if self.pltOn:
                colorby = 'continuous interference 1'

                predvsobsC(pls_raman_calibration, x_test_snv_savgol, self.y_test, CLASSID=self.id_test,colorby=colorby, xaxis_label='Observed',
                          yaxis_label='Predicted',na=na, SNV=self.SNV, excludeCat=self.excludeCat)

                if self.excludeCat:
                    predvsobsC3(pls_raman_calibration, x_val_snv_savgol, self.y_val, CLASSID=self.id_val, colorby=colorby,
                            xaxis_label='Observed', yaxis_label='Predicted', na=na)

                if self.pltHisOn:
                    # todo write as a function
                    Yhat_cat = divCatNoSubCatHist(pls_raman_calibration, x_test_snv_savgol, self.y_test, self.rob_def,
                                                  CLASSID=self.id_test, colorby=1)

                    np.random.seed(12345)

                    cat1 = Yhat_cat[0]  # normal(0, 1, 50)
                    cat2 = Yhat_cat[1]  # normal(0, 1, 20)
                    cat3 = Yhat_cat[2]  # normal(0, 1, 60)

                    bns = np.linspace(0.0, 0.25, 15)

                    default_cycler = (cycler(color=['r', 'g', 'b', 'y']) +
                                      cycler(linestyle=['-', '--', ':', '-.']))

                    fig, axs = plt.subplots(1)
                    plt.xlabel('Output Prediction', fontsize=22)
                    plt.ylabel('Frequency', fontsize=22)
                    plt.xticks(fontsize=18)
                    plt.yticks(np.arange(0.0, 21.0, 5.0),fontsize=18)
                    axs.set_prop_cycle(default_cycler)
                    cats=[cat1, cat2, cat3]

                    n, bins, patches = axs.hist(cats, bns, alpha=0.65, histtype='bar',
                                                color=['r', 'g', 'b'],
                                                label=['$\hat{y}_1$', '$\hat{y}_2$', '$\hat{y}_3$']) #['0', '0.33', '0.66']

                    hatches = ['...', '///', '']
                    for patch_set, hatch in zip(patches, hatches):
                        for patch in patch_set.patches:
                            patch.set_hatch(hatch)

                    plt.legend(loc="upper left", fontsize=18)

                    axs.vlines(np.mean(cat1), 0, 45, 'r', '-', label='$\hat{\widetilde{y}}_1$')
                    axs.vlines(np.mean(cat2), 0, 45, 'g', '--',  label='$\hat{\widetilde{y}}_2$')
                    axs.vlines(np.mean(cat3), 0, 45,'b', ':', label='$\hat{\widetilde{y}}_3$')
                    plt.setp(axs, xlim=(0.0, 0.25), ylim=(0, 20))
                    plt.legend(loc="upper right",fontsize=18)
                    plt.tight_layout()
                    plt.savefig("problemSetting1his1_" + str(na) + ".pdf")
                    plt.show()

                    fig1, axs1 = plt.subplots(1)
                    plt.xlabel('Output Prediction', fontsize=22)
                    plt.ylabel('Frequency', fontsize=22)
                    plt.xticks(fontsize=18)
                    plt.yticks(np.arange(0.0, 21.0, 5.0), fontsize=18)
                    axs1.set_prop_cycle(default_cycler)
                    axs1.vlines(np.mean(cat1), 0, 45, 'r', '-', label='$\hat{\widetilde{y}}_1$')
                    axs1.vlines(np.mean(cat2), 0, 45, 'g', '--', label='$\hat{\widetilde{y}}_2$')
                    axs1.vlines(np.mean(cat3), 0, 45,'b', ':', label='$\hat{\widetilde{y}}_3$')
                    plt.setp(axs1, xlim=(0.08, 0.13), ylim=(0, 20))
                    plt.legend(loc="upper right",fontsize=18)
                    plt.tight_layout()
                    plt.savefig("problemSetting1hisZ1_" + str(na) + ".pdf")
                    plt.show()

            return obj

import sys
original_stdout = sys.stdout # Save a reference to the original standard output

DataFile = 'data/dataSimulated.csv'
Robmetric = 1 # categorical for which rob metric; allowed values 1,2,3 ; default=1; optional argument
DisMeas = 2 # categorical for which distance metric; allowed values 1 (euclidean),2(manhattan); default=1; optional argument
beta = 15 # relative weight between first and second moment

# # # Define bounds for the decision variables
ws_lb = 1
ws_ub = 7
od_lb = 0
od_ub = 2
op_lb = 1
op_ub = 4
na_lb = 1
na_ub = 8

bb_func0 = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, beta=beta, SG=False, SNV=False, Mcenter=False, pltOn=True)
for i in range(1,5):
    bb_func0.evaluate([i])

bb_func1 = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, beta=beta, SG=False, pltHisOn=True, pltOn=True)
for i in range(1,5):
    bb_func1.evaluate([i])

bb_func2 = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, beta=beta, SG=False, excludeCat=True, pltOn=True)
for i in range(1,9):
    bb_func2.evaluate([i])

# spectra plots
bb_func3 = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, beta=beta, pltOn=True, pltName='CS1_spectraACC')
bb_func3.evaluate([3, 0, 2, 3]) #acc
bb_func3 = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, beta=beta, pltOn=True, OF=2, pltName='CS1_spectraROB')
bb_func3.evaluate([5, 2, 4, 2]) #rob

for i in range(0,1): # random splits
    for j in range(1,3): #different objectives
        bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, OF=j, random=i, beta=beta)
        res = entmoot_minimize(
            bb_func.evaluate,
            [(ws_lb, ws_ub), (od_lb, od_ub), (op_lb, op_ub), (na_lb, na_ub)],
            n_calls=200,  # 70,
            # n_points=10000,
            base_estimator="GBRT",
            # std_estimator="BDD",
            n_initial_points=90,  # 45,
            initial_point_generator="random",
            acq_func="LCB",
            acq_optimizer="global",
            x0=None,
            y0=None,
            random_state=250,
            acq_func_kwargs={
                "kappa": 1.96
            },
            base_estimator_kwargs={
                "min_child_samples": 2
            },
            verbose=True,
        )
        bb_func.evaluate(res.x)
        with open("resultsInitialExample.txt", "a") as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print("\n" + "ENTMOOT calls 200 init 90" + "\n")
            print("\n" + "random" + str(i) + "\n")
            print("\n" +"obj" + str(j) + "\n")
            print("parameters " + str(res.x) + "\n")
            print("objective " + "%0.05f" % res.fun + "\n")
            bb_func.evaluate(res.x)
            sys.stdout = original_stdout  # Reset the standard output to its original value

# complete enumeration
for o in range(1,3):
    objValue=1e10
    objX= []
    for i in range(ws_lb,ws_ub+1):
        for j in range(od_lb,od_ub+1):
            for k in range(op_lb,op_ub+1):
                for l in range(na_lb,na_ub+1):
                    bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, beta=beta, OF=o)
                    res=bb_func.evaluate([i, j, k, l])
                    if res<objValue:
                        objValue=res
                        objX=[i, j, k, l]
    print(objValue, objX)
    bb_func.evaluate(objX)
    with open("resultsInitialExample.txt", "a") as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print("1b: complete enumeration" + "\n")
        print("\n" + "obj" + str(o) + "\n")
        print("parameters " + str(objX) + "\n")
        print("objective " + "%0.05f" % objValue + "\n")
        bb_func.evaluate(objX)
        sys.stdout = original_stdout  # Reset the standard output to its original value
