##
# ExampleNIRandRaman.py
#
# NIR and Raman spectroscopy example
#
# C.D. Kappatou, J.Odgers, S. Garcia, R.Misener: "Optimization Methods for Developing Efficient Chemometric Models", 2022.
# 
##

import pandas as pd
import numpy as np
import pyphi as phi
import matplotlib.pyplot as plt
import scipy.io
from sklearn.model_selection import train_test_split
from entmoot.optimizer.entmoot_minimize import entmoot_minimize

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
    x1=x.tolist() * y.shape[0]
    y1=y.tolist()
    for i in range(len(x1)):
        ax.plot(np.array(x1[i]).astype(int),y1[i])
    plt.xticks(np.arange(7500,10500+1,1000))
    ax.xaxis.set_minor_locator(plt.MaxNLocator(7))
    plt.tight_layout()
    if pltName:
        plt.savefig(str(pltName)+".pdf")
    else:
        plt.savefig("CS2_spectra.pdf")
    plt.show()
    return

class BBFunc:
    def __init__(self, Data, *, Rob=1, Dis=1, beta=False, SNV=True, SG=True, OF=1, MMsource=0, random=0, stepWs=True, pltOn=False, pltName=False):
        self.rob_def = Rob
        self.dis = Dis
        self.of = OF
        self.beta = beta
        self.msource = MMsource
        self.stepWs = stepWs
        self.pltOn = pltOn
        self.pltName = pltName

        # # # Data loading; The following part of reading the data will generally change by application; Action Point
        # Raman example; Load the data from Excel
        X = pd.read_excel(Data, 'Raman', index_col=None, na_values=np.nan)
        self.Y = pd.read_excel(Data, 'Y', index_col=None, na_values=np.nan)
        self.Z = pd.read_excel(Data, 'Categorical', index_col=None,
                                          na_values=np.nan)

        # # # Clean data; optional
        # X, _ = phi.clean_low_variances(X)
        # self.Y, _ = phi.clean_low_variances(self.Y)
        # self.Z, _ = phi.clean_low_variances(self.Z)

        # # # Split data set to train and validation/test
        try:
            x_test
        except NameError:
            x_test = None
        try:
            x_test
        except NameError:
            x_test = None
        if x_test is None: # we do not have an additional independent test set
            x_train0, x_val, y_train0, self.y_val, id_train0, self.id_val \
                = train_test_split(X, self.Y, self.Z, test_size=0.2, random_state=random)
            x_train, x_test, self.y_train, self.y_test, self.id_train, self.id_test \
                = train_test_split(x_train0, y_train0, id_train0,test_size=0.125, random_state=random)  # 0.125 x 0.8 = 0.1
        else: # we have an additional independent test set
            x_train, x_val, self.y_train, self.y_val, self.id_train, self.id_val \
                = train_test_split(X, self.Y, self.Z, test_size=0.3, random_state=random)

        # # Test Plot 1 ; Raw Data
        if self.pltOn:
            plot_spectra(x_train, xaxis_label='Wavelength', yaxis_label='Intensity')

        # # # Write splitted sets to .mat file; optional
        x_train1 = x_train.iloc[:,1:]
        x_train1 = x_train1.to_numpy()
        x_val1 = x_val.iloc[:, 1:]
        x_val1 = x_val1.to_numpy()
        x_test1 = x_test.iloc[:, 1:]
        x_test1 = x_test1.to_numpy()
        self.y_train1 = self.y_train.iloc[:, 1:]
        self.y_train1 = self.y_train1.to_numpy()
        self.y_val1 = self.y_val.iloc[:, 1:]
        self.y_val1 = self.y_val1.to_numpy()
        self.y_test1 = self.y_test.iloc[:, 1:]
        self.y_test1 = self.y_test1.to_numpy()
        # note the following are the samples names not the categories as indicated above by id_val and id_train
        s_id_train1 = x_train.iloc[:, 0]
        s_id_train1 = s_id_train1.to_numpy()
        s_id_val1 = x_val.iloc[:, 0]
        s_id_val1 =s_id_val1.to_numpy()
        s_id_test1 = x_test.iloc[:, 0]
        s_id_test1 =s_id_test1.to_numpy()

        scipy.io.savemat('cs2matlab.mat', dict(x=x_train1, y=self.y_train1, z=s_id_train1, x_t=x_val1, y_t=self.y_val1,
                                           z_t=s_id_val1, x_v=x_test1, y_v=self.y_test1,
                                           z_v=s_id_test1))

        # # # Preprocessing Step 1:
        if SNV: # turn on svn
            self.x_train_snv = phi.snv(x_train)
            self.x_val_snv = phi.snv(x_val)
            self.x_test_snv = phi.snv(x_test)
        else: # turn off svn
            self.x_train_snv = x_train
            self.x_val_snv = x_val
            self.x_test_snv = x_test

        self.SG = SG
    def evaluate(self,x):
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
                x_val_snv_savgol, M_v = phi.savgol(ws, do, po, self.x_val_snv)
                x_test_snv_savgol, M_t = phi.savgol(ws, do, po, self.x_test_snv)
        else: # turn off SG
            na = x[0]
            x_train_snv_savgol = self.x_train_snv
            x_val_snv_savgol = self.x_val_snv
            x_test_snv_savgol = self.x_test_snv

        # # Test Plot 2; Pre-processed Data
        if self.pltOn:
            plot_spectra(x_train_snv_savgol, xaxis_label='Wavelength', yaxis_label='Intensity', pltName=self.pltName)

        # # # Create the Regression Model Using the Training Set; Optional Action Point
        # Here you can change the algorithm (nipals or svd) and \
        # column-wise pre-processing (False for none, True for autoscaling & mean centering, 'center' \
        # for only mean centering and output detail shush True or False
        pls_raman_calibration = phi.pls(x_train_snv_savgol, self.y_train, na, force_nipals=True, mcsX='center',
                                        mcsY='center', shush=True)

        # # # Predictions Using Aboved-created Regression Model
        # Prediction Validation Set
        pls_raman_calibration_predictions = phi.pls_pred(x_val_snv_savgol,
                                                         pls_raman_calibration)

        # Prediction Training/Calibration Set
        pls_raman_calibration_predictionsC = phi.pls_pred(x_train_snv_savgol,
                                                          pls_raman_calibration)

        # Prediction Accuracy Test Set
        pls_raman_calibration_predictionsT = phi.pls_pred(x_test_snv_savgol,
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

        # Accuracy Validation Set
        if isinstance(self.y_val, np.ndarray):
            y_val_ = self.y_val
        else:
            y_val_ = np.array(self.y_val.values[:, 1:]).astype(float)
        Yhat_ = pls_raman_calibration_predictions['Yhat']
        errorY_ = y_val_ - Yhat_
        PRESSY = np.sum(errorY_ ** 2)
        RMSPEY = np.sqrt(PRESSY / Yhat_.size)
        print("%0.05f" % RMSPEY)

        # Accuracy Test Set
        if isinstance(self.y_test, np.ndarray):
            y_test_ = self.y_test
        else:
            y_test_ = np.array(self.y_test.values[:, 1:]).astype(float)
        Yhatt_ = pls_raman_calibration_predictionsT['Yhat']
        errorYt_ = y_test_ - Yhatt_
        PRESSYt = np.sum(errorYt_ ** 2)
        RMSPEYt = np.sqrt(PRESSYt / Yhatt_.size)
        print("%0.05f" % RMSPEYt)

        # Moment Matching; Action Point
        # Here you need to specify if you have subcategories and call the moment matching function \
        # multiple times accordingly to the variability categories/subcategories of your choice
        # The following can be done alternative with training set (change test with train)
        M1sumA, M2sumA = mm.momMatch(pls_raman_calibration, x_val_snv_savgol, self.y_val,
                                CLASSID=self.id_val, cat_rob_index=2, subclass_index=1, \
                                     rob_def=self.rob_def, dis=self.dis)

        M1sumB, M2sumB = mm.momMatch(pls_raman_calibration, x_val_snv_savgol, self.y_val,
                                CLASSID=self.id_val, cat_rob_index=1, subclass_index=2, \
                                     rob_def=self.rob_def, dis=self.dis)

        if self.msource == 0:
            p_sumA = 1
            p_sumB = 1

        elif self.msource == 1:
            p_sumA = 1
            p_sumB = 0

        elif self.msource == 2:
            p_sumA = 0
            p_sumB = 1

        # Here set the weights among the different variation sources if no intuition set to unity/
        # if wished the different sources my get different weights or turned off (setting weight to zero)
        varSourceWeightA = 1 * 2
        varSourceWeightB = 1 * 1
        M1sum = p_sumA*varSourceWeightA*M1sumA + p_sumB*varSourceWeightB*M1sumB
        M2sum = p_sumA*varSourceWeightA*M2sumA + p_sumB*varSourceWeightB*M2sumB

        print(M1sumA, M1sumB)
        print(M2sumA, M2sumB)

        # # # Define Objective Function; Action Point

        if self.of == 1:  # accuracy
            alphaRob = 0
            alphaAcc = 1
        elif self.of == 2:  # robustness
            alphaRob = 1
            alphaAcc = 0
        elif self.of == 3:  # both
            alphaRob = 1  # weight for robustness
            alphaAcc = 1.5  # weight for accuarcy

        MM = M1sum + self.beta * M2sum
        obj = alphaAcc * RMSPEY + alphaRob * MM

        MMA = M1sumA + self.beta * M2sumA # scale
        MMB = M1sumB + self.beta * M2sumB  # dos
        print("%0.05f" % MMA + "&" + "%0.05f" % MMB)

        print(M1sum, M2sum)
        print("%0.05f" % MM)
        print(x, obj)

        return obj

import sys
original_stdout = sys.stdout # Save a reference to the original standard output

DataFile = 'data/Raman.xlsx'
Robmetric = 1 # categorical for which rob metric; allowed values 1,2,3 ; default=1; optional argument
DisMeas = 1 # categorical for which distance metric; allowed values 1 (euclidean),2(manhattan); default=1; optional argument
beta = 0.005 # relative weight between first and second moment

# # # Define bounds for the decision variables
ws_lb = 1
ws_ub = 7
od_lb = 0
od_ub = 2
op_lb = 1
op_ub = 4
na_lb = 1
na_ub = 8

# 1:random splits for different objectives
for i in range(0,5):
    for j in range(1,3):
        bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, beta=beta, OF=j, random=i)
        res = entmoot_minimize(
            bb_func.evaluate,
            [(ws_lb, ws_ub), (od_lb, od_ub), (op_lb, op_ub), (na_lb, na_ub)],
            n_calls=200,  # 70,
            # n_points=10000,
            base_estimator="GBRT",
            std_estimator="BDD",
            n_initial_points=90,  # 45,
            initial_point_generator="random",
            acq_func="LCB",
            acq_optimizer="global",
            x0=None,
            y0=None,
            random_state=100,
            acq_func_kwargs={
                "kappa": 1.96
            },
            base_estimator_kwargs={
                "min_child_samples": 2
            },
            verbose=True,
        )
        # bb_func.evaluate(res.x)
        with open("resultsExample2.txt", "a") as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print("random" + str(i) + "\n")
            print("\n" +"obj" + str(j) + "\n")
            print("parameters " + str(res.x) + "\n")
            print("objective " + "%0.05f" % res.fun + "\n")
            bb_func.evaluate(res.x)
            sys.stdout = original_stdout  # Reset the standard output to its original value

# 1b:complete enumeration
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
    with open("resultsExample2.txt", "a") as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print("1b: complete enumeration" + "\n")
        print("\n" + "obj" + str(o) + "\n")
        print("parameters " + str(objX) + "\n")
        print("objective " + "%0.05f" % objValue + "\n")
        bb_func.evaluate(objX)
        sys.stdout = original_stdout  # Reset the standard output to its original value

# 2: random split different robustness sources for MM objective
for i in range(0,5):
    for j in range(1,3):
        bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, beta=beta, OF=2, random=i, MMsource=j)
        res = entmoot_minimize(
            bb_func.evaluate,
            [(ws_lb, ws_ub), (od_lb, od_ub), (op_lb, op_ub), (na_lb, na_ub)],
            n_calls=200,  # 70,
            # n_points=10000,
            base_estimator="GBRT",
            std_estimator="BDD",
            n_initial_points=90,  # 45,
            initial_point_generator="random",
            acq_func="LCB",
            acq_optimizer="global",
            x0=None,
            y0=None,
            random_state=100,
            acq_func_kwargs={
                "kappa": 1.96
            },
            base_estimator_kwargs={
                "min_child_samples": 2
            },
            verbose=True,
        )
        # bb_func.evaluate(res.x)
        with open("resultsExample2.txt", "a") as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print("random" + str(i) + "\n")
            print("\n" +"obj" + str(j) + "\n")
            print("parameters " + str(res.x) + "\n")
            print("objective " + "%0.05f" % res.fun + "\n")
            bb_func.evaluate(res.x)
            sys.stdout = original_stdout  # Reset the standard output to its original value

# 2b:complete enumeration
for o in range(1,3):
    objValue=1e10
    objX= []
    for i in range(ws_lb,ws_ub+1):
        for j in range(od_lb,od_ub+1):
            for k in range(op_lb,op_ub+1):
                for l in range(na_lb,na_ub+1):
                    bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, beta=beta, OF=2, MMsource=o)
                    res=bb_func.evaluate([i, j, k, l])
                    if res<objValue:
                        objValue=res
                        objX=[i, j, k, l]
    print(objValue, objX)
    bb_func.evaluate(objX)
    with open("resultsExample2.txt", "a") as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print("2b: complete enumeration" + "\n")
        print("\n" + "obj" + str(o) + "\n")
        print("parameters " + str(objX) + "\n")
        print("objective " + "%0.05f" % objValue + "\n")
        bb_func.evaluate(objX)
        sys.stdout = original_stdout  # Reset the standard output to its original value

# spectra plots
bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, pltOn=True, beta=beta, pltName='CS2_spectraACC')
bb_func.evaluate([1, 0, 1, 7]) #acc
bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, pltOn=True, beta=beta, OF=2, pltName='CS2_spectraSC')
bb_func.evaluate([4, 2, 3, 1]) #rob sc
bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, pltOn=True, beta=beta, OF=2, pltName='CS2_spectraDOS')
bb_func.evaluate([7, 0, 1, 1]) #rob dos
bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, pltOn=True, beta=beta, OF=2, pltName='CS2_spectraROB')
bb_func.evaluate([5, 2, 4, 1]) #rob

