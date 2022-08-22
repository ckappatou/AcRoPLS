# NIR spectroscopy example solved in ENTMOOT for different objectives

# C.Kappatou, J.Odgers, R. Misener, S. Garcia
# 22.08.2022

import pandas as pd
import numpy as np
import pyphi as phi
import matplotlib.pyplot as plt
import momMatch as mm
from sklearn.model_selection import train_test_split
from entmoot.optimizer.entmoot_minimize import entmoot_minimize
from scipy.io import loadmat

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
        ax.plot(x1[i], y1[i])
    plt.tight_layout()
    if pltName:
        plt.savefig(str(pltName)+".pdf")
    else:
        plt.savefig("CS3_spectra.pdf")
    plt.show()
    return

class BBFunc:
    def __init__(self, Data, *, Rob=1, SNV=True, SG=True,Dis=1, OF=1, MMsource=0, random=0, completeTest=False, stepWs=True, valSet=1, pltOn=False, pltName=False):
        self.rob_def = Rob
        self.dis = Dis
        self.of = OF
        self.msource = MMsource
        self.completeTest = completeTest
        self.stepWs = stepWs
        self.pltOn=pltOn
        self.pltName=pltName

        # # # Data loading; The following part of reading the data will generally change by application; Action Point
        # Industrial example; Load data from matlab
        mat = loadmat(Data)

        y_bc_train = mat['Y_BC_TRAIN']
        y_b_c_train_api = y_bc_train[:, :1]
        self.Y = np.concatenate((y_b_c_train_api, y_b_c_train_api))

        spec_b_train = mat['SPEC_B_PLS_TRAINING']
        spec_c_train = mat['SPEC_C_PLS_TRAINING']
        X = np.concatenate((spec_b_train, spec_c_train))

        id_b_train = mat['ID_B']
        id_b_train = np.append(id_b_train, np.ones((len(id_b_train), 1)), axis=1)
        id_b_train[:, [0, 1, 2, 3, 4]] = id_b_train[:, [1, 3, 4, 2, 0]]
        id_c_train = mat['ID_C']
        id_c_train = np.append(id_c_train, np.zeros((len(id_c_train), 1)), axis=1)
        id_c_train[:, [0, 1, 2, 3, 4]] = id_c_train[:, [1, 3, 4, 2, 0]]  # RH, PS, InstB, MR
        self.Z = np.concatenate((id_b_train, id_c_train))
        self.Z = self.Z[:, :-1]

        # y val
        y_b_val1 = mat['TEST'+str(valSet)+'_HPLC'] #mat['TEST1_HPLC'] # 100% variation from target
        self.target = 35.71
        y_b_val1[:, 1] = y_b_val1[:, 1] * self.target / 100
        self.y_val = y_b_val1[:, -1]
        # cat val
        self.id_val = mat['TEST_SETS_ID'][valSet-1, :]
        # self.id_b_test1[[0, 1, 2, 3]] = self.id_b_test1[[1, 0, 2, 3]]
        self.id_val = self.id_val[:-1]
        # x val
        spec_b_val = mat['TEST_SET_'+str(valSet)] #mat['TEST_SET_1']
        spec_b_val_used_list = []
        for i in range(len(spec_b_val)):
            for j in range(len(y_b_val1)):
                if y_b_val1[j, 0] == float(i):
                    spec_b_val_used_list.append(spec_b_val[i, :])
        x_val = np.array(spec_b_val_used_list)

        # # # Clean data; optional
        # X, _ = phi.clean_low_variances(X)
        # self.Y, _ = phi.clean_low_variances(self.Y)
        # self.Z, _ = phi.clean_low_variances(self.Z)

        # # # Split data set to train and test/validation
        if self.completeTest:
            x_train = X
            self.y_train = self.Y
            self.id_train = self.Z
            x_test = x_val
            self.y_test = self.y_val
            self.id_test = self.id_val
        else:
            try:
                x_val
            except NameError:
                x_val = None
            if x_val is None: # we do not have an additional independent validation set
                x_train0, x_test, y_train0, self.y_test, id_train0, self.id_test \
                    = train_test_split(X, self.Y, self.Z, test_size=0.2, random_state=random)
                x_train, x_val, self.y_train, self.y_val, self.id_train, self.id_val \
                    = train_test_split(x_train0, y_train0, id_train0, test_size=0.125,
                                       random_state=random)  # 0.125 x 0.8 = 0.1
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
        # scipy.io.savemat('cs3mat.mat', dict(x=x_train1, y=self.y_train1, z=s_id_train1, x_v=x_test1, y_v=self.y_test1,
        #                                    z_v=s_id_test1))

        # # # Preprocessing Step 1:
        if SNV: # turn on svn
            self.x_train_snv = phi.snv(x_train)
            self.x_test_snv = phi.snv(x_test)
            self.x_val_snv = phi.snv(x_val)
        else: # turn off svn
            self.x_train_snv = x_train
            self.x_test_snv = x_test
            self.x_val_snv = x_val

        self.SG = SG
    def evaluate(self,x):
        if self.SG: # turn on SG
            if self.stepWs:
                ws = x[0] * 5 + 5
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
        else: # turn off SG
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
        pls_raman_calibration = phi.pls(x_train_snv_savgol, self.y_train, na, force_nipals=True, mcsX='center',
                                        mcsY='center', shush=True)

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
        # The following could be done alternative with training set (change test with train)
        if self.completeTest:
            MM = 0
        else:
            # RH, PS, InstB, MR
            # ps
            M1sum0, M2sum0 = mm.momMatch(pls_raman_calibration, x_test_snv_savgol, self.y_test,
                                         CLASSID=self.id_test, cat_rob_index=1, subclass_index=2, subclass2_index=0,
                                         subclass3_index=3, rob_def=self.rob_def, dis=self.dis)

            # rh
            M1sum1, M2sum1 = mm.momMatch(pls_raman_calibration, x_test_snv_savgol, self.y_test,
                                         CLASSID=self.id_test, cat_rob_index=0, subclass_index=2, subclass2_index=1,
                                         subclass3_index=3, rob_def=self.rob_def, dis=self.dis)

            # inst
            M1sum2, M2sum2 = mm.momMatch(pls_raman_calibration, x_test_snv_savgol, self.y_test,
                                         CLASSID=self.id_test, cat_rob_index=2, subclass_index=0, subclass2_index=1,
                                         subclass3_index=3, rob_def=self.rob_def, dis=self.dis)

            # mr
            M1sum3, M2sum3 = mm.momMatch(pls_raman_calibration, x_test_snv_savgol, self.y_test,
                                         CLASSID=self.id_test, cat_rob_index=3, subclass_index=0, subclass2_index=1,
                                         subclass3_index=2, rob_def=self.rob_def, dis=self.dis)

            if self.msource==0:
                p_sum0=1
                p_sum1=1
                p_sum2=1
                p_sum3=1

            elif self.msource==1:
                p_sum0=1
                p_sum1=0
                p_sum2=0
                p_sum3=0

            elif self.msource == 2:
                p_sum0=0
                p_sum1=1
                p_sum2=0
                p_sum3=0

            elif self.msource == 3:
                p_sum0=0
                p_sum1=0
                p_sum2=1
                p_sum3=0

            elif self.msource == 4:
                p_sum0=0
                p_sum1=0
                p_sum2=0
                p_sum3=1

            M1sum = p_sum0*M1sum0/2+p_sum1*M1sum1+p_sum3*25*M1sum3+p_sum2*M1sum2
            M2sum = p_sum0*M2sum0/2+p_sum1*M2sum1+p_sum3*25*M2sum3+p_sum2*M2sum2

            beta = 0.0001  # relative weight between first and second moment

            # PS, RH, InstB, MR
            MM = M1sum + beta * M2sum
            MM0 = M1sum0 + beta * M2sum0 #ps
            MM1 = M1sum1 + beta * M2sum1 #rh
            MM2 = M1sum2 + beta * M2sum2 #inst
            MM3 = M1sum3 + beta * M2sum3 #mr
            print("%0.05f" % MM0 +"&"+ "%0.05f" % MM1+"&"+ "%0.05f" % MM2+"&"+ "%0.05f" % MM3 )
            print("%0.05f" % M1sum, "%0.05f" % M2sum)

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

        print("%0.05f" % MM)
        print(x, obj)

        return obj

import sys
original_stdout = sys.stdout # Save a reference to the original standard output

DataFile = 'NIR DOE data with test sets.mat'
Robmetric = 1 # categorical for which rob metric; allowed values 1,2,3 ; default=1; optional argument
DisMeas = 1 # categorical for which distance metric; allowed values 1 (euclidean),2(manhattan); default=1; optional argument

# 1: complete training test and validation 1 as test set
bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, OF=1, completeTest=True)
res = entmoot_minimize(
    bb_func.evaluate,
    [(1, 7), (0, 2), (1, 4), (1, 8)],
    n_calls=2 * 100,  # 70,
    # n_points=10000,
    base_estimator="GBRT",
    std_estimator="BDD",
    n_initial_points=2 * 45,  # 45,
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
bb_func.evaluate(res.x)
with open("resultsExample3.txt", "a") as f:
    sys.stdout = f  # Change the standard output to the file we created.
    print("1: complete training test and validation 1 as test set" + "\n")
    print("parameters " + str(res.x) + "\n")
    print("objective " + "%0.05f" % res.fun + "\n")
    bb_func.evaluate(res.x)
    sys.stdout = original_stdout  # Reset the standard output to its original value

# 1b: complete enumeration
objValue=1e10
objX= []
for i in range(1,8):
    for j in range(0,3):
        for k in range(1,5):
            for l in range(1,9):
                bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, OF=1, completeTest=True)
                res=bb_func.evaluate([i, j, k, l])
                if res<objValue:
                    objValue=res
                    objX=[i, j, k, l]
print(objValue, objX)
bb_func.evaluate(objX)
with open("resultsExample3.txt", "a") as f:
    sys.stdout = f  # Change the standard output to the file we created.
    print("1b: complete enumeration" + "\n")
    print("parameters " + str(objX) + "\n")
    print("objective " + "%0.05f" % objValue + "\n")
    bb_func.evaluate(objX)
    sys.stdout = original_stdout  # Reset the standard output to its original value

# 1c: complete training test and different validation sets as test set
for i in range(2,8):
    bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, OF=1, completeTest=True, valSet=i)
    res = entmoot_minimize(
        bb_func.evaluate,
        [(1, 7), (0, 2), (1, 4), (1, 8)],
        n_calls=2 * 100,  # 70,
        # n_points=10000,
        base_estimator="GBRT",
        std_estimator="BDD",
        n_initial_points=2 * 45,  # 45,
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
    bb_func.evaluate(res.x)
    with open("resultsExample3.txt", "a") as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print("1c: complete training test and different validation sets as test set" + "\n")
        print("test" + str(i) + "\n")
        print("parameters " + str(res.x) + "\n")
        print("objective " + "%0.05f" % res.fun + "\n")
        bb_func.evaluate(res.x)
        bb_func1 = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, OF=1, valSet=i)
        print("Acc")
        bb_func1.evaluate([1, 2, 4, 8])
        print("Rob")
        bb_func1.evaluate([7, 1, 3, 3])
        sys.stdout = original_stdout  # Reset the standard output to its original value

# 2:random splits for different objectives
for i in range(0,5):
    for j in range(1,3):
        bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, OF=j, random=i)
        res = entmoot_minimize(
            bb_func.evaluate,
            [(1, 7), (0, 2), (1, 4), (1, 8)],
            n_calls=2*100,  # 70,
            # n_points=10000,
            base_estimator="GBRT",
            std_estimator="BDD",
            n_initial_points=2*45,  # 45,
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
        with open("resultsExample3.txt", "a") as f:
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
    for i in range(1,8):
        for j in range(0,3):
            for k in range(1,5):
                for l in range(1,9):
                    bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, OF=o)
                    res=bb_func.evaluate([i, j, k, l])
                    if res<objValue:
                        objValue=res
                        objX=[i, j, k, l]
    print(objValue, objX)
    bb_func.evaluate(objX)
    with open("resultsExample3.txt", "a") as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print("2b: complete enumeration" + "\n")
        print("\n" + "obj" + str(o) + "\n")
        print("parameters " + str(objX) + "\n")
        print("objective " + "%0.05f" % objValue + "\n")
        bb_func.evaluate(objX)
        sys.stdout = original_stdout  # Reset the standard output to its original value

# 3: random split different robustness sources for MM objective
for i in range(0,5):
    for j in range(1,5):
        bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, OF=2, random=i, MMsource=j)
        res = entmoot_minimize(
            bb_func.evaluate,
            [(1, 7), (0, 2), (1, 4), (1, 8)],
            n_calls=2*100,  # 70,
            # n_points=10000,
            base_estimator="GBRT",
            std_estimator="BDD",
            n_initial_points=2*45,  # 45,
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
        with open("resultsExample3.txt", "a") as f:
            sys.stdout = f  # Change the standard output to the file we created.
            print("random" + str(i) + "\n")
            print("\n" +"obj" + str(j) + "\n")
            print("parameters " + str(res.x) + "\n")
            print("objective " + "%0.05f" % res.fun + "\n")
            bb_func.evaluate(res.x)
            sys.stdout = original_stdout  # Reset the standard output to its original value

# 3b:complete enumeration
for o in range(1,5):
    objValue=1e10
    objX= []
    for i in range(1,8):
        for j in range(0,3):
            for k in range(1,5):
                for l in range(1,9):
                    bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, OF=2, MMsource=o)
                    res=bb_func.evaluate([i, j, k, l])
                    if res<objValue:
                        objValue=res
                        objX=[i, j, k, l]
    print(objValue, objX)
    bb_func.evaluate(objX)
    with open("resultsExample3.txt", "a") as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print("3b: complete enumeration" + "\n")
        print("\n" + "obj" + str(o) + "\n")
        print("parameters " + str(objX) + "\n")
        print("objective " + "%0.05f" % objValue + "\n")
        bb_func.evaluate(objX)
        sys.stdout = original_stdout  # Reset the standard output to its original value

# 4:Reference
bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, OF=1, stepWs=False)
x = [11, 1, 2, 3]
bb_func.evaluate(x)
with open("resultsExample3.txt", "a") as f:
    sys.stdout = f  # Change the standard output to the file we created.
    print("4: Reference" + "\n")
    print("parameters " + str(x) + "\n")
    bb_func.evaluate(x)
    sys.stdout = original_stdout  # Reset the standard output to its original value

# # spectra plots
bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, pltOn=True, pltName='CS3_spectraACC')
bb_func.evaluate([1, 2, 4, 8]) #acc
bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, pltOn=True, OF=2, pltName='CS3_spectraPS')
bb_func.evaluate([5, 1, 2, 2]) #ps
bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, pltOn=True, OF=2, pltName='CS3_spectraRH')
bb_func.evaluate([7, 0, 2, 1]) #rh
bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, pltOn=True, OF=2, pltName='CS3_spectraINST')
bb_func.evaluate([6, 0, 4, 3]) #inst
bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, pltOn=True, OF=2, pltName='CS3_spectraMR')
bb_func.evaluate([5, 2, 3, 2]) #mr
bb_func = BBFunc(DataFile, Rob=Robmetric, Dis=DisMeas, pltOn=True, OF=2, pltName='CS3_spectraROB')
bb_func.evaluate([7, 1, 3, 3]) #rob