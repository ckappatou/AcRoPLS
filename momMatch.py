##
# momMatch.py
#
# Moment matching and divide categories functions
#
# C.D. Kappatou, J.Odgers, S. Garcia, R.Misener: "Optimization Methods for Developing Efficient Chemometric Models", 2022.
# 
##

import pandas as pd
import numpy as np
import pyphi as phi
import pyphi_plots as pp
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# Todo: change function to take as arguments subclasses (true/false) index as many input arg, where depending on its length the code is automatically adjusted
def divCat(mvmobj, X, Y, *, rob_def=1, CLASSID=False, colorby=False, subclass=False, \
           subclass_index=False, subclass2=False, subclass2_index=False, subclass3=False, \
           subclass3_index=False, x_space=False):
    """
    divide categories

    mvmobj: A model created with phi.pca or phi.pls

    X/Y:     Data [numpy arrays or pandas dataframes]

    optional:
    rob_def: robustness definition 1 will be used unless otherwise specified by the user
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
        # class_mean_pred_corrected = []

        ols = linear_model.LinearRegression()
        reg = ols.fit(y_mes_.reshape(-1, 1), y_pred_.reshape(-1, 1))
        # plt.figure()
        # plt.plot(y_mes_, reg.predict(y_mes_.reshape(-1, 1)), linewidth=2, color="blue")

        if not (isinstance(CLASSID, bool)):  #I have classes
            Classes_ = np.unique(CLASSID[:, colorby]).tolist()
            classid_ = list(CLASSID[:, colorby])
            # different_colors = len(Classes_)
            for classid_in_turn in Classes_:
                y_mes_aux = []
                y_pred_aux = []
                obsid_aux = []
                classid_aux = []
                y_olsOfPls_aux = []

                if not (isinstance(subclass, bool)): #I have subclasses
                    if not (isinstance(subclass2, bool)): #I have at least 2 subclasses
                        if not (isinstance(subclass3, bool)):  # I have 3 subclasses
                            for i in list(range(len(ObsID_))):
                                if classid_[i] == classid_in_turn \
                                        and not (np.isnan(y_mes_[i])) and CLASSID[i, subclass_index] == subclass \
                                        and CLASSID[i, subclass2_index] == subclass2 \
                                        and CLASSID[i, subclass3_index] == subclass3:
                                    y_mes_aux.append(y_mes_[i])
                                    y_pred_aux.append(y_pred_[i])
                                    obsid_aux.append(ObsID_[i])
                                    classid_aux.append(str(classid_in_turn) + '_' + str(subclass) + '_' + str(subclass2) + '_' + str(
                                            subclass3))
                                    y_olsOfPls_aux.append(reg.predict(np.c_[y_mes_.reshape(-1, 1)[i]]))
                        else: #two subclasses
                            for i in list(range(len(ObsID_))):
                                if classid_[i] == classid_in_turn \
                                        and not (np.isnan(y_mes_[i])) and CLASSID[i, subclass_index] == subclass \
                                        and CLASSID[i, subclass2_index] == subclass2:
                                    y_mes_aux.append(y_mes_[i])
                                    y_pred_aux.append(y_pred_[i])
                                    obsid_aux.append(ObsID_[i])
                                    classid_aux.append(str(classid_in_turn) + '_' + str(subclass) + '_' + str(subclass2))
                                    y_olsOfPls_aux.append(reg.predict(np.c_[y_mes_.reshape(-1, 1)[i]]))
                    else: #one subclass
                        for i in list(range(len(ObsID_))):
                            if classid_[i] == classid_in_turn and not (np.isnan(y_mes_[i])) \
                                    and CLASSID[i, subclass_index] == subclass:
                                y_mes_aux.append(y_mes_[i])
                                y_pred_aux.append(y_pred_[i])
                                obsid_aux.append(ObsID_[i])
                                classid_aux.append(str(classid_in_turn) + '_' + str(subclass))
                                y_olsOfPls_aux.append(reg.predict(np.c_[y_mes_.reshape(-1, 1)[i]]))
                else: #no subclasses
                    for i in list(range(len(ObsID_))):
                        if classid_[i] == classid_in_turn and not (np.isnan(y_mes_[i])):
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
                    # todo option alternative ddof=1
                    class_errorY_var.append(np.var(errorY_, ddof=0))
                    class_var_pred.append(np.var(y_pred_aux, ddof=0))
                    class_mean_pred.append(np.mean(y_pred_aux))
                    # class_mean_pred_corrected.append(np.mean(y_pred_aux)-np.mean(y_mes_aux))
                    errorYolsOfPls_= (y_mes_aux_ - y_olsOfPls_aux)
                    class_errorYolsOfPls_avg.append(np.mean(errorYolsOfPls_))
                    class_errorYolsOfPls_var.append(np.var(errorYolsOfPls_, ddof=0))
    if rob_def == 1:
        return class_mean_pred,class_var_pred
    elif rob_def == 2:
        return class_errorY_avg, class_errorY_var
    elif rob_def == 3:
        return class_errorYolsOfPls_avg, class_errorYolsOfPls_var

# Todo: change function to take as arguments subclasses (true/false) index as many input arg, where depending on its length the code is automatically adjusted
def momMatch(mvmobj, X, Y, *, rob_def=1, CLASSID=False, cat_rob_index=False, subclass_index=False,\
                     subclass2_index=False, subclass3_index=False, dis=1):
    # Moment matching
    M1 = []
    M2 = []
    if not (isinstance(CLASSID, bool)):  # I have classes
        if isinstance(CLASSID, np.ndarray):
            CLASSIDarr = CLASSID
        else:
            CLASSIDarr = CLASSID.values #make sure I pass CLASSID as an array CLASSIDarr
        if not (isinstance(subclass_index, bool)):  # I have subclasses
            if not (isinstance(subclass2_index, bool)):  # I have at least 2 subclasses
                if not (isinstance(subclass3_index, bool)):  # I have 3 subclasses
                    #   First division
                    Classes_ = np.unique(CLASSIDarr[:, subclass_index]).tolist()
                    for classid_in_turn in Classes_:
                        # Second division
                        Classes_2 = np.unique(CLASSIDarr[:, subclass2_index]).tolist()
                        for classid_in_turn2 in Classes_2:
                            # Third division
                            Classes_3 = np.unique(CLASSIDarr[:, subclass3_index]).tolist()
                            for classid_in_turn3 in Classes_3:
                                # Division in last category; the one we want to be robust to
                                Cmean, Cvar = divCat(mvmobj, X, Y, rob_def=rob_def, CLASSID=CLASSIDarr, \
                                                     colorby=cat_rob_index, subclass=classid_in_turn,
                                                     subclass_index=subclass_index, subclass2=classid_in_turn2,
                                                     subclass2_index=subclass2_index, subclass3=classid_in_turn3,
                                                     subclass3_index=subclass3_index)
                                if Cmean and Cvar:
                                    M1.append(Cmean)
                                    M2.append(np.square(Cmean) + Cvar)
                else:  # two subclasses
                    #   First division
                    Classes_ = np.unique(CLASSIDarr[:, subclass_index]).tolist()
                    for classid_in_turn in Classes_:
                        # Second division
                        Classes_2 = np.unique(CLASSIDarr[:, subclass2_index]).tolist()
                        for classid_in_turn2 in Classes_2:
                            # Division in last category; the one we want to be robust to
                            Cmean, Cvar = divCat(mvmobj, X, Y, rob_def=rob_def, CLASSID=CLASSIDarr,\
                                                 colorby=cat_rob_index, subclass=classid_in_turn,\
                                                 subclass_index=subclass_index, subclass2=classid_in_turn2,\
                                                 subclass2_index=subclass2_index)
                            if Cmean and Cvar:
                                M1.append(Cmean)
                                M2.append(np.square(Cmean) + Cvar)
            else:  # one subclass
                #   First division
                Classes_ = np.unique(CLASSIDarr[:, subclass_index]).tolist()
                for classid_in_turn in Classes_:
                    # Division in last category; the one we want to be robust to
                    Cmean, Cvar = divCat(mvmobj, X, Y,  rob_def=rob_def, CLASSID=CLASSIDarr, colorby=cat_rob_index, \
                                         subclass=classid_in_turn, subclass_index=subclass_index)
                    if Cmean and Cvar:
                        M1.append(Cmean)
                        M2.append(np.square(Cmean) + Cvar)
        else: #no subclasses
            Cmean, Cvar = divCat(mvmobj, X, Y, rob_def=rob_def, CLASSID=CLASSIDarr, colorby=cat_rob_index)
            if Cmean and Cvar:
                M1.append(Cmean)
                M2.append(np.square(Cmean) + Cvar)

    # Distance (square) of M1 for the different Type across all different scales
    M1_dis_sum = []
    for i in range(len(M1)):
        M1_dis_obj = product(M1[i], repeat=2)
        M1_dis_list = list(map(list, M1_dis_obj))
        M1_dis = []
        for item in M1_dis_list:
            M1_dis.append(abs(item[0] - item[1]))
        M1_dis = np.reshape(M1_dis, (len(M1[i]), len(M1[i])))
        M1_disU_elements = M1_dis[np.triu_indices(len(M1[i]), k=1)]
        # M1_dis_sum.append(sum(np.square(M1_disU_elements)))
        if M1_disU_elements.size>0:
            if dis==1: # L2 norm = Eucledean(squared) distance
                M1_dis_sum.append(sum(np.square(M1_disU_elements))/len(M1_disU_elements)) #if only one var source div by len is not necessary
            elif dis==2: # L1 norm = Manhattan or Taxicab distance
                M1_dis_sum.append(sum(M1_disU_elements) / len(M1_disU_elements)) #if only one var source div by len is not necessary
    # if len(M1)>=1:
    #     M1sum = sum(M1_dis_sum) / len(M1)
    # it can be that not all categories contribute distances (i.e. cat with only one element) in that case although we divide with len(M1) M1_dis_sum may have fewer elements
    # correction for division only with categories that contribute distances
    if len(M1_dis_sum)>=1:
        M1sum = sum(M1_dis_sum) / len(M1_dis_sum)
    else:
        M1sum = sum(M1_dis_sum)

    # Distance between M2 for the different Type across all different scales
    M2_dis_sum = []
    for i in range(len(M2)):
        M2_dis_obj = product(M2[i], repeat=2)
        M2_dis_list = list(map(list, M2_dis_obj))
        M2_dis = []
        for item in M2_dis_list:
            M2_dis.append(abs(item[0] - item[1]))
        M2_dis = np.reshape(M2_dis, (len(M2[i]), len(M2[i])))
        M2_disU_elements = M2_dis[np.triu_indices(len(M2[i]), k=1)]
        # M2_dis_sum.append(sum(np.square(M2_disU_elements)))
        if M2_disU_elements.size>0:
            if dis==1: # L2 norm = Eucledean(squared) distance
                M2_dis_sum.append(sum(np.square(M2_disU_elements)) / len(M2_disU_elements)) #if only one var source div by len is not necessary
            elif dis==2:# L1 norm = Manhattan or Taxicab distance
                M2_dis_sum.append(sum(M2_disU_elements) / len(M2_disU_elements)) #if only one var source div by len is not necessary
    # if len(M2)>=1:
    #     M2sum = sum(M2_dis_sum) / len(M2)
    # correction for division only with categories that contribute distances
    if len(M2_dis_sum)>=1:
        M2sum = sum(M2_dis_sum) / len(M2_dis_sum)
    else:
        M2sum = sum(M2_dis_sum)
    return M1sum, M2sum
