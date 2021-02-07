from scipy.io import loadmat
import numpy as np
from sklearn import svm
# from sklearn.svm import LinearSVC
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from scikits.learn.svm import sparse
import utilities
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


#  hyper parameters
cross_valid = 5
step_thr = .01

# constant values
seed_num = 8
line_width = 2
output_path ='./Results/'

np.random.seed(seed_num)
svm_model = True  # svm
svm_l1_model = False  # svm with lasso
gp_model = True  # gaussian process
nb_model = True  # gaussian naive bayesian
rf_model = True  # random forest
kn_model = True  # K-Nearest Neighbor
mlp_model = True  # mlp
LR_model = True  # Logistic Regression


data_mode_list = ['combination', 'high_gamma', 'lp_ds']

# data_mode = 'combination' # high_gamma, lp_ds, combination
# DATA_type = ['FLICKER', 'SQ'] # FLICKER, SQ
DATA_type = ['FLICKER'] # FLICKER, SQ
FLICKER_type = ['color', 'shape']
for data_mode in data_mode_list:
    for data_type in DATA_type:
        for flicker_type in FLICKER_type:
            if data_type == 'SQ':
                if flicker_type == 'shape':
                    break

            if data_type == 'FLICKER':
                data_path = './data/FLICKER/'
                list_num = [9, 10, 11, 12, 13, 14, 15]
                data_info = data_type + '_' + flicker_type

            elif data_type == 'SQ':
                data_path = './data/SQ/'
                list_num = [1, 2, 3, 4, 5, 6, 7, 8, 13]
                data_info = data_type




            # list_num = [11]
            for i in range(len(list_num)):

                if list_num[i] < 10:
                    patient = 'p0' + str(list_num[i])
                else:
                    patient = 'p' + str(list_num[i])

                print('----------------\n patient:  ' + patient + '\n-----------------')

                if data_type == 'FLICKER':
                    tail_path = patient + '_' + flicker_type + '_' + data_type + '.mat'
                elif data_type == 'SQ':
                    tail_path = patient  + '_' + data_type + '.mat'

                if data_mode == 'combination':
                    trail_onset_lp_ds = loadmat(data_path + 'trail_onset_lp_ds_' + tail_path)
                    trail_onset_high_gamma = loadmat(data_path + 'trail_onset_high_gamma_' + tail_path)
                else:
                    trail_onset = loadmat(data_path + 'trail_onset_' + data_mode + '_' + tail_path)



                if  data_mode == 'combination':
                    black_trail_onset_lp_ds = np.array(trail_onset_lp_ds['black_trail_onset_lp_ds'])
                    white_trail_onset_lp_ds = np.array(trail_onset_lp_ds['white_trail_onset_lp_ds'])

                    black_trail_onset_high_gamma = np.array(trail_onset_high_gamma['black_trail_onset_high_gamma'])
                    white_trail_onset_high_gamma = np.array(trail_onset_high_gamma['white_trail_onset_high_gamma'])

                    black_trail_onset = np.concatenate((black_trail_onset_lp_ds, black_trail_onset_high_gamma), axis=1)
                    white_trail_onset = np.concatenate((white_trail_onset_lp_ds, white_trail_onset_high_gamma), axis=1)
                else:
                    black_trail_onset = np.array(trail_onset['black_trail_onset_' + data_mode])
                    white_trail_onset = np.array(trail_onset['white_trail_onset_' + data_mode])

                m_b = black_trail_onset.shape
                m_w = white_trail_onset.shape

                num_trial_b = m_b[0]
                num_trial_w = m_w[0]
                feature_num = m_b[2]
                ch_num = m_b[1]

                feat_tot = np.concatenate((black_trail_onset, white_trail_onset), axis=0)
                lbl_tot = np.array([-1]*(num_trial_b) + [1]*(num_trial_w))
                data_l = lbl_tot.shape[0]

                for cross_valid in range(cross_valid, cross_valid+1):
                    acc_svm_cv = np.zeros((ch_num, cross_valid))
                    acc_svm_mean = np.zeros((ch_num, 1))
                    acc_svm_std = np.zeros((ch_num, 1))
                    acc_svm_l1_mean = np.zeros((ch_num, 1))
                    acc_svm_l1_std = np.zeros((ch_num, 1))
                    acc_gp_cv = np.zeros((ch_num, cross_valid))
                    acc_gp_mean = np.zeros((ch_num, 1))
                    acc_gp_std = np.zeros((ch_num, 1))
                    acc_nb_cv = np.zeros((ch_num, cross_valid))
                    acc_nb_mean = np.zeros((ch_num, 1))
                    acc_nb_std = np.zeros((ch_num, 1))
                    acc_rf_cv = np.zeros((ch_num, cross_valid))
                    acc_rf_mean = np.zeros((ch_num, 1))
                    acc_rf_std = np.zeros((ch_num, 1))
                    acc_kn_cv = np.zeros((ch_num, cross_valid))
                    acc_kn_mean = np.zeros((ch_num, 1))
                    acc_kn_std = np.zeros((ch_num, 1))
                    acc_mlp_cv = np.zeros((ch_num, cross_valid))
                    acc_mlp_mean = np.zeros((ch_num, 1))
                    acc_mlp_std = np.zeros((ch_num, 1))
                    acc_lr_cv = np.zeros((ch_num, cross_valid))
                    acc_lr_mean = np.zeros((ch_num, 1))
                    acc_lr_std = np.zeros((ch_num, 1))

                    pred_svm_tot = np.zeros((ch_num, data_l))
                    pred_svm_l1_tot = np.zeros((ch_num, data_l))
                    pred_gp_tot = np.zeros((ch_num, data_l))
                    pred_nb_tot = np.zeros((ch_num, data_l))
                    pred_nb_tot = np.zeros((ch_num, data_l))
                    pred_rf_tot = np.zeros((ch_num, data_l))
                    pred_kn_tot = np.zeros((ch_num, data_l))
                    pred_mlp_tot = np.zeros((ch_num, data_l))
                    pred_lr_tot = np.zeros((ch_num, data_l))

                    for ch in range(ch_num): # different channels
                        print(ch)
                        # DATA for each channel
                        feat_ch = feat_tot[:, ch, :]

                        # %%  SVM classifier
                        if svm_model:
                            # non linear kernel
                            clf_svm = svm.SVC(C=10)
                            scores_svm = cross_val_score(
                                clf_svm, feat_ch, lbl_tot, cv=cross_valid, scoring='accuracy')
                            acc_svm_cv[ch, :] = scores_svm
                            acc_svm_mean[ch, 0] = scores_svm.mean()
                            acc_svm_std[ch, 0] = scores_svm.std()

                            pred_svm_ch = cross_val_predict(clf_svm, feat_ch, lbl_tot, cv=cross_valid)
                            pred_svm_tot[ch, :] = pred_svm_ch

                        # linear kernel- l1 regulizer

                        if svm_l1_model:
                            clf_svm_l1 = svm.LinearSVC(C=10, intercept_scaling=100, penalty='l1', dual=False)
                            scores_svm_l1 = cross_val_score(
                                clf_svm_l1, feat_ch, lbl_tot, cv=cross_valid, scoring='accuracy')

                            # print(clf_svm_l1.coef_)
                            acc_svm_l1_mean[ch, 0] = scores_svm_l1.mean()
                            acc_svm_l1_std[ch, 0] = scores_svm_l1.std()

                            pred_svm_l1_ch = cross_val_predict(clf_svm_l1, feat_ch, lbl_tot, cv=cross_valid)
                            pred_svm_l1_tot[ch, :] = pred_svm_l1_ch

                        # %% Gaussian Process classifier
                        if gp_model:
                            kernel = 1.0 * RBF(1.0)
                            clf_gp = GaussianProcessClassifier(kernel=kernel, random_state=0)

                            scores_gp = cross_val_score(
                                clf_gp, feat_ch, lbl_tot, cv=cross_valid, scoring='accuracy')

                            acc_gp_cv[ch, :] = scores_gp
                            acc_gp_mean[ch, 0] = scores_gp.mean()
                            acc_gp_std[ch, 0] = scores_gp.std()

                            pred_gp_ch = cross_val_predict(clf_gp, feat_ch, lbl_tot, cv=cross_valid)
                            pred_gp_tot[ch, :] = pred_gp_ch

                        # %% Naive Bayesian
                        if nb_model:
                            kernel = 1.0 * RBF(1.0)
                            clf_nb = GaussianNB(var_smoothing=.01)

                            scores_nb = cross_val_score(
                                clf_nb, feat_ch, lbl_tot, cv=cross_valid, scoring='accuracy')

                            acc_nb_cv[ch, :] = scores_nb
                            acc_nb_mean[ch, 0] = scores_nb.mean()
                            acc_nb_std[ch, 0] = scores_nb.std()

                            pred_nb_ch = cross_val_predict(clf_nb, feat_ch, lbl_tot, cv=cross_valid)
                            pred_nb_tot[ch, :] = pred_nb_ch

                        # %% Random Forest classifier
                        if rf_model:
                            clf_rf = RandomForestClassifier(random_state=0)
                            scores_rf = cross_val_score(
                                clf_rf, feat_ch, lbl_tot, cv=cross_valid, scoring='accuracy')

                            acc_rf_cv[ch, :] = scores_rf
                            acc_rf_mean[ch, 0] = scores_rf.mean()
                            acc_rf_std[ch, 0] = scores_rf.std()

                            pred_rf_ch = cross_val_predict(clf_rf, feat_ch, lbl_tot, cv=cross_valid)
                            pred_rf_tot[ch, :] = pred_rf_ch

                        # %% K-Nearest Neighbor
                        if kn_model:
                            clf_kn = KNeighborsClassifier()

                            scores_kn = cross_val_score(
                                clf_kn, feat_ch, lbl_tot, cv=cross_valid, scoring='accuracy')

                            acc_kn_cv[ch, :] = scores_kn
                            acc_kn_mean[ch, 0] = scores_kn.mean()
                            acc_kn_std[ch, 0] = scores_kn.std()

                            pred_kn_ch = cross_val_predict(clf_kn, feat_ch, lbl_tot, cv=cross_valid)
                            pred_kn_tot[ch, :] = pred_kn_ch

                        # %% Multi-layer Perceptorn
                        if mlp_model:
                            clf_NN = MLPClassifier(max_iter=10000, alpha=1, hidden_layer_sizes=(20, 20,))

                            scores_mlp = cross_val_score(
                                clf_NN, feat_ch, lbl_tot, cv=cross_valid, scoring='accuracy')

                            acc_mlp_cv[ch, :] = scores_mlp
                            acc_mlp_mean[ch, 0] = scores_mlp.mean()
                            acc_mlp_std[ch, 0] = scores_mlp.std()

                            pred_mlp_ch = cross_val_predict(clf_NN, feat_ch, lbl_tot, cv=cross_valid)
                            pred_mlp_tot[ch, :] = pred_mlp_ch

                        # %%  Logistic Regression
                        if LR_model:
                            # non linear kernel
                            clf_lr = LogisticRegression(random_state=0)
                            scores_lr = cross_val_score(
                                clf_lr, feat_ch, lbl_tot, cv=cross_valid, scoring='accuracy')

                            acc_lr_cv[ch, :] = scores_lr
                            acc_lr_mean[ch, 0] = scores_lr.mean()
                            acc_lr_std[ch, 0] = scores_lr.std()

                            pred_lr_ch = cross_val_predict(clf_lr, feat_ch, lbl_tot, cv=cross_valid)
                            pred_lr_tot[ch, :] = pred_lr_ch

                    # %% majority voting & plot
                    # SVM
                    odd_state = True
                    if svm_model:
                        model_n = 'SVM'
                        utilities.save_info_single_max(acc_svm_mean, acc_svm_cv, output_path, model_n, cross_valid, patient, data_mode, data_info)


                        # hard voting
                        # thr_list_svm, ch_no_list_svm, acc_list_svm = utilities.hard_voting(pred_svm_tot, acc_svm_mean, step_thr, lbl_tot, model_n)
                        thr_list_svm, ch_no_list_svm, acc_mean_list_svm, acc_min_list_svm, acc_max_list_svm = utilities.hard_voting_2(pred_svm_tot,
                                                                                                                                      acc_svm_mean,
                                                                                                                                      step_thr, lbl_tot,
                                                                                                                                      model_n,
                                                                                                                                      output_path,
                                                                                                                                      cross_valid,
                                                                                                                                      patient,
                                                                                                                                      data_mode,
                                                                                                                                      data_info,
                                                                                                                                      odd_mode=odd_state)
                        # plot
                        utilities.plt_signle_ch(acc_svm_mean, data_l, model_n, output_path, line_width, cross_valid, patient, data_mode, data_info)
                        # utilities.plt_best_majority(thr_list_svm, ch_no_list_svm, acc_list_svm, model_n, output_path, line_width, cross_valid, patient, data_mode)
                        utilities.plt_best_majority_2(thr_list_svm, ch_no_list_svm, acc_mean_list_svm, acc_min_list_svm, acc_max_list_svm, model_n,
                                                      output_path, line_width, cross_valid, patient, data_mode, data_info)

                    # SVM - l1
                    if svm_l1_model:
                        model_n = 'SVM_l1'

                        # hard voting
                        thr_list_svm, ch_no_list_svm, acc_list_svm = utilities.hard_voting(pred_svm_l1_tot, acc_svm_l1_mean, step_thr, lbl_tot, model_n)

                        # plot
                        utilities.plt_signle_ch(acc_svm_l1_mean, data_l, model_n, output_path, line_width, cross_valid, patient, data_mode, data_info)
                        utilities.plt_best_majority(thr_list_svm, ch_no_list_svm, acc_list_svm, model_n, output_path, line_width, cross_valid, patient,
                                                    data_mode, data_info)

                    # kn
                    if kn_model:
                        model_n = 'KN'
                        utilities.save_info_single_max(acc_kn_mean, acc_kn_cv, output_path, model_n, cross_valid, patient, data_mode, data_info)
                        # hard voting
                        # thr_list_kn, ch_no_list_kn, acc_list_kn = utilities.hard_voting(pred_kn_tot, acc_kn_mean, step_thr, lbl_tot, model_n)

                        thr_list_kn, ch_no_list_kn, acc_mean_list_kn, acc_min_list_kn, acc_max_list_kn = utilities.hard_voting_2(pred_kn_tot,
                                                                                                                                      acc_kn_mean,
                                                                                                                                      step_thr, lbl_tot,
                                                                                                                                      model_n,
                                                                                                                                 output_path,
                                                                                                                                 cross_valid,
                                                                                                                                 patient,
                                                                                                                                 data_mode,
                                                                                                                                 data_info,
                                                                                                                                 odd_mode=odd_state)
                        # plot
                        utilities.plt_signle_ch(acc_kn_mean, data_l, model_n, output_path, line_width, cross_valid, patient, data_mode, data_info)
                        utilities.plt_best_majority_2(thr_list_kn, ch_no_list_kn, acc_mean_list_kn, acc_min_list_kn, acc_max_list_kn, model_n,
                                                      output_path, line_width, cross_valid, patient, data_mode, data_info)


                    # NN
                    if mlp_model:
                        model_n = 'NN'
                        utilities.save_info_single_max(acc_mlp_mean, acc_mlp_cv, output_path, model_n, cross_valid, patient, data_mode, data_info)
                        # hard voting
                        # thr_list_NN, ch_no_list_NN, acc_list_NN = utilities.hard_voting(pred_mlp_tot, acc_mlp_mean, step_thr, lbl_tot, model_n)

                        # plot
                        # utilities.plt_best_majority(thr_list_NN, ch_no_list_NN, acc_list_NN, model_n, output_path, line_width, cross_valid, patient, data_mode)

                        thr_list_mlp, ch_no_list_mlp, acc_mean_list_mlp, acc_min_list_mlp, acc_max_list_mlp = utilities.hard_voting_2(pred_mlp_tot,
                                                                                                                                 acc_mlp_mean,
                                                                                                                                 step_thr, lbl_tot,
                                                                                                                                 model_n,
                                                                                                                                      output_path,
                                                                                                                                      cross_valid,
                                                                                                                                      patient,
                                                                                                                                      data_mode,
                                                                                                                                      data_info,
                                                                                                                                      odd_mode= odd_state)
                        # plot
                        utilities.plt_signle_ch(acc_mlp_mean, data_l, model_n, output_path, line_width, cross_valid, patient, data_mode, data_info)
                        utilities.plt_best_majority_2(thr_list_mlp, ch_no_list_mlp, acc_mean_list_mlp, acc_min_list_mlp, acc_max_list_mlp, model_n,
                                                      output_path, line_width, cross_valid, patient, data_mode, data_info)


                    # Gasussian Process
                    if gp_model:
                        model_n = 'GP'
                        utilities.save_info_single_max(acc_gp_mean, acc_gp_cv, output_path, model_n, cross_valid, patient, data_mode, data_info)
                        # hard voting
                        # thr_list_gp, ch_no_list_gp, acc_list_gp = utilities.hard_voting(pred_gp_tot, acc_gp_mean, step_thr, lbl_tot, model_n)

                        # plot
                        # utilities.plt_best_majority(thr_list_gp, ch_no_list_gp, acc_list_gp, model_n, output_path, line_width, cross_valid, patient, data_mode)

                        thr_list_gp, ch_no_list_gp, acc_mean_list_gp, acc_min_list_gp, acc_max_list_gp = utilities.hard_voting_2(pred_gp_tot,
                                                                                                                                      acc_gp_mean,
                                                                                                                                      step_thr, lbl_tot,
                                                                                                                                      model_n,
                                                                                                                                 output_path,
                                                                                                                                 cross_valid,
                                                                                                                                 patient,
                                                                                                                                 data_mode,
                                                                                                                                 data_info,
                                                                                                                                 odd_mode= odd_state)
                        # plot
                        utilities.plt_signle_ch(acc_gp_mean, data_l, model_n, output_path, line_width, cross_valid, patient, data_mode, data_info)
                        utilities.plt_best_majority_2(thr_list_gp, ch_no_list_gp, acc_mean_list_gp, acc_min_list_gp, acc_max_list_gp, model_n,
                                                      output_path, line_width, cross_valid, patient, data_mode, data_info)



                    # Naive Bayesian
                    if nb_model:
                        model_n = 'NB'
                        utilities.save_info_single_max(acc_nb_mean, acc_nb_cv, output_path, model_n, cross_valid, patient, data_mode, data_info)
                        # hard voting
                        # thr_list_nb, ch_no_list_nb, acc_list_nb = utilities.hard_voting(pred_nb_tot, acc_nb_mean, step_thr, lbl_tot, model_n)

                        # plot
                        # utilities.plt_best_majority(thr_list_nb, ch_no_list_nb, acc_list_nb, model_n, output_path, line_width, cross_valid, patient, data_mode)

                        thr_list_nb, ch_no_list_nb, acc_mean_list_nb, acc_min_list_nb, acc_max_list_nb = utilities.hard_voting_2(pred_nb_tot,
                                                                                                                                 acc_nb_mean,
                                                                                                                                 step_thr, lbl_tot,
                                                                                                                                 model_n,
                                                                                                                                 output_path,
                                                                                                                                 cross_valid,
                                                                                                                                 patient,
                                                                                                                                 data_mode,
                                                                                                                                 data_info,
                                                                                                                                 odd_mode= odd_state)
                        # plot
                        utilities.plt_signle_ch(acc_nb_mean, data_l, model_n, output_path, line_width, cross_valid, patient, data_mode, data_info)
                        utilities.plt_best_majority_2(thr_list_nb, ch_no_list_nb, acc_mean_list_nb, acc_min_list_nb, acc_max_list_nb, model_n,
                                                      output_path, line_width, cross_valid, patient, data_mode, data_info)

                    # Random Forest
                    if rf_model:
                        model_n = 'RF'
                        utilities.save_info_single_max(acc_rf_mean, acc_rf_cv, output_path, model_n, cross_valid, patient, data_mode, data_info)
                        # voting
                        # thr_list_rf, ch_no_list_rf, acc_list_rf = utilities.hard_voting(pred_rf_tot, acc_rf_mean, step_thr, lbl_tot, model_n)

                        # plot
                        # utilities.plt_best_majority(thr_list_rf, ch_no_list_rf, acc_list_rf, model_n, output_path, line_width, cross_valid, patient, data_mode)

                        thr_list_rf, ch_no_list_rf, acc_mean_list_rf, acc_min_list_rf, acc_max_list_rf = utilities.hard_voting_2(pred_rf_tot,
                                                                                                                                 acc_rf_mean,
                                                                                                                                 step_thr, lbl_tot,
                                                                                                                                 model_n,
                                                                                                                                 output_path,
                                                                                                                                 cross_valid,
                                                                                                                                 patient,
                                                                                                                                 data_mode,
                                                                                                                                 data_info,
                                                                                                                                 odd_mode= odd_state)
                        # plot
                        utilities.plt_signle_ch(acc_rf_mean, data_l, model_n, output_path, line_width, cross_valid, patient, data_mode, data_info)
                        utilities.plt_best_majority_2(thr_list_rf, ch_no_list_rf, acc_mean_list_rf, acc_min_list_rf, acc_max_list_rf, model_n,
                                                      output_path, line_width, cross_valid, patient, data_mode, data_info)

                    # Logistic Regression
                    if LR_model:
                        model_n = 'LR'
                        utilities.save_info_single_max(acc_lr_mean, acc_lr_cv, output_path, model_n, cross_valid, patient, data_mode, data_info)
                        # hard voting
                        # thr_list_lr, ch_no_list_lr, acc_list_lr = utilities.hard_voting(pred_lr_tot, acc_lr_mean, step_thr, lbl_tot, model_n)
                        thr_list_lr, ch_no_list_lr, acc_mean_list_lr, acc_min_list_lr, acc_max_list_lr = utilities.hard_voting_2(pred_lr_tot,
                                                                                                                                 acc_lr_mean,
                                                                                                                                 step_thr, lbl_tot,
                                                                                                                                 model_n,
                                                                                                                                 output_path,
                                                                                                                                 cross_valid,
                                                                                                                                 patient,
                                                                                                                                 data_mode,
                                                                                                                                 data_info,
                                                                                                                                 odd_mode=False)
                        # plot
                        utilities.plt_signle_ch(acc_lr_mean, data_l, model_n, output_path, line_width, cross_valid, patient, data_mode, data_info)
                        # utilities.plt_best_majority(thr_list_lr, ch_no_list_lr, acc_list_lr, model_n, output_path, line_width, cross_valid, patient, data_mode)
                        utilities.plt_best_majority_2(thr_list_lr, ch_no_list_lr, acc_mean_list_lr, acc_min_list_lr, acc_max_list_lr, model_n,
                                                      output_path, line_width, cross_valid, patient, data_mode, data_info)

                    K=1
