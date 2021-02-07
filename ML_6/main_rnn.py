from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import KFold
import utilities
import tensorflow as tf

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# hyper param
epoch_num = 300
fold_num = 5

step_thr = 0.01
# constant values
line_width = 2
output_path = './Results/'

max_cell = 13
min_all_cell = np.zeros((1, max_cell - 1))
mean_all_cell = np.zeros((1, max_cell - 1))
max_all_cell = np.zeros((1, max_cell - 1))


data_mode = 'combination' # high_gamma, lp_ds, combination
# DATA_type = ['SQ', 'FLICKER'] # FLICKER, SQ
DATA_type = ['FLICKER'] # FLICKER, SQ
FLICKER_type = ['color', 'shape'] #color,  shape
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



        for num_cell in range(max_cell, max_cell + 1):
            '''Creating Model'''
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.LSTM(units=num_cell, activation='tanh', name='rnn_layer', input_shape=(15, 1)))
            model.add(tf.keras.layers.Dense(units=1, activation='sigmoid', name='output_layer'))

            # plotting model
            tf.keras.utils.plot_model(model, 'LSTM_model.png', show_shapes=True, show_layer_names=True, expand_nested=True)

            # %% compelling the model
            optimzer_n = 'rmsprop'
            loss_n = 'binary_crossentropy'
            metric_n = 'accuracy'
            model.compile(optimizer=optimzer_n, loss=loss_n, metrics=["accuracy"])

            model.summary()

            '''loading and preparing data'''
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
                    tail_path = patient + '_' + data_type + '.mat'

                if data_mode == 'combination':
                    trail_onset_lp_ds = loadmat(data_path + 'trail_onset_lp_ds_' + tail_path)
                    trail_onset_high_gamma = loadmat(data_path + 'trail_onset_high_gamma_' + tail_path)
                else:
                    trail_onset = loadmat(data_path + 'trail_onset_' + data_mode + '_' + tail_path)

                if data_mode == 'combination':
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
                lbl_tot = np.array([0] * (num_trial_b) + [1] * (num_trial_w))
                data_l = lbl_tot.shape[0]

                acc_rnn_mean = np.zeros((ch_num, 1))
                acc_rnn_cv = np.zeros((ch_num, fold_num))
                pred_rnn_tot = np.zeros((ch_num, num_trial_b + num_trial_w))

                acc_test_ch_all_fold = np.zeros(shape=(ch_num, fold_num))

                for ch in range(ch_num):  # different channel
                    print('===========================')
                    print('Ch: ' + str(ch))
                    print('===========================')
                    # DATA for each channel
                    feat_ch = feat_tot[:, ch, :]
                    kf = KFold(n_splits=fold_num)
                    acc_fold_ch = []
                    prob_test_ch = []
                    i = 0
                    for train_index, test_index in kf.split(feat_ch):  # fold
                        i += 1
                        print('#fold = ' + str(i))
                        print('             TRAIN           ')
                        # print("TRAIN:", train_index, "TEST:", test_index)
                        feat_tr, feat_te = feat_ch[train_index], feat_ch[test_index]
                        lbl_tr, lbl_te = lbl_tot[train_index], lbl_tot[test_index]

                        m_ft_tr = feat_tr.shape
                        m_ft_te = feat_te.shape
                        feat_tr = tf.reshape(feat_tr, shape=(m_ft_tr[0], m_ft_tr[1], 1))
                        feat_te = tf.reshape(feat_te, shape=(m_ft_te[0], m_ft_te[1], 1))

                        model.fit(feat_tr, lbl_tr, epochs=epoch_num, verbose=0)

                        print(' ------------------------------ ')
                        print('             TEST           ')
                        test_loss, test_accuracy = model.evaluate(feat_te, lbl_te)
                        test_prob = model.predict(feat_te)
                        print(' ------------------------------ ')
                        acc_fold_ch.append(test_accuracy)
                        prob_test_ch.extend(test_prob)

                    # adding prediction of each chanel for all test
                    prob_test_ch_all = np.reshape(np.array(prob_test_ch), (np.array(prob_test_ch).shape[1], np.array(prob_test_ch).shape[0]))
                    ind_1 = np.where(prob_test_ch_all > .5)
                    pred_ch = np.zeros((prob_test_ch_all.shape[0], prob_test_ch_all.shape[1]))
                    pred_ch[ind_1] = 1
                    pred_rnn_tot[ch, :] = pred_ch

                    # adding accuracy of test
                    acc_rnn_mean[ch, 0] = np.mean(acc_fold_ch)
                    acc_rnn_cv[ch, :] = acc_fold_ch
                    print(' ******************* \n patient: ' + patient + ' -- #cell' + str(num_cell) + ' -- #ch :' + str(ch))
                    print('acc of ' + str(fold_num) + ' fold : ' + str(acc_fold_ch))
                    print('average acc of ch ' + str(ch) + ' : ' + str(acc_rnn_mean[ch, 0]))
                    print(' ******************* ')

                    acc_test_ch_all_fold[ch, :] = acc_fold_ch
                    k = 1



                model_n = 'lstm'
                utilities.save_info_single_max(acc_rnn_mean, acc_rnn_cv, output_path, model_n, fold_num, patient, data_mode, data_info)

                # hard voting
                # thr_list_rnn, ch_no_list_rnn, acc_list_rnn = utilities.hard_voting(pred_rnn_tot, acc_rnn_mean, step_thr, lbl_tot, model_n)

                # plot
                # utilities.plt_best_majority(thr_list_rnn, ch_no_list_rnn, acc_list_rnn, model_n, output_path, line_width, fold_num, patient, data_mode)

                thr_list_rnn, ch_no_list_rnn, acc_mean_list_rnn, acc_min_list_rnn, acc_max_list_rnn = utilities.hard_voting_2(pred_rnn_tot,
                                                                                                                              acc_rnn_mean,
                                                                                                                              step_thr, lbl_tot,
                                                                                                                              model_n,
                                                                                                                              output_path,
                                                                                                                              fold_num,
                                                                                                                              patient,
                                                                                                                              data_mode,
                                                                                                                              data_info,
                                                                                                                              odd_mode=True)
                # plot
                utilities.plt_signle_ch(acc_rnn_mean, data_l, model_n, output_path, line_width, fold_num, patient, data_mode, data_info)
                utilities.plt_best_majority_2(thr_list_rnn, ch_no_list_rnn, acc_mean_list_rnn, acc_min_list_rnn, acc_max_list_rnn, model_n,
                                              output_path, line_width, fold_num, patient, data_mode, data_info)


                k = 1

