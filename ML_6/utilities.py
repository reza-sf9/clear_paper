"""
@author: Reza_SF
"""
import numpy as np
import matplotlib.pyplot as plt

seed_num = 123

def hard_voting_2(pred_mat, acc_mean, step_thr, lbl_tot, model_n, output_path, cross_valid, patient, data_mode, data_info, odd_mode):
    # string print

    b_clf_val = -1
    if model_n == 'SVM':
        str_model_print = 'SUPPORT VECTOR MACHINE'
    elif model_n == 'SVM_l1':
        str_model_print = 'Linear SVM - l1'
    elif model_n == 'GP':
        str_model_print = 'GASUSSIAN PROCESS'
    elif model_n == 'NB':
        str_model_print = 'Naive Bayesian'
    elif model_n == 'RF':
        str_model_print = 'RANDOM FOREST'
    elif model_n == 'KN':
        str_model_print = 'K-Nearest Neighbor'
    elif model_n == 'NN':
        str_model_print = 'Multi-Layer Perceptorn'
    elif model_n == 'LR':
        str_model_print = 'Logistic Regression'
    elif model_n == 'lstm':
        str_model_print = 'LSTM'
        b_clf_val = 0

    thr_acc_list, ch_des_list, acc_mean_des_list, acc_min_des_list, acc_max_des_list, ind_used_list, acc_cv_list = [], [], [], [], [], [], []
    cnt = 0
    for thr_acc in np.arange(.51, np.max(acc_mean), step_thr):
        mp = pred_mat.shape # ind_1 = ch, ind_2: num square
        num_ch = mp[0]
        num_square = mp[1]

        # voting for good score
        ind_larger_thr = np.array(np.where(np.reshape(acc_mean, (num_ch,)) >= thr_acc))

        calc_vote = 1
        if odd_mode:
            calc_vote = 0
            if np.mod(ind_larger_thr.shape[1], 2) == 1:
                calc_vote = 1

        if calc_vote == 1:
                # voting for total data set
                lbl_vote_des = np.zeros((num_square, 1))
                vec_error_des = np.zeros((num_square, 1))

                cnt += 1
                ind_used_list.append(ind_larger_thr)

                for sq in range(num_square):
                    pred_sq = pred_mat[:, sq]

                    #
                    pred_sq_des = pred_sq[ind_larger_thr]
                    num_b_des = np.array(np.where(pred_sq_des == b_clf_val)).shape[1]
                    num_w_des = np.array(np.where(pred_sq_des == 1)).shape[1]

                    if num_b_des > num_w_des:
                        lbl_vote_des[sq, 0] = b_clf_val
                    else:
                        lbl_vote_des[sq, 0] = 1

                    # finding number of error
                    if lbl_vote_des[sq, 0] != lbl_tot[sq]:
                        vec_error_des[sq, 0] = 1


                step_size_cv = num_square/5
                list_cv = np.arange(0, num_square+5, step_size_cv)
                vec_error_cv_1 = vec_error_des[int(list_cv[0]): int(list_cv[1]), 0]
                error_cv_1 = np.sum(vec_error_cv_1) / step_size_cv
                accuracy_cv_1 = 1 - error_cv_1
                vec_error_cv_2 = vec_error_des[int(list_cv[1]): int(list_cv[2]), 0]
                error_cv_2 = np.sum(vec_error_cv_2) / step_size_cv
                accuracy_cv_2 = 1 - error_cv_2
                vec_error_cv_3 = vec_error_des[int(list_cv[2]): int(list_cv[3]), 0]
                error_cv_3 = np.sum(vec_error_cv_3) / step_size_cv
                accuracy_cv_3 = 1 - error_cv_3
                vec_error_cv_4 = vec_error_des[int(list_cv[3]): int(list_cv[4]), 0]
                error_cv_4 = np.sum(vec_error_cv_4) / step_size_cv
                accuracy_cv_4 = 1 - error_cv_4
                vec_error_cv_5 = vec_error_des[int(list_cv[4]): int(list_cv[5]), 0]
                error_cv_5 = np.sum(vec_error_cv_5) / step_size_cv
                accuracy_cv_5 = 1 - error_cv_5

                acc_cv = [accuracy_cv_1, accuracy_cv_2, accuracy_cv_3, accuracy_cv_4, accuracy_cv_5]
                acc_cv_list.append(acc_cv)
                min_acc_cv = np.min(acc_cv)
                mean_acc_cv = np.mean(acc_cv)
                max_acc_cv = np.max(acc_cv)

                # error_des = np.sum(vec_error_des) / num_square
                # accuracy_des = 1 - error_des
                accuracy_des = mean_acc_cv

                thr_acc_list.append(thr_acc)
                ch_des_list.append(ind_larger_thr.shape[1])
                acc_mean_des_list.append(accuracy_des)
                acc_min_des_list.append(min_acc_cv)
                acc_max_des_list.append(max_acc_cv)


    max_ind = np.array(np.where(acc_mean_des_list == np.max(acc_mean_des_list)))
    max_ind = np.reshape(max_ind, (max_ind.shape[1], ))
    max_ind_val = max_ind[-1]
    chosen_ch = np.array(ind_used_list[max_ind_val])
    chosen_ch = np.reshape(chosen_ch, (chosen_ch.shape[1], ))
    acc_cv_max = np.array(acc_cv_list[max_ind_val])
    str_save_chosen_ch = output_path + 'chosen_ch_' + model_n + '_cv' + str(cross_valid) + '_' + patient + '_' +  data_mode + '_' + data_info +'.npy'
    str_save_acc_cv = output_path + 'acc_cv_max_' + model_n + '_cv' + str(cross_valid) + '_' + patient + '_' +  data_mode + '_' + data_info \
                     +'.npy'

    np.save(str_save_chosen_ch, chosen_ch)
    np.save(str_save_acc_cv, acc_cv_max)

    # remove repeating results and printing
    ch_no_rep = [ch_des_list[0]]
    thr_acc_no_rep = [thr_acc_list[0]]
    acc_mean_des_no_rep = [acc_mean_des_list[0]]
    acc_min_des_no_rep = [acc_min_des_list[0]]
    acc_max_des_no_rep = [acc_max_des_list[0]]

    # voting
    print('=======================================')
    print('         ' + str_model_print + '       ')
    print('---------------------------------------')
    print('| % Threshold | Num Channel | Accuracy ')
    print_results(thr_acc_no_rep[-1], ch_no_rep[-1], acc_mean_des_no_rep[-1])

    for i in range(1, len(ch_des_list)):
        prev_ch = ch_des_list[i-1]
        curr_ch = ch_des_list[i]
        if curr_ch != prev_ch:
            ch_no_rep.append(curr_ch)
            thr_acc_no_rep.append(thr_acc_list[i])
            acc_mean_des_no_rep.append(acc_mean_des_list[i])
            acc_min_des_no_rep.append(acc_min_des_list[i])
            acc_max_des_no_rep.append(acc_max_des_list[i])

            print_results(thr_acc_no_rep[-1], ch_no_rep[-1], acc_mean_des_no_rep[-1])

    ch_no_list = ch_no_rep
    thr_list = thr_acc_no_rep
    acc_mean_list = acc_mean_des_no_rep
    acc_min_list = acc_min_des_no_rep
    acc_max_list = acc_max_des_no_rep

    return thr_list, ch_no_list, acc_mean_list, acc_min_list, acc_max_list


def hard_voting(pred_mat, acc_mean, step_thr, lbl_tot, model_n):
    # string print

    b_clf_val = -1
    if model_n == 'SVM':
        str_model_print = 'SUPPORT VECTOR MACHINE'
    elif model_n == 'SVM_l1':
        str_model_print = 'Linear SVM - l1'
    elif model_n == 'GP':
        str_model_print = 'GASUSSIAN PROCESS'
    elif model_n == 'NB':
        str_model_print = 'Naive Bayesian'
    elif model_n == 'RF':
        str_model_print = 'RANDOM FOREST'
    elif model_n == 'KN':
        str_model_print = 'K-Nearest Neighbor'
    elif model_n == 'NN':
        str_model_print = 'Multi-Layer Perceptorn'
    elif model_n == 'lstm':
        str_model_print = 'LSTM'
        b_clf_val = 0

    thr_acc_list, ch_des_list, acc_des_list = [], [], [],
    for thr_acc in np.arange(.51, np.max(acc_mean), step_thr):
        mp = pred_mat.shape # ind_1 = ch, ind_2: num square
        num_ch = mp[0]
        num_square = mp[1]

        # voting for good score
        ind_larger_thr = np.array(np.where(np.reshape(acc_mean, (num_ch,)) >= thr_acc))

        # voting for total data set
        lbl_vote_des = np.zeros((num_square, 1))
        vec_error_des = np.zeros((num_square, 1))

        for sq in range(num_square):
            pred_sq = pred_mat[:, sq]

            #
            pred_sq_des = pred_sq[ind_larger_thr]
            num_b_des = np.array(np.where(pred_sq_des == b_clf_val)).shape[1]
            num_w_des = np.array(np.where(pred_sq_des == 1)).shape[1]

            if num_b_des > num_w_des:
                lbl_vote_des[sq, 0] = b_clf_val
            else:
                lbl_vote_des[sq, 0] = 1

            # finding number of error
            if lbl_vote_des[sq, 0] != lbl_tot[sq]:
                vec_error_des[sq, 0] = 1



        error_des = np.sum(vec_error_des) / num_square
        accuracy_des = 1 - error_des

        thr_acc_list.append(thr_acc)
        ch_des_list.append(ind_larger_thr.shape[1])
        acc_des_list.append(accuracy_des)

    # remove repeating results and printing
    ch_no_rep = [ch_des_list[0]]
    thr_acc_no_rep = [thr_acc_list[0]]
    acc_des_no_rep = [acc_des_list[0]]
    # voting
    print('=======================================')
    print('         ' + str_model_print + '       ')
    print('---------------------------------------')
    print('| % Threshold | Num Channel | Accuracy ')
    print_results(thr_acc_no_rep[-1], ch_no_rep[-1], acc_des_no_rep[-1])

    for i in range(1, len(ch_des_list)):
        prev_ch = ch_des_list[i-1]
        curr_ch = ch_des_list[i]
        if curr_ch != prev_ch:
            ch_no_rep.append(curr_ch)
            thr_acc_no_rep.append(thr_acc_list[i])
            acc_des_no_rep.append(acc_des_list[i])

            print_results(thr_acc_no_rep[-1], ch_no_rep[-1], acc_des_no_rep[-1])

    ch_no_list = ch_no_rep
    thr_list = thr_acc_no_rep
    acc_list = acc_des_no_rep

    return thr_list, ch_no_list, acc_list

def split_train_test(black_ch, white_ch, folds, f, feature_num):

    # seperate test and train data
    black_ch_train = black_ch[folds != f]
    black_ch_test = black_ch[folds == f]

    white_ch_train = white_ch[folds != f]
    white_ch_test = white_ch[folds == f]

    # mix black and white data set
    lbl_train = np.array([-1] * len(black_ch_train) + [1] * len(white_ch_train))
    lbl_test = np.array([-1] * len(black_ch_test) + [1] * len(white_ch_test))
    feat_train = np.zeros((len(lbl_train), feature_num))
    feat_test = np.zeros((len(lbl_test), feature_num))

    np.random.seed(seed_num)
    np.random.shuffle(lbl_train)
    np.random.shuffle(lbl_test)
    ind_b_train = np.where(lbl_train == -1)
    ind_w_train = np.where(lbl_train == 1)
    ind_b_test = np.where(lbl_test == -1)
    ind_w_test = np.where(lbl_test == 1)

    feat_train[ind_b_train, :] = black_ch_train
    feat_train[ind_w_train, :] = white_ch_train
    feat_test[ind_b_test, :] = black_ch_test
    feat_test[ind_w_test, :] = white_ch_test

    return feat_train, lbl_train, feat_test, lbl_test

def plt_signle_ch(acc_mean, data_l, model_n, output_path, line_width, cross_valid, patient, data_mode, data_info):
    num_correct = np.array(np.where(acc_mean > .5)).shape[1]
    ch_num = acc_mean.shape[0]

    # BEST ACURACCY
    val_max = np.max(acc_mean)
    ind_max = np.max(np.where(acc_mean == val_max))
    best_acc = "{:.2f}".format(val_max * 100)

    # string title
    if model_n == 'SVM':
        str_model_tit = 'SUPPORT VECTOR MACHINE'
    if model_n == 'SVM_l1':
        str_model_tit = 'Linear SVM - l1'
    elif model_n=='GP':
        str_model_tit = 'GASUSSIAN PROCESS'
    elif model_n == 'NB':
        str_model_tit = 'Naive Bayesian'
    elif model_n == 'RF':
        str_model_tit = 'RANDOM FOREST'
    elif model_n == 'KN':
        str_model_tit = 'K-Nearest Neighbor'
    elif model_n == 'NN':
        str_model_tit = 'Multi-Layer Perceptorn'
    elif model_n == 'LR':
        str_model_tit = 'Logistic Regression'
    elif model_n == 'lstm':
        str_model_tit = 'LSTM'


    # plot
    plt.figure()
    x_ind = np.reshape(np.arange(ch_num), (ch_num, 1))
    plt.plot(x_ind, acc_mean, linewidth=line_width)
    # plt.grid()
    plt.plot(ind_max, val_max, marker="o",
             markerfacecolor='green', markeredgecolor='green', markersize=15)
    half_line = np.array([.5] * ch_num)
    half_line = np.reshape(half_line, (ch_num, 1))
    plt.plot(x_ind, half_line, color='red')
    plt.xlabel('Channel')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xlim(0, ch_num)
    if data_mode == 'combination':
        x_v = np.floor(x_ind.shape[0]/2)
        plt.vlines(x_v, 0, 1, colors='green', linestyles='dashed')


    plt.title(str_model_tit + '---' + data_info + ' --- ' + patient + ' -- ' + data_mode + '\nbest ch: '+ str(ind_max) + " | best acc: "+ str(best_acc)
              + ' | #correct pred: '+ str(num_correct)+'/'+str(ch_num) + ' | CV: ' +  str(cross_valid))
    plt.savefig(output_path + 'sing_ch_' + model_n + '_cv' + str(cross_valid) + '_' + patient + '_' +  data_mode + '_' + data_info + '.svg')
    plt.savefig(output_path + 'sing_ch_' + model_n + '_cv' + str(cross_valid) + '_' + patient + '_' + data_mode + '_' + data_info + '.png')
    plt.show()
    pass

def plt_best_majority(thr_acc_list, ch_des_list, acc_des_list, model_n, output_path, line_width, cross_valid, patient, data_mode):
    # string title
    if model_n == 'SVM':
        str_model_tit = 'SUPPORT VECTOR MACHINE'
    elif model_n == 'SVM_l1':
        str_model_tit = 'Linear SVM - l1'
    elif model_n == 'GP':
        str_model_tit = 'GASUSSIAN PROCESS'
    elif model_n == 'NB':
        str_model_tit = 'Naive Bayesian'
    elif model_n == 'RF':
        str_model_tit = 'RANDOM FOREST'
    elif model_n == 'KN':
        str_model_tit = 'K-Nearest Neighbor'
    elif model_n == 'NN':
        str_model_tit = 'Multi-Layer Perceptorn'
    elif model_n == 'lstm':
        str_model_tit = 'LSTM'




    max_acc = np.max(np.array(acc_des_list))

    list_max_bool = list(acc_des_list == max_acc)
    ind_max = list_max_bool.index(True)
    best_ch = ch_des_list[ind_max]
    best_acc = "{:.2f}".format(max_acc * 100)

    plt.figure()
    plt.plot(ch_des_list, acc_des_list)
    plt.plot(best_ch, max_acc, marker="o",
                          markerfacecolor='green', markeredgecolor='green', markersize=15)

    plt.xticks(ch_des_list, ch_des_list)
    plt.xlabel('Number of Channel')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.title(str_model_tit + ' --- ' + patient + ' -- ' + data_mode + '\n Best Num of Ch: ' + str(best_ch) + "  |  Best Acc: " + str(best_acc) + ' | CV = ' + str(
        cross_valid))
    plt.savefig(output_path + 'voting_' + model_n + '_cv' + str(cross_valid) + '_' + patient + '_' +  data_mode + '.svg')
    plt.show()

    pass


def plt_best_majority_2(thr_acc_list, ch_des_list, acc_mean_list, acc_min_list, acc_max_list, model_n, output_path, line_width, cross_valid,
                        patient,
                        data_mode, data_info):
    # string title
    if model_n == 'SVM':
        str_model_tit = 'SUPPORT VECTOR MACHINE'
    elif model_n == 'SVM_l1':
        str_model_tit = 'Linear SVM - l1'
    elif model_n == 'GP':
        str_model_tit = 'GASUSSIAN PROCESS'
    elif model_n == 'NB':
        str_model_tit = 'Naive Bayesian'
    elif model_n == 'RF':
        str_model_tit = 'RANDOM FOREST'
    elif model_n == 'KN':
        str_model_tit = 'K-Nearest Neighbor'
    elif model_n == 'NN':
        str_model_tit = 'Multi-Layer Perceptorn'
    elif model_n == 'LR':
        str_model_tit = 'Logistic Regression'
    elif model_n == 'lstm':
        str_model_tit = 'LSTM'

    up_error = np.array(acc_max_list) - np.array(acc_mean_list)
    lower_erro = np.array(acc_mean_list) - np.array(acc_min_list)
    assym_error = np.array([lower_erro, up_error])


    max_acc = np.max(np.array(acc_mean_list))

    list_max_bool = list(acc_mean_list == max_acc)
    ind_max = list_max_bool.index(True)
    best_ch = ch_des_list[ind_max]
    best_acc = "{:.2f}".format(max_acc * 100)

    plt.figure()
    plt.errorbar(ch_des_list, acc_mean_list, yerr=assym_error,  fmt='-s', ecolor='red', solid_capstyle='projecting', capsize=5)
    plt.plot(best_ch, max_acc, marker="o",
                          markerfacecolor='green', markeredgecolor='green', markersize=15)

    plt.xticks(ch_des_list, ch_des_list)
    plt.xlabel('Number of Channel')
    plt.ylabel('Accuracy')
    plt.ylim([np.min(acc_min_list)-.1, 1])
    # plt.grid()
    plt.title(str_model_tit + '---' + data_info + ' --- ' + patient + ' -- ' + data_mode + '\n Best Num of Ch: ' + str(best_ch) + "  |  Best " \
                                                                                                                                 "Acc: " + str(
        best_acc) + ' | CV = ' + str(
        cross_valid))
    plt.savefig(output_path + 'error_voting_' + model_n + '_cv' + str(cross_valid) + '_' + patient + '_' +  data_mode + '_' + data_info +'.svg')
    plt.savefig(output_path + 'error_voting_' + model_n + '_cv' + str(cross_valid) + '_' + patient + '_' + data_mode + '_' + data_info + '.png')
    plt.show()

    pass


def print_results(thr_acc, ch_num, accuracy):
    '''printing scores in console'''

    print('       ', f'{thr_acc:.3f}', end='')
    print('       ', f'{ch_num:.0f}', end='')
    print('    ', f'{accuracy:.2f}')

def save_info_single_max(acc_METHOD_mean, acc_METHOD_cv, output_path, model_n, cross_valid, patient, data_mode, data_info):
    acc_METHOD_mean = np.reshape(acc_METHOD_mean, (acc_METHOD_mean.shape[0],))
    max_ind_mean = np.array(np.where(acc_METHOD_mean == np.max(acc_METHOD_mean)))
    max_ind_mean = np.reshape(max_ind_mean, (max_ind_mean.shape[1],))
    max_ind_val = max_ind_mean[-1]
    acc_cv_max = np.array(acc_METHOD_cv[max_ind_val, :])
    str_save_single_acc_cv = output_path + 'single_acc_cv_max_' + model_n + '_cv' + str(
        cross_valid) + '_' + patient + '_' + data_mode + '_' + data_info \
                             + '.npy'

    np.save(str_save_single_acc_cv, acc_cv_max)