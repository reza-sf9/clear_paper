import numpy as np
import matplotlib.pyplot as plt

def print_result_mean_std(mean_val_signle, std_val_signle, mean_val_voting, std_val_voting, num_ch, model_name):

    model_name = model_name.upper()
    cnt = len(model_name)
    if cnt == 2:
        model_name += '  '
    elif cnt == 3:
        model_name += ' '

    print('   ' + model_name + '   ', end='')
    print('          ', f'{mean_val_signle:.2f}', end='')
    print('  ', f'{std_val_signle:.2f}', end='')
    print('               ', f'{mean_val_voting:.2f}', end='')
    print('  ', f'{std_val_voting:.2f}', end='')
    print('             ', f'{num_ch:.0f}')


    pass

key_list = ['BTsC (likelihood)', 'BTsC (voting)', 'SVM', 'Gaussian Process', 'Naive Bayes', 'Random Forest', 'KNN', 'MLP',
            'Logistic Regressoin', 'LSTM']

data_mode = 'combination' # high_gamma, lp_ds, combination
DATA_type = ['FLICKER', 'SQ'] # FLICKER, SQ
FLICKER_type = ['color', 'shape']
data_type = 'FLICKER' # FLICKER, SQ
flicker_type = 'color'  # color, shape
model_list_name = ['SVM', 'GP', 'NB', 'RF', 'KN', 'NN', 'LR', 'lstm']
load_path_acc_voting = './Results/acc_result/acc_cv_max_'
load_path_acc_signle = './Results/single_acc_result/single_acc_cv_max_'
load_path_chosen_ch = './Results/chosen_ch/chosen_ch_'

for data_type in DATA_type:

    for flicker_type in FLICKER_type:

        if data_type == 'SQ':
            if flicker_type == 'shape':
                break

        if data_type == 'FLICKER':
            data_path = './data/FLICKER/'
            list_num = [9, 10, 11, 12, 13, 14]
            data_info = data_type + '_' + flicker_type

        elif data_type == 'SQ':
            data_path = './data/SQ/'
            list_num = [1, 2, 3, 4, 5, 6, 7, 8, 13]
            data_info = data_type

        print('\n=============================        ' + data_info  + '      =============================')

        for p_id in list_num:

            if p_id<10:
                str_p_id = 'p0' + str(p_id)
            else:
                str_p_id = 'p' + str(p_id)

            print('\n-----------------------------           ' + str_p_id +'              ---------------------------')
            print('             SINGLE  (mean  - std) ||   VOTING  (mean - std)  ||         #CH ')

            for model_name in model_list_name:

                # voting Results
                str_load_acc_voting = load_path_acc_voting + model_name + '_cv5_' + str_p_id + '_' + data_mode + '_' + data_info + '.npy'
                acc_cv_voting = np.load(str_load_acc_voting)
                std_val_voting = np.std(acc_cv_voting)
                std_val_voting = float("{0:.2f}".format(std_val_voting))
                mean_val_voting = np.mean(acc_cv_voting)
                mean_val_voting = float("{0:.2f}".format(mean_val_voting))

                # Single Results
                str_load_acc_signle = load_path_acc_signle + model_name + '_cv5_' + str_p_id + '_' + data_mode + '_' + data_info + '.npy'
                acc_cv_signle = np.load(str_load_acc_signle)
                std_val_signle = np.std(acc_cv_signle)
                std_val_signle = float("{0:.2f}".format(std_val_signle))
                mean_val_signle = np.mean(acc_cv_signle)
                mean_val_signle = float("{0:.2f}".format(mean_val_signle))

                # Number of Chosen Channel
                str_load_chosen_ch = load_path_chosen_ch + model_name + '_cv5_' + str_p_id + '_' + data_mode + '_' + data_info + '.npy'
                chosen_ch = np.load(str_load_chosen_ch)
                num_ch = chosen_ch.shape[0]

                print_result_mean_std(mean_val_signle, std_val_signle, mean_val_voting, std_val_voting, num_ch, model_name)

                k=1






# plt.errorbar(ch_des_list, acc_mean_list, yerr=assym_error,  fmt='-s', ecolor='red', solid_capstyle='projecting', capsize=5)