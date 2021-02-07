import numpy as np
import matplotlib.pyplot as plt

mean_plt = False


# aa = np.load('y_p09_mean.npy')
k=1
def plt_acc(x, y, title, data_info, output_path):

    plt.figure(figsize=(8, 6))

    markerline, stemlines, baseline = plt.stem(x, y, use_line_collection = True)
    plt.setp(stemlines, 'linewidth', 7)
    plt.setp(markerline, markersize=15)
    plt.ylim([.5, 1.05])
    # plt.grid()
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.xticks(x, key_list, rotation='vertical')
    plt.title(data_info + '  ------  ' + title)
    plt.rcParams.update({'font.size': 12})
    plt.savefig(output_path + 'acc_1sec_' + title + '_' + data_info + '.png')
    plt.savefig(output_path + 'acc_1sec_' + title + '_' + data_info + '.svg')
    plt.show()
    pass

def plt_acc_with_std(x, acc_mean, std_error, title, data_info, output_path):


    sym_error = np.array([std_error, std_error])

    plt.figure(figsize=(8, 6))
    plt.errorbar(x, acc_mean, yerr=sym_error, fmt='o', ecolor='red', solid_capstyle='projecting', capsize=10, ms=10, mew=1)
    # plt.setp(stemlines, 'linewidth', 7)
    # plt.setp(markerline, markersize=15)
    plt.ylim([.2, 1.05])
    # # plt.grid()
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.xticks(x, key_list, rotation='vertical')
    plt.title(data_info + '  ------  ' + title)
    plt.rcParams.update({'font.size': 12})
    plt.savefig(output_path + 'acc_STD_1sec_' + title + '_' + data_info + '.png')
    plt.savefig(output_path + 'acc_STD_1sec_' + title + '_' + data_info + '.svg')
    plt.show()

def print_result_mean_std(key_list, y_mean_TOT, y_std_TOT, data_info):
    print('\n============================================================   ' + data_info + '  '
                                                                                          '================================================================\n')
    print( '             ' + key_list[0] + ' |' + '| '+ key_list[1] + ' |' +'| '+ key_list[2] + ' |' +'| '+ key_list[3] + ' |' +
          '| '+ key_list[4] + ' |' +'| '+ key_list[5] + ' |' +'| '+ key_list[6] + ' |' +'| '+ key_list[7] + ' |' +
          '| '+ key_list[8] + ' |' +'| '+ key_list[9] + ' |')

    print('   MEAN:     ', end='')
    print('        ', f'{y_mean_TOT[0]:.2f}', end='')
    print('           ', f'{y_mean_TOT[1]:.2f}', end='')
    print('        ', f'{y_mean_TOT[2]:.2f}', end='')
    print('         ', f'{y_mean_TOT[3]:.2f}', end='')
    print('           ', f'{y_mean_TOT[4]:.2f}', end='')
    print('           ', f'{y_mean_TOT[5]:.2f}', end='')
    print('        ', f'{y_mean_TOT[6]:.2f}', end='')
    print('  ', f'{y_mean_TOT[7]:.2f}', end='')
    print('          ', f'{y_mean_TOT[8]:.2f}', end='')
    print('          ', f'{y_mean_TOT[9]:.2f}')

    print('   STD:      ', end='')
    print('        ', f'{y_std_TOT[0]:.2f}', end='')
    print('           ', f'{y_std_TOT[1]:.2f}', end='')
    print('        ', f'{y_std_TOT[2]:.2f}', end='')
    print('         ', f'{y_std_TOT[3]:.2f}', end='')
    print('           ', f'{y_std_TOT[4]:.2f}', end='')
    print('           ', f'{y_std_TOT[5]:.2f}', end='')
    print('        ', f'{y_std_TOT[6]:.2f}', end='')
    print('  ', f'{y_std_TOT[7]:.2f}', end='')
    print('          ', f'{y_std_TOT[8]:.2f}', end='')
    print('          ', f'{y_std_TOT[9]:.2f}')

    pass


key_list = ['BTsC (likelihood)', 'BTsC (voting)', 'SVM', 'Gaussian Process', 'Naive Bayes', 'Random Forest', 'KNN', 'MLP',
            'Logistic Regressoin', 'LSTM']
output_path = './Results/plot_paper/'

data_type = 'SQ' # FLICKER, SQ
flicker_type = 'color'  # color, shape

if data_type == 'FLICKER':
    data_info = data_type + '_' + flicker_type

    if flicker_type == 'color':
        dict_voting_p09_mean = {key_list[0]: 0.97,
                    key_list[1]: 0.99,
                    key_list[2]: 0.92,
                    key_list[3]: 0.88,
                    key_list[4]: 0.89,
                    key_list[5]: 0.87,
                    key_list[6]: 0.89,
                    key_list[7]: 0.85,
                    key_list[8]: 0.90,
                    key_list[9]: 0.88,
                    }
        dict_voting_p09_std = {key_list[0]: 0.04,
                         key_list[1]: 0.02,
                         key_list[2]: 0.04,
                         key_list[3]: 0.11,
                         key_list[4]: 0.06,
                         key_list[5]: 0.07,
                         key_list[6]: 0.05,
                         key_list[7]: 0.11,
                         key_list[8]: 0.06,
                         key_list[9]: 0.24,
                         }

        dict_voting_p10_mean = {key_list[0]: 0.88,
                    key_list[1]: 0.91,
                    key_list[2]: 0.81,
                    key_list[3]: 0.77,
                    key_list[4]: 0.80,
                    key_list[5]: 0.78,
                    key_list[6]: 0.75,
                    key_list[7]: 0.71,
                    key_list[8]: 0.82,
                    key_list[9]: 0.68,
                    }
        dict_voting_p10_std = {key_list[0]: 0.07,
                    key_list[1]: 0.06,
                    key_list[2]: 0.11,
                    key_list[3]: 0.04,
                    key_list[4]: 0.11,
                    key_list[5]: 0.08,
                    key_list[6]: 0.14,
                    key_list[7]: 0.10,
                    key_list[8]: 0.13,
                    key_list[9]: 0.16,
                    }

        dict_voting_p11_mean = {key_list[0]: 0.88,
                    key_list[1]: 0.91,
                    key_list[2]: 0.78,
                    key_list[3]: 0.74,
                    key_list[4]: 0.78,
                    key_list[5]: 0.82,
                    key_list[6]: 0.86,
                    key_list[7]: 0.72,
                    key_list[8]: 0.77,
                    key_list[9]: 0.69,
                    }
        dict_voting_p11_std = {key_list[0]: 0.07,
                         key_list[1]: 0.03,
                         key_list[2]: 0.07,
                         key_list[3]: 0.12,
                         key_list[4]: 0.14,
                         key_list[5]: 0.07,
                         key_list[6]: 0.07,
                         key_list[7]: 0.09,
                         key_list[8]: 0.11,
                         key_list[9]: 0.22,
                         }

        dict_voting_p12_mean = {key_list[0]: 0.92,
                    key_list[1]: 0.99,
                    key_list[2]: 0.79,
                    key_list[3]: 0.84,
                    key_list[4]: 0.82,
                    key_list[5]: 0.86,
                    key_list[6]: 0.82,
                    key_list[7]: 0.71,
                    key_list[8]: 0.81,
                    key_list[9]: 0.66,
                    }
        dict_voting_p12_std = {key_list[0]: 0.06,
                         key_list[1]: 0.02,
                         key_list[2]: 0.15,
                         key_list[3]: 0.10,
                         key_list[4]: 0.05,
                         key_list[5]: 0.11,
                         key_list[6]: 0.05,
                         key_list[7]: 0.10,
                         key_list[8]: 0.04,
                         key_list[9]: 0.14,
                         }

        dict_voting_p13_mean = {key_list[0]: 0.88,
                    key_list[1]: 0.88,
                    key_list[2]: 0.74,
                    key_list[3]: 0.65,
                    key_list[4]: 0.73,
                    key_list[5]: 0.68,
                    key_list[6]: 0.67,
                    key_list[7]: 0.67,
                    key_list[8]: 0.74,
                    key_list[9]: 0.62,
                    }
        dict_voting_p13_std = {key_list[0]: 0.07,
                         key_list[1]: 0.09,
                         key_list[2]: 0.19,
                         key_list[3]: 0.33,
                         key_list[4]: 0.17,
                         key_list[5]: 0.26,
                         key_list[6]: 0.31,
                         key_list[7]: 0.11,
                         key_list[8]: 0.11,
                         key_list[9]: 0.08,
                         }

        dict_voting_p14_mean = {key_list[0]: 0.98,
                    key_list[1]: 0.98,
                    key_list[2]: 0.85,
                    key_list[3]: 0.79,
                    key_list[4]: 0.88,
                    key_list[5]: 0.82,
                    key_list[6]: 0.88,
                    key_list[7]: 0.73,
                    key_list[8]: 0.91,
                    key_list[9]: 0.66,
                    }
        dict_voting_p14_std = {key_list[0]: 0.03,
                    key_list[1]: 0.03,
                    key_list[2]: 0.05,
                    key_list[3]: 0.07,
                    key_list[4]: 0.11,
                    key_list[5]: 0.14,
                    key_list[6]: 0.07,
                    key_list[7]: 0.13,
                    key_list[8]: 0.07,
                    key_list[9]: 0.12,
                    }

    elif flicker_type == 'shape':
        dict_voting_p09_mean = {key_list[0]: 0.96,
                    key_list[1]: 0.92,
                    key_list[2]: 0.85,
                    key_list[3]: 0.70,
                    key_list[4]: 0.81,
                    key_list[5]: 0.75,
                    key_list[6]: 0.78,
                    key_list[7]: 0.75,
                    key_list[8]: 0.76,
                    key_list[9]: 0.83,
                    }
        dict_voting_p09_std = {key_list[0]: 0.05,
                         key_list[1]: 0.06,
                         key_list[2]: 0.15,
                         key_list[3]: 0.26,
                         key_list[4]: 0.12,
                         key_list[5]: 0.20,
                         key_list[6]: 0.15,
                         key_list[7]: 0.25,
                         key_list[8]: 0.20,
                         key_list[9]: 0.29,
                         }

        dict_voting_p10_mean = {key_list[0]: 0.82,
                         key_list[1]: 0.80,
                         key_list[2]: 0.81,
                         key_list[3]: 0.66,
                         key_list[4]: 0.77,
                         key_list[5]: 0.73,
                         key_list[6]: 0.73,
                         key_list[7]: 0.72,
                         key_list[8]: 0.81,
                         key_list[9]: 0.73,
                         }
        dict_voting_p10_std = {key_list[0]: 0.08,
                        key_list[1]: 0.07,
                        key_list[2]: 0.15,
                        key_list[3]: 0.34,
                        key_list[4]: 0.28,
                        key_list[5]: 0.25,
                        key_list[6]: 0.12,
                        key_list[7]: 0.21,
                        key_list[8]: 0.15,
                        key_list[9]: 0.18,
                        }

        dict_voting_p11_mean = {key_list[0]: 0.88,
                         key_list[1]: 0.84,
                         key_list[2]: 0.77,
                         key_list[3]: 0.64,
                         key_list[4]: 0.72,
                         key_list[5]: 0.70,
                         key_list[6]: 0.76,
                         key_list[7]: 0.77,
                         key_list[8]: 0.77,
                         key_list[9]: 0.65,
                         }
        dict_voting_p11_std = {key_list[0]: 0.07,
                        key_list[1]: 0.08,
                        key_list[2]: 0.22,
                        key_list[3]: 0.32,
                        key_list[4]: 0.31,
                        key_list[5]: 0.27,
                        key_list[6]: 0.19,
                        key_list[7]: 0.09,
                        key_list[8]: 0.22,
                        key_list[9]: 0.23,
                        }

        dict_voting_p12_mean = {key_list[0]: 0.91,
                         key_list[1]: 0.93,
                         key_list[2]: 0.87,
                         key_list[3]: 0.80,
                         key_list[4]: 0.80,
                         key_list[5]: 0.84,
                         key_list[6]: 0.76,
                         key_list[7]: 0.73,
                         key_list[8]: 0.81,
                         key_list[9]: 0.66,
                         }
        dict_voting_p12_std = {key_list[0]: 0.06,
                        key_list[1]: 0.06,
                        key_list[2]: 0.10,
                        key_list[3]: 0.16,
                        key_list[4]: 0.11,
                        key_list[5]: 0.18,
                        key_list[6]: 0.15,
                        key_list[7]: 0.16,
                        key_list[8]: 0.09,
                        key_list[9]: 0.13,
                        }

        dict_voting_p13_mean = {key_list[0]: 0.88,
                         key_list[1]: 0.89,
                         key_list[2]: 0.83,
                         key_list[3]: 0.67,
                         key_list[4]: 0.67,
                         key_list[5]: 0.73,
                         key_list[6]: 0.76,
                         key_list[7]: 0.65,
                         key_list[8]: 0.74,
                         key_list[9]: 0.53,
                         }
        dict_voting_p13_std = {key_list[0]: 0.07,
                        key_list[1]: 0.07,
                        key_list[2]: 0.09,
                        key_list[3]: 0.26,
                        key_list[4]: 0.17,
                        key_list[5]: 0.09,
                        key_list[6]: 0.13,
                        key_list[7]: 0.11,
                        key_list[8]: 0.11,
                        key_list[9]: 0.07,
                        }

        dict_voting_p14_mean = {key_list[0]: 0.88,
                         key_list[1]: 0.85,
                         key_list[2]: 0.85,
                         key_list[3]: 0.66,
                         key_list[4]: 0.80,
                         key_list[5]: 0.75,
                         key_list[6]: 0.79,
                         key_list[7]: 0.77,
                         key_list[8]: 0.80,
                         key_list[9]: 0.63,
                         }
        dict_voting_p14_std = {key_list[0]: 0.07,
                        key_list[1]: 0.10,
                        key_list[2]: 0.15,
                        key_list[3]: 0.20,
                        key_list[4]: 0.23,
                        key_list[5]: 0.23,
                        key_list[6]: 0.12,
                        key_list[7]: 0.19,
                        key_list[8]: 0.21,
                        key_list[9]: 0.63,
                        }

    x = []
    y_p09_mean = []
    y_p10_mean = []
    y_p11_mean = []
    y_p12_mean = []
    y_p13_mean = []
    y_p14_mean = []

    y_p09_std = []
    y_p10_std = []
    y_p11_std = []
    y_p12_std = []
    y_p13_std = []
    y_p14_std = []

    for i in range(len(dict_voting_p09_mean)):
        x.append(i)
        y_p09_mean.append(dict_voting_p09_mean[key_list[i]])
        y_p10_mean.append(dict_voting_p10_mean[key_list[i]])
        y_p11_mean.append(dict_voting_p11_mean[key_list[i]])
        y_p12_mean.append(dict_voting_p12_mean[key_list[i]])
        y_p13_mean.append(dict_voting_p13_mean[key_list[i]])
        y_p14_mean.append(dict_voting_p14_mean[key_list[i]])

        y_p09_std.append(dict_voting_p09_std[key_list[i]])
        y_p10_std.append(dict_voting_p10_std[key_list[i]])
        y_p11_std.append(dict_voting_p11_std[key_list[i]])
        y_p12_std.append(dict_voting_p12_std[key_list[i]])
        y_p13_std.append(dict_voting_p13_std[key_list[i]])
        y_p14_std.append(dict_voting_p14_std[key_list[i]])

    y_TOT_mean = np.array([y_p09_mean, y_p10_mean, y_p11_mean, y_p12_mean, y_p13_mean, y_p14_mean])
    y_mean_TOT = np.mean(y_TOT_mean, axis=0)
    y_std_TOT = np.std(y_TOT_mean, axis=0)

    #
    if mean_plt:
        plt_acc(x, y_p09_mean, 'p09', data_info, output_path)
        plt_acc(x, y_p10_mean, 'p10', data_info, output_path)
        plt_acc(x, y_p11_mean, 'p11', data_info, output_path)
        plt_acc(x, y_p12_mean, 'p12', data_info, output_path)
        plt_acc(x, y_p13_mean, 'p13', data_info, output_path)
        plt_acc(x, y_p14_mean, 'p14', data_info, output_path)

        plt_acc(x, y_mean_TOT, 'pTOT', data_info, output_path)


    plt_acc_with_std(x, y_p09_mean, y_p09_std, 'p09', data_info, output_path)
    plt_acc_with_std(x, y_p10_mean, y_p10_std, 'p10', data_info, output_path)
    plt_acc_with_std(x, y_p11_mean, y_p11_std, 'p11', data_info, output_path)
    plt_acc_with_std(x, y_p12_mean, y_p12_std, 'p12', data_info, output_path)
    plt_acc_with_std(x, y_p13_mean, y_p13_std, 'p13', data_info, output_path)
    plt_acc_with_std(x, y_p14_mean, y_p14_std, 'p14', data_info, output_path)

    plt_acc_with_std(x, y_mean_TOT, y_std_TOT, 'pTOT', data_info, output_path)

    print_result_mean_std(key_list, y_mean_TOT, y_std_TOT, data_info)

    K=1
elif data_type == 'SQ':
    data_info = data_type


    dict_voting_p01_mean = {key_list[0]: 0.98,
                key_list[1]: 0.97,
                key_list[2]: 0.91,
                key_list[3]: 0.88,
                key_list[4]: 0.92,
                key_list[5]: 0.90,
                key_list[6]: 0.87,
                key_list[7]: 0.82,
                key_list[8]: 0.89,
                key_list[9]: 0.93,
                }
    dict_voting_p01_std = {key_list[0]: 0.03,
                     key_list[1]: 0.04,
                     key_list[2]: 0.04,
                     key_list[3]: 0.05,
                     key_list[4]: 0.05,
                     key_list[5]: 0.04,
                     key_list[6]: 0.08,
                     key_list[7]: 0.09,
                     key_list[8]: 0.08,
                     key_list[9]: 0.12,
                     }

    dict_voting_p02_mean = {key_list[0]: 0.98,
                     key_list[1]: 0.98,
                     key_list[2]: 0.90,
                     key_list[3]: 0.93,
                     key_list[4]: 0.92,
                     key_list[5]: 0.88,
                     key_list[6]: 0.93,
                     key_list[7]: 0.84,
                     key_list[8]: 0.91,
                     key_list[9]: 0.64,
                     }
    dict_voting_p02_std = {key_list[0]: 0.04,
                     key_list[1]: 0.03,
                     key_list[2]: 0.03,
                     key_list[3]: 0.05,
                     key_list[4]: 0.08,
                     key_list[5]: 0.12,
                     key_list[6]: 0.04,
                     key_list[7]: 0.09,
                     key_list[8]: 0.08,
                     key_list[9]: 0.15,
                     }

    dict_voting_p03_mean = {key_list[0]: 0.98,
                key_list[1]: 1,
                key_list[2]: 0.87,
                key_list[3]: 0.83,
                key_list[4]: 0.86,
                key_list[5]: 0.91,
                key_list[6]: 0.90,
                key_list[7]: 0.79,
                key_list[8]: 0.91,
                key_list[9]: 0.62,
                }
    dict_voting_p03_std = {key_list[0]: 0.03,
                     key_list[1]: 0.00,
                     key_list[2]: 0.08,
                     key_list[3]: 0.11,
                     key_list[4]: 0.10,
                     key_list[5]: 0.07,
                     key_list[6]: 0.06,
                     key_list[7]: 0.10,
                     key_list[8]: 0.09,
                     key_list[9]: 0.11,
                     }

    dict_voting_p04_mean = {key_list[0]: 0.98,
                     key_list[1]: 0.98,
                     key_list[2]: 0.81,
                     key_list[3]: 0.69,
                     key_list[4]: 0.81,
                     key_list[5]: 0.78,
                     key_list[6]: 0.81,
                     key_list[7]: 0.77,
                     key_list[8]: 0.79,
                     key_list[9]: 0.58,
                     }
    dict_voting_p04_std = {key_list[0]: 0.03,
                     key_list[1]: 0.04,
                     key_list[2]: 0.18,
                     key_list[3]: 0.14,
                     key_list[4]: 0.09,
                     key_list[5]: 0.18,
                     key_list[6]: 0.16,
                     key_list[7]: 0.13,
                     key_list[8]: 0.14,
                     key_list[9]: 0.09,
                     }

    dict_voting_p05_mean = {key_list[0]: 1,
                key_list[1]: 1,
                key_list[2]: 0.94,
                key_list[3]: 0.87,
                key_list[4]: 0.90,
                key_list[5]: 0.92,
                key_list[6]: 0.95,
                key_list[7]: 0.93,
                key_list[8]: 0.95,
                key_list[9]: 0.67,
                }
    dict_voting_p05_std = {key_list[0]: 0.00,
                     key_list[1]: 0.00,
                     key_list[2]: 0.06,
                     key_list[3]: 0.11,
                     key_list[4]: 0.11,
                     key_list[5]: 0.07,
                     key_list[6]: 0.06,
                     key_list[7]: 0.09,
                     key_list[8]: 0.08,
                     key_list[9]: 0.13,
                     }

    dict_voting_p06_mean = {key_list[0]: 0.96,
                key_list[1]: 0.94,
                key_list[2]: 0.94,
                key_list[3]: 0.95,
                key_list[4]: 0.96,
                key_list[5]: 0.94,
                key_list[6]: 0.95,
                key_list[7]: 0.86,
                key_list[8]: 0.96,
                key_list[9]: 0.72,
                }
    dict_voting_p06_std = {key_list[0]: 0.05,
                     key_list[1]: 0.04,
                     key_list[2]: 0.04,
                     key_list[3]: 0.04,
                     key_list[4]: 0.04,
                     key_list[5]: 0.05,
                     key_list[6]: 0.03,
                     key_list[7]: 0.06,
                     key_list[8]: 0.02,
                     key_list[9]: 0.10,
                     }

    dict_voting_p07_mean = {key_list[0]: 0.88,
                key_list[1]: 0.84,
                key_list[2]: 0.91,
                key_list[3]: 0.75,
                key_list[4]: 0.89,
                key_list[5]: 0.78,
                key_list[6]: 0.83,
                key_list[7]: 0.81,
                key_list[8]: 0.89,
                key_list[9]: 0.62,
                }
    dict_voting_p07_std = {key_list[0]: 0.05,
                     key_list[1]: 0.06,
                     key_list[2]: 0.09,
                     key_list[3]: 0.07,
                     key_list[4]: 0.09,
                     key_list[5]: 0.15,
                     key_list[6]: 0.15,
                     key_list[7]: 0.17,
                     key_list[8]: 0.12,
                     key_list[9]: 0.26,
                     }

    dict_voting_p08_mean = {key_list[0]: 0.92,
                key_list[1]: 0.95,
                key_list[2]: 0.86,
                key_list[3]: 0.81,
                key_list[4]: 0.81,
                key_list[5]: 0.81,
                key_list[6]: 0.75,
                key_list[7]: 0.75,
                key_list[8]: 0.82,
                key_list[9]: 0.63,
                }
    dict_voting_p08_std = {key_list[0]: 0.06,
                     key_list[1]: 0.03,
                     key_list[2]: 0.10,
                     key_list[3]: 0.14,
                     key_list[4]: 0.11,
                     key_list[5]: 0.10,
                     key_list[6]: 0.17,
                     key_list[7]: 0.19,
                     key_list[8]: 0.21,
                     key_list[9]: 0.25,
                     }

    dict_voting_p13_mean = {key_list[0]: 0.82,
                     key_list[1]: 0.86,
                     key_list[2]: 0.73,
                     key_list[3]: 0.69,
                     key_list[4]: 0.73,
                     key_list[5]: 0.72,
                     key_list[6]: 0.76,
                     key_list[7]: 0.71,
                     key_list[8]: 0.74,
                     key_list[9]: 0.56,
                     }
    dict_voting_p13_std = {key_list[0]: 0.08,
                     key_list[1]: 0.09,
                     key_list[2]: 0.18,
                     key_list[3]: 0.30,
                     key_list[4]: 0.08,
                     key_list[5]: 0.34,
                     key_list[6]: 0.21,
                     key_list[7]: 0.09,
                     key_list[8]: 0.22,
                     key_list[9]: 0.31,
                     }



    x = []
    y_p01_mean = []
    y_p02_mean = []
    y_p03_mean = []
    y_p04_mean = []
    y_p05_mean = []
    y_p06_mean = []
    y_p07_mean = []
    y_p08_mean = []
    y_p13_mean = []

    y_p01_std = []
    y_p02_std = []
    y_p03_std = []
    y_p04_std = []
    y_p05_std = []
    y_p06_std = []
    y_p07_std = []
    y_p08_std = []
    y_p13_std = []

    for i in range(len(dict_voting_p01_mean)):
        x.append(i)
        y_p01_mean.append(dict_voting_p01_mean[key_list[i]])
        y_p02_mean.append(dict_voting_p02_mean[key_list[i]])
        y_p03_mean.append(dict_voting_p03_mean[key_list[i]])
        y_p04_mean.append(dict_voting_p04_mean[key_list[i]])
        y_p05_mean.append(dict_voting_p05_mean[key_list[i]])
        y_p06_mean.append(dict_voting_p06_mean[key_list[i]])
        y_p07_mean.append(dict_voting_p07_mean[key_list[i]])
        y_p08_mean.append(dict_voting_p08_mean[key_list[i]])
        y_p13_mean.append(dict_voting_p13_mean[key_list[i]])

        y_p01_std.append(dict_voting_p01_std[key_list[i]])
        y_p02_std.append(dict_voting_p02_std[key_list[i]])
        y_p03_std.append(dict_voting_p03_std[key_list[i]])
        y_p04_std.append(dict_voting_p04_std[key_list[i]])
        y_p05_std.append(dict_voting_p05_std[key_list[i]])
        y_p06_std.append(dict_voting_p06_std[key_list[i]])
        y_p07_std.append(dict_voting_p07_std[key_list[i]])
        y_p08_std.append(dict_voting_p08_std[key_list[i]])
        y_p13_std.append(dict_voting_p13_std[key_list[i]])

    y_TOT_mean = np.array([y_p01_mean, y_p02_mean, y_p03_mean, y_p04_mean, y_p05_mean, y_p06_mean, y_p07_mean, y_p08_mean, y_p13_mean])
    y_mean_TOT = np.mean(y_TOT_mean, axis=0)
    y_std_TOT = np.std(y_TOT_mean, axis=0)

    #

    if mean_plt:
        plt_acc(x, y_p01_mean, 'p01', data_info, output_path)
        plt_acc(x, y_p02_mean, 'p02', data_info, output_path)
        plt_acc(x, y_p03_mean, 'p03', data_info, output_path)
        plt_acc(x, y_p04_mean, 'p04', data_info, output_path)
        plt_acc(x, y_p05_mean, 'p05', data_info, output_path)
        plt_acc(x, y_p06_mean, 'p06', data_info, output_path)
        plt_acc(x, y_p07_mean, 'p07', data_info, output_path)
        plt_acc(x, y_p08_mean, 'p08', data_info, output_path)
        plt_acc(x, y_p13_mean, 'p13', data_info, output_path)

        plt_acc(x, y_mean_TOT, 'pTOT', data_info, output_path)


    # np.save('order_model.npy', key_list)
    #
    # np.save('y_p01_mean.npy', y_p01_mean)
    # np.save('y_p01_std.npy', y_p01_std)


    plt_acc_with_std(x, y_p01_mean, y_p01_std, 'p01', data_info, output_path)
    plt_acc_with_std(x, y_p02_mean, y_p02_std, 'p02', data_info, output_path)
    plt_acc_with_std(x, y_p03_mean, y_p03_std, 'p03', data_info, output_path)
    plt_acc_with_std(x, y_p04_mean, y_p04_std, 'p04', data_info, output_path)
    plt_acc_with_std(x, y_p05_mean, y_p05_std, 'p05', data_info, output_path)
    plt_acc_with_std(x, y_p06_mean, y_p06_std, 'p06', data_info, output_path)
    plt_acc_with_std(x, y_p07_mean, y_p07_std, 'p07', data_info, output_path)
    plt_acc_with_std(x, y_p08_mean, y_p08_std, 'p08', data_info, output_path)
    plt_acc_with_std(x, y_p13_mean, y_p13_std, 'p13', data_info, output_path)

    plt_acc_with_std(x, y_mean_TOT, y_std_TOT, 'pTOT', data_info, output_path)

    print_result_mean_std(key_list, y_mean_TOT, y_std_TOT, data_info)




