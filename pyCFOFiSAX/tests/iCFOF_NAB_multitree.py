# Authors: Lucas Foulon <lucas.foulon@gmail.com>
# License: BSD 3 clause

from pyCFOFiSAX import CFOFiSAX
 
import time

import numpy as np
import pandas as pd

import json

import os
import psutil


def print_rss():
    process = psutil.Process(os.getpid())
    # print(process)
    print(str(process.memory_info()[0]/10**6)+" Mo")


def size_probative_nab(thefilepath):
    f = open(thefilepath, "r")
    f_lines = f.readlines()
    count = len(f_lines)
    f.close() 
    if count <= 5000:
        probative_length = int(count*0.15)
    else:
        probative_length = 750
    return probative_length


def icfof_nab(dataset_type, name_dataset, period_probative,
              update=False, num_tree=1, each_tree_score=False,
              type_split_node={0: 30}):
    """
    Fonction pour le calcul des scores iCFOF des séquences NAB

    Parametres
    ----------
    :param dataset_type:
        famille de jeux de donnees

    :param name_dataset:
        nom du jeu de donnees

    :param update:
        si vrai, insere la derniere sequence dans l'arbre iSAX, met a jour l'arbre et stats entre noeuds et sequences,
        avant le calcul du score de la prochaine sequence

    :param period_probative:
        taille du jeu de réference

    :param num_tree:
        nombre d'arbre iSAX utilise dans la foret iSAX

    :param each_tree_score:
        seulement si num_tree > 1
        retourne les scores obtenus dans chaque arbre
        INDISPENSABLE pour le calcul des scores agrege a partir des scores iCFOF

    :param dict type_split_node:
        type de separation des noeuds feuilles selon seuil
        {0: 10, 3000: 20} : seuil a 10 si nb de sequences a l'init de l'arbre est < 3000, 20 sinon
        {0: 30} : seuil a 30
    """

    # recuperation des donnees
    thefilepath = "./pyCFOFiSAX/tests/data_test/data/NAB-master/"+dataset_type+"/"+name_dataset
    df_dataset_probative_length = size_probative_nab(thefilepath)
    df_dataset_root = pd.read_csv(thefilepath, index_col='timestamp', parse_dates=True)

    rho_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    # copie dataframe final avec score
    my_df_test_seq_score = df_dataset_root.copy()
    for r in rho_list:
        my_df_test_seq_score['p'+str(r)] = 0
    if num_tree > 1 and each_tree_score:
        for tmp_num_tree in range(num_tree):
            for r in rho_list:
                my_df_test_seq_score['p' + str(r) + '_tree' + str(tmp_num_tree)] = 0


    # recuperation de l'etiquetage des anomalies
    with open("./pyCFOFiSAX/tests/data_test/data/NAB-master/labels/combined_windows.json", 'r') as f:
        json_label = json.load(f)

    # etiquetage des anomalies dans le dataframe
    my_df_test_seq_score['label'] = 0
    for interval_anomaly in json_label[dataset_type+"/"+name_dataset]:
        mask_tmp = (my_df_test_seq_score.index >= interval_anomaly[0]) & (my_df_test_seq_score.index <= interval_anomaly[1])
        my_df_test_seq_score.loc[mask_tmp,'label'] = 1



    #
    # Parametre sequences
    #

    print("\n***** paramètres *****\n")

    print("\tJeu : ", dataset_type+"/"+name_dataset)
    print("\tMet à jour l'arbre : ", end="")
    if update:
        print("TRUE")
    else:
        print("FALSE")
    print("\tTaille de la période probatoire (seq de référence) = ", str(period_probative))
    print("\tDNombre d'arbre iSAX dans la forêt : num_tree = ", str(num_tree))
    print("\tRetourne le score pour tous les arbres (seulement si num_tree > 1) : ", end="")
    if each_tree_score:
        print("TRUE")
    else:
        print("FALSE")
    print("\tType de seuil de séparation des noeuds feuilles : ", str(type_split_node))

    print("\n\tNombre de valeurs totals : ", str(len(df_dataset_root)))
    print("\tTaille de la plus petite periode probatoire : ", end="")
    print(df_dataset_probative_length)

    ## calcul des differentes tailles de valeur
    ratio_divide_seq = 2
    print("\tRatio taille fenêtre :", str(ratio_divide_seq))

    seq_len_list = []
    size_tmp = int(df_dataset_probative_length//ratio_divide_seq)
    while size_tmp > 5:
        
        size_tmp //= ratio_divide_seq
        seq_len_list.append(int(size_tmp))
        
    print("\tTaille des fenêtres de la séquence : ", str(seq_len_list))
    number_of_values = sum(seq_len_list)
    print("\tTotal de valeurs de la fenêtre : ", str(number_of_values))
    sequence_length = len(seq_len_list)
    print("\tNombre de fenêtre par séquence pour 1 indicateur = ", str(sequence_length))

    #
    # Choix de la periode test, n*df_dataset_probative_length
    #

    number_of_probative_period = int(period_probative / int(df_dataset_probative_length))
    del period_probative
    print("\tNuméro de la période à évaluer = ", str(number_of_probative_period))
    n_prob_length = df_dataset_probative_length * number_of_probative_period
    print("\tEdit : Taille de la période probatoire (seq de référence) = ", str(n_prob_length))

    print("\n***** fin paramètres *****\n")

    # ##### Préparation des indicateurs

    for aggregation in seq_len_list:
        df_dataset_root['doc_count_mean_' + str(aggregation)] = df_dataset_root['value'].rolling(aggregation, min_periods=aggregation).sum()
        df_dataset_root['doc_count_mean_' + str(aggregation) + '_by_value'] = df_dataset_root['doc_count_mean_' + str(aggregation)] / aggregation

        df_dataset_root['doc_count_std_' + str(aggregation)] = df_dataset_root['value'].rolling(aggregation, min_periods=aggregation).std(ddof=0)

    # ##### Creation des sequences 

    seq_list = []
    start_time = time.time()
    count_seq_hist = 0

    for i in range(0, len(df_dataset_root)-int(number_of_values)+1, 1):

        """
        C'est ici que je construis mon mot
        """
        start = i
        tmp_move = 0
        tmp_list_avg = []
        tmp_list_std = []
        tmp_list_index = []
        for aggregation in seq_len_list:
            tmp_list_index.append(df_dataset_root.iloc[aggregation+start-1+tmp_move].name)
            tmp_list_avg.append(df_dataset_root['doc_count_mean_'+str(aggregation)+'_by_value'][aggregation+start-1+tmp_move])
            tmp_list_std.append(df_dataset_root['doc_count_std_'+str(aggregation)][aggregation+start-1+tmp_move])
            tmp_move += aggregation
        s1 = pd.Series(tmp_list_avg, index=tmp_list_index)
        s2 = pd.Series(tmp_list_std, index=tmp_list_index) 
        
        seq_list.append(s1.append(s2).rename("st_" + str(i)))
        count_seq_hist += 1

    print("--- temps creation sequence :  %s seconds ---" % (time.time() - start_time))
    print("\tfor",count_seq_hist,"sequences")
    print("\ttaille RSS = ", end="")
    print_rss()

    start_time = time.time()
    list_columns = ['d'+str(x) for x in range(sequence_length*2)]
    df_dataset = pd.DataFrame(columns=list_columns)
    # On crée la liste des n points
    for seq in seq_list:
        if seq.index[-1] not in df_dataset.index:
            df_dataset.loc[seq.index[-1]] = seq.tolist()
        else:
            df_dataset_tmp = pd.DataFrame(columns=list_columns)
            df_dataset_tmp.loc[seq.index[-1]] = seq.tolist()
            df_dataset = df_dataset.append(df_dataset_tmp)
    print("--- temps seq in df :  %s seconds ---" % (time.time() - start_time))
    print("\ttaille RSS = ", end="")
    print_rss()
    print("\n")

    if n_prob_length == 0:
        start_test = int(number_of_values / 2)
    else:
        start_test = n_prob_length - number_of_values
    step_test = df_dataset_probative_length

    # TRAIN
    date_start_train = df_dataset.index[0]
    # TODO nb_elt_train a parametrer
    date_end_train = df_dataset.index[start_test]

    print("--- préparation date train et test")
    print("\tinterval date train [", str(date_start_train), ", ", str(date_end_train), "]")

    # TEST
    date_start_test = df_dataset.index[start_test]
    if start_test-1+step_test >= df_dataset.shape[0]:
        date_end_test = df_dataset.index[-1]
    else:
        if n_prob_length == 0:
            date_end_test = df_dataset_root.index[step_test-1]
        else:
            date_end_test = df_dataset.index[start_test+step_test]

    print("\tdate début test ", str(date_start_test))

    # préparation données train et test en numpy.ndarray
    ndarray_dataset_train = df_dataset.loc[df_dataset.index <= date_end_train].values
    ndarray_dataset_test = df_dataset.loc[(df_dataset.index > date_start_test) & (df_dataset.index <= date_end_test)].values
    index_dataset_test = df_dataset.loc[(df_dataset.index > date_start_test) & (df_dataset.index <= date_end_test)].index

    if n_prob_length == 0:
        my_df_test_seq_score = my_df_test_seq_score.loc[my_df_test_seq_score.index <= index_dataset_test[-1]]
    else:
        my_df_test_seq_score = my_df_test_seq_score.loc[(my_df_test_seq_score.index >= index_dataset_test[0]) &
                                                        (my_df_test_seq_score.index <= index_dataset_test[-1])]

    # while n_prob_length < len(df_dataset):
    if n_prob_length < len(df_dataset)+number_of_values:

        print("\n ***** Démarrage construction et insertion dans forêt *****\n")

        seuil = type_split_node[0]
        for k_dict in sorted(type_split_node.keys()):
            if len(ndarray_dataset_train) < type_split_node[k_dict]:
                seuil = type_split_node[k_dict]
            else:
                break

        tmp_divider = sequence_length*2 / num_tree
        split_tree_num = sequence_length*2 / tmp_divider
        i_partition = np.array_split(np.arange(sequence_length*2), split_tree_num)

        print("\tindice de répartition multi arbre : "+str(i_partition))

        cfof_isax = CFOFiSAX()
        cfof_isax.init_forest_isax(size_word=sequence_length*2,
                                   threshold=seuil,
                                   data_ts=ndarray_dataset_train,
                                   base_cardinality=2, number_tree=num_tree,
                                   indices_partition=i_partition)

        start_time = time.time()
        cfof_isax.forest_isax.index_data(ndarray_dataset_train)
        print("\tinsertion arbre(s) --- %s seconds ---" % (time.time() - start_time))
        print("\tmémoire RSS avant pretraitement", end=" : ")
        print_rss()

        cfof_isax.forest_isax.preprocessing_forest_for_icfof(ndarray_dataset_train, bool_print=False)

        print("\tmémoire RSS après pretraitement", end=" : ")
        print_rss()

        print("\n ***** Fin construction et insertion dans forêt *****\n")

        tmp_columns = []
        for tmp_num_tree in range(num_tree):
            tmp_columns.append("nb_node_total_tree" + str(tmp_num_tree))
        for tmp_num_tree in range(num_tree):
            tmp_columns.append("nb_node_visited_mean_tree" + str(tmp_num_tree))

        dataframe_node = pd.DataFrame(columns=tmp_columns)

        start_time = time.time()
        nbr_scored = 0

        print("\n ***** Démarrage évaluation séq... *****\n")

        for row_loc, row_test in enumerate(ndarray_dataset_test[:]):

            print("\trow loc " + str(row_loc), end="\r")

            ite_loc = row_loc+start_test+1

            score_list = cfof_isax.score_icfof(row_test, ndarray_dataset_train,
                                               rho_list, each_tree_score=each_tree_score,
                                               fast_method=True)

            num_nodes = cfof_isax.forest_isax.number_nodes_visited(row_test, ndarray_dataset_train)
            dataframe_node.loc[ite_loc] = num_nodes

            if num_tree > 1 and each_tree_score:
                for tmp_i in range(len(score_list[0])):
                    my_df_test_seq_score.iloc[row_loc, tmp_i+1] = score_list[0][tmp_i]
                for tmp_i, tmp_score_by_tree in enumerate(score_list[1:][0]):
                    for tmp_j, tmp_score in enumerate(tmp_score_by_tree):
                        tmp_pos = (tmp_i+1)*len(rho_list) + tmp_j + 1
                        my_df_test_seq_score.iloc[row_loc, tmp_pos] = tmp_score

            else:
                for i in range(len(rho_list)):
                    my_df_test_seq_score.iloc[row_loc, i + 1] = score_list[i]

            nbr_scored += 1

            if update:
                cmpt_insert = cfof_isax.forest_isax.index_data([row_test])
                if cmpt_insert[0] > 0:
                    ndarray_dataset_train = np.vstack((ndarray_dataset_train, row_test))
                    cfof_isax.forest_isax.preprocessing_forest_for_icfof(ndarray_dataset_train, bool_print=False)

        print("\tFini             ")
        print("\t" + str(nbr_scored) + " scores calculés en " + str(time.time() - start_time) + "secs")
        print("\n ***** Fin évaluation séq *****\n")

    else:
        print("Warning : la période train est plus grande que la taille du jeu")
        return -1

    return my_df_test_seq_score, dataframe_node
