# Authors: Lucas Foulon <lucas.foulon@gmail.com>
# License: BSD 3 clause

from pyCFOFiSAX import CFOFiSAX

from .profiling import ProfilingContext

import numpy as np
import pandas as pd

import json

import os
import psutil
import logging
from tqdm import tqdm


def print_rss():
    return psutil.Process(os.getpid()).memory_info()[0]/10**6


def size_probative_nab(file_path: str) -> int:
    try:
        f = open(file_path, "r")
        f_lines = f.readlines()
        count = len(f_lines)
        f.close()
    except Exception as e:
        logging.error(f"Error, can't open file {file_path} : {type(e)} : {e}")
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

    logging.basicConfig(format='%(message)s', level=logging.INFO)

    # recuperation des donnees
    dir_path = os.path.join("./pyCFOFiSAX/tests/data_test/data/NAB-master", dataset_type)
    file_path = os.path.join(dir_path, name_dataset)
    df_dataset_probative_length = size_probative_nab(file_path)
    df_dataset_root = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)

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
        mask_tmp = (my_df_test_seq_score.index >= interval_anomaly[0]) & \
                   (my_df_test_seq_score.index <= interval_anomaly[1])
        my_df_test_seq_score.loc[mask_tmp,'label'] = 1

    #
    # Parametre sequences
    #

    logging.info("\n***** paramètres *****\n")

    logging.info(f"\tJeu : {os.path.join(dataset_type, name_dataset)}")
    logging.info(f"\tMet à jour l'arbre : {'true' if update else 'false'}")
    logging.info(f"\tTaille de la période probatoire (seq de référence) = {period_probative}")
    logging.info(f"\tNombre d'arbre iSAX dans la forêt : num_tree = {num_tree}")
    logging.info("\tRetourne le score pour tous les arbres (seulement si num_tree > 1) : "
                 f"{'true' if each_tree_score else 'false'}")
    logging.info(f"\tType de seuil de séparation des noeuds feuilles : {type_split_node}")

    logging.info(f"\n\tNombre de valeurs totals : {len(df_dataset_root)}")
    logging.info(f"\tTaille de la plus petite periode probatoire : {df_dataset_probative_length}\n")

    ## calcul des differentes tailles de valeur
    ratio_divide_seq = 2
    logging.info(f"\tRatio taille fenêtre : {ratio_divide_seq}")

    seq_len_list = []
    size_tmp = int(df_dataset_probative_length//ratio_divide_seq)
    while size_tmp > 5:
        
        size_tmp //= ratio_divide_seq
        seq_len_list.append(int(size_tmp))
        
    logging.info(f"\tTaille des fenêtres de la séquence : {seq_len_list}")
    number_of_values = sum(seq_len_list)
    logging.info(f"\tTotal de valeurs de la fenêtre : {number_of_values}")
    sequence_length = len(seq_len_list)
    logging.info(f"\tNombre de fenêtre par séquence pour 1 indicateur = {sequence_length}")

    #
    # Choix de la periode test, n*df_dataset_probative_length
    #

    number_of_probative_period = int(period_probative / int(df_dataset_probative_length))
    del period_probative
    logging.info(f"\tNuméro de la période à évaluer = {number_of_probative_period}")
    n_prob_length = df_dataset_probative_length * number_of_probative_period
    logging.info(f"\tEdit : Taille de la période probatoire (seq de référence) = {n_prob_length}")

    logging.info("\n***** fin paramètres *****\n")

    # ##### Préparation des indicateurs

    for aggregation in seq_len_list:
        df_dataset_root[f'doc_count_mean_{aggregation}'] = df_dataset_root['value']\
            .rolling(aggregation, min_periods=aggregation).sum()
        df_dataset_root[f'doc_count_mean_{aggregation}_by_value'] = \
            df_dataset_root[f'doc_count_mean_{aggregation}'] / aggregation

        df_dataset_root[f'doc_count_std_{aggregation}'] = df_dataset_root['value']\
            .rolling(aggregation, min_periods=aggregation).std(ddof=0)

    # ##### Creation des sequences 

    seq_list = []
    count_seq_hist = 0

    with ProfilingContext("temps création sequence"):

        for i in range(0, len(df_dataset_root)-int(number_of_values)+1, 1):

            """
            C'est ici que je construis ma séquence
            """
            start = i
            tmp_move = 0
            tmp_list_avg = []
            tmp_list_std = []
            tmp_list_index = []
            for aggregation in seq_len_list:
                tmp_list_index.append(df_dataset_root.iloc[aggregation+start-1+tmp_move].name)
                tmp_list_avg.append(df_dataset_root[f'doc_count_mean_{aggregation}_by_value']\
                                        [aggregation+start-1+tmp_move])
                tmp_list_std.append(df_dataset_root[f'doc_count_std_{aggregation}'][aggregation+start-1+tmp_move])
                tmp_move += aggregation
            s1 = pd.Series(tmp_list_avg, index=tmp_list_index)
            s2 = pd.Series(tmp_list_std, index=tmp_list_index)

            seq_list.append(s1.append(s2).rename(f"st_{i}"))
            count_seq_hist += 1

    ProfilingContext.print_summary()
    logging.info(f"\tfor {count_seq_hist} sequences")
    logging.info(f"\ttaille RSS = {print_rss()} mo")

    with ProfilingContext("temps seq in df"):
        list_columns = [f'd{int(x)}' for x in range(sequence_length*2)]
        df_dataset = pd.DataFrame(columns=list_columns)
        # On crée la liste des n points
        for seq in seq_list:
            if seq.index[-1] not in df_dataset.index:
                df_dataset.loc[seq.index[-1]] = seq.tolist()
            else:
                df_dataset_tmp = pd.DataFrame(columns=list_columns)
                df_dataset_tmp.loc[seq.index[-1]] = seq.tolist()
                df_dataset = df_dataset.append(df_dataset_tmp)
    ProfilingContext.print_summary()
    logging.info(f"\ttaille RSS = {print_rss()} mo\n")

    if n_prob_length == 0:
        start_test = int(number_of_values / 2)
    else:
        start_test = n_prob_length - number_of_values
    step_test = df_dataset_probative_length

    # TRAIN
    date_start_train = df_dataset.index[0]
    # TODO nb_elt_train a parametrer
    date_end_train = df_dataset.index[start_test]

    logging.info("--- préparation date train et test")
    logging.info(f"\tinterval date train [ {date_start_train}, {date_end_train}]")

    # TEST
    date_start_test = df_dataset.index[start_test]
    if start_test-1+step_test >= df_dataset.shape[0]:
        date_end_test = df_dataset.index[-1]
    else:
        if n_prob_length == 0:
            date_end_test = df_dataset_root.index[step_test-1]
        else:
            date_end_test = df_dataset.index[start_test+step_test]

    logging.info(f"\tdate début test {date_start_test}")

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

        logging.info("\n ***** Démarrage construction et insertion dans forêt *****\n")

        seuil = type_split_node[0]
        for k_dict in sorted(type_split_node.keys()):
            if len(ndarray_dataset_train) < type_split_node[k_dict]:
                seuil = type_split_node[k_dict]
            else:
                break

        tmp_divider = sequence_length*2 / num_tree
        split_tree_num = sequence_length*2 / tmp_divider
        i_partition = np.array_split(np.arange(sequence_length*2), split_tree_num)

        logging.info(f"\tindice de répartition multi arbre : {i_partition}")

        cfof_isax = CFOFiSAX()
        cfof_isax.init_forest_isax(size_word=sequence_length*2,
                                   threshold=seuil,
                                   data_ts=ndarray_dataset_train,
                                   base_cardinality=2, number_tree=num_tree,
                                   indices_partition=i_partition)

        with ProfilingContext("insertion arbre(s)"):
            cfof_isax.forest_isax.index_data(ndarray_dataset_train)
        ProfilingContext.print_summary()
        logging.info(f"\tmémoire RSS avant pretraitement = {print_rss()} mo")

        cfof_isax.forest_isax.preprocessing_forest_for_icfof(ndarray_dataset_train, bool_print=False)

        logging.info(f"\tmémoire RSS après pretraitement = {print_rss()} mo")

        logging.info("\n ***** Fin construction et insertion dans forêt *****\n")

        tmp_columns = []
        for tmp_num_tree in range(num_tree):
            tmp_columns.append(f"nb_node_total_tree {tmp_num_tree}")
        for tmp_num_tree in range(num_tree):
            tmp_columns.append(f"nb_node_visited_mean_tree {tmp_num_tree}")

        dataframe_node = pd.DataFrame(columns=tmp_columns)

        with ProfilingContext(f"{len(ndarray_dataset_test)} scores calculés en"):
            nbr_scored = 0

            logging.info("\n ***** Démarrage évaluation séq... *****\n")

            for row_loc, row_test in tqdm(enumerate(ndarray_dataset_test[:]), desc="\trow loc"):

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
                    cmpt_insert = cfof_isax.forest_isax.index_data(np.array([row_test]))
                    if cmpt_insert[0] > 0:
                        ndarray_dataset_train = np.vstack((ndarray_dataset_train, row_test))
                        cfof_isax.forest_isax.preprocessing_forest_for_icfof(ndarray_dataset_train, bool_print=False)

        logging.info("\tFini             ")
        ProfilingContext.print_summary()
        logging.info("\n ***** Fin évaluation séq *****\n")

    else:
        logging.info("Warning : la période train est plus grande que la taille du jeu")
        return -1

    return my_df_test_seq_score, dataframe_node
