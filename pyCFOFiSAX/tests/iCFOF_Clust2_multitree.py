# Authors: Lucas Foulon <lucas.foulon@gmail.com>
# License: BSD 3 clause

from pyCFOFiSAX import CFOFiSAX

from .profiling import ProfilingContext

import numpy as np
import pandas as pd

import os
from tqdm import tqdm

from scipy.stats import norm
from scipy.special import ndtri
from scipy.stats import kurtosis


def cfof_by_cdf(dataframe, num_tree, rho, kurto):
    """
    Calcul du score approxime en agregeant les scores de chacun des arbres 
    (fonctionne si num_tree de icfof_Clust2 >1)
    Retourne index et scores

    dataframe :
        le jeu en cours d'evaluation

    num_tree :
        nombre d'arbre iSAX utilise dans la foret iSAX

    rho :
        le parametre cfof

    kurto :
        le kurtosis du jeu du dataframe
    """
    
    col_list_subtree = []
    for tmp_num_tree in range(num_tree):
        col_list_subtree.append(f"score_tree{tmp_num_tree}")

    tmp_index = dataframe[col_list_subtree].dropna().index
    # kurto = 3
    
    # cste = (np.sqrt(num_tree)-1)*2*ndtri(rho)/np.sqrt(num_tree*(kurto+3))
    # cste = 2*ndtri(rho)/np.sqrt(kurto+3) - 2*num_tree*ndtri(rho)/np.sqrt(num_tree*(kurto+3))
    cste = (np.sqrt(num_tree)-num_tree)*2*ndtri(rho)/np.sqrt(num_tree*(kurto+3))
    
    tmp_eq = np.vectorize(ndtri)(dataframe[col_list_subtree].dropna()).sum(axis=1)
    tmp_eq = tmp_eq / np.sqrt(num_tree) + cste
    
    tmp_eq = norm.cdf(tmp_eq)
    
    return tmp_index, tmp_eq


def icfof_clust2(name_dataset, dim_dataset, num_tree, each_tree_score, ite_to_evaluate):
    """
    Fonction pour le calcul des scores iCFOF des objets Clust2

    Parametres
    ----------
    name_dataset :
        nom du jeu de donnees

    dim_dataset :
        dimension des objets du Clust2

    num_tree :
        nombre d'arbre iSAX utilise dans la foret iSAX

    each_tree_score :
        seulement si num_tree > 1
        retourne les scores obtenus dans chaque arbre
        INDISPENSABLE pour le calcul des scores agrege a partir des scores iCFOF

    ite_to_evaluate :
        position des objets a evaluer (pour le calcul des scores)

    """

    rho_list = [0.01, 0.05, 0.1]
    col_use = list(range(0, dim_dataset+len(rho_list)))

    file_path = os.path.join("pyCFOFiSAX/tests/data_test/data/", '.'.join([name_dataset, 'csv']))
    ndarray_dataset = np.genfromtxt(file_path,
                                    delimiter=',',
                                    skip_header=1,
                                    usecols=col_use)

    ndarray_dataset_no_score = ndarray_dataset[:, :dim_dataset]

    threshold = 30
    cfof_isax = CFOFiSAX()
    cfof_isax.init_forest_isax(size_word=dim_dataset,
                               threshold=threshold,
                               data_ts=ndarray_dataset_no_score,
                               base_cardinality=2, number_tree=num_tree)

    cfof_isax.forest_isax.index_data(ndarray_dataset_no_score)

    cfof_isax.forest_isax.preprocessing_forest_for_icfof(ndarray_dataset_no_score,
                                                         bool_print=True, count_num_node=True)

    dataframe_score = pd.DataFrame(columns=["score_real", "score_approx_meanvrang"])

    tmp_columns = []
    for tmp_num_tree in range(num_tree):
        tmp_columns.append(f"nb_node_total_tree{tmp_num_tree}")
    for tmp_num_tree in range(num_tree):
        tmp_columns.append(f"nb_node_visited_mean_tree{tmp_num_tree}")

    dataframe_node = pd.DataFrame(columns=tmp_columns)

    score_list = []

    for tmp_ite in tqdm(ite_to_evaluate):

        with ProfilingContext("temps scores"):
            score = cfof_isax.score_icfof(
                ndarray_dataset_no_score[tmp_ite], ndarray_dataset_no_score,
                rho=rho_list, each_tree_score=each_tree_score,
                fast_method=True)

        with ProfilingContext("temps nodes"):
            num_nodes = cfof_isax.forest_isax.number_nodes_visited(ndarray_dataset_no_score[tmp_ite],
                                                                   ndarray_dataset_no_score)
            dataframe_node.loc[tmp_ite] = num_nodes

        if num_tree > 1 and each_tree_score:
            score_list.append(score)
            dataframe_score.loc[tmp_ite, "score_real"] = ndarray_dataset[tmp_ite, dim_dataset+len(rho_list)-1]
            dataframe_score.loc[tmp_ite, "score_approx_meanvrang"] = score[0][len(rho_list)-1]
            tmp_num_tree = 0
            for tmp_tree_score in score[1]:
                dataframe_score.loc[tmp_ite, f"score_tree{tmp_num_tree}"] = tmp_tree_score[len(rho_list)-1]
                tmp_num_tree += 1
        elif num_tree == 1 or not each_tree_score:
            score_list.append(score)
            dataframe_score.loc[tmp_ite, "score_real"] = ndarray_dataset[tmp_ite, dim_dataset+len(rho_list)-1]
            dataframe_score.loc[tmp_ite, "score_approx_meanvrang"] = score[len(rho_list)-1]

    ProfilingContext.print_summary()

    if num_tree > 1 and each_tree_score:
        print(kurtosis(ndarray_dataset_no_score, fisher=False, axis=None))
        kurto = kurtosis(ndarray_dataset_no_score, fisher=False, axis=None)
        index_res, val_res = cfof_by_cdf(dataframe_score, num_tree, 0.1, kurto)

        dataframe_score.loc[index_res, 'theorical_score'] = val_res

    return dataframe_score, dataframe_node
