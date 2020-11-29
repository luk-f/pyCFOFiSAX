# Authors: Lucas Foulon <lucas.foulon@gmail.com>
# License: BSD 3 clause

from numpy import array as np_array
from numpy import ndarray as np_ndarray
from numpy import zeros as np_zeros
from numpy import add as np_add
from numpy import sort as np_sort

from pyCFOFiSAX import ForestISAX


def _convert_rho_to_krho(rho, size_ds: int):
    """
    Convertit le paramètre ``rho`` (noté aussi :math:`\\varrho`) compris entre :math:`0 < \\varrho < 1` en une valeur comprise entre 0 et la taille du jeu de réference.

    :param list rho: la ou les valeurs de :math:`\\varrho` à convertir
    :param int size_ds: la taille du jeu de réference

    :returns: la liste des :math:`\\varrho \\times size\_ds`
    :rtype: list
    """

    if isinstance(rho, list):
        return [x * size_ds for x in rho]
    elif isinstance(rho, float) and rho < 1.0:
        return [size_ds * rho]
    else:
        raise ValueError("rho doit etre un float ou une liste de float entre 0.0 et 1.0")


def score_by_listvrang(k_list_result, k_rho):
    """
    Calcul des approximations CFOF à partir de la liste des vrang et selon la valeur des :math:`\\varrho` contenues dans la liste ``k_rho``.
    Une approximation CFOF obtenue pour chaque :math:`\\varrho` contenue dans la liste ``k_rho``.

    :param list(float) k_list_result: la liste des vrang pour la séquence à évaluer
    :param list(float) k_rho: la liste des :math:`\\varrho` pour le calcul des approximations CFOF.

    :returns: la liste des approximations CFOF
    :rtype: list(float)
    """

    nb_obj_total = len(k_list_result)

    score_list = np_zeros(len(k_rho))
    need_nn_prec = 0
    for k_rho_ite, k_rho_var in enumerate(k_rho):

        need_nn = k_rho_var - need_nn_prec

        while need_nn > 0:
            need_nn -= 1
            estim_final = k_list_result.pop(0)

        score_list[k_rho_ite] = estim_final / nb_obj_total
        need_nn_prec = k_rho_var

    return score_list


class CFOFiSAX:
    """
    La classe pour l'approximation *i*\ CFOF à l'aide d'arbres d'indexation *i*\ SAX.
    Contient la forêt d'arbres *i*\ SAX.
    """

    def __init__(self):
        """
        Fonction d'initialisation de la classe CFOFiSAX.

        :returns: contient seulement l'attribut ``forest_isax = None``.
        :rtype: CFOFiSAX
        """

        self.forest_isax = None

    def init_forest_isax(self, size_word: int, threshold: int,
                         data_ts: np_ndarray, base_cardinality: int = 2,
                         number_tree: int = 1,
                         indices_partition: list = None,
                         max_card_alphabet: int = 128, boolean_card_max: bool = True):
        """
        Initialise la forêt d'arbres *i*\ SAX.
        Nécessite les paramètre de la classe :class:`~pyCFOFiSAX._forest_iSAX.ForestISAX`.

        :param int size_word: la taille des mots SAX
        :param int threshold: le seuil maximal des nœuds
        :param numpy.ndarray data_ts: les séquences à insérer pour en extraire les stats
        :param int base_cardinality: la cardinalité la plus petite pour l'encodage *i*\ SAX
        :param int number_tree: le nombre d'arbres TreeISAX dans la forêt
        :param list indices_partition: une liste de liste d'indices où, pour chaque arbre, précise les indices des séquences à insérer
        :param int max_card_alphabet: si ``boolean_card_max == True``, la cardinalité maximale de l'encodage *i*\ SAX dans chacun des arbres
        :param bool boolean_card_max: si ``== True``, définit une cardinalité maximale pour l'encodage *i*\ SAX des séquences dans chacun des arbres
        """

        self.forest_isax = ForestISAX(size_word, threshold, data_ts, base_cardinality, number_tree,
                                      indices_partition, max_card_alphabet, boolean_card_max)

    def score_icfof(self, query: np_array, ntss: np_ndarray, rho=[0.001, 0.005, 0.01, 0.05, 0.1],
                    each_tree_score: bool = False, fast_method: bool = True):
        """
        Calcul les approximations *i*\ CFOF.
        Appelle une des deux fonctions selon le paramètre ``fast_method`` :
         - si ``True`` (par défaut) : :func:`~pyCFOFiSAX._forest_iSAX.ForestISAX.vranglist_by_idtree_faster`
         - si ``False`` : :func:`~pyCFOFiSAX._forest_iSAX.ForestISAX.vranglist_by_idtree`
        Puis trie la liste des vrang pour obtenir les approximations CFOF en fonction des valeurs de ``rho`` en paramètre.

        :param numpy.array query: la séquence à évaluer
        :param numpy.ndarray ntss: les séquences de réference
        :param list rho: les valeurs de rho pour le calcul des approximation
        :param bool each_tree_score: si vrai, retourne les scores obtenus dans chacun des arbres
        :param bool fast_method: si vrai, utilise les fonctions numpy pour le calcul, sinon parcourt l'arbre via une liste FIFO de nœuds

        :returns: les approximations *i*\ CFOF
        :rtype: numpy.ndarray
        """

        k_rho = _convert_rho_to_krho(rho, len(ntss))

        k_list_result_mean = np_zeros(len(ntss))

        if each_tree_score:
            k_list_result_ndarray = np_ndarray(shape=(self.forest_isax.number_tree, len(ntss)))

        for id_tree, tree in self.forest_isax.forest.items():

            ntss_tmp = np_array(ntss)[:, self.forest_isax.indices_partition[id_tree]]
            sub_query = query[self.forest_isax.indices_partition[id_tree]]

            if fast_method:
                k_list_result_tmp = tree.vrang_list_faster(sub_query, ntss_tmp)
            else:
                k_list_result_tmp = tree.vrang_list(sub_query, ntss_tmp)

            ratio_klist_tmp = (len(self.forest_isax.indices_partition[id_tree]) / self.forest_isax.size_word)
            k_list_result_mean = np_add(k_list_result_mean, np_array(k_list_result_tmp) * ratio_klist_tmp)
            if each_tree_score:
                k_list_result_ndarray[id_tree] = k_list_result_tmp

        k_list_result_mean = np_sort(k_list_result_mean, axis=None)

        if each_tree_score:
            k_list_result_ndarray.sort()
            return score_by_listvrang(k_list_result_mean.tolist(), k_rho), \
                   [score_by_listvrang(list(k_l_r), k_rho) for k_l_r in k_list_result_ndarray]

        return score_by_listvrang(k_list_result_mean.tolist(), k_rho)
