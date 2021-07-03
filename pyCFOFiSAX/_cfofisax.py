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
    Converts the parameter ``rho`` (also noted: math:`\\varrho`) between :math:`0 < \\varrho < 1` in a value between 0 and the size of the reference game.

    :param list rho: The values (s) of :math:`\\varrho` to convert
    :param int size_ds: The size of the reference game

    :returns: the list of :math:`\\varrho \\times size\_ds`
    :rtype: list
    """

    if isinstance(rho, list):
        return [x * size_ds for x in rho]
    elif isinstance(rho, float) and rho < 1.0:
        return [size_ds * rho]
    else:
        raise ValueError("Rho must be a float or a list of floats between 0.0 and 1.0")


def score_by_listvrang(k_list_result, k_rho):
    """
    Calculation of CFOF approximations from the vrang list and according to the value of :math:`\\varrho` contained in the ``k_rho`` list.
    A CFOF approximation obtained for each :math:`\\varrho` contained in the list``k_rho``.

    :param list(float) k_list_result: The list of vrang for the sequence to be evaluated
    :param list(float) k_rho: The list of :math:`\\varrho` for calculation of CFOF approximations.

    :returns: The list of CFOF approximations
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
    The class for approximation *i*\ CFOF using indexing trees *i*\ SAX.
    Contains the tree forest *i*\ SAX.
    """

    def __init__(self):
        """
        Initialization function of the CFOFiSAX class.

        :returns: contains only the attribute ``forest_isax = None``.
        :rtype: CFOFiSAX
        """

        self.forest_isax = None

    def init_forest_isax(self, size_word: int, threshold: int,
                         data_ts: np_ndarray, base_cardinality: int = 2,
                         number_tree: int = 1,
                         indices_partition: list = None,
                         max_card_alphabet: int = 128, boolean_card_max: bool = True):
        """
        Initializes the forest of *i*\ SAX trees.
        Requires the parameters of the class :class:`~pyCFOFiSAX._forest_iSAX.ForestISAX`.

        :param int size_word: The size of the SAX words
        :param int threshold: The maximum threshold of nodes
        :param numpy.ndarray data_ts: The sequences to be inserted to extract the stats
        :param int base_cardinality: The smallest cardinality for encoding *i*\ SAX
        :param int number_tree: The number of TreeISAX trees in the forest
        :param list indices_partition: A list of list of indices where, for each tree, specifies the indices of the sequences to be inserted
        :param int max_card_alphabet: if ``boolean_card_max == True``, The maximum cardinality of encoding *i*\ SAX in each tree
        :param bool boolean_card_max: if ``== True``, Defines a maximum cardinality for encoding *i*\ SAX sequences in each of the trees
        """

        self.forest_isax = ForestISAX(size_word, threshold, data_ts, base_cardinality, number_tree,
                                      indices_partition, max_card_alphabet, boolean_card_max)

    def score_icfof(self, query: np_array, ntss: np_ndarray, rho=[0.001, 0.005, 0.01, 0.05, 0.1],
                    each_tree_score: bool = False, fast_method: bool = True):
        """
        Calculate *i*\ CFOF Approximations.
        Call one of the two functions according to the parameter ``fast_method`` :
         - if ``True`` (default) : :func:`~pyCFOFiSAX._forest_iSAX.ForestISAX.vranglist_by_idtree_faster`
         - if ``False`` : :func:`~pyCFOFiSAX._forest_iSAX.ForestISAX.vranglist_by_idtree`
        Then sort the vrang list to get CFOF approximations based on ``rho`` parameter values.

        :param numpy.array query: The sequence to be evaluated
        :param numpy.ndarray ntss: Reference sequences
        :param list rho: Rho values for calculating approximation
        :param bool each_tree_score: if vrai, teturns the scores obtained in each of the trees
        :param bool fast_method: if vrai, uses the NUMPY functions for the calculation, otherwise travels the tree via a FIFO list of nodes

        :returns: *i*\ CFOF approximations
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
