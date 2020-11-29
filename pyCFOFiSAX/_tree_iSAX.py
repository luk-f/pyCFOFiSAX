# Authors: Lucas Foulon <lucas.foulon@gmail.com>
# License: BSD 3 clause

from pyCFOFiSAX._node import RootNode
from pyCFOFiSAX._isax import IndexableSymbolicAggregateApproximation

from anytree import RenderTree

from scipy.stats import norm
from scipy.spatial.distance import cdist

from numpy import array as np_array
from numpy import ndarray as np_ndarray
from numpy import empty as np_empty
from numpy import zeros as np_zeros
from numpy import uint32 as np_uint32
from numpy import float64 as np_float64
from numpy import add as np_add
from numpy import sum as np_sum
from numpy import log2 as np_log2
from numpy import sqrt as np_sqrt
from numpy import square as np_square
from numpy import divide as np_divide
from numpy import mean as np_mean
from numpy import max as np_max
from numpy import minimum as np_minimum
from numpy import maximum as np_maximum
from numpy import place as np_place
from numpy import repeat as np_repeat
from numpy import logical_and as np_logical_and
from numpy import less_equal as np_less_equal
from numpy import greater as np_greater
from numpy import logical_not as np_logical_not
from numpy import linspace as np_linspace
from numpy import searchsorted as np_searchsorted
from numpy import multiply as np_multiply

from scipy.stats import norm as scipy_norm

from bisect import bisect as bisect_bisect

from sys import getsizeof

from time import time as time_time
from sys import stdout

from numba import njit, prange


@njit(nogil=True)
def vrang_seq_ref(distance, max_array, min_array, cdf_mean, cdf_std, num_ts_by_node,
                  index_cdf_bin, cdf_bins):
    """
    Calcule le vrang à partir de la distance entre la séquence à évaluer et la séquence de réference.

    :param float distance: la distance entre les deux séquences
    :param np_array max_array: les distances max entre les nœuds feuilles de l'arbre et la séquence de réference
    :param np_array min_array: les distances min entre les nœuds feuilles de l'arbre et la séquence de réference
    :param np_array cdf_mean: les distances moyennes entre les nœuds feuilles de l'arbre et la séquence de réference
    :param np_array cdf_std: la dispersion des distances dans chaque nœud feuille
    :param np_array num_ts_by_node: le nombre de séquence dans chaque nœud feuille
    :param np_array index_cdf_bin: l'index de la CDF ``cdf_bins``
    :param np_array cdf_bins: les valeurs de CDF de loi normale centrée à l'origine et d'écart-type 1

    :returns: le vrang
    :rtype: int
    """
    vrang = 0

    vrang += num_ts_by_node[
        np_greater(distance, max_array)
    ].sum()

    # tous les nœuds borderlines
    boolean_grp = np_logical_and(np_less_equal(distance, max_array),
                                 np_greater(distance, min_array))

    cdf_mean_grp = cdf_mean[boolean_grp]
    cdf_std_grp = cdf_std[boolean_grp]
    num_ts_by_node_grp = num_ts_by_node[boolean_grp]

    vrang += num_ts_by_node_grp[
        np_logical_and(cdf_std_grp <= 0.0, cdf_mean_grp < distance)
    ].sum()

    cdf_mean_grp = cdf_mean_grp[cdf_std_grp > 0.0]
    num_ts_by_node_grp = num_ts_by_node_grp[cdf_std_grp > 0.0]
    cdf_std_grp = cdf_std_grp[cdf_std_grp > 0.0]

    distance_normalized = (distance - cdf_mean_grp) / cdf_std_grp

    vrang += num_ts_by_node_grp[
        distance_normalized > 4.0
        ].sum()

    new_boolean_grp = np_logical_and(distance_normalized <= 4.0,
                                     distance_normalized >= -4.0)

    num_ts_by_node_grp = num_ts_by_node_grp[new_boolean_grp]
    distance_normalized_grp = distance_normalized[new_boolean_grp]

    index_for_bin = np_searchsorted(index_cdf_bin, distance_normalized_grp)
    vrang += np_multiply(cdf_bins[index_for_bin], num_ts_by_node_grp).sum()
    return vrang


@njit(nogil=True, parallel=True)
def vrang_list_for_all_seq_ref(len_seq_list, distance,
                               max_array, min_array,
                               cdf_mean, cdf_std,
                               num_ts_by_node,
                               index_cdf_bin, cdf_bins):
    """
    Fait appel à la fonction :func:`~pyCFOFiSAX.tree_iSAX.vrang_seq_ref` pour chaque séquence de réference.

    :param float len_seq_list: le nombre de séquence de réference
    :param np_array distance: la distance entre les deux séquences
    :param np_ndarray max_array: les distances max entre les nœuds feuilles de l'arbre et la séquence de réference
    :param np_ndarray min_array: les distances min entre les nœuds feuilles de l'arbre et la séquence de réference
    :param np_ndarray cdf_mean: les distances moyennes entre les nœuds feuilles de l'arbre et la séquence de réference
    :param np_array cdf_std: la dispersion des distances dans chaque nœud feuille
    :param np_array num_ts_by_node: le nombre de séquence dans chaque nœud feuille
    :param np_array index_cdf_bin: l'index de la CDF ``cdf_bins``
    :param np_array cdf_bins: les valeurs de CDF de loi normale centrée à l'origine et d'écart-type 1

    :returns: la liste des vrang
    :rtype: np_array
    """
    vrang_array = np_zeros(len_seq_list)
    for ii_tmp in prange(len_seq_list):
        vrang_array[ii_tmp] = vrang_seq_ref(distance[ii_tmp],
                                            max_array[ii_tmp], min_array[ii_tmp],
                                            cdf_mean[ii_tmp], cdf_std,
                                            num_ts_by_node,
                                            index_cdf_bin, cdf_bins)
    return vrang_array


@njit(nogil=True)
def nodes_visited_for_seq_ref(distance, max_array, min_array, list_parent_node):
    """

    """

    boolean_grp = np_logical_and(np_less_equal(distance, max_array),
                                 np_greater(distance, min_array))
    count_visited_nodes = np_sum(boolean_grp)
    not_boolean_grp = np_logical_not(boolean_grp)
    not_node_parent = list_parent_node[not_boolean_grp]
    boolean_grp = np_logical_and(np_less_equal(distance, max_array[not_node_parent]),
                                 np_greater(distance, min_array[not_node_parent]))
    count_visited_nodes += np_sum(boolean_grp)
    # root is not counted
    count_visited_nodes -= 1
    return count_visited_nodes


@njit(nogil=True, parallel=True)
def nodes_visited_for_all_seq_ref(len_seq_list, distance,
                                  max_array, min_array,
                                  list_parent_node):
    """

    """
    count_visited_nodes = np_zeros(len_seq_list)
    for ii_tmp in prange(len_seq_list):
        count_visited_nodes[ii_tmp] = nodes_visited_for_seq_ref(distance[ii_tmp],
                                                                max_array[ii_tmp], min_array[ii_tmp],
                                                                list_parent_node)
    return count_visited_nodes


class TreeISAX:
    """
    La classe TreeISAX contenant
     * la distribution iSAX des séquences à indexer
     * et vers le premier nœud root

    .. WARNING::
        Dans cette version, les data_ts sont obligatoires pour définir à l'avance les futurs breakpoints

    :param int size_word: le nombre de discrétisation SAX pour chaque séquence
    :param int threshold: la capacité maximale des nœuds feuilles de l'arbre
    :param numpy.ndarray data_ts: array de séquences à insérer
    :param int base_cardinality: la plus petite cardinalité pour l'encodage iSAX
    :param int max_card_alphabet: si self.boolean_card_max == True, cardinalite max pour l'encodage iSAX


    :ivar int size_word: Nombre de lettres contenues dans les mots SAX indexés dans l'arbre
    :ivar int threshold: Seuil avant la séparation d'une feuille en deux nœuds feuilles
    """

    def __init__(self, size_word, threshold, data_ts, base_cardinality=2, max_card_alphabet=128,
                 boolean_card_max=True):
        """
        Fonction d'initialisation de la classe TreeISAX

        :returns: un arbre iSAX
        :rtype: TreeISAX
        """
        
        # Nombre de lettre contenues dans les mots SAX indexés dans l'arbre
        self.size_word = size_word
        # Seuil avant la séparation d'une feuille en deux nœuds feuilles
        self.threshold = threshold
        # cardinalité de chacune des lettres au niveau 1 de l'arbre
        # n'est pas très utile...
        self._base_cardinality = base_cardinality
        # cardinalité max courante
        self.bigger_current_cardinality = base_cardinality
        # Si vrai, défini une cardinalité maximale pour l'alphabet de l'arbre
        self.boolean_card_max = boolean_card_max
        self.max_card_alphabet = max_card_alphabet

        # mean, variance des sequences data_ts
        self.mu, self.sig = norm.fit(data_ts)

        self.min_max = np_empty(shape=(self.size_word, 2))
        for i, dim in enumerate(np_array(data_ts).T.tolist()):
            self.min_max[i][0] = min(dim)-self.mu
            self.min_max[i][1] = max(dim)+self.mu

        self.isax = IndexableSymbolicAggregateApproximation(self.size_word, mean=self.mu, std=self.sig)
        # verif si toutes les valeurs sont correctement indexable
        tmp_max = np_max(abs(np_array(data_ts) - self.mu))
        bkpt_max = abs(self.isax._card_to_bkpt(self.max_card_alphabet)[-1] - self.mu)
        if tmp_max > bkpt_max:
            ratio = tmp_max / bkpt_max
            self.isax = IndexableSymbolicAggregateApproximation(self.size_word, mean=self.mu, std=self.sig*ratio)
        # et on transmet tout cela au nœud root
        self.root = RootNode(tree=self, parent=None, sax=[0]*self.size_word,
                             cardinality=np_array([int(self._base_cardinality / 2)] * self.size_word))
        self.num_nodes = 1

        # attributs pour le pré-traitement
        self._minmax_nodes_computed = False
        self.node_list = None
        self.node_list_leaf = None
        self.node_leaf_ndarray_mean = None

        # attributs pour le calcul *i*\ CFOF
        self._preprocessing_computed = False
        self.min_array = None
        self.max_array = None
        self.min_array_leaf = None
        self.max_array_leaf = None
        self.cdf_mean = None
        self.cdf_std = None

        # attribut après une mise à jour de l'arbre
        self._new_insertion_after_preproc = False
        self._new_insertion_after_minmax_nodes = False

    def insert(self, new_sequence):
        """
        La fonction insert converti en PAA puis appel la fonction insert_paa

        :param numpy.array new_sequence: la nouvelle séquence à insérer
        """
        
        # convert to paa
        paa = self.isax.transform_paa([new_sequence])[0]
        self.insert_paa(paa)

    def insert_paa(self, new_paa):
        """
        La fonction insert qui appelle directement celle de son nœud root

        :param numpy.array new_paa: la nouvelle séquence à insérer
        """

        if len(new_paa) < self.size_word:
            print("Erreur !! "+new_paa+" est plus petit que size.word = "+self.size_word+". FIN")
        else:
            self.root.insert_paa(new_paa)
            self._new_insertion_after_preproc = True
            self._new_insertion_after_minmax_nodes = True

    def preprocessing_for_icfof(self, ntss_tmp, bool_print: bool = False, count_num_node: bool = False):
        """
        Permet de faire appel, pour l'arbre ``id_tree`` aux deux méthodes de pré-traitements:
         * :func:`~pyCFOFiSAX.tree_iSAX.TreeISAX._minmax_obj_vs_node`,
         * :func:`~pyCFOFiSAX.tree_iSAX.TreeISAX.distrib_nn_for_cdf`.

        :param ntss_tmp: les séquences de réference
        :param boolean bool_print: si True, affiche les temps de chaque étape de pré-traitement
        :param boolean count_num_node: si True, compte le nombre de nœuds

        :returns: si ``count_num_node`` True, retourne le nombre de nœuds dans l'arbre
        :rtypes: int
        """

        start_time = time_time()
        self.min_array, self.max_array = self._minmax_obj_vs_node(ntss_tmp, bool_print)
        if bool_print:
            print("_minmax_obj_vs_node --- %s seconds ---" % (time_time() - start_time))
            stdout.flush()

        start_time = time_time()
        self.distrib_nn_for_cdf(ntss_tmp, bool_print)
        if bool_print:
            print("pretrait cdf --- %s seconds ---" % (time_time() - start_time))
            stdout.flush()

        start_time = time_time()
        self._minmax_obj_vs_nodeleaf()
        if bool_print:
            print("pretrait _minmax_obj_vs_node_leaf --- %s seconds ---" % (time_time() - start_time))
            stdout.flush()

        self._preprocessing_computed = True
        if count_num_node:
            return self.num_nodes

    def _minmax_nodes(self):
        """
        Retourne les bornes des nœuds de l'arbre.
        Fait appel à :func:`~pyCFOFiSAX.node.RootNode._do_bkpt`.

        :returns: les bornes min et max des nœuds de l'arbre
        :rtype: numpy.ndarray
        """

        self.get_list_nodes_and_barycentre()

        bkpt_ndarray = np_ndarray(shape=(2, len(self.node_list), self.size_word), dtype=float)
        for num_n, node in enumerate(self.node_list):
            bkpt_ndarray[0][num_n], bkpt_ndarray[1][num_n] = node._do_bkpt()
        return bkpt_ndarray

    def _minmax_obj_vs_node(self, ntss_tmp, bool_print: bool = False):
        """
        Calcule les distances min et max entre les séquences ``ntss_tmp`` et les nœuds de l'arbre.

        :param numpy.ndarray ntss_tmp: les séquences de réference
        :param boolean bool_print: si True, affiche les temps de chaque étape de pré-traitement

        :returns: les distances minimales entre les séquences et les nœuds, les distances maximales entre les séquences et les nœuds
        :rtype: numpy.ndarray, numpy.ndarray
        """

        start_time = time_time()
        bkpt_ndarray = self._minmax_nodes()
        if bool_print:
            print("\t_minmax_nodes --- %s seconds ---" % (time_time() - start_time))
            stdout.flush()
        start_time = time_time()
        row_sax_word, _ = self.isax._row_sax_word_array(ntss_tmp, self.bigger_current_cardinality,
                                                        self.size_word)
        if bool_print:
            print("\t_row_sax_word_array --- %s seconds ---" % (time_time() - start_time))
            stdout.flush()

        node_list_isax = np_array([node.iSAX_word for node in self.node_list])
        node_list_isax = node_list_isax.transpose((1, 0, 2))

        # TODO card_current = tree.base_cardinality 2/2
        # et donc
        # np_log2(node_list_isax[:,:,1]) - np_log2(tree.base_cardinality)
        node_list_isax[:, :, 1] = np_log2(node_list_isax[:, :, 1])
        node_list_isax[:, :, 1] = node_list_isax[:, :, 1].astype('int64')

        bkpt_ndarray = bkpt_ndarray.transpose(0, 2, 1)
        bkpt_ndarray = bkpt_ndarray.reshape(bkpt_ndarray.shape + (1,))

        tmp_length_node_list = len(self.node_list)
        tmp_length_ntss_tmp = len(ntss_tmp)
        min_array_total = np_zeros((tmp_length_node_list, tmp_length_ntss_tmp), dtype=np_float64)
        max_array_total = np_zeros((tmp_length_node_list, tmp_length_ntss_tmp), dtype=np_float64)

        row_sax_word = row_sax_word.transpose((0, 2, 1))

        ntss_tmp = ntss_tmp.transpose()
        ntss_tmp = ntss_tmp.reshape(ntss_tmp.shape + (1,))

        """ TODO np.vectorize ?"""
        for dim in range(self.size_word):

            for line_node, node_isax in enumerate(node_list_isax[dim]):
                dist_bkptmin = cdist([bkpt_ndarray[0][dim][line_node]], ntss_tmp[dim])[0]
                dist_bkptmax = cdist([bkpt_ndarray[1][dim][line_node]], ntss_tmp[dim])[0]

                distmin = np_minimum(dist_bkptmin, dist_bkptmax)
                max_array_total[line_node] += np_square(np_maximum(dist_bkptmin, dist_bkptmax))

                mask = row_sax_word[dim][node_isax[1]] == node_isax[0]
                np_place(distmin, mask, [0])

                min_array_total[line_node] += np_square(distmin)

        return np_sqrt(min_array_total.transpose()), np_sqrt(max_array_total.transpose())

    def distrib_nn_for_cdf(self, ntss_tmp, bool_print: bool = False):
        """
        Calcule les deux indicateurs, moyenne et écart-type des distances, nécessaires pour l'utilisation de la CDF de la loi normale.
        Le calcul de ces indicateurs sont décrit dans `Scoring Message Stream Anomalies in Railway Communication Systems, L.Foulon et al., 2019, ICDMWorkshop <https://ieeexplore.ieee.org/abstract/document/8955558>`_.

        :param numpy.ndarray ntss_tmp: les séquences de réference
        :param boolean bool_print: si True, affiche les stats nœuds sur la sortie standard

        :returns:
        :rtype: list(numpy.ndarray, numpy.array)
        """

        start_time = time_time()
        node_list, node_list_leaf, node_leaf_ndarray_mean = self.get_list_nodes_and_barycentre()
        if bool_print:
            print("pretrait node --- %s seconds ---" % (time_time() - start_time))
            stdout.flush()
            print(len(node_list), " nodes whose ", len(node_list_leaf), " leafs in tree")
            stdout.flush()

        nb_leaf = len(node_list_leaf)

        cdf_mean = np_zeros((nb_leaf, len(ntss_tmp)))
        cdf_std = np_zeros(nb_leaf)
        nb_ts_by_node = np_zeros(nb_leaf, dtype=np_uint32)
        centroid_dist = np_square(cdist(node_leaf_ndarray_mean, ntss_tmp))

        for num, node in enumerate(node_list_leaf):
            cdf_std[node.id_numpy_leaf] = np_mean(node.std)
            nb_ts_by_node[node.id_numpy_leaf] = node.get_nb_sequences()

        dist_list = np_array([np_zeros(i) for i in nb_ts_by_node], dtype=object)

        # calcul distance au carre entre [barycentre et ts] du meme nœud
        """ TODO np.vectorize ?"""
        for node_nn in node_list_leaf:
            dist_list[node_nn.id_numpy_leaf] = cdist([node_nn.mean], node_nn.get_sequences())[0]
        dist_list = np_square(dist_list)

        """ TODO np.vectorize ?"""
        for num, node in enumerate(node_list_leaf):
            node_id = node.id_numpy_leaf

            centroid_dist_tmp = centroid_dist[node_id]
            centroid_dist_tmp = centroid_dist_tmp.reshape(centroid_dist_tmp.shape + (1,))
            centroid_dist_tmp = np_repeat(centroid_dist_tmp, nb_ts_by_node[node_id], axis=1)

            cdf_mean_tmp = np_add(centroid_dist_tmp, dist_list[node_id])
            cdf_mean[node_id] = np_sum(cdf_mean_tmp, axis=1)

        del dist_list
        del cdf_mean_tmp
        del centroid_dist_tmp

        cdf_mean = np_divide(cdf_mean.T, nb_ts_by_node)
        cdf_mean = np_sqrt(cdf_mean)

        self.cdf_mean = cdf_mean
        self.cdf_std = cdf_std

    def _minmax_obj_vs_nodeleaf(self):
        """
        Calcule les distances min et max entre les séquences ``ntss_tmp`` et les nœuds feuilles de l'arbre.

        .. WARNING::
            Attention doit être exécuté après :func:`~pyCFOFiSAX._tree_iSAX.TreeISAX._minmax_obj_vs_node` et :func:`~pyCFOFiSAX._tree_iSAX.TreeISAX.distrib_nn_for_cdf`.
        """
        id_leaf_array = np_zeros(len(self.node_list_leaf), dtype=np_uint32)
        for i_tmp in range(len(self.node_list_leaf)):
            id_leaf_array[i_tmp] = self.node_list_leaf[i_tmp].id_numpy
        self.min_array_leaf = self.min_array[:, id_leaf_array]
        self.max_array_leaf = self.max_array[:, id_leaf_array]

    def get_level_max(self):
        """
        Fonction permettant de retourner le niveau max sachant root niveau 0

        :returns: le niveau de profondeur max
        :rtype: int
        """

        lvl_max = 0
        for _, _, node in RenderTree(self.root):
            if node.level > lvl_max: lvl_max = node.level
        return lvl_max

    def get_width_of_level(self, level: int):
        """
        Fonction permettant de retourner la largeur d'un niveau sachant root niveau 0

        :returns: le nombre de nœud sur le niveau de l'arbre
        :rtype: int
        """

        cmpt = 0
        for _, _, node in RenderTree(self.root):
            if node.level == level: cmpt += 1
        return cmpt

    def get_width_of_all_level(self):
        """
        Fonction permettant de retourner la largeur de tous les niveaux dans une liste sachant root niveau 0

        :returns: le nombre de nœud sur chaque niveau de l'arbre
        :rtype: list
        """

        cmpt = []
        for _, _, node in RenderTree(self.root):
            while node.level > len(cmpt)-1:
                cmpt.append(0)
            cmpt[node.level] += 1
        return cmpt

    def get_nodes_of_level(self, level: int):
        """
        Fonction permettant de retourner les nœuds d'un niveau sachant root niveau 0

        :param int level: le niveau de l'arbre à évaluer

        :returns: les nœuds du level-ième niveau de l'arbre
        :rtype: list
        """
        
        node_list = []
        for _, _, node in RenderTree(self.root):
            if node.level == level:
                node_list.append(node)
        return node_list

    def get_number_internal_and_terminal(self):
        """
        Fonction permettant de retourner le nombre de nœuds feuilles et nœuds internes

        :returns: le nombre de nœuds internes, le nombre de nœuds feuilles
        :rtype: int, int
        """

        cmpt_leaf = 0
        cmpt_internal = 0
        for _, _, node in RenderTree(self.root):
            if node.terminal:
                cmpt_leaf += 1
            else:
                cmpt_internal += 1
        return cmpt_internal, cmpt_leaf

    def count_nodes_by_tree(self):
        """
        La fonction count_nodes_by_tree retourne le nombre de nœuds et de nœuds feuilles de l'arbre.
        Fait appel à :func:`~pyCFOFiSAX.tree_iSAX.TreeISAX.get_number_internal_and_terminal`.

        :returns: le nombre de nœuds internes, le nombre de nœuds feuilles
        :rtype: int, int
        """

        cmpt_internal, cmpt_leaf = self.get_number_internal_and_terminal()
        cmpt_node = cmpt_leaf + cmpt_internal
        return cmpt_node, cmpt_leaf

    def get_list_nodes_and_barycentre(self):
        """
        Retourne listes des nœuds et barycentres

        :returns: liste des nœuds, liste des nœuds feuilles, liste des barycentres des feuilles
        :rtype: list, list, list
        """

        if self._minmax_nodes_computed and not self._new_insertion_after_minmax_nodes:
            return self.node_list, self.node_list_leaf, self.node_leaf_ndarray_mean

        self.node_list = []
        self.node_list_leaf = []
        node_leaf_mean = []
        cmpt_node = 0
        count_leaf = 0

        for _, _, node in RenderTree(self.root):
            self. node_list.append(node)
            node.id_numpy = cmpt_node
            cmpt_node += 1
            if node.is_leaf:
                self.node_list_leaf.append(node)
                node_leaf_mean.append(node.mean)
                node.id_numpy_leaf = count_leaf
                count_leaf += 1

        if cmpt_node != self.num_nodes:
            print("ATTENTION !!!")
            exit(-1)
        self.node_leaf_ndarray_mean = np_array(node_leaf_mean)
        self._minmax_nodes_computed = True
        self._new_insertion_after_minmax_nodes = False
        return self.node_list, self.node_list_leaf, self.node_leaf_ndarray_mean

    def get_list_nodes_leaf(self):
        """
        Retourne liste des nœuds feuilles

        :returns: liste des nœuds feuilles
        :rtype: list
        """
        self.get_list_nodes_and_barycentre()
        return self.node_list_leaf

    def get_size(self):
        """
        Fonction permettant de retourner la taille mémoire de l'arbre, des nœuds
        et des séquences contenues dans l'arbre

        :returns: taille mémoire totale, taille mémoire des nœuds, taille mémoire des sequences
        :rtype: int, int, int
        """

        total_size = 0
        ts_size = 0
        node_size = 0
        total_size += getsizeof(self)
        for _, _, node in RenderTree(self.root):
            total_size += getsizeof(node)
            node_size += getsizeof(node)
            if node.terminal:
                for ts in node.get_sequences():
                    total_size += getsizeof(ts)
                    ts_size += getsizeof(ts)
        return total_size, node_size, ts_size

    def get_size_and_width_and_number_types_nodes(self):
        """
        Fonction regroupant :
            -  get_size()
            -  get_width_of_all_level()
            -  get_number_internal_and_terminal()

        :returns: taille mémoire totale, taille mémoire des nœuds, taille mémoire des sequences,
            le nombre de nœuds sur chaque niveau, le nombre de nœuds internes, le nombre de nœuds feuilles,
            le nombre de séquence insérées dans l'arbre
        :rtype: int, int, int, list, int, int, int
        """

        total_size = 0
        node_size = 0
        ts_size = 0

        cmpt_nodes = []

        number_terminal = 0
        number_internal = 0

        cmpt_sequences = []

        total_size += getsizeof(self)
        total_size += getsizeof(self.root)
        for _, _, node in RenderTree(self.root):
            while node.level > len(cmpt_nodes)-1:
                cmpt_nodes.append(0)
                cmpt_sequences.append(0)
            cmpt_nodes[node.level] += 1
            if node.terminal:
                number_terminal += 1
                cmpt_sequences[node.level] += len(node.get_sequences())
            else:
                number_internal += 1
            total_size += getsizeof(node)
            node_size += getsizeof(node)
            if node.terminal:
                for ts in node.get_sequences():
                    total_size += getsizeof(ts)
                    ts_size += getsizeof(ts)
        return total_size, node_size, ts_size, cmpt_nodes, number_internal, number_terminal, cmpt_sequences

    def vrang_list_faster(self, sub_query: np_array, ntss_tmp: np_ndarray):
        """
        Obtenir la liste des vrang pour la séquence ``sub_query`` dans l'arbre.
        Nécessaire pour le calcul de l'approximation.
        Cette méthode est la version rapide de la méthode :func:`~pyCFOFiSAX._tree_iSAX.TreeISAX.vrang_list`.

        .. note::
            Cette méthode ne parcourt pas l'arbre, mais élague directement les nœuds feuilles.
            Les feuilles conservées (non élaguées) seront utilisées par la fonction d'approximation.

        :param sub_query: la séquence à évaluer
        :param ntss_tmp: les séquences de réference (ie. l'historique de réference) au format PAA

        :returns: la liste de vrang de ``sub_query``
        :rtype: list(float)
        """

        num_ts_by_node = []
        for i, node in enumerate(self.get_list_nodes_leaf()):
            num_ts_by_node.append(node.get_nb_sequences())
        num_ts_by_node = np_array(num_ts_by_node)

        if not hasattr(self, 'index_cdf_bin'):
            self.index_cdf_bin = np_linspace(-4.0, 4.0, num=1000)
        if not hasattr(self, 'cdf_bins'):
            self.cdf_bins = scipy_norm.cdf(self.index_cdf_bin, 0, 1)

        q_paa = self.isax.transform_paa([sub_query])[0]

        distance_q_p = cdist([q_paa.reshape(q_paa.shape[:-1])], ntss_tmp)[0]

        return vrang_list_for_all_seq_ref(len(ntss_tmp), distance_q_p,
                                          self.max_array_leaf, self.min_array_leaf,
                                          self.cdf_mean, self.cdf_std,
                                          num_ts_by_node,
                                          self.index_cdf_bin, self.cdf_bins)

    def vrang_list(self, sub_query: np_array, ntss_tmp: np_ndarray):
        """
        Obtenir la liste des vrang pour la séquence ``sub_query`` dans l'arbre.
        Nécessaire pour le calcul de l'approximation.
        La même méthode plus rapide mais sans le parcours d'arbre :
        :func:`~pyCFOFiSAX._tree_iSAX.TreeISAX.vrang_list_faster`.

        :param sub_query: la séquence à évaluer
        :param ntss_tmp: les séquences de réference (ie. l'historique de réference)

        :returns: la liste de vrang de ``sub_query``
        :rtype: list(float)
        """

        if not hasattr(self, 'index_cdf_bin'):
            self.index_cdf_bin = np_linspace(-4.0, 4.0, num=1000)
        if not hasattr(self, 'cdf_bins'):
            self.cdf_bins = scipy_norm.cdf(self.index_cdf_bin, 0, 1)

        # liste des vrang
        # TODO np_array
        k_list_result = []

        q_paa = self.isax.transform_paa([sub_query])[0]

        distance_q_p = cdist([q_paa.reshape(q_paa.shape[:-1])], ntss_tmp)[0]

        # pour tout objet p
        for stop_ite, p_paa in enumerate(ntss_tmp):

            p_name = stop_ite

            # nombres d'objets trop près
            count_ts_too_nn = 0

            # Distance reelle
            distance = distance_q_p[stop_ite]

            max_array_p_bool = self.max_array[p_name] < distance
            min_array_p_bool = self.min_array[p_name] <= distance

            nodes_list_fifo = []
            nodes_list_fifo.extend(self.root.nodes)

            while nodes_list_fifo:

                node_nn = nodes_list_fifo.pop(0)

                if max_array_p_bool[node_nn.id_numpy]:

                    count_ts_too_nn += node_nn.get_nb_sequences()

                elif node_nn.terminal:

                    if min_array_p_bool[node_nn.id_numpy]:

                        cdf_mean_tmp = self.cdf_mean[p_name][node_nn.id_numpy_leaf]
                        cdf_std_tmp = self.cdf_std[node_nn.id_numpy_leaf]

                        if cdf_std_tmp > 0.0:
                            distance_normalized = (distance - cdf_mean_tmp) / cdf_std_tmp
                            if distance_normalized > 4:
                                count_ts_too_nn += node_nn.get_nb_sequences()
                            elif -4 <= distance_normalized:
                                index_for_bin = bisect_bisect(self.index_cdf_bin, distance_normalized)
                                count_ts_too_nn += self.cdf_bins[index_for_bin] * node_nn.get_nb_sequences()
                        else:
                            if distance > cdf_mean_tmp:
                                count_ts_too_nn += node_nn.get_nb_sequences()

                elif min_array_p_bool[node_nn.id_numpy]:
                    nodes_list_fifo.extend(node_nn.nodes)

            # On sauvegarde l'estimation de la position du centroid de query par rapport a p
            k_list_result.append(count_ts_too_nn)

        return k_list_result

    def number_nodes_visited(self, sub_query: np_array, ntss_tmp: np_ndarray):
        """
        Compte le nombre de nœuds visités moyen dans l'arbre pour le calcul de l'approximation.

        :param numpy.array sub_query: la séquence à évaluer
        :param numpy.ndarray ntss_tmp: les séquences de réference

        :returns: retourne le nombre de nœuds visités dans l'arbre pour l'approximation *i*\ CFOF
        :rtype: numpy.array
        """
        q_paa = self.isax.transform_paa([sub_query])[0]
        ntss_tmp_paa = self.isax.transform_paa(ntss_tmp)

        distance_q_p = cdist([q_paa.reshape(q_paa.shape[:-1])],
                             ntss_tmp_paa.reshape(ntss_tmp_paa.shape[:-1]))[0]

        list_parent_node = np_zeros(len(self.node_list), dtype=np_uint32)

        for tmp_node in self.node_list:
            if tmp_node.id_numpy == 0:
                continue
            list_parent_node[tmp_node.id_numpy] = tmp_node.parent.id_numpy

        count_visited_nodes_list = nodes_visited_for_all_seq_ref(len(ntss_tmp_paa), distance_q_p,
                                                                 self.max_array, self.min_array,
                                                                 list_parent_node)

        return self.num_nodes, count_visited_nodes_list.mean()
