# Authors: Lucas Foulon <lucas.foulon@gmail.com>
# License: BSD 3 clause

from numpy import array as np_array
from numpy import ndarray as np_ndarray
from numpy import zeros as np_zeros

from pyCFOFiSAX._tree_iSAX import TreeISAX
from tslearn.piecewise import PiecewiseAggregateApproximation


class ForestISAX:
    """
    La classe ForestISAX contenant un ou plusieurs arbres et les fonctions de prétraitement sur les données contenues
    dans ces arbres

    :param int size_word: la taille des mots SAX
    :param int threshold: le seuil maximal des nœuds
    :param numpy.ndarray data_ts: les séquences à insérer pour en extraire les stats
    :param int base_cardinality: la cardinalité la plus petite pour l'encodage *i*\ SAX
    :param int number_tree: le nombre d'arbres TreeISAX dans la forêt
    :param list indices_partition: une liste de liste d'indices où, pour chaque arbre, précise les indices des
    séquences à insérer
    :param int max_card_alphabet: si ``boolean_card_max == True``, la cardinalité maximale de l'encodage *i*\ SAX dans
    chacun des arbres
    :param boolean boolean_card_max: si ``== True``, définit une cardinalité maximale pour l'encodage *i*\ SAX des
    séquences dans chacun des arbres

    :ivar list length_partition: la longueur des mots SAX dans chacun des arbres (``== [size_word]`` si ``number_tree
    == 1``)
    """

    def __init__(self, size_word: int, threshold: int, data_ts: np_ndarray, base_cardinality: int = 2,
                 number_tree: int = 1,
                 indices_partition: list = None,
                 max_card_alphabet: int = 128, boolean_card_max: bool = True):
        """
        Fonction d'initialisation de la classe TreeISAX

        :returns: une forêt pointant vers un ou plusieurs arbres iSAX
        :rtype: ForestISAX
        """

        # nombre de lettre contenu dans le mot SAX
        self.size_word = size_word
        # seuil ou le nœud split
        self.threshold = threshold
        # cardinalité de chacune des lettres au niveau 1 de l'arbre
        self.base_cardinality = base_cardinality
        # cardinalite max
        self.max_cardinality = base_cardinality

        self._paa = PiecewiseAggregateApproximation(self.size_word)

        self.forest = {}
        self.number_tree = number_tree

        self.indices_partition = indices_partition

        self._init_trees(data_ts, max_card_alphabet, boolean_card_max)

    def _init_trees(self, data_ts: np_ndarray, max_card_alphabet: int, boolean_card_max: bool):
        """
        Fonction qui initialise le(s) arbre(s) lors de la création d'un objet ForestISAX

        :param numpy.ndarray data_ts: les séquences à insérer pour en extraire les stats
        :param int max_card_alphabet: si ``boolean_card_max == True``, la cardinalité maximale de l'encodage *i*\ SAX
        dans chacun des arbres
        :param boolean boolean_card_max: si ``== True``, définit une cardinalité maximale pour l'encodage *i*\ SAX
        des séquences dans chacun des arbres
        """

        if self.number_tree == 1:
            """ si il n'y a qu'un seul arbre"""

            self.forest[0] = TreeISAX(
                size_word=self.size_word,
                threshold=self.threshold, data_ts=data_ts,
                base_cardinality=self.base_cardinality,
                max_card_alphabet=max_card_alphabet,
                boolean_card_max=boolean_card_max
            )
            self.length_partition = [self.size_word]
            self.indices_partition = [list(range(self.size_word))]

        elif self.indices_partition is None:
            """ si il n'y a pas qu'un seul arbre et que les indices ne sont pas défini """

            self.length_partition = [int(self.size_word / self.number_tree)] * self.number_tree
            for reste in range(self.size_word - sum(self.length_partition)):
                self.length_partition[reste] += 1

            self.indices_partition = []

            for i in range(self.number_tree):
                self.forest[i] = TreeISAX(
                    size_word=self.length_partition[i],
                    threshold=self.threshold,
                    data_ts=data_ts[:, i:self.size_word:self.number_tree],
                    base_cardinality=2,
                    max_card_alphabet=max_card_alphabet,
                    boolean_card_max=boolean_card_max
                )
                self.indices_partition.append(list(range(i, self.size_word, self.number_tree)))

        else:
            # liste du nombre de lettre dans chaque arbre
            self.length_partition = []
            for part_tmp in self.indices_partition:
                self.length_partition.append(len(part_tmp))

            for i in range(self.number_tree):
                self.forest[i] = TreeISAX(
                    size_word=self.length_partition[i],
                    threshold=self.threshold,
                    data_ts=data_ts[:, self.indices_partition[i]],
                    base_cardinality=2,
                    max_card_alphabet=max_card_alphabet,
                    boolean_card_max=boolean_card_max
                )

    def index_data(self, new_sequences: np_ndarray):
        """
        La fonction index_data permet d'insérer un grand nombre de séquences

        :param numpy.ndarray new_sequences: les séquences à insérer

        :returns: le nombre de séquences (sous-séquences) insérer dans l'arbre (dans les arbres)
        :rtype: numpy.array
        """

        # conversion des ts en paa
        if new_sequences.shape[-1] > 1:
            # add dim to avoid tslearn warning
            new_sequences = new_sequences.reshape(new_sequences.shape + (1,))
        npaa = self._paa.fit_transform(new_sequences)

        # pour compter le nombre d'objets inserés dans chaque arbre
        cmpt_insert = np_zeros(shape=self.number_tree)

        for i, tree in self.forest.items():
            # récupère les indices de l'arbre, dans le cas multi-arbre
            npaa_tmp = npaa[:, self.indices_partition[i]]
            npaa_tmp = npaa_tmp.reshape(npaa_tmp.shape[:-1])

            for npa_tp in npaa_tmp:
                tree.insert_paa(npa_tp)
                cmpt_insert[i] += 1

        # retourne array[indice de l'arbre] avec le nombre d'objets inseres pour chaque arbre
        return cmpt_insert

    def _count_nodes(self, id_tree: int):
        """
        La fonction _count_nodes retourne le nombre de nœuds et de nœuds feuilles pour un arbre donné.
        Fait appel à :func:`~pyCFOFiSAX.tree_iSAX.TreeISAX.count_nodes_by_tree`.

        :param int id_tree: l'id de l'arbre à analyser

        :returns: le nombre de nœuds internes, le nombre de nœuds feuilles
        :rtype: int, int
        """

        tree = self.forest[id_tree]
        return tree.count_nodes_by_tree()

    def list_nodes(self, id_tree: int, bool_print: bool = False):
        """
        Retourne listes des nœuds et barycentres de l'arbre id_tree. Affiche les statistiques sur la sortie standard
        si ``bool_print == True``
        Fait appel à :func:`~pyCFOFiSAX.tree_iSAX.TreeISAX.get_list_nodes_and_barycentre`.

        :param int id_tree: l'id de l'arbre à analyser
        :param boolean bool_print: affiche les stats nœuds sur la sortie standard

        :returns: la liste des nœuds, la liste des nœuds internes, la liste des barycentres
        :rtype: list, list, list
        """

        tree = self.forest[id_tree]
        node_list, node_list_leaf, node_leaf_ndarray_mean = tree.get_list_nodes_and_barycentre()
        if bool_print:
            print(f"{len(node_list)} nodes whose {len(node_list_leaf)} leafs in tree {id_tree}")

        return node_list, node_list_leaf, node_leaf_ndarray_mean

    def preprocessing_forest_for_icfof(self, ntss: np_ndarray, bool_print: bool = False, count_num_node: bool = False):
        """
        Permet de faire appel, pour l'arbre ``id_tree`` au pré-traitement pour le calcul *i*\ CFOF

        :param ntss: les séquences de réference
        :param boolean bool_print: si True, affiche les temps de chaque étape de pré-traitement
        :param boolean count_num_node: si True, compte le nombre de nœuds

        :returns: si count_num_node, retourne le nombre de nœuds contenus dans chaque arbre
        :rtype: numpy.array
        """

        total_num_node = np_zeros(self.number_tree)
        for id_tree, tmp_tree in self.forest.items():
            ntss_tmp = ntss[:, self.indices_partition[id_tree]]
            total_num_node[id_tree] = tmp_tree.preprocessing_for_icfof(ntss_tmp,
                                                                       bool_print=bool_print,
                                                                       count_num_node=count_num_node)

        if count_num_node:
            return total_num_node

    def number_nodes_visited(self, query: np_array, ntss: np_ndarray):
        """
        Compte le nombre de nœuds visités moyen dans chaque arbre pour le calcul de l'approximation.

        :param numpy.array query: la séquence à évaluer
        :param numpy.ndarray ntss: les séquences de réference

        :returns: retourne le nombre de nœuds visités dans chaque arbre pour l'approximation *i*\ CFOF
        :rtype: numpy.array
        """

        total_num_node = np_zeros(self.number_tree*2)

        for id_tree, tmp_tree in self.forest.items():

            sub_query = query[self.indices_partition[id_tree]]
            ntss_tmp = np_array(ntss)[:, self.indices_partition[id_tree]]

            total_num_node[id_tree], total_num_node[self.number_tree + id_tree] = \
                tmp_tree.number_nodes_visited(sub_query, ntss_tmp)

        return total_num_node
