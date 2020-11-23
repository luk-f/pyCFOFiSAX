# Authors: Lucas Foulon <lucas.foulon@gmail.com>
# License: BSD 3 clause

from anytree import Node
from numpy import all as np_all
from numpy import copy as np_copy
from numpy import mean as np_mean
from numpy import std as np_std
from numpy import sqrt as np_sqrt
from numpy import zeros as np_zeros
from numpy import array as np_array
from numpy import empty as np_empty
from numpy import argmin as np_argmin

""" Module regroupant les trois types de nœuds utilisés par l'arbre iSAX """


class RootNode(Node):
    """
    La classe RootNode crée l'unique nœud de l'arbre ancêtre commun à tous les autres nœuds

    :param tree_iSAX tree: l'arbre dans lequel le nœud est contenu
    :param Node parent: le nœud parent direct
    :param numpy.array sax: les valeurs SAX du nœud
    :param numpy.array cardinality: les cardinalites des valeurs SAX
    """

    #: Attribut permettant de définir un id pour chaque nœud
    id_global = 0

    def __init__(self, tree, parent, sax, cardinality):
        """
        Fonction d'initialisation de la classe RootNode

        :returns: un nœud root
        :rtype: RootNode
        """

        self.iSAX_word = np_array([sax, cardinality]).T

        Node.__init__(self, parent=parent, name=str(self.iSAX_word))

        self.tree = tree
        self.sax = sax
        self.cardinality = cardinality  

        self.cardinality_next = np_copy(self.cardinality)
        self.cardinality_next = np_array([x*2 for x in self.cardinality_next])

        # Nombre de séquences contenues dans le nœud (ou par ses fils)
        self.nb_sequences = 0

        """ La partie calcul incrémental pour CFOF """
        self.mean = np_empty(shape=self.tree.size_word)
        # Permet le calcul incrémental de self.mean
        self.sum = np_empty(shape=self.tree.size_word)

        self.std = np_empty(shape=self.tree.size_word)
        # Permet le calcul incrémental de self.std
        self.sn = np_empty(shape=self.tree.size_word)

        # Spécifique aux nœuds internes
        self.nodes = []
        self.key_nodes = {}

        self.terminal = False
        self.level = 0

        self.id = RootNode.id_global
        RootNode.id_global += 1

    def insert_paa(self, new_paa):
        """
        La fonction insert_paa(new_paa) permettant d'insérer une nouvelle séquence convertie en PAA

        :param new_paa: la séquence convertie en PAA à insérer
        """

        i_sax_word = self.tree.isax.transform_paa_to_isax(new_paa, self.cardinality_next)[0]
        # pour i_sax_word, on retourne le premier élement de chaque tuple et on test si le mot apparaît dans les nœuds
        if str([i[0] for i in i_sax_word]) in self.key_nodes:
            # on récupère le nœud qui colle au mot
            current_node = self.key_nodes[str([i[0] for i in i_sax_word])]

            # Si c'est une feuille
            if current_node.terminal:
                # et que l'on ne dépasse pas le seuil max ou que le nœud feuille ne soit plus splitable
                # nb : cette seconde condition n'est pas proposée par shieh et kheogh
                if current_node.nb_sequences < self.tree.threshold or not current_node.splitable:
                    current_node.insert_paa(new_paa)
                # mais sinon (on dépasse le seuil max et la feuille est splitable)
                else:
                    # création du nouveau nœud interne
                    new_node = InternalNode(self.tree, current_node.parent, np_copy(current_node.sax),
                                            np_copy(current_node.cardinality), current_node.sequences)
                    # on insert la nouvelle séquence dans ce nouveau nœud interne
                    new_node.insert_paa(new_paa)
                    # pour chacune des séquences de la feuille courante on insert ses séquences dans le nouveau nœud interne
                    # ce nœud interne va créer une ou plusieurs feuilles pour insérer ces séquences
                    for ts in current_node.sequences:
                        new_node.insert_paa(ts)
                    # et on supprime la feuille courante de la liste des nœuds
                    self.nodes.remove(current_node)
                    # que l'on retire également du dict
                    del self.key_nodes[str(current_node.sax)]
                    # et l'on rajoute au dict le nouveau nœud interne
                    self.key_nodes[str(current_node.sax)] = new_node
                    self.nodes.append(new_node)
                    current_node.parent = None
                    # et on supprime définitivement la feuille courante
                    del current_node

            # sinon (ce n'est pas une feuille) on continue le parcours de l'arbre
            else:
                current_node.insert_paa(new_paa)

        # sinon (le nœud sax n'existe pas) on crée une nouvelle feuille
        else:
            new_node = TerminalNode(self.tree, self, [i[0] for i in i_sax_word], np_array(self.cardinality_next))
            new_node.insert_paa(new_paa)
            self.key_nodes[str([i[0] for i in i_sax_word])] = new_node
            self.nodes.append(new_node)
            self.tree.num_nodes += 1

        # maj des indicateurs du nœud
        self.nb_sequences += 1
        # calcul mean et std
        if self.nb_sequences == 1:
            self.sum = np_copy(new_paa)
            self.mean = np_copy(new_paa)
            self.std = np_zeros(self.tree.size_word)
            self.sn = np_zeros(self.tree.size_word)
        else:
            mean_moins_1 = np_copy(self.mean)
            self.sum += new_paa
            self.mean = self.sum / self.nb_sequences
            self.sn += (new_paa - mean_moins_1) * (new_paa - self.mean)
            self.std = np_sqrt(self.sn / self.nb_sequences)

    def _do_bkpt(self):
        """
        La fonction _do_bkpt calcule les bornes min et max du nœud sur chaque dimension du nœud.

        :returns: une array contenant les bornes min et une contenant les bornes max
        :rtype: numpy.array, numpy.array
        """

        bkpt_list_min = np_empty(self.tree.size_word)
        bkpt_list_max = np_empty(self.tree.size_word)
        for i, iSAX_letter in enumerate(self.iSAX_word):
            bkpt_tmp = self.tree.isax._card_to_bkpt(iSAX_letter[1])
            # le cas où il n'y a pas de bkpt (nœud root)
            if iSAX_letter[1] < 2:
                bkpt_list_min[i] = self.tree.min_max[i][0]
                bkpt_list_max[i] = self.tree.min_max[i][1]
            # le cas où il n'y a pas de bkpt inf
            elif iSAX_letter[0] == 0:
                bkpt_list_min[i] = self.tree.min_max[i][0]
                bkpt_list_max[i] = bkpt_tmp[iSAX_letter[0]]
            # le cas où il n'y a pas de bkpt sup
            elif iSAX_letter[0] == iSAX_letter[1]-1:
                bkpt_list_min[i] = bkpt_tmp[iSAX_letter[0]-1]
                bkpt_list_max[i] = self.tree.min_max[i][1]
            # le cas général
            else:
                bkpt_list_min[i] = bkpt_tmp[iSAX_letter[0]-1]
                bkpt_list_max[i] = bkpt_tmp[iSAX_letter[0]]

        return bkpt_list_min, bkpt_list_max

    def get_sequences(self):
        """
        Retourne les séquences contenues dans le nœud (cas feuille seulement) ou ses descendants

        :returns: les séquences
        :rtype: numpy.ndarray
        """
        sequences = []
        for node in self.nodes:
            for ts in node.get_sequences():
                sequences.append(ts)
        return sequences

    def get_nb_sequences(self) -> int:
        """
        Retourne le nombre de séquences contenues dans le nœud et ses descendants

        :returns: le nombre de séquences du sous-arbre
        :rtype: int
        """
        return self.nb_sequences

    def __str__(self):
        """
        Définition de la fonction d'affichage pour le nœud

        :returns: les infos à afficher
        :rtype: str
        """

        str_print = "RootNode\n\tiSAX : " + str(self.iSAX_word) + "\n\tcardinalité : " + str(self.cardinality) + \
                    "\n\tcardinalité suiv : " + str(self.cardinality_next) + "\n\tnbr nœud fils : " + \
                    str(len(self.nodes))
        return str_print


class InternalNode(RootNode):
    """
    La classe InternalNode crée les nœuds interne ayant au moins un descendant direct, et un seul ascendant direct

    :param tree_iSAX tree: l'arbre dans lequel le nœud est contenu
    :param Node parent: le nœud parent direct
    :param list sax: les valeurs SAX du nœud
    :param numpy.array cardinality: les cardinalités des valeurs SAX
    :param numpy.ndarray sequences: les séquences à insérer dans ce nœud
    """

    def __init__(self, tree, parent, sax, cardinality, sequences):
        """
        Fonction d'initialisation de la classe InternalNode

        :returns: un nœud root
        :rtype: RootNode
        """

        """ hérite de la fonction d'init de la classe RootNode """
        RootNode.__init__(self, tree=tree, parent=parent,
                          sax=sax, cardinality=cardinality)

        """ transforme les séquences en liste de PAA"""
        list_ts_paa = self.tree.isax.transform_paa(sequences)
        tmp_mean = np_mean(list_ts_paa, axis=0)
        tmp_stdev = np_std(list_ts_paa, axis=0)

        """ comme c'est un nœud interne, il a forcement au moins un nœud descendant donc : """
        """ on calcule les futurs cardinalités candidates """
        cardinality_next_tmp = np_copy(self.cardinality)
        # si max_card
        if self.tree.boolean_card_max:
            # on multiplie par 2 seulement les cardinalités ne dépassant pas le seuil autorisé
            cardinality_next_tmp[cardinality_next_tmp <= self.tree.max_card_alphabet] *= 2
        else:
            # on multiplie par 2 toutes les cardinalités (ils sont tous candidats)
            cardinality_next_tmp *= 2
        # la fonction self.split choisi l'indice de la cardinalité à multiplier par 2
        position_min = self.split(cardinality_next_tmp, tmp_mean, tmp_stdev)

        """ on écrit la prochaine cardinalité (pour ses nœuds feuilles) """
        self.cardinality_next = np_copy(self.cardinality)
        self.cardinality_next[position_min] *= 2
        if self.tree.bigger_current_cardinality < self.cardinality_next[position_min]:
            self.tree.bigger_current_cardinality = self.cardinality_next[position_min]

        self.level = parent.level + 1

    def split(self, next_cardinality, mean, stdev):
        """
        Calcule la prochaine cardinalité à spliter en deux

        :param numpy.array next_cardinality: la liste des prochaines cardinalités
        :param numpy.array mean: la liste des moyennes de répartition des valeurs des séquences sur chaque dimension
        :param numpy.array stdev: la liste des écarts-types de répartition des valeurs des séquences sur chaque dimension
        """

        # segment_to_split : idem notation iSAX 2.0 (A Camerra, T Palpanas, J Shieh et al. - 2010)
        segment_to_split = None
        seg_to_spli_dist = float('inf')

        """ on parcourt les bkpt obtenus pour chaque dim et on cherche la dimension qui sépare le mieux nos sequences
        selon nos critères """
        # liste des bkpt pour la cardinalité choisie (ie la plus petite, cf fonction init de InternalNode)
        bkpt_list = [self.tree.isax._card_to_bkpt_only(next_c) for next_c in next_cardinality]
        for i in range(self.tree.size_word):
            # test si on ne dépasse pas la card max
            if next_cardinality[i] <= self.tree.max_card_alphabet and self.tree.boolean_card_max:
                # ici breakpoint le plus proche à la moyenne des valeurs de la i^ieme dimension
                nearest_bkpt = min(bkpt_list[i], key=lambda x: abs(x-mean[i]))
                """ 1er critère : si l'ecart-type des valeurs de la i^ieme dimension n'est pas nul """
                if stdev[i] != 0:
                    """ 2nd critère : pour tre la meilleure candidate, la distance entre bkpt et barycentre 
                    est divisée par l'écart-type """
                    if (abs((nearest_bkpt - mean[i]) / stdev[i])) < seg_to_spli_dist:
                        segment_to_split = i
                        seg_to_spli_dist = abs((nearest_bkpt - mean[i]) / stdev[i])

        """ attention, si aucun candidat, choisir la plus petite cardinalité """
        if segment_to_split is None:
            segment_to_split = np_argmin(self.cardinality)
        return segment_to_split

    def __str__(self) -> str:
        """
        Définition de la fonction d'affichage pour le nœud

        :returns: les infos à afficher
        :rtype: str
        """

        str_print = "InternalNode\n\tiSAX : " + str(self.name) + "\n\tparent iSAX : " + str(self.parent.name) + \
                    "\n\tcardinalité : " + str(self.cardinality) + "\n\tcardinalité suiv : " + \
                    str(self.cardinality_next) + "\n\tnbr nœud fils : " + str(len(self.nodes))
        return str_print


class TerminalNode(RootNode):
    """
    La classe TerminalNode crée les nœuds feuilles ayant aucun descendant, et un seul ascendant direct

    :param tree_iSAX tree: l'arbre dans lequel le nœud est contenu
    :param Node parent: le nœud parent direct
    :param list sax: les valeurs SAX du nœud
    :param numpy.array cardinality: les cardinalités des valeurs SAX
    """

    def __init__(self, tree, parent, sax, cardinality):
        """
        Fonction d'initialisation de la classe TerminalNode

        :returns: un nœud root
        :rtype: RootNode
        """

        RootNode.__init__(self, tree=tree, parent=parent,
                          sax=sax, cardinality=cardinality)

        del self.cardinality_next

        """ partie spécifique aux nœuds terminaux
        (quoi? on dit des nœuds terminals?) """
        # variable pour les bkpt (non incrémental)
        self.bkpt_min, self.bkpt_max = np_array([]), np_array([])

        self.terminal = True
        if parent is None:
            self.level = 0
        else:
            self.level = parent.level + 1

        self.splitable = True
        if np_all(np_array(self.cardinality) >= self.tree.max_card_alphabet) and self.tree.boolean_card_max:
            self.splitable = False

        """ Important, la liste des sequences PAA que la feuille contient"""
        self.sequences = []

    def insert_paa(self, ts_paa):
        """
        Fonction qui insert une nouvelle séquence au format PAA

        :param ts_paa: la nouvelle séquence PAA
        """

        self.sequences.append(ts_paa)
        """ maj des indicateurs """
        self.nb_sequences += 1
        # calcul mean et std
        if self.nb_sequences == 1:
            self.sum = np_copy(ts_paa)
            self.mean = np_copy(ts_paa)
            self.std = np_zeros(self.tree.size_word)
            self.sn = np_zeros(self.tree.size_word)
        else:
            mean_moins_1 = np_copy(self.mean)
            self.sum += ts_paa
            self.mean = self.sum / self.nb_sequences
            self.sn += (ts_paa - mean_moins_1) * (ts_paa - self.mean)
            self.std = np_sqrt(self.sn / self.nb_sequences)

    def get_sequences(self):
        """
        Retourne les séquences contenues dans le nœud

        :returns: les séquences contenues dans le nœud
        :rtype: list
        """
        return self.sequences

    def __str__(self) -> str:
        """
        Définition de la fonction d'affichage pour le nœud

        :returns: les infos à afficher
        :rtype: str
        """
        str_print = "TerminalNode\n\tiSAX : " + str(self.name) + "\n\tparent iSAX : " + str(self.parent.name) + \
                    "\n\tcardinalité : " + str(self.cardinality) + "\n\tcardinalité suiv : " + \
                    str(self.cardinality_next) + "\n\tnbr sequences : " + str(self.nb_sequences)
        return str_print
