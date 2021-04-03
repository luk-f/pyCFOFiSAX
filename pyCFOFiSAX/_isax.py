# Authors: Lucas Foulon <lucas.foulon@gmail.com>
# License: BSD 3 clause

from numpy import int as np_int
from numpy import array as np_array
from numpy import zeros as np_zeros

from math import log as math_log

from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise.piecewise import _paa_to_symbols, _breakpoints, _bin_medians
from tslearn.utils import to_time_series_dataset


class IndexableSymbolicAggregateApproximation(PiecewiseAggregateApproximation):
    """Indexable Symbolic Aggregate approXimation (iSAX) transformation.

    First presented by J. Shieh & E. Keogh in *i*\ SAX: Indexing and Mining Terabyte Sized Time Series.
    Classe qui hérite de la classe ``PiecewiseAggregateApproximation`` proposée `par Romain Tavenard dans \`\`tslearn\`\` disponible ici <https://tslearn.readthedocs.io/en/stable/>`_.

    :param int n_segments: le nombre de lettre dans le mot SAX
    :param int alphabet_size_min: la taille minimum de l'alphabet SAX à l'initialisation (2 par défaut)
    :param float mean: la moyenne de la distribution des séquences à encoder (0.0 par défaut)
    :param float std: l'écart-type de la distribution des séquences à encoder (0.0 par défaut)
    """

    def __init__(self, n_segments, alphabet_size_min=2, mean=0.0, std=1.0):
        """
        Fonction d'initialisation de la classe IndexableSymbolicAggregateApproximation

        :returns: une classe d'encodage *i*\ SAX
        :rtype: IndexableSymbolicAggregateApproximation
        """

        PiecewiseAggregateApproximation.__init__(self, n_segments)
        self.n_segments = n_segments
        self.alphabet_size_min = alphabet_size_min
        self.alphabet_size_max = alphabet_size_min

        self.mean = mean
        self.std = std

        self.card_to_bkpt_ = dict()
        self.card_to_bkpt_only_ = dict()
        self.card_to_bkpt_middle_ = dict()
        self.card_to_bkpt_[self.alphabet_size_min] = _breakpoints(self.alphabet_size_min, scale=self.std) + self.mean
        self.card_to_bkpt_only_[self.alphabet_size_min] = _breakpoints(self.alphabet_size_min,
                                                                       scale=self.std) + self.mean
        self.card_to_bkpt_middle_[self.alphabet_size_min] = _bin_medians(self.alphabet_size_min,
                                                                         scale=self.std) + self.mean

    def fit(self, X):
        """
        Prépare les données pour l'encodage *i*\ SAX selon ``PiecewiseAggregateApproximation``

        :returns: données prêtes pour l'encodage, défini par ``tslearn``
        :rtype: numpy.ndarray of PiecewiseAggregateApproximation
        """

        return PiecewiseAggregateApproximation.fit(self, X)

    def _card_to_bkpt(self, max_cardinality):
        """
        Retourne les breakpoints associés aux cardinalités <= max_cardinality.
        La fonction calcule et stocke les bkpt s'ils n'ont jamais été calculé.

        :param int max_cardinality: la cardinalité maximum

        :returns: breakpoints associés aux cardinalités <= max_cardinality
        :rtype: dict
        """

        if max_cardinality not in self.card_to_bkpt_:

            self.card_to_bkpt_[max_cardinality] = _breakpoints(max_cardinality, scale=self.std) + self.mean
            self.card_to_bkpt_middle_[max_cardinality] = _bin_medians(max_cardinality, scale=self.std) + self.mean

        if max_cardinality > self.alphabet_size_max:
            self.alphabet_size_max = max_cardinality

        return self.card_to_bkpt_[max_cardinality]

    def _card_to_bkpt_only(self, max_cardinality):
        """
        Retourne les breakpoints associés à la cardinalité == max_cardinality.
        La fonction calcule et stocke les bkpt ss'ils n'ont jamais été calculé.

        :param int max_cardinality: la cardinalité

        :returns: les breakpoints associés à la cardinalité == max_cardinality
        :rtype: list
        """

        if max_cardinality not in self.card_to_bkpt_only_:

            bkpt_up = self._card_to_bkpt(max_cardinality)
            bkpt_low = self._card_to_bkpt(max_cardinality/2)

            self.card_to_bkpt_only_[max_cardinality] = [i for i in bkpt_up if i not in bkpt_low]

        return self.card_to_bkpt_only_[max_cardinality]

    def fit_transform(self, X, card, **fit_params):
        """
        Prépare les données ``X`` fournies en paramètre pour l'encodage ``tslearn``.
        Puis transforme les données ``X`` fournies en paramètre d'abord en PAA puis en SAX selon la cardinalité ``card``.

        :param numpy.ndarray X: les données à transformer
        :param int card: la cardinalité à utiliser pour la transformation

        :returns: les données transformées en SAX
        :rtype: numpy.ndarray
        """

        X_ = to_time_series_dataset(X)
        return self._fit(X_)._transform(X_, card)

    def _transform_paa_to_isax(self, X_paa, card):
        """
        Transforme les données ``X_paa`` en paramètre en *i*\ SAX selon les cardinalités ``card``.

        :param numpy.ndarray X_paa: les données PAA à transformer en *i*\ SAX
        :param list card: les cardinalités à utiliser pour la transformation

        :returns: les données transformées en SAX
        :rtype: numpy.ndarray
        """
        
        max_card = max(card)
        isax_list = _paa_to_symbols(X_paa, self._card_to_bkpt(max_card))
        return np_array([[(symbol[0]*card[idx_sym] / max_card,
                           card[idx_sym]) for idx_sym, symbol in enumerate(word)] for word in isax_list], dtype=np_int)

    def _transform(self, X, card):
        """
        Transforme les données ``X`` en paramètre d'abord en PAA puis en SAX selon la cardinalité ``card``.

        :param numpy.ndarray X: les données à transformer
        :param int card: la cardinalité à utiliser pour la transformation

        :returns: les données transformées en SAX
        :rtype: numpy.ndarray
        """
        
        X_paa = PiecewiseAggregateApproximation._transform(self, X)
        return self._transform_paa_to_isax(X_paa, card)

    def transform_paa_to_isax(self, X_paa, card):
        """
        Prépare les données ``X_paa`` fournies en paramètre pour l'encodage ``tslearn``.
        Puis transforme les données ``X_paa`` en paramètre en *i*\ SAX selon les cardinalités ``card``.

        :param numpy.ndarray X_paa: les données PAA à transformer en *i*\ SAX
        :param list card: les cardinalités à utiliser pour la transformation

        :returns: les données transformées en SAX
        :rtype: numpy.ndarray
        """

        X_paa_ = to_time_series_dataset(X_paa)
        return self._transform_paa_to_isax(X_paa_, card)

    def transform(self, X, card):
        """
        Prépare les données ``X`` fournies en paramètre pour l'encodage ``tslearn``.
        Puis transforme les données ``X`` en paramètre d'abord en PAA puis en SAX selon la cardinalité ``card``.

        :param numpy.ndarray X: les données à transformer
        :param int card: la cardinalité à utiliser pour la transformation

        :returns: les données transformées en SAX
        :rtype: numpy.ndarray
        """

        X_ = to_time_series_dataset(X)
        return self._transform(X_, card)

    def _transform_sax(self, X, card):
        """
        Transforme les données ``X`` en paramètre d'abord en PAA puis en SAX selon la cardinalité ``card``.

        :param numpy.ndarray X: les données à transformer
        :param int card: la cardinalité à utiliser pour la transformation

        :returns: les données transformées en SAX
        :rtype: numpy.ndarray
        """
        
        X_paa = PiecewiseAggregateApproximation._transform(self, X)
        # isax_list = _paa_to_symbols(X_paa, self._card_to_bkpt(card))
        return _paa_to_symbols(X_paa, self._card_to_bkpt(card))

    def transform_sax(self, X, card):
        """
        Prépare les données ``X`` fournies en paramètre pour l'encodage ``tslearn``.
        Puis transforme les données ``X`` en paramètre d'abord en PAA puis en SAX selon la cardinalité ``card``.

        :param numpy.ndarray X: les données à transformer
        :param int card: la cardinalité à utiliser pour la transformation

        :returns: les données transformées en SAX
        :rtype: numpy.ndarray
        """

        X_ = to_time_series_dataset(X)
        return self._transform_sax(X_, card)

    def transform_paa(self, X):
        """
        Prépare les données ``X`` fournies en paramètre pour l'encodage ``tslearn``.
        Puis transforme les données ``X`` en paramètre en PAA.

        :param numpy.ndarray X: les données à transformer

        :returns: les données transformées en PAA
        :rtype: numpy.ndarray
        """

        X_ = to_time_series_dataset(X)
        return PiecewiseAggregateApproximation._transform(self, X_)

    def _row_sax_word_array(self, ntss_tmp, bigger_cardinality, size_word):
        """
        Converti toutes les séquences selon les différentes cardinalité de l'arbre.
        Pour chaque cardinalité, fait appel à :func:`~pyCFOFiSAX.isax.IndexableSymbolicAggregateApproximation.transform_sax`.

        :param ntss_tmp: les séquences à analyser
        :param int bigger_cardinality: la plus grande cardinalité *i*\ SAX de l'arbre
        :param int size_word: la taille des séquences SAX de l'arbre

        :returns: les mots SAX de toutes les séquences ``ntss_tmp`` selon toutes les cardinalités de l'arbre, un dict retournant l'indice des cardinalités *i*\ SAX
        :rtype: numpy.ndarray, dict
        """

        # TODO integrer tree.base_cardinality dans le calcul
        # si tree.base_cardinality = 3 par exemple...
        number_of_card = int(math_log(bigger_cardinality, 2))

        row_sax_word = np_zeros((number_of_card + 1, len(ntss_tmp), size_word, 1))
        # TODO card_current = tree.base_cardinality 1/2
        card_current = 1
        card_index = 0
        card_to_index = dict()
        """ TODO np.vectorize
        >>> vtransform_sax = np.vectorize(tree.isax.transform_sax)
        >>> card_list = 2**np.arange(1,int(np.sqrt(tree.bigger_current_cardinality))-1)
        >>> vtransform_sax(ntss_tmp, card_list)"""
        while card_current <= bigger_cardinality:
            card_to_index[card_current] = card_index
            row_sax_word[card_index] = self.transform_sax(ntss_tmp, card_current)
            card_current = card_current * 2
            card_index += 1
        row_sax_word = row_sax_word.transpose((2, 1, 0, 3))

        return row_sax_word.reshape(row_sax_word.shape[:-1]), card_to_index
