.. pyiCFOF documentation master file, created by
   sphinx-quickstart on Fri Nov 13 09:09:46 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to py-\ *i*\ CFOF's documentation!
===========================================

Cette documentation présente le fonctionnement des classes Python pour l'approximation des scores CFOF à l'aide d'un ou plusieurs arbres d'indexation *i*\ SAX.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ./cfofisax.rst
   ./forest_iSAX.rst
   ./tree_iSAX.rst
   ./node.rst
   ./isax.rst

Pour commencer
-----------------------

Installation
~~~~~~~~~~~~~~~~~

Lancer ``pip install -r requirements.txt`` ou ``python3 -m pip install -r requirements.txt``.

Utilisation
~~~~~~~~~~~~~~~~~

>>> from pyCFOFiSAX import CFOFiSAX
>>> cfof_isax = CFOFiSAX()

Un jeu de données artificiel `Clust2` contenant 10000 séquences en dimension 200 est disponible dans le projet avec les résultats CFOF (attention : ce ne sont pas les scores CFOFiSAX) pour :math:`\varrho = [0.01, 0.05, 0.1]` :

>>> import numpy as np
>>> # chargement du jeu, avec usecols=list(range(0, 200)) pour ne pas charger les scores
>>> ndarray_dataset = np.genfromtxt("pyCFOFiSAX/tests/data_test/data/clust2_200d_20200319_125226_withrealcfof.csv",
>>>                                 delimiter=',',
>>>                                 skip_header=1,
>>>                                 usecols=list(range(0, 200)))
>>> ndarray_dataset.shape
(10000, 200)


Initialisation de la forêt avec 20 arbres *i*\ SAX :

>>> cfof_isax.init_forest_isax(size_word=200,
>>>                            threshold=30,
>>>                            data_ts=ndarray_dataset,
>>>                            base_cardinality=2, number_tree=20)

Insertion des données :

>>> cfof_isax.forest_isax.index_data(ndarray_dataset)

Puis pré-traitement :

>>> # va afficher les temps de pré-traitement pour chaque arbre et afficher le nombre de nœuds dans chaque arbre
>>> cfof_isax.forest_isax.preprocessing_forest_for_icfof(ndarray_dataset,
>>>                                                      bool_print=True, count_num_node=True)

Ensuite, le calcul de l'approximation *i*\ CFOF, pour la ième séquence est effectué avec la fonction :

>>> tmp_ite = np.random.randint(0,10000)
>>> tmp_ite
8526
>>> score = cfof_isax.score_icfof(
>>>         ndarray_dataset[tmp_ite], ndarray_dataset,
>>>         rho=[0.1], each_tree_score=True,
>>>         fast_method=True)
>>> # les approximations pour chaque valeur de rho sont dans score[0]
>>> score[0]
array([0.21357929])
>>> # si each_tree_score=True,
>>> # il est possible de regarder les scores obtenus dans chaque arbre avec score[1]

Pour connaître le nombre de nœuds visités dans chaque arbre lors du calcul :

>>> cfof_isax.forest_isax.number_nodes_visited(ndarray_dataset[tmp_ite],
>>>                                            ndarray_dataset)
array([1086.    , 1044.    , 1061.    , 1087.    , 1013.    , 1115.    ,
       1067.    , 1101.    , 1069.    , 1065.    , 1104.    , 1104.    ,
       1115.    , 1059.    , 1009.    , 1001.    , 1097.    , 1018.    ,
       1081.    , 1084.    ,  838.215 ,  804.8092,  826.3052,  853.1703,
        769.1603,  870.4031,  829.9387,  853.9431,  833.5567,  830.6371,
        849.1152,  861.5018,  864.5723,  820.4267,  762.5597,  763.3633,
        863.6749,  785.884 ,  850.051 ,  832.9658])


Réferences
-----------------
* `Scoring Message Stream Anomalies in Railway Communication Systems, L.Foulon et al., 2019, ICDMWorkshop <https://ieeexplore.ieee.org/abstract/document/8955558>`_
* `Approximation du score CFOF de détection d’anomalie dans un arbre d’indexation iSAX: Application au contexte SI de la SNCF, L.Foulon et al., 2019, Actes de la conférence EGC'2019 <https://hal.archives-ouvertes.fr/hal-02019035>`_
* `CFOF: A Concentration Free Measure for Anomaly Detection, F. Angiulli, 2020, TKDD <https://ieeexplore.ieee.org/abstract/document/8955558>`_
* `Concentration Free Outlier Detection, F. Angiulli, 2017, ECML-PKDD <https://link.springer.com/chapter/10.1007/978-3-319-71249-9_1>`_
* `iSAX 2.0: Indexing and Mining One Billion Time Series, A. Camerrea et al., 2010, ICDM <https://ieeexplore.ieee.org/abstract/document/5693959>`_
* `iSAX: Disk-Aware Mining and Indexing of Massive Time Series Datasets, J. Shieh et al., 2009, DMKD <https://link.springer.com/article/10.1007/s10618-009-0125-6>`_
* `Unsupervised real-time anomaly detection for streaming data, Ahmad, S. et al., 2017, Neurocomputing <https://doi.org/10.1016/j.neucom.2017.04.070>`_
* `Tslearn, A Machine Learning Toolkit for Time Series Data, R.Tavenard et al., 2020 , JMLR <http://jmlr.org/papers/v21/20-091.html>`_


Autres
-----------------

* :ref:`genindex`
