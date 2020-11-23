# Calcul CFOF via iSAX

[![Documentation Status](https://readthedocs.org/projects/pycfofisax/badge/?version=main)](https://pycfofisax.readthedocs.io/fr/main/?badge=main)

Projet thèse Lucas \
2017-2020

## Pour commencer

### Installation

Lancer `pip install -r requirements.txt` ou `python3 -m pip install -r requirements.txt`.

Le package `tslearn` requière `Cython`, `numba` et `llvmlite` (`pip` téléchargera automatiquement ces packages).
Cependant pour `Cython`, il est [nécessaire d'avoir un compileur C (comme décrit sur ce site)](https://cython.readthedocs.io/en/latest/src/quickstart/install.html),
et le [package `llvmlite`](https://llvmlite.readthedocs.io/en/latest/) n'a été, pour ce projet, testé qu'avec Python 3.7 et 3.8
que nous recommandons.

Sinon, l'utilisation de `conda` simplifie l'installation (comme recommandé sur le site de [tslearn](https://tslearn.readthedocs.io/en/latest/index.html)), car ne nécessite pas 
l'installation de compileur C et des packages `Cython` et `llvmlite`.

### Utilisation

La documentation est disponible ici : [https://pycfofisax.readthedocs.io/fr/main/](https://pycfofisax.readthedocs.io/fr/main/)

## Remerciements

Développements des travaux de :
 - [CFOF: A Concentration Free Measure for Anomaly Detection, par Fabrizio Angiulli](https://arxiv.org/abs/1901.04992),
 - [iSAX: Indexing and Mining Terabyte Sized Time Series, par Jin Shieh et Eamonn Keogh](http://www.cs.ucr.edu/~eamonn/iSAX/iSAX.html),
 - [iSAX 2.0: Indexing and Mining One Billion Time Series, par Alessandro Camerra, Themis Palpanas, Jin Shieh et Eamonn Keogh](https://www.cs.ucr.edu/~eamonn/iSAX_2.0.pdf).
 - [Scoring Message Stream Anomalies in Railway Communication Systems, par Lucas Foulon, Serge Fenet, Christophe Rigotti et Denis Jouvin](https://hal.archives-ouvertes.fr/hal-02357924/)

Utilisation du code de :
 - [tslearn de Romain Tavenard et al. (2017)](https://tslearn.readthedocs.io/en/latest/index.html).

Utilisation des jeux de :
 - [Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017). Unsupervised real-time anomaly detection for streaming data. Neurocomputing, Available online 2 June 2017, ISSN 0925-2312](https://doi.org/10.1016/j.neucom.2017.04.070)
