# Authors: Lucas Foulon <lucas.foulon@gmail.com>
# License: BSD 3 clause

import unittest
import os

from pyCFOFiSAX._forest_iSAX import ForestISAX
import numpy as np
import pandas as pd

from .iCFOF_NAB_multitree import icfof_nab, size_probative_nab
from .iCFOF_Clust2_multitree import icfof_clust2

NOT_TEST_FOREST_WITH_NAB = True
NOT_TEST_FOREST_WITH_CLUST2 = False


class TestiCFOFMethods(unittest.TestCase):

    def test_ForestISAX_init(self):

        my_array = np.array([[1, 2], [3, 4], [5, 6]])
        my_forest = ForestISAX(size_word=2, threshold=10, data_ts=my_array)

        # tests pour la forêt
        self.assertEqual(my_forest.size_word, 2)
        self.assertEqual(my_forest.threshold, 10)
        self.assertEqual(my_forest.base_cardinality, 2)
        self.assertEqual(my_forest.max_cardinality, 2)

        self.assertEqual(my_forest.number_tree, 1)
        self.assertEqual(my_forest.indices_partition, [[0, 1]])

        # tests pour l'arbre
        my_tree = my_forest.forest[0]
        self.assertEqual(my_tree.size_word, 2)
        self.assertEqual(my_tree.threshold, 10)
        self.assertEqual(my_tree.max_card_alphabet, 128)
        self.assertTrue(my_tree.boolean_card_max)
        self.assertEqual(my_tree._base_cardinality, 2)
        self.assertEqual(my_tree.bigger_current_cardinality, 2)

    @unittest.skipIf(NOT_TEST_FOREST_WITH_NAB, "variable NOT_TEST_FOREST_WITH_NAB is True")
    def test_ForestISAX_NAB(self):

        # big dataset
        # dataset_dict = [("realTweets", "Twitter_volume_AMZN.csv", 1),
        #                 ("realTweets", "Twitter_volume_FB.csv", 3),
        #                 ("realTweets", "Twitter_volume_UPS.csv", 12)]
        # small dataset
        dataset_dict = [("realTweets", "Twitter_volume_AMZN.csv", 1),
                        ("realTweets", "Twitter_volume_FB.csv", 3)]

        for dataset_type, name_dataset, num_period_probative in dataset_dict:

            # setting
            # os.path.join("./pyCFOFiSAX/tests/data_test/data/NAB-master", '.'.join((name_dataset, filename_suffix)))
            the_path_file = "./pyCFOFiSAX/tests/data_test/data/NAB-master/" + dataset_type + "/" + name_dataset
            size_probative_period = size_probative_nab(the_path_file)
            probative_period = int(size_probative_period*num_period_probative)

            # calcul des scores
            scores_icfof, dataframe_node = icfof_nab(dataset_type, name_dataset, probative_period,
                                                     update=True, num_tree=1, each_tree_score=False,
                                                     type_split_node={0: 30})

            # test des scores
            the_path_file = "./pyCFOFiSAX/tests/data_test/results/NAB-master/scores_icfof/iCFOF-4stream-30-01/" \
                            + dataset_type + "/iCFOF-4stream-30-01_" + name_dataset
            real_scores_icfof = pd.read_csv(the_path_file, parse_dates=True, index_col=0)

            scores_icfof = scores_icfof[['value', 'p0.1', 'label']].to_numpy()
            real_scores_icfof = real_scores_icfof[['value',
                                                   'anomaly_score',
                                                   'label']].iloc[
                                probative_period:probative_period+size_probative_period]

            np.testing.assert_allclose(scores_icfof, real_scores_icfof, rtol=1e-5)

            # test du nombre de noeuds parcourus
            real_res_nodes = np.genfromtxt("./pyCFOFiSAX/tests/data_test/results/NAB-master/node_visited/"
                                           + "iCFOF-4stream-1tree-30/" + dataset_type + "/" + name_dataset
                                           + "_1tree_30$" + str(num_period_probative) + "num_nodes.txt",
                                           delimiter=",", skip_header=1, usecols=(1, 2, 3))

            dataframe_node = np.concatenate((np.expand_dims(dataframe_node.index.to_numpy(dtype=np.float), axis=0).T,
                                             dataframe_node.to_numpy(dtype=np.float)), axis=1)

            np.testing.assert_allclose(dataframe_node, real_res_nodes, rtol=1e-5)

    @unittest.skipIf(NOT_TEST_FOREST_WITH_CLUST2, "variable NOT_TEST_FOREST_WITH_CLUST2 is True")
    def test_ForestISAX_Clust2(self):

        # dataset_dict = [("clust2_200d_20200319_125226_withrealcfof", 200, 1, False),
        #                 ("clust2_200d_20200319_125226_withrealcfof", 200, 2, False),
        #                 ("clust2_200d_20200319_125226_withrealcfof", 200, 20, True)]
        dataset_dict = [("clust2_200d_20200319_125226_withrealcfof", 200, 20, True)]

        for name_dataset, dim_dataset, num_tree, each_tree_score in dataset_dict:

            ite_to_evaluate = np.random.randint(0, 10000, size=5)

            scores_icfof, dataframe_node = icfof_clust2(name_dataset, dim_dataset, num_tree,
                                                        each_tree_score, ite_to_evaluate)

            thepathfile = "pyCFOFiSAX/tests/data_test/results/Clust2/"\
                          + "clust2_200d_20200319_125226_withrealcfof/" + str(num_tree) + "tree/"\
                          + "clust2_200d_20200319_125226_approx_score_" + str(num_tree) + "tree.csv"
            real_scores_icfof = pd.read_csv(thepathfile, parse_dates=True, index_col=0)

            if each_tree_score and num_tree > 1:
                column_list = ["score_approx_meanvrang"]
                for tmp_num_tree in range(num_tree):
                    column_list.append("score_tree"+str(tmp_num_tree))
                
                scores_icfof = scores_icfof[column_list].to_numpy(dtype=np.float)
                real_scores_icfof = real_scores_icfof[column_list].loc[ite_to_evaluate].to_numpy()

            else:
                scores_icfof = scores_icfof[["score_approx_meanvrang"]].to_numpy(dtype=np.float)
                real_scores_icfof = real_scores_icfof[["score_approx_meanvrang"]].loc[ite_to_evaluate].to_numpy()

            np.testing.assert_allclose(scores_icfof, real_scores_icfof, rtol=1e-5)

            # test du nombre de noeuds parcourus
            real_res_nodes = np.genfromtxt("pyCFOFiSAX/tests/data_test/results/Clust2/"
                                           + name_dataset + "/" + str(num_tree) + "tree/"
                                           + name_dataset + "_" + str(num_tree) + "tree_num_nodes.csv",
                                           delimiter=",", skip_header=1, usecols=(range(1, num_tree+1)))
            tmp_columns = []
            for tmp_num_tree in range(num_tree):
                tmp_columns.append("nb_node_visited_mean_tree" + str(tmp_num_tree))
            dataframe_node = dataframe_node[tmp_columns]
            dataframe_node = dataframe_node.to_numpy(dtype=np.float)

            real_res_nodes = real_res_nodes[ite_to_evaluate]

            np.testing.assert_allclose(dataframe_node, real_res_nodes, rtol=1e-5)
        

if __name__ == '__main__':
    unittest.main()
