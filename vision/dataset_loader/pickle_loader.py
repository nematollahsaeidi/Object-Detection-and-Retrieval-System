import os
import pickle
import time
import numpy as np
from search.faiss.faiss_manager import FaissManager

embedding_dir = "./embeddings"
models_dir = "./models"
dataset_dir = './dataset/'


class PickleLoader:
    def __init__(self, embedding_manager, model_list):
        self.faiss_dict = {}
        self.labels = []
        self.embedding_manager = embedding_manager
        self.model_list = model_list
        self.db_embeddings = None

    def load(self):
        # dataset_name = '_'.join(['digistyle', '53548'])
        dataset_name = '_'.join(['digistyle', '43013'])
        embedding_path = os.path.join(embedding_dir, dataset_name + '.pkl')

        if os.path.exists(embedding_path):
            with open(embedding_path, 'rb') as handle:
                self.db_embeddings = pickle.load(handle)
        else:
            self.db_embeddings = self.embedding_manager.generate_embeddings(dataset_dir)
            dataset_name = '_'.join(['digistyle', f'{len(self.db_embeddings)}'])
            with open(os.path.join(embedding_dir, dataset_name + '.pkl'), 'wb') as f:
                pickle.dump(self.db_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.close()

        if self.db_embeddings is not None:
            for model_name in self.model_list:
                self.faiss_dict[model_name] = FaissManager()
                model_embeddings = []
                for element in self.db_embeddings:
                    model_embeddings.append(self.db_embeddings[element][model_name][0, :])
                    self.labels.append(element)
                index_start_time = time.time()
                self.faiss_dict[model_name].create_l2_index(np.array(model_embeddings))
                # self.faiss_dict[model_name].create_HNSW_index(np.array(model_embeddings))
                # self.faiss_dict[model_name + 'l2'] = self.faiss_dict[model_name].create_l2_index(
                #     np.array(model_embeddings))
                print(f"--- creating index takes {(time.time() - index_start_time):.3f} seconds  ---")
            return self.faiss_dict, self.labels
        else:
            print("We don't have any embeddings db.")
            exit(-1)
