import configparser

from vision.dataset_loader.elastic_search_index_loader import ElasticSearchLoader
from vision.dataset_loader.pickle_loader import PickleLoader

config = configparser.ConfigParser()
config.read('cfg/config.cfg')


class DatasetLoader:
    def __init__(self, embedding_manager, model_list):
        if config.get('settings', 'dataset_source') == 'elastic':
            es_loader = ElasticSearchLoader(model_list)
            self.faiss_dict, self.dataset = es_loader.load()
        else:
            pickle_loader = PickleLoader(embedding_manager, model_list)
            self.faiss_dict, self.dataset = pickle_loader.load()
