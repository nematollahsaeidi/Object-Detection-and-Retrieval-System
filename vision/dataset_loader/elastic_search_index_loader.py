import configparser
import time
import elasticsearch.helpers
import numpy as np
from search.faiss.faiss_manager import FaissManager

config = configparser.ConfigParser()
config.read('cfg/config.cfg')


class ElasticSearchLoader:
    def __init__(self, model_list):
        self.model_list = model_list
        self.dataset = []
        self.faiss_dict = {}
        self.es = elasticsearch.Elasticsearch(
            [config.get('elastic', 'ip')],
            http_auth=(config.get('elastic', 'user'), config.get('elastic', 'password')),
            scheme=config.get('elastic', 'scheme'),
            port=int(config.get('elastic', 'port')),
            timeout=int(config.get('elastic', 'timeout')),
            verify_certs=config.getboolean('elastic', 'verify_certs'))
        self.res = elasticsearch.helpers.scan(
            self.es,
            query={"query": {"match_all": {}}},
            index=config.get('elastic', 'base_index'),
            size=1000,
            doc_type='image',
            preserve_order=True
        )

    def load(self):
        # if not es.indices.exists(index='cbir'):
        #     # es = ElasticManager(['InceptionResNetV2']).generate_elastic_index(dataset_dir_main)
        #     print('cbir indexes were made')
        # else:
        for model_name in self.model_list:
            self.faiss_dict[model_name] = FaissManager()
            model_embeddings = []
            iteration = 0
            good_samples = 0
            index_start_time = time.time()
            for hit in self.res:
                record = hit["_source"]
                try:
                    if iteration < 9_999:
                        if type(record) is dict and len(hit['_source']) == 15:
                            model_embeddings.append(np.float32(record['irn2_embedding'][0]))
                            self.dataset.append(record)
                            good_samples += 1
                        print(iteration)
                        iteration += 1
                    else:
                        break
                except KeyError as e:
                    print(e)
                    print(f"irn2_embedding field of {iteration}, doesn't exist")
                    print(f'{record}')
                except Exception as e:
                    print(e)
                    break
            print(good_samples)
            print(f"--- Reading from elastic takes {(time.time() - index_start_time):.3f} seconds  ---")
            model_embeddings = np.array(model_embeddings)
            self.faiss_dict[model_name].create_l2_index(model_embeddings)

        return self.faiss_dict, self.dataset
