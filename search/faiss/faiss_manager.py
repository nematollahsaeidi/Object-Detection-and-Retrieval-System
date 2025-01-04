import faiss


class FaissManager:
    def __init__(self):
        self.dimension = None
        self.n = None
        self.index = None
        self.number_neighbors = 32
        self.ef_construction = 40

    def create_l2_index(self, db_vectors):
        self.n, self.dimension = db_vectors.shape
        index_flat = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.index_cpu_to_all_gpus(index_flat)
        self.index.add(db_vectors)

    def create_hnsw_index(self, db_vectors):
        self.n, self.dimension = db_vectors.shape
        index_flat_hnsw = faiss.IndexHNSWFlat(self.dimension, self.number_neighbors)
        index_flat_hnsw.hnsw.efConstruction = self.ef_construction  # higher is more accurate
        self.index = faiss.index_cpu_to_all_gpus(index_flat_hnsw)
        self.index.train(db_vectors)
        self.index.add(db_vectors)

    def search_knn(self, query_vector, k):
        # query_vectors = np.random.random((n_query, dimension)).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        return distances, indices
