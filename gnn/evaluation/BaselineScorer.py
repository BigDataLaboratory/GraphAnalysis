class BaselineScorer:
    def score(self, data, edge_index):
        """
        Score candidate edges. Returns a 1-D tensor of scores in [0, 1].
        """
        raise NotImplementedError
