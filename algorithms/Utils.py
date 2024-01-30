from itertools import combinations

import mmh3


class Utils:

    @staticmethod
    def hash(x):
        """
        Compute the not signed hash of the input element
        """
        return mmh3.hash64(str(x), 0)[0]

    @staticmethod
    def compute_hash(x):
        """
        Compute the not signed hash of the input element
        """
        if x is not None:
            return mmh3.hash64(x, 0)[0]

    @staticmethod
    def combinations_list(x):
        """
        Create all the possible combinations within hashtag in the same tweet, using their hashes
        """
        if x is not None:
            hashed = []
            for ht in x:
                hashed.append(mmh3.hash64(ht, 0)[0])
            hashed.sort()
            return list(combinations(hashed, 2))

    @staticmethod
    def persist_to_file(obj, file_path='./graph.csv', format="csv"):
        if format == "csv":
            obj.to_csv(file_path + ".csv", index=False)
        elif format == "gml":
            obj.save(file_path + ".gml")
