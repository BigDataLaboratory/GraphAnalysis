from itertools import combinations
import mmh3


class Utils:


    def hash(self, x):
        """
        Compute the not signed hash of the input element
        """
        return mmh3.hash64(str(x), 0)[0]

    def compute_hash(self, x):
        """
        Compute the not signed hash of the input element
        """
        if x is not None:
            return mmh3.hash64(x, 0)[0]

    def combinations_list(self, x):
        """
        Create all the possible combinations within hashtag in the same tweet, using their hashes
        """
        if x is not None:
            hashed = []
            for ht in x:
                hashed.append(mmh3.hash64(ht, 0)[0])
            hashed.sort()
            return list(combinations(hashed, 2))

    def persist_to_file(self, df, file_path='./graph.csv', format="csv"):
        if format == "csv":
            df.to_csv(file_path + ".csv", index=False)