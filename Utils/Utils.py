from itertools import combinations
from datetime import datetime
import mmh3


class Utils:

    @staticmethod
    def hash(x):
        """
        Compute the not signed hash of the input element
        This method uses MurmurHash3 to compute a 64-bit hash of the input.
        The hash is not signed, meaning it will always return a non-negative integer.

        :param x: The input element to hash. It can be any object that can be converted to a string.
        :return: A 64-bit hash of the input element as an
        """
        return mmh3.hash64(str(x), 0)[0]

    @staticmethod
    def compute_hash(x):
        """
        Compute the not signed hash of the input element
        This method uses MurmurHash3 to compute a 64-bit hash of the input.
        The hash is not signed, meaning it will always return a non-negative integer.

        :param x: The input element to hash. It can be any object that can be converted to a string.
        :return: A 64-bit hash of the input element as an integer.
        :raises TypeError: If the input is None.
        """
        if x is not None:
            return mmh3.hash64(x, 0, signed=False)[0]

    @staticmethod
    def combinations_list(x):
        """
        Create all the possible combinations within hashtag in the same tweet, using their hashes
        This method computes the hash of each element in the input list, sorts them, and returns all unique combinations
        of two hashes.

        :param x: A list of elements to compute combinations from. Each element is expected to be hashable.
        :return: A list of tuples, where each tuple contains two hashes representing a combination of two elements.
        :raises TypeError: If the input is None.
        :raises ValueError: If the input list is empty.
        """
        if x is not None:
            hashed = []
            for ht in x:
                hashed.append(mmh3.hash64(ht, 0, signed=False)[0])
            hashed.sort()
            return list(combinations(hashed, 2))
        else:
            return list()

    @staticmethod
    def persist_to_file(obj, file_path: str = './graph', format: str = "csv"):
        if format == "csv":
            obj.to_csv(file_path + "_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".csv", index=False, mode='a')
        elif format == "gml":
            obj.save(file_path + "_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".gml")
        elif format == "json":
            obj.to_json(file_path + "_" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".json", lines=True, orient='records', mode='a')
