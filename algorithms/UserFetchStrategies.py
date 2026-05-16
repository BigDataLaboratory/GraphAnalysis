from abc import ABC, abstractmethod
import itertools
import math
import os
from collections import defaultdict

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import algorithms.mongoQueries as mongoQueries

class ExtractionStrategy(ABC):
    @abstractmethod
    def partition_work(self, collection, n_workers, max_users, logger):
        pass

    @abstractmethod
    def iter_users(self, collection, work_item):
        pass

    @abstractmethod
    def get_community_id(self, user_id):
        pass

class TweetsSortedByUserScanStrategy(ExtractionStrategy):
    """
    One global iteration over all tweets, sorted by user.id.
    STEP 1: divide the id into ranges, these ranges are assigned to workers
      example: worker0: [0, 999], worker1: [1000, 1999], worker2: [2000, 2999] ...
    STEP 2: each worker iterates over the tweets in their range and compute features

    (+) Only one iteration over the collection
    (-) fetches all the tweet, not good when need only certain users (community users)

    Complexity: O(t): iteration over all tweets
    """
    def partition_work(self, collection, n_workers, max_users, logger):
        logger.info("Scanning collection to determine user.id range...")
        bounds = list(collection.aggregate([
            {"$group": {
                "_id": None,
                "min_uid": {"$min": "$user.id"},
                "max_uid": {"$max": "$user.id"},
            }}
        ]))
        if not bounds:
            return []
        
        global_min = bounds[0]["min_uid"]
        global_max = bounds[0]["max_uid"] + 1
        
        step = max(1, (global_max - global_min + n_workers - 1) // n_workers)
        ranges = []
        for i in range(n_workers):
            uid_start = global_min + i * step
            uid_end   = min(global_min + (i + 1) * step, global_max)
            if uid_start >= global_max:
                break
            ranges.append((uid_start, uid_end))
        return ranges

    def iter_users(self, collection, work_item):
        uid_start, uid_end = work_item
        _, proj = mongoQueries.extract_tweets_for_user_graph()
        query = {"user.id": {"$gte": uid_start, "$lt": uid_end}}
        cursor = collection.find(query, proj).sort("user.id", 1)
        for user_id, user_tweets in itertools.groupby(cursor, key=lambda t: t['user']['id']):
            yield user_id, user_tweets

    def get_community_id(self, user_id):
        return -1

class CommunityUserBatchStrategy(ExtractionStrategy):
    """
    Fetches tweets of community users by splitting them into $in batches.

    STEP 1: Sort community user IDs and divide them into batches of size `batch_size`.
            Batches are distributed across workers (one batch = one work item).
            example: worker0: [user_1 - user_999], worker1: [user_1000 - user_1999], ...
    STEP 2: Each worker executes one MongoDB $in query per batch and iterates
            over the returned tweets, grouped by user.id.

    (+) Fetches ONLY the tweets of community users — no wasted I/O.
    (-) Random I/O pattern: each $in triggers b·log(t) independent BTree lookups.
    (-) Load imbalance possible: a batch may contain users from a dense community
        (many tweets) and block a worker while others are idle.
    (-) Batch size is a trade-off: too small → many round-trips; too large → heavy queries.

    Recommended for: small-to-medium community sets where index fits in RAM.
    Not recommended for: highly skewed community sizes (use CommunityAwareShardStrategy).

    Complexity: O(u·log(t) + (u/b)·C + u·d)
        u  : number of users in the community
        t  : total number of tweets in the collection
        b  : batch_size (number of user IDs per $in query)
        C  : fixed network/query overhead per $in request (ms)
        d  : average tweets per user = t / u_total (e.g. 500M / 16M ≈ 31.25)

        Term breakdown:
            u·log(t)   — BTree index lookups across all batches (dominant, constant in b)
            (u/b)·C    — network round-trip overhead; decreases as b increases
            u·d        — Python-side processing of returned tweets (constant in b)

        Sweet spot for b: sqrt(u·C/k) where k is the memory/timeout penalty per ID.
        In practice: b=500 for u=37k gives ~75 $in queries (negligible overhead).
    """
    def __init__(self, user_community_map, batch_size=10000):
        self.user_community_map = user_community_map
        self.batch_size = batch_size

    def partition_work(self, collection, n_workers, max_users, logger):
        user_ids = sorted(self.user_community_map.keys())
        if max_users is not None:
            user_ids = user_ids[:max_users]

        n_users = len(user_ids)
        min_batches = n_workers * 2
        batch_size = min(self.batch_size, max(1, n_users // min_batches))

        logger.info(
            f"CommunityUserBatchStrategy: {n_users:,} users -> "
            f"batch_size={batch_size} -> ~{math.ceil(n_users / batch_size)} batches "
            f"for {n_workers} workers"
        )

        def chunked(iterable, n):
            it = iter(iterable)
            while batch := list(itertools.islice(it, n)):
                yield batch

        return list(chunked(user_ids, batch_size))

    def iter_users(self, collection, work_item):
        user_ids = work_item
        _, proj = mongoQueries.extract_tweets_for_user_graph()
        query = {"user.id": {"$in": user_ids}}
        cursor = collection.find(query, proj).sort("user.id", 1)
        for user_id, user_tweets in itertools.groupby(cursor, key=lambda t: t['user']['id']):
            yield user_id, user_tweets

    def get_community_id(self, user_id):
        return self.user_community_map.get(user_id, -1)

class CommunityAwareShardStrategy(ExtractionStrategy):
    """
    Fetches tweets of community users by partitioning work at the community level.

    STEP 1: Build an inverse map (community_id → [user_ids]) from the user_community_map.
            Sort communities by size (descending) and distribute them across workers
            using a round-robin greedy algorithm to minimize load imbalance.
            example: worker0: [comm_5 (800 users), comm_2 (120 users), ...]
                     worker1: [comm_1 (750 users), comm_3 (200 users), ...]
    STEP 2: Each worker iterates over its assigned communities sequentially.
            For each community, user IDs are sub-batched into $in queries of size `batch_size`.
            Tweets are streamed and grouped by user.id via itertools.groupby.

    (+) Fetches ONLY the tweets of community users — no wasted I/O.
    (+) Community-level partitioning ensures checkpoints are coherent units
        (one community = one logical block), making partial reruns easier.
    (+) Round-robin by community size minimizes worker idle time at the end.
    (+) Natural fit for downstream community-level analysis (community_id is always known).
    (-) Slightly more complex partitioning logic than CommunityUserBatchStrategy.
    (-) Minimum granularity is one community (~40 users min), so very uneven community
        sizes can still cause minor imbalance within a worker's local queue.

    Recommended for: community-oriented extractions with 100–1000 communities.
    Optimal parameters (u=37k, 278 communities): batch_size=500, n_workers=4.

    Complexity: O(u·log(t) + (u/b)·C + u·d + K)
        u  : total number of users across all communities
        t  : total number of tweets in the collection
        b  : batch_size (number of user IDs per $in query)
        C  : fixed network/query overhead per $in request (ms)
        d  : average tweets per user = t / u_total (e.g. 500M / 16M ≈ 31.25)
        K  : partitioning overhead = n_communities·log(n_communities) (negligible)

        Term breakdown:
            u·log(t)             — BTree index lookups (identical to Batch strategy)
            (u/b)·C              — network round-trip overhead (~75 queries for u=37k, b=500)
            u·d                  — Python-side processing of returned tweets
            K = c·log(c)         — one-time community sort at partition_work time (c=278 → ~2300 ops)

        vs CommunityUserBatchStrategy: identical asymptotic complexity, but better
        load balancing in practice due to community-aware round-robin distribution.
    """
    def __init__(self, user_community_map: dict, batch_size: int = 500):
        self.user_community_map = user_community_map
        self.batch_size = batch_size
        # Inverse map : community_id -> [user_ids]
        self.community_users: dict = defaultdict(list)
        for uid, cid in user_community_map.items():
            self.community_users[cid].append(uid)

    def partition_work(self, collection, n_workers, max_users, logger):
        # Sort communities by size (desc) for better load balancing
        communities = sorted(
            self.community_users.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        # Round-robin on workers to balance the load
        worker_loads = [[] for _ in range(n_workers)]
        worker_sizes = [0] * n_workers
        for cid, uids in communities:
            lightest = min(range(n_workers), key=lambda i: worker_sizes[i])
            worker_loads[lightest].append((cid, sorted(uids)))
            worker_sizes[lightest] += len(uids)

        logger.info(
            f"CommunityAwareShard: {len(communities)} communities | "
            f"{sum(worker_sizes):,} users | "
            f"load per worker: {worker_sizes}"
        )
        return worker_loads  # work_item = list of (cid, [uids])

    def iter_users(self, collection, work_item):
        _, proj = mongoQueries.extract_tweets_for_user_graph()
        for cid, user_ids in work_item:
            for i in range(0, len(user_ids), self.batch_size):
                batch = user_ids[i : i + self.batch_size]
                print(f"[CommunityAwareShardStrategy] Query:{{'user.id': {{'$in': {batch}}}}}\n")
                cursor = collection.find(
                    {"user.id": {"$in": batch}}, proj
                ).sort("user.id", 1)
                for user_id, user_tweets in itertools.groupby(
                    cursor, key=lambda t: t['user']['id']
                ):
                    yield user_id, user_tweets

    def get_community_id(self, user_id):
        return self.user_community_map.get(user_id, -1)


class CommunitySortedLinearScanStrategy(ExtractionStrategy):
    """
    Fetches community tweets by scanning the ENTIRE collection linearly,
    filtering in Python to keep only users present in the community map.

    STEP 1: Compute the global [min_uid, max_uid] range via a MongoDB aggregation.
            Split the range into n_workers equal sub-ranges (by user.id value).
            example: worker0: [uid_0 - uid_4M], worker1: [uid_4M - uid_8M], ...
    STEP 2: Each worker opens a single sorted cursor over its sub-range.
            Tweets are streamed in user.id order and grouped via itertools.groupby.
            Users not in the community map are skipped in O(1) (Python set lookup).

    (+) Single sequential cursor per worker — optimal disk I/O pattern for MongoDB.
    (+) No $in queries — eliminates BTree random lookup overhead entirely.
    (+) Simple and robust: no batching logic, no community distribution complexity.
    (+) Scales linearly with collection size regardless of community structure.
    (-) Reads ALL tweets in the collection, including the 99.77% that are irrelevant.
        At 500M tweets × ~2KB/doc ≈ 1TB of data transferred vs ~2.3GB for $in strategies.
    (-) Python-side skip is O(1) per document but network transfer cost is unavoidable:
        every document must be serialized by MongoDB, sent over the socket, and
        deserialized by PyMongo before the user.id can be checked.
    (-) Extremely inefficient when community coverage ratio is low (< ~5%).

    Recommended for: community coverage ratio > 30% of total users.
    NOT recommended for: this dataset (ratio = 37k / 16M = 0.23% → 430x more I/O
                         than CommunityAwareShardStrategy).

    Break-even point vs $in strategies:
        Linear scan wins when: t_scan < t_batch
        t / u_total > ~0.30  (i.e. targeting > 30% of all users)
        For this dataset: 37k / 16M = 0.23% → $in is ~430x more efficient.

    Complexity: O(t·log(t) + t·C_stream)
        t        : total number of tweets in the collection (500M — dominates everything)
        C_stream : per-document deserialization + Python filter cost (constant, small)

        Term breakdown:
            t·log(t)      — full collection scan cost (sort is pre-existing on user.id index)
            t·C_stream    — PyMongo deserialization + set lookup for every document
            u·d           — actual processing of the ~1.17M matching tweets (same as others)

        vs CommunityAwareShardStrategy: same u·d processing cost, but t·log(t) >> u·log(t)
        by a factor of t/u = 500M/37k ≈ 13 500x more index work, plus full data transfer.
    """
    def __init__(self, user_community_map, batch_size=5000):
        self.user_community_map = user_community_map
        self.batch_size = batch_size
        # On pré-calcule un set pour une recherche O(1)
        self.target_users_set = set(user_community_map.keys())

    def partition_work(self, collection, n_workers, max_users, logger):
        """
        Cuts the collection into equal ID ranges (Min/Max).
        each worker scans its segment linearly.
        """
        logger.info("Calculating UID ranges for linear scan...")
        bounds = list(collection.aggregate([
            {"$group": {
                "_id": None,
                "min_uid": {"$min": "$user.id"},
                "max_uid": {"$max": "$user.id"},
            }}
        ]))
        if not bounds: return []
        
        g_min, g_max = bounds[0]["min_uid"], bounds[0]["max_uid"] + 1
        step = (g_max - g_min + n_workers - 1) // n_workers
        
        ranges = []
        for i in range(n_workers):
            s = g_min + i * step
            e = min(s + step, g_max)
            ranges.append((s, e))
        return ranges

    def iter_users(self, collection, work_item):
        uid_start, uid_end = work_item
        _, proj = mongoQueries.extract_tweets_for_user_graph()
        
        # Filtre de plage pour le worker
        query = {"user.id": {"$gte": uid_start, "$lt": uid_end}}
        
        # Le curseur ne charge pas tout en RAM, il 'stream' les données
        cursor = collection.find(query, proj).sort("user.id", 1).batch_size(self.batch_size)
        
        # Groupby par utilisateur
        for user_id, user_tweets in itertools.groupby(cursor, key=lambda t: t['user']['id']):
            # LA MAGIE : On ignore instantanément l'utilisateur s'il n'est pas dans Leiden
            if user_id not in self.target_users_set:
                continue
            
            yield user_id, user_tweets

    def get_community_id(self, user_id):
        return self.user_community_map.get(user_id, -1)