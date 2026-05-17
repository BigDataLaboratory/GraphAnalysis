import logging
import os
import pandas as pd

from Utils.Writer import Writer, SUPPORTED_FORMATS
from algorithms.CommunityText import CommunityText
from algorithms.GraphGeneration import GraphGeneration
from algorithms.GraphGenerationUser import GraphGenerationUser
from algorithms.UserFetchStrategies import (
    TweetsSortedByUserScanStrategy,
    CommunityUserBatchStrategy,
    CommunityAwareShardStrategy,
    CommunitySortedLinearScanStrategy
)
from community.Combo import Combo
from community.Leiden import Leiden
from algorithms import mongoQueries
from config import AppConfig
import pandas as pd

class GraphAnalysis:
    logger = logging.getLogger("GraphAnalysis")

    def __init__(self, config: AppConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        import time
        start = time.time()

        self._validate_formats()

        if self.config.graph_generation.to_execute:
            self._run_graph_generation()

        g = self._run_community_detection()   # returns graph or None

        if self.config.get_users_text.to_execute:
            self._run_get_text()

        if self.config.topic_builder.to_execute:
            self._run_topic_builder()

        elapsed = time.time() - start
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        self.logger.info(f"GraphAnalysis complete in {int(h):02}:{int(m):02}:{int(s):02}.")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_formats(self) -> None:
        out = self.config.graph_generation.parameters.output
        for fmt in (out.intermediate_file_format, out.final_file_format):
            if fmt not in SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported file_format '{fmt}'. Choose from {SUPPORTED_FORMATS}."
                )

    # ------------------------------------------------------------------
    # Graph generation
    # ------------------------------------------------------------------

    def _run_graph_generation(self) -> None:
        gg_cfg = self.config.graph_generation
        p = gg_cfg.parameters
        src = p.input.conf
        out = p.output
        gt = p.graph_type
        intermediate_fmt = out.intermediate_file_format
        final_fmt = out.final_file_format

        if gt.user_user:
            if getattr(p, "is_community", False) and getattr(p, "community_file", ""):
                df = pd.read_csv(p.community_file)
                user_community_map = dict(
                    zip(df["user_id"].astype(int), df["community"].astype(int))
                )

                # ── automatical selection of strategy ──────────────────────────
                strategy_name = getattr(p, "community_strategy")
                batch_size = getattr(p, "community_batch_size")

                if strategy_name == "batch":
                    strategy = CommunityUserBatchStrategy(user_community_map)
                elif strategy_name == "linear":
                    strategy = CommunitySortedLinearScanStrategy(user_community_map, batch_size=batch_size)
                else:
                    strategy = CommunityAwareShardStrategy(user_community_map, batch_size=batch_size)
                    
                is_comm = True
            else:
                strategy = TweetsSortedByUserScanStrategy()
                is_comm = False

            ggu = GraphGenerationUser(
                uri=src.uri,
                username=src.username,
                password=src.password,
                auth_source=src.authName,
                auth_mechanism=src.authMechanism,
                database_name=src.db_name,
                collection=src.collection,
                output_file_path=out.path,
                delete_tmp_after_merge=gg_cfg.delete_tmp_after_merge,
                intermediate_file_format=intermediate_fmt,
                final_file_format=final_fmt,
                fast_rt_threshold=p.fast_rt_threshold,
                strategy=strategy,
                is_community_run=is_comm,
                community_file_path=getattr(p, "community_file", None) if is_comm else None,
                load_snapshot_status=p.load_snapshot.status if p.load_snapshot else False,
                load_snapshot_tmp_path=p.load_snapshot.tmp_path if p.load_snapshot else ""
            )
            ggu.run(checkpoint_every=p.checkpoint_every, n_workers=p.n_workers)

        needs_typed_edges = any([
            gt.retweet, gt.tweet_retweet, gt.user_hashtag,
            gt.hashtag_cooccurrences, gt.response, gt.mention,
        ])

        if needs_typed_edges:
            graph_gen = GraphGeneration(
                uri=src.uri,
                username=src.username,
                password=src.password,
                auth_source=src.authName,
                auth_mechanism=src.authMechanism,
                database_name=src.db_name,
                collection=src.collection,
                start_date=src.start_date,
                end_date=src.end_date,
                method=src.method,
                input_type=p.input.type,
                output_file_path=out.path,
                retweet=gt.retweet,
                tweet_retweet=gt.tweet_retweet,
                user_hashtag=gt.user_hashtag,
                hashtag_cooccurrences=gt.hashtag_cooccurrences,
                response=gt.response,
                mention=gt.mention,
                file_format=final_fmt,
            )
            w, s = mongoQueries.extract_tweets_if_contains_hashtags_or_is_retweet_or_reply()
            graph_gen.query_data_in_chunks(w, s, method=src.method)
        else:
            self.logger.info("Skipping typed-edge GraphGeneration: no graph type enabled.")

    # ------------------------------------------------------------------
    # Community detection  — returns the loaded graph (or None)
    # ------------------------------------------------------------------

    def _run_community_detection(self):
        cd = self.config.community_detection
        shared = cd.parameters

        any_cd = cd.leiden.to_execute or cd.combo.to_execute or cd.hierarchical.to_execute
        if not any_cd:
            return None

        # Build the right Writer
        if cd.leiden.to_execute or cd.hierarchical.to_execute:
            w = Writer("igraph", shared.temporal)
        else:
            w = Writer("nx")

        # Load graph
        g = None
        if shared.read_from_edge_list:
            w.read_csv_in_batch(shared.graph_file_path[0], shared.pickle_graph_file_path, 300_000)
        if shared.read_from_file:
            g = w.read_pickle_parallel_preserve_time(shared.pickle_graph_file_path)

        if g is None and any_cd:
            raise RuntimeError(
                "Community detection is enabled but no graph was loaded. "
                "Set read_from_file or read_from_edge_list to true."
            )

        if cd.combo.to_execute:
            self._run_combo(g, cd.combo.parameters)

        if cd.leiden.to_execute:
            self._run_leiden(g, shared.temporal, cd.leiden.parameters)

        if cd.hierarchical.to_execute:
            self._run_hierarchical(g, w, cd.hierarchical.parameters)

        return g

    def _run_combo(self, g, params) -> None:
        combo = Combo()
        rps = combo.compute_combo_in_parallel(g)
        for rp in rps:
            combo.export_partition(g, rp, params.community_output_file_path, ["type", str(rp)])
        combo.export_graph(g, params.community_output_file_path)

    def _run_leiden(self, g, temporal: bool, params) -> None:
        leiden = Leiden(g)
        if not temporal:
            rps = leiden.compute_leiden((0.1, 1.0))
            for rp in rps:
                leiden.export_partition(
                    leiden.get_graph(), rp,
                    params.community_output_file_path,
                    ["name", "type", str(rp)],
                )
            leiden.export_graph(leiden.get_graph(), params.community_output_file_path)
        else:
            rps = leiden.compute_leiden_temporal_incremental(
                method="CPM",
                resolution_parameter_range=(0.1, 1.0),
                lambda_temporal=0.1,
                cap_bonus=1,
                n_iterations=2,
            )
            for rp in rps:
                leiden.export_partition(
                    leiden.get_graph(), rp,
                    params.community_output_file_path,
                    ["name", "type", str(rp)],
                )

    def _run_hierarchical(self, g, w, params) -> None:
        communities = pd.read_csv(
            params.first_level_communities_file, sep=",", header=0, low_memory=False
        )
        labels = communities[params.community_col_name].to_numpy()

        if params.collapse_nodes:
            col = "collapsed"
            self.logger.info(f"Collapsing nodes with min_size={params.min_size}")
            self.logger.info(f"Communities before collapse: {len(set(labels))}")
            labels = w.collapse_nodes(labels, params.min_size)
            self.logger.info(f"Communities after collapse: {len(set(labels))}")
            communities[col] = labels
        else:
            col = params.community_col_name

        vc = pd.Series(labels)
        vc = vc[vc != -1].value_counts()
        top_communities = vc.head(params.top_k).index
        top_node_indices = communities[communities[col].isin(top_communities)]["id"].to_numpy()

        subgraph = g.subgraph(top_node_indices)
        self.logger.info(f"Subgraph: {len(subgraph.vs)} nodes, {len(subgraph.es)} edges")

        # Persist subgraph — path comes from config, not hard-coded
        pickle_path = os.path.join(params.pickle_output_path, "top500_final.pkl")
        subgraph.write_pickle(pickle_path)
        self.logger.info(f"Subgraph saved to {pickle_path}")

        leiden = Leiden(subgraph)
        rps = leiden.compute_leiden((0.1, 1.0), number_of_resolutions=10)
        for rp in rps:
            leiden.export_partition(
                leiden.get_graph(), rp,
                params.community_output_file_path,
                ["name", "type", str(rp)],
            )
        leiden.export_graph(leiden.get_graph(), params.community_output_file_path)

    # ------------------------------------------------------------------
    # Get users text
    # ------------------------------------------------------------------

    def _run_get_text(self) -> None:
        cfg = self.config.get_users_text.parameters
        src = cfg.input.conf
        comms = cfg.communities
        maps = cfg.map_file_path

        if not comms.read_communities_from_file:
            raise AttributeError("Runtime-generated communities are not supported; set read_communities_from_file=true.")
        if not maps.read_maps_from_file:
            raise AttributeError("Runtime-generated maps are not supported; set read_maps_from_file=true.")

        ct = CommunityText(
            comms.indexes or [],
            uri=src.uri,
            username=src.username,
            password=src.password,
            auth_source=src.authName,
            auth_mechanism=src.authMechanism,
            collection=src.collection,
        )
        ct.connect(src.db_name)
        ct.set_comms_file_path(comms.community_file_path)
        ct.set_maps([maps.user_map, maps.retweet_user_map])

        project = {
            "$project": {
                "text": {
                    "$cond": {
                        "if": {"$gt": ["$retweeted_status", None]},
                        "then": "$retweeted_status.text",
                        "else": "$text",
                    }
                },
                "_id": 0,
                "user.id": 1,
                "created_at": 1,
                "type": {
                    "$cond": {
                        "if": {"$gt": ["$retweeted_status", None]},
                        "then": "normal",
                        "else": "retweet",
                    }
                },
            }
        }
        ct.get_users_tweet_text(
            0, project,
            col_comms=["id", "name", "type", "pagerank", "0.6"],
        )
        self.logger.debug("Generated final intermediate result with text data.")

    # ------------------------------------------------------------------
    # Topic builder (stub — implementation commented out upstream)
    # ------------------------------------------------------------------

    def _run_topic_builder(self) -> None:
        self.logger.info("Topic builder stage is a stub — skipping.")