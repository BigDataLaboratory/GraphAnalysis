import logging
import pandas as pd
import os

from collections import namedtuple

from Utils.Writer import Writer
from algorithms.CommunityText import CommunityText
from algorithms.GraphGeneration import GraphGeneration
# from community.Combo import Combo
from community.Leiden import Leiden


class GraphAnalysis:
    logger = logging.getLogger("GraphAnalysis")

    def __init__(self, parameters: namedtuple):
        self.parameters = parameters
        self.graph = None 

    def _load_graph(self):

        # Community detection flag
        community_detection = (
            self.parameters.do_community_detection_combo
            or self.parameters.do_community_detection_leiden
            or self.parameters.do_community_hierarchical
        )

        if not community_detection:
            return None

        # Writer: igraph se Leiden/hierarchical, nx se combo
        if self.parameters.do_community_detection_leiden or self.parameters.do_community_hierarchical:
            w = Writer("igraph", self.parameters.temporal)
        elif self.parameters.do_community_detection_combo:
            w = Writer("nx")
        else:
            w = Writer("igraph", self.parameters.temporal)

        # Leggi edge list se richiesto
        if self.parameters.do_read_from_edge_list:
            w.read_csv_in_batch(self.parameters.graph_file_path[0], self.parameters.pickle_graph_path, 300000)

        # Leggi i pickle snapshot (quello che fai di solito)
        if self.parameters.do_read_graph_from_file:
            g = w.read_pickle_parallel_preserve_time(self.parameters.pickle_graph_path)
            return g

        return None

    def _run_leiden_temporal(self, g):
        """
        Esegue Leiden temporale incrementale in due modalità:
        - single run (cap_mode + tenure_mode da cfg)
        - experiments (cap_modes × tenure_modes × rp_list)
        """
        cfg = self.parameters.community_leiden_prop or {}

        # fallback se Main.py non passa ancora community_leiden_exp_prop
        exp = getattr(self.parameters, "community_leiden_exp_prop", {}) or {}



        cfg0 = self.parameters.community_leiden_prop or {}

        # se il parser ti ha passato tutto il blocco leiden, estrai "parameters"
        cfg = cfg0.get("parameters", cfg0)

        # esperimenti: se non esiste l’attributo, prova a prenderli dal JSON annidato
        exp = getattr(self.parameters, "community_leiden_exp_prop", None)
        if not exp:
            exp = cfg0.get("experiments", {}) or {}


        method = cfg.get("method", "CPM")

        # rp range nel tuo JSON è lista [min,max]
        rp_range = cfg.get("resolution_parameter_range", [0.06, 0.06])
        if isinstance(rp_range, (list, tuple)) and len(rp_range) == 2:
            rp_default = float(rp_range[0])
        else:
            rp_default = float(cfg.get("resolution_parameter", 0.06))

        n_iterations = int(cfg.get("n_iterations", 2))
        lambda_temporal = float(cfg.get("lambda_temporal", 0.0))

        # nuove opzioni
        cap_mode = cfg.get("cap_mode", "static")  # off/static/dynamic
        cap_bonus = float(cfg.get("cap_bonus", cfg.get("cap_bonus_base", 0.30)))

        tenure_mode = cfg.get("tenure_mode", "linear")  # none/linear/log/exp
        tenure_exp_k = float(cfg.get("tenure_exp_k", 0.15))

        dynamic_cap_conf = cfg.get("dynamic_cap", {}) or {}

        # controlli / limiti
        max_slices = cfg.get("max_slices", None)
        debug_sample_nodes = cfg.get("debug_sample_nodes", None)
        max_edges = cfg.get("max_edges", None)

        out_dir = cfg.get("community_output_file_path", "./output/communities_leiden")
        os.makedirs(out_dir, exist_ok=True)

        leiden_instance = Leiden(g)


        # MODALITÀ ESPERIMENTI     
        if exp.get("to_execute", False):
            rps_list = exp.get("resolution_parameters", [rp_default])
            cap_modes = exp.get("cap_modes", ["off", "static", "dynamic"])
            tenure_modes = exp.get("tenure_modes", ["linear", "log", "exp"])

            sample_slices = exp.get("sample_slices", max_slices)
            
            for rp in rps_list:
                rp = float(rp)
                for cm in cap_modes:
                    for tm in tenure_modes:
                        run_tag = f"rp{rp}_cap{cm}_ten{tm}"

                        self.logger.info(f"[EXPERIMENT] Start {run_tag}")
                        leiden_instance.run_temporal_experiment(
                            method=method,
                            resolution_parameter=rp,
                            lambda_temporal=lambda_temporal,
                            cap_mode=cm,
                            cap_bonus=cap_bonus,
                            tenure_mode=tm,
                            tenure_exp_k=tenure_exp_k,
                            dynamic_cap_conf=dynamic_cap_conf,
                            n_iterations=n_iterations,
                            max_slices=sample_slices,
                            debug_sample_nodes=debug_sample_nodes,
                            max_edges=max_edges,
                            output_dir=out_dir,
                            run_tag=run_tag
                        )





        # MODALITÀ SINGLE RUN

        else:
            run_tag = f"single_rp{rp_default}_cap{cap_mode}_ten{tenure_mode}"
            self.logger.info(f"[SINGLE] Start {run_tag}")

            leiden_instance.run_temporal_experiment(
                method=method,
                resolution_parameter=rp_default,
                lambda_temporal=lambda_temporal,
                cap_mode=cap_mode,
                cap_bonus=cap_bonus,
                tenure_mode=tenure_mode,
                tenure_exp_k=tenure_exp_k,
                dynamic_cap_conf=dynamic_cap_conf,
                n_iterations=n_iterations,
                max_slices=max_slices,
                debug_sample_nodes=debug_sample_nodes,
                max_edges=max_edges,
                output_dir=out_dir,
                run_tag=run_tag
            )

    def run(self):
        print("🔎 Flag Leiden:", self.parameters.do_community_detection_leiden)
        print("🔎 Flag Combo:", self.parameters.do_community_detection_combo)
        print("🔎 Flag Hierarchical:", self.parameters.do_community_hierarchical)
        print("📂 Cerco snapshot in:", self.parameters.pickle_graph_path)

        try:
            if os.path.isdir(self.parameters.pickle_graph_path):
                print("📁 File trovati:", len(os.listdir(self.parameters.pickle_graph_path)))
            else:
                print("⚠️ pickle_graph_path non è una cartella:", self.parameters.pickle_graph_path)
        except Exception as e:
            print("⚠️ Impossibile leggere snapshot dir:", e)

        
        # 1) (Opzionale) Graph generation
        
        if self.parameters.do_graph_generation:
            gg = GraphGeneration(
                uri=self.parameters.source_uri,
                username=self.parameters.source_username,
                password=self.parameters.source_password,
                auth_source=self.parameters.source_auth_source,
                auth_mechanism=self.parameters.source_auth_mechanism,
                database_name=self.parameters.source_db_name,
                collection=self.parameters.source_collection,
                start_date=self.parameters.source_chunk_start_date,
                end_date=self.parameters.source_chunk_end_date,
                method=self.parameters.source_method,
                input_type=self.parameters.source_input_type,
                output_file_path=self.parameters.output_graph_path,
                retweet=self.parameters.do_retweet_graph,
                tweet_retweet=self.parameters.do_tweet_retweet_graph,
                user_hashtag=self.parameters.do_hashtag_graph,
                hashtag_cooccurrences=self.parameters.do_hashtag_cooccurrences_graph,
                response=self.parameters.do_response_graph,
                mention=self.parameters.do_mention_graph
            )

            # use it when mongo is available again
            w = {
                "$or": [
                    {"hashtagEntities": {"$exists": True}},
                    {"retweeted_status": {"$exists": True}},
                    {"in_reply_to_status_id": {"$exists": True}}
                ]
            }
            s = {
                "_id": 0,
                "id": 1,
                "in_reply_to_status_id": 1,
                "in_reply_to_user_id": 1,
                "retweeted_status.created_at": 1,
                "retweeted_status.id": 1,
                "retweeted_status.user.id": 1,
                "retweeted_status.user.screen_name": 1,
                "user.id": 1,
                "user.screen_name": 1,
                "hashtagEntities": 1,
                "created_at": 1,
                "userMentionEntities": 1
            }

            gg.query_data_in_chunks(w, s, method=self.parameters.source_method)

        # 2) Load graph (pickle snapshots)
        
        g = self._load_graph()
        self.graph = g  # salva per Main.py

        if g is None:
            self.logger.warning("Nessun grafo caricato (g=None).")
            return None

        
        # 3) Combo (se attivo) 
     
        if self.parameters.do_community_detection_combo:
            # combo_instance = Combo()
            # rps = combo_instance.compute_combo_in_parallel(g)
            # for rp in rps:
            #     combo_instance.export_partition(g, rp,
            #                                     self.parameters.community_combo_prop["community_output_file_path"],
            #                                     ["type", "{}".format(rp)])
            # combo_instance.export_graph(g, self.parameters.community_combo_prop["community_output_file_path"])
            self.logger.warning("Combo attivo ma la classe Combo è commentata/import non presente.")
            pass

        
        # 4) Leiden (static o temporal)
    
        if self.parameters.do_community_detection_leiden:
            if not self.parameters.temporal:
                # static Leiden
                leiden_instance = Leiden(g)
                rps = leiden_instance.compute_leiden((0.1, 1.0))
                for rp in rps:
                    leiden_instance.export_partition(
                        leiden_instance.get_graph(),
                        rp,
                        self.parameters.community_leiden_prop["community_output_file_path"],
                        ["name", "type", "{}".format(rp)]
                    )
                leiden_instance.export_graph(
                    leiden_instance.get_graph(),
                    self.parameters.community_leiden_prop["community_output_file_path"]
                )
            else:
                # temporal incremental: nuova logica
                self._run_leiden_temporal(g)

       
        # 5) Hierarchical 
        if self.parameters.do_community_hierarchical:
            collapse_nodes = self.parameters.community_hierarchical_prop["collapse_nodes"]
            resolution_col = self.parameters.community_hierarchical_prop["community_col_name"]
            top_k = self.parameters.community_hierarchical_prop["top_k"]

            w = Writer("igraph", self.parameters.temporal)

            communities = pd.read_csv(
                self.parameters.community_hierarchical_prop["first_level_communities_file"],
                sep=",",
                header=0,
                low_memory=False
            )
            labels = communities[resolution_col].to_numpy()

            if collapse_nodes:
                col_to_extract_communities = "collapsed"
                self.logger.info("Collapsing nodes with min size {}".format(self.parameters.community_hierarchical_prop["min_size"]))
                self.logger.info("Number of communities before collapsing: {}".format(len(set(labels))))
                labels_c = w.collapse_nodes(labels, self.parameters.community_hierarchical_prop["min_size"])
                self.logger.info("Number of communities after collapsing: {}".format(len(set(labels_c))))
                communities[col_to_extract_communities] = labels_c
                vc = pd.Series(labels_c)
            else:
                col_to_extract_communities = resolution_col
                vc = pd.Series(labels)

            vc = vc[vc != -1].value_counts()
            top_communities = vc.head(top_k).index

            top_node_indices = communities[communities[col_to_extract_communities].isin(top_communities)]["id"].to_numpy()

            subgraph = g.subgraph(top_node_indices)
            self.logger.info("Number of nodes in subgraph: {}".format(len(subgraph.vs)))
            self.logger.info("Number of edges in subgraph: {}".format(len(subgraph.es)))

            filename = "top500_final.pkl"
            full_path = os.path.join("/ipazianas/pasquini/twitter_graph_dump", filename)
            with open(full_path, "wb") as f:
                subgraph.write_pickle(full_path)
            self.logger.info("Final graph saved.")

            leiden_instance = Leiden(subgraph)
            rps = leiden_instance.compute_leiden((0.1, 1.0), number_of_resolutions=10)
            for rp in rps:
                leiden_instance.export_partition(
                    leiden_instance.get_graph(),
                    rp,
                    self.parameters.community_hierarchical_prop["community_output_file_path"],
                    ["name", "type", "{}".format(rp)]
                )
            leiden_instance.export_graph(leiden_instance.get_graph(), self.parameters.community_hierarchical_prop["community_output_file_path"])

       
        # 6) Get text / topic builder 
        if self.parameters.do_get_text:
            ct = CommunityText(
                self.parameters.community_indexes if self.parameters.community_indexes else [],
                uri=self.parameters.td_uri,
                username=self.parameters.td_username,
                password=self.parameters.td_password,
                auth_source=self.parameters.td_auth_source,
                auth_mechanism=self.parameters.td_auth_mechanism,
                collection=self.parameters.td_collection
            )
            ct.connect(self.parameters.td_db_name)

            if self.parameters.do_read_communities_from_file:
                ct.set_comms_file_path(self.parameters.community_file_path)
            else:
                raise AttributeError("It's not possible to use communities generated at runtime")

            if not self.parameters.do_read_maps_from_file:
                raise AttributeError("It's not possible to use maps generated at runtime")
            else:
                maps = [self.parameters.user_map, self.parameters.retweet_user_map]
                ct.set_maps(maps)

            
            # ct.get_users_tweet_text(...)
            self.logger.debug("Generated final intermediate result with text data")

        if self.parameters.do_topic_builder:
            pass

        return g
