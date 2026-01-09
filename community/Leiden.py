# -*- coding: utf-8 -*-
import logging
import os
import time
import uuid
import csv
import datetime as dt
from collections import defaultdict

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd

from Utils.Writer import Writer



class Leiden:
    logger = logging.getLogger("Leiden")

    def __init__(self, data_graph: ig.Graph):
        self.id = uuid.uuid1().hex
        self.data_graph = data_graph

    def _print_memory_usage(self, note=""):
        try:
            import psutil, os as _os
            process = psutil.Process(_os.getpid())
            mem_mb = process.memory_info().rss / (1024 * 1024)
            self.logger.info(f"[MEM] {note} | {mem_mb:.2f} MB")
        except Exception:
            pass

    def get_graph(self):
        """
        Returns the graph associated with this Leiden instance.
        This method is useful for accessing the graph after it has been processed or modified.
        
        :return: The igraph.Graph instance associated with this Leiden instance.
        """
        return self.data_graph

  
    def compute_pagerank(self, data_graph):
         
        """
        Computes the PageRank values for nodes in the graph, excluding nodes of type 'hashtag'.
        This method first creates a subgraph excluding nodes of type 'hashtag', then computes the
        PageRank values for the remaining nodes. The results are mapped back to the original graph.

        :param data_graph: The graph to compute PageRank on, an igraph.Graph instance.
        """
        self.logger.info("Start removing nodes of type hashtag")
        exclude_type = "h"
        nodes_to_keep = [v.index for v in data_graph.vs if v["type"] != exclude_type]
        subgraph = data_graph.induced_subgraph(nodes_to_keep)

        self.logger.info("Start computation of PageRank")
        start = time.time()
        pagerank_values = subgraph.pagerank(directed=True, weights="weight", implementation="prpack")
        end = time.time()

        data_graph.vs["pagerank"] = [None] * data_graph.vcount()
        for subgraph_node, pagerank in zip(subgraph.vs, pagerank_values):
            original_index = subgraph.vs["name"].index(subgraph_node["name"])
            data_graph.vs[original_index]["pagerank"] = pagerank

        self.logger.info("Computation of PageRank completed!")
        self.logger.info("Elapsed time: " + str(end - start))

    def build_daily_slices(self, g, first_n_days=10):
        edges_by_day = defaultdict(list)
        for e in g.es:
            day = dt.datetime.utcfromtimestamp(e["time"]).date()
            edges_by_day[day].append((e.tuple, e["weight"] if "weight" in e.attributes() else 1.0))

        if first_n_days > 0:
            sorted_days = sorted(edges_by_day)[:first_n_days]
            self.logger.info(f"Using the first {first_n_days} days with activity: {[d.isoformat() for d in sorted_days]}")
        else:
            sorted_days = sorted(edges_by_day)
            self.logger.info("Using all days with activity")

        slices = []
        for day in sorted_days:
            edge_tuples = [tpl for tpl, _ in edges_by_day[day]]
            weights = [w for _, w in edges_by_day[day]]

            Gd = ig.Graph(n=g.vcount(), edges=edge_tuples, directed=g.is_directed())
            for attr in g.vs.attributes():
                Gd.vs[attr] = list(g.vs[attr])
            Gd.vs["slice"] = [day.isoformat()] * Gd.vcount()
            Gd.es["weight"] = weights
            slices.append((day, Gd))
        return slices

    def iter_weekly_slices(self, g: ig.Graph):
        """
        Yields weekly slices (ISO week) aggregating weights per (u,v) within each week.
        IMPORTANT: qui ricostruiamo un grafo settimanale con solo 'weight' sugli archi.
        Il calcolo del CAP DINAMICO (come Dan) NON usa queste slice, ma il grafo originale.
        """
        weekly_edge_weights = defaultdict(lambda: defaultdict(float))

        for e in g.es:
            ts = dt.datetime.fromtimestamp(e["time"], tz=dt.timezone.utc)
            year, week, _ = ts.isocalendar()
            week_str = f"{year}-W{week:02d}"

            u, v = e.tuple
            w = float(e["weight"]) if "weight" in e.attributes() and e["weight"] is not None else 1.0
            weekly_edge_weights[week_str][(int(u), int(v))] += w

        for week_str, edge_dict in sorted(weekly_edge_weights.items()):
            edge_list = list(edge_dict.keys())
            weight_list = list(edge_dict.values())

            Gw = ig.Graph(n=g.vcount(), edges=edge_list, directed=True)
            for attr in g.vs.attributes():
                Gw.vs[attr] = list(g.vs[attr])
            Gw.vs["slice"] = [week_str] * Gw.vcount()
            Gw.es["weight"] = weight_list
            yield week_str, Gw

   
    def collapse_nodes(self, labels, min_size=30, dummy=-1):
        self.logger.info("Start collapsing nodes")
        vc = pd.Series(labels).value_counts()
        big = vc[vc >= min_size].index
        self.logger.info("Finished collapsing nodes")
        return np.where(pd.Series(labels).isin(big), labels, dummy)

    def _relabel_with_overlap(self, prev_comm_by_name, curr_names, curr_labels):
        overlap_counts = defaultdict(lambda: defaultdict(int))
        for name, curr_label in zip(curr_names, curr_labels):
            prev_label = prev_comm_by_name.get(name)
            if prev_label is not None:
                overlap_counts[curr_label][prev_label] += 1

        overlaps = []
        for lab_curr, prev_counts in overlap_counts.items():
            if prev_counts:
                best_prev, best_size = max(prev_counts.items(), key=lambda item: item[1])
                overlaps.append((lab_curr, best_prev, best_size))

        used_prev = set()
        remap = {}
        for lab_curr, lab_prev, _ in sorted(overlaps, key=lambda x: -x[2]):
            if lab_prev not in used_prev:
                remap[lab_curr] = lab_prev
                used_prev.add(lab_prev)

        next_label = max(prev_comm_by_name.values(), default=-1) + 1
        for lab_curr in set(curr_labels):
            if lab_curr not in remap:
                remap[lab_curr] = next_label
                next_label += 1

        return [remap[lab] for lab in curr_labels]

    
    def _tenure_bonus(self, tau, lambda_temporal, cap_value, tenure_mode="linear", tenure_exp_k=0.15):
        """
        Bonus per archi 'stabili' (endpoints nella stessa community precedente).
        tau = min(tenure(u), tenure(v)) per ciascun arco stabile.
        bonus = f(lambda, tau) poi cappato a cap_value (cap static o cap(t->t+1)).
        """
        if lambda_temporal <= 0:
            return np.zeros_like(tau, dtype=np.float32)

        tau = tau.astype(np.float32)
        lam = float(lambda_temporal)
        capv = float(cap_value)

        if tenure_mode == "none":
            raw = np.full_like(tau, lam, dtype=np.float32)
        elif tenure_mode == "linear":
            raw = lam * (1.0 + tau)
        elif tenure_mode == "log":
            raw = lam * (1.0 + np.log1p(tau))
        elif tenure_mode == "exp":
            k = float(tenure_exp_k)
            raw = lam * np.exp(k * tau)
        else:
            raise ValueError(f"Unknown tenure_mode: {tenure_mode}")

        return np.minimum(raw, capv).astype(np.float32)

   
    def _weekly_mutual_retweet_weights_from_original(self, retweet_edge_types=None, max_slices=None):
        """
        Costruisce, per ogni settimana, le coppie MUTUAL (u,v)->W_t(u,v)
        usando il grafo originale (con edge 'time' e 'type').

        - si considerano SOLO archi retweet (type in retweet_edge_types)
        - mutual = esiste (u->v) e (v->u) nella stessa settimana
        - W_t(u,v) = w(u->v) + w(v->u) nella settimana t
        """
        if retweet_edge_types is None:
            retweet_edge_types = {"0"}

        if "time" not in self.data_graph.es.attribute_names():
            raise RuntimeError("Edge attribute 'time' non trovato: impossibile costruire settimane.")
        if "type" not in self.data_graph.es.attribute_names():
            raise RuntimeError("Edge attribute 'type' non trovato: impossibile filtrare SOLO retweet")

        # week -> directed weights
        dir_w_by_week = defaultdict(lambda: defaultdict(float))

        for e in self.data_graph.es:
            if str(e["type"]) not in retweet_edge_types:
                continue

            ts = dt.datetime.fromtimestamp(e["time"], tz=dt.timezone.utc)
            year, week, _ = ts.isocalendar()
            week_str = f"{year}-W{week:02d}"

            u, v = e.tuple
            w = float(e["weight"]) if "weight" in e.attributes() and e["weight"] is not None else 1.0
            dir_w_by_week[week_str][(int(u), int(v))] += w

        weeks_sorted = sorted(dir_w_by_week.keys())
        if max_slices is not None:
            weeks_sorted = weeks_sorted[:int(max_slices)]

        mutual_by_week = {}
        for wk in weeks_sorted:
            dir_w = dir_w_by_week[wk]
            mutual = {}
            for (u, v), wuv in dir_w.items():
                if u < v:
                    wvu = dir_w.get((v, u))
                    if wvu is not None:
                        mutual[(u, v)] = float(wuv + wvu)
            mutual_by_week[wk] = mutual

        return weeks_sorted, mutual_by_week

    def precompute_dynamic_cap_dan(self, max_slices=None, retweet_edge_types=None, stat="median", fallback=0.30):
        """
        Per ogni transizione t -> t+1 (settimane consecutive):
          - considera le coppie mutual in t (solo retweet)
          - seleziona quelle che NON sono mutual in t+1
          - prendi i valori W_t(u,v) di quelle coppie (peso in t)
          - cap(t->t+1) = mediana di quei valori

        Ritorna dict: week_str (t+1) -> cap_value
        (la prima settimana ha cap=fallback perché non esiste transizione in ingresso).
        """
        weeks, mutual_by_week = self._weekly_mutual_retweet_weights_from_original(
            retweet_edge_types=set(retweet_edge_types) if retweet_edge_types is not None else {"0"},
            max_slices=max_slices
        )

        caps = {}
        if not weeks:
            return caps

        caps[weeks[0]] = float(fallback)

        for i in range(1, len(weeks)):
            prev_wk = weeks[i - 1]
            curr_wk = weeks[i]

            prev_mutual = mutual_by_week.get(prev_wk, {})
            curr_mutual = mutual_by_week.get(curr_wk, {})

            # mutual in t but NOT mutual in t+1
            lost = [w for pair, w in prev_mutual.items() if pair not in curr_mutual]

            if len(lost) == 0:
                cap_val = float(fallback)
            else:
                arr = np.array(lost, dtype=np.float32)
                if stat == "median":
                    cap_val = float(np.median(arr))
                elif stat == "mean":
                    cap_val = float(np.mean(arr))
                else:
                    raise ValueError(f"Unknown stat: {stat}")

            caps[curr_wk] = cap_val

        return caps


    # ============================================================
    # Incremental progressive Leiden (con cap_mode + tenure_mode)
    # ============================================================
    def compute_leiden_incremental_progressive(
        self,
        method="CPM",
        resolution_parameter=0.06,
        lambda_temporal=0.02,
        cap_mode="static",                 # off/static/dynamic
        cap_bonus=0.30,                    # cap static o fallback
        dynamic_cap_by_week=None,          # dict: week_str -> cap(t->week)
        tenure_mode="linear",              # none/linear/log/exp
        tenure_exp_k=0.15,
        debug_sample_nodes=None,
        max_edges=None,
        max_slices=None,
        n_iterations=2,
        edge_types_keep=None,              # (opzionale) filtri sugli archi delle slice (se esistesse 'type' nella slice)
        output_dir=None,
        run_tag=None
    ):
        """
        Incremental Leiden temporale:
        - slicing settimanale (iter_weekly_slices)
        - warm start
        - bonus temporale sugli archi tra nodi che erano nella stessa community a t-1
        - bonus = min( f(lambda, tenure), cap ), dove:
            cap = cap_bonus (static) oppure cap_dinamico(t->t+1) (dynamic)
        - scrittura progressive CSV: name,date,community
        """
        self.logger.info(f"Start Leiden incrementale ({method}) [progressive CSV mode]")
        self._print_memory_usage("Dopo setup iniziale")

        if output_dir is None:
            output_dir = "./output/leiden_runs"
        os.makedirs(output_dir, exist_ok=True)

        tag = run_tag if run_tag else f"rp{resolution_parameter}"
        master_file = os.path.join(output_dir, f"leiden_progressive_{tag}.csv")

        # header una sola volta
        if not os.path.exists(master_file):
            with open(master_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["name", "date", "community"])

        prev_comm_by_name = {}
        tenure_by_name = {}
        memberships = []
        dates = []
        processed = 0

        slices_iter = self.iter_weekly_slices(self.data_graph)

        for week_str, G in slices_iter:
            if max_slices is not None and processed >= max_slices:
                break
            processed += 1
            dates.append(week_str)
            self._print_memory_usage(f"Prima della slice {week_str}")

            # (opzionale) filtri archi per tipo/top-k (nota: 'type' non esiste nelle slice ricostruite)
            if (edge_types_keep is not None and "type" in G.es.attributes()) or (max_edges is not None):
                mask = np.ones(G.ecount(), dtype=bool)

                if edge_types_keep is not None and "type" in G.es.attributes():
                    et = np.array([str(t) for t in G.es["type"]])
                    mask &= np.isin(et, list(edge_types_keep))

                if max_edges is not None and G.ecount() > max_edges:
                    w = np.array(G.es["weight"], dtype=float) if "weight" in G.es.attribute_names() else np.ones(G.ecount())
                    topk = np.argsort(-w)[:int(max_edges)]
                    m2 = np.zeros(G.ecount(), dtype=bool)
                    m2[topk] = True
                    mask &= m2

                idx_keep = np.where(mask)[0]
                G = G.subgraph_edges(idx_keep, delete_vertices=False)
                self.logger.info(f"[{week_str}] Filtered edges -> {G.vcount():,} nodes, {G.ecount():,} edges")

            # debug subsample
            if debug_sample_nodes is not None and G.vcount() > int(debug_sample_nodes):
                G = G.induced_subgraph(range(int(debug_sample_nodes)))
                self.logger.info(f"[{week_str}] DEBUG sample nodes -> {G.vcount():,} nodes, {G.ecount():,} edges")

            # ensure weights
            if "weight" not in G.es.attribute_names():
                G.es["weight"] = [1.0] * G.ecount()
            else:
                G.es["weight"] = [float(w) if w is not None else 1.0 for w in G.es["weight"]]

            names = np.array(G.vs["name"], dtype=object)

            # warm start
            initial_membership = None
            if prev_comm_by_name:
                next_label = max(prev_comm_by_name.values(), default=-1) + 1
                init_labels = np.full(len(names), -1, dtype=int)
                for i, n in enumerate(names):
                    if n in prev_comm_by_name:
                        init_labels[i] = prev_comm_by_name[n]
                    else:
                        init_labels[i] = next_label
                        next_label += 1
                _, initial_membership = np.unique(init_labels, return_inverse=True)
                initial_membership = initial_membership.tolist()

            # temporal bonus
            boosted = 0
            total_bonus = 0.0

            if prev_comm_by_name and cap_mode != "off" and lambda_temporal > 0 and G.ecount() > 0:
                edges = np.asarray(G.get_edgelist(), dtype=np.int32)
                src, dst = edges[:, 0], edges[:, 1]
                name_to_idx = {n: i for i, n in enumerate(names)}

                comm = np.full(len(names), -1, dtype=np.int32)
                ten = np.zeros(len(names), dtype=np.float32)

                for n, c in prev_comm_by_name.items():
                    i = name_to_idx.get(n)
                    if i is not None:
                        comm[i] = c
                for n, t in tenure_by_name.items():
                    i = name_to_idx.get(n)
                    if i is not None:
                        ten[i] = t

                cu, cv = comm[src], comm[dst]
                tau = np.minimum(ten[src], ten[dst])
                stable_mask = (cu == cv) & (cu >= 0)

                if np.any(stable_mask):
                    # cap da usare nella settimana corrente (cap della transizione t-1 -> t)
                    cap_value = float(cap_bonus)
                    if cap_mode == "dynamic" and dynamic_cap_by_week is not None:
                        cap_value = float(dynamic_cap_by_week.get(str(week_str), cap_bonus))

                    bonuses = self._tenure_bonus(
                        tau=tau[stable_mask],
                        lambda_temporal=lambda_temporal,
                        cap_value=cap_value,
                        tenure_mode=tenure_mode,
                        tenure_exp_k=tenure_exp_k
                    )

                    w = np.asarray(G.es["weight"], dtype=np.float32)
                    w[stable_mask] += bonuses
                    G.es["weight"] = w.tolist()

                    boosted = int(stable_mask.sum())
                    total_bonus = float(bonuses.sum())

            if boosted:
                self.logger.info(f"[{week_str}] boosted_edges={boosted:,}, bonus~{total_bonus:.2f}")

            # run Leiden
            self.logger.info(f"[{week_str}] start Leiden (nodes={G.vcount():,}, edges={G.ecount():,})")
            partition_cls = la.CPMVertexPartition if method == "CPM" else la.ModularityVertexPartition

            part = la.find_partition(
                G,
                partition_cls,
                weights="weight",
                resolution_parameter=resolution_parameter if method == "CPM" else None,
                initial_membership=initial_membership,
                n_iterations=int(n_iterations),
                seed=42
            )

            curr_labels = np.array(part.membership, dtype=int)
            self.logger.info(f"Leiden {method} quality = {part.quality():.4f}")

            # align labels
            if prev_comm_by_name:
                curr_labels = np.array(self._relabel_with_overlap(prev_comm_by_name, names, curr_labels), dtype=int)

            memberships.append(list(curr_labels))

            # progressive csv append
            with open(master_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for n, lab in zip(names, curr_labels):
                    writer.writerow([n, str(week_str), int(lab)])

            # update tenure + prev labels
            new_prev = {}
            new_ten = {}
            for n, lab in zip(names, curr_labels):
                old = prev_comm_by_name.get(n)
                if old is not None and old == int(lab):
                    new_ten[n] = tenure_by_name.get(n, 0) + 1
                else:
                    new_ten[n] = 0
                new_prev[n] = int(lab)

            prev_comm_by_name = new_prev
            tenure_by_name = new_ten

            self._print_memory_usage(f"Dopo la slice {week_str}")

        return memberships, dates


    # Runner per GraphAnalysis: esegue UNA run

    def run_temporal_experiment(
        self,
        method,
        resolution_parameter,
        lambda_temporal,
        cap_mode,                 # off/static/dynamic
        cap_bonus,
        tenure_mode,              # none/linear/log/exp
        tenure_exp_k,
        dynamic_cap_conf,         # dict da prop.json
        n_iterations,
        max_slices,
        debug_sample_nodes,
        max_edges,
        output_dir,
        run_tag
    ):
        """
        Esegue la run e salva:
          - leiden_progressive_<run_tag>.csv
        """
        os.makedirs(output_dir, exist_ok=True)

        dynamic_cap_by_week = None
        if cap_mode == "dynamic":
            enabled = bool(dynamic_cap_conf.get("enabled", True))
            if not enabled:
                # se chiedi cap_mode dynamic ma dynamic_cap disabled, usiamo fallback cap_bonus
                dynamic_cap_by_week = None
            else:
                retweet_edge_types = dynamic_cap_conf.get("retweet_edge_types", ["0"])
                stat = dynamic_cap_conf.get("stat", "median")
                fallback = float(cap_bonus) if bool(dynamic_cap_conf.get("fallback_to_cap_bonus", True)) else 0.0

                dynamic_cap_by_week = self.precompute_dynamic_cap_dan(
                    max_slices=max_slices,
                    retweet_edge_types=retweet_edge_types,
                    stat=stat,
                    fallback=fallback
                )


        memberships, dates = self.compute_leiden_incremental_progressive(
            method=method,
            resolution_parameter=float(resolution_parameter),
            lambda_temporal=float(lambda_temporal),
            cap_mode=cap_mode,
            cap_bonus=float(cap_bonus),
            dynamic_cap_by_week=dynamic_cap_by_week,
            tenure_mode=tenure_mode,
            tenure_exp_k=float(tenure_exp_k),
            debug_sample_nodes=debug_sample_nodes,
            max_edges=max_edges,
            max_slices=max_slices,
            n_iterations=int(n_iterations),
            output_dir=output_dir,
            run_tag=run_tag
        )


        
        
        

        return {
            "run_tag": run_tag,
            "method": method,
            "resolution_parameter": float(resolution_parameter),
            "lambda_temporal": float(lambda_temporal),
            "cap_mode": cap_mode,
            "cap_bonus": float(cap_bonus),
            "tenure_mode": tenure_mode,
            "tenure_exp_k": float(tenure_exp_k),
            "max_slices": max_slices
        }

    #funzione incremental
   
    def compute_leiden_temporal_incremental(
        self,
        method="CPM",
        resolution_parameter_range=(0.1, 1.0),
        lambda_temporal=0.1,
        cap_bonus=1,
        n_iterations=10
    ):
        """
        Rimane per compatibilità: prova 10 rp.
        Usa: cap static, tenure linear (come prima).
        """
        all_memberships = defaultdict(list)

        for rp in np.linspace(resolution_parameter_range[0], resolution_parameter_range[1], num=10):
            start = time.time()
            rp_round = round(float(rp), 1)

            memberships, dates = self.compute_leiden_incremental_progressive(
                method=method,
                resolution_parameter=rp_round,
                lambda_temporal=lambda_temporal,
                cap_mode="static",
                cap_bonus=cap_bonus,
                tenure_mode="linear",
                n_iterations=n_iterations,
                output_dir="./output/leiden_runs",
                run_tag=f"legacy_rp{rp_round}"
            )

            all_memberships[rp_round] = memberships
            series_per_node = list(zip(*all_memberships[rp_round]))
            self.data_graph.vs[f"{rp_round}"] = [
                {date: int(lbl.item() if isinstance(lbl, np.generic) else lbl) for date, lbl in zip(dates, labels)}
                for labels in series_per_node
            ]

            end = time.time()
            self.logger.info(f"Finished Leiden incremental (legacy) rp={rp_round} elapsed={end-start:.2f}s")
            yield rp_round

    
    # Multilayer temporal 
   
    def compute_leiden_temporal(self, resolution_parameter_range=(0.4, 0.5)):
        self.logger.info("Start Leiden computation for temporal data")
        all_memberships = defaultdict(list)

        if "id" not in self.data_graph.vs.attribute_names():
            self.data_graph.vs["id"] = self.data_graph.vs["name"]

        weekly_slices = self.build_weekly_slices(self.data_graph)
        dates = [d for d, _ in weekly_slices]
        graphs = [G for _, G in weekly_slices]

        for rp in np.linspace(float(resolution_parameter_range[0]), float(resolution_parameter_range[1]), num=1):
            start = time.time()
            rp_round = round(float(rp), 1)

            memberships, dQ = la.find_partition_temporal(
                graphs,
                la.CPMVertexPartition,
                interslice_weight=0.1,
                resolution_parameter=rp_round,
                seed=42
            )
            all_memberships[rp_round] = memberships
            series_per_node = list(zip(*all_memberships[rp_round]))

            self.data_graph.vs[f"{rp_round}"] = [
                {date: int(lbl) for date, lbl in zip(dates, labels)}
                for labels in series_per_node
            ]
            self.data_graph[f"cpm_quality_{rp_round}"] = dQ

            end = time.time()
            self.logger.info(f"Finished temporal Leiden rp={rp_round} elapsed={end-start:.2f}s")
            yield rp_round

   
    # Static Leiden
    
    def compute_leiden(self, resolution_parameter_range=(0.1, 1.0), number_of_resolutions=10):
        self.logger.info("Start Leiden computation")
        for rp in np.linspace(resolution_parameter_range[0], resolution_parameter_range[1], num=number_of_resolutions):
            start = time.time()
            rp_round = round(float(rp), 1)

            self.logger.info(f"Starting Leiden CPM rp={rp_round}")
            partition = la.find_partition(
                self.data_graph,
                la.CPMVertexPartition,
                resolution_parameter=rp_round,
                weights="weight",
                seed=42
            )
            cpm = partition.quality()

            self.data_graph.vs[f"{rp_round}"] = partition.membership
            self.data_graph["cpm_quality"] = cpm

            end = time.time()
            self.logger.info(f"Finished Leiden rp={rp_round} elapsed={end-start:.2f}s")
            yield rp_round

  
    # Export
    
    def export_partition(self, g, resolution_parameter, file_path, attr=None):
        Writer.create_dir(file_path, self.id)
        out = os.sep.join([file_path, self.id, f"communities_rp_{resolution_parameter}.csv"])
        Writer.export_nodes_with_attributes(g, out, attr)

    def export_graph(self, g, file_path):
        Writer.create_dir(file_path, self.id)
        out = os.sep.join([file_path, self.id, "nodes_with_communities.csv"])
        Writer.export_nodes_with_attributes(g, out)

    
    # Helper (usata da compute_leiden_temporal): build_weekly_slices
  
    def build_weekly_slices(self, g):
        weekly_edge_weights = defaultdict(lambda: defaultdict(int))
        for e in g.es:
            ts = dt.datetime.utcfromtimestamp(e["time"])
            year, week, _ = ts.isocalendar()
            weekly_edge_weights[(year, week)][e.tuple] += e["weight"] if "weight" in e.attributes() else 1

        slices = []
        for (year, week), edge_dict in sorted(weekly_edge_weights.items()):
            edge_list = list(edge_dict.keys())
            weight_list = list(edge_dict.values())
            Gw = ig.Graph(n=g.vcount(), edges=edge_list, directed=True)
            for attr in ("id", "name", "type"):
                if attr in g.vs.attribute_names():
                    Gw.vs[attr] = list(g.vs[attr])
            Gw.vs["slice"] = [f"{year}-W{week:02d}"] * Gw.vcount()
            Gw.es["weight"] = weight_list
            slices.append((f"{year}-W{week:02d}", Gw))
        return slices
