# -*- coding: utf-8 -*-
import concurrent.futures
import logging
import os
import time
import uuid

import igraph as ig
import leidenalg as la
import numpy as np
from collections import defaultdict
import datetime as dt

from Utils.Writer import Writer


class Leiden:
    logger = logging.getLogger('Leiden')

    def __init__(self, data_graph):
        self.id = uuid.uuid1().hex
        self.data_graph = data_graph
    
    def _print_memory_usage(self, note=""):
        try:
            import psutil, os as _os
            process = psutil.Process(_os.getpid())
            mem_mb = process.memory_info().rss / (1024 * 1024)
            self.logger.info(f"[MEM] {note} → {mem_mb:.2f} MB")
        except Exception:
            # psutil non disponibile: fallback silenzioso
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
        # Specify the type of nodes to exclude
        # Get the indices of nodes to keep (i.e., nodes not of the excluded type)
        # Create the subgraph with the selected nodes
        exclude_type = "h"
        nodes_to_keep = [v.index for v in data_graph.vs if v["type"] != exclude_type]
        subgraph = data_graph.induced_subgraph(nodes_to_keep)

        self.logger.info("Start computation of PageRank")

        start = time.time()
        pagerank_values = subgraph.pagerank(directed=True, weights='weight', implementation="prpack")
        end = time.time()

        # Map the PageRank values back to the original graph
        # Initialize all PageRank values in the original graph to None
        data_graph.vs["pagerank"] = [None] * data_graph.vcount()
        # Copy the PageRank values from the subgraph back to the original graph
        for subgraph_node, pagerank in zip(subgraph.vs, pagerank_values):
            original_index = subgraph.vs["name"].index(subgraph_node["name"])  # Get the original index from the subgraph
            data_graph.vs[original_index]["pagerank"] = pagerank
        
        self.logger.info('Computation of PageRank completed!')
        self.logger.info("Elapsed time: " + str(end - start))


    def build_daily_slices(self, g, first_n_days=10):
        """
        Builds daily slices of a temporal graph by grouping edges by their timestamp.
        Each slice is a subgraph representing the edges that occurred on a specific day.

        :param g: The temporal graph to slice, an igraph.Graph instance.
        :return: A list of tuples, each containing a date and the corresponding daily subgraph.
        :raises ValueError: If the graph does not have a 'time' attribute for its edges.
        :raises RuntimeError: If the graph is not directed or if the 'time' attribute is not present in the edges.
        :raises Exception: If the graph is not connected or if the partitioning fails.
        """
        # Group edges by day
        edges_by_day = defaultdict(list)
        for e in g.es:
            day = dt.datetime.utcfromtimestamp(e['time']).date()
            edges_by_day[day].append((e.tuple, e['weight']))

        # Create daily slices
        if first_n_days > 0:
            sorted_days = sorted(edges_by_day)[:first_n_days] # Get the first 14 days with activity
            self.logger.info(f"Using the first {first_n_days} days with activity: {[day.isoformat() for day in sorted_days]}")
        else:
            sorted_days = sorted(edges_by_day)
            self.logger.info(f"Using all days with activity")

        slices = []
        for day in sorted_days:                # chronological order
            edge_tuples = [tpl for tpl, _ in edges_by_day[day]]
            weights     = [w for _, w in edges_by_day[day]]

            # Create a subgraph for the day
            Gd = ig.Graph(n=g.vcount(), edges=edge_tuples, directed=g.is_directed())

            # copy attributes of vertex
            Gd.vs['id']    = g.vs['id']
            Gd.vs['name']  = g.vs['name']
            Gd.vs['type']  = g.vs['type']
            Gd.vs['slice'] = [day.isoformat()] * Gd.vcount()   # add slice date to vertices

            # copy attributes of edge
            Gd.es['weight'] = weights

            slices.append((day, Gd))
        return slices

    def iter_weekly_slices(self, g):
        """
        Lazily yields weekly slices of a temporal directed graph by grouping edges by ISO calendar week.
        For each week, edges with the same source and target are aggregated by summing their weights.

        :param g: A directed igraph.Graph with edge attributes 'time' (Unix timestamp) and optionally 'weight'.
        :yield: (week_str, igraph.Graph for that week), where week_str = "YYYY-Www".
        """
        # Aggregate edge weights per week
        weekly_edge_weights = defaultdict(lambda: defaultdict(float))

        for e in g.es:
            ts = dt.datetime.fromtimestamp(e["time"], tz=dt.timezone.utc)
            year, week, _ = ts.isocalendar()
            week_str = f"{year}-W{week:02d}"
            edge_key = e.tuple

            weight = e["weight"] if "weight" in e.attributes() and e["weight"] is not None else 1.0
            weekly_edge_weights[week_str][edge_key] += float(weight)

        # Yield graphs one by one
        for week_str, edge_dict in sorted(weekly_edge_weights.items()):
            edge_list = list(edge_dict.keys())
            weight_list = list(edge_dict.values())

            Gw = ig.Graph(n=g.vcount(), edges=edge_list, directed=True)

            # Copy all vertex attributes
            for attr in g.vs.attributes():
                Gw.vs[attr] = g.vs[attr]

            Gw.vs["slice"] = [week_str] * Gw.vcount()
            Gw.es["weight"] = weight_list

            yield week_str, Gw

    def build_weekly_slices(self, g):
        """
        Builds weekly slices of a temporal directed graph by grouping edges by ISO calendar week.
        For each week, edges with the same source and target are aggregated by summing their weights.

        :param g: A directed igraph.Graph with edge attributes 'time' (Unix timestamp) and 'weight'.
        :return: A list of tuples: ((year, week_number), igraph.Graph for that week).
        """
        # Aggregate edge weights per week
        weekly_edge_weights = defaultdict(lambda: defaultdict(int))  # {(year, week): {(src, tgt): total_weight}}

        for e in g.es:
            ts = dt.datetime.fromtimestamp(e["time"], tz=dt.timezone.utc)
            year, week, _ = ts.isocalendar()
            week_str = f"{year}-W{week:02d}"
            edge_key = e.tuple  # directed edge (src, tgt)

            # Use weight if present, otherwise default to 1.0
            weight = e["weight"] if "weight" in e.attributes() and e["weight"] is not None else 1.0
            weekly_edge_weights[week_str][edge_key] += float(weight)

        # Build graph per week
        slices = []
        for week_str, edge_dict in sorted(weekly_edge_weights.items()):
            edge_list = list(edge_dict.keys())
            weight_list = list(edge_dict.values())

            Gw = ig.Graph(n=g.vcount(), edges=edge_list, directed=True)

            # Copy vertex attributes
            for attr in g.vs.attributes():
                Gw.vs[attr] = g.vs[attr]
            
            Gw.vs["slice"] = [week_str] * Gw.vcount()

            # Copy edge weights
            Gw.es['weight'] = weight_list

            slices.append((week_str, Gw))

        return slices

    def collapse_nodes(self, labels, min_size=30, dummy=-1):
        """
        Collapses nodes in the graph based on their labels, keeping only those with a size greater than or equal to `min_size`.
        Nodes that do not meet this criterion are replaced with a dummy value.
        This method is useful for simplifying the graph by merging less significant nodes into a single dummy community.

        :param labels: A list or array-like structure containing the labels of the nodes.
        :param min_size: The minimum size for a label to be retained. Nodes with fewer than `min_size` occurrences will be replaced with the dummy value.
        :param dummy: The value to replace nodes community that do not meet the `min_size` criterion.
        """
        self.logger.info("Start collapsing nodes")
        vc = pd.Series(labels).value_counts()
        big = vc[vc >= min_size].index
        self.logger.info("Finished collapsing nodes")
        return np.where(pd.Series(labels).isin(big), labels, dummy)

    def _relabel_with_overlap(self, prev_comm_by_name, curr_names, curr_labels):
        
        """Realigns current community labels to match previous labels by maximizing overlap.

        This method ensures temporal consistency of community labels. It maps the
        labels from the current time slice to the labels from the previous slice
        based on the number of shared members. A greedy approach is used, where
        the largest overlapping communities are matched first. New communities that
        cannot be matched to a previous one are assigned a new, unused integer label.

        Args:
            prev_comm_by_name (dict): A mapping from node name to community label
                from the previous time slice.
            curr_names (list): The list of node names in the current time slice.
            curr_labels (list): The list of community labels for the current nodes,
                as assigned by the community detection algorithm.

        Returns:
            list: A new list of labels for the current nodes, relabeled to be
                consistent with the previous time slice.
        """
        #  compute overlap counts in a single pass
        overlap_counts = defaultdict(lambda: defaultdict(int))
        for name, curr_label in zip(curr_names, curr_labels):
            prev_label = prev_comm_by_name.get(name)
            if prev_label is not None:
                overlap_counts[curr_label][prev_label] += 1

        # Find the best previous label for each new community
        overlaps = []
        for lab_curr, prev_counts in overlap_counts.items():
            if prev_counts:
                # Find the previous label with the maximum number of shared members
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

    def compute_leiden_incremental(self,
                                   method="CPM",
                                   resolution_parameter=0.06,
                                   lambda_temporal=0.02,
                                   cap_bonus=0.30,
                                   debug_sample_nodes=None,
                                   max_edges=None,
                                   max_slices=None,
                                   n_iterations=2,
                                   edge_types_keep=None):
        """Performs incremental community detection with a temporal bonus.

        This method processes a temporal graph slice by slice. For each slice, it runs
        the Leiden algorithm, incorporating information from the previous slice to
        ensure temporal stability of the communities. This is achieved through:
        1.  Warm Start: Using the previous slice's partition as the initial state
            for the current slice's optimization.
        2.  Temporal Bonus: A vectorized bonus is added to the weights of edges
            connecting nodes that were in the same community previously. The bonus
            increases with "tenure" i.e., the number of consecutive slices the
            nodes have shared a community.

        The final community labels are realigned across slices to maintain consistent
        identifiers for continuous communities over time.

        Args:
            method (str): The quality function for Leiden ('CPM' or 'Modularity').
            resolution_parameter (float): The resolution parameter for the CPM method.
            lambda_temporal (float): The base factor for the temporal bonus.
            cap_bonus (float): The maximum bonus an edge can receive from the
                temporal smoothing logic.
            debug_sample_nodes (int, optional): If set, randomly samples this
                many nodes for faster debugging. Defaults to None.
            max_edges (int, optional): If set, keeps only the top N edges by
                weight for faster debugging. Defaults to None.
            max_slices (int, optional): If set, processes only this many time
                slices. Defaults to None.
            n_iterations (int): Number of iterations for the Leiden algorithm.
            edge_types_keep (list, optional): A list of edge type strings to
                keep; all others are filtered out. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - memberships (list): A list of tuples, where each tuple represents a
                  time slice and contains `(list_of_names, list_of_labels)`.
                - dates (list): A list of datetime objects corresponding to each slice.
        """
                
        self.logger.info(f"Start Leiden incrementale ({method})")
        self._print_memory_usage("Dopo setup iniziale")

        prev_comm_by_name = {} # serve a mappare nodo nella community precedente
        tenure_by_name = {} # serve a mappare nodo nei valori di tenure
        memberships = [] # questa e le seguenti per raccogliere l'andamento
        dates = []

        processed = 0

        slices_iter = self.iter_weekly_slices(self.data_graph)

        for date_val, G in slices_iter: # Loop sulle slice (una per volta)
            if max_slices is not None and processed >= max_slices:  # interrompe se supera max_slices (parametro)
                break
            processed += 1

            # salva la data della slice corrente (Viene dal generatore read_slices_streaming / merge_slices_by_date, 
            # che per ogni sotto-grafo ti dà (data, grafo).), 
            # la aggiunge a dates in modo che alla fine abbiamo Alla fine, 
            # dates conterrà tutte le date in cui hai analizzato una slice, nell’ordine giusto.

            dates.append(date_val)
            self._print_memory_usage(f"Prima della slice {date_val}")

            '''
                Filtro per tipo di arco (si imposta tramite parametro edge_types_keep).
                Normalizza type a int.
                Tiene solo gli archi con type desiderato.

                 Usa subgraph_edges(..., delete_vertices=False) per non perdere i nodi inattivi (grande per coerenza temporale).
            '''

            # filtro tipi di archi opzionale
            if edge_types_keep is not None and "type" in G.es.attributes():
                # filtra solo i tipi desiderati
                idx_keep = [e.index for e in G.es if str(e["type"]) in edge_types_keep]
                G = G.subgraph_edges(idx_keep, delete_vertices=False)

            names = G.vs["name"] #chiave univoca tra slice diverse 

            # sample nodi (debug)
            if debug_sample_nodes is not None and G.vcount() > debug_sample_nodes:
                G = G.induced_subgraph(range(debug_sample_nodes))
                names = G.vs["name"]
                self.logger.info(f"[{date_val}] DEBUG sample nodes -> {G.vcount():,} nodi, {G.ecount():,} archi")


            '''
                Tiene i top-K archi più pesanti (se max_edges è settato).
                Usa subgraph_edges(..., delete_vertices=False) per non perdere i nodi inattivi
            '''

            # limita archi (debug)
            if max_edges is not None and G.ecount() > max_edges:
                if "weight" in G.es.attribute_names():
                    weights = np.asarray([float(w) if w is not None else 1.0 for w in G.es["weight"]], dtype=float)
                else:
                    G.es["weight"] = [1.0] * G.ecount()
                    weights = np.asarray(G.es["weight"], dtype=float)
                idx = np.argsort(-weights)[:max_edges]
                G = G.subgraph_edges(idx.tolist(), delete_vertices=False)
                names = G.vs["name"]
                self.logger.info(f"[{date_val}] DEBUG sample edges -> {G.vcount():,} nodi, {G.ecount():,} archi")

            '''
            imposta tutti i pesi come float (se non ci sono, li crea tutti a 1.0).
            '''

            # pesi float
            if "weight" not in G.es.attribute_names():
                G.es["weight"] = [1.0] * G.ecount()
            else:
                G.es["weight"] = [float(w) if w is not None else 1.0 for w in G.es["weight"]]
            

            '''
            Parte fondamentale: inizializza leiden utilizzando le comunità precedenti (warm start).
            ES: senza questa parte, ogni slice sarebbe indipendente e non avrebbe continuità temporale, cioè le comunità avrebbero una label diversa ogni slice.
            '''

            # warm start
            initial_membership = None
            if prev_comm_by_name: # se abbiamo già la partizione del giorno precedente
                next_label = max(prev_comm_by_name.values(), default=-1) + 1
                init_labels = []
                for n in names:
                    label = prev_comm_by_name.get(n)
                    if label is None:
                        # Assign a new, unique label for each new node
                        init_labels.append(next_label)
                        next_label += 1
                    else:
                        init_labels.append(label)

                # Use np.unique for a highly efficient re-compacting of labels
                _, initial_membership = np.unique(init_labels, return_inverse=True)
                initial_membership = initial_membership.tolist()

            # bonus temporale sugli archi intra-comunità (stessa community precedente)

            '''
            Crea array comm (label del giorno precedente per ogni nodo attuale) e ten (tenure).

            prev_comm[n]: se il nodo esisteva ieri, prende la comunità; se no, -1 (significa “nessuna comunità precedente”).
            prev_tenure[n]: se il nodo esisteva ieri, prende la tenure; se no, 0.

            comm = np.array([prev_comm[n] for n in names]) → per ogni nodo nell’ordine della slice attuale, inserisce la label precedente.

            ten = np.array([prev_tenure[n] for n in names]) → stesso discorso ma con la tenure.

            mask: veri e falsi per ogni arco: True se i due nodi dell’arco erano nella stessa comunità ieri (cu==cv>=0).
            tau: tenure “di coppia”: prende il min fra le due tenure (conservativo). → più a lungo due nodi restano insieme, più bonus all’arco (fino a cap_bonus).
            Applica il bonus solo agli archi mask==True, riscrivendo i pesi.
            
            '''

            boosted = 0
            total_bonus = 0.0
            if prev_comm_by_name and lambda_temporal > 0 and G.ecount() > 0:
                edges = np.array(G.get_edgelist(), dtype=int)
                sources = edges[:, 0]
                targets = edges[:, 1]

                # build NumPy arrays, avoiding intermediate dictionaries
                comm = np.array([prev_comm_by_name.get(n, -1) for n in names], dtype=int)
                ten = np.array([tenure_by_name.get(n, 0) for n in names], dtype=int)

                cu = comm[sources]
                cv = comm[targets]
                tau = np.minimum(ten[sources], ten[targets])

                mask = (cu == cv) & (cu >= 0)

                '''
                    conta e logga quanti archi hanno ricevuto il bonus e la somma dei bonus applicati (fino a if boosted: self llogger.info(...)).
                    mask è un array booleano lungo #archi che dice per ogni arco se i due nodi erano nella stessa comunità ieri (True = arco idoneo al bonus).

                    mask.sum() → conta quanti True ci sono, cioè quanti archi hanno ricevuto un bonus.

                    bonuses.sum() → somma effettiva dei bonus aggiunti a quegli archi.
                '''

                if np.any(mask):
                    bonuses = np.minimum(lambda_temporal * (1 + tau[mask]), cap_bonus)
                    w = np.array(G.es["weight"], dtype=float)
                    w[mask] += bonuses
                    G.es["weight"] = w.tolist()

                    boosted = int(mask.sum())
                    total_bonus = float(bonuses.sum())

            if boosted:
                self.logger.info(f"[{date_val}] boosted_edges={boosted:,}, bonus≈{total_bonus:.2f}")

            # Leiden
            '''

            Sceglie la partizione (CPM o Modularity), usa sempre i pesi e il warm start se presente.

            seed=42 per riproducibilità.

            n_iterations: più alto → più rifinitura (ma occhio ai tempi).
                    
            '''
            self.logger.info(f"[{date_val}] start Leiden (nodi={G.vcount():,}, archi={G.ecount():,})")
            if method == "CPM":
                part = la.find_partition(
                    G,
                    la.CPMVertexPartition,
                    weights="weight",
                    resolution_parameter=resolution_parameter,
                    initial_membership=initial_membership,
                    n_iterations=n_iterations,
                    seed=42
                )
            elif method == "Modularity":
                part = la.find_partition(
                    G,
                    la.ModularityVertexPartition,
                    weights="weight",
                    initial_membership=initial_membership,
                    n_iterations=n_iterations,
                    seed=42
                )
            else:
                raise ValueError(f"Metodo non supportato: {method}")

            curr_labels = part.membership

            # relabel con overlap alle label precedenti

            '''
            _relabel_with_overlap: mappa le label nuove sulle vecchie cercando il matching con maggior overlap (greedy).
            Così “la comunità 5” di ieri resta “5” oggi, se c’è continuità di membri. Evita che le etichette saltino ogni giorno.
            '''


            if prev_comm_by_name:
                curr_labels = self._relabel_with_overlap(prev_comm_by_name, names, curr_labels)

            memberships.append((list(names), list(curr_labels)))

            # aggiorna tenure e label precedenti

            '''
            Se il nodo resta nella stessa comunità → tenure + 1, altrimenti 0.

            Aggiorna la mappa prev_comm_by_name con le label riallineate 
            (importante: il bonus del prossimo giorno userà queste).
            '''

            new_prev_comm_by_name = {}
            new_tenure_by_name = {}
            for n, lab in zip(names, curr_labels):
                old_lab = prev_comm_by_name.get(n)
                if old_lab is not None and old_lab == lab:
                    new_tenure_by_name[n] = tenure_by_name.get(n, 0) + 1
                else:
                    new_tenure_by_name[n] = 0
                new_prev_comm_by_name[n] = lab

            prev_comm_by_name = new_prev_comm_by_name
            tenure_by_name = new_tenure_by_name

            self._print_memory_usage(f"Dopo la slice {date_val}")

        return memberships, dates

    def compute_leiden_temporal_incremental(self,
                                        method="CPM",
                                        resolution_parameter_range=(0.4, 0.5),
                                        lambda_temporal=0.1,
                                        cap_bonus=1,
                                        n_iterations=10):
        """Computes incremental temporal community detection using the Leiden algorithm.

        This method iterates through a specified range of resolution parameters. For
        each parameter, it applies an incremental version of the Leiden algorithm
        to detect community structures over time.

        The resulting community membership for each node across different timestamps
        is stored as a new vertex attribute in `self.data_graph.vs`. The key for
        this new attribute is the string representation of the resolution
        parameter used for that computation. The value for each vertex is a
        dictionary mapping ISO-formatted dates to community labels.

        The method functions as a generator, yielding the key of the newly created
        vertex attribute after each iteration.

        Args:
            method (str): The quality function to use with the Leiden algorithm,
                such as "CPM" or "Modularity".
            resolution_parameter_range (tuple[float, float]): A tuple `(min, max)`
                defining the range of resolution parameters to test.
            lambda_temporal (float): The temporal smoothing parameter. Higher values
                increase the penalty for a node changing its community between
                consecutive timestamps.
            cap_bonus (int): The maximum bonus granted for temporal stability.
            n_iterations (int): The number of iterations for the Leiden algorithm
                at each temporal step.

        Yields:
            float: The resolution parameter used for the completed iteration. This
                value also serves as the key for the newly created vertex
                attribute (e.g., `data_graph.vs['0.4']`).
        """
        all_memberships = defaultdict(list)

        for rp in np.linspace(resolution_parameter_range[0], resolution_parameter_range[1], num=1):
            start = time.time()
            rp_round = round(rp, 1)

            memberships, dates = self.compute_leiden_incremental(method=method, 
                                                                 resolution_parameter=rp_round, 
                                                                 lambda_temporal=lambda_temporal, 
                                                                 cap_bonus=cap_bonus, 
                                                                 n_iterations=n_iterations)

            all_memberships[rp_round] = memberships
            series_per_node = list(zip(*all_memberships[rp_round]))
            self.data_graph.vs["{}".format(rp_round)] = [
                        {date.isoformat(): lbl for date, lbl in zip(dates, labels)}
                    for labels in series_per_node
                ]

            end = time.time()

            self.logger.info(
                "Finished Leiden with CPM Quality Function and resolution parameter = {}".format(rp_round))
            self.logger.info("Elapsed time: " + str(end - start))

            yield rp_round

    def compute_leiden_temporal(self, resolution_parameter_range=(0.4, 0.5)):
        """
        Computes the Leiden partitioning of a temporal graph using the CPM quality function.
        This method iterates over a range of resolution parameters, applies the Leiden algorithm,
        and updates the graph with the partitioning results for each time slice.

        :param data_graph: The temporal graph to partition, an igraph.Graph instance.
        :param resolution_parameter_range: A tuple specifying the range of resolution parameters to test.

        :return: Yields the resolution parameter used for each partitioning.
        :raises Exception: If the graph is not connected or if the partitioning fails.
        :raises ValueError: If the resolution parameter is not within the specified range.
        :raises RuntimeError: If the Leiden algorithm fails to compute a partition.
        :raises RuntimeError: If the Leiden algorithm fails to compute a partition for the temporal graph
        """
        self.logger.info("Start Leiden computation for temporal data")
        all_memberships = defaultdict(list) # dict: res → [ [labels], … ]

        if 'id' not in self.data_graph.vs.attribute_names():
            self.data_graph.vs['id'] = self.data_graph.vs['name']
        if True:
            self.logger.info("Start removing edges not equal to 0 or 2")
            for e in self.data_graph.es:
                print(f"Edge {e.index} type: {e['type']}, type is {type(e['type'])}")
            edges_to_keep = [e.index for e in self.data_graph.es if e["type"] == "0" or e["type"] == "2"] 
            self.logger.info("Number of edges to keep: {}".format(len(edges_to_keep)))
            self.data_graph = self.data_graph.subgraph_edges(edges_to_keep, delete_vertices=False)
            self.logger.info("Finished removing edges not equal to 0 or 2, created the subgraph")
        
        if True:
            # Build daily slices of the temporal graph
            self.logger.info(f"Building daily slices of the temporal graph with {self.data_graph.vcount()} nodes and {self.data_graph.ecount()} edges")
            daily_slices = self.build_daily_slices(self.data_graph)
            dates = [d for d, _ in daily_slices]
            self.logger.info("Number of daily slices: {}".format(len(daily_slices)))
            graphs = [G for _, G in daily_slices]
            self.logger.info("Number of graphs in daily slices: {}".format(len(graphs)))
        if False:
            self.logger.info(f"Building weekly slices of the temporal graph with {self.data_graph.vcount()} nodes and {self.data_graph.ecount()} edges")
            weekly_slices = self.build_weekly_slices(self.data_graph)
            # Extract (year, week) identifiers
            weeks = [f"{year}-W{week:02d}" for (year, week), _ in weekly_slices]
            self.logger.info("Number of weekly slices: {}".format(len(weekly_slices)))
            # Extract igraph.Graph objects only
            graphs = [G for _, G in weekly_slices]
            self.logger.info("Number of graphs in weekly slices: {}".format(len(graphs)))

        for rp in np.linspace(resolution_parameter_range[0], resolution_parameter_range[1], num=1):
            start = time.time()

            rp_round = round(rp, 1)
            self.logger.info("Starting Leiden with CPM Quality Function and resolution parameter = {}".format(rp_round))
            # Apply the Leiden algorithm to each daily slice
            # and collect the memberships for each resolution parameter
            memberships, dQ = la.find_partition_temporal(graphs, 
                                            la.CPMVertexPartition, 
                                            interslice_weight=0.2,
                                            resolution_parameter=rp_round,
                                            seed=42)
            all_memberships[rp_round] = memberships
            series_per_node = list(zip(*all_memberships[rp_round]))

            if True:
                self.data_graph.vs["{}".format(rp_round)] = [
                        {date.isoformat(): lbl for date, lbl in zip(dates, labels)}
                    for labels in series_per_node
                ]
            if False:
                # Convert (year, week) → "YYYY-Www" strings
                week_labels = [f"{year}-W{week:02d}" for (year, week), _ in weekly_slices]
                print("Week labels: ", week_labels)

                self.data_graph.vs["{}".format(rp_round)] = [
                    {week: lbl for week, lbl in zip(week_labels, labels)}
                    for labels in series_per_node
                ]

            self.data_graph[f"cpm_quality_{rp_round}"] = dQ
            self.logger.info("Leiden CPM quality value is: {}, resolution parameter is {}".format(dQ, rp_round))

            end = time.time()

            self.logger.info(
                "Finished Leiden with CPM Quality Function and resolution parameter = {}".format(rp_round))
            self.logger.info("Elapsed time: " + str(end - start))
            yield rp_round

    def compute_leiden(self, resolution_parameter_range=(0.1, 1.0), number_of_resolutions=10):
        """
        Computes the Leiden partitioning of the graph using the CPM quality function.
        This method iterates over a range of resolution parameters, applies the Leiden algorithm,
        and updates the graph with the partitioning results.
        
        :param resolution_parameter_range: A tuple specifying the range of resolution parameters to test.
        
        :return: Yields the resolution parameter used for each partitioning.
        :raises Exception: If the graph is not connected or if the partitioning fails.
        :raises ValueError: If the resolution parameter is not within the specified range.
        :raises RuntimeError: If the Leiden algorithm fails to compute a partition. 
        """
        self.logger.info("Start Leiden computation")
        for rp in np.linspace(resolution_parameter_range[0], resolution_parameter_range[1], num=number_of_resolutions):
            start = time.time()

            rp_round = round(rp, 1)
            self.logger.info("Starting Leiden with CPM Quality Function and resolution parameter = {}".format(rp_round))

            partition = la.find_partition(self.data_graph, la.CPMVertexPartition, resolution_parameter=rp_round,
                                          weights='weight',
                                          seed=0)
            cpm = partition.quality()

            self.data_graph.vs["{}".format(rp_round)] = partition.membership
            self.data_graph["cpm_quality"] = cpm
            self.logger.info("Leiden CPM quality value is: {}, resolution parameter is {}".format(cpm, rp_round))

            end = time.time()
            self.logger.info(
                "Finished Leiden with CPM Quality Function and resolution parameter = {}".format(rp_round))
            self.logger.info("Elapsed time: " + str(end - start))
            yield rp_round

    def export_partition(self, g, resolution_parameter, file_path, attr=None):
        """
        Exports the partition of the graph to a CSV file with nodes and their attributes.
        The file is saved in a directory structure based on the instance ID.
        The directory is created if it does not exist.
        The nodes are exported with their attributes, including 'id', 'name', 'type',
        and 'pagerank' if available.

        :param g: The graph to export, an igraph.Graph instance.
        :param resolution_parameter: The resolution parameter used for the partitioning.
        :param file_path: The base path where the partition will be exported.
        :param attr: Optional list of attributes to include in the export. If None, defaults
        """
        Writer.create_dir(file_path, self.id)
        file_path = os.sep.join([file_path, self.id, "communities_rp_{}.csv".format(resolution_parameter)])
        Writer.export_nodes_with_attributes(g, file_path, attr)

    def export_graph(self, g, file_path):
        """
        Exports the graph to a CSV file with nodes and their attributes.
        The file is saved in a directory structure based on the instance ID.
        The directory is created if it does not exist.
        The nodes are exported with their attributes, including 'id', 'name', 'type',
        and 'pagerank' if available.

        :param g: The graph to export, an igraph.Graph instance.
        :param file_path: The base path where the graph will be exported.
        """
        Writer.create_dir(file_path, self.id)
        file_path = os.sep.join([file_path, self.id, "nodes_with_communities.csv"])
        Writer.export_nodes_with_attributes(g, file_path)
