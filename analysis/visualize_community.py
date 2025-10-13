#!/usr/bin/env python3
"""
visualize_community.py

Extracts and visualizes the internal graph structure (retweet and reply) for a
single, specified community from your Leiden analysis. Edge thickness is now
based on the interaction weight, and reciprocal edges are drawn as separate arcs.

Usage:
    python visualize_community.py \\
        --leiden_csv /path/to/retweet_leiden.csv \\
        --retweet_edges /path/to/retweet_edges.csv \\
        --reply_edges /path/to/reply_edges.csv \\
        --community_id 1023 \\
        --resolution 0.9 \\
        --outdir community_visuals
"""
import argparse
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt

def get_community_members(leiden_df, community_id, resolution):
    """Gets the set of user names for a specific community ID and resolution."""
    res_str = str(resolution)
    comm_id_str = str(community_id)
    
    # Filter for the specific community and resolution
    members = leiden_df[
        (leiden_df['type'].str.lower() == 'u') &
        (leiden_df[res_str] == comm_id_str)
    ]['name'].tolist()
    
    return set(members)

def create_and_draw_subgraph(members, edge_file, outfile, graph_type, args, sep=',', type='retweet'):
    """
    Creates an induced subgraph for a set of members from an edge file and
    draws it, with edge widths based on interaction weight.
    """
    print(f"  > Building {graph_type} subgraph for {len(members)} members...")
    
    # Using a directed graph as retweets and replies have a clear direction
    G = nx.DiGraph()
    G.add_nodes_from(members)
    
    edges_found = 0
    with open(edge_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # Assumes format: src,dst,weight
                parts = line.strip().split(sep)
                if len(parts) < 3: # Ensure weight is present
                    continue
                source, target, weight_str = parts[1], parts[2], parts[3]
                weight = float(weight_str)
                
                # Check if both source and target are in our community
                if source in members and target in members:
                    G.add_edge(source, target, weight=weight)
                    edges_found += 1
            except (ValueError, IndexError):
                # Skip malformed lines
                continue
    
    if edges_found == 0:
        print(f"  > Warning: Found no internal {graph_type} edges for this community.")
        return

    print(f"  > Found {G.number_of_nodes()} nodes and {G.number_of_edges()} internal edges.")

    # --- Visualization ---
    print(f"  > Generating {graph_type} plot...")
    fig, ax = plt.subplots(figsize=(15, 15))
    
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    in_degrees = [G.in_degree(n) for n in G.nodes()]
    node_sizes = [d * 2 + 20 for d in in_degrees]

    # --- Separate edges into regular and self-loops for custom drawing ---
    self_loops = list(nx.selfloop_edges(G, data=True))
    regular_edges = [e for e in G.edges(data=True) if not e[0] == e[1]]

    # --- Draw nodes first ---
    n_color = 'skyblue' if type == 'reply' else 'red'
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_size=node_sizes,
                           node_color='white',
                           edgecolors=n_color,
                           linewidths=1.5)

    # --- Draw REGULAR edges with a slight curve ---
    if regular_edges:
        regular_weights = [data['weight'] for _, _, data in regular_edges]
        max_w_reg = max(regular_weights) if regular_weights else 1.0
        regular_widths = [0.2 + 2.8 * (w / max_w_reg) for w in regular_weights]
        regular_arrow_sizes = [10 + w * 5 for w in regular_widths]

        nx.draw_networkx_edges(G, pos, ax=ax,
                               edgelist=[(u, v) for u, v, _ in regular_edges],
                               node_size=node_sizes,
                               width=regular_widths,
                               edge_color='gray',
                               arrowsize=regular_arrow_sizes,
                               connectionstyle='arc3,rad=0.1')

    # --- Draw SELF-LOOP edges with a large curve to make them visible ---
    if self_loops:
        loop_weights = [data['weight'] for _, _, data in self_loops]
        max_w_loop = max(loop_weights) if loop_weights else 1.0
        loop_widths = [0.2 + 2.8 * (w / max_w_loop) for w in loop_weights]
        
        nx.draw_networkx_edges(G, pos, ax=ax,
                               edgelist=[(u, v) for u, v, _ in self_loops],
                               node_size=node_sizes,
                               width=loop_widths,
                               edge_color='gray',  # Make self-loops stand out
                               arrowsize=10,
                               # A large radius creates a visible loop for self-edges
                               connectionstyle='arc3,rad=0.6')

    #ax.set_title(f'Internal {graph_type} Structure of Community {args.community_id} (Res {args.resolution})', fontsize=20)
    ax.axis('off')
    plt.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"  > Plot saved to: {outfile}")


def main(args):
    """
    Main function to orchestrate the visualization process.
    """
    os.makedirs(args.outdir, exist_ok=True)

    # --- 1. Load Leiden data and get community members ---
    try:
        print("Loading Leiden CSV:", args.leiden_csv)
        df_leiden = pd.read_csv(args.leiden_csv, dtype=str, keep_default_na=False)
    except FileNotFoundError as e:
        print(f"Error: {e}. Exiting.")
        return
        
    print(f"\nFinding members for Community '{args.community_id}' at Resolution '{args.resolution}'...")
    members = get_community_members(df_leiden, args.community_id, args.resolution)
    
    if not members:
        print("\nError: Could not find any members for this community. Please check your inputs.")
        return
    print(f"Found {len(members)} members.")
    
    # --- 2. Create and draw the retweet subgraph ---
    print("\n--- Processing Retweet Graph ---")
    try:
        outfile_rt = os.path.join(args.outdir, f"community_{args.community_id}_res_{args.resolution}_retweets.png")
        create_and_draw_subgraph(members, args.retweet_edges, outfile_rt, "Retweet", args, sep=args.sep, type='retweet')
    except FileNotFoundError as e:
        print(f"Error: Retweet edge file not found at '{args.retweet_edges}'. Skipping.")
        
    # --- 3. Create and draw the reply subgraph ---
    print("\n--- Processing Reply Graph ---")
    try:
        outfile_reply = os.path.join(args.outdir, f"community_{args.community_id}_res_{args.resolution}_replies.png")
        create_and_draw_subgraph(members, args.reply_edges, outfile_reply, "Reply", args, sep=args.sep, type='reply')
    except FileNotFoundError as e:
        print(f"Error: Reply edge file not found at '{args.reply_edges}'. Skipping.")
    
    print("\n✅ Visualization complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the internal structure of a specific community.")
    parser.add_argument('--leiden_csv', required=True, help='Path to the source (e.g., retweet) Leiden CSV to get community members.')
    parser.add_argument('--retweet_edges', required=True, help='Path to the user-user retweet edge file (e.g., src,dst,weight).')
    parser.add_argument('--reply_edges', required=True, help='Path to the user-user reply edge file.')
    parser.add_argument('--community_id', required=True, type=str, help='The ID of the community you want to visualize.')
    parser.add_argument('--resolution', required=True, type=str, help='The resolution at which the community was identified (e.g., "0.9").')
    parser.add_argument('--outdir', required=True, help='Directory to save the output plots.')
    parser.add_argument('--sep', default=',', help='Separator for the edge files (default: ",").')
    args = parser.parse_args()

    main(args)

