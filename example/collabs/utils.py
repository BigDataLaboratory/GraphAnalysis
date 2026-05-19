import os
import re
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter

current_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(current_dir, 'files')

def read_file(path):
    with open(os.path.join(folder, path), "r", encoding="utf-8") as f:
        return f.read()

def plot_extraction_progress(path):
    raw_string_data = read_file(path)
    # Split lines and filter out empty ones
    lines = [line.strip() for line in raw_string_data.strip().split('\n') if line.strip()]

    parsed_rows = []
    for line in lines:
        # Split by the pipe character and clean whitespace
        columns = [col.strip() for col in line.split('|')]
        if len(columns) == 4:
            batch_id = int(columns[0])
            timestamp = pd.to_datetime(columns[1]) # Properly parse Date + Time
            current_value = float(columns[2])
            accumulated_value = float(columns[3])

            parsed_rows.append([batch_id, timestamp, current_value, accumulated_value])

    # Create DataFrame and sort chronologically
    df = pd.DataFrame(parsed_rows, columns=['Batch', 'Timestamp', 'Value', 'Accumulated'])
    df = df.sort_values('Timestamp')

    # Plotting configuration (without using plt.figure to comply with environment constraints)
    ax = df.plot(
        x='Timestamp',
        y='Accumulated',
        kind='line',
        marker='o',
        color='#1f77b4',
        linewidth=2,
        figsize=(12, 6)
    )

    # Customizing the axes and titles
    ax.set_title("Tweet Extraction Progress: Evolution of Accumulated Tweets Over Time", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Time (YYYY-MM-DD HH:MM)", fontsize=11, labelpad=10)
    ax.set_ylabel("Accumulated Tweets Extracted", fontsize=11, labelpad=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    # Formatting X-axis dates to show date + time nicely without overlaps
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    plt.xticks(rotation=30, ha='right')

    # Tight layout to avoid truncated labels
    plt.tight_layout()

def human_format(num, pos=None):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.1f%s' % (num, ['', 'k', 'M', 'G', 'T'][magnitude])

def plot_extraction_dashboard(path, batch_label):
    raw_string = read_file(path)
    pattern = re.compile(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),\d+\s+-\s+GraphGenerationUser\s+-\s+INFO\s+-\s+\[Worker\s+(\d+).*?\]\s+(\d+)\s+users')

    worker_last_value = {} # Store the last value seen per worker
    data = []

    for line in io.StringIO(raw_string):
        match = pattern.search(line)
        if match:
            time = match.group(1)
            worker_id = match.group(2)
            current_total_for_worker = int(match.group(3))

            # Calculate the actual progress (the "delta")
            last_value = worker_last_value.get(worker_id, 0)
            delta = current_total_for_worker - last_value

            if delta > 0 or len(data) == 0:
                data.append({
                    'time': time,
                    'worker': f"W{worker_id}",
                    'count': delta  # Only add NEW users
                })
                worker_last_value[worker_id] = current_total_for_worker

    if not data:
        print(f"No data found for {batch_label}.")
        return

    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    df['cumulative'] = df['count'].cumsum()

    # 2. Setup Figure (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    fig.suptitle(f"User Extraction Dashboard ({batch_label})", fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(wspace=0.2)
    formatter = FuncFormatter(human_format)

    # --- LEFT GRAPH: Cumulative Progress ---
    ax1.plot(df['time'], df['cumulative'], color='#007acc', linewidth=3, label='Total Users')
    ax1.fill_between(df['time'], df['cumulative'], color='#007acc', alpha=0.1)
    ax1.set_title(f'[{batch_label}] User Extraction Cumulative Progress', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (HH:MM:SS)')
    ax1.set_ylabel('Total Extracted Users')
    ax1.yaxis.set_major_formatter(formatter)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- RIGHT GRAPH: Worker Performance ---
    worker_perf = df.groupby('worker')['count'].sum().sort_values(ascending=False)
    bars = worker_perf.plot(kind='bar', ax=ax2, color='#e67e22', edgecolor='black', width=0.8)
    ax2.set_title(f'[{batch_label}] Extracted Users per Worker', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Worker ID')
    ax2.set_ylabel('Total Extracted Users')
    ax2.yaxis.set_major_formatter(formatter)
    ax2.tick_params(axis='x', rotation=0)

    # Add labels on top of bars
    for bar in ax2.patches:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 human_format(bar.get_height()), ha='center', va='bottom', fontsize=10)

    plt.show()

    # Summary Print
    print(f"Summary for {batch_label}: {df['cumulative'].iloc[-1]:,} users in {len(df)} batches.")
    print("-" * 100)

def plot_community_distribution(csv_path):
    """
    Plots a multi-panel visual dashboard showing the distribution of user IDs 
    and community sizes from a community detection CSV output.
    """
    # 1. Load data safely
    if not os.path.isabs(csv_path):
        # Resolve relative to the current dir if not absolute
        csv_path = os.path.join(current_dir, csv_path)
        
    if not os.path.exists(csv_path):
        # Try resolving relative to project root
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        csv_path_alt = os.path.join(project_root, csv_path)
        if os.path.exists(csv_path_alt):
            csv_path = csv_path_alt
        else:
            print(f"Error: Community CSV file not found at '{csv_path}'.")
            return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Check columns
    required_cols = {'user_id', 'community'}
    if not required_cols.issubset(df.columns):
        print(f"Error: CSV must contain 'user_id' and 'community' columns. Found: {df.columns.tolist()}")
        return

    # Drop any nulls
    df = df.dropna(subset=['user_id', 'community'])
    df['community'] = df['community'].astype(str) # treat as category

    # Prepare figure layout
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # Sleek palette
    colors = ['#1e3d59', '#17b978', '#ff6e40', '#ffc13b', '#6200ea', '#00e5ff', '#ff007f']

    # --- PANEL 1: Top 10 Largest Communities ---
    ax_top = axs[0, 0]
    comm_counts = df['community'].value_counts()
    top_n = comm_counts.head(10)
    
    top_n.plot(kind='bar', ax=ax_top, color=colors[:len(top_n)], edgecolor='black', width=0.7)
    ax_top.set_title("Top 10 Largest Communities by User Count", fontsize=12, fontweight='bold', pad=10)
    ax_top.set_xlabel("Community ID", fontsize=10)
    ax_top.set_ylabel("Number of Users", fontsize=10)
    ax_top.grid(True, linestyle='--', alpha=0.5)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    
    # Add values on top of bars
    for bar in ax_top.patches:
        height = bar.get_height()
        ax_top.text(
            bar.get_x() + bar.get_width()/2, 
            height + (height * 0.01),
            f"{int(height):,}", 
            ha='center', 
            va='bottom', 
            fontsize=9,
            fontweight='semibold'
        )

    # --- PANEL 2: User ID Density (Chronological Account Age Distribution) ---
    ax_dist = axs[0, 1]
    # Human-readable formatter for large Twitter IDs
    def id_formatter(x, pos):
        if x >= 1e18:
            return f"{x/1e18:.1f}E18"
        elif x >= 1e15:
            return f"{x/1e15:.1f}E15"
        elif x >= 1e9:
            return f"{x/1e9:.1f}B"
        return f"{x:.0f}"

    # Draw User ID distribution histogram
    df['user_id'].plot(kind='hist', bins=50, ax=ax_dist, color='#17b978', edgecolor='white', alpha=0.8)
    ax_dist.set_title("Global User ID Distribution (Account Age)", fontsize=12, fontweight='bold', pad=10)
    ax_dist.set_xlabel("User ID (Larger IDs = Newer Accounts)", fontsize=10)
    ax_dist.set_ylabel("Frequency", fontsize=10)
    ax_dist.xaxis.set_major_formatter(FuncFormatter(id_formatter))
    ax_dist.grid(True, linestyle='--', alpha=0.5)
    ax_dist.spines['top'].set_visible(False)
    ax_dist.spines['right'].set_visible(False)

    # --- PANEL 3: Boxplot of User IDs within Top 5 Communities (Age Profile) ---
    ax_box = axs[1, 0]
    top_5_comms = comm_counts.head(5).index.tolist()
    sub_df = df[df['community'].isin(top_5_comms)].copy()
    
    # Create boxplot grouped by community
    boxplot_data = [sub_df[sub_df['community'] == c]['user_id'].values for c in top_5_comms]
    bp = ax_box.boxplot(
        boxplot_data, 
        labels=top_5_comms, 
        patch_artist=True,
        medianprops={'color': 'black', 'linewidth': 1.5},
        flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 3, 'alpha': 0.3}
    )
    
    for patch, color in zip(bp['boxes'], colors[:5]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax_box.set_title("User ID Range within Top 5 Communities", fontsize=12, fontweight='bold', pad=10)
    ax_box.set_xlabel("Community ID", fontsize=10)
    ax_box.set_ylabel("User ID", fontsize=10)
    ax_box.yaxis.set_major_formatter(FuncFormatter(id_formatter))
    ax_box.grid(True, linestyle='--', alpha=0.5)
    ax_box.spines['top'].set_visible(False)
    ax_box.spines['right'].set_visible(False)

    # --- PANEL 4: Pie Chart of Community Representation ---
    ax_pie = axs[1, 1]
    other_count = comm_counts.iloc[5:].sum() if len(comm_counts) > 5 else 0
    pie_data = comm_counts.head(5).copy()
    if other_count > 0:
        pie_data['Other'] = other_count
        
    labels = [f"C{x}" if x != 'Other' else 'Other' for x in pie_data.index]
    
    ax_pie.pie(
        pie_data,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors[:5] + ['#cfd8dc'],
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True}
    )
    ax_pie.set_title("Community Distribution Share", fontsize=12, fontweight='bold', pad=10)

    # Global title & optimization
    plt.suptitle(
        f"Community & User Profile Analysis Dashboard\nSource: {os.path.basename(csv_path)}",
        fontsize=16,
        fontweight='bold',
        y=0.98
    )
    plt.tight_layout()

def analyze_user_id_distribution(csv_path, n_workers=20, global_min=None, global_max=None):
    """
    Analyzes the distribution of user_ids in a community relative to worker ranges.
    Generates a chart showing the target payload per worker and identifies workers to cancel.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File {csv_path} not found.")

    # 1. Load community users
    df = pd.read_csv(csv_path)
    # Ensure columns are clean
    user_ids = df['user_id'].dropna().astype('int64').values
    
    print(f"--- DISTRIBUTION ANALYSIS ---")
    print(f"Number of target users in the CSV: {len(user_ids):,}")

    # 2. Determine global boundaries (Ideally supplied from MongoDB bounds)
    if global_min is None:
        global_min = int(user_ids.min())
        print(f"Warning: 'global_min' not provided. Using community minimum: {global_min:,}")
    if global_max is None:
        global_max = int(user_ids.max()) + 1
        print(f"Warning: 'global_max' not provided. Using community maximum: {global_max:,}")

    # 3. Simulate step range calculation for Linear Scan Strategy
    step = (global_max - global_min + n_workers - 1) // n_workers
    
    worker_data = []
    
    for i in range(n_workers):
        uid_start = global_min + i * step
        uid_end = min(global_min + (i + 1) * step, global_max)
        
        # Count how many community users fall within this range [start, end[
        count = np.sum((user_ids >= uid_start) & (user_ids < uid_end))
        
        worker_data.append({
            'Worker_ID': i,
            'Range_Start': uid_start,
            'Range_End': uid_end,
            'Target_Users_Count': int(count)
        })
        
    distribution_df = pd.DataFrame(worker_data)
    
    # 4. Generate Charts
    # Subplots to view sequential distribution AND efficiency ranking
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # --- CHART 1: Sequential Order (To see distribution shape) ---
    ax1.bar(distribution_df['Worker_ID'], distribution_df['Target_Users_Count'], 
            color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_xticks(distribution_df['Worker_ID'])
    ax1.set_xticklabels(distribution_df['Worker_ID'], fontsize=10)
    ax1.set_title("Spatial Distribution: Number of Target Users per Worker (Sequential ID Ranges)", 
                  fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Worker ID (From lowest range to highest)", fontsize=11)
    ax1.set_ylabel("Number of Present Users", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Add value labels on top of bars
    max_val = distribution_df['Target_Users_Count'].max()
    for _, row in distribution_df.iterrows():
        ax1.text(row['Worker_ID'], row['Target_Users_Count'] + (max_val * 0.01), 
                 f"{row['Target_Users_Count']:,}" if row['Target_Users_Count'] > 0 else "0", 
                 ha='center', va='bottom', fontsize=9, 
                 fontweight='bold' if row['Target_Users_Count'] == 0 else 'normal')

    # --- CHART 2: Sorted by Volume (To instantly identify redundant workers) ---
    df_sorted = distribution_df.sort_values(by='Target_Users_Count', ascending=False).reset_index(drop=True)
    
    # Color workers with 0 users in red
    colors = ['#1f77b4' if v > 0 else '#d62728' for v in df_sorted['Target_Users_Count']]
    
    ax2.bar(df_sorted.index, df_sorted['Target_Users_Count'], 
            color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_xticks(df_sorted.index)
    ax2.set_xticklabels([f"W{int(w)}" for w in df_sorted['Worker_ID']], fontsize=10, rotation=45)
    ax2.set_title("Efficiency Ranking: Number of Target Users per Worker (Sorted Decreasing)", 
                  fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Worker ID (Ordered from most useful to least useful)", fontsize=11)
    ax2.set_ylabel("Number of Present Users", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    for idx, row in df_sorted.iterrows():
        ax2.text(idx, row['Target_Users_Count'] + (max_val * 0.01), 
                 f"{row['Target_Users_Count']:,}", 
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()
    
    # 5. Conclusions and gain statistics
    dead_workers = distribution_df[distribution_df['Target_Users_Count'] == 0]['Worker_ID'].tolist()
    total_dead = len(dead_workers)
    gain_pct = (total_dead / n_workers) * 100
    
    print(f"\n--- CONCLUSIONS ---")
    print(f"Workers to CANCEL/REMOVE (0 target users in their range): {dead_workers}")
    print(f"Number of active useful workers: {n_workers - total_dead} / {n_workers}")
    print(f"🚀 ESTIMATED RESOURCE & TIME SAVING: {gain_pct:.1f}%")
    
    return distribution_df

def plot_community_id_distribution(csv_path, n_workers=2, global_min=None, global_max=None):
    """
    Displays two Matplotlib subplots directly (without file saving):
    1. The general user ID distribution curve (sorted) to visualize the index structure.
    2. The bar chart showing payload per worker to detect empty ranges.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File {csv_path} not found.")

    # 1. Load and sort IDs
    df = pd.read_csv(csv_path)
    user_ids = np.sort(df['user_id'].dropna().astype('int64').values)
    total_users = len(user_ids)
    
    # If boundaries are not provided, use the min/max of the community users
    if global_min is None:
        global_min = user_ids[0]
    if global_max is None:
        global_max = user_ids[-1] + 1

    print(f"--- DISTRIBUTION ANALYSIS ---")
    print(f"Total community users: {total_users}")
    print(f"Analyzed ID range: [{global_min:,} to {global_max:,}]")

    # 2. Create single figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # =========================================================================
    # CHART 1: General ID Distribution (Shape and Jumps)
    # =========================================================================
    ax1.plot(user_ids, 'o-', color='#1f77b4', markersize=6, linewidth=2, label="User ID")
    
    # If 2 workers, draw the mid-range cut line
    if n_workers == 2:
        mid_id = (global_min + global_max) // 2
        ax1.axhline(y=mid_id, color='#d62728', linestyle='--', linewidth=1.5, 
                    label="Theoretical Cutoff Line (Mid-Range)")
    
    ax1.set_title("General User ID Distribution\nIndex Structure Visualization (Old vs New IDs)", 
                  fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Sequential index of community users", fontsize=10)
    ax1.set_ylabel("Numerical ID value (user.id)", fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc="upper left")

    # =========================================================================
    # CHART 2: Worker Payload Distribution
    # =========================================================================
    # Simulate step range calculation for workers
    step = (global_max - global_min + n_workers - 1) // n_workers
    worker_counts = []
    worker_labels = []
    
    for i in range(n_workers):
        uid_start = global_min + i * step
        uid_end = min(global_min + (i + 1) * step, global_max)
        
        # Count users in this range [start, end[
        count = np.sum((user_ids >= uid_start) & (user_ids < uid_end))
        worker_counts.append(int(count))
        
        if n_workers <= 4:
            worker_labels.append(f"Worker {i}\n[{uid_start:.1e}, {uid_end:.1e})")
        else:
            worker_labels.append(f"Worker {i}")

    # Green if worker has work, red if empty
    colors = ['#2ca02c' if c > 0 else '#d62728' for c in worker_counts]
    
    bars = ax2.bar(worker_labels, worker_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
    
    ax2.set_title(f"Payload Distribution Across {n_workers} Workers\n(Linear Scan Strategy with Equal-Width Ranges)", 
                  fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylabel("Target Users to Extract", fontsize=10)
    ax2.set_xlabel("Worker Tasks", fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Adjust Y-axis to make space for labels
    max_count = max(worker_counts) if max(worker_counts) > 0 else 1
    ax2.set_ylim(0, max_count * 1.2)
    
    # Add text labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + (max_count * 0.02), 
                 f"{int(yval)} users\n({(yval/total_users)*100:.1f}%)" if yval > 0 else "0\n(Dead Zone)", 
                 ha='center', va='bottom', fontsize=9, 
                 fontweight='bold' if yval == 0 else 'normal')

    # 3. Interactive display
    plt.tight_layout()
    plt.show()
    
    # Brief console summary
    print("\n💡 Payload Summary:")
    for idx, count in enumerate(worker_counts):
        print(f"   - Worker {idx} will process: {count} users")
        
    return worker_counts

def plot_id_distribution_bands(csv_path, n_divisions=20, global_min=None, global_max=None):
    """
    Displays a 1D density band heatmap and prints exact min/max bounds for each division in the console.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File {csv_path} not found.")

    # 1. Load data
    df = pd.read_csv(csv_path)
    user_ids = df['user_id'].dropna().astype('int64').values
    total_users = len(user_ids)
    
    if global_min is None:
        global_min = int(user_ids.min())
    if global_max is None:
        global_max = int(user_ids.max()) + 1

    # 2. Calculate divisions and metrics
    step = (global_max - global_min + n_divisions - 1) // n_divisions
    counts = []
    bands_bounds = []
    
    for i in range(n_divisions):
        uid_start = global_min + i * step
        uid_end = min(global_min + (i + 1) * step, global_max)
        
        # Filter community users inside this division
        users_in_band = user_ids[(user_ids >= uid_start) & (user_ids < uid_end)]
        count = len(users_in_band)
        counts.append(count)
        
        # Find real min/max inside this division
        if count > 0:
            real_min = users_in_band.min()
            real_max = users_in_band.max()
        else:
            real_min = "N/A"
            real_max = "N/A"
            
        bands_bounds.append((uid_start, uid_end, real_min, real_max))
        
    counts = np.array(counts)
    
    print("=" * 115)
    print(f"📋 USER ID BOUNDS AND LIMITS PER BAND ({n_divisions} DIVISIONS)")
    print("=" * 115)
    print(f"{'Division':<8} | {'Users':<6} | {'% Community':<12} | {'Real Min ID (band)':<22} | {'Real Max ID (band)':<22} | Status")
    print("-" * 115)
    
    for i in range(n_divisions):
        _, _, real_min, real_max = bands_bounds[i]
        pct = (counts[i] / total_users) * 100
        status_flag = "❌ EMPTY" if counts[i] == 0 else "✅ USEFUL"
        
        # Format display without scientific notation for integers
        str_min = f"{real_min:,}" if isinstance(real_min, (int, np.integer)) else str(real_min)
        str_max = f"{real_max:,}" if isinstance(real_max, (int, np.integer)) else str(real_max)
        
        print(f"Div {i:<4} | {counts[i]:<6,} | {pct:<10.2f}% | {str_min:<22} | {str_max:<22} | {status_flag}")
        
    print("-" * 115)
    empty_bands = np.sum(counts == 0)
    print(f"📊 DIAGNOSTIC: {empty_bands} / {n_divisions} divisions are empty and can be deactivated. ({(empty_bands/n_divisions)*100:.1f}%)\n")

    # 4. Construct Matplotlib figure (horizontal band)
    fig, ax = plt.subplots(figsize=(14, 3.5))
    grid = counts.reshape(1, -1)
    
    # Progressive color palette
    cmap = plt.cm.get_cmap('YlOrRd')
    cmap.set_under('#f0f0f0') # Gray background for empty zones
    
    # Display the band heatmap
    im = ax.imshow(grid, cmap=cmap, aspect='auto', vmin=0.1, extent=[0, n_divisions, 0, 1])
    
    # Axes configuration
    ax.set_yticks([]) 
    ax.set_xticks(np.arange(n_divisions) + 0.5)
    ax.set_xticklabels([f"Div {i}" for i in range(n_divisions)], rotation=45, fontsize=10)
    
    # Write text inside the bands
    for i in range(n_divisions):
        text_color = "white" if counts[i] > (counts.max() * 0.6) else "black"
        ax.text(i + 0.5, 0.5, f"{counts[i]:,}\n({(counts[i]/total_users)*100:.1f}%)" if counts[i] > 0 else "0", 
                ha='center', va='center', color=text_color, fontsize=9,
                fontweight='bold' if counts[i] == 0 else 'normal')
                
    ax.set_title(f"1D Band Density Map of User IDs ({n_divisions} Divisions)\nGlobal Range: [{global_min:,} to {global_max:,})", 
                  fontsize=12, fontweight='bold', pad=15)
    
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.28, shrink=0.6)
    cbar.set_label("Community User Concentration", fontsize=10)

def calculate_optimal_user_ranges(csv_path, n_workers=20, global_min=None, global_max=None):
    """
    Calculates dynamic and optimal ID ranges for workers.
    Includes the user count, real bounds without scientific notation, and range distance.
    """
    df = pd.read_csv(csv_path)
    # Cast to uint64 to avoid negative ID bugs
    user_ids = np.sort(df['user_id'].dropna().astype(np.uint64).values)
    total_users = len(user_ids)
    
    if global_min is None:
        global_min = int(user_ids.min())
    if global_max is None:
        global_max = int(user_ids.max()) + 1
        
    total_global_db_span = int(global_max) - int(global_min)

    # Quantile cutting
    indices = np.linspace(0, total_users, n_workers + 1, dtype=int)
    
    optimal_ranges = []
    total_scanned_span = 0
    
    print("=" * 115)
    print(f"🚀 CALCULATING OPTIMAL DYNAMIC ID RANGES ({n_workers} WORKERS)")
    print("=" * 115)
    print(f"{'Worker':<9} | {'Users':<6} | {'user.id MIN Real':<20} | {'user.id MAX Real':<20} | {'Range Distance':<20}")
    print("-" * 115)
    
    for i in range(n_workers):
        start_idx = indices[i]
        end_idx = indices[i+1]
        
        worker_users = user_ids[start_idx:end_idx]
        
        uid_min_reel = int(worker_users.min())
        uid_max_reel = int(worker_users.max())
        
        # Calculate raw difference distance
        distance = uid_max_reel - uid_min_reel
        
        total_scanned_span += distance
        optimal_ranges.append((uid_min_reel, uid_max_reel))
        
        # Clean formatting without scientific notation for integer IDs
        print(f"Worker {i:<2} | {len(worker_users):<6} | {str(uid_min_reel):<20} | {str(uid_max_reel):<20} | {str(distance):<20}")
        
    print("-" * 115)
    
    if total_global_db_span > 0:
        ignored_span = total_global_db_span - total_scanned_span
        ignored_span = max(0, ignored_span) 
        pct_ignored = (ignored_span / total_global_db_span) * 100
    else:
        pct_ignored = 0.0

    print(f"💡 SCAN OPTIMIZATION SUMMARY:")
    print(f"   - Thanks to these dynamic ranges, workers will ignore {pct_ignored:.2f}% of the index empty space.")
    print("=" * 115)
    
    return optimal_ranges

import pandas as pd
import matplotlib.pyplot as plt

def plot_user_distribution(path_csv):

    df = pd.read_csv(
        path_csv,
        usecols=["user_id"],
        dtype={"user_id": "int64"}
    )

    user_ids = df["user_id"]

    plt.figure(figsize=(18, 6))

    plt.hist(
        user_ids,
        bins=500,
        log=True
    )

    plt.xlabel("Twitter User ID")
    plt.ylabel("Nombre d'utilisateurs")
    plt.title("Distribution des IDs Twitter")

    plt.grid(True)

    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN


def detect_dense_id_ranges(
    path_csv,
    eps=5e15,
    min_samples=10
):

    df = pd.read_csv(
        path_csv,
        usecols=["user_id"],
        dtype={"user_id": "int64"}
    )

    ids = np.sort(df["user_id"].values)

    X = ids.reshape(-1, 1)

    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples
    ).fit(X)

    labels = clustering.labels_

    unique_labels = sorted(set(labels))

    ranges = []

    print("\n=== DENSE USER ID RANGES ===\n")

    print(
        f"{'Cluster':<10}"
        f"{'Users':<10}"
        f"{'Range Size':<25}"
        f"{'Min ID':<25}"
        f"{'Max ID'}"
    )

    print("-" * 95)

    for label in unique_labels:

        if label == -1:
            continue

        cluster_ids = ids[labels == label]

        min_id = int(cluster_ids.min())
        max_id = int(cluster_ids.max())

        count = len(cluster_ids)

        range_size = max_id - min_id

        ranges.append({
            "cluster": label,
            "users": count,
            "min_id": min_id,
            "max_id": max_id,
            "range_size": range_size
        })

        print(
            f"{label:<10}"
            f"{count:<10}"
            f"{range_size:<25}"
            f"{min_id:<25}"
            f"{max_id}"
        )

    # VISUALISATION
    plt.figure(figsize=(22, 5))

    scatter = plt.scatter(
        ids,
        np.zeros_like(ids),
        c=labels,
        s=6,
        cmap="tab20"
    )

    plt.yticks([])

    plt.xlabel("Twitter User ID")
    plt.title("Dense Twitter ID Clusters")

    plt.grid(True)

    plt.show()

    return ranges