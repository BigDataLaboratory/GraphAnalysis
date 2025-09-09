import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

prefix = "/ipazianas/pasquini/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/result/images"


def generate_visualizations(csv_data, output_prefix='community_comparison'):
    """
    Reads CSV data, creates three summary plots comparing Retweet and Reply graphs,
    and saves them as PNG files.

    Args:
        csv_data (str): A string containing the CSV formatted data.
        output_prefix (str): A prefix for the output image filenames.
    """
    try:
        df = pd.read_csv(io.StringIO(csv_data))
        print("Data loaded successfully. Generating plots...")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Set a professional plot style
    sns.set_theme(style="whitegrid", context="talk")
    
    # --- Plot 1: Community Similarity Score (NMI) ---
    fig1, ax1 = plt.subplots(figsize=(10, 8)) # Create a figure with a single subplot
    #fig1.suptitle('NMI (Structural Stability): Impact of Hashtags', fontsize=20, weight='bold')

    # Create the line plot for NMI
    sns.lineplot(ax=ax1, data=df, x='resolution', y='nmi', hue='pair', style='pair', markers=True, markersize=10)

    # Set titles and labels
    #ax1.set_title('Comparison of Retweet vs. Reply Graphs', fontsize=16)
    ax1.set_xlabel('Resolution (Gamma)', fontsize=14)
    ax1.set_ylabel('NMI Score', fontsize=14)
    ax1.legend(title='Comparison Pair')
    ax1.set_ylim(0.98, 1.001) # Zoom in on the high NMI values
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot1_path = f"{prefix}/{output_prefix}_nmi_scores_combined.png"
    plt.savefig(plot1_path, dpi=300)
    print(f"✅ Plot 1 saved to: {plot1_path}")
    plt.close(fig1)

    # --- Plot 2: Community Similarity Score (ARI) ---
    fig2, ax2 = plt.subplots(figsize=(10, 8)) # Create a second figure with a single subplot
    #fig2.suptitle('ARI: Impact of Hashtags', fontsize=20, weight='bold')

    # Create the line plot for ARI
    sns.lineplot(ax=ax2, data=df, x='resolution', y='ari', hue='pair', style='pair', markers=True, markersize=10)

    # Set titles and labels
    #ax2.set_title('Comparison of Retweet vs. Reply Graphs', fontsize=16)
    ax2.set_xlabel('Resolution (Gamma)', fontsize=14)
    ax2.set_ylabel('ARI Score', fontsize=14)
    ax2.legend(title='Comparison Pair')

    # Final adjustments and saving
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot2_path = f"{prefix}/{output_prefix}_ari_score.png"
    plt.savefig(plot2_path, dpi=300)
    print(f"✅ ARI plot saved to: {plot2_path}")
    plt.close(fig2) # Close the figure


    # --- Plot 2: Community Size Inequality (Gini Coefficient) ---
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    #fig2.suptitle('Community Size Inequality: Retweet vs. Reply Graph', fontsize=20, weight='bold')

    # Plotting only the 'base' Gini for clarity, as augmented is nearly identical
    sns.lineplot(ax=ax, data=df[df['pair'] == 'R_vs_RH'], x='resolution', y='gini_base', marker='o', label='Retweet Graph (RT)')
    sns.lineplot(ax=ax, data=df[df['pair'] == 'P_vs_PH'], x='resolution', y='gini_base', marker='s', label='Reply Graph (R)')
    
    #ax.set_title('Gini coefficient of base graphs across resolutions', fontsize=16)
    ax.set_xlabel('Resolution (Gamma)', fontsize=14)
    ax.set_ylabel('Gini Coefficient', fontsize=14)
    ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot2_path = f"{prefix}/{output_prefix}_gini_coefficient_combined.png"
    plt.savefig(plot2_path, dpi=300)
    print(f"✅ Plot 2 saved to: {plot2_path}")
    plt.close(fig2)


    # --- Plot 3: Number of Communities Detected ---
    fig3, ax = plt.subplots(1, 1, figsize=(12, 8))
    #fig3.suptitle('Number of Communities Detected: Retweet vs. Reply Graph', fontsize=20, weight='bold')

    # Plotting only the 'base' counts for clarity
    sns.lineplot(ax=ax, data=df[df['pair'] == 'R_vs_RH'], x='resolution', y='n_comms_base', marker='o', label='Retweet Graph (RT)')
    sns.lineplot(ax=ax, data=df[df['pair'] == 'P_vs_PH'], x='resolution', y='n_comms_base', marker='s', label='Reply Graph (R)')

    #ax.set_title('Community counts of base graphs across resolutions', fontsize=16)
    ax.set_xlabel('Resolution (Gamma)', fontsize=14)
    ax.set_ylabel('Number of Communities (in millions)', fontsize=14)
    ax.ticklabel_format(style='plain', axis='y')
    # Format y-axis to be in millions for readability
    ax.yaxis.set_major_formatter(lambda x, pos: f'{x/1e6:.1f}M')
    ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot3_path = f"{prefix}/{output_prefix}_community_counts_combined.png"
    plt.savefig(plot3_path, dpi=300)
    print(f"✅ Plot 3 saved to: {plot3_path}")
    plt.close(fig3)

if __name__ == '__main__':
    # Paste your provided data here as a multi-line string
    csv_data_string = """pair,resolution,nmi,ari,vi,n_comms_base,n_comms_aug,mean_size_base,mean_size_aug,gini_base,gini_aug
R_vs_RH,0.1,0.9894607164362109,0.7199841308679926,13.527450040246567,3924522,3947041,1.3721564052896122,1.3643278597815427,0.24610065324542418,0.2428540273116695
R_vs_RH,0.2,0.9929364583785402,0.7573117057237898,13.58875468054018,4268232,4296826,1.261660097201839,1.253264153586857,0.18548306069813147,0.18161441618950502
R_vs_RH,0.3,0.9947160030485278,0.7796711195144032,13.63803177455151,4558708,4578227,1.1812684646614786,1.1762321964376166,0.13702866099705902,0.13463412771206773
R_vs_RH,0.4,0.9950484500669564,0.7795615226418852,13.645216387483359,4582636,4612796,1.1751005316590712,1.1674173321343497,0.1325350765393749,0.1283519984196122
R_vs_RH,0.5,0.9978097334305217,0.7642495619066442,13.687650029673645,5117195,5115441,1.0523456698445144,1.0527065017463793,0.048987564997379884,0.049292349192946006
R_vs_RH,0.6,0.9985763006532734,0.7903881762408375,13.678783019627016,5156165,5161032,1.0443921014940367,1.043407210030862,0.0419922226227063,0.041121499475549994
R_vs_RH,0.7,0.9986394092767077,0.784143185570281,13.681793440080455,5162903,5168867,1.0430290865429779,1.041825607043091,0.04073215114939477,0.03966677869630386
R_vs_RH,0.8,0.9987775466065673,0.7955732797207895,13.68351091197756,5174483,5180866,1.0406948868128467,1.039412715943628,0.03859032144966368,0.037452025972132486
R_vs_RH,0.9,0.998802649896261,0.7805643556094355,13.685810193504693,5178675,5185498,1.0398524719160789,1.0384842497287627,0.03780524891041104,0.03658958087557185
R_vs_RH,1.0,0.9989473459649603,0.7819824177997063,13.695131430223025,5231880,5233860,1.0292778121822366,1.0288884303363102,0.028239390058227176,0.027881720871328453
P_vs_PH,0.1,0.9896753346430888,0.6468143317215006,13.481221695107486,3674210,3706978,2.063900811330871,2.045656866590522,0.32887091911191413,0.3270610248108434
P_vs_PH,0.2,0.9917290494412097,0.6598218649581395,13.633993076973706,4335107,4366168,1.749254401333116,1.7368101731312218,0.2664757564114528,0.26585011751716703
P_vs_PH,0.3,0.992378180469855,0.6693294666974829,13.796575703860224,5101751,5124537,1.4863926130459915,1.4797834418992388,0.21065911881195776,0.2105212618935799
P_vs_PH,0.4,0.9925604292991178,0.6397761746127562,13.808260476199525,5143341,5166649,1.474373369372165,1.467722115436911,0.2055342946542762,0.20548616408727294
P_vs_PH,0.5,0.9976563182335143,0.5931725793460378,13.958905268930344,6981792,6979058,1.0861402058382719,1.0865656941094342,0.07653804349499005,0.07689289485905437
P_vs_PH,0.6,0.9983449812855275,0.6337841640791477,13.950342523109796,7027707,7028456,1.0790439897394697,1.0789289994843818,0.07077892816862064,0.07072356499017096
P_vs_PH,0.7,0.9984067659124166,0.6008264062176963,13.953832540120136,7039135,7040402,1.0772921672904412,1.0770982963756899,0.06924417460043686,0.06912671008650073
P_vs_PH,0.8,0.9984706997212504,0.5796130662358527,13.9594271733706,7064348,7065300,1.0734472593932236,1.073302619846291,0.06595199002105523,0.06588665937052762
P_vs_PH,0.9,0.9985569247381747,0.6144325846255897,13.960020737171995,7070917,7072487,1.072450008959234,1.0722119390251266,0.06507352528974275,0.06492462835434654
P_vs_PH,1.0,0.9988106308568903,0.5985613205625117,13.986827118446673,7272865,7270479,1.0426709419190374,1.0430131219689927,0.04033308091638799,0.04064885351835157
"""
    generate_visualizations(csv_data_string)