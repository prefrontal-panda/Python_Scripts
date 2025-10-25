# This Python script contains functions that can be used for the analysis of NAPLAN data at a campu-level 
# The main code to run can be found at NAPLAN_CampusLevelAnalysis.ipynb

# Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from io import BytesIO
import base64
import inspect
import math

# For loop to subset
def subset_loop(dataset_name, subset_col):
    """
    Note: assign the function to a variable to save it to different dictionaries 
        (e.g. gender_df = subset_loop(dataset_name, subset_col))
    """
    dictionary_name = {} # Creating an empty dictionary to store the individual names
    # for loop
    for i in dataset_name[subset_col].unique(): # Getting the unique names
        dictionary_name[i] = dataset_name[dataset_name[subset_col] == i]
        print(f"{i} is stored in dictionary_name[{i!r}]") # printing key within dictionary
    return dictionary_name


# Finding summary statistics for each domain within a campus
def summary_stats(dataset_name, cols_of_interest):
    """
    cols_of_interest: pass as a list of strings (e.g. ['READING', 'WRITING', 'SPELLING', 'GRAMMAR & PUNCTUATION']).
    """
    # Making a summary table to store the values required.
    summary = dataset_name[cols_of_interest].agg([ # '.agg()' applies the summary functions we want to the entire dataframe
        'count',
        'mean',
        'median',
        'std',
        'min',
        lambda x: x.quantile(0.25), # 'lambda' here is used to find the 25th quantile of the numeric column.
        lambda x: x.quantile(0.75), # Finding the 75th quantile of the numeric column.
        'max'
    ]).round(2)

    # Transposing so each row becomes a variable
    summary = summary.transpose()

    # Renaming columns
    summary.columns = ['n', 'Mean', 'Median', 'Std', 'Min', '25th %ile', '75th %ile', 'Max']

    # Returning final output table
    return summary


# For loop to run summary statistics through a dictionary
def summary_loop(dataset_name, cols_to_summarise, print_output=True):
    """
    This function loops through a dictionary of dataframes and calculates the summary statistics required.
    Note that the columns to summarise should be defined outside the function.

    dataset_name: dictionary where the keys are the groups of interest and the values are the dataframes
    cols_to_summarise: list of the columns to generate summary stats of
    print_output (True/False): optionally print the summary stats for each group
    """
    summaries = {} # Empty dictionary
    # For loop 
    for group_name, name_df in dataset_name.items():
        summaries[group_name] = summary_stats(name_df, cols_to_summarise)
        print(f"Summary statistics calculated for {group_name}")
    # Viewing
    if print_output:
        for group_name, summary_df in summaries.items():                   
            print(f"\n{'='*60}") # print 60x "=" and add a blank line before the sign for better readability
            print(f"Summary Statistics for {group_name}") # print header for the campus
            print(f"{'='*60}") # separator
            print(summary_df) # print the summary dataframe for each campus

    return summaries

# Renaming columns for concatenating and plotting
def col_rename(dataset_name, mapping_dict, verbose=True):
    """
    Function to rename columns using a mapping dictionary.
    Please define the dictionary outside the function.
    """
    # Setting warnings to check for missing columns
    existing_cols = set(dataset_name.columns)
    renaming_cols = set(mapping_dict.keys())

    missing_cols = renaming_cols - existing_cols
    if verbose and missing_cols:
        print(f"These columns are not in the dataframe: {missing_cols}")

    # Loop through current mapping dictionary and removes columns not present in current dataframe
    return dataset_name.rename(columns={k: v for k, v in mapping_dict.items() if k in existing_cols})


# Converting wide dataframe to long dataframe
def df_melt(dataset_name, id_vars, value_vars, var_name='Statistic', value_name='Score'):
    """
    Converts a wide dataframe to long format for plotting.

    id_vars: list of strings (e.g. ['Domain', 'Gender']) based on renamed columns. gives grouping information.
    value_vars: list of strings to plot (e.g. ['q1', 'mean'] or ['READING', 'WRITING']). gives category information for plotting.
    var_name: column name for 'variable_values'
    value_name: column name for the values of each variable
    """
    return dataset_name.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name        
    )

# Use cases:
#domain_name = ['READING', 'WRITING', 'SPELLING', 'NUMERACY', 'GRAMMAR & PUNCTUATION']
#df_long = melt_to_long(dt_subset, id_vars=['Home School Name'], value_vars=domain_name, var_name='Domain', value_name='Score')
# OR
#data_genderlong = melt_to_long(dt_genderconcat, id_vars=['Domain', 'Gender'], value_vars=['q1', 'median', 'mean', 'q3'])
#lbote_long = melt_to_long(lbote_all, id_vars=['Domain', 'label'], value_vars=['q1', 'median', 'mean', 'q3'])


# Plotting boxplots for comparison against nation datasets
def plot_comparison(dataset, x='Statistic', y='Score', hue=None, title=' ',
                    col='Domain', col_wrap=3,
                    x_labels=['Q1', 'Median', 'Mean', 'Q3'],
                    palette='Set2', height=4, aspect=1.2):
    """
    Generates bar plots for the summary statistics identified.

    hue: column used for coloring
    col: column to facet by (passed as string)
    x_labels: x-axis labels
    palette: color palette to use
    """
    # Setting theme
    sns.set_theme(style="whitegrid")
    
    # Plotting
    g = sns.catplot(
        data=dataset,
        x=x,
        y=y,
        hue=hue,
        col=col,
        kind='bar',
        col_wrap=col_wrap,
        height=height,
        aspect=aspect,
        palette=palette
    )
    
    # Setting x-axis labels
    for ax in g.axes.flat:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=0, ha='center')
    
    # Setting titles
    g.set_titles("{col_name}")
    g.figure.subplots_adjust(top=0.9)
    g.figure.suptitle(title, fontsize=16)
    g.set_axis_labels(x, y)
    
    # Returning plot
    plt.show()
    return g


# Plotting Median vs national percentile spread
def school_vs_national_figs(
    school_stats, 
    national_stats, 
    subgroups=None, 
    school_label_map=None,
    ncols=2, 
    figsize_per_plot=(6,4),
    jitter_width=0.03,
    random_seed=42,
    title_template="{domain}"
):
    """
    Creates shaded-band plots for median comparison against national score spread.
    
    school_stats : dictionary with school medians. Format: {label: {domain: median}}
    national_stats : dictionary with national stats. Format: {subgroup: {domain: {'p10':..,'p25':..,'p50':..,'p75':..,'p90':..}}}
    subgroups : optional. List of national subgroups to plot. Defaults to all keys in national_stats.
    school_label_map : optional. mapping dictionary of national subgroup to school labels to plot. 
        Example: {'ALL': ['Whole School'], 'BOYS': ['BOYS'], 'GIRLS': ['GIRLS'], 'LBOTE': ['LBOTE']}
    jitter_width: adds optional jitter to the dots on the map for clearer visualisation
    """
    
    if subgroups is None: # if no subgroup is defined, plot all (not recommended)
        subgroups = list(national_stats.keys())
    
    if school_label_map is None: # provides mapping identifiers for dots
        school_label_map = {sg: list(school_stats.keys()) for sg in subgroups} # default: use all school labels for all subgroups
    
    domains = list(next(iter(school_stats.values())).keys())
    n_plots = len(domains) * len(subgroups)
    nrows = math.ceil(n_plots / ncols)
    
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, 
        figsize=(figsize_per_plot[0]*ncols, figsize_per_plot[1]*nrows)
    )
    axes = axes.flatten()

    # generating seed for reproducible jitters
    rng = np.random.default_rng(seed=random_seed) 
    label_jitters = {
        label: rng.uniform(-jitter_width, jitter_width) for label in school_stats.keys()
    }
    
    plot_idx = 0
    for subgroup in subgroups:
        for domain in domains:
            ax = axes[plot_idx]
            ns = national_stats[subgroup][domain]
            
            # Shaded bands
            ax.fill_between([0,1], ns['p10'], ns['p90'], color='gray', alpha=0.2, label='National p10–p90')
            ax.fill_between([0,1], ns['p25'], ns['p75'], color='gray', alpha=0.5, label='National IQR')
            ax.hlines(ns['p50'], 0, 1, colors='black', linestyles='dashed', label='National Median')
            
            # Only plot school labels mapped to this subgroup
            labels_to_plot = school_label_map.get(subgroup, [])
            for label in labels_to_plot:
                if label in school_stats:
                    jitter = label_jitters[label]  # adding jitter so dots are more visible
                    ax.plot(0.5 + jitter, school_stats[label][domain], 'o', markersize=10, label=label)
            
            ax.set_xlim(0,1)
            ax.set_xticks([])
            ax.set_ylabel(domain)
            ax.set_title(title_template.format(domain=domain))
            #ax.legend()
            
            plot_idx += 1
    
    for j in range(plot_idx, len(axes)):
        fig.delaxes(axes[j])
    
    # Settin legends
    # --- Collect all legend handles and labels ---
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    # Deduplicate (preserves order)
    by_label = dict(zip(labels, handles))
    # Plotting legend
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc='upper center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,            # adjust number of columns for layout
        frameon=False
    )
    # Plotting and returning
    plt.tight_layout()
    return fig


# Saving whole report into a html file format
def save_report(filename="report.html", title="Analysis Report",
                figures=None, tables=None, namespace=None):
    """
    Gathers all figures and dataframes from the current notebook and saves them in a single HTML file.

    filename: saved as string. the name of the output file
    title: gives the report a title
    namespace: optional. goes through the notebook and maps the global variables
    figures/tables: can be passed as a dictionary to manually set titles
    """

    # --- Detect namespace if needed ---
    if namespace is None:
        frame = inspect.currentframe().f_back # Go back one code chunk
        namespace = frame.f_globals # map variables

    html_parts = [
        f"<html><head><meta charset='utf-8'><title>{title}</title></head><body>",
        f"<h1>{title}</h1><hr>"
    ]

    # Collecting Figures
    if figures is None:
        figures = {f"Plot {i+1}": plt.figure(num) for i, num in enumerate(plt.get_fignums())} # automatically collects figures
    if figures:
        html_parts.append("<h2>Plots</h2>")
        for title, fig in figures.items(): # loops through dictionary and gives cutom title
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
            html_parts.append(f"<h3>{title}</h3>")  # Use custom title
            html_parts.append(f'<img src="data:image/png;base64,{img_base64}" style="width:100%; max-width:1200px;"><br><br>') # changing size of plot displayed
    else:
        html_parts.append("<p><em>No active matplotlib figures found.</em></p>")

    # Collecting tables
    if tables is None:
        dfs = {name: obj for name, obj in namespace.items() if isinstance(obj, pd.DataFrame)}
    else:
        # If a list or dict of DataFrames is passed
        if isinstance(tables, dict):
            dfs = tables
        else:
            dfs = {f"Table {i+1}": df for i, df in enumerate(tables)}

    if dfs:
        html_parts.append("<h2>Tables</h2>")
        for title, df in dfs.items():
            # Handle nested dicts of DataFrames
            if isinstance(df, dict):
                html_parts.append(f"<h3>{title}</h3>")
                for subname, subdf in df.items():
                    html_parts.append(f"<h4>{subname}</h4>")
                    html_parts.append(subdf.to_html(index=False, border=1))
                    html_parts.append("<br>")
            elif isinstance(df, pd.DataFrame):
                html_parts.append(f"<h3>{title}</h3>")
                html_parts.append(df.to_html(index=False, border=1))
                html_parts.append("<br><br>")
            else:
                html_parts.append(f"<p><em>{title} is not a DataFrame</em></p>")

    else:
        html_parts.append("<p><em>No pandas DataFrames found.</em></p>")

    html_parts.append("</body></html>")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    print(f"Report saved to {os.path.abspath(filename)}")