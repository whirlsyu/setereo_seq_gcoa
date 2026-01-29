import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
from sklearn.isotonic import spearmanr
matplotlib.use('Agg')  #  Agg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from statsmodels.stats.multitest import multipletests
plt.rcParams['figure.dpi'] = 300  #  dpi
from cell2location.utils.filtering import filter_genes
from coexistence_utils import calculate_coexistence_ratio, calculate_coexistence_ratio_probability, add_gene_statistics, zero_diagonal
import seaborn as sns
import scanpy as sc
import anndata
import scipy.sparse as sp
from scipy import sparse

def calculate_cpm(adata):
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)
    
    total_counts_per_cell = np.asarray(adata.X.sum(axis=1)).flatten()
    cpm = adata.X.copy()
    cpm = cpm.multiply(1e6 / total_counts_per_cell[:, np.newaxis])
    mean_cpm = np.asarray(cpm.mean(axis=0)).flatten()
    
    return pd.DataFrame({
        'gene': adata.var_names,
        'mean_cpm': mean_cpm
    }).sort_values('mean_cpm', ascending=False)

def calculate_counts(adata):
    if sparse.issparse(adata.X):
        counts = adata.X.toarray()
    else:
        counts = adata.X
    
    mean_counts = np.mean(counts, axis=0)
    
    return pd.DataFrame({
        'gene': adata.var_names,
        'mean_counts': mean_counts
    }).sort_values('mean_counts', ascending=False)

def calculate_coexpression_ratio(adata):
    print(f"Detected sparse matrix: {sp.issparse(adata.X)}")
    data = pd.DataFrame(
        data=adata.X.toarray() if sp.issparse(adata.X) else adata.X,
        columns=adata.var_names
    )
    # Initialize correlation and p-value matrices
    n_vars = data.shape[1]
    corr_matrix = np.zeros((n_vars, n_vars))
    pval_matrix = np.zeros((n_vars, n_vars))

    # Iterate to compute Spearman correlation and p-value for each pair of variables
    for i in range(n_vars):
        for j in range(i, n_vars):
            corr, pval = spearmanr(data.iloc[:, i], data.iloc[:, j])
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
            pval_matrix[i, j] = pval
            pval_matrix[j, i] = pval

    # Convert to DataFrame
    corr_df = pd.DataFrame(corr_matrix, columns=data.columns, index=data.columns)
    pval_df = pd.DataFrame(pval_matrix, columns=data.columns, index=data.columns)

    # Flatten p-value matrix to 1D array (excluding diagonal)
    p_values_flat = pval_df.stack()[pval_df.stack() != 1].reset_index(drop=True)
    valid_p_values = p_values_flat[~p_values_flat.isna()]

    # Validate input
    n_vars = data.shape[1]
    expected_length = n_vars * (n_vars - 1) // 2
    if len(valid_p_values) != expected_length:
        print(f"Warning: only detected {len(valid_p_values)} valid p-values (expected {expected_length})")
    # Validate number of valid tests
    if len(p_values_flat) < 1:
        return corr_df, pval_df.fillna(1)  # If no valid p-values, return p-value matrix filled with 1
    else:
        # Apply Benjamini-Hochberg correction (FDR control)
        _, corrected_pvals, _, _ = multipletests(p_values_flat, method='fdr_bh')

        # Reconstruct corrected p-value matrix (fill only valid positions)
        corrected_pval_matrix = np.full_like(pval_matrix, np.nan)
        idx = 0
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                if not np.isnan(pval_df.iloc[i, j]):
                    corrected_pval_matrix[i, j] = corrected_pvals[idx]
                    corrected_pval_matrix[j, i] = corrected_pvals[idx]
                    idx += 1
        corrected_pval_df = pd.DataFrame(corrected_pval_matrix, columns=data.columns, index=data.columns)
        return corr_df, corrected_pval_df


def filter_and_rename_genes(adata, filter_file):
    # read filter file
    filter_df = pd.read_csv(filter_file, sep='\t')
    
    # create mapping table, Locus to EXO70_info 
    locus_to_exo70 = dict(zip(filter_df['Locus'], filter_df['EXO70_info']))
    
    # get gene names in adata
    genes_to_keep = [gene for gene in adata.var_names if gene in filter_df['Locus'].values]
    
    # filter adata 
    adata_filtered = adata[:, genes_to_keep]
    
    # rename to genes
    new_var_names = [locus_to_exo70.get(gene, gene) for gene in adata_filtered.var_names]
    adata_filtered.var_names = new_var_names
    adata_filtered.var_names_make_unique()
    
    return adata_filtered

def plot_spatial_distribution(adata, output_path, title):
    # get spatial range
    x_range = adata.obsm['spatial'][:, 0].max() - adata.obsm['spatial'][:, 0].min()
    y_range = adata.obsm['spatial'][:, 1].max() - adata.obsm['spatial'][:, 1].min()
    
    # Determine image orientation based on coordinate range    
    if x_range > y_range:
        figsize = (8, 10)  # Vertical orientation        
        img_key = "hires"
    else:
        figsize = (10, 8)  # Horizontal orientation
        img_key = None  # don't use background image
    
    fig, ax = plt.subplots(figsize=figsize)
    sc.pl.spatial(adata, color="cell_type", img_key=img_key, spot_size=40, alpha=0.8, show=False, ax=ax)
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_spatial_distribution_filted_umap(adata, output_path, title):
    # get spatial range
    x_range = adata.obsm['spatial'][:, 0].max() - adata.obsm['spatial'][:, 0].min()
    y_range = adata.obsm['spatial'][:, 1].max() - adata.obsm['spatial'][:, 1].min()
    
    # Determine image orientation based on coordinate range
    if x_range > y_range:
        figsize = (8, 10)  # Vertical orientation
        img_key = "hires"
    else:
        figsize = (10, 8)  # Horizontal orientation
        img_key = None  # don't use background image
    
    fig, ax = plt.subplots(figsize=figsize)
    # plot cell locations using Leiden clustering
    sc.pl.spatial(adata, 
                color="leiden",  # Leiden clustering
                img_key=img_key,  # use hires image
                spot_size=40,    
                alpha=0.8,       # set dot transparency
                ax=ax,
                show=False,
                title=title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()




def analyze_spatial_data(input_file, output_folder, filter_file):
    # create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # read h5ad file
    adata = anndata.read_h5ad(input_file)
    
    # basic info
    print(f"cell quantity: {adata.n_obs}")
    print(f"gene quantity: {adata.n_vars}")
    
    # calculate non-zero ratio
    if sp.issparse(adata.X):
        non_zero_count = adata.X.nnz
        total_elements = adata.X.shape[0] * adata.X.shape[1]
    else:
        non_zero_count = np.count_nonzero(adata.X)
        total_elements = adata.X.size
    
    print(f"none zero ratio: {non_zero_count / total_elements:.2%}")
    
    # spatial distribution
    output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_spatial_distribution.png")
    title = f"spatial_distribution - {os.path.splitext(os.path.basename(input_file))[0]}"
    plot_spatial_distribution(adata, output_path, title)

    
    # UMAP
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.louvain(adata)
    sc.pl.umap(adata, color='louvain', title='UMAP', show=False)
    plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_umap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # QC
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts'], jitter=0.4, multi_panel=True, show=False)
    plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_qc_violin.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', show=False)
    plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_qc_scatter.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # CPM
    cpm_df = calculate_cpm(adata)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(cpm_df['mean_cpm'], bins=100, kde=True, log_scale=True)
    plt.title('Distribution of Mean CPM')
    plt.xlabel('Mean CPM (log scale)')
    plt.ylabel('Number of Genes')
    plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_cpm_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(data=cpm_df, x='mean_cpm')
    plt.xscale('log')
    plt.title('Cumulative Distribution Function of Mean CPM')
    plt.xlabel('Mean CPM (log scale)')
    plt.ylabel('Cumulative Proportion')
    plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_cpm_ecdf.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # counts
    counts_df = calculate_counts(adata)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(counts_df['mean_counts'], bins=100, kde=True, log_scale=True)
    plt.title('Distribution of Mean Counts')
    plt.xlabel('Mean Counts (log scale)')
    plt.ylabel('Number of Genes')
    plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_counts_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.ecdfplot(data=counts_df, x='mean_counts')
    plt.xscale('log')
    plt.title('Cumulative Distribution Function of Mean Counts')
    plt.xlabel('Mean Counts (log scale)')
    plt.ylabel('Cumulative Proportion')
    plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_counts_ecdf.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # filter genes
    adata_filtered = filter_and_rename_genes(adata, filter_file)

    # UMAP
    sc.pp.neighbors(adata_filtered)
    sc.tl.umap(adata_filtered)
    sc.tl.leiden(adata_filtered)
    sc.pl.umap(adata_filtered, color='leiden', title='UMAP', show=False)
    plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_filtered_umap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    output_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_filtered_umap_spatial_distribution.png")
    title = f"filtered_umap_spatial_distribution - {os.path.splitext(os.path.basename(input_file))[0]}"
    plot_spatial_distribution_filted_umap(adata_filtered, output_path, title)    
    # gene correlation or coexistence
    cell_types = adata_filtered.obs['cell_type'].unique()
    for cell_type in cell_types:
        cell_type_data = adata_filtered[adata_filtered.obs['cell_type'] == cell_type]
        corr, corrected_pval_df = calculate_coexpression_ratio(cell_type_data)
        gene_stats = add_gene_statistics(cell_type_data)
        gene_stats.to_csv(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_gene_statistics_{cell_type}.csv"))
   
        # order genes by names
        sorted_genes = sorted(corr.columns)
        corr = corr.loc[sorted_genes, sorted_genes]
        # save correlation data
        corr.to_csv(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_correlation_{cell_type}.csv"))
        corrected_pval_df.to_csv(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_correlation_pval_{cell_type}.csv")) 
        corr = zero_diagonal(corr)         
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=True, yticklabels=True)
        plt.title(f'Gene Correlation Heatmap - {cell_type}')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_correlation_{cell_type}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # gene correlation or coexistence
    for cell_type in cell_types:
        cell_type_data = adata_filtered[adata_filtered.obs['cell_type'] == cell_type].X
        ratio = calculate_coexistence_ratio(cell_type_data)
        
        if ratio.size == 0:
            print(f"warning: {cell_type} data is empty, skip this cell type")
            continue
        
        ratio_df = pd.DataFrame(ratio, index=adata_filtered.var_names, columns=adata_filtered.var_names)
        sorted_genes = sorted(ratio_df.columns)
        ratio_df = ratio_df.loc[sorted_genes, sorted_genes]
        # save coexistence data
        ratio_df.to_csv(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_coexistence_{cell_type}.csv"))
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(ratio_df, cmap='YlOrRd', vmin=0, vmax=1, 
                    xticklabels=True, yticklabels=True)
        plt.title(f'Gene Coexistence Ratio Heatmap - {cell_type}')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_coexistence_{cell_type}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
    # gene correlation or coexistence
    for cell_type in cell_types:
        cell_type_data = adata_filtered[adata_filtered.obs['cell_type'] == cell_type].X
        ratio, observed, expected  = calculate_coexistence_ratio_probability(cell_type_data)
        
        if ratio.size == 0:
            print(f"warning: {cell_type} data is empty, skip this cell type")
            continue
        
        ratio_df = pd.DataFrame(ratio, index=adata_filtered.var_names, columns=adata_filtered.var_names)
        
        observed_df = pd.DataFrame(observed, index=adata_filtered.var_names, columns=adata_filtered.var_names)
        expected_df = pd.DataFrame(expected, index=adata_filtered.var_names, columns=adata_filtered.var_names)
        
        sorted_genes = sorted(ratio_df.columns)
        ratio_df = ratio_df.loc[sorted_genes, sorted_genes]

        ratio_df.to_csv(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_coexistence_probability_{cell_type}.csv"))
        observed_df.to_csv(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_coexistence_observed_{cell_type}.csv"))
        expected_df.to_csv(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_coexistence_expected_{cell_type}.csv"))
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(ratio_df, cmap='YlOrRd', vmin=ratio_df.min().min()-0.1, vmax=ratio_df.max().max()+0.1, 
                    xticklabels=True, yticklabels=True)
        plt.title(f'Gene Coexistence Probability Heatmap - {cell_type}')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(input_file))[0]}_coexistence_probability_{cell_type}.png"), dpi=300, bbox_inches='tight')
        plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze spatial transcriptomics data")
    parser.add_argument("input_file", help="Path to the input h5ad file")
    parser.add_argument("output_folder", help="Path to the output folder")
    parser.add_argument("--filter_file", help="Path to the filter file", default='filtered_gene_results.tsv')
    args = parser.parse_args()
    
    analyze_spatial_data(args.input_file, args.output_folder, args.filter_file)