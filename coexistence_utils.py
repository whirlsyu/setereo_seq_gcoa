# coexistence_utils.py

import numpy as np
import scipy.sparse as sp
import pandas as pd
from statsmodels.stats.multitest import multipletests

def calculate_coexistence_ratio(data, epsilon=1e-10):
    print(f"Input data type: {type(data)}")
    print(f"Input data shape: {data.shape}")
    
    if sp.issparse(data):
        binary_data = data.astype(bool).astype(int)
    else:
        binary_data = (data > 0).astype(int)
    
    print(f"Binary data shape: {binary_data.shape}")
    
    if binary_data.shape[1] == 1:
        # if only one gene, return 1.0
        return np.array([[1.0]])
    
    if sp.issparse(binary_data):
        coexistence_matrix = binary_data.T @ binary_data
        gene_sums = binary_data.sum(axis=0).A1
    else:
        coexistence_matrix = np.dot(binary_data.T, binary_data)
        gene_sums = binary_data.sum(axis=0)
    
    print(f"Coexistence matrix shape: {coexistence_matrix.shape}")
    

    denominator_matrix = gene_sums[:, np.newaxis] + gene_sums[np.newaxis, :] - coexistence_matrix
    
    # avioding division by zero
    denominator_matrix[denominator_matrix == 0] = epsilon
    
    result = coexistence_matrix / denominator_matrix
    
    print(f"Result shape: {result.shape}")
    print(f"Result type: {type(result)}")
    
    if sp.issparse(result):
        result = result.toarray()
    
    return result


def calculate_coexistence_ratio_all_as_base(data, epsilon=1e-10):
    print(f"Input data type: {type(data)}")
    print(f"Input data shape: {data.shape}")
    
    if sp.issparse(data):
        coexistence = (data > 0).astype(int)
        print(f"Coexistence shape: {coexistence.shape}")
        coexistence_matrix = coexistence.T @ coexistence
    else:
        coexistence = (data > 0).astype(int)
        print(f"Coexistence shape: {coexistence.shape}")
        coexistence_matrix = np.dot(coexistence.T, coexistence)
    
    print(f"Coexistence matrix shape: {coexistence_matrix.shape}")
    
    total_cells = data.shape[0]
    print(f"Total cells: {total_cells}")
    
    result = coexistence_matrix / (total_cells + epsilon)
    print(f"Result shape: {result.shape}")
    print(f"Result type: {type(result)}")
    
    if result.ndim != 2:
        print(f"Warning: Resulting matrix is not 2D. Shape: {result.shape}")
        if result.size == 1:
            result = np.array([[result]])
        else:
            result = np.atleast_2d(result)
    
    return result


def calculate_coexistence_ratio_probability(data, epsilon=1e-10):
    print(f"Input data type: {type(data)}")
    print(f"Input data shape: {data.shape}")
    
    if sp.issparse(data):
        binary_data = data.astype(bool).astype(int)
    else:
        binary_data = (data > 0).astype(int)
    
    print(f"Binary data shape: {binary_data.shape}")
    
    if binary_data.shape[1] == 1:
        return np.array([[1.0]])
    
    total_cells = binary_data.shape[0]  # total cells
    coexistence_matrix = binary_data.T @ binary_data
    
    gene_expression_ratios = binary_data.sum(axis=0) / total_cells  # 每个基因的表达比率
    
    # make sure gene_expression_ratios is a 1D array
    gene_expression_ratios = np.asarray(gene_expression_ratios).flatten()

    # calculate  coexistence ratio matrix
    
    expected_data = (gene_expression_ratios[:, np.newaxis] * gene_expression_ratios[np.newaxis, :])
    observed_data = (coexistence_matrix / total_cells)
    result = observed_data - expected_data
    
    print(f"Result shape: {result.shape}")
    print(f"Result type: {type(result)}")
    
    if sp.issparse(result):
        result = result.toarray()
    if sp.issparse(expected_data):
        expected_data = expected_data.toarray()
    if sp.issparse(observed_data):
        observed_data = observed_data.toarray()
    
    return (result,observed_data,expected_data)

def permutation_test(observed_data, expected_data, n_permutations=10000):
    observed_diff = np.mean(observed_data) - np.mean(expected_data)

    combined_data = np.concatenate([observed_data, expected_data])
    
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined_data)  # 随机重排
        perm_observed = combined_data[:len(observed_data)]
        perm_expected = combined_data[len(observed_data):]
        perm_diff = np.mean(perm_observed) - np.mean(perm_expected)
        perm_diffs.append(perm_diff)

    p_value = (np.abs(np.array(perm_diffs)) >= np.abs(observed_diff)).mean()
    
    return observed_diff, p_value

def element_wise_permutation_test(observed_matrix, expected_matrix, n_permutations=1000):
    """
    perform element-wise permutation test
    
    params:
    observed_matrix: from calculate_coexistence_ratio_probability calculated observed_data
    expected_matrix: from calculate_coexistence_ratio_probability calculated expected_data
    n_permutations: permutation times
    
    return:
    p_value_matrix: p value matrix
    diff_matrix: difference matrix
    """
    assert observed_matrix.shape == expected_matrix.shape, "观察矩阵和期望矩阵形状必须相同"
    
    rows, cols = observed_matrix.shape
    p_value_matrix = np.zeros((rows, cols))
    diff_matrix = observed_matrix - expected_matrix
    
    for i in range(rows):
        for j in range(cols):
            observed_val = observed_matrix[i, j]
            expected_val = expected_matrix[i, j]
            

            if observed_val == expected_val:
                p_value_matrix[i, j] = 1.0
                continue
            
            observed_diff = observed_val - expected_val
            
            values = np.array([observed_val, expected_val])
            
            perm_diffs = []
            for _ in range(n_permutations):
                np.random.shuffle(values)  # 随机重排
                perm_diff = values[0] - values[1]
                perm_diffs.append(perm_diff)
            
            p_value = np.mean(np.abs(np.array(perm_diffs)) >= np.abs(observed_diff))
            p_value_matrix[i, j] = p_value
    
    return p_value_matrix, diff_matrix

def merge_matrices_with_pvalue(observed_matrices_list, expected_matrices_list, gene_names, cell_type="Unknown", n_permutations=1000):
    """
    combine multiple observed and expected matrices into a single matrix,
    and calculate p-values for each element.
    
    params:
    observed_matrices_list: observerved matrix list, every element is a n×n numpy array
    expected_matrices_list: expected matrix list, every element is a n×n numpy array
    gene_names: gene names list, every element is a string, length is n, n is the number of genes
    cell_type: cell type, for instance: 'Bcell'
    n_permutations: permutation times, default is 1000
    
    return:
    merged_result: a dict containing merged observed and expected matrices, p-values, gene names and cell types
    """
    print(f"starting to merge matrices for cell type: {cell_type}")
    
    if len(observed_matrices_list) != len(expected_matrices_list):
        raise ValueError("observed_matrices_list and expected_matrices_list must have the same length")
    
    if len(observed_matrices_list) == 0:
        raise ValueError("matrices list must not be empty")
    
    # 检查所有矩阵的形状是否相同
    matrix_shape = observed_matrices_list[0].shape
    for obs_mat, exp_mat in zip(observed_matrices_list, expected_matrices_list):
        if obs_mat.shape != matrix_shape or exp_mat.shape != matrix_shape:
            raise ValueError("the shape of all matrices must be the same")
    
    n_matrices = len(observed_matrices_list)
    merged_observed = np.zeros(matrix_shape)
    merged_expected = np.zeros(matrix_shape)
    
    for obs_mat, exp_mat in zip(observed_matrices_list, expected_matrices_list):
        merged_observed += obs_mat
        merged_expected += exp_mat
    

    merged_observed /= n_matrices
    merged_expected /= n_matrices
    
    print(f" {n_matrices} matrices have been merged ")
    print(f"the shape of merged matrices: {merged_observed.shape}")
    
    p_value_matrix, diff_matrix = element_wise_permutation_test(
        merged_observed, merged_expected, n_permutations
    )
    
    # return the merged result
    merged_result = {
        'observed': merged_observed,
        'expected': merged_expected,
        'diff': diff_matrix,
        'p_value': p_value_matrix,
        'gene_names': gene_names,
        'cell_type': cell_type
    }
    
    return merged_result

def add_gene_statistics(adata):
    """
    Add gene expression statistics to an AnnData object (supports sparse/dense matrices)

    Parameters:
    adata: input AnnData object

    Returns:
    Pandas DataFrame containing the statistics
    """
    # get the expression matrix
    X = adata.X

    # initialize statistics dictionary
    stats = {
        'mean_expression': [],
        'expression_cell_count': [],
        'expression_cell_ratio': [],
        'expression_range': [],
        'variance': []
    }

    # compute statistics for each gene
    for gene_idx in range(X.shape[1]):
        gene_data = X[:, gene_idx]

        # compute mean
        mean = gene_data.mean() if sp.issparse(gene_data) else gene_data.mean()
        stats['mean_expression'].append(mean)

        # number of expressing cells
        expr_cells = gene_data.getnnz() if sp.issparse(gene_data) else np.count_nonzero(gene_data)
        ratio = expr_cells / X.shape[0]
        stats['expression_cell_ratio'].append(ratio)
        stats['expression_cell_count'].append(expr_cells)

        # expression range (min, max)
        min_val = gene_data.min() if sp.issparse(gene_data) else gene_data.min()
        max_val = gene_data.max() if sp.issparse(gene_data) else gene_data.max()
        stats['expression_range'].append((min_val, max_val))

        # variance (sparse-specialized calculation)
        if sp.issparse(gene_data):
            # variance = E[X^2] - (E[X])^2
            square_data = gene_data.power(2)
            var = square_data.mean() - mean**2
        else:
            var = gene_data.var()
        stats['variance'].append(var)

    # create statistics DataFrame
    gene_stats_df = pd.DataFrame(stats, index=adata.var_names)
    gene_stats_df.index.name = 'gene'

    # add to AnnData var
    adata.var = adata.var.assign(**gene_stats_df)

    return gene_stats_df

def zero_diagonal(df):
    # ensure input is a square matrix
    if df.shape[0] != df.shape[1]:
        raise ValueError("Input must be a square matrix")

    # operate directly on the underlying numpy array for efficiency
    np.fill_diagonal(df.values, 0)
    return df

def symmetric_pval_to_qval(pval_matrix, method='fdr_bh'):
    """
    Convert a symmetric p-value DataFrame to a symmetric q-value DataFrame
    using multiple testing correction on the upper triangle.

    :param pval_matrix: symmetric p-value DataFrame
    :param method: multiple testing correction method, default 'fdr_bh'
    :return: symmetric q-value DataFrame
    """
    pval_matrix = pval_matrix.copy().astype(float)

    # number of genes
    n_genes = pval_matrix.shape[0]

    # initialize all-NaN matrix (critical fix)
    qval_matrix = pd.DataFrame(np.full((n_genes, n_genes), np.nan),
                               index=pval_matrix.index,
                               columns=pval_matrix.columns)

    # upper triangle indices (excluding diagonal)
    upper_indices = np.triu_indices(n_genes, k=1)

    # extract upper-triangle p-values and flatten
    pval_flat = pval_matrix.values[upper_indices]

    # compute q-values (Benjamini-Hochberg correction)
    _, qval_flat, _, _ = multipletests(pval_flat, method=method)

    # fill upper triangle
    qval_matrix.values[upper_indices] = qval_flat

    # mirror to lower triangle (critical fix)
    qval_matrix = qval_matrix.combine_first(qval_matrix.T)

    # set diagonal elements
    np.fill_diagonal(qval_matrix.values, 1.0)

    return qval_matrix