from math import comb, log, log1p

def log_repeat_evalue(bg, n, r, l, k):

    p_match = sum(freq**2 for freq in bg.values())
    p_all_flanks_match = sum(freq**r for freq in bg.values())

    if p_all_flanks_match == 1: return float("-inf")

    log_arm = log(comb(l, k)) + (l - k) * log(p_match)
    if k: log_arm += k * log1p(-p_match)

    return log(comb(n, r)) + (r - 1) * log_arm + 2 * log1p(-p_all_flanks_match)

def repeat_evalue(bg, n, r, l, k):

    p_match = sum(freq**2 for freq in bg.values())

    p_arm = (
        comb(l, k)
        * p_match**(l - k)
        * (1 - p_match)**k
    )

    p_all_flanks_match = sum(freq**r for freq in bg.values())

    p_maximal = (1 - p_all_flanks_match)**2

    search_space = comb(n, r)

    return search_space * p_arm**(r - 1) * p_maximal

if __name__ == "__main__":

    bg = {"A": 0.3000 , "C": 0.1867, "G": 0.1693, "T": 0.3440}

    n = 1000
    r = 2
    l = 14
    k = 1

    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    u = 2

    if u == 1:

        k_values = np.arange(0, 4)
        evalue_values = [repeat_evalue(bg, n, r, l, k) for k in k_values]
        sns.lineplot(x=k_values, y=evalue_values)
        plt.yscale('log')
        plt.show()

    if u == 2:

        k_values = np.arange(0, 4)
        l_values = np.arange(10, 30)
        evalue_values = np.array(
            [[repeat_evalue(bg, n, r, l, k) for k in k_values] for l in l_values]
        )

        evalue_values = np.where(evalue_values > 1e-2, np.nan, evalue_values)

        log_evalue_values = np.log10(np.clip(evalue_values, np.finfo(float).tiny, None))
        ax = sns.heatmap(
            log_evalue_values,
            cmap='viridis',
            xticklabels=k_values,
            yticklabels=l_values,
        )
        ax.set_xlabel('k')
        ax.set_ylabel('l')
        ax.collections[0].colorbar.set_label(r'$\log_{10} E$')
        plt.show()
