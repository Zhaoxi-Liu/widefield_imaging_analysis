#%%
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA, FastICA
import matplotlib.pyplot as plt

#%%
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(1 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal
s4 = signal.sawtooth(2 * np.pi * time, 0.5)  # Triangular wave



S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# # Mixing matrix 行对应混合信号，列对应原始信号
# A = np.array([[1, 1, 1, 1], [0.5, 4, 1.0, 2], [1.5, 1.0, 4.0, 0], 
#               [4, 0, 1, 3], [2, 3, 4, 6], [5,4,3,2]])  
A = np.array([[1, 1, 1], [0.5, 4, 1.0], [1.5, 1.0, 4.0], 
              [4, 0, 1], [2, 3, 4], [5,4,3]])
X = np.dot(S, A.T)  # Generate observations

#%% fastICA
ica = FastICA(n_components=3, whiten="arbitrary-variance")
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

# We can `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

#%% PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

#%% jader
from jadeR import jadeR
Xt= X.T
demix_matrix=np.asarray(jadeR(Xt, m=3, verbose=True))
IC = demix_matrix @ Xt

#%% plot
plt.figure(figsize=(14, 16))

# models = [X, S, IC.T, S_, H]
models = [X, S, IC.T]

names = [
    "Observations (mixed signal)",
    "True Sources",
    "ICA jade recovered signals",
    # "fastICA recovered signals",
    # "PCA recovered signals",
]
colors = ["red", "steelblue", "orange", "green", 'purple', 'brown']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(3, 1, ii)
    plt.title(name,fontsize=20)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.tight_layout()
plt.savefig('ICA results comparison.png')
plt.show()



# %%
def calculate_DW(e):
    numerator = np.sum((e[1:] - e[:-1]) ** 2)
    denominator = np.sum(e ** 2)
    DW = numerator / denominator
    return DW

def DW_ICA(X, min_IC=2, max_IC=None, nIC_step=1, verbose=False):
    N = X.shape[0]
    if max_IC == None: max_IC=N
    
    DW_values = []
    for n_IC in range(min_IC, max_IC, nIC_step):
        print(f'\ntest n_IC = {n_IC}')
        demix_matrix = jadeR(X, m=n_IC, verbose=verbose)
        IC = demix_matrix @ X

        Xk = np.linalg.pinv(demix_matrix) @ IC
        Rk = X - Xk
        DW_sample_values = [calculate_DW(np.asarray(Rk[i, :]).ravel()) for i in range(N)]
        DW_values.append(DW_sample_values)
        print(f"DW: {DW_sample_values}")

    IC_num = range(min_IC, max_IC, nIC_step)

    return IC_num, np.asarray(DW_values)

        
    
#%%
def LCC_ICA(X, min_IC=2, max_IC=None, nIC_step=1, verbose=False):
    N = X.shape[0]
    if max_IC == None: max_IC=N-1

    for n_IC in range(min_IC, max_IC, nIC_step):
        print(f'\ntest n_IC = {n_IC}')
        demix_matrix=jadeR(X, m=n_IC, verbose=verbose)
        IC = demix_matrix @ X
#
        corr_matrix = np.corrcoef(IC)
        np.fill_diagonal(corr_matrix, 0)
        if np.max(np.abs(corr_matrix)) > 0.1:
            k = n_IC - 1
            print(f'\n\nk = {k}')
            selected_demix_matrix=jadeR(X, m=k, verbose=False)
            selected_IC = selected_demix_matrix @ X
            print(f'finish LCC_algorithm')
            return k, selected_IC, selected_demix_matrix
        


#%%
Xt= X.T
k, ICs, demix_matrix = LCC_ICA(Xt)
print(f'Optimal IC number: {k}')


# %%
Xt= X.T
IC_num, DW_value = DW_ICA(Xt)
plot_heatmap(DW_value, xlabel=None, ylabel=IC_num, xtitle='sample index', ytitle='n_ICA', cmap='coolwarm', title='ICA DW_values', outfile=None)

# %%
# %%
