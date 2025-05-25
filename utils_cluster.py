from scipy.linalg import qr, svd
from scipy.sparse.linalg import eigsh
from scipy import sparse
import numpy as np
import warnings
import numbers
from sklearn.cluster import k_means
from scipy.linalg import LinAlgError, qr, svd
from scipy.sparse import csc_matrix
from scipy.sparse.csgraph import laplacian as csgraph_laplacian

class SpectralClustering:
    def __init__(
        self,
        n_clusters,
        affinity,
        random_state,
        assign_labels,
        n_init = 10
    ) -> None:
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.random_state = random_state
        self.assign_labels = assign_labels
        self.n_init = n_init
        pass

    def fit_predict(self, X):
        if self.assign_labels == 'kmeans':
            eigenvalues, eigenvectors = eigsh(X, which='SM', k=self.n_clusters)
            eigenvalues = eigenvalues.real
            idx_min = np.argsort(eigenvalues)[range(1, self.n_clusters)]
        else:
            eigenvalues, eigenvectors = np.linalg.eig(X)
            eigenvalues = eigenvalues.real
            idx_min = np.argsort(eigenvalues)[range(self.n_clusters)]
        maps = eigenvectors[:, idx_min]
        maps = maps.real
        if self.assign_labels == "kmeans":
            _, labels, _ = k_means(
                maps,
                self.n_clusters,
                random_state=self.random_state,
                n_init=self.n_init,
                verbose=False,
            )
        elif self.assign_labels == "cluster_qr":
            labels = cluster_qr(maps)
        else:
            labels = discretize(maps, random_state=self.random_state)
        return labels


def discretize(
    vectors, *, copy=True, max_svd_restarts=30, n_iter_max=20, random_state=None
):
    """Search for a partition matrix which is closest to the eigenvector embedding.

    This implementation was proposed in [1]_.

    Parameters
    ----------
    vectors : array-like of shape (n_samples, n_clusters)
        The embedding space of the samples.

    copy : bool, default=True
        Whether to copy vectors, or perform in-place normalization.

    max_svd_restarts : int, default=30
        Maximum number of attempts to restart SVD if convergence fails

    n_iter_max : int, default=30
        Maximum number of iterations to attempt in rotation and partition
        matrix search if machine precision convergence is not reached

    random_state : int, RandomState instance, default=None
        Determines random number generation for rotation matrix initialization.
        Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    labels : array of integers, shape: n_samples
        The labels of the clusters.

    References
    ----------

    .. [1] `Multiclass spectral clustering, 2003
           Stella X. Yu, Jianbo Shi
           <https://people.eecs.berkeley.edu/~jordan/courses/281B-spring04/readings/yu-shi.pdf>`_

    Notes
    -----

    The eigenvector embedding is used to iteratively search for the
    closest discrete partition.  First, the eigenvector embedding is
    normalized to the space of partition matrices. An optimal discrete
    partition matrix closest to this normalized embedding multiplied by
    an initial rotation is calculated.  Fixing this discrete partition
    matrix, an optimal rotation matrix is calculated.  These two
    calculations are performed until convergence.  The discrete partition
    matrix is returned as the clustering solution.  Used in spectral
    clustering, this method tends to be faster and more robust to random
    initialization than k-means.

    """

    random_state = check_random_state(random_state)

    eps = np.finfo(float).eps
    n_samples, n_components = vectors.shape

    # Normalize the eigenvectors to an equal length of a vector of ones.
    # Reorient the eigenvectors to point in the negative direction with respect
    # to the first element.  This may have to do with constraining the
    # eigenvectors to lie in a specific quadrant to make the discretization
    # search easier.
    norm_ones = np.sqrt(n_samples)
    for i in range(vectors.shape[1]):
        vectors[:, i] = (vectors[:, i] / np.linalg.norm(vectors[:, i])) * norm_ones
        if vectors[0, i] != 0:
            vectors[:, i] = -1 * vectors[:, i] * np.sign(vectors[0, i])

    # Normalize the rows of the eigenvectors.  Samples should lie on the unit
    # hypersphere centered at the origin.  This transforms the samples in the
    # embedding space to the space of partition matrices.
    vectors = vectors / np.sqrt((vectors**2).sum(axis=1))[:, np.newaxis]

    svd_restarts = 0
    has_converged = False

    # If there is an exception we try to randomize and rerun SVD again
    # do this max_svd_restarts times.
    while (svd_restarts < max_svd_restarts) and not has_converged:
        # Initialize first column of rotation matrix with a row of the
        # eigenvectors
        rotation = np.zeros((n_components, n_components))
        rotation[:, 0] = vectors[random_state.randint(n_samples), :].T

        # To initialize the rest of the rotation matrix, find the rows
        # of the eigenvectors that are as orthogonal to each other as
        # possible
        c = np.zeros(n_samples)
        for j in range(1, n_components):
            # Accumulate c to ensure row is as orthogonal as possible to
            # previous picks as well as current one
            c += np.abs(np.dot(vectors, rotation[:, j - 1]))
            rotation[:, j] = vectors[c.argmin(), :].T

        last_objective_value = 0.0
        n_iter = 0

        while not has_converged:
            n_iter += 1

            t_discrete = np.dot(vectors, rotation)

            labels = t_discrete.argmax(axis=1)
            vectors_discrete = csc_matrix(
                (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
                shape=(n_samples, n_components),
            )

            t_svd = vectors_discrete.T * vectors

            try:
                U, S, Vh = np.linalg.svd(t_svd)
            except LinAlgError:
                svd_restarts += 1
                print("SVD did not converge, randomizing and trying again")
                break

            ncut_value = 2.0 * (n_samples - S.sum())
            if (abs(ncut_value - last_objective_value) < eps) or (n_iter > n_iter_max):
                has_converged = True
            else:
                # otherwise calculate rotation and continue
                last_objective_value = ncut_value
                rotation = np.dot(Vh.T, U.T)

    if not has_converged:
        raise LinAlgError("SVD did not converge")
    return labels


def _set_diag(laplacian, value, norm_laplacian):
    """Set the diagonal of the laplacian matrix and convert it to a
    sparse format well suited for eigenvalue decomposition.

    Parameters
    ----------
    laplacian : {ndarray, sparse matrix}
        The graph laplacian.

    value : float
        The value of the diagonal.

    norm_laplacian : bool
        Whether the value of the diagonal should be changed or not.

    Returns
    -------
    laplacian : {array, sparse matrix}
        An array of matrix in a form that is well suited to fast
        eigenvalue decomposition, depending on the band width of the
        matrix.
    """
    n_nodes = laplacian.shape[0]
    # We need all entries in the diagonal to values
    if not sparse.isspmatrix(laplacian):
        if norm_laplacian:
            laplacian.flat[:: n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        if norm_laplacian:
            diag_idx = laplacian.row == laplacian.col
            laplacian.data[diag_idx] = value
        # If the matrix has a small number of diagonals (as in the
        # case of structured matrices coming from images), the
        # dia format might be best suited for matvec products:
        n_diags = np.unique(laplacian.row - laplacian.col).size
        if n_diags <= 7:
            # 3 or less outer diagonals on each side
            laplacian = laplacian.todia()
        else:
            # csr has the fastest matvec and is thus best suited to
            # arpack
            laplacian = laplacian.tocsr()
    return laplacian


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )


def check_symmetric(array, *, tol=1e-10, raise_warning=True, raise_exception=False):
    """Make sure that array is 2D, square and symmetric.

    If the array is not symmetric, then a symmetrized version is returned.
    Optionally, a warning or exception is raised if the matrix is not
    symmetric.

    Parameters
    ----------
    array : {ndarray, sparse matrix}
        Input object to check / convert. Must be two-dimensional and square,
        otherwise a ValueError will be raised.

    tol : float, default=1e-10
        Absolute tolerance for equivalence of arrays. Default = 1E-10.

    raise_warning : bool, default=True
        If True then raise a warning if conversion is required.

    raise_exception : bool, default=False
        If True then raise an exception if array is not symmetric.

    Returns
    -------
    array_sym : {ndarray, sparse matrix}
        Symmetrized version of the input array, i.e. the average of array
        and array.transpose(). If sparse, then duplicate entries are first
        summed and zeros are eliminated.
    """
    if (array.ndim != 2) or (array.shape[0] != array.shape[1]):
        raise ValueError(
            "array must be 2-dimensional and square. shape = {0}".format(array.shape)
        )

    symmetric = np.allclose(array, array.T, atol=tol)

    if not symmetric:
        if raise_exception:
            raise ValueError("Array must be symmetric")
        if raise_warning:
            warnings.warn(
                "Array is not symmetric, and will be converted "
                "to symmetric by average with its transpose.",
                stacklevel=2,
            )
        array = 0.5 * (array + array.T)

    return array


def _init_arpack_v0(size, random_state):
    """Initialize the starting vector for iteration in ARPACK functions.

    Initialize a ndarray with values sampled from the uniform distribution on
    [-1, 1]. This initialization model has been chosen to be consistent with
    the ARPACK one as another initialization can lead to convergence issues.

    Parameters
    ----------
    size : int
        The size of the eigenvalue vector to be initialized.

    random_state : int, RandomState instance or None, default=None
        The seed of the pseudo random number generator used to generate a
        uniform distribution. If int, random_state is the seed used by the
        random number generator; If RandomState instance, random_state is the
        random number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    Returns
    -------
    v0 : ndarray of shape (size,)
        The initialized vector.
    """
    random_state = check_random_state(random_state)
    v0 = random_state.uniform(-1, 1, size)
    return v0


def _deterministic_vector_sign_flip(u):
    """Modify the sign of vectors for reproducibility.

    Flips the sign of elements of all the vectors (rows of u) such that
    the absolute maximum element of each vector is positive.

    Parameters
    ----------
    u : ndarray
        Array with vectors as its rows.

    Returns
    -------
    u_flipped : ndarray with same shape as u
        Array with the sign flipped vectors as its rows.
    """
    max_abs_rows = np.argmax(np.abs(u), axis=1)
    signs = np.sign(u[range(u.shape[0]), max_abs_rows])
    u *= signs[:, np.newaxis]
    return u


def spectral_embedding(
    adjacency,
    *,
    n_components=8,
    eigen_solver=None,
    random_state=None,
    eigen_tol=0.0,
    norm_laplacian=True,
    drop_first=True,
):
    """Project the sample on the first eigenvectors of the graph Laplacian.

    The adjacency matrix is used to compute a normalized graph Laplacian
    whose spectrum (especially the eigenvectors associated to the
    smallest eigenvalues) has an interpretation in terms of minimal
    number of cuts necessary to split the graph into comparably sized
    components.

    This embedding can also 'work' even if the ``adjacency`` variable is
    not strictly the adjacency matrix of a graph but more generally
    an affinity or similarity matrix between samples (for instance the
    heat kernel of a euclidean distance matrix or a k-NN matrix).

    However care must taken to always make the affinity matrix symmetric
    so that the eigenvector decomposition works as expected.

    Note : Laplacian Eigenmaps is the actual algorithm implemented here.

    Read more in the :ref:`User Guide <spectral_embedding>`.

    Parameters
    ----------
    adjacency : {array-like, sparse graph} of shape (n_samples, n_samples)
        The adjacency matrix of the graph to embed.

    n_components : int, default=8
        The dimension of the projection subspace.

    eigen_solver : {'arpack', 'lobpcg', 'amg'}, default=None
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities. If None, then ``'arpack'`` is
        used.

    random_state : int, RandomState instance or None, default=None
        A pseudo random number generator used for the initialization
        of the lobpcg eigen vectors decomposition when `eigen_solver ==
        'amg'`, and for the K-Means initialization. Use an int to make
        the results deterministic across calls (See
        :term:`Glossary <random_state>`).

        .. note::
            When using `eigen_solver == 'amg'`,
            it is necessary to also fix the global numpy seed with
            `np.random.seed(int)` to get deterministic results. See
            https://github.com/pyamg/pyamg/issues/139 for further
            information.

    eigen_tol : float, default=0.0
        Stopping criterion for eigendecomposition of the Laplacian matrix
        when using arpack eigen_solver.

    norm_laplacian : bool, default=True
        If True, then compute symmetric normalized Laplacian.

    drop_first : bool, default=True
        Whether to drop the first eigenvector. For spectral embedding, this
        should be True as the first eigenvector should be constant vector for
        connected graph, but for spectral clustering, this should be kept as
        False to retain the first eigenvector.

    Returns
    -------
    embedding : ndarray of shape (n_samples, n_components)
        The reduced samples.

    Notes
    -----
    Spectral Embedding (Laplacian Eigenmaps) is most useful when the graph
    has one connected component. If there graph has many components, the first
    few eigenvectors will simply uncover the connected components of the graph.

    References
    ----------
    * https://en.wikipedia.org/wiki/LOBPCG

    * :doi:`"Toward the Optimal Preconditioned Eigensolver: Locally Optimal
      Block Preconditioned Conjugate Gradient Method",
      Andrew V. Knyazev
      <10.1137/S1064827500366124>`
    """
    adjacency = check_symmetric(adjacency)

    if eigen_solver is None:
        eigen_solver = "arpack"
    elif eigen_solver not in ("arpack", "lobpcg", "amg"):
        raise ValueError(
            "Unknown value for eigen_solver: '%s'."
            "Should be 'amg', 'arpack', or 'lobpcg'" % eigen_solver
        )

    random_state = check_random_state(random_state)

    n_nodes = adjacency.shape[0]
    # Whether to drop the first eigenvector
    if drop_first:
        n_components = n_components + 1

    # if not _graph_is_connected(adjacency):
    #     warnings.warn(
    #         "Graph is not fully connected, spectral embedding may not work as expected."
    #     )

    laplacian, dd = csgraph_laplacian(
        adjacency, normed=norm_laplacian, return_diag=True
    )
    # laplacian = np.copy(adjacency)
    if (
        eigen_solver == "arpack"
        or eigen_solver != "lobpcg"
        and (not sparse.isspmatrix(laplacian) or n_nodes < 5 * n_components)
    ):
        # lobpcg used with eigen_solver='amg' has bugs for low number of nodes
        # for details see the source code in scipy:
        # https://github.com/scipy/scipy/blob/v0.11.0/scipy/sparse/linalg/eigen
        # /lobpcg/lobpcg.py#L237
        # or matlab:
        # https://www.mathworks.com/matlabcentral/fileexchange/48-lobpcg-m
        # laplacian = _set_diag(laplacian, 1, norm_laplacian)

        # Here we'll use shift-invert mode for fast eigenvalues
        # (see https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
        #  for a short explanation of what this means)
        # Because the normalized Laplacian has eigenvalues between 0 and 2,
        # I - L has eigenvalues between -1 and 1.  ARPACK is most efficient
        # when finding eigenvalues of largest magnitude (keyword which='LM')
        # and when these eigenvalues are very large compared to the rest.
        # For very large, very sparse graphs, I - L can have many, many
        # eigenvalues very near 1.0.  This leads to slow convergence.  So
        # instead, we'll use ARPACK's shift-invert mode, asking for the
        # eigenvalues near 1.0.  This effectively spreads-out the spectrum
        # near 1.0 and leads to much faster convergence: potentially an
        # orders-of-magnitude speedup over simply using keyword which='LA'
        # in standard mode.
        try:
            # We are computing the opposite of the laplacian inplace so as
            # to spare a memory allocation of a possibly very large array
            laplacian *= -1
            v0 = _init_arpack_v0(laplacian.shape[0], random_state)
            _, diffusion_map = eigsh(
                laplacian, k=n_components, sigma=1.0, which="LM", tol=eigen_tol, v0=v0
            )
            embedding = diffusion_map.T[n_components::-1]
            if norm_laplacian:
                # recover u = D^-1/2 x from the eigenvector output x
                embedding = embedding / dd
        except RuntimeError:
            # When submatrices are exactly singular, an LU decomposition
            # in arpack fails. We fallback to lobpcg
            eigen_solver = "lobpcg"
            # Revert the laplacian to its opposite to have lobpcg work
            laplacian *= -1

    embedding = _deterministic_vector_sign_flip(embedding)
    if drop_first:
        return embedding[1:n_components].T
    else:
        return embedding[:n_components].T


def cluster_qr(vectors):
    """Find the discrete partition closest to the eigenvector embedding.

        This implementation was proposed in [1]_.

    .. versionadded:: 1.1

        Parameters
        ----------
        vectors : array-like, shape: (n_samples, n_clusters)
            The embedding space of the samples.

        Returns
        -------
        labels : array of integers, shape: n_samples
            The cluster labels of vectors.

        References
        ----------
        .. [1] :doi:`Simple, direct, and efficient multi-way spectral clustering, 2019
            Anil Damle, Victor Minden, Lexing Ying
            <10.1093/imaiai/iay008>`

    """

    k = vectors.shape[1]
    _, _, piv = qr(vectors.T, pivoting=True)
    ut, _, v = svd(vectors[piv[:k], :].T)
    vectors = abs(np.dot(vectors, np.dot(ut, v.conj())))
    return vectors.argmax(axis=1)


def spectral_clustering(
    affinity,
    *,
    n_clusters=8,
    n_components=None,
    eigen_solver=None,
    random_state=None,
    n_init=10,
    eigen_tol=0.0,
    assign_labels="kmeans",
    verbose=False,
):
    """Apply clustering to a projection of the normalized Laplacian.

    In practice Spectral Clustering is very useful when the structure of
    the individual clusters is highly non-convex or more generally when
    a measure of the center and spread of the cluster is not a suitable
    description of the complete cluster. For instance, when clusters are
    nested circles on the 2D plane.

    If affinity is the adjacency matrix of a graph, this method can be
    used to find normalized graph cuts [1]_, [2]_.

    Read more in the :ref:`User Guide <spectral_clustering>`.

    Parameters
    ----------
    affinity : {array-like, sparse matrix} of shape (n_samples, n_samples)
        The affinity matrix describing the relationship of the samples to
        embed. **Must be symmetric**.

        Possible examples:
          - adjacency matrix of a graph,
          - heat kernel of the pairwise distance matrix of the samples,
          - symmetric k-nearest neighbours connectivity matrix of the samples.

    n_clusters : int, default=None
        Number of clusters to extract.

    n_components : int, default=n_clusters
        Number of eigenvectors to use for the spectral embedding.

    eigen_solver : {None, 'arpack', 'lobpcg', or 'amg'}
        The eigenvalue decomposition method. If None then ``'arpack'`` is used.
        See [4]_ for more details regarding ``'lobpcg'``.
        Eigensolver ``'amg'`` runs ``'lobpcg'`` with optional
        Algebraic MultiGrid preconditioning and requires pyamg to be installed.
        It can be faster on very large sparse problems [6]_ and [7]_.

    random_state : int, RandomState instance, default=None
        A pseudo random number generator used for the initialization
        of the lobpcg eigenvectors decomposition when `eigen_solver ==
        'amg'`, and for the K-Means initialization. Use an int to make
        the results deterministic across calls (See
        :term:`Glossary <random_state>`).

        .. note::
            When using `eigen_solver == 'amg'`,
            it is necessary to also fix the global numpy seed with
            `np.random.seed(int)` to get deterministic results. See
            https://github.com/pyamg/pyamg/issues/139 for further
            information.

    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of n_init
        consecutive runs in terms of inertia. Only used if
        ``assign_labels='kmeans'``.

    eigen_tol : float, default=0.0
        Stopping criterion for eigendecomposition of the Laplacian matrix
        when using arpack eigen_solver.

    assign_labels : {'kmeans', 'discretize', 'cluster_qr'}, default='kmeans'
        The strategy to use to assign labels in the embedding
        space.  There are three ways to assign labels after the Laplacian
        embedding.  k-means can be applied and is a popular choice. But it can
        also be sensitive to initialization. Discretization is another
        approach which is less sensitive to random initialization [3]_.
        The cluster_qr method [5]_ directly extracts clusters from eigenvectors
        in spectral clustering. In contrast to k-means and discretization, cluster_qr
        has no tuning parameters and is not an iterative method, yet may outperform
        k-means and discretization in terms of both quality and speed.

        .. versionchanged:: 1.1
           Added new labeling method 'cluster_qr'.

    verbose : bool, default=False
        Verbosity mode.

        .. versionadded:: 0.24

    Returns
    -------
    labels : array of integers, shape: n_samples
        The labels of the clusters.

    Notes
    -----
    The graph should contain only one connected component, elsewhere
    the results make little sense.

    This algorithm solves the normalized cut for `k=2`: it is a
    normalized spectral clustering.

    References
    ----------

    .. [1] :doi:`Normalized cuts and image segmentation, 2000
           Jianbo Shi, Jitendra Malik
           <10.1109/34.868688>`

    .. [2] :doi:`A Tutorial on Spectral Clustering, 2007
           Ulrike von Luxburg
           <10.1007/s11222-007-9033-z>`

    .. [3] `Multiclass spectral clustering, 2003
           Stella X. Yu, Jianbo Shi
           <https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf>`_

    .. [4] :doi:`Toward the Optimal Preconditioned Eigensolver:
           Locally Optimal Block Preconditioned Conjugate Gradient Method, 2001
           A. V. Knyazev
           SIAM Journal on Scientific Computing 23, no. 2, pp. 517-541.
           <10.1137/S1064827500366124>`

    .. [5] :doi:`Simple, direct, and efficient multi-way spectral clustering, 2019
           Anil Damle, Victor Minden, Lexing Ying
           <10.1093/imaiai/iay008>`

    .. [6] :doi:`Multiscale Spectral Image Segmentation Multiscale preconditioning
           for computing eigenvalues of graph Laplacians in image segmentation, 2006
           Andrew Knyazev
           <10.13140/RG.2.2.35280.02565>`

    .. [7] :doi:`Preconditioned spectral clustering for stochastic block partition
           streaming graph challenge (Preliminary version at arXiv.)
           David Zhuzhunashvili, Andrew Knyazev
           <10.1109/HPEC.2017.8091045>`
    """
    if assign_labels not in ("kmeans", "discretize", "cluster_qr"):
        raise ValueError(
            "The 'assign_labels' parameter should be "
            "'kmeans' or 'discretize', or 'cluster_qr', "
            f"but {assign_labels!r} was given"
        )
    if isinstance(affinity, np.matrix):
        raise TypeError(
            "spectral_clustering does not support passing in affinity as an "
            "np.matrix. Please convert to a numpy array with np.asarray. For "
            "more information see: "
            "https://numpy.org/doc/stable/reference/generated/numpy.matrix.html",  # noqa
        )

    # random_state = check_random_state(random_state)
    n_components = n_clusters if n_components is None else n_components

    # We now obtain the real valued solution matrix to the
    # relaxed Ncut problem, solving the eigenvalue problem
    # L_sym x = lambda x  and recovering u = D^-1/2 x.
    # The first eigenvector is constant only for fully connected graphs
    # and should be kept for spectral clustering (drop_first = False)
    # See spectral_embedding documentation.
    maps = spectral_embedding(
        affinity,
        n_components=n_components,
        eigen_solver=eigen_solver,
        random_state=random_state,
        eigen_tol=eigen_tol,
        drop_first=False,
    )
    # maps = maps[:, :-1]
    # maps = maps[:, 1:]
    if verbose:
        print(f"Computing label assignment using {assign_labels}")

    if assign_labels == "kmeans":
        _, labels, _ = k_means(
            maps, n_clusters, random_state=random_state, n_init=n_init, verbose=verbose
        )
    elif assign_labels == "cluster_qr":
        labels = cluster_qr(maps)
    else:
        labels = discretize(maps, random_state=random_state)

    return labels
