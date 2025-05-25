# Output: a dict with three paired elements:
# 1. data (dict): with R paired elements.
# > Each paired element is (l, l_data) where l_data is a dict with three paired elements
# > l_data = {'obs': np.array, 'tre': np.array, 'outcome': np.array, 'interior': bool}
# 2. design (str): 'I', 'G', 'C'
# 3. adj (np.array): R-by-R matrix

import numpy as np
from scipy.linalg import block_diag
from utils import cov2cor

def individual_policy(obs, p=0.5):
    """
    obs (np.array): R-by-d matrix
    p (float): a scalar
    """
    R = obs.shape[0]
    A = np.random.binomial(n=1, p=p, size=(R, 1))
    return A


def global_policy(obs, p=0.5):
    R = obs.shape[0]
    A = np.random.binomial(n=1, p=p, size=1) * np.ones((R, 1))
    return A


def AA_policy(obs, p=1.0):
    R = obs.shape[0]
    A = np.random.binomial(n=1, p=p, size=1) * np.ones((R, 1))
    return A


def cluster_policy(obs, cluster, p=0.5):
    """
    cluster: a dict including the indices of region in each cluster, e.g.,
    cluster[0] = [0, 2, 4] means the first, third, fifth regions belong to the first cluster
    Notice that, the union of cluster[0], ..., cluster[m-1] must be [0, 1, ..., R-1]
    """

    def fixed_cluster_policy(obs, p=0.5):
        R = obs.shape[0]
        A = np.zeros((R, 1))
        for _, value in cluster.items():
            A[value] = np.random.binomial(n=1, p=p, size=1) * np.ones((len(value), 1))
        return A

    return fixed_cluster_policy


class EnvSimulator:
    def rectangle_hexagon_model(self, grid_size):
        positions = []
        r = 1  # The radius of a single hexagon (distance from center to any vertex)
        dx = r * 3 / 2  # Horizontal distance between centers
        dy = np.sqrt(3) * r  # Vertical distance between centers

        # Calculate number of hexagons possible in the horizontal and vertical dimensions
        height = 11
        nx = int(grid_size / dx) + 1
        ny = int(height / dy) + 1

        grid = {}
        index = 0
        for x in range(nx):
            for y in range(ny):
                x_pos = x * dx
                y_pos = y * dy + (x % 2) * dy / 2  # Stagger odd columns
                if x_pos <= grid_size and y_pos <= height:
                    positions.append((x_pos, y_pos))
                    grid[(x, y)] = index
                    index += 1

        # Establish adjacency
        adjacency = []
        for (x, y), idx in grid.items():
            neighbors = [
                (x + 1, y), (x - 1, y),                      # Right and left
                (x, y + 1), (x, y - 1),                      # Top and bottom
                (x - 1, y + 1) if x % 2 == 1 else (x + 1, y - 1),  # Top-right or bottom-left if odd row
                (x + 1, y + 1) if x % 2 == 1 else (x - 1, y - 1)   # Top-left or bottom-right if odd row
            ]
            # Only include valid neighbors that are within the grid
            adjacency.append([grid[n] for n in neighbors if n in grid])

        R = len(adjacency)
        return positions, adjacency, R

    def fan_hexagon_model(self, grid_size, fan_angle_degrees=120):
        """Generate positions for hexagons tiling within a fan-shaped region and return adjacency."""
        positions = []
        r = 1  # The radius of a single hexagon (side length)
        dx = r * 3 / 2  # Horizontal distance between centers
        dy = np.sqrt(3) * r  # Vertical distance between centers
        fan_angle = np.radians(fan_angle_degrees)  # Convert degrees to radians

        # Determine the range of hexagon centers that fit in the circle
        x_max = int((grid_size * 2) / dx) + 1
        y_max = int((grid_size * 2) / dy) + 1

        grid = {}
        index = 0
        for x in range(-x_max, x_max + 1):
            for y in range(-y_max, y_max + 1):
                x_pos = x * dx
                y_pos = y * dy + (x % 2) * dy / 2  # Stagger odd columns
                # Convert to polar coordinates to check if within fan angle
                theta = np.arctan2(y_pos, x_pos)
                if x_pos**2 + y_pos**2 <= grid_size**2 and abs(theta) <= fan_angle / 2:
                    positions.append((x_pos, y_pos))
                    grid[(x, y)] = index
                    index += 1

        # Establish adjacency
        adjacency = []
        for (x, y), idx in grid.items():
            neighbors = [
                (x + 1, y), (x - 1, y),  # Right and left
                (x, y + 1), (x, y - 1),  # Top and bottom
                (x + 1, y - 1) if x % 2 == 0 else (x - 1, y + 1),  # Top-left or bottom-right
                (x - 1, y - 1) if x % 2 == 0 else (x + 1, y + 1)   # Top-right or bottom-left
            ]
            adjacency.append([grid[n] for n in neighbors if n in grid])

        R = len(adjacency)
        return positions, adjacency, R

    def circle_hexagon_model(self, grid_size):
        """Generate positions for hexagons tiling within a circle and their adjacency."""
        positions = []
        r = 1  # The radius of a single hexagon (side length)
        dx = r * 3 / 2  # Horizontal distance between centers
        dy = np.sqrt(3) * r  # Vertical distance between centers

        # Determine the range of hexagon centers that fit in the circle
        # Compute number of hexagons possible along the width of the circle diameter
        x_max = int((grid_size * 2) / dx) + 1
        y_max = int((grid_size * 2) / dy) + 1

        # Populate the grid and check which centers are inside the circle
        grid = {}
        index = 0
        for x in range(-x_max, x_max + 1):
            for y in range(-y_max, y_max + 1):
                x_pos = x * dx
                y_pos = y * dy + (x % 2) * dy / 2  # Stagger odd columns
                # Check if the center of the hexagon is within the circle
                if x_pos**2 + y_pos**2 <= grid_size**2:
                    positions.append((x_pos, y_pos))
                    grid[(x, y)] = index
                    index += 1

        # Establish adjacency
        adjacency = []
        for (x, y), idx in grid.items():
            # Adjacent hexagons for a hexagonal grid
            neighbors = [
                (x + 1, y), (x - 1, y),  # Right and left
                (x, y + 1), (x, y - 1),  # Top and bottom
                (x + 1, y - 1) if x % 2 == 0 else (x - 1, y + 1),  # Top-left or bottom-right
                (x - 1, y - 1) if x % 2 == 0 else (x + 1, y + 1)   # Top-right or bottom-left
            ]
            # Filter valid neighbors that are within the circle
            adjacency.append([grid[n] for n in neighbors if n in grid])

        return positions, adjacency, len(adjacency)

    def hexagon_model(self, grid_size):
        """
        Output:
        > grid (np.array): an R-by-2 matrix used for generating alpha and beta in (3) in Yang, et al (2024)
        > adj_indices (list): each element of `adj_indices` is list that includes the indices of adjacent region, e.g., `adj_indices[0]=[1, 3, 5]` means the first region's neighbours is the second, fourth, and sixth regions.
        > R (int): the number of region
        """

        R = grid_size**2

        # grid: coordinates
        grid = []
        labels = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = j * 3 / 2
                y = np.sqrt(3) * (i + 0.5 * (j % 2))
                grid.append((x, y))
                labels.append((i, j))
        grid = np.array(grid)
        # grid[:, 0] = np.max(grid[:, 0])
        # grid[:, 1] = np.max(grid[:, 1])

        # adj_indices
        adj_indices = [[] for _ in range(R)]
        for q, r in labels:
            index = q * grid_size + r
            directions_even = [(0, -1), (-1, -1), (-1, 0), (1, 0), (0, 1), (-1, 1)]
            directions_odd = [(1, -1), (0, -1), (-1, 0), (1, 0), (0, 1), (1, 1)]
            directions = directions_even if r % 2 == 0 else directions_odd
            for dq, dr in directions:
                nq, nr = q + dq, r + dr
                for i in range(self.exposure):
                    if 0 <= nq+i < grid_size and 0 <= nr+i < grid_size:
                        neighbor_index = (nq+i) * grid_size + (nr+i)
                        adj_indices[index].append(neighbor_index)
                        adj_indices[neighbor_index].append(index)
                    if (i > 1) and (0 <= nq-i < grid_size and 0 <= nr-i < grid_size):
                        neighbor_index = (nq-i) * grid_size + (nr-i)
                        adj_indices[index].append(neighbor_index)
                        adj_indices[neighbor_index].append(index)

        adj_indices = [list(set(x)) for x in adj_indices]
        return grid, adj_indices, R

    def square_model(self, grid_size):
        R = grid_size**2

        # grid: coordinates
        grid = []
        labels = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = j
                y = i
                grid.append((x, y))
                labels.append((i, j))
        grid = np.array(grid)

        # adj_indices
        adj_indices = [[] for _ in range(R)]
        for q, r in labels:
            index = q * grid_size + r
            directions = [(0, -1), (-1, 0), (1, 0), (0, 1)]
            for dq, dr in directions:
                nq, nr = q + dq, r + dr
                for i in range(self.exposure):
                    if 0 <= nq+i < grid_size and 0 <= nr+i < grid_size:
                        neighbor_index = (nq+i) * grid_size + (nr+i)
                        adj_indices[index].append(neighbor_index)
                        adj_indices[neighbor_index].append(index)
                    if (i > 1) and (0 <= nq-i < grid_size and 0 <= nr-i < grid_size):
                        neighbor_index = (nq-i) * grid_size + (nr-i)
                        adj_indices[index].append(neighbor_index)
                        adj_indices[neighbor_index].append(index)
        return grid, adj_indices, R

    def triangle_model(self, grid_size):
        ## TODO
        # return grid, adj_indices, R
        pass

    def init_obs_model1(self):
        init_obs = np.random.normal(size=(self.R, self.dim_obs), loc=4.0, scale=1.0)
        return init_obs

    def init_obs_model2(self):
        next_obs = np.random.normal(size=(self.R, self.dim_obs), loc=4.0, scale=1.0)
        next_obs = np.clip(next_obs, a_min=3, a_max=5)
        return next_obs

    def init_obs_model3(self):
        next_obs = np.random.normal(size=(self.R, self.dim_obs), loc=4.0, scale=1.0)
        next_obs = np.clip(next_obs, a_min=3, a_max=5)
        return next_obs

    def f1_model1(self, x):
        """
        @ Description: see Numerical Experiments in page 21 of Yang et al (2024)
        """
        a0, a1, a2, a3, b1, b2, b3 = self.f1_model1_params
        result = (
            a0
            + a1 * np.cos(1 * np.pi * x)
            + a2 * np.cos(2 * np.pi * x)
            + a3 * np.cos(3 * np.pi * x)
            + b1 * np.sin(1 * np.pi * x)
            + b2 * np.sin(2 * np.pi * x)
            + b3 * np.sin(3 * np.pi * x)
        )
        return result

    def g1_model1(self, y):
        a0, a1, a2, a3, b1, b2, b3 = self.g1_model1_params
        result = (
            a0
            + a1 * np.cos(1 * np.pi * y)
            + a2 * np.cos(2 * np.pi * y)
            + a3 * np.cos(3 * np.pi * y)
            + b1 * np.sin(1 * np.pi * y)
            + b2 * np.sin(2 * np.pi * y)
            + b3 * np.sin(3 * np.pi * y)
        )
        return result

    def f2_model1(self, x):
        a0, a1, a2, a3, b1, b2, b3 = self.f2_model1_params
        result = (
            a0
            + a1 * np.cos(1 * np.pi * x)
            + a2 * np.cos(2 * np.pi * x)
            + a3 * np.cos(3 * np.pi * x)
            + b1 * np.sin(1 * np.pi * x)
            + b2 * np.sin(2 * np.pi * x)
            + b3 * np.sin(3 * np.pi * x)
        )
        return result

    def g2_model1(self, y):
        a0, a1, a2, a3, b1, b2, b3 = self.g2_model1_params
        result = (
            a0
            + a1 * np.cos(1 * np.pi * y)
            + a2 * np.cos(2 * np.pi * y)
            + a3 * np.cos(3 * np.pi * y)
            + b1 * np.sin(1 * np.pi * y)
            + b2 * np.sin(2 * np.pi * y)
            + b3 * np.sin(3 * np.pi * y)
        )
        return result

    def param_outcome_model1(self):
        EX = 4
        alpha = 8 + 2 * (
            self.f1_model1(self.grid[:, 0]) + self.g1_model1(self.grid[:, 1])
        )
        alpha = alpha.reshape(-1, 1)
        beta = self.f2_model1(self.grid[:, 0]) + self.g2_model1(self.grid[:, 1])
        beta = beta.reshape(-1, 1)
        gamma_strength = np.sum(alpha + EX * beta) * self.s
        gamma = gamma_strength * alpha / np.sum(alpha)
        theta = gamma_strength * 0.6 * beta / np.sum(beta)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta

    def param_outcome_model2(self):
        self.coord_sum = self.grid[:, 0] + self.grid[:, 1]

    def param_outcome_model3(self):
        pass

    def param_outcome_model4(self):
        self.coord_sum = (self.grid[:, 0] + self.grid[:, 1]).reshape(-1, 1)

    def tau_model1(self):
        self.tau = np.sum(self.gamma + self.theta)

    def tau_model2(self):
        self.tau = 24 * np.sum(np.sin(self.coord_sum * (np.pi / 8) + 1.5 * self.s) - np.sin(self.coord_sum * (np.pi / 8)))

    def tau_model3(self):
        self.tau = 24 * self.R * np.sin(1.5 * self.s)

    def tau_model4(self):
        self.SIN_SCALE = np.max(self.coord_sum)
        self.tau = (
            12
            * np.sum(
                np.sin((self.coord_sum / self.SIN_SCALE) + self.s * 1.5)
                - np.sin((self.coord_sum / self.SIN_SCALE))
            )
        )

    def exchangeable_cor(self):
        R = self.R
        cov_mat = np.ones((R, R)) * self.rho
        np.fill_diagonal(cov_mat, 1.0)
        return cov_mat

    def exponential_decay_cor(self):
        R = self.R
        cov_mat = np.zeros((R, R))
        for i in range(R):
            for j in range(i, R):
                # here, we divide 2 two time as self.grid[i, :] is two times in Yang's paper
                dist = np.linalg.norm(self.grid[i, :] - self.grid[j, :]) / 2 / 2
                cov_value = self.rho**dist
                cov_mat[i, j] = cov_value
                cov_mat[j, i] = cov_value
        return cov_mat

    def d_dependent_cor(self):
        R = self.R
        cov_mat = np.zeros((R, R))
        for i in range(R):
            for j in range(R):
                if i == j:
                    cov_mat[i, j] = 1.0
                elif abs(i - j) <= self.rho * R:
                    cov_mat[i, j] = self.rho - (abs(i - j) / R)
        return cov_mat

    def horizon_fail_cor(self):
        R = self.R
        sub_cov_mat2 = np.zeros((R >> 1, R >> 1))
        sub_cov_mat1 = self.rho * np.ones((R >> 1, R >> 1))
        np.fill_diagonal(sub_cov_mat1, val=1.0)
        cov_mat = np.vstack(
            [
                np.hstack([sub_cov_mat1, sub_cov_mat2]),
                np.hstack([sub_cov_mat2, sub_cov_mat1]),
            ]
        )
        return cov_mat

    def yang_example3_cor(self):
        R = self.R
        np.random.seed(1)
        v1 = 0.25 * np.random.uniform(size=R) + 0.75
        v1[int(np.ceil(R * self.rho)) :] = 0

        cov_mat = np.outer(v1, v1)
        np.fill_diagonal(cov_mat, 1)
        return cov_mat

    def low_rank_cor(self):
        rank = int(self.R * self.rho)
        L = np.random.uniform(low=0, high=1, size=(self.R, rank))
        cov_mat = L @ L.transpose()
        delta = np.real(np.max(np.linalg.eigvals(cov_mat))) / (self.R - 1)
        cov_mat = cov_mat + delta * np.eye(self.R)
        # np.max(np.linalg.eigvals(cov_mat)) / np.min(np.linalg.eigvals(cov_mat))
        cov_mat = cov2cor(cov_mat)
        return cov_mat

    def irregular_d_dependent_cor(self):
        block_size = np.random.poisson(lam=4, size=self.R)
        cov_mat = np.zeros((self.R, self.R))
        for i in range(self.R):
            for j in range(self.R):
                if i == j:
                    cov_mat[i, j] = 2.0
                elif abs(i - j) <= block_size[i]:
                    cov_mat[i, j] = self.rho
        cov_mat = (cov_mat + cov_mat.transpose()) / 2.0
        return cov_mat

    def uniform_cor(self):
        R = self.R
        # cov_mat = np.random.uniform(low=self.rho-0.1, high=self.rho+0.1, size=(R, R))
        cov_mat = np.random.normal(loc=self.rho, scale=0.1, size=(R, R))
        cov_mat = (cov_mat + cov_mat.transpose()) / 2
        np.fill_diagonal(cov_mat, 2.0)
        return cov_mat
    
    def exchangeable_cor_entry_small(self):
        R = self.R
        cov_mat = np.ones((R, R)) * self.rho
        cov_mat += np.random.normal(loc=0, scale=0.1, size=(R, R))
        num_small_entry = int(R / 2)
        row_ind = np.random.choice(R, num_small_entry, replace=True)
        col_ind = np.random.choice(R, num_small_entry, replace=True)
        for i in range(num_small_entry):
            cov_mat[row_ind[i], col_ind[i]] = 0.01
            cov_mat[col_ind[i], row_ind[i]] = 0.01
        
        np.fill_diagonal(cov_mat, 1.0)
        return cov_mat

    def exchangeable_cor_row_small(self):
        R = self.R
        value = 0.01
        cov_mat = np.ones((R, R)) * self.rho
        num_small_entry = int(9 * R / 10)
        row_ind = np.random.choice(R, 1)  # Only select one row
        col_ind = np.random.choice(R, num_small_entry, replace=False)
        cov_mat[row_ind, col_ind] = value
        cov_mat[col_ind, row_ind] = value
        np.fill_diagonal(cov_mat, 1.0)
        return cov_mat
    
    def block_cor(self):
        R = self.R
        m = 6
        block_size = int(R // m)
        cov_mat = np.zeros((R, R))
        for i in range(m):
            start = i * block_size
            end = start + block_size
            cov_mat[start:end, start:end] = self.rho

        np.fill_diagonal(cov_mat, 1.0)
        return cov_mat

    def outcome_model1(self, obs, A, random):
        R = A.shape[0]

        main_effect = (
            self.alpha + np.sum(obs * self.beta, axis=1).reshape(-1, 1) + self.gamma * A
        )

        bar_A = np.zeros((R, 1))
        for key, value in enumerate(self.adj_indices):
            bar_A[key, 0] = np.mean(A[value])
        spillover_effect = self.theta * bar_A

        if random:
            e_noise = np.random.multivariate_normal(
                mean=np.zeros(R), cov=self.cov_mat, size=1
            )
            e_noise = e_noise.reshape(-1, 1)
        else:
            e_noise = np.zeros((R, 1))

        # print('main_effect:', main_effect)
        # print('spillover_effect:', spillover_effect)
        # print('e_noise:', e_noise)

        outcome = main_effect + spillover_effect + e_noise
        return outcome

    def outcome_model2(self, obs, A, random):
        R = A.shape[0]
        bar_A = np.zeros((R, 1))
        bar_O = np.zeros((R, obs.shape[1]))
        for key, value in enumerate(self.adj_indices):
            bar_A[key, 0] = np.mean(A[value])
            bar_O[key, 0] = np.mean(obs[value, :], axis=0)

        spillover_effect = (
            3
            * (obs + bar_O)
            * np.sin((self.coord_sum.reshape(-1, 1) * (np.pi / 8.0)) + self.s * (A + 0.5 * bar_A))
        )

        if random:
            e_noise = np.random.multivariate_normal(mean=np.zeros(R), cov=self.cov_mat, size=1)
            e_noise = e_noise.reshape(-1, 1)
        else:
            e_noise = np.zeros((R, 1))
        outcome = 5 + spillover_effect + 0.5 * e_noise
        return outcome

    def outcome_model3(self, obs, A, random):
        R = A.shape[0]
        bar_A = np.zeros((R, 1))
        bar_O = np.zeros((R, obs.shape[1]))
        for key, value in enumerate(self.adj_indices):
            bar_A[key, 0] = np.mean(A[value])
            bar_O[key, 0] = np.mean(obs[value, :], axis=0)

        spillover_effect = 3 * (obs + bar_O) * np.sin(self.s * (A + 0.5 * bar_A))

        if random:
            e_noise = np.random.multivariate_normal(mean=np.zeros(R), cov=self.cov_mat, size=1,)
            e_noise = e_noise.reshape(-1, 1)
        else:
            e_noise = np.zeros((R, 1))
        outcome = 5 + spillover_effect + 0.5 * e_noise
        return outcome

    def outcome_model4(self, obs, A, random):
        R = A.shape[0]
        bar_A = np.zeros((R, 1))
        for key, value in enumerate(self.adj_indices):
            bar_A[key, 0] = np.mean(A[value])

        spillover_effect = (
            3
            * obs
            * np.sin(
                (self.coord_sum.reshape(-1, 1) / self.SIN_SCALE)
                + self.s * (A + 0.5 * bar_A)
            )
        )

        if random:
            e_noise = np.random.multivariate_normal(mean=np.zeros(R), cov=self.cov_mat, size=1,)
            e_noise = e_noise.reshape(-1, 1)
        else:
            e_noise = np.zeros((R, 1))
        outcome = 5 + spillover_effect + 0.5 * e_noise
        return outcome

    def next_obs_model1(self, obs, A):
        next_obs = np.random.normal(size=obs.shape, loc=4.0, scale=1.0)
        return next_obs

    def next_obs_model2(self, obs, A):
        next_obs = np.random.normal(size=obs.shape, loc=4.0, scale=1.0)
        next_obs = np.clip(next_obs, a_min=3, a_max=5)
        return next_obs

    def next_obs_model3(self, obs, A):
        next_obs = np.random.normal(size=obs.shape, loc=4.0, scale=1.0)
        next_obs = np.clip(next_obs, a_min=3, a_max=5)
        return next_obs

    def __init__(
        self,
        cor_type="example1",
        model_type="static",
        exposure=1,
        pattern="hexagon",
        grid_size=12,
        grid_noise=('uniform', 0.0),
        obs_dim=1,
        signal=0.025,
        rho=0.3,
        env_seed=0,
    ):
        self.dim_obs = obs_dim
        self.exposure = exposure
        self.s = signal
        self.rho = rho
        np.random.seed(env_seed)

        if pattern == "rectangle_hexagon":
            self.spatial_model = self.rectangle_hexagon_model
        if pattern == "fan_hexagon":
            self.spatial_model = self.fan_hexagon_model
        if pattern == "circle_hexagon":
            self.spatial_model = self.circle_hexagon_model
        if pattern == "hexagon":
            self.spatial_model = self.hexagon_model
        elif pattern == "square":
            self.spatial_model = self.square_model
        elif pattern == "triangle":
            self.spatial_model = self.triangle_model
        self.grid, self.adj_indices, self.R = self.spatial_model(grid_size)
        if grid_noise[0] == 'uniform':
            range_value = grid_noise[1]
            noise = np.random.uniform(-range_value, range_value, (self.R, 2))
        elif grid_noise[0] == 'normal':
            cov_value = grid_noise[1]
            noise = np.random.multivariate_normal([0, 0], [[cov_value, 0.0], [0, cov_value]], self.R)
        self.grid = self.grid + noise

        if cor_type == "example1":
            self.cor_model = self.exchangeable_cor
        elif cor_type == "example2":
            self.cor_model = self.exponential_decay_cor
        elif cor_type == "example3":
            self.cor_model = self.d_dependent_cor
        elif cor_type == "yang_example3":
            self.cor_model = self.yang_example3_cor
        elif cor_type == "example4":
            self.cor_model = self.horizon_fail_cor
        elif cor_type == "example5":
            self.cor_model = self.low_rank_cor
        elif cor_type == "example6":
            self.cor_model = self.irregular_d_dependent_cor
        elif cor_type == "example7":
            self.cor_model = self.uniform_cor
        elif cor_type == "example8":
            self.cor_model = self.exchangeable_cor_entry_small
        elif cor_type == "example9":
            self.cor_model = self.exchangeable_cor_row_small
        elif cor_type == "example10":
            self.cor_model = self.block_cor
        self.cov_mat = self.cor_model()

        if model_type == "static":
            self.init_obs_model = self.init_obs_model1
            self.outcome_model = self.outcome_model1
            self.next_obs_model = self.next_obs_model1
            self.param_outcome_model = self.param_outcome_model1
            self.tau_model = self.tau_model1
            self.f1_model1_params = np.random.uniform(0, 1, 7)
            self.f2_model1_params = np.random.uniform(0, 1, 7)
            self.g1_model1_params = np.random.uniform(0, 1, 7)
            self.g2_model1_params = np.random.uniform(0, 1, 7)
        elif model_type == "semi-static":
            self.init_obs_model = self.init_obs_model2
            self.outcome_model = self.outcome_model4
            self.next_obs_model = self.next_obs_model2
            self.param_outcome_model = self.param_outcome_model2
            self.tau_model = self.tau_model4
        elif model_type == "complex-semi-static":
            self.init_obs_model = self.init_obs_model2
            self.outcome_model = self.outcome_model2
            self.next_obs_model = self.next_obs_model2
            self.param_outcome_model = self.param_outcome_model2
            self.tau_model = self.tau_model2
        elif model_type == "homo-semi-static":
            self.init_obs_model = self.init_obs_model3
            self.outcome_model = self.outcome_model3
            self.next_obs_model = self.next_obs_model3
            self.param_outcome_model = self.param_outcome_model3
            self.tau_model = self.tau_model3
        elif model_type == "dynamic":
            pass
        elif model_type == "toy":
            pass
        else:
            pass

        self.param_outcome_model()
        self.tau_model()

    def set_env_seed(self, env_seed):
        np.random.seed(env_seed)

    def sample_data(
        self,
        interior,
        policy=None,
        N=30,
        seed=1,
        burn_in_N=0,
        random=True,
    ):
        """
        Output: data (dict): with R paired elements.
        > Each paired element in data is (l, l_data) where l_data is a dict with three paired elements
        > l_data = {'obs': np.array, 'tre': np.array, 'outcome': np.array, 'interior': bool}
        """
        np.random.seed(seed)
        if burn_in_N > 0:
            burn_in = True
            N += burn_in_N
        else:
            burn_in = False

        init_obs = self.init_obs_model()
        R = init_obs.shape[0]
        random_obs = np.zeros((N + 1, R, self.dim_obs))
        random_action = np.zeros((N, R, 1))
        random_outcome = np.zeros((N, R, 1))

        random_obs[0, :, :] = init_obs
        for i in range(N):
            random_action[i, :, :] = policy(random_obs[i, :, :])
            random_outcome[i, :, :] = self.outcome_model(
                random_obs[i, :, :], random_action[i, :, :], random=random
            )
            random_obs[i + 1, :, :] = self.next_obs_model(
                random_obs[i, :, :], random_action[i, :, :]
            )
            pass

        if burn_in:
            valid_index = range(burn_in_N, N + 1)
            random_obs = random_obs[valid_index, :, :]
            valid_index = range(burn_in_N, N)
            random_action = random_action[valid_index, :, :]
            random_outcome = random_outcome[valid_index, :, :]

        random_obs = random_obs[:(-1), :, :]
        data = dict()
        for i in range(R):
            l_data = dict()
            l_data["obs"] = random_obs[:, i, :]
            l_data["tre"] = random_action[:, i, :]
            l_data["outcome"] = random_outcome[:, i, :]
            l_data["interior"] = interior[i]
            data[i] = l_data
        return data

    def get_adj_matrix(self):
        adj_mat = np.zeros((self.R, self.R))
        for row, nnz_index in enumerate(self.adj_indices):
            for col in nnz_index:
                adj_mat[row, col] = 1

        return adj_mat

    def get_cov_matrix(self):
        return np.copy(self.cov_mat)
