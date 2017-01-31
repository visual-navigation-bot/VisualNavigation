import numpy as np
import matplotlib.pyplot as plt
import pygame

class TwoD_Gaussian_Process:
    """
    This is the gaussian process function that initialized by two kernel functions
    and can train by 2D points and output the predictions of future 2D points
    """
    def __init__(self, x_kernel, y_kernel):
        self.x_kernel = x_kernel
        self.y_kernel = y_kernel
        self.xgp = Gaussian_Process(x_kernel)
        self.ygp = Gaussian_Process(y_kernel)
        self.first_test_time = None
        # the test set, in vertical vector form
        self.test_time = None

        self.goal_time = None
        self.goal_position = None
        self.debug_mode = []

    def set_debug_mode(self, debug_mode = []):
        self.debug_mode = debug_mode
        return

    def sample(self, path_history, goal_time, goal_position, sample_count):
        """
        This is the only function other than intialization can be called by functions outside this class
        It will refresh, check the correctness of history and goal
        and then add histroy, add goal and finaly sampling
        Input:
            path_history: Path_History; the path history for training
            goal_time: int; goal time as time index
            goal_position: np.2darray; the goal position
            sample_count: int; the number of samples
        Return:
            future_time: np.1darray; time array
            sample_future_path: np.3darray; axis 0 is time, axis 1 is sample, axis 2 is x and y
        Note:
            goal_time should always be greater than last path history time
            path history should never be empty, so do goal time and goal_position
        """
        if 2 in self.debug_mode:
            assert len(path_history.past_time) != 0, "No Past Path in 2dGP"
            last_past_time = path_history.past_time[-1]
            assert goal_time > last_past_time, "goal time less than or equal to final past time"
        self._refresh()
        self._add_history(path_history)
        self._add_goal(goal_time, goal_position)
        future_time, sample_future_path = self._sample(sample_count)
        return future_time, sample_future_path

    def _refresh(self):
        # refresh all the training data, only preserve kernel model
        self.xgp.refresh()
        self.ygp.refresh()
        self.first_test_time = None
        self.test_time = None
        self.goal_time = None
        self.goal_position = None
        return

    def _add_history(self, path_history):
        """
        add a path history for training
        Input:
            path_history: Path_History; the path history for training
        """

        train_time = path_history.past_time.reshape(-1, 1)
        train_x = path_history.past_path[:, 0].reshape(-1, 1)
        train_y = path_history.past_path[:, 1].reshape(-1, 1)
        self.xgp.add_train_set(train_time, train_x)
        self.ygp.add_train_set(train_time, train_y)
        self.first_test_time = path_history.past_time[-1]
        return
        
    def _add_goal(self, goal_time, goal_position):
        """
        add the goal for training purpose
        Input:
            goal_time: int; goal time as time index
            goal_position: np.2darray; the goal position
        """
        self.goal_time = goal_time
        self.goal_position = goal_position.copy()

        test_time = np.arange(self.first_test_time + 1, goal_time).reshape(-1, 1)
        self.test_time = test_time
        _goal_time = np.array([[goal_time]])
        _goal_x = np.array([[goal_position[0]]])
        _goal_y = np.array([[goal_position[1]]])
        self.xgp.add_train_set(_goal_time, _goal_x)
        self.ygp.add_train_set(_goal_time, _goal_y)

        self.xgp.add_test_set(test_time)
        self.ygp.add_test_set(test_time)

        return

    def _sample(self, sample_count):
        """
        sampling the future path to the goal time (include goal time)
        Input:
            sample_count: int; the number of samples
        Return:
            future_time: np.1darray; time array
            sample_future_path: np.3darray; axis 0 is time, axis 1 is sample, axis 2 is x and y
        """
        if len(self.test_time) == 0:
            future_time = np.array([self.goal_time])
            sample_future_path = np.tile(self.goal_position, (1, sample_count, 1))
            return (future_time, sample_future_path)

        test_x_sample = self.xgp.sample(sample_count)
        test_y_sample = self.ygp.sample(sample_count)
        test_path_sample = np.dstack((test_x_sample, test_y_sample))
        sample_future_path = np.vstack((test_path_sample, np.tile(self.goal_position, (1,sample_count,1))))
        future_time = np.hstack((self.test_time.reshape(-1), self.goal_time))
        return (future_time.copy(), sample_future_path.copy())

class Gaussian_Process:
    """
    A very simple one dimensional gaussian process
    There will never include any repete training set and test set is completely different from training set
    Should not test by empty array
    When sampling, there will always have training set and testing set
    """
    def __init__(self, kernel):
        self.kernel = kernel
        # x, y, test should all be vertical vectors
        self.x = None
        self.y = None
        self.test = None
        self.debug_mode = []

    def set_debug_mode(self, debug_mode = []):
        self.debug_mode = debug_mode
        return


    def add_train_set(self, x, y):
        """
        add the training set (or append the training set)
        Input:
            x: np.2darray; vertical vector
            y: np.2darray; vertical vector
        """
        if 1 in self.debug_mode:
            if self.x is not None:
                overlap = False
                for xtrain in x:
                    if xtrain in self.x:
                        overlap = True
                assert (not overlap), "train set overlapping"
            assert len(x) == len(y), "length of x and y are different"


        if self.x is None:
            self.x = x
        else:
            self.x = np.vstack((self.x, x))

        if self.y is None:
            self.y = y
        else:
            self.y = np.vstack((self.y, y))
        return


    def add_test_set(self, test):
        """
        add the testing set (or append the testing set)
        Input:
            xtest: np.2darray; vertical vector
        """
        if self.test == None:
            self.test = test
        else:
            self.test = np.vstack((self.test, test))

    def refresh(self):
        """
        refresh all training set and testing set
        """
        self.x = None
        self.y = None
        self.test = None
        return

    def sample(self, sample_count):
        """
        sample the testing set by sample_count number of samples
        Input:
            sample_count: int; the amount of samples that we are generating
        Return:
            y_samples: np.2darray; axis = 0 is corresponding to test index
                                   axis = 1 is corresponding to sample index
        """
        if 1 in self.debug_mode:
            assert (self.test is not None), "None Type Test Set Error"
            assert (len(self.test) != 0), "Empty Test Set Error"
            assert (self.x is not None), "None Type Train Set Error"
            assert(len(self.x) != 0), "Empty Train Set Error"

        K = self.kernel(self.x, self.x, "train")
        L = np.linalg.cholesky(K + 1e-9*np.eye(K.shape[0]))
        
        # the kernele calculated below won't add noise
        Lk = np.linalg.solve(L, self.kernel(self.x, self.test, "crossed") )
        mu = np.dot(Lk.T, np.linalg.solve(L, self.y)).reshape(-1)

        # compute the variance at test points
        K_ = self.kernel(self.test, self.test, "test")
        tuned_K_ = K_ + 1e-9*np.eye(K_.shape[0]) - np.dot(Lk.T, Lk)
        y_samples = np.random.multivariate_normal(mu, tuned_K_, sample_count).swapaxes(0,1)
        return y_samples



class Path_History:
    # for any pedestrians, its path history should always contains something
    # add the history whenever started a new path history or moved yourself
    # no functions can change existing history
    def __init__(self, init_time, init_position):
        """
        The path history of a certain pedestrian
        Input:
            init_time: can be a complete past time index or a start time index
            init_position: can be a complete past path or a start position
        past path is a np.2darray: axis 0 is the position, axis 1 is the x or y
        past time is a np.1darray: axis 0 is the time index of each position
        """
        if type(init_time) == int:
            # initializing a totally new path history
            self.past_time = np.array([init_time])
            self.past_path = np.expand_dims(init_position.copy(), axis = 0)
        else:
            # initializing a sliced path history
            self.past_time = init_time
            self.past_path = init_position

    def add_history(self, time, position):
        """
        Add in a new history set
        Input:
            time: int; the time index of this new position
            position: np.1darray; the position of this new history set
        """
        self.past_path = np.vstack((self.past_path, position.copy()))
        self.past_time = np.hstack((self.past_time, time))
        return

    def sliced_history(self, init_time):
        """
        Given a time index, get the sliced history with time 
        greater equal to this init time index
        Input:
            init_time: int; we want to get history after this time index
        """
        if init_time <= self.past_time[0]:
            return self
        else:
            start_index = np.argwhere(self.past_time == init_time)[0][0]
            sliced_time = self.past_time[start_index:]
            sliced_position = self.past_path[start_index:, :]
            return Path_History(sliced_time, sliced_position)
        return

    def plot(self, mode):
        past_x = self.past_path[:, 0]
        past_y = self.past_path[:, 1]
        plt.plot(past_x, past_y, mode)
        return

    def pprint(self):
        print "Path History Detail: "
        print "    past path: ", self.past_path
        print "    past time: ", self.past_time



######################################################################################
# POPULAR KERNELS
######################################################################################

def square_exponential_kernel(l, sigma = 1.):
    """
    Square exponential kernel. One of the most well-known stationary covariance function
    Input:
        l: float; length scale of distance between input vectors
        sigma: float; the sigma of the square exponential kernel 
    Return:
        kernel: function; the kernel function that gives covariance of two vectors
            vector1: np.2darray; a vertical vector
            vector2: np.2darray; a horizontal vector
            mode: string; either "train" or "test"
    """
    def kernel(vector1, vector2, mode):
        r = np.sqrt(np.abs(vector1**2 + vector2.T**2 - 2 * np.outer(vector1, vector2)))
        return sigma * np.exp(- .5 * r / l)
    return kernel

def matern_kernel(l, sigma = 1.):
    """
	Matern kernel. See "Gaussian Processes for Machine Learning", by
	Rasmussen and Williams, Chapter 4.
	Specifically, this kernel is Matern class with v=5/2, multiplied by an
	optional signal variance sigma2.
    Input:
        l: float; length scale of distance between input vectors
        sigma: float; the sigma of the matern kernel 
    Return:
        kernel: function; the kernel function that gives covariance of two vectors
            vector1: np.2darray; a vertical vector
            vector2: np.2darray; a vertical vector
            mode: string; either "train" or "test"
	"""
    # r = dist(v1, v2)
    # kernel = (1 + sqrt(5)*r/l + 5*r**2/(3*l**2)) * exp(-sqrt(5)*r/l)
    def kernel(vector1, vector2, mode):
        r = np.sqrt(np.abs(vector1**2 + vector2.T**2 - 2 * np.outer(vector1, vector2)))
        return sigma * (1 + np.sqrt(5)*r/l + 5*r**2 / (3*l**2)) * np.exp(-np.sqrt(5) * r / l)
    return kernel

def linear_kernel(sigma = 1.):
    """
    the linear kernel function factory
    Input:
        sigma: float; the sigma of the linear kernel
    Return:
        kernel: function; the kernel function that gives covariance of two vectors
            vector1: 2darray; a vertical vector
            vector2: 2darray; a vertical vector
            mode: string; either "train" or "test"
    """
    def kernel(vector1, vector2, mode):
        return (1 + np.outer(vector1, vector2)) * sigma
    return kernel



def noise_kernel(sigma = 0.):
    """
    the noise kernel function factory
    Note: Here, we only put noise on the known training points instead of correlations
    Therefore, if the input vector1 and vector2 are not the same, this kernel will be 0
    In testing case, we add noise to the whole k instead of just trace
    Input:
        sigma: float; the sigma of the noise kernel
    Return:
        kernel: function; the kernel function that gives covariance of two vectors
            vector1: 2darray; a vertical vector
            vector2: 2darray; a vertical vector
            mode: string; either "train" or "test"
    """
    def kernel(vector1, vector2, mode):
        if mode == "train":
            assert(len(vector1) == len(vector2)), "vector length not aligned in noise kernel"
            return sigma * np.eye(len(vector1))
        elif mode == "test":
            return sigma
        else:
            return 0.
    return kernel


def summed_kernel(*args):
    """
    summed all the input kernels
    Input:
        k1: function; the kernel function that will be added
        k2: function; the kernel function that will be added
        ...
    Return:
        k: function: the kernel function that will be returned
    """
    def kernel(vector1, vector2, mode):
        K = np.zeros((len(vector1), len(vector2)))
        for k in args:
            K += k(vector1, vector2, mode)
        return K
    return kernel

######################################################################################
# INTERACTIVE WEIGHT FUNCTIONS
######################################################################################

            
def weight_calculation(ped1_future_time, ped1_sample_future_path,
        ped2_future_time, ped2_sample_future_path, 
        interaction_parameters, debug_mode = []):
    """
    calculate the weight between paths
    Input:
        ped1_future_time: np.1darray; first pedestrian future time list
        ped1_sample_future_path: np.3darray; axis 0 is time, axis 1 is sample, axis 2 is x and y
        ped2_future_time: np.1darray; second pedestrian future time list
        ped2_sample_future_path: np.3darray; axis 0 is time, axis 1 is sample, axis 2 is x and y
        interaction_parameters: dict
            'alpha': alpha in the paper interaction section
            'h': h in the paper interaction section
        debug_mode: list of int; debug mode 3 belongs to this
    Output:
        weight: np.1darray; the weight of this path pair (accross all samples)
    """
    alpha = interaction_parameters['alpha']
    h = interaction_parameters['h']

    if 3 in debug_mode:
        assert ped1_future_time[0] == ped2_future_time[0], "two pedestrian not start from same time"
    minimal_time = min(len(ped1_future_time), len(ped2_future_time))
    ped1_sample_sliced_path = ped1_sample_future_path[0:minimal_time,:,:]
    ped2_sample_sliced_path = ped2_sample_future_path[0:minimal_time,:,:]

    path_dist_vector = ped2_sample_sliced_path - ped1_sample_sliced_path
    path_square_dist = np.sum(path_dist_vector**2, axis = 2)
    rbf = 1 - alpha * np.exp(- path_square_dist / (2*h**2))
    weight = np.prod(rbf, axis = 0)
    return weight

def weighted_next_position(sample_path, sample_weight, debug_mode = []):
    """
    find next position from the sampled paths
    Input:
        sample_path: a list of Path_History; the sampled path
        sample_weight: a list of float; the sampled weight
        debug_mode: list of int; debug mode 3 belongs to this
    Output:
        next_position: np.1darray; the next weighted position
    """
    if 3 in debug_mode:
        assert sample_path.shape[1] == len(sample_weight), "sample count not matched"
    sample_next_position = sample_path[0,:,:]
    weighted_next_position = np.tile(sample_weight, (2,1)).T * sample_next_position
    next_position = np.sum(weighted_next_position, axis = 0) / np.sum(sample_weight)
    return next_position

