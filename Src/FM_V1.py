import pandas as pd
import numpy as np
import torch

#from Src.ClassModelFM import RankFM

df = pd.read_csv('Data/Raw/transactions_train.csv')

df_1 = pd.read_csv('Data/Preprocessed/AllDataOneTable.csv')

df_test = df_1[['customer_id','article_id']]

all_users = df_test.customer_id.unique()
all_items = df_test.article_id.unique()

np.random.seed(42)
s_users = np.random.choice(all_users, size=10000, replace=False)

s_interactions = df_test[df_test.customer_id.isin(s_users)].copy()
s_interactions.shape

s_items = s_interactions.article_id.unique()
len(s_items)

n_s_users = len(s_users)
n_s_items = len(s_items)

print("sample users:", n_s_users)
print("sample items:", n_s_items)
print("sample interactions:", s_interactions.shape)

s_sparsity = 1 - (s_interactions[['customer_id', 'article_id']].drop_duplicates().shape[0] / (n_s_users * n_s_items))
print("sample interaction data sparsity: {}".format(round(100 * s_sparsity, 2)))

shuffle_index = np.arange(len(s_interactions))
np.random.shuffle(shuffle_index)

s_interactions = s_interactions.iloc[shuffle_index]
s_interactions['random'] = np.random.random(size=len(s_interactions))
s_interactions.head()

test_pct = 0.25
train_mask = s_interactions['random'] <  (1 - test_pct)
valid_mask = s_interactions['random'] >= (1 - test_pct)
          
interactions_total = s_interactions[['customer_id', 'article_id']]
interactions_total = interactions_total.iloc[shuffle_index]

interactions_train = s_interactions[train_mask].groupby(['customer_id', 'article_id']).size().to_frame('orders').reset_index()
interactions_valid = s_interactions[valid_mask].groupby(['customer_id', 'article_id']).size().to_frame('orders').reset_index()

# sample_weight_train = interactions_train['orders']
# sample_weight_valid = interactions_vwlid['orders']
sample_weight_train = np.log2(interactions_train['orders'] + 1)
sample_weight_valid = np.log2(interactions_valid['orders'] + 1)

interactions_train = interactions_train[['customer_id', 'article_id']]
interactions_valid = interactions_valid[['customer_id', 'article_id']]

train_users = np.sort(interactions_train.customer_id.unique())
valid_users = np.sort(interactions_valid.customer_id.unique())
cold_start_users = set(valid_users) - set(train_users)

train_items = np.sort(interactions_train.article_id.unique())
valid_items = np.sort(interactions_valid.article_id.unique())
cold_start_items = set(valid_items) - set(train_items)

#item_features_train = item_features[item_features.article_id.isin(train_items)]
#item_features_valid = item_features[item_features.produarticle_idct_id.isin(valid_items)]

print("total shape: {}".format(interactions_total.shape))
print("train shape: {}".format(interactions_train.shape))
print("valid shape: {}".format(interactions_valid.shape))

print("\ntrain weights shape: {}".format(sample_weight_train.shape))
print("valid weights shape: {}".format(sample_weight_valid.shape))

print("\ntrain users: {}".format(len(train_users)))
print("valid users: {}".format(len(valid_users)))
print("cold-start users: {}".format(len(cold_start_users)))

print("\ntrain items: {}".format(len(train_items)))
print("valid items: {}".format(len(valid_items)))
print("number of cold-start items: {}".format(len(cold_start_items)))

#print("\ntrain item features: {}".format(item_features_train.shape))
#print("valid item features: {}".format(item_features_valid.shape))

model = RankFM(factors=50, loss='warp', max_samples=50, alpha=0.01, learning_rate=0.1, learning_schedule='invscaling')

model.fit(interactions_train, sample_weight=sample_weight_train, epochs=30, verbose=True)

import torch
import numpy as np
import pandas as pd

class RankFM():
    def __init__(self, factors=10, loss='bpr', max_samples=10, alpha=0.01, beta=0.1, sigma=0.1, learning_rate=0.1, learning_schedule='constant', learning_exponent=0.25):
        self.factors = factors
        self.loss = loss
        self.max_samples = max_samples
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.learning_schedule = learning_schedule
        self.learning_exponent = learning_exponent
    def _init_all(self, interactions, user_features=None, item_features=None, sample_weight=None):
        """index the interaction data and user/item features and initialize model weights
        :param interactions: dataframe of observed user/item interactions: [user_id, item_id]
        :param user_features: dataframe of user metadata features: [user_id, uf_1, ..., uf_n]
        :param item_features: dataframe of item metadata features: [item_id, if_1, ..., if_n]
        :param sample_weight: vector of importance weights for each observed interaction
        :return: None
        """

        assert isinstance(interactions, (np.ndarray, pd.DataFrame)), "[interactions] must be np.ndarray or pd.dataframe"
        assert interactions.shape[1] == 2, "[interactions] should be: [user_id, item_id]"

        # save unique arrays of users/items in terms of original identifiers
        interactions_df = pd.DataFrame(get_data(interactions), columns=['user_id', 'item_id'])
        self.user_id = pd.Series(np.sort(np.unique(interactions_df['user_id'])))
        self.item_id = pd.Series(np.sort(np.unique(interactions_df['item_id'])))

        # create zero-based index to identifier mappings
        self.index_to_user = self.user_id
        self.index_to_item = self.item_id

        # create reverse mappings from identifiers to zero-based index positions
        self.user_to_index = pd.Series(data=self.index_to_user.index, index=self.index_to_user.values)
        self.item_to_index = pd.Series(data=self.index_to_item.index, index=self.index_to_item.values)

        # store unique values of user/item indexes and observed interactions for each user
        self.user_idx = np.arange(len(self.user_id), dtype=np.int32)
        self.item_idx = np.arange(len(self.item_id), dtype=np.int32)

        # map the interactions to internal index positions
        self._init_interactions(interactions, sample_weight)

        # map the user/item features to internal index positions
        self._init_features(user_features, item_features)

        # initialize the model weights after the user/item/feature dimensions have been established
        self._init_weights(user_features, item_features)
    def _init_interactions(self, interactions, sample_weight):
        """map new interaction data to existing internal user/item indexes
        :param interactions: dataframe of observed user/item interactions: [user_id, item_id]
        :param sample_weight: vector of importance weights for each observed interaction
        :return: None
        """

        assert isinstance(interactions, (np.ndarray, pd.DataFrame)), "[interactions] must be np.ndarray or pd.dataframe"
        assert interactions.shape[1] == 2, "[interactions] should be: [customer_id, item_id]"

        # map the raw user/item identifiers to internal zero-based index positions
        # NOTE: any user/item pairs not found in the existing indexes will be dropped
        self.interactions = interactions
        self.interactions['customer_id'] = self.interactions['customer_id'].map(self.user_to_index).astype(np.int32)
        self.interactions['article_id'] = self.interactions['article_id'].map(self.item_to_index).astype(np.int32)
        self.interactions = self.interactions.rename({'customer_id': 'user_idx', 'article_id': 'item_idx'}, axis=1).dropna()

        # store the sample weights internally or generate a vector of ones if not given
        if sample_weight is not None:
            assert isinstance(sample_weight, (np.ndarray, pd.Series)), "[sample_weight] must be np.ndarray or pd.series"
            assert sample_weight.ndim == 1, "[sample_weight] must a vector (ndim=1)"
            assert len(sample_weight) == len(interactions), "[sample_weight] must have the same length as [interactions]"
            self.sample_weight = np.ascontiguousarray(sample_weight)
        else:
            self.sample_weight = np.ones(len(self.interactions), dtype=np.float32)

        # create a dictionary containing the set of observed items for each user
        # NOTE: if the model has been previously fit extend rather than replace the itemset for each user

        if self.is_fit:
            new_user_items = self.interactions.groupby('user_idx')['item_idx'].apply(set).to_dict()
            self.user_items = {user: np.sort(np.array(list(set(self.user_items[user]) | set(new_user_items[user])), dtype=np.int32)) for user in self.user_items.keys()}
        else:
            self.user_items = self.interactions.sort_values(['user_idx', 'item_idx']).groupby('user_idx')['item_idx'].apply(np.array, dtype=np.int32).to_dict()

        # format the interactions data as a c-contiguous integer array for cython use
        #self.interactions = np.ascontiguousarray(self.interactions, dtype=np.int32)

    def _init_weights(self, user_features=None, item_features=None):
        """initialize model weights given user/item and user-feature/item-feature indexes/shapes
        :param user_features: dataframe of user metadata features: [user_id, uf_1, ... , uf_n]
        :param item_features: dataframe of item metadata features: [item_id, if_1, ... , if_n]
        :return: None
        """

        # initialize scalar weights as ndarrays of zeros
        self.w_i = np.zeros(len(self.item_idx)).astype(np.float32)
        self.w_if = np.zeros(self.x_if.shape[1]).astype(np.float32)

        # initialize latent factors by drawing random samples from a normal distribution
        self.v_u = np.random.normal(loc=0, scale=self.sigma, size=(len(self.user_idx), self.factors)).astype(np.float32)
        self.v_i = np.random.normal(loc=0, scale=self.sigma, size=(len(self.item_idx), self.factors)).astype(np.float32)

        # randomly initialize user feature factors if user features were supplied
        # NOTE: set all user feature factor weights to zero to prevent random scoring influence otherwise
        if user_features is not None:
            scale = (self.alpha / self.beta) * self.sigma
            self.v_uf = np.random.normal(loc=0, scale=scale, size=[self.x_uf.shape[1], self.factors]).astype(np.float32)
        else:
            self.v_uf = np.zeros([self.x_uf.shape[1], self.factors], dtype=np.float32)

        # randomly initialize item feature factors if item features were supplied
        # NOTE: set all item feature factor weights to zero to prevent random scoring influence otherwise
        if item_features is not None:
            scale = (self.alpha / self.beta) * self.sigma
            self.v_if = np.random.normal(loc=0, scale=scale, size=[self.x_if.shape[1], self.factors]).astype(np.float32)
        else:
            self.v_if = np.zeros([self.x_if.shape[1], self.factors], dtype=np.float32)
    def _reset_state(self):
        """initialize or reset internal model state"""

        # [ID, IDX] arrays
        self.customer_id = None
        self.article_id = None
        self.user_idx = None
        self.item_idx = None

        # [ID <-> IDX] mappings
        self.index_to_user = None
        self.index_to_item = None
        self.user_to_index = None
        self.item_to_index = None

        # user/item interactions and importance weights
        self.interactions = None
        self.sample_weight = None

        # set of observed items for each user
        self.user_items = None

        # [user, item] features
        self.x_uf = None
        self.x_if = None

        # [item, item-feature] scalar weights
        self.w_i = None
        self.w_if = None

        # [user, item, user-feature, item-feature] latent factors
        self.v_u = None
        self.v_i = None
        self.v_uf = None
        self.v_if = None

        # internal model state indicator
        self.is_fit = True

    def fit(self, interactions, user_features=None, item_features=None, sample_weight=None, epochs=1, verbose=False):
        """clear previous model state and learn new model weights using the input data
        :param interactions: dataframe of observed user/item interactions: [user_id, item_id]
        :param user_features: dataframe of user metadata features: [user_id, uf_1, ... , uf_n]
        :param item_features: dataframe of item metadata features: [item_id, if_1, ... , if_n]
        :param sample_weight: vector of importance weights for each observed interaction
        :param epochs: number of training epochs (full passes through observed interactions)
        :param verbose: whether to print epoch number and log-likelihood during training
        :return: self
        """

        self._reset_state()
        self.fit_partial(interactions, user_features, item_features, sample_weight, epochs, verbose)
        return self

    def fit_partial(self, interactions, user_features=None, item_features=None, sample_weight=None, epochs=1, verbose=False):
            """learn or update model weights using the input data and resuming from the current model state
            :param interactions: dataframe of observed user/item interactions: [user_id, item_id]
            :param user_features: dataframe of user metadata features: [user_id, uf_1, ... , uf_n]
            :param item_features: dataframe of item metadata features: [item_id, if_1, ... , if_n]
            :param sample_weight: vector of importance weights for each observed interaction
            :param epochs: number of training epochs (full passes through observed interactions)
            :param verbose: whether to print epoch number and log-likelihood during training
            :return: self
            """

            assert isinstance(epochs, int) and epochs >= 1, "[epochs] must be a positive integer"
            assert isinstance(verbose, bool), "[verbose] must be a boolean value"

            if self.is_fit:
                self._init_interactions(interactions, sample_weight)
                #self._init_features(user_features, item_features)
            else:
                self._init_all(interactions, user_features, item_features, sample_weight)

            # determine the number of negative samples to draw depending on the loss function
            # NOTE: if [loss == 'bpr'] -> [max_samples == 1] and [multiplier ~= 1] for all updates
            # NOTE: the [multiplier] is scaled by total number of items so it's always [0, 1]

            if self.loss == 'bpr':
                max_samples = 1
            elif self.loss == 'warp':
                max_samples = self.max_samples
            else:
                raise ValueError('[loss] function not recognized')

            # NOTE: the cython private _fit() method updates the model weights in-place via typed memoryviews
            # NOTE: therefore there's nothing returned explicitly by either method
            """
            _fit(
                self.interactions,
                self.sample_weight,
                self.user_items,
                self.x_uf,
                self.x_if,
                self.w_i,
                self.w_if,
                self.v_u,
                self.v_i,
                self.v_uf,
                self.v_if,
                self.alpha,
                self.beta,
                self.learning_rate,
                self.learning_schedule,
                self.learning_exponent,
                max_samples,
                epochs,
                verbose
            )"""

            self.is_fit = True
            return self
