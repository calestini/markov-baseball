import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from operator import itemgetter
import concurrent.futures
import datetime

RUNS=[
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [3,2,2,2,1,1,1,0,2,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [3,2,2,2,1,1,1,0,2,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [3,2,2,2,1,1,1,0,2,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [4,3,3,3,2,2,2,1,3,2,2,2,1,1,1,0,2,1,1,1,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,2,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,2,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,2,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,3,2,2,2,1,1,1,0,2,1,1,1,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,3,2,2,2,1,1,1,0,2,1,1,1,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,3,2,2,2,1,1,1,0,2,1,1,1,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,4,3,3,3,2,2,2,1,3,2,2,2,1,1,1,0,0,1,2,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,1,1,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,1,1,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,1,1,1,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,2,2,2,1,1,1,0,0,1,2,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,2,2,2,1,1,1,0,0,1,2,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,2,2,2,1,1,1,0,0,1,2,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,3,3,3,2,2,2,1,0,1,2,3]
#    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ]

OUTS=[
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,2,2,2,2,0,0,0,0,3,0,0,0],
    [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,2,2,2,2,0,0,0,0,3,0,0,0],
    [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,2,2,2,2,0,0,0,0,3,0,0,0],
    [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,0,3,3,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,2,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,2,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,2,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,2,2,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,2,2,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,2,2,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1]
    #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    #[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ]

PRIOR = [
    [0.0238,0.2554,0.0466,0.0075,0.0,0.0,0.0,0.0,0.6668,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    [0.0211,0.0003,0.0124,0.0073,0.1951,0.0491,0.0328,0.0,0.0006,0.4092,0.1544,0.0028,0.0,0.0,0.0,0.0,0.1148,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    [0.0171,0.0554,0.0437,0.0076,0.0993,0.0954,0.0054,0.0,0.0035,0.018,0.3803,0.2664,0.0,0.0,0.0,0.0,0.008,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    [0.0178,0.173,0.047,0.0104,0.0,0.1127,0.0002,0.0,0.206,0.0061,0.0015,0.4197,0.0,0.0,0.0,0.0,0.0057,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    [0.021,0.0004,0.0122,0.0074,0.0465,0.0322,0.0338,0.1603,0.0006,0.0027,0.0033,0.0016,0.3215,0.0952,0.1583,0.0,0.0003,0.0108,0.012,0.078,0.0,0.0,0.0,0.0,0.0018,0.0,0.0,0.0],
    [0.022,0.0004,0.0121,0.0077,0.1387,0.0511,0.039,0.0801,0.001,0.182,0.0531,0.003,0.0287,0.2304,0.0417,0.0,0.086,0.0055,0.0034,0.0134,0.0,0.0,0.0,0.0,0.0005,0.0,0.0,0.0],
    [0.0182,0.0603,0.0485,0.0096,0.0072,0.0803,0.0077,0.1515,0.0047,0.0048,0.0938,0.1358,0.0058,0.0105,0.35,0.0,0.0029,0.0003,0.0032,0.0044,0.0,0.0,0.0,0.0,0.0003,0.0,0.0,0.0],
    [0.0236,0.0001,0.0123,0.008,0.0565,0.0357,0.0409,0.1634,0.0008,0.0036,0.0039,0.003,0.0783,0.0964,0.0464,0.3144,0.0004,0.0025,0.0029,0.0675,0.0065,0.0037,0.0283,0.0,0.001,0.0002,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0217,0.2532,0.0442,0.007,0.0,0.0,0.0,0.0,0.6739,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0235,0.0004,0.0147,0.0082,0.203,0.0551,0.0326,0.0,0.0008,0.4362,0.0913,0.0029,0.0,0.0,0.0,0.0,0.1312,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0193,0.0707,0.0503,0.0088,0.1471,0.0612,0.0041,0.0,0.0049,0.014,0.4407,0.1681,0.0,0.0,0.0,0.0,0.0109,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0175,0.1824,0.0475,0.0094,0.0,0.1378,0.0005,0.0,0.2136,0.0267,0.0074,0.3434,0.0,0.0,0.0,0.0,0.0138,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0232,0.0004,0.0154,0.0088,0.0603,0.0396,0.0354,0.1416,0.0008,0.0047,0.0048,0.002,0.3579,0.0968,0.0734,0.0,0.1344,0.0002,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0208,0.0005,0.0133,0.0086,0.1405,0.0498,0.0356,0.0812,0.0008,0.1847,0.0687,0.0041,0.0248,0.2085,0.0228,0.0,0.1306,0.0048,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0131,0.0619,0.0434,0.0079,0.0056,0.0637,0.0063,0.273,0.005,0.0056,0.0882,0.1099,0.0085,0.0248,0.2674,0.0,0.0132,0.0025,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0222,0.0002,0.0126,0.0095,0.0541,0.0362,0.0393,0.1502,0.0009,0.0053,0.0053,0.0027,0.0911,0.0921,0.0489,0.2877,0.1368,0.0048,0.0002,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0223,0.2574,0.0438,0.0063,0.0,0.0,0.0,0.0,0.6703,0.0,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0239,0.001,0.0209,0.009,0.1944,0.0529,0.0216,0.0,0.6753,0.0009,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0183,0.0923,0.0534,0.0079,0.1711,0.0305,0.0005,0.0,0.6208,0.0051,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0181,0.1503,0.0411,0.0073,0.0,0.1556,0.0,0.0,0.6256,0.002,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0209,0.001,0.0226,0.0098,0.0609,0.0445,0.0254,0.1204,0.6863,0.0073,0.0009,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0206,0.001,0.0205,0.01,0.1133,0.0497,0.0233,0.0931,0.6614,0.006,0.0011,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0159,0.0824,0.0472,0.0072,0.0008,0.0311,0.0007,0.2222,0.5821,0.0054,0.0051,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0214,0.0011,0.0218,0.0106,0.0544,0.046,0.0274,0.1205,0.683,0.0054,0.0073,0.001]
]

#initializematrices:
###Initializematrices
#T=npzeros([28,28])

class markov():
    """
    """
    def __init__(self):
        self.data = None
        self.initalize_matrices()
        self.batter_list = ['goldp001','donaj001','cruzn002','vottj001','davic003']
        self.batting_line = [
                ['goldp001', 'Paul GoldSchmidt'],
                ['donaj001', 'Josh Donaldson'],
                ['cruzn002', 'Nelson Cruz'],
                ['cabrm001', 'Migruel Cabrera'],
                ['vottj001', 'Joey Votto'],
                ['mccua001', 'Andrew McCutchen'],
                ['machm001', 'Manny Machado'],
                ['heywj001', 'Jason Heyward'],
                ['davic003', 'Chris Davis']
        ]

    #T is a block matrix of A,B,C,D,E,F,0s,1s
    def initalize_matrices(self):
        """
        """
        A = np.zeros([8,8])
        B = np.zeros([8,8])
        C = np.zeros([8,8])

        D = np.zeros([8,4])
        E = np.zeros([8,4])
        F = np.zeros([8,4])

        P0_1 = np.zeros([8,8])
        P0_4 = np.zeros([4,8])
        P1 = np.repeat([[1,0,0,0]],4,axis=0)

        T = np.block([
            [ A  , B ,  C  , D ],
            [P0_1, A ,  B  , E ],
            [P0_1,P0_1, A  , F ],
            [P0_4,P0_4,P0_4 ,P1]
        ])
        outs = np.array(OUTS)
        runs = np.array(RUNS)
        prior = np.array(PRIOR)

        #self.T = T

        self.runs = runs
        self.outs = outs
        self.prior = prior

        #return True


    def get_data(self, path_str = '../markov.csv'):
        if self.data is None:
            self.data = pd.read_csv(path_str)
            return True
        return False


    def pad(self, array, shape=[24,28]):
        rows = shape[0] - array.shape[0]
        cols = shape[1] - array.shape[1]

        padded = np.pad(array,  ((rows,0),(0,cols)), constant_values = 0, mode='constant')
        return padded


    def transition(self, player_id, precision = 2):
        """Transition matrix for the player
        """
        if self.data is None:
            self.get_data()

        pre_post_cnt = self.data[self.data['player_id']==player_id]\
            .groupby(['pre_state','post_state']).count().reset_index()[['pre_state','post_state','play_runs']]\
            .pivot('pre_state', 'post_state', 'play_runs').fillna(0).values

        all_pre_cnt = self.data[self.data['player_id']==player_id].groupby(['pre_state']).count()[['post_state']].values
        Tp = (pre_post_cnt / all_pre_cnt)
        Tp = self.pad(Tp, [24,28]) #correct shape for multiplication and broadcasting

        return np.round(Tp,precision)


    def er(self, Tp=None):
        """Function to calculate expected run
        """
        Er = np.sum(self.runs*Tp , axis=1).reshape([3,8])
        print (Er)
        return Er

    def eo(self, Tp=None):
        """Function to calculate expected run
        """
        Eo = np.sum(self.outs*Tp , axis=1).reshape([3,8])
        print (Eo)
        return Eo


    def plot(self, array, title):
        plt.figure(figsize=(16,8))
        plt.title(title)
        sns.heatmap(array, annot=True)
        plt.show()


    def plot_T(self, Tp=None, player_id=None):
        """Plot transition matrix
        """
        if player_id:
            Tp = self.transition(player_id=player_id)
            self.plot(Tp, title=player_id)
        else: print ('No player id or dataset passed.')


    def plot_er(self, player_id=None):
        """Plot expected run
        """
        if player_id:
            Tp = self.transition(player_id=player_id)
            Er = self.er(Tp)
            self.plot(Er, title=player_id)
        else: print ('No player id or dataset passed.')


    def plot_eo(self, player_id=None):
        """Plot expected run
        """
        if player_id:
            Tp = self.transition(player_id=player_id)
            Eo = self.eo(Tp)
            self.plot(Eo, title=player_id)
        else: print ('No player id or dataset passed.')


    def batting_line_T(self, batter_list):
        bT = []
        for batter in batter_list:
            bT.append(self.transition(player_id = batter, precision = 10))
        return bT


    def move_state(self, pre_state, Tp):
        post_state = None
        random_prob = np.random.uniform()
        Tp_cum = np.cumsum(Tp, axis=1)
        transition = Tp_cum[pre_state]
        while post_state is None:
            try:
                post_state = int(np.min(np.where(transition >= random_prob)))
            except:
                random_prob = np.random.uniform()

        return post_state

    def play_game(self, T=None, innings=9, N =1000):
        '''Game simulations for one T matrix
        '''
        if not T:
            T = self.prior
        tot_runs = []
        for n in range(N):
            runs = 0

            for i in range(0, innings, 1): #full game
                pre_state = 0

                while pre_state < 24:

                    post_state = self.move_state(pre_state, T)

                    runs += self.runs[pre_state, post_state]
                    pre_state = post_state

            tot_runs.append(runs)
        return np.mean(tot_runs)

    def simulate_games(self, batter_list=None, N = 10000, innings = 9):
        """it runs random selection of states for innings
        """
        if batter_list is None:
            batter_list = self.batter_list

        batting_line_T = self.batting_line_T(self.batter_list)

        tot_runs = []
        for n in range(N):

            runs = 0
            batting = 0

            for i in range(0, innings, 1): #full game
                pre_state = 0

                while pre_state < 24:
                    if batting > len(batting_line_T)-1: #allows for any number of batters, including one
                        batting = 0
                    post_state = self.move_state(pre_state, batting_line_T[batting])

                    runs += self.runs[pre_state, post_state]
                    pre_state = post_state

                    batting += 1

            tot_runs.append(runs)

        return np.mean(tot_runs)
        #sns.distplot((tot_runs))
        #plt.title(np.mean(tot_runs))
        #plt.show()

    def batter_permutations(self, batter_list = None):
        """create all possible permutations of batters
        """
        if batter_list is None:
            batter_list = ['goldp001','donaj001','cruzn002','vottj001','cabrm001','mccua001','machm001','heywj001','davic003']

        permutations = itertools.permutations(batter_list)
        self.permutations = [i for i in permutations]

        return [i for i in range(len(self.permutations))]


    def optimize_batting(self, permutations):

        results = []
        for loop, batter_list in enumerate(permutations):
            print (batter_list)
            #print (loop)
            avg_runs = self.simulate_games(batter_list=batter_list, N=2)
            results.append([batter_list, avg_runs])

        #results = sorted(results, key=itemgetter(-1), reverse=True)
        #plt.plot(np.array(results)[:,-1])
        #plt.show()
        #print (results)
        return results

    def optimize_batting2(self, index):

        #for index in indices:
        batter_list = self.permutations[index]
        #print (batter_list)
        #print (loop)
        avg_runs = self.simulate_games(batter_list=batter_list, N=10)
        results = [batter_list, avg_runs]
        print (index) if index % 10000 == 0 else None
        #results = sorted(results, key=itemgetter(-1), reverse=True)
        #plt.plot(np.array(results)[:,-1])
        #plt.show()
        #print (results)
        return results


if __name__ == '__main__':
    mk = markov()
    #goldp001 - Paul GoldSchmidt
    #donaj001 - Josh Donaldson
    #cruzn002 - Nelson Cruz
    #cabrm001 - Migruel Cabrera
    #vottj001 - Joey Votto
    #mccua001 - Andrew McCutchen
    #machm001 - Manny Machado
    #heywj001 - Jason Heyward
    #davic003 - Chris Davis
    #print (mk.play_game())
    #mk.simulate_games()

    #using parallel code (max 10 cores)
    max_cores_to_use = 10
    chunksize = 10000

    print ('Starting:\t', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    permutations_indices = mk.batter_permutations()
    print (len(permutations_indices))
    results = []
    #for result in map(mk.optimize_batting2,permutations_indices):
    #    results.append(result)


    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores_to_use) as executor:
        #for result in executor.map(mk.optimize_batting2, permutations_indices):
        results = zip(executor.map(mk.optimize_batting2, permutations_indices))

    #print (results)
    print ('Finishing:\t', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    #plt.plot(np.array(results)[:,-1])
    #plt.show()
