import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
#import concurrent.futures
import time
#from operator import itemgetter
import datetime

from helper import PRIOR, OUTS, RUNS, timeit

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
        #print (Er)
        return Er


    def eo(self, Tp=None):
        """Function to calculate expected out
        """
        Eo = np.sum(self.outs*Tp , axis=1).reshape([3,8])
        #print (Eo)
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
        """final results has 3D --> shape[N, 24, 24] where N is no of batters
        """
        bT = np.zeros([len(batter_list),24,24], dtype=float)
        #bT = []
        #bT[i].append(self.transition(player_id = batter, precision = 10)[:24,:24])
        for i in range(len(batter_list)):
            bT[i,:,:] = self.transition(player_id = batter_list[i], precision = 10)[:24,:24]
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

    def simulate_games(self, batter_list=['goldp001','donaj001','cruzn002','vottj001','cabrm001','mccua001','machm001','heywj001','davic003'], N = 10, innings = 9):
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

    # def batter_permutations(self, batter_list = ['goldp001','donaj001','cruzn002','vottj001','cabrm001','mccua001','machm001','heywj001','davic003']):
    #     """create all possible permutations of batters
    #     """
    #     permutations = itertools.permutations(batter_list)
    #     self.permutations = [i for i in permutations]
    #
    #     return [i for i in range(len(self.permutations))]


    # def optimize_batting(self, permutations):
    #
    #     results = []
    #     for loop, batter_list in enumerate(permutations):
    #         #print (batter_list)
    #         #print (loop)
    #         avg_runs = self.simulate_games(batter_list=batter_list, N=2)
    #         results.append([batter_list, avg_runs])
    #
    #     #results = sorted(results, key=itemgetter(-1), reverse=True)
    #     #plt.plot(np.array(results)[:,-1])
    #     #plt.show()
    #     #print (results)
    #     return results

    # def optimize_batting2(self, index):
    #
    #     batter_list = self.permutations[index]
    #     avg_runs = self.simulate_games(batter_list=batter_list, N=10)
    #     results = [batter_list, avg_runs]
    #     #print (index) if index % 10000 == 0 else None
    #     #results = sorted(results, key=itemgetter(-1), reverse=True)
    #     #plt.plot(np.array(results)[:,-1])
    #     #plt.show()
    #     #print (results)
    #     return results

    #def expected_run(self, Tp):
    #    Er = np.sum(self.runs[:24,:24]*Tp , axis=1).reshape([24,1])
    #    return Er

    def rotate_batters(self, current_batter_list):
        """rotate batters once, to the left"""
        next_batter_list = current_batter_list[1:]
        next_batter_list.append(current_batter_list[0])
        return next_batter_list


    def line_runs_transitions(self, batter_list=['goldp001','donaj001','cruzn002','vottj001','cabrm001','mccua001','machm001','heywj001','davic003']):
        """dictionaries with transitions and runs for batting line
        """
        transitions = {}
        runs = {}

        for player in batter_list:#9 loops
            transitions[player] = self.transition(player_id = player, precision = 10)[:24,:24]
            runs[player] = np.sum(self.runs[:24,:24]*transitions[player], axis=1)

        return transitions, runs

    #have to create an ordered permutation to reduce lookup time in optimization
    @timeit
    def batter_permutation(self, batter_list):
        """it creates 3X times the number of batters, to account for potential
        long innings, where a full rotation happens (~0.01% of cases)
        """
        n = len(batter_list)
        #first get all possibe sequences permutaitons of (n-1)
        combinations = itertools.permutations(batter_list[1:])
        ordered_batter_list = []
        for combination in combinations:
            batter_seq = [batter_list[0]] + list(combination)
            ordered_batter_list.append(batter_seq+batter_seq+batter_seq)
            for i in range(n-1):
                batter_seq = self.rotate_batters(batter_seq)
                ordered_batter_list.append(batter_seq+batter_seq+batter_seq)

        return ordered_batter_list

    #@timeit
    def Er_out_Matrix(self, permutations_list, runs, transitions, dim):
        """Returns earned run and out matrix for each batter lineup
        It is the backbone for the optimal batter calculation
        """


        ER_matrix = np.zeros([len(permutations_list),24])
        Outs_matrix = np.zeros([len(permutations_list),27]) #Out prob, starting from that batter

        #batters_list = []
        for loop, batters in enumerate(permutations_list):
            #batters_list.append(list(batters[:9]))

            #T_list = np.zeros([len(batters),24,24])
            Er_list = np.zeros([len(batters),24,1])

            current_T = np.identity(24)

            for i in range(len(batters)):

                current_T = np.dot(current_T, transitions[batters[i]])
                current_run = runs[batters[i]]

                if i == 0:
                    ER = np.dot(np.identity(24), current_run)
                    current_U = transitions[batters[i]][0,:]
                    prob = 1-np.sum(current_U)
                    Outs_matrix[loop][i] = prob
                else:
                    ER += np.dot(current_T, current_run)
                    prob = 1-np.sum(np.dot(current_U, transitions[batters[i]]))
                    current_U = np.dot(current_U, transitions[batters[i]])
                    Outs_matrix[loop][i] = prob - np.sum(Outs_matrix[loop][2:])

            ER_matrix[loop] = ER.reshape([24,])

        Outs_matrix[:,:2] = 0.

        return ER_matrix, Outs_matrix


    #@timeit
    def expected_run(self,batters_out_prob, batters_er_on_00, dim):
        er_total = 0
        new = np.zeros((dim,dim))
        new[:,1:] = batters_out_prob[:,:-1]
        new[:,0] = batters_out_prob[:,-1]

        er1 = batters_er_on_00[0,0]

        start_second = new[0,:]
        first_ER = batters_er_on_00[:,0]
        er2 = np.dot(start_second, first_ER)

        start_third = start_second.reshape(dim,1) * new
        er3 = np.sum(start_third * batters_er_on_00)

        er_total = er1 + er2 + er3

        start_previous = start_third
        for last_innings in [4,5,6,7,8,9]:
            start_next = np.zeros((dim,1))

            for i in range(dim):

                if i == 8: #diaognal 0 = last player = player 8
                    x = 0
                else:
                    x = i + 1

                start_next[i,:] = np.trace(start_previous[:,::-1], -x) + np.trace(start_previous[:,::-1], dim - x) #adding up the diagonals

            start_next = start_next.reshape(dim,1) * new
            start_previous = start_next.copy()
            er_total += np.sum(start_next * batters_er_on_00)

        return er_total


    #@timeit
    def out_er(self, ER_matrix, Outs_matrix, batters_list, dim, index_st):
        batters_out_prob = np.zeros([dim,dim])
        batters_er_on_00 = np.zeros([dim,1])

        index_end = index_st + dim
        batters_out_prob = Outs_matrix[index_st:index_end,0:dim]
        batters_er_on_00[:,0] = ER_matrix[index_st:index_end,0]

        batters_out_prob[:,:2] = 0.

        return batters_out_prob, batters_er_on_00


    @timeit
    def optimize_line(self, batter_list=['goldp001','donaj001','cruzn002','vottj001','cabrm001','mccua001','machm001','heywj001','davic003']):
        dim = 9

        transitions, runs = self.line_runs_transitions(batter_list)
        permutations_list = self.batter_permutation(batter_list)

        ER_matrix, Outs_matrix = self.Er_out_Matrix(permutations_list=permutations_list, runs=runs, transitions=transitions,dim=dim)

        er_total = np.zeros((len(permutations_list), 1))

        for loop in range(0,len(permutations_list),dim):

            for i in range(dim):
                batters_out_prob = np.zeros([dim,dim])
                batters_er_on_00 = np.zeros([dim,1])

                if i == 0:
                    index_st = loop
                    index_end = loop + dim
                    batters_out_prob = Outs_matrix[index_st:index_end,0:dim]
                    batters_er_on_00[:,0] = ER_matrix[index_st:index_end,0]
                    batters_out_prob[:,:2] = 0.

                    p_batters_out_prob = batters_out_prob
                    p_batters_er_on_00 = batters_er_on_00
                else: #rotate matrices for different starting batters
                    batters_out_prob[:,:-i] = p_batters_out_prob[:,i:]
                    batters_out_prob[:,-i:] = p_batters_out_prob[:,:i]

                    batters_er_on_00[:-i,0] = p_batters_er_on_00[i:,0]
                    batters_er_on_00[-i:,0] = p_batters_er_on_00[:i:,0]

                er_total[loop+i,0] = self.expected_run(batters_out_prob=batters_out_prob, batters_er_on_00=batters_er_on_00, dim=dim)

        best_lines = np.array(permutations_list).reshape([len(permutations_list),27])[:,:9]

        return er_total, best_lines


if __name__ == '__main__':
    mk = markov()

    #using parallel code (max 10 cores)'''
    print ('Starting:\t', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    max_cores_to_use = 10
    '''
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

    #plt.plot(np.array(results)[:,-1])
    #plt.show()
    '''
    #Altuve, Blackmon, J.D. Martinez, Stanton, Inciarte, Jose Ramirez, Votto, Pujols, Trout
    best_2017 = ['altuj001','martj006','blacc001','stanm004','incie001','ramij003','vottj001','pujoa001','troum001']

    red_socks = ['bettm001','benia002','martj006','bogax001','holtb002','vazqc001','nunee002','leons001','bradj001']
    er_total, bline = mk.optimize_line(batter_list = best_2017)
    #er_games = mk.simulate_games()

    print ('Finishing:\t', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
