import numpy as np
import pickle
import datetime
import scipy.integrate
import math
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

"""
Parameters:
    prod_mat (the production matrix of f_xy, default: f_xy = sqrt(1.1^x*1.5^y))
    r (discount rate, default: 0.05):
    gamma (worker's bargaining power, default: 0.55)
    rho (coefficient for Poisson Process arrival rate, default: 1)
    delta (probability of matches ending, default: 0.5)
    theta (Pareto distribution shape parameter, default: 10)

When the production and distribution matrices are modified to contain different
    number of firm and worker types than in the paper, the code accounts for
    such changes in the solution algorithm so long as the number of workers
    and firms are consistent.
"""

class Market_WP:
    def __init__(self, unem, vaca, lhs_wage, lhs_prof, prod_mat =
            np.array(list(
                [np.sqrt((1.1**x)*(1.5**y)) for y in range(10)] for x in range(10)
            )),
            r = 0.05, gamma = 0.55, rho = 1,
            delta = 0.5, theta = 10):
        """
        Constructs a labor market of given unemployment/vacancy rates and
        reservation wages/profits.
        """
        self.lhs_wage = lhs_wage
        self.lhs_prof = lhs_prof
        self.prod_mat = prod_mat
        self.gamma = gamma
        self.rho = rho
        self.delta = delta
        self.theta = theta
        self.unem = unem
        self.vaca = vaca
        self.r = r

    def res_prod(self):
        """
        Calculates reservation productivity with (7).
        """
        res_prod = []
        for x, wage in np.ndenumerate(self.lhs_wage):
            res_worker = []
            for y, prof in np.ndenumerate(self.lhs_prof):
                res_worker.append((wage + prof) / self.prod_mat[x][y])
            res_prod.append(res_worker)
        return np.array(res_prod)

    def match_rate(self):
        """
        Calculate match rates with (10).
        """
        match_rate = []
        res_prod = self.res_prod()
        for x, u in np.ndenumerate(self.unem):
            match_worker = []
            for y, v in np.ndenumerate(self.vaca):
                match_worker.append(u * v * res_prod[x][y]**(- self.theta))
            match_rate.append(match_worker)

        return self.rho/self.delta*np.array(match_rate)

    def rhs_wage(self):
        """
        Calculates the right hand side of (8).
        """
        rhs_wage = []
        match_rates = self.match_rate()
        for x, w in np.ndenumerate(self.lhs_wage):
            worker_wage = 0
            for y, pi in np.ndenumerate(self.lhs_prof):
                worker_wage += match_rates[x][y] * (w + pi)
            rhs_wage.append(worker_wage / self.unem[x])
        return self.gamma * self.delta /((self.r + self.delta) * (self.theta -1)) * np.array(rhs_wage)

    def rhs_prof(self):
        """
        Calculates the right hand side of (9).
        """
        rhs_prof = []
        match_rates = self.match_rate()
        for y, pi in np.ndenumerate(self.lhs_prof):
            firm_prof = 0
            for x, w in np.ndenumerate(self.lhs_wage):
                firm_prof += match_rates[x][y] * (w + pi)
            rhs_prof.append(firm_prof / self.vaca[y])
        return (1 - self.gamma) * self.delta /((self.r + self.delta) * (self.theta -1)) * np.array(rhs_prof)

    def is_equilibrium(self):
        """
        Determines whether the reservation wages/profits satisfy (8)(9) for
        fixed unemployment and vacancy rates.
        """
        wage_diff = np.absolute(self.rhs_wage() - self.lhs_wage)
        prof_diff = np.absolute(self.rhs_prof() - self.lhs_prof)
        return np.amax(np.concatenate((wage_diff, prof_diff))) < 0.001

    def update_WP(self):
        """
        Guesses a lower reseravtion wage/profit if last guess is higher than the
        right hand side fo (8)/(9) using gradient descent.
        Guesses higher if lower than right hand side.
        """
        rhs_wage = self.rhs_wage()
        for x, w in np.ndenumerate(self.lhs_wage):
            wage = rhs_wage[x]
            if w > wage + 1.:
                self.lhs_wage[x] -= 0.01
            elif w > wage + 0.001:
                self.lhs_wage[x] -= 0.00005
            elif wage - 1 < w < wage - 0.001:
                self.lhs_wage[x] += 0.00005
            elif w < wage - 1.:
                self.lhs_wage[x] += 0.01

        rhs_prof = self.rhs_prof()
        for y, pi in np.ndenumerate(self.lhs_prof):
            prof = rhs_prof[y]
            if pi > prof + 1.:
                self.lhs_prof[y] -= 0.01
            elif pi > prof + 0.001:
                self.lhs_prof[y] -= 0.00005
            elif prof - 1 < pi < prof - 0.001:
                self.lhs_prof[y] += 0.00005
            elif pi < prof - 1:
                self.lhs_prof[y] += 0.01

def solve_WP(guess_W, guess_Pi, unem, vaca, prod_mat, r, gamma, rho, delta, theta):
    market = Market_WP(unem, vaca, guess_W, guess_Pi, prod_mat, r, gamma, rho,
        delta, theta)
    """
    Solves for the reseravtion wage and profit that satisfy (8)(9) for given
    unemployment and vacancy rates by updating the guess after comparing the LHS
    and RHS of both equation until there is approximate equality.
    """
    while not market.is_equilibrium():
        market.update_WP()

    return market.match_rate(), market.rhs_wage(), market.rhs_prof(), market.res_prod()

class Market_UV:
    def __init__(self, lhs_unem, lhs_vaca, prod_mat =
            np.array(list(
                [np.sqrt((1.1**x)*(1.5**y)) for y in range(10)] for x in range(10)
            )),
            r = 0.05, gamma = 0.55, rho = 1,
            delta = 0.5, theta = 10, dist_X = np.full((10), 0.1),
            dist_Y = np.full((10), 0.1)):
        """
        Constructs a labor market of given unemployment and vacancy rate.
        The solve_WP function is called to characterize the reservation
        wages and profits given undmployment and vacancy rate.
        """
        self.prod_mat = prod_mat
        self.r = r
        self.gamma = gamma
        self.rho = rho
        self.delta = delta
        self.theta = theta
        self.lhs_unem = lhs_unem
        self.lhs_vaca = lhs_vaca
        self.dist_X = dist_X
        self.dist_Y = dist_Y
        self.match_rate, self.res_wage, self.res_prof, self.res_prod = solve_WP(np.full((len(self.dist_X)), 1.), np.full((len(self.dist_Y)), 1.),
            self.lhs_unem, self.lhs_vaca, self.prod_mat,self.r, self.gamma, self.rho, self.delta, self.theta)

    def rhs_unem(self):
        """
        Calulates the right hand side of (11).
        """
        rhs_unem = []
        for x, m in np.ndenumerate(self.dist_X):
            rhs_unem.append(m - np.sum(self.match_rate[x, :]))
        return np.array(rhs_unem)

    def rhs_vaca(self):
        """
        Calculates the right hand side of (12).
        """
        rhs_vaca = []
        for y, n in np.ndenumerate(self.dist_Y):
            rhs_vaca.append(n - np.sum(self.match_rate[:, y]))
        return np.array(rhs_vaca)

    def update_UV(self):
        """
        Guesses a lower unemployment/vacancy rate if last guess is higher than
        the right hand side fo (11)/(12) using gradient descent.
        Guesses higher if lower than right hand side.
        """
        rhs_unem = self.rhs_unem()
        for x, u in np.ndenumerate(self.lhs_unem):
            if u > rhs_unem[x] + 0.01:
                self.lhs_unem[x] -= 0.0001
            elif u > rhs_unem[x] + 0.0001:
                self.lhs_unem[x] -= 0.000001
            elif rhs_unem[x] - 0.01 < u < rhs_unem[x] - 0.0001:
                self.lhs_unem[x] += 0.000001
            elif u < rhs_unem[x] - 0.01:
                self.lhs_unem[x] += 0.0001

        rhs_vaca = self.rhs_vaca()
        for y, v in np.ndenumerate(self.lhs_vaca):
            if v > rhs_vaca[y] + 0.01:
                self.lhs_vaca[y] -= 0.0001
            elif v > rhs_vaca[y] + 0.001:
                self.lhs_vaca[y] -= 0.00002
            elif rhs_vaca[y] - 0.01 < v < rhs_vaca[y] - 0.001:
                self.lhs_vaca[y] += 0.00002
            elif v < rhs_vaca[y] - 0.01:
                self.lhs_vaca[y] += 0.0001

        self.match_rate, self.res_wage, self.res_prof, self.res_prod = solve_WP(self.res_wage, self.res_prof,
                self.lhs_unem, self.lhs_vaca, self.prod_mat, self.r, self.gamma, self.rho,
                self.delta, self.theta)

    def is_equilibrium(self):
        unem_diff = np.absolute(self.rhs_unem() - self.lhs_unem)
        vaca_diff = np.absolute(self.rhs_vaca() - self.lhs_vaca)
        return np.amax(np.concatenate((unem_diff, vaca_diff))) < 0.001

def run(market):
    # while not market.is_equilibrium(): #Optional dumping into pickle for times
    ## when the code takes too long to run
        # print("Dumping at ", datetime.datetime.now())
        # with open(r"market.pickle", "wb") as output_file:
        #     pickle.dump(market, output_file)

    return market


def solve_UV(prod_mat = np.array(list([np.sqrt((1.1**x)*(1.5**y))
        for y in range(10)] for x in range(10))), r = 0.05, gamma = 0.55,
            rho = 1, delta = 0.5, theta = 10, dist_X = np.full((10), 0.1), dist_Y =
            np.full((10), 0.1)):
    market = Market_UV(np.full((len(dist_X)), 0.01), np.full((len(dist_Y)), 0.01), prod_mat, r,
            gamma, rho, delta, theta, dist_X, dist_Y)

    return run(market)

class LaborMarket:
    """
    Constructs a labor market in equilibrium. That is, the reservation wages and
    profits and the unemployment and vacany rates statisfy (8)(9)(11)(12).
    """
    def __init__(self, prod_mat = np.array(list([np.sqrt((1.1**x)*(1.5**y))
        for y in range(10)] for x in range(10))), r = 0.05, gamma = 0.55,
        rho = 1, delta = 0.5, theta = 10, dist_X = np.full((10), 0.1), dist_Y =
        np.full((10), 0.1)):
        self.prod_mat = prod_mat
        self.r = r
        self.gamma = gamma
        self.rho = rho
        self.delta = delta
        self.theta = theta
        self.dist_X = dist_X
        self.dist_Y = dist_Y
        self.equilibirium = solve_UV(self.prod_mat, self.r, self.gamma,
            self.rho, self.delta, self.theta, self.dist_X, self.dist_Y)
        self.res_wage = self.equilibirium.res_wage
        self.res_prof = self.equilibirium.res_prof
        self.res_prod = self.equilibirium.res_prod
        self.unem = self.equilibirium.rhs_unem()
        self.vaca = self.equilibirium.rhs_vaca()
        self.matches = self.equilibirium.match_rate

    def get_ave_log_wage(self):
        """
        Calculates average log wage received by worker from firm.
        """
        ave_log_wage = []
        for w in self.res_wage:
            worker_wage = []
            for pi in self.res_prof:
                def g(q):
                    return math.log(w + self.gamma * (w + pi)*q) * (self.theta *
                        (1 + q)**(-self.theta - 1))
                worker_wage.append(scipy.integrate.quad(g, 0, np.inf)[0])
            ave_log_wage.append(worker_wage)
        return ave_log_wage

    def get_ave_wage_offer(self):
        """
        Calculates average wage offer paid by firm to worker.
        """
        z_0 = np.min(self.res_prod)
        firm_offers = []
        for x, w in np.ndenumerate(self.res_wage):
            worker_offers = []
            for y, pi in np.ndenumerate(self.res_prof):
                z = max(z_0, (pi - (1 - self.gamma)*w / self.gamma)
                    / self.prod_mat[x][y])
                def g(q):
                    return math.log(self.gamma*(q*self.prod_mat[x][y] - w - pi)
                        + w)*(q**(-self.theta -1))
                worker_offers.append(self.theta / (z**(-self.theta)) *
                    scipy.integrate.quad(g, z, np.inf)[0])
            firm_offers.append(worker_offers)
        return firm_offers

    def plot_worker_distribution(self, ax, cm):
        """
        Replicates bar chart on worker distribution.
        """
        plt = ax
        num_firms = len(self.dist_Y)
        firms = tuple(y + 1 for y in range(num_firms))
        worker_percents = {}
        firm_matches = []
        for y in range(num_firms):
            firm_matches.append(np.sum(self.matches[:, y]))

        for x, matches in enumerate(self.matches):
            percents = []
            for y, match in enumerate(matches):
                percents.append(match / firm_matches[y])
            worker_percents[x] = percents
        bottom = np.zeros(num_firms)

        for x, share in worker_percents.items():
            p = plt.bar(firms, share, 0.4, label = f"x = {x + 1}",
                bottom = bottom, color=cm[x])
            bottom += share
            plt.bar_label(p, labels = ['' for i in range(len(self.dist_Y))],
                label_type = 'center', color=cm[x])

        plt.set_xlabel('firm type y')
        plt.set_xticks(firms)
        plt.set_ylabel('share of worker types')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #plt.show() #Shows figure
        #plt.savefig('worker_dist.png') #Saves figure

    def plot_ave_log_wage(self, ax, cm):
        """
        Replicates graph plotting average log wage.
        """
        num_firms = len(self.dist_Y)
        firms = tuple(y + 1 for y in range(num_firms))
        plt = ax
        for x, wages in enumerate(self.get_ave_log_wage()):
            plt.plot(firms, wages, marker = "x", label = f"x = {x + 1}", color=cm[x])
        plt.set_xticks(firms)
        plt.set_xlabel("firm type y")
        plt.set_ylabel("average log wage")


    def plot_wage_offers(self, ax, cm):
        """
        Replicates graph plotting average log wage offer.
        """
        num_firms = len(self.dist_Y)
        firms = tuple(y + 1 for y in range(num_firms))
        plt = ax
        for x, wages in enumerate(self.get_ave_wage_offer()):
            plt.plot(firms, wages, marker = "x", label = f"x = {x + 1}", color=cm[x])
        plt.set_xticks(firms)
        plt.set_xlabel("firm type y")
        plt.set_ylabel("average log wage offer")

    def replicate(self, filename = 'Borovičková-Shimer-2024.pdf', fig_size=(7, 10)):
        """
        Creates and saves in current directory the three graphs
        as one figure, specifying the parameter values.
        """
        matplotlib.use('Agg')
        num_workers = len(self.dist_X)
        fig, axs = plt.subplots(2, 2)
        fig.set_figheight(fig_size[0])
        fig.set_figwidth(fig_size[1])

        cm = plt.cm.RdBu(np.linspace(0, 1, num_workers))

        self.plot_ave_log_wage(axs[0, 0], cm)
        self.plot_worker_distribution(axs[0, 1], cm)
        self.plot_wage_offers(axs[1, 0], cm)
        axs[1, 1].axis("off")
        textstr = '\n\n'.join((rf'Discount factor $r = {self.r}$', rf'Worker bargaining power $\gamma = {self.gamma}$', rf'Arrival rate $\rho = {self.rho}$',rf'Match end rate $\delta = {self.delta}$',
            rf'Pareto shape parameter $\theta = {self.theta}$'))
        axs[1, 1].text(0., 0., textstr)
        plt.tight_layout(pad = 1.5)
        # plt.show()
        fig.savefig(filename, bbox_inches='tight')



if __name__ == "__main__":
    market = LaborMarket()
    market.replicate()
