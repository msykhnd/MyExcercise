import numpy as np


def RosenBrock(x):
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2)


class RealCodedGeneticAlgorithm:
    def __init__(self):
        self.input_min = -2.048
        self.input_max = 2.048
        self.num_variables = 50
        self.num_genes = 20
        self.num_iteration = 100000
        self.mutant_scale = 0.001
        self.mutant_rate = 0.8
        self.crossover_rate = 0.7
        self.env = self.init_env()  # (num_genes, num_variable)
        self.generation_log=[]
        self.elite_log=[]

    def gen_gene(self):
        return (self.input_max - self.input_min) * np.random.rand(self.num_variables) + self.input_min

    def init_env(self):
        gen_list = []
        for i in range(self.num_genes):
            gen_list.append(self.gen_gene())
        return np.array(gen_list)

    def evaluator(self, x):
        return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2)

    def env_evaluate(self):
        eval_list = []
        for x in self.env:
            eval_list.append(self.evaluator(x))
        return np.array(eval_list)

    def elite_selector(self):
        sorted_env = self.env[np.argsort(self.env_evaluate())]
        elite_gene = sorted_env[0]
        # print(RosenBrock(sorted_env[0]))
        else_genes = sorted_env[1:]
        return elite_gene, else_genes

    def roulette_selector(self, genes):
        eval_list = []
        for x in genes:
            eval_list.append(self.evaluator(x))
        eval_list = np.abs(eval_list - np.max(eval_list))
        total = np.sum(eval_list)

        selected_index = []
        for _ in range(2):
            chose_point = np.random.uniform(0.0, total)
            sum = 0.0
            for index, value in enumerate(eval_list):
                sum += value
                if sum >= chose_point:
                    selected_index.append(index)
                    eval_list[index] = 0.
                    total -= value
                    break
        return selected_index

    def uniform_crossover(self, p1, p2):
        x = np.copy(p1)
        y = np.copy(p2)
        for i in range(len(p1)):
            if np.random.rand() < 0.5:
                x[i], y[i] = y[i], x[i]
        return x, y

    def gen_mutant(self, x):
        x = x + np.random.normal(loc=0.0, scale=self.mutant_scale, size=self.env[0].shape)
        x = np.where(x < self.input_min, self.input_min, x)
        x = np.where(x > self.input_max, self.input_max, x)
        return x

    def new_generation(self):
        children = []
        # print("env shape ",self.env.shape)

        elite, elses = self.elite_selector()

        # print(elses.shape)
        np.random.shuffle(elses)

        sep_point = int(self.crossover_rate * len(self.env))
        if sep_point % 2 == 1 & sep_point != 0:
            sep_point -= 1

        target, static = elses[:sep_point], elses[sep_point:]
        # print(sep_point)
        # print(target.shape)

        for i in range(0, int(sep_point / 2)):
            parents_index = self.roulette_selector(elses)
            parent_1 = elses[parents_index[0]]
            parent_2 = elses[parents_index[1]]
            child_1, child_2 = self.uniform_crossover(parent_1, parent_2)
            children.extend([child_1, child_2])

        for j in range(0, len(children)):
            if np.random.rand() < self.mutant_rate:
                children[j] = self.gen_mutant(children[j])

        new_generation = np.vstack((elite.reshape(1, len(elite)), static, np.array(children)))
        self.env = new_generation

    def fit(self):
        for i in range(self.num_iteration):
            self.new_generation()
            if i % 1000== 0:
                best, _ = self.elite_selector()
                eval = self.evaluator(best)
                print(eval)
                self.generation_log.append(sum(self.env_evaluate()))
                self.elite_log.append(eval)


if __name__ == "__main__":
    # x = np.arange(50)
    RGA = RealCodedGeneticAlgorithm()
    # elite, elses = RGA.elite_selector()
    # print(elses.shape)
    # print(RGA.env_evaluate())
    # parents_index = RGA.roulette_selector(elses)
    # mother = elses[parents_index[0]]
    # father = elses[parents_index[1]]
    #
    # print("母", mother)
    # print("父", father)
    # print("kogomo", RGA.uniform_crossover(mother, father))
    RGA.fit()
    print(RGA.elite_log)
