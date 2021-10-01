class LinearRegression:
    def __init__(self, features, Y, max_itter=10000, max_precision=0.00001, learning_rate=0.0001):
        self.features = [np.ones(len(Y), dtype=int), *features] # Features to be trained on
        self.Y = Y # Actual Output Data

        self.theetas = np.random.random_sample((1, len(self.features)))[0] # Initialization of theeta (Hypothesis Variables)
        self.learning_rate = learning_rate

        self.max_itter = max_itter
        self.max_precision = max_precision

    def h(self, x):
        return np.dot(x, self.theetas)

    def features_at(self, i):
        return [f[i] for f in self.features]

    def J(self):
        return (1/2) * sum([(self.h(self.features_at(i)) - self.Y[i])**2 for i in range(len(self.Y))])
        
    def gradient_descent_at_i(self, i):
        n = 0
        precision = 1
        prev_theeta = 0

        while n < self.max_itter and self.max_precision < precision:
            prev_theeta = self.theetas[i]

            self.theetas[i] = self.theetas[i] - self.learning_rate * sum([(self.h(self.features_at(j)) - self.Y[j])*self.features[i][j] for j in range(len(self.Y))])

            precision = abs(self.theetas[i] - prev_theeta)
            n += 1

    def gradient_descent(self):
        # for i in range(len(self.features)):
        #     self.gradient_descent_at_i(i)

        leny = len(self.Y)

        for _ in range(self.max_itter):
            for i in range(len(self.features)):
                self.theetas[i] = self.theetas[i] - self.learning_rate * sum([(self.h(self.features_at(j)) - self.Y[j])*self.features[i][j] for j in range(leny)])

        return self.theetas

    def graph_acc(self):
        import matplotlib.pyplot as plt

        if len(self.features) == 3:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            
            ax.scatter3D(self.features[1], self.features[2], self.Y)
            minf1, maxf1 = min(self.features[1]), max(self.features[1])
            minf2, maxf2 = min(self.features[2]), max(self.features[2])
            minout, maxout = self.h([1, minf1, minf2]), self.h([1, maxf1, maxf2])
            ax.plot3D([minf1, maxf1], [minf2, maxf2], [minout, maxout], c='r')
            plt.show()

        elif len(self.features) == 2:
            plt.scatter(self.features[1], self.Y)
            plt.plot(self.features[1], [self.h(self.features_at(i)) for i in range(len(self.Y))])
            plt.show()

        else:
            raise "Can not plot for more than 3 dimentions"
