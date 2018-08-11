import numpy as np
import copy

TAU = 1e-12
INT_MAX = 2147483647


class Kernel(object):
    '''
    定义核函数类别
    '''

    def __init__(self, dataMatIn, kernel_type='rbf', degree=3, gamma=0.1, coef0=0.0):
        self.kernel_type = kernel_type
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.dataMatIn = dataMatIn

    def kernel_function(self, i, j):
        if self.kernel_type == 'linear':
            return self.dataMatIn[i] * self.dataMatIn[j].T
        elif self.kernel_type == 'rbf':
            return np.exp(-self.gamma * (np.linalg.norm(self.dataMatIn[i] - self.dataMatIn[j]) ** 2))
        elif self.kernel_type == 'poly':
            return (self.gamma * (self.dataMatIn[i] * self.dataMatIn[j].T) + self.coef0) ** self.degree
        elif self.kernel_type == 'sigmoid':
            return (np.tanh(self.gamma * self.dataMatIn[i] * self.dataMatIn[j].T) + self.coef0)
        elif self.kernel_type == 'laplacian':
            return np.exp(-self.gamma * np.linalg.norm(self.dataMatIn[i] - self.dataMatIn[j]))
        elif self.kernel_type == 'morlet':
            morlet = 1
            for k in range(self.dataMatIn.shape[1]):
                morlet *= (np.cos(1.75 * np.linalg.norm(self.dataMatIn[i, k] - self.dataMatIn[j, k]) / self.coef0) *
                           np.exp(-(np.linalg.norm(self.dataMatIn[i, k] - self.dataMatIn[j, k]) ** 2) / (2 * self.coef0 ** 2)))
            return morlet
        else:
            raise NameError('Houston We Have a Problem -- \
                That Kernel is not recognized')


class SVR_Q_Solve(Kernel):
    def __init__(self, dataMatIn, labelMat,
                 epsilon_svr, C, eps=1e-3,
                 kernel_type='rbf', degree=3, gamma=0.1, coef0=0.0):
        super().__init__(dataMatIn, kernel_type, degree, gamma, coef0)
        self.x = dataMatIn
        self.y = labelMat

        self.l, self.n = np.shape(dataMatIn)
        self.alpha_status = []
        for i in range(2 * self.l):
            self.alpha_status.append('free')

        self.alphas = np.mat(np.zeros((2 * self.l, 1)))
        self.C = C
        self.K = np.mat(np.zeros((self.l, self.l)))
        self.Q = np.mat(np.zeros((2 * self.l, 2 * self.l)))  # 为2lx2l包含了4个k阵
        self.sign = np.mat(np.zeros((2 * self.l, 1)))  # SVR定义的y值，前l个为1后l个为-1
        self.index = np.mat(np.zeros((1, 2 * self.l)))  # 2l个index

        self.G = np.mat(np.zeros((1, 2 * self.l)))
        self.G_bar = np.mat(np.zeros((1, 2 * self.l)))
        self.eps = eps
        self.p = np.mat(np.zeros((1, 2 * self.l)))

        self.rho = 0
        self.obj = np.mat(np.zeros((self.l, 1)))

        # 求解P
        for i in range(self.l):
            # 此处libsvm源码为 '-'，下一行为 '+'( 根据公式计算也是如此)
            self.p[:, i] = epsilon_svr - labelMat[i]
            self.p[:, i + self.l] = epsilon_svr + labelMat[i]

        # 定义y，index，求解核矩阵k，变形矩阵Q
        for k in range(self.l):
            self.sign[k] = 1
            self.sign[k + self.l] = -1
            self.index[:, k] = k
            self.index[:, k + self.l] = k

        for i in range(self.l):
            for j in range(self.l):
                self.K[i, j] = self.get_K(i, j)
        self.Q[:self.l, :self.l] = self.K
        self.Q[self.l:, self.l:] = self.K

        self.Q[:self.l, self.l:] = -self.K
        self.Q[self.l:, :self.l] = -self.K

    def update_alpha_status(self, i):
        if self.alphas[i] >= self.C:
            self.alpha_status[i] = 'upper_bound'
        elif self.alphas[i] <= 0:
            self.alpha_status[i] = 'lower_bound'
        else:
            self.alpha_status[i] = 'free'

    def is_upper_bound(self, i):
        return self.alpha_status[i] == 'upper_bound'

    def is_lower_bound(self, i):
        return self.alpha_status[i] == 'lower_bound'

    def is_free(self, i):
        return self.alpha_status[i] == 'free'

    def get_K(self, i, j):
        kij = self.kernel_function(i, j)
        return kij

    def select_working_set(self, out_i, out_j):
        '''
        return i, j such that
        i: maximizes - y_i * grad(f)_i, i in I_up(\alpha)
        j: minimizes the decrease of obj value
        (if quadratic coefficeint <= 0, replace it with tau)
        -y_j * grad(f)_j < -y_i * grad(f)_i, j in I_low(\alpha)
        '''
        Gmax = -float('inf')  # -yi*G(alphai)
        Gmax2 = -float('inf')  # yj*G(alphaj)
        Gmax_idx = -1
        Gmin_idx = -1
        obj_diff_min = float('inf')

        # 寻找working set B中的i
        for t in range(2 * self.l):
            if self.sign[t] == 1:
                if self.is_upper_bound(t) is False:  # 对应于yi=1,alphai<c
                    if -self.G[:, t] >= Gmax:
                        # 寻找最大的-yi*G(alphai)，以使违反条件最严重
                        Gmax = -self.G[:, t] .copy()
                        Gmax_idx = t
            else:
                if self.is_lower_bound(t) is False:
                    if self.G[:, t] > Gmax:
                        Gmax = self.G[:, t].copy()
                        Gmax_idx = t
        i = Gmax_idx

        if i == -1:
            self.Q[:, i] = 0
            self.Q[i, :] = 0

        # 寻找working set B中的j
        '''
        working set selection using second order information for training support vector machines
        中的 Theorenm3, j使目标值最小
        '''
        for j in range(2 * self.l):
            if self.sign[j] == 1:
                if self.is_lower_bound(j) is False:
                    grad_diff = Gmax + self.G[:, j].copy()  # 分子
                    if self.G[:, j] >= Gmax2:
                        Gmax2 = self.G[:, j].copy()
                    if grad_diff > 0:  # 保证不满足KKT条件
                        quad_coef = self.Q[i, i] + self.Q[j, j] - \
                            2 * self.sign[i] * self.Q[j, i]  # 分母
                        if quad_coef > 0:
                            obj_diff = -(grad_diff ** 2) / quad_coef
                        else:
                            obj_diff = -(grad_diff ** 2) / TAU
                        if obj_diff <= obj_diff_min:
                            Gmin_idx = j
                            obj_diff_min = obj_diff
            else:
                if self.is_upper_bound(j) is False:
                    grad_diff = Gmax - self.G[:, j].copy()
                    if -self.G[:, j] >= Gmax2:
                        Gmax2 = -self.G[:, j].copy()
                    if grad_diff > 0:
                        quad_coef = self.Q[i, i] + self.Q[j,
                                                          j] + 2 * self.sign[i] * self.Q[i, j]
                        if quad_coef > 0:
                            obj_diff = -(grad_diff ** 2) / quad_coef
                        else:
                            obj_diff = -(grad_diff ** 2) / TAU
                        if obj_diff <= obj_diff_min:
                            Gmin_idx = j
                            obj_diff_min = obj_diff

        if Gmax + Gmax2 < self.eps or Gmin_idx == -1:
            print('Gmax+Gmax2: ', Gmax + Gmax2)
            return 1, out_i, out_j  # 表示已经完全优化

        out_i = Gmax_idx
        out_j = Gmin_idx
        return 0, out_i, out_j

    def solve(self):
        # initialize alpha_status
        for i in range(2 * self.l):
            self.update_alpha_status(i)

        # initialize gradient
        for i in range(2 * self.l):
            self.G[:, i] = self.p[:, i]

        for i in range(2 * self.l):
            if self.is_lower_bound(i) is False:
                for j in range(2 * self.l):
                    self.G[:, j] += self.alphas[i] * self.Q[i, j]
                if self.is_upper_bound(i) is True:
                    for j in range(2 * self.l):
                        self.G_bar[:, j] += self.C * self.Q[i, j]

        # optimization step
        iter = 0
        iter_max = max(10000000, max(INT_MAX, 100 * self.l))

        while iter < iter_max:
            i = 0
            j = 0
            judge, i, j = self.select_working_set(i, j)
            if judge != 0:
                # reset active set size and check
                judge, i, j = self.select_working_set(i, j)
                if judge != 0:
                    print("****break****")
                    break

            iter += 1
            # update alpha[i] and alpha[j], handle bounds carefully
            old_alpha_i = copy.deepcopy(self.alphas[i])
            old_alpha_j = copy.deepcopy(self.alphas[j])

            # yi,yj异号
            if self.sign[i] != self.sign[j]:
                quad_coef = self.Q[i, i] + self.Q[j, j] + 2 * \
                    self.Q[i, j]  # 最后一个为+号因为Qij为kij*sign[i]*sign[j]
                if quad_coef <= 0:
                    quad_coef = TAU
                delta = (-self.G[:, i] - self.G[:, j]) / quad_coef  # alpha改变量
                # 根据此项判断alpha(i)-alpha(j)=constant与约束框(0~c)的交点
                diff = self.alphas[i] - self.alphas[j]
                self.alphas[i] += delta
                self.alphas[j] += delta
                if diff > 0:
                    if self.alphas[j] < 0:
                        self.alphas[j] = 0
                        self.alphas[i] = diff
                    if self.alphas[i] > self.C:
                        self.alphas[i] = self.C
                        self.alphas[j] = self.C - diff
                else:
                    if self.alphas[i] < 0:
                        self.alphas[i] = 0
                        self.alphas[j] = -diff
                    if self.alphas[j] > self.C:
                        self.alphas[j] = self.C
                        self.alphas[i] = self.C + diff
            else:
                quad_coef = self.Q[i, i] + self.Q[j, j] - 2 * self.Q[i, j]
                if quad_coef <= 0:
                    quad_coef = TAU
                delta = (self.G[:, i] - self.G[:, j]) / quad_coef  # alpha该变量
                # 根据此项判断alpha(i)-alpha(j)=constant与约束框(0~c)的交点
                sum = self.alphas[i] + self.alphas[j]
                self.alphas[i] -= delta
                self.alphas[j] += delta
                if sum > self.C:
                    if self.alphas[i] > self.C:
                        self.alphas[i] = self.C
                        self.alphas[j] = sum - self.C
                    if self.alphas[j] > self.C:
                        self.alphas[j] = self.C
                        self.alphas[i] = sum - self.C
                else:
                    if self.alphas[j] < 0:
                        self.alphas[j] = 0
                        self.alphas[i] = sum
                    if self.alphas[i] < 0:
                        self.alphas[i] = 0
                        self.alphas[j] = sum

            # update G
            delta_alpha_i = self.alphas[i] - old_alpha_i
            delta_alpha_j = self.alphas[j] - old_alpha_j
            for k in range(2 * self.l):
                self.G[:, k] += self.Q[i, k] * \
                    delta_alpha_i + self.Q[j, k] * delta_alpha_j

            # update G_bar and alpha_status
            ui = self.is_upper_bound(i)
            uj = self.is_upper_bound(j)
            self.update_alpha_status(i)
            self.update_alpha_status(j)
            if ui != self.is_upper_bound(i):
                if ui:
                    for k in range(self.l):
                        self.G_bar[:, k] -= self.C * self.Q[i, k]
                else:
                    for k in range(self.l):
                        self.G_bar[:, k] += self.C * self.Q[i, k]
            if uj != self.is_upper_bound(j):
                if uj:
                    for k in range(self.l):
                        self.G_bar[:, k] -= self.C * self.Q[j, k]
                else:
                    for k in range(self.l):
                        self.G_bar[:, k] += self.C * self.Q[j, k]

            #print('i:', i, ',j:', j, ',alpha_i:', self.alphas[i], ',alpha_j:', self.alphas[j])

        print('iter: ', iter, '\n', '=================')
        # caculate rho
        rho = self.caculate_rho()

        # caculate object value
        v = 0
        for i in range(2 * self.l):
            v += self.alphas[i] * (self.G[:, i] + self.p[:, i])
        obj = v / 2  # 目标值为(alpha^T*Q*alpha + p^T*alpha)

        return rho, obj

    def caculate_rho(self):
        nr_free = 0
        ub = float('inf')
        lb = -float('inf')
        sum_free = 0
        for i in range(2 * self.l):
            yG = self.sign[i] * self.G[:, i]
            if self.is_upper_bound(i):
                if self.sign[i] == -1:
                    ub = min(ub, yG)
                else:
                    lb = max(lb, yG)
            elif self.is_lower_bound(i):
                if self.sign[i] == +1:
                    ub = min(ub, yG)
                else:
                    lb = max(lb, yG)
            else:
                nr_free += 1
                sum_free += yG

        if nr_free > 0:
            r = sum_free / nr_free
        else:
            r = (ub + lb) / 2
        return r


def solve_epsilon_svr(dataMatIn, labelMat,
                      epsilon_svr, C, eps=1e-3, kernel_type='rbf',
                      degree=3, gamma=0.1, coef0=0.0):
    s = SVR_Q_Solve(dataMatIn, labelMat,
                    epsilon_svr, C, eps, kernel_type,
                    degree, gamma, coef0)
    rho, obj = s.solve()
    alpha = np.mat(np.zeros((s.l, 1)))
    for i in range(s.l):
        alpha[i] = s.alphas[i] - s.alphas[i + s.l]
    return alpha, rho, obj


def svm_train_decision(dataMatIn, labelMat,
                       epsilon_svr, C, eps=1e-3, kernel_type='rbf',
                       degree=3, gamma=0.1, coef0=0.0):
    alpha, rho, obj = solve_epsilon_svr(dataMatIn, labelMat,
                                        epsilon_svr, C, eps, kernel_type,
                                        degree, gamma, coef0)
    print('obj = ', obj, ', rho = ', rho)

    # output SVs
    # nSV代表支持向量的个数，nBSV表示处于边界的支持向量个数。
    nSV = 0
    nBSV = 0
    l = np.shape(dataMatIn)[0]
    for i in range(l):
        if np.abs(alpha[i]) > 0:
            nSV += 1
            if np.abs(alpha[i] >= C):
                nBSV += 1
    print('nSV=', nSV, '\t', 'nBSV=', nBSV)
    return alpha, rho


def svm_train(dataMatIn, labelMat,
              epsilon_svr, C, eps=1e-3, kernel_type='rbf',
              degree=3, gamma=0.1, coef0=0.0):
    svm_model = {
        'epsilon': epsilon_svr,  # 对应svm_parameter中的p
        'C': C,
        'eps': eps,
        'kernel_type': kernel_type,
        'degree': degree,
        'gamma': gamma,
        'coef0': coef0,
        'alpha': np.mat(np.zeros((np.shape(dataMatIn)[0], 1))),
        'rho': 0,
        'lSV': 0,  # 支持向量个数
        'SV': [],  # 支持向量
        'sv_coef': [],  # 支持向量系数
        'sv_indices': []  # 记录支持向量在训练数据中的index
    }
    svm_model['alpha'], svm_model['rho'] = svm_train_decision(dataMatIn, labelMat,
                                                              epsilon_svr, C, eps, kernel_type,
                                                              degree, gamma, coef0)
    nSV = 0
    for i in range(np.shape(dataMatIn)[0]):
        if np.abs(svm_model['alpha'][i]) > 0:
            nSV += 1
    svm_model['lSV'] = nSV

    svm_model['SV'] = []
    for i in range(nSV):
        svm_model['SV'].append(np.zeros((1, np.shape(dataMatIn)[1])))

    svm_model['sv_coef'] = np.mat(np.zeros((nSV, 1)))
    svm_model['sv_indices'] = np.mat(np.zeros((nSV, 1)))
    j = 0
    for i in range(np.shape(dataMatIn)[0]):
        if np.abs(svm_model['alpha'][i]) > 0:
            svm_model['SV'][j] = dataMatIn[i]
            svm_model['sv_coef'][j] = svm_model['alpha'][i]
            svm_model['sv_indices'][j] = i + 1
            j += 1
    return svm_model


def k_function(dataTrain_i, dataTest_i, svm_model):  # dataTrain,dataTest都是一个样本
    kernel_type = svm_model['kernel_type']
    degree = svm_model['degree']
    gamma = svm_model['gamma']
    coef0 = svm_model['coef0']
    if kernel_type == 'linear':
        return dataTrain_i * dataTest_i.T
    elif kernel_type == 'rbf':
        return np.exp(-gamma * (np.linalg.norm(dataTrain_i - dataTest_i) ** 2))
    elif kernel_type == 'poly':
        return (gamma * (dataTrain_i * dataTest_i.T) + coef0) ** degree
    elif kernel_type == 'sigmoid':
        return (np.tanh(gamma * dataTest_i * dataTrain_i.T) + coef0)
    elif kernel_type == 'laplacian':
        return np.exp(-gamma * np.linalg.norm(dataTest_i - dataTrain_i))
    elif kernel_type == 'morlet':
        morlet = 1
        for k in range(dataTrain_i.shape[1]):
            morlet *= (np.cos(1.75 * (dataTest_i[:, k] - dataTrain_i[:, k]) / coef0) * np.exp(- (
                np.linalg.norm(dataTest_i[:, k] - dataTrain_i[:, k]) ** 2) / (2 * coef0 ** 2)))
        return morlet
    else:
        raise NameError('Houston We Have a Problem -- \
            That Kernel is not recognized')


def svm_predict(svm_model, dataTest_i):
    sv_coef = svm_model['sv_coef']
    SV = svm_model['SV']
    rho = svm_model['rho']
    sum = 0
    for i in range(svm_model['lSV']):
        sum += sv_coef[i] * k_function(SV[i], dataTest_i, svm_model)
    sum -= rho
    return sum
