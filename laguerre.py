from typing import List, Any
import numpy as np
from tqdm import tqdm
import time

class Laguerre:
    def __init__(self):
        self._maxX = 10
        self._curmaxX = None
        self._curmaxXHier = []
        self._length = 101
        self.x = []
        self.f = None
        self.baseline = []
        self.alpha = None
        # коэффициенты простого разложения
        self.coefs = []
        # просто эрмиты
        self.laguerres = []
        # коэффициенты разложения с автоматич. масштабом
        self.autoScaleCoefs = []
        # растянутые под сигнал эрмиты
        self.autoScaleLaguerres = []
        # разница между исходной функцией и суммой autoScale, т.е. погрешность
        self.autoScaleResidue = None

    #     def initialize(self, maxX: float, funcToDecompose: List[float], alpha: int):
    #         self.initialize_parameters(funcToDecompose, alpha)
    #         self._maxX = maxX
    #         self.initX(self._maxX, alpha)

    #     def initialize_parameters(self, funcToDecompose: List[float], alpha: int):
    #         self._maxX = 10.
    #         self._length = len(funcToDecompose)
    #         self.alpha = alpha

    #         self.x = []
    #         self.initX(self._maxX, alpha)

    #         self.baseline = []
    #         self.f = []
    #         assert self._length is not None
    #         assert funcToDecompose is not None
    #         for i in range(self._length):
    #             self.baseline.append(funcToDecompose[i] + i * (
    #                     funcToDecompose[self._length - 1] - funcToDecompose[0]) / float(self._length))
    #             self.f.append(funcToDecompose[i] - self.baseline[i])

    #     def initX(self, maxX: float, alpha: int):
    #         step = maxX / (self._length - 1)
    #         fi_n = []
    #         fi_n1 = []
    #         fi_n2 = []
    #         exp = []

    #         for i in range(self._length):
    #             self.x.append(i * step)
    #             print(alpha, self.x[i])
    #             exp.append(np.exp(-self.x[i] / 2.) * self.x[i] ** (alpha / 2.))
    #             fi_n2.append(exp[i])
    #             fi_n1.append(exp[i] * (1 + alpha - self.x[i]))
    #             fi_n.append((fi_n1[i] * (2 * 1 + 1 + alpha - self.x[i]) - fi_n2[i] * (1 + alpha)) / (1 + 1))

    #         self.laguerres = []
    #         self.laguerres.extend(fi_n2)
    #         self.laguerres.extend(fi_n1)
    #         self.laguerres.extend(fi_n)

    def create_functions(self, max_number, alpha, x, a) -> List[Any]:
        """
        рассчитывает систему функций Лагерра для фиксированного α и n = 0, . . . , N и
        возвращает рассчитанные функции в виде массива массивов
        :param max_number: задает количество функций N
        :param alpha: порядок преобразования Ганкеля
        :param x: сетка для расчетов
        :param a: параметр масштабирования сетки
        :return:
        """
        if max_number < 0:
            return None

        #         if len(x) != self._length:
        #             print('Warning! lengths are not equal!', self._length, len(x))
        #             self._length = len(x)

        functions = [[]] * max_number
        norm = None

        if max_number >= 0:
            norm = 1
            for i in range(1, alpha + 1):
                norm *= i
            norm = (1 / norm) ** 0.5


            for i in range(len(x)):
                if alpha < 0:
                    mul_alpha = 1 / ((a * x[i]) ** (-alpha / 2) + 1e-7)
                else:
                    mul_alpha = (a * x[i]) ** (alpha / 2)
                if np.isnan(norm * np.exp(-a * x[i] / 2) * mul_alpha):
                    print(mul_alpha, np.exp(-a * x[i] / 2), norm)
                functions[0].append(norm * np.exp(-a * x[i] / 2) * mul_alpha)

        if max_number == 1:
            return functions

        if max_number >= 1:
            norm = (1 / (alpha + 1)) ** 0.5

            functions[1] = []
            for i in range(len(x)):
                functions[1].append(norm * (1 + alpha - a * x[i]) * functions[0][i])

        if max_number == 2:
            return functions

        for j in range(2, max_number):
            functions[j] = []
            for i in range(len(x)):
                functions[j].append(
                    functions[j - 1][i] * (-x[i] * a + 2 * j - 1 + alpha) / (j * (j + alpha)) ** 0.5 -
                    functions[j - 2][i] * ((j - 1) * (alpha + j - 1) / (j * (j + alpha))) ** 0.5)

        return functions

    def create_functions_x2_2sqrtx(self, max_number: int, alpha: int, x: List[float], a: float) -> List[Any]:
        """
        рассчитывает систему собственных функций преобразования Ганкеля
        :param max_number: задает количество функций N
        :param alpha: порядок преобразования Ганкеля
        :param x: сетка для расчетов
        :param a: параметр масштабирования сетки
        :return:
        """

        if max_number < 0:
            return None
        x2 = []
        for i in range(len(x)):
            x2.append(x[i] * x[i])

        functions = self.create_functions(max_number, alpha, x2, a * a)

        for m in range(max_number):
            for i in range(len(x)):
                functions[m][i] *= (2 * a * x[i]) ** 0.5

        return functions

    def transform_forward(self, numberOfFunctions: int, data: List[Any],
                          lag: List[List[float]], x: List[float], a: float) -> List[float]:
        """
        Реализует расчет коэффициентов разложения функции, заданной параметром data,
        в ряд по системе функций, заданной параметром lag, и возвращает их в виде массива
        :param numberOfFunctions: число членов разложения
        :param data: функция, которую необходимо разложить на коэффициенты
        :param lag: система функций Лагерра
        :param x: сетка
        :param a: параметр масштабирования сетки
        :return: коэффициенты разложения
        """
        coefs = []
        dx = a * (x[-1] - x[0]) / (len(x) - 1)
        for n in range(numberOfFunctions):
            coefs.append((np.array(lag[n]) * data).sum() * dx)
        return coefs

    def transform_backward(self, numberOfFunctions: int, coefs: List[float],
                           lag: List[List[float]]) -> List[float]:
        """
        реализует восстановление функции, возвращаемой в виде массива
        :param numberOfFunctions: число членов разложения
        :param coefs: коэффициенты разложения
        :param lag: система функций Лагерра
        :return: восстановленная функция
        """

        res = []
        for _ in range(len(lag[0])):
            res.append(0)

        for n in range(numberOfFunctions):
            if n % 2 == 1:
                cur_coef = -coefs[n]
            else:
                cur_coef = coefs[n]
            for x in range(len(lag[0])):
                res[x] += cur_coef * lag[n][x]
        return res

    def lag_function(self, number, alpha, x):
        exp = np.exp(-x / (2.0 * (number + 1)))
        ln_m1 = 0
        ln = exp

        if number == 0:
            return np.exp(-x / 2)

        n = 0

        while n < number:
            ln_p1 = (exp * ln * (2 * n + 1 + alpha - x) - exp * exp * ln_m1 * (n + alpha)) / (n + 1)
            ln_m1 = ln
            ln = ln_p1
            n += 1
        return ln

    def laguerre_zeros1(self, n, alpha):
        zeros = np.zeros(n)
        x1 = 0
        epsilon = 4.94065645841247e-324

        for roots in range(n):
            if (roots / n) < 0.1:
                x1 = 1e-6
            cond = True
            while cond:
                x0 = x1
                lag = self.create_functions(n + 1, alpha, [x0], 1)
                l = lag[n][0]
                ln_1 = lag[n - 1][0]
                l_sharp = (n * l - (n + alpha) * ln_1) / (x0 + epsilon)
                l_sum = 0
                for i in range(roots):
                    l_sum += 1 / (x0 - zeros[i] + epsilon)

                x1 = x0 - l / (l_sharp - l * l_sum + epsilon)
                cond = abs(x0 - x1) > 5e-16 * (1 + abs(x1)) * 100

            zeros[roots] = x1
            if np.isnan(x1):
                print(x0, x1, l, l_sharp, l_sum, epsilon)
            x1 *= 0.5

        return zeros

    def laguerre_zeros(self, n, alpha, abort_after=1):
        zeros = np.zeros(n)
        x1 = 0
        epsilon = np.finfo('float64').resolution

        for roots in range(n):
            if (roots / n) < 0.1:
                x1 = 1e-6
            cond = True
            start = time.time()
            while cond:
                delta = time.time() - start
                if delta >= abort_after:
                    break
                x0 = x1
                l = self.lag_function(n, alpha, x0)
                ln_1 = self.lag_function(n - 1, alpha, x0)

                if np.isnan(l):
                    l = 0.
                if np.isnan(ln_1):
                    ln_1 = 0.

                l_sharp = (n * l - (n + alpha) * ln_1) / (x0 + epsilon)
                l_sum = 0
                for i in range(roots):
                    l_sum += 1 / (x0 - zeros[i] + epsilon)

                x1 = x0 - l / (l_sharp - l * l_sum + epsilon)
                cond = abs(x0 - x1) > 5e-16 * (1 + abs(x1)) * 100
            zeros[roots] = x1
            x1 *= 0.5

        return zeros

    def transform_forward_fast_x2sqrtx_laguerre_quad(self, laguerre_zeros, n_coef, alpha, data, x):
        quadrature_order = len(laguerre_zeros)
        mu = np.zeros(n_coef * quadrature_order)

        lag = self.create_functions(quadrature_order + 2, alpha, laguerre_zeros, 1)
        ksi = lag[quadrature_order + 1]
        laguerre_func = self.create_functions(n_coef, alpha, laguerre_zeros, 1)

        for j in range(n_coef):
            for i in range(quadrature_order):
                mu[quadrature_order * j + i] = (laguerre_zeros[i]) ** 0.75\
                                               * laguerre_func[j][i] / (ksi[i] * ksi[i] + 1e-7)

        data_interpolated = np.zeros(quadrature_order)

        #         sqrtZeros = np.zeros(quadrature_order)
        #         for i in range(quadrature_order):
        #             sqrtZeros[i] = laguerre_zeros[i] ** 0.5
        sqrtZeros = laguerre_zeros ** 0.5

        index = 1
        for i in range(quadrature_order):
            while (index < len(data) - 1) and (x[index] <= sqrtZeros[i]):
                index += 1

            if index < len(data):
                data_interpolated[i] = ((x[index] - sqrtZeros[i]) * data[index - 1] + (
                        sqrtZeros[i] - x[index - 1]) * data[index]) / (x[index] - x[index - 1])

        laguerre_coeffs = np.zeros(n_coef)

        for j in range(n_coef):
            total_sum = 0
            for i in range(quadrature_order):
                total_sum += data_interpolated[i] * mu[quadrature_order * j + i]
            laguerre_coeffs[j] = total_sum / (2 ** 0.5 * (quadrature_order + 1) * (quadrature_order + alpha + 1))

        return laguerre_coeffs
