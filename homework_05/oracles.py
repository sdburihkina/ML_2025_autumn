import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
       # TODO: Implement
       # Вычисление значения функции логистической регрессии с регуляризацией
    # f(x) = (1/m) * Σ log(1 + exp(-b_i * A_i x)) + (λ/2) * ||x||^2
    
    # Вычисляем аргументы для logsumexp: [0, -b_i * A_i x]
    # Это эквивалентно log(1 + exp(-b_i * A_i x)) через logsumexp
      logsumexp_args = np.array([np.zeros(len(self.b)), -self.b * self.matvec_Ax(x)])
    
    # Вычисляем logsumexp по axis=0 и усредняем по всем примерам
      logistic_loss = np.logsumexp(logsumexp_args, axis=0).mean()
    
    # Добавляем L2-регуляризацию
      regularization_term = 0.5 * self.regcoef * np.dot(x, x)
    
      return logistic_loss + regularization_term

    def grad(self, x):
       # TODO: Implement
        # Вычисление градиента функции логистической регрессии
    # ∇f(x) = -A^T(b * σ(-b * Ax)) / m + λx
    
    # Вычисляем вероятности: σ(-b_i * A_i x) = 1 / (1 + exp(b_i * A_i x))
      probabilities = expit(-self.b * self.matvec_Ax(x))
    
    # Вычисляем вектор весов для градиента: b * σ(-b * Ax)
      weight_vector = self.b * probabilities
    
    # Вычисляем градиент функции потерь: -A^T(b * σ(-b * Ax)) / m
      loss_gradient = -self.matvec_ATx(weight_vector) / len(self.b)
    
    # Добавляем градиент регуляризатора: λx
      regularization_gradient = self.regcoef * x
    
      return loss_gradient + regularization_gradient
    def hess(self, x):
         # TODO: Implement
            # Вычисление гессиана (матрицы вторых производных)
        # H(x) = A^T * diag(s) * A / m + λI
        # где s_i = σ(-b_i * A_i x) * σ(b_i * A_i x) = σ(z_i) * (1 - σ(z_i)), z_i = -b_i * A_i x
        
        # Вычисляем диагональные элементы гессиана:
        # s_i = σ(-b_i * A_i x) * σ(b_i * A_i x) = σ(z_i) * σ(-z_i)
        z = -self.b * self.matvec_Ax(x)
        diagonal_elements = expit(z) * expit(-z)  # σ(z) * (1 - σ(z))
        
        # Вычисляем гессиан функции потерь: A^T * diag(s) * A / m
        loss_hessian = self.matmat_ATsA(diagonal_elements) / len(self.b)
        
        # Добавляем гессиан регуляризатора: λI
        regularization_hessian = self.regcoef * np.eye(len(x))
        
        return loss_hessian + regularization_hessian


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """
def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
    # Инициализация кэшей для произведений матриц на векторы
    self.Ax = None  # Кэш для A*x
    self.Ad = None  # Кэш для A*d
    # Вызов конструктора родительского класса
    super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

def func_directional(self, x, d, alpha):
     # TODO: Implement optimized version with pre-computation of Ax and Ad
    # Вычисление функции вдоль направления: f(x + alpha * d)
    
    # Кэшируем A*x если еще не вычислено
    if self.Ax is None:
        self.Ax = self.matvec_Ax(x)
    # Кэшируем A*d если еще не вычислено  
    if self.Ad is None:
        self.Ad = self.matvec_Ax(d)
    
    # Вычисляем A*(x + alpha*d) = A*x + alpha*A*d
    Ax_alpha_Ad = self.Ax + alpha * self.Ad
    
    # Вычисляем логистическую функцию потерь:
    # log(1 + exp(-b_i * A_i(x + alpha * d))) через logsumexp
    logsumexp_args = np.array([np.zeros(len(self.b)), -self.b * Ax_alpha_Ad])
    logistic_loss = np.logsumexp(logsumexp_args, axis=0).mean()
    
    # Вычисляем регуляризационный член: 0.5 * λ * ||x + alpha*d||^2
    x_alpha_d = x + alpha * d
    regularization = 0.5 * self.regcoef * np.dot(x_alpha_d, x_alpha_d)
    
    return logistic_loss + regularization

def grad_directional(self, x, d, alpha):
     # TODO: Implement optimized version with pre-computation of Ax and Ad
    # Вычисление производной по направлению: ⟨∇f(x + alpha*d), d⟩
    
    # Кэшируем A*x и A*d если еще не вычислено
    if self.Ax is None:
        self.Ax = self.matvec_Ax(x)
    if self.Ad is None:
        self.Ad = self.matvec_Ax(d)
    
    # Вычисляем A*(x + alpha*d) = A*x + alpha*A*d
    Ax_alpha_Ad = self.Ax + alpha * self.Ad
    
    # Вычисляем производную от функции потерь по направлению:
    # ⟨-A^T(b * σ(-b * A(x+αd)))/m, d⟩ = -⟨b * σ(-b * A(x+αd)), Ad⟩/m
    probabilities = expit(-self.b * Ax_alpha_Ad)
    data_derivative = -np.dot(self.b * probabilities, self.Ad) / len(self.b)
    
    # Вычисляем производную от регуляризатора по направлению:
    # ⟨λ(x + αd), d⟩
    x_alpha_d = x + alpha * d
    reg_derivative = self.regcoef * np.dot(x_alpha_d, d)
    
    return data_derivative + reg_derivative

def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    
# Определение матрично-векторных операций для oracle
# Умножение матрицы A на вектор x: A * x
    matvec_Ax = lambda x: A @ x  # TODO: Implement

    # Умножение транспонированной матрицы A^T на вектор x: A^T * x  
    matvec_ATx = lambda x: A.T @ x  # TODO: Implement

    # Умножение: A^T * diag(s) * A (для вычисления гессиана)
    def matmat_ATsA(s):
         # TODO: Implement
        return A.T @ (s[:, np.newaxis] * A)  # s - диагональная матрица в виде вектора

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    else:
        raise 'Unknown oracle_type=%s' % oracle_type
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)



def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # TODO: Implement numerical estimation of the gradient
       # Инициализация вектора градиента (той же размерности, что и x)
    grad = np.zeros_like(x)
    # Вычисление значения функции в исходной точке
    f_x = func(x)

    # Вычисление частных производных по каждой координате
    for i in range(x.size):
        # Создаем вектор возмущения: нулевой вектор с eps в i-ой компоненте
        dx = np.zeros_like(x)
        dx.flat[i] = eps
        
        # Формула конечной разности для частной производной:
        # ∂f/∂x_i ≈ [f(x + eps*e_i) - f(x)] / eps
        grad.flat[i] = (func(x + dx) - f_x) / eps

    return grad


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # TODO: Implement numerical estimation of the Hessian
    n = x.size  # Размерность пространства
    hess = np.zeros((n, n))  # Инициализация матрицы гессиана
    f_x = func(x)  # Значение функции в точке x

    # Предварительно вычисляем f(x + eps * e_i) для всех координат
    # Это оптимизация для избежания повторных вычислений
    f_plus_i = np.zeros(n)
    for i in range(n):
        # Создаем вектор возмущения по i-ой координате
        dx = np.zeros_like(x)
        dx.flat[i] = eps
        f_plus_i[i] = func(x + dx)

    # Вычисляем вторые производные (гессиан)
    for i in range(n):
        # Вектор возмущения по i-ой координате
        dx_i = np.zeros_like(x)
        dx_i.flat[i] = eps

        for j in range(i, n):  # Используем симметрию гессиана
            # Вектор возмущения по j-ой координате
            dx_j = np.zeros_like(x)
            dx_j.flat[j] = eps

            # Вычисляем f(x + eps*e_i + eps*e_j)
            f_plus_ij = func(x + dx_i + dx_j)
            
            # Формула для смешанной производной второго порядка:
            # ∂²f/∂x_i∂x_j ≈ [f(x+eps_i+eps_j) - f(x+eps_i) - f(x+eps_j) + f(x)] / eps²
            hess_ij = (f_plus_ij - f_plus_i[i] - f_plus_i[j] + f_x) / (eps**2)

            # Заполняем симметричную матрицу
            hess[i, j] = hess_ij
            if i != j:
                hess[j, i] = hess_ij  # Симметрия гессиана

    return hess