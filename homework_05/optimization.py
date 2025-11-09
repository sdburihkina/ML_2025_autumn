import numpy as np
from numpy.linalg import LinAlgError
import scipy
from scipy.optimize import line_search
from scipy.linalg import cho_factor, cho_solve
from datetime import datetime
from collections import defaultdict




class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
         # TODO: Implement line search procedures for Armijo, Wolfe and Constant steps.
        if self._method == 'Wolfe':
                alpha = previous_alpha if previous_alpha is not None else self.alpha_0
                
                # Пробуем найти шаг по Вольфе
                wolfe_result = line_search(oracle.func, oracle.grad, x_k, d_k, c1=self.c1, c2=self.c2)
                if wolfe_result[0] is not None:
                    return wolfe_result[0]
                
                # Иначе Армихо
                alpha = previous_alpha if previous_alpha is not None else self.alpha_0
                while oracle.func_directional(x_k, d_k, alpha) > oracle.func(x_k) + self.c1 * alpha * np.dot(oracle.grad(x_k), d_k):
                    alpha /= 2
                return alpha

        elif self._method == 'Armijo':
                alpha = previous_alpha if previous_alpha is not None else self.alpha_0
                while oracle.func_directional(x_k, d_k, alpha) > oracle.func(x_k) + self.c1 * alpha * np.dot(oracle.grad(x_k), d_k):
                    alpha /= 2
                return alpha

        elif self._method == 'Constant':
                return self.c if previous_alpha is None else previous_alpha

        return None            
      

def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """def gradient_descent
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    # Инициализация истории оптимизации 
    history = defaultdict(list) if trace else None
        # Получение инструмента для поиска шага
    line_search_tool = get_line_search_tool(line_search_options)
        # Начальная точка
    x_k = np.copy(x_0)
    # TODO: Implement gradient descent
    # Use line_search_tool.line_search() for adaptive step size.
        # Засекаем время начала оптимизации
    start_time = datetime.now()

        # Используем line_search_tool.line_search() для адаптивного выбора шага.
    for i in range(max_iter):
        # Выполняем шаг градиентного спуска: x_{k+1} = x_k - alpha * grad(f(x_k))
      x_k = x_k - line_search_tool.line_search(oracle, x_k, -oracle.grad(x_k))

       
    if history:
            # Сохраняем время с начала оптимизации
            history['time'].append((datetime.now() - start_time).total_seconds())
            # Сохраняем значение функции
            history['func'].append(oracle.func(x_k))
            # Сохраняем норму градиента
            history['grad_norm'].append(np.linalg.norm(oracle.grad(x_k)))
            # Для 1D и 2D случаев сохраняем координаты точки
            if x_k.size <= 2:
                history['x'].append(x_k)

        # Вывод информации о текущей итерации 
    if display:
            print('Итерация: {}, x_k: {}, f(x_k): {}, ||g(x_k)||: {}'.format(
                i, x_k, oracle.func(x_k), np.linalg.norm(oracle.grad(x_k))))

        # Критерий остановки: норма градиента стала достаточно малой
    if np.linalg.norm(oracle.grad(x_k))**2 < (tolerance * np.linalg.norm(oracle.grad(x_0))**2):
            if display:
                print('norm(grad(x_k))^2 < epsilon * norm(grad(x_0))^2!')
                print('Количество итераций: {}'.format(i))
            return x_k, 'success', history

        # Проверка достижения максимального числа итераций
    if i == max_iter - 1:
            if display:
                print('Достигнуто максимальное число итераций!')
                print('Количество итераций: {}'.format(i))
            return x_k, 'iterations_exceeded', history

        # Возврат в случае выхода из цикла 
    return x_k, 'computational_error', history

   


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    # Инициализация истории оптимизации 
    history = defaultdict(list) if trace else None
    # Получение инструмента для поиска шага
    line_search_tool = get_line_search_tool(line_search_options)
    # Начальная точка
    x_k = np.copy(x_0)
    # TODO: Implement Newton's method.
    # Use line_search_tool.line_search() for adaptive step size.
    # Засекаем время начала оптимизации
    start_time = datetime.now()

    # Основной цикл оптимизации (метод Ньютона)
    for i in range(max_iter):
        # Вычисление направления Ньютона: H(x_k)^{-1} * grad(f(x_k))
        # cho_solve решает систему линейных уравнений с использованием разложения Холецкого
        newton_direction = scipy.linalg.cho_solve(oracle.hess(x_k), oracle.grad(x_k))
        
        # Выполняем шаг метода Ньютона: x_{k+1} = x_k - alpha * H^{-1} * grad(f(x_k))
        # line_search адаптивно выбирает размер шага alpha
        x_k = x_k - line_search_tool.line_search(oracle, x_k, -newton_direction)
        
        # Сохранение истории оптимизации 
        if history:
            # Время с начала оптимизации
            history['time'].append((datetime.now() - start_time).total_seconds())
            # Значение целевой функции
            history['func'].append(oracle.func(x_k))
            # Норма градиента
            history['grad_norm'].append(np.linalg.norm(oracle.grad(x_k)))
            # Координаты точки (для 1D и 2D случаев)
            if x_k.size <= 2:
                history['x'].append(x_k)
        
        # Вывод информации о текущей итерации
        if display:
            print('Итерация: {}, x_k: {}, f(x_k): {}, ||g(x_k)||: {}'.format(
                i, x_k, oracle.func(x_k), np.linalg.norm(oracle.grad(x_k))))
        
        # Проверка критерия остановки: норма градиента стала достаточно малой
        grad_norm_squared = np.linalg.norm(oracle.grad(x_k)) ** 2
        initial_grad_norm_squared = np.linalg.norm(oracle.grad(x_0)) ** 2
        
        if grad_norm_squared < tolerance * initial_grad_norm_squared:
            if display:
                print('Условие сходимости выполнено: norm(grad(x_k))^2 < epsilon * norm(grad(x_0))^2!')
                print('Всего итераций: {}'.format(i))
            return x_k, 'success', history
        
        # Проверка достижения максимального числа итераций
        if i == max_iter - 1:
            if display:
                print('Достигнуто максимальное число итераций!')
                print('Всего итераций: {}'.format(i))
            return x_k, 'iterations_exceeded', history

    # Резервный возврат 
    return x_k, 'computational_error', history