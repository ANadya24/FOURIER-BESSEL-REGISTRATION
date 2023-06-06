from scipy.interpolate import interp1d


class FunctionScale:

    def x_compress_function(self, x, y, a):
        x_new = x.copy() / a
        interp_func = interp1d(x_new, y, bounds_error=False, fill_value="extrapolate")
        return interp_func(x)

    def x_unclench_function(self, x, y, a):
        x_new = x.copy() * a
        interp_func = interp1d(x, y, bounds_error=False, fill_value="extrapolate")
        return interp_func(x_new)

    def interpolate_2_new_grid(self, x, y, new_x):
        f_interp = interp1d(x, y)
        return f_interp(new_x)

    def func_sqrtx(self, x, func):
        return func * x ** 0.5