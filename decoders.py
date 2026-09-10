import numpy as np
from scipy.optimize import newton


class MPPDecoder:
    def __init__(self, max_iter=1000, rtol=1e-3, newton_max_iter=50, newton_atol=1e-8):
        self.max_iter = int(max_iter)
        self.rtol = rtol

        # For updating state
        self.newton_max_iter = int(newton_max_iter)
        self.newton_atol = newton_atol

    def __call__(self, x0: float, vx: float, r: np.ndarray, r0: float, r1: float, vr: float):
        # Fixed parameters for MPP occurrence
        b0, b1 = self.get_mpp_occurrence_parameters(r)

        # Expectation-Maximization (EM) algorithm
        vx_prev, r0_prev, r1_prev, vr_prev = vx, r0, r1, vr
        for i in range(self.max_iter):
            # E-step: Estimate states with Kalman filtering and smoothing
            x_pred, v_pred, x_updt, v_updt = self.kalman_filter(x0, vx_prev, b0, b1, r, r0_prev, r1_prev, vr_prev)
            x_smth, v_smth = self.kalman_smoother(x_pred, v_pred, x_updt, v_updt)
            x0 = x_smth[0]  # Update initial state

            # M-step: Optimize parameters
            vx = self.get_state_noise_variance(v_pred, v_updt, x_smth, v_smth)
            r0, r1 = self.get_mpp_amplitude_parameters(x_smth, v_smth, r)
            vr = self.get_mpp_amplitude_noise_variance(x_smth, v_smth, r, r0, r1)

            # Print current parameters
            print(f"i: {i}, x0: {x0:.6f}, vx: {vx:.6f}, r0: {r0:.6f}, r1: {r1:.6f}, vr: {vr:.6f}")

            # Check feasibility
            if r1 <= 0:
                # Revert to previous parameters
                vx, r0, r1, vr = vx_prev, r0_prev, r1_prev, vr_prev
                status = "halted"
                break

            # Check convergence
            if np.allclose([vx, r0, r1, vr], [vx_prev, r0_prev, r1_prev, vr_prev], rtol=self.rtol):
                status = "converged"
                break

            # Update parameters
            vx_prev, r0_prev, r1_prev, vr_prev = vx, r0, r1, vr

        # If reached max iteration
        else:
            status = "terminated"

        # Remove initial state
        x_smth, v_smth = x_smth[1:], v_smth[1:]

        # Return results
        results = {
            "status": status,
            "iteration": i,
            "x_smth": x_smth,
            "v_smth": v_smth,
            "x0": x0,
            "vx": vx,
            "r0": r0,
            "r1": r1,
            "vr": vr,
        }
        return results

    def get_mpp_occurrence_parameters(self, r):
        p0 = np.sum(r > 0) / len(r)  # Baseline probability of MPP occurrence
        b0 = np.log(p0 / (1 - p0))  # Intercept of logistic function
        b1 = 1  # Rate of logistic function
        return b0, b1

    def kalman_filter(self, x0, vx, b0, b1, r, r0, r1, vr):
        K = len(r)

        x_pred = np.zeros(K)  # x_{k+1|k}
        x_updt = np.zeros(K + 1)  # x_{k|k}, +1 for initial state
        v_pred = np.zeros(K)  # v_{k+1|k}
        v_updt = np.zeros(K + 1)  # v_{k|k}, +1 for initial state

        x_updt[0] = x0  # Initial state x_{0|0}
        v_updt[0] = vx  # Initial variance v_{0|0}

        for k in range(K):
            # Predict
            x_pred[k] = x_updt[k]  # x_{k+1|k}
            v_pred[k] = v_updt[k] + vx  # v_{k+1|k}

            # Update
            x_updt[k + 1] = self.update_state(x_pred[k], v_pred[k], b0, b1, r[k], r0, r1, vr)  # x_{k+1|k+1}
            v_updt[k + 1] = self.update_variance(v_pred[k], x_updt[k + 1], b0, b1, r[k], r1, vr)  # v_{k+1|k+1}

        return x_pred, v_pred, x_updt, v_updt

    def kalman_smoother(self, x_pred, v_pred, x_updt, v_updt):
        K = len(x_pred)

        x_smth = np.zeros(K + 1)  # x_{k|K}, +1 for initial state
        v_smth = np.zeros(K + 1)  # v_{k|K}, +1 for initial state

        x_smth[K] = x_updt[K]  # x_{K|K}
        v_smth[K] = v_updt[K]  # v_{K|K}

        for k in range(K - 1, -1, -1):
            ak = v_updt[k] / v_pred[k]  # v_{k|k} / v_{k+1|k}
            x_smth[k] = x_updt[k] + ak * (x_smth[k + 1] - x_pred[k])  # x_{k|K}
            v_smth[k] = v_updt[k] + ak**2 * (v_smth[k + 1] - v_pred[k])  # v_{k|K}

        return x_smth, v_smth

    def get_mpp_amplitude_parameters(self, x_smth, v_smth, r):
        n = r > 0
        r = r[n]
        x = x_smth[1:][n]
        v = v_smth[1:][n]

        A = np.array([[len(r), np.sum(x)], [np.sum(x), np.sum(x**2 + v)]])
        B = np.array([np.sum(r), np.sum(r * x)])
        r0, r1 = np.linalg.solve(A, B)
        return r0, r1

    def get_mpp_amplitude_noise_variance(self, x_smth, v_smth, r, r0, r1):
        n = r > 0
        r = r[n]
        x = x_smth[1:][n]
        v = v_smth[1:][n]

        vr = np.sum(r**2 + r0**2 + r1**2 * (x**2 + v) - 2 * r0 * r - 2 * r1 * r * x + 2 * r0 * r1 * x) / len(r)
        return vr

    def get_state_noise_variance(self, v_pred, v_updt, x_smth, v_smth):
        a = v_updt[:-1] / v_pred  # v_{k|k} / v_{k+1|k}
        u = x_smth**2 + v_smth  # u_k
        u_ = x_smth[1:] * x_smth[:-1] + a * v_smth[1:]  # u_{k, k+1}
        vx = np.sum(u[1:] - 2 * u_ + u[:-1]) / (len(u) - 1)
        return vx

    def update_state(self, x_pred_k, v_pred_k, b0, b1, rk, r0, r1, vr):
        """
        Args:
            x_pred_k: x_{k+1|k}
            v_pred_k: v_{k+1|k}
        Returns:
            x_updt_k: x_{k+1|k+1}
        """
        # Newton-Raphson method
        ck = v_pred_k / (r1**2 * v_pred_k + vr)

        def f(x):
            pk = 1 / (1 + np.exp(-(b0 + b1 * x)))
            if rk > 0:
                return x_pred_k + ck * (vr * b1 * (1 - pk) + r1 * (rk - r0 - r1 * x_pred_k)) - x
            else:
                return x_pred_k - v_pred_k * b1 * pk - x

        def df(x):
            pk = 1 / (1 + np.exp(-(b0 + b1 * x)))
            if rk > 0:
                return -ck * vr * b1**2 * pk * (1 - pk) - 1
            else:
                return -v_pred_k * b1**2 * pk * (1 - pk) - 1

        x_updt_k = newton(f, x_pred_k, fprime=df, maxiter=self.newton_max_iter, tol=self.newton_atol)
        return x_updt_k

    def update_variance(self, v_pred_k, x_updt_k, b0, b1, rk, r1, vr):
        """
        Args:
            x_updt_k: x_{k+1|k+1}
            v_pred_k: v_{k+1|k}
        Returns:
            v_updt_k: v_{k+1|k+1}
        """
        pk = 1 / (1 + np.exp(-(b0 + b1 * x_updt_k)))
        if rk > 0:
            v_updt_k = 1 / (1 / v_pred_k + b1**2 * pk * (1 - pk) + r1**2 / vr)
        else:
            v_updt_k = 1 / (1 / v_pred_k + b1**2 * pk * (1 - pk))
        return v_updt_k


class MPPContDecoder(MPPDecoder):
    def __call__(
        self,
        x0: float,
        vx: float,
        r: np.ndarray,
        r0: float,
        r1: float,
        vr: float,
        s: np.ndarray,
        s0: float,
        s1: float,
        vs: float,
        s_lambda: float = 1,
        vs_stop: float = 0,
    ):
        # Fixed parameters for MPP occurrence
        b0, b1 = self.get_mpp_occurrence_parameters(r)

        # Expectation-Maximization (EM) algorithm
        vx_prev, r0_prev, r1_prev, vr_prev, s0_prev, s1_prev, vs_prev = vx, r0, r1, vr, s0, s1, vs
        for i in range(self.max_iter):
            # E-step: Estimate states with Kalman filtering and smoothing
            x_pred, v_pred, x_updt, v_updt = self.kalman_filter(
                x0, vx_prev, b0, b1, r, r0_prev, r1_prev, vr_prev, s, s0_prev, s1_prev, vs_prev
            )
            x_smth, v_smth = self.kalman_smoother(x_pred, v_pred, x_updt, v_updt)
            x0 = x_smth[0]  # Update initial state

            # M-step: Optimize parameters
            vx = self.get_state_noise_variance(v_pred, v_updt, x_smth, v_smth)
            r0, r1 = self.get_mpp_amplitude_parameters(x_smth, v_smth, r)
            vr = self.get_mpp_amplitude_noise_variance(x_smth, v_smth, r, r0, r1)
            s0, s1 = self.get_cont_parameters(x_smth, v_smth, s)
            vs = self.get_cont_noise_variance(x_smth, v_smth, s, s0, s1)

            # Prevent overfitting to continuous component
            s0 = s0_prev + s_lambda * (s0 - s0_prev)  # Update with exponential smoothing
            s1 = s1_prev + s_lambda * (s1 - s1_prev)
            vs = vs_prev + s_lambda * (vs - vs_prev)
            if vs < vs_stop:  # Stop updating if variance is too small
                s0 = s0_prev
                s1 = s1_prev
                vs = vs_prev

            # Print current parameters
            print(
                f"i: {i}, x0: {x0:.6f}, vx: {vx:.6f}, r0: {r0:.6f}, r1: {r1:.6f}, vr: {vr:.6f}, s0: {s0:.6f}, s1: {s1:.6f}, vs: {vs:.6f}"
            )

            # Check feasibility
            if r1 <= 0 or s1 <= 0:
                # Revert to previous parameters
                vx, r0, r1, vr, s0, s1, vs = vx_prev, r0_prev, r1_prev, vr_prev, s0_prev, s1_prev, vs_prev
                status = "halted"
                break

            # Check convergence
            if np.allclose(
                [vx, r0, r1, vr, s0, s1, vs],
                [vx_prev, r0_prev, r1_prev, vr_prev, s0_prev, s1_prev, vs_prev],
                rtol=self.rtol,
            ):
                status = "converged"
                break

            # Update parameters
            vx_prev, r0_prev, r1_prev, vr_prev, s0_prev, s1_prev, vs_prev = vx, r0, r1, vr, s0, s1, vs

        # If reached max iteration
        else:
            status = "terminated"

        # Remove initial state
        x_smth, v_smth = x_smth[1:], v_smth[1:]

        # Return results
        results = {
            "status": status,
            "iteration": i,
            "x_smth": x_smth,
            "v_smth": v_smth,
            "x0": x0,
            "vx": vx,
            "r0": r0,
            "r1": r1,
            "vr": vr,
            "s0": s0,
            "s1": s1,
            "vs": vs,
        }
        return results

    def kalman_filter(self, x0, vx, b0, b1, r, r0, r1, vr, s, s0, s1, vs):
        K = len(r)

        x_pred = np.zeros(K)  # x_{k+1|k}
        x_updt = np.zeros(K + 1)  # x_{k|k}, +1 for initial state
        v_pred = np.zeros(K)  # v_{k+1|k}
        v_updt = np.zeros(K + 1)  # v_{k|k}, +1 for initial state

        x_updt[0] = x0  # Initial state x_{0|0}
        v_updt[0] = vx  # Initial variance v_{0|0}

        for k in range(K):
            # Predict
            x_pred[k] = x_updt[k]  # x_{k+1|k}
            v_pred[k] = v_updt[k] + vx  # v_{k+1|k}

            # Update
            x_updt[k + 1] = self.update_state(
                x_pred[k], v_pred[k], b0, b1, r[k], r0, r1, vr, s[k], s0, s1, vs
            )  # x_{k+1|k+1}
            v_updt[k + 1] = self.update_variance(v_pred[k], x_updt[k + 1], b0, b1, r[k], r1, vr, s1, vs)  # v_{k+1|k+1}

        return x_pred, v_pred, x_updt, v_updt

    def get_cont_parameters(self, x_smth, v_smth, s):
        x = x_smth[1:]
        v = v_smth[1:]

        A = np.array([[len(s), np.sum(x)], [np.sum(x), np.sum(x**2 + v)]])
        B = np.array([np.sum(s), np.sum(s * x)])
        r0, r1 = np.linalg.solve(A, B)
        return r0, r1

    def get_cont_noise_variance(self, x_smth, v_smth, s, s0, s1):
        x = x_smth[1:]
        v = v_smth[1:]

        vs = np.sum(s**2 + s0**2 + s1**2 * (x**2 + v) - 2 * s0 * s - 2 * s1 * s * x + 2 * s0 * s1 * x) / len(s)
        return vs

    def update_state(self, x_pred_k, v_pred_k, b0, b1, rk, r0, r1, vr, sk, s0, s1, vs):
        """
        Args:
            x_pred_k: x_{k+1|k}
            v_pred_k: v_{k+1|k}
        Returns:
            x_updt_k: x_{k+1|k+1}
        """
        # Newton-Raphson method
        ck_1 = v_pred_k / ((r1**2 * vs + s1**2 * vr) * v_pred_k + vr * vs)
        ck_0 = v_pred_k / (s1**2 * v_pred_k + vs)

        def f(x):
            pk = 1 / (1 + np.exp(-(b0 + b1 * x)))
            if rk > 0:
                ck = ck_1
                return (
                    x_pred_k
                    + ck
                    * (
                        vr * vs * b1 * (1 - pk)
                        + r1 * vs * (rk - r0 - r1 * x_pred_k)
                        + s1 * vr * (sk - s0 - s1 * x_pred_k)
                    )
                    - x
                )
            else:
                ck = ck_0
                return x_pred_k + ck * (-vs * b1 * pk + s1 * (sk - s0 - s1 * x_pred_k)) - x

        def df(x):
            pk = 1 / (1 + np.exp(-(b0 + b1 * x)))
            if rk > 0:
                ck = ck_1
                return -ck * vr * vs * b1**2 * pk * (1 - pk) - 1
            else:
                ck = ck_0
                return -ck * vs * b1**2 * pk * (1 - pk) - 1

        x_updt_k = newton(f, x_pred_k, fprime=df, maxiter=self.newton_max_iter, tol=self.newton_atol)
        return x_updt_k

    def update_variance(self, v_pred_k, x_updt_k, b0, b1, rk, r1, vr, s1, vs):
        """
        Args:
            x_updt_k: x_{k+1|k+1}
            v_pred_k: v_{k+1|k}
        Returns:
            v_updt_k: v_{k+1|k+1}
        """
        pk = 1 / (1 + np.exp(-(b0 + b1 * x_updt_k)))
        if rk > 0:
            v_updt_k = 1 / (1 / v_pred_k + b1**2 * pk * (1 - pk) + r1**2 / vr + s1**2 / vs)
        else:
            v_updt_k = 1 / (1 / v_pred_k + b1**2 * pk * (1 - pk) + s1**2 / vs)
        return v_updt_k
