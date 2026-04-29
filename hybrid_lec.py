import numpy as np
import matplotlib.pyplot as plt

def prepare_curve(loss, aep):
    loss = np.asarray(loss, dtype=float)
    aep = np.asarray(aep, dtype=float)

    mask = np.isfinite(loss) & np.isfinite(aep) & (loss >= 0) & (aep > 0)
    loss, aep = loss[mask], aep[mask]

    idx = np.argsort(-aep)
    loss, aep = loss[idx], aep[idx]

    uniq_aep = np.unique(aep)[::-1]
    uniq_loss = np.array([loss[aep == p].max() for p in uniq_aep])
    return uniq_loss, uniq_aep


def interp_loglog(x_new, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)

    mask = (x > 0) & (y > 0)
    x, y = x[mask], y[mask]

    idx = np.argsort(x)
    x, y = x[idx], y[idx]

    return np.exp(np.interp(np.log(x_new), np.log(x), np.log(y)))


def enforce_strict_monotonicity(loss, aep, rel_eps_loss=1e-10, rel_eps_aep=1e-10):
    idx = np.argsort(loss)
    loss = np.asarray(loss, dtype=float)[idx]
    aep = np.asarray(aep, dtype=float)[idx]

    uniq_loss = np.unique(loss)
    aep = np.array([aep[loss == L].max() for L in uniq_loss])
    loss = uniq_loss.copy()

    for i in range(1, len(loss)):
        if loss[i] <= loss[i - 1]:
            loss[i] = loss[i - 1] * (1 + rel_eps_loss)

    for i in range(1, len(aep)):
        if aep[i] >= aep[i - 1]:
            aep[i] = aep[i - 1] * (1 - rel_eps_aep)

    mask = (loss >= 0) & (aep > 0)
    return loss[mask], aep[mask]


def run_hybrid_lec(
    emp_loss,
    emp_aep,
    tail_loss,
    tail_aep,
    n_candidates=500,
    w_level=3.0,
    w_slope=1.0,
    w_curv=0.5,
    blend_ratio=1.5,
    n_blend=80,
    enforce_monotonic=True,
):
    emp_loss, emp_aep = prepare_curve(emp_loss, emp_aep)
    tail_loss, tail_aep = prepare_curve(tail_loss, tail_aep)

    p_lo = max(emp_aep.min(), tail_aep.min())
    p_hi = min(emp_aep.max(), tail_aep.max())

    if p_lo < p_hi:
        p_grid = np.geomspace(p_lo, p_hi, n_candidates)

        emp_grid = interp_loglog(p_grid, emp_aep[::-1], emp_loss[::-1])
        tail_grid = interp_loglog(p_grid, tail_aep[::-1], tail_loss[::-1])

        logp = np.log(p_grid)
        emp_slope = np.gradient(np.log(emp_grid), logp)
        tail_slope = np.gradient(np.log(tail_grid), logp)
        emp_curv = np.gradient(emp_slope, logp)
        tail_curv = np.gradient(tail_slope, logp)

        score = (
            w_level * np.abs(np.log(emp_grid) - np.log(tail_grid))
            + w_slope * np.abs(emp_slope - tail_slope)
            + w_curv * np.abs(emp_curv - tail_curv)
        )

        join_aep = p_grid[np.argmin(score)]
        aep_hi = min(p_hi, join_aep * blend_ratio)
        aep_lo = max(p_lo, join_aep / blend_ratio)

        if aep_lo >= aep_hi:
            join_loss = 0.5 * (
                interp_loglog([join_aep], emp_aep[::-1], emp_loss[::-1])[0]
                + interp_loglog([join_aep], tail_aep[::-1], tail_loss[::-1])[0]
            )
            hybrid_loss = np.concatenate((
                emp_loss[emp_aep > join_aep],
                [join_loss],
                tail_loss[tail_aep < join_aep],
            ))
            hybrid_aep = np.concatenate((
                emp_aep[emp_aep > join_aep],
                [join_aep],
                tail_aep[tail_aep < join_aep],
            ))
        else:
            p_blend = np.geomspace(aep_lo, aep_hi, n_blend)
            emp_blend = interp_loglog(p_blend, emp_aep[::-1], emp_loss[::-1])
            tail_blend = interp_loglog(p_blend, tail_aep[::-1], tail_loss[::-1])

            t = (np.log(p_blend) - np.log(p_blend[0])) / (np.log(p_blend[-1]) - np.log(p_blend[0]))
            w = np.clip(t, 0.0, 1.0)
            w = w * w * (3 - 2 * w)

            blend_loss = np.exp((1 - w) * np.log(tail_blend) + w * np.log(emp_blend))

            hybrid_loss = np.concatenate((
                emp_loss[emp_aep > aep_hi],
                blend_loss,
                tail_loss[tail_aep < aep_lo],
            ))
            hybrid_aep = np.concatenate((
                emp_aep[emp_aep > aep_hi],
                p_blend,
                tail_aep[tail_aep < aep_lo],
            ))
    else:
        emp_end_loss, emp_end_aep = emp_loss[-1], emp_aep[-1]
        tail_start_loss, tail_start_aep = tail_loss[0], tail_aep[0]

        if np.isclose(emp_end_aep, tail_start_aep):
            hybrid_loss = np.concatenate((
                emp_loss[:-1],
                [0.5 * (emp_end_loss + tail_start_loss)],
                tail_loss[1:],
            ))
            hybrid_aep = np.concatenate((
                emp_aep[:-1],
                [emp_end_aep],
                tail_aep[1:],
            ))
        else:
            hybrid_loss = np.concatenate((emp_loss, tail_loss))
            hybrid_aep = np.concatenate((emp_aep, tail_aep))

    idx = np.argsort(hybrid_loss)
    hybrid_loss, hybrid_aep = hybrid_loss[idx], hybrid_aep[idx]

    if enforce_monotonic:
        hybrid_loss, hybrid_aep = enforce_strict_monotonicity(hybrid_loss, hybrid_aep)

    return hybrid_loss, hybrid_aep
