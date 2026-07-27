import os

import numpy as np
import pandas as pd
from decoders import MPPContDecoder, MPPDecoder
from plot import plot_results
from scipy.io import savemat


def main(exp_name, decoder="mpp_cont"):
    # Define the path to save results
    save_path = f"results/log/{exp_name}/{{subject}}_{{phase}}.mat"
    save_path_fig = f"results/plots/{exp_name}/{{subject}}_{{phase}}.pdf"

    # Load dataset
    dataset = pd.read_csv("dataset.csv")
    dataset = dataset.groupby(["subject", "phase"])

    # Main loop
    for (subject, phase), data in dataset:
        print(f"Processing {subject} {phase}...")

        # Load data
        y = data["y"].values
        r = data["r"].values  # MPP component
        s = data["s"].values  # Cont component

        # Run decoder
        if decoder == "mpp_cont":
            results = MPPContDecoder(max_iter=500, rtol=1e-3)(
                x0=0,
                vx=0.005,
                r=r,
                r0=0,
                r1=10,
                vr=0.1 * np.var(r[r > 0]),
                s=s,
                s0=s[0],
                s1=0.1,
                vs=10 * np.var(s),
                s_lambda=0.1,
                vs_stop=0.1 * np.var(s),
            )
        elif decoder == "mpp":
            results = MPPDecoder(max_iter=500, rtol=1e-3)(
                x0=0,
                vx=0.005,
                r=r,
                r0=0,
                r1=10,
                vr=0.1 * np.var(r[r > 0]),
            )
        else:
            raise ValueError(f"Unknown decoder: {decoder}")

        # Save results
        results.update(
            {
                "t": np.arange(len(y)) / 4,  # 4 Hz
                "y": y,
                "r": r,
            }
        )
        if decoder == "mpp_cont":
            results.update({"s": s})
        _save_path = save_path.format(subject=subject, phase=phase)
        os.makedirs(os.path.dirname(_save_path), exist_ok=True)
        savemat(_save_path, results)

        # Plot results
        plot_results(
            t=results["t"],
            y=results["y"],
            x_pred=results["x_smth"],
            v_pred=results["v_smth"],
            vx=results["vx"],
            r=results["r"],
            r0=results["r0"],
            r1=results["r1"],
            vr=results["vr"],
            s=results.get("s"),
            s0=results.get("s0"),
            s1=results.get("s1"),
            vs=results.get("vs"),
            title=f"{subject} {phase}\n({results['status']} at iteration {results['iteration']})",
            save_path=save_path_fig.format(subject=subject, phase=phase),
        )

    print(f"{exp_name} done!")


if __name__ == "__main__":
    main("mpp_cont_best", decoder="mpp_cont")
    main("mpp_best", decoder="mpp")
