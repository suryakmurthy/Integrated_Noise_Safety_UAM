# Noise Aware Behavior in UAM Systems

This is the GitHub repository for the noise-aware agent developed for the NASA ULI project. It extends the [D2MAV-A model](https://arxiv.org/pdf/2003.08353) developed by Brittain et al.

## License

This repository contains modified code based on the D2MAV-A implementation, which is licensed under the GNU General Public License v3.0 (GPLv3). Accordingly, this project is also licensed under GPLv3.

You should have received a copy of the GNU General Public License along with this program. If not, see: https://www.gnu.org/licenses/

> **Modification Notice**: We have modified the original D2MAV-A code by adjusting model hyperparameters, the state and action spaces, and the reward function. These changes are marked in the relevant files and documented where applicable.

---

## Installation (Ubuntu)

### 1. Install project dependencies

1. Navigate to the `ULI_noise_aware_agent` directory:
    ```bash
    cd ULI_noise_aware_agent
    ```

2. Install BlueSky:
    ```bash
    pip install -e .
    ```

For more information on the BlueSky simulator, please see: [BlueSky GitHub](https://github.com/TUDelft-CNS-ATM/bluesky)

---

### Reward Shaping Configuration (`conf/config.gin`)

The training behavior of the agent is governed by **three reward shaping weights** defined in the config file:

```
Driver.weighting_factor_noise = [X₁, X₂, X₃]
Driver.weighting_factor_energy = [Y₁, Y₂, Y₃]
Driver.weighting_factor_separation = [Z₁, Z₂, Z₃]
```

Each index `(Xᵢ, Yᵢ, Zᵢ)` corresponds to a different reward configuration for a separate training run. This allows you to train **multiple models in sequence**, each with a different tradeoff between noise, energy, and separation objectives.

For example:

```
Driver.weighting_factor_noise = [0.5, 0.0, 1.0]
Driver.weighting_factor_energy = [0.0, 1.0, 0.0]
Driver.weighting_factor_separation = [0.5, 1.0, 0.0]
```

This configuration will trigger **three separate training runs** with the following weight combinations:

1. `(0.5, 0.0, 0.5)`
2. `(0.0, 1.0, 1.0)`
3. `(1.0, 0.0, 0.0)`

---

### Model Naming Convention

Each trained policy is saved under:

```
models/<run_name>/noise_XX_energy_YY_separation_ZZ/
```

Where `XX`, `YY`, and `ZZ` represent the values of `X`, `Y`, and `Z` (rounded to two decimal places) for that specific run.

Using the example above, the following subdirectories will be created:

```
models/train_D2MAV_full_results/
├── noise_05_energy_00_separation_05/
├── noise_00_energy_10_separation_10/
└── noise_10_energy_00_separation_00/
```

This naming convention helps you easily organize and compare models trained with different reward priorities.

### Evaluation Configuration (`conf/config_test.gin`)

This file defines how learned policies are evaluated in the BlueSky simulator.

---

#### Key Parameters

- `Driver.run_type = 'eval'`  
  Specifies that this run is for evaluation only — no training will occur.

- `Driver.weights_file`  
  A list of model checkpoint paths corresponding to different reward tradeoffs. Each entry defines a separate evaluation run.

  Example:
  ```python
  Driver.weights_file = [
      'models/train_D2MAV_full_results/noise_0175_energy_0825_separation_00/best_model.h5',
      'models/train_D2MAV_full_results/noise_02_energy_08_separation_00/best_model.h5',
      'models/train_D2MAV_full_results/noise_00_energy_045_separation_055/best_model.h5'
  ]
  ```

  These directories must follow the naming convention used during training. Each model will be evaluated independently using the same environment and scenario settings.


## Running the Project

1. Navigate to the project directory:
    ```bash
    cd ULI_noise_aware_agent
    ```

2. Run the main script:
    ```bash
    python main.py
    ```

---

## Simulation Visualization

1. Complete step 1 above in a single terminal (**Terminal 1**).
2. Open a **second terminal** (**Terminal 2**).

3. In Terminal 2:
    ```bash
    cd ULI_noise_aware_agent
    python BlueSky.py
    ```

4. The BlueSky GUI should open. After it loads, return to Terminal 1 and run the simulation script (step 2 in “Running the Project”).

5. In the BlueSky GUI, open the **Nodes** tab (lower-right). Select a different simulation node to view the Austin Environment sim.

---
## Training and Testing Results Visualization

1. Open the `Visualization_Results.ipynb` notebook.
2. Run the individual cells to generate the plots included in the journal paper.
3. TikZ-compatible versions of the plots will be saved in the `plot_results/` directory.

## Acknowledgements

This project uses the D2MAV-A model developed by Brittain et al.: [https://arxiv.org/pdf/2003.08353](https://arxiv.org/pdf/2003.08353)

We thank the original authors for their contribution. Our modified version is shared under the same GPLv3 license in the interest of open research and reproducibility.