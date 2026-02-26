# CAAC-Checking-Base-PyScript
Here is a detailed, explanatory list of all command-line flags (arguments) that this Python script supports, written in clear English with explanations, what each one does, typical use cases, and how to use it properly.
General Rules for Using Flags

All flags start with -- (double dash)
Most take a value right after them, like: --requests 8000
Boolean-like flags (only one) don't take a value — just writing the flag enables it: --no-plots
Order of flags doesn't matter
You can combine many flags in one command
If you don't specify a flag → it uses the default value
To see this list yourself while running:Bashpython your_script.py --help

Full List of All Supported Flags (with Detailed Explanation)

--seed INTEGER
Default: 42
Sets the global random seed.
→ Use the same seed → you get exactly the same simulation results every time (great for comparing changes).
→ Change it → get different random users, attacks, timings, etc.
Example: --seed 123
--requests INTEGER
Default: 4000
Total number of requests (events) to simulate.
This is usually the most important flag you change.
Small runs (testing): 500–2000
Normal experiments: 3000–10000
Large scale: 15000+ (takes longer)
Example: --requests 7500
--users INTEGER
Default: 120
How many synthetic users to create (only used if you don't provide --users-csv).
More users = more realistic diversity in roles, departments, etc.
Typical range: 50–500
Example: --users 200
--attack-ratio FLOAT (between 0.0 and 1.0)
Default: 0.30
What percentage of all requests are attack attempts.
0.10–0.25 = low attack volume (realistic for many systems)
0.30–0.50 = stress testing / red-team style
Example: --attack-ratio 0.42
--business-hour-start INTEGER (0–23)
Default: 8
Start of "business hours" — used to detect off-hours sensitive actions.
Example: --business-hour-start 9
--business-hour-end INTEGER (0–23)
Default: 18
End of business hours.
Must be > start hour.
Example: --business-hour-end 17
--dataset-mode STRING
Default: hybrid
Allowed values: synthetic, hybrid, replay
synthetic → generate everything fake (fastest, fully controlled)
hybrid → mix real requests + synthetic (most realistic)
replay → only use real requests from file (requires --requests-csv)
Example: --dataset-mode replay

--real-request-ratio FLOAT (0.0–1.0)
Default: 0.40
Only used in hybrid mode.
What percentage of requests should come from your real CSV file.
Example: --real-request-ratio 0.65
--max-episode-length INTEGER
Default: 4
Maximum number of steps (tool calls) in one agent conversation/loop.
Longer = more complex legitimate & attack chains.
Typical: 3–8
Example: --max-episode-length 6
--multi-step-attack-ratio FLOAT (0.0–1.0)
Default: 0.70
Of all attack episodes, what percentage should be multi-step (chained) attacks.
Example: --multi-step-attack-ratio 0.55
--caac-policy-weight FLOAT
Default: 0.52
How much the rule-based (hard-coded policy) risk contributes to CAAC decision.
Sum of this + --caac-ml-weight should be ≈ 1.0
--caac-ml-weight FLOAT
Default: 0.48
How much the learned ML risk score contributes to CAAC.
--caac-deny-threshold FLOAT (0.0–1.0)
Default: 0.57
If total risk ≥ this value → CAAC denies the request.
Higher value = more permissive
Lower value = more strict
Very important tuning parameter.
Example: --caac-deny-threshold 0.62
--network-jitter-ms FLOAT
Default: 16.0
How much random variation in network delay (ms).
--network-spike-prob FLOAT (0.0–1.0)
Default: 0.03
Chance of a big sudden network delay spike.
--transient-error-prob FLOAT (0.0–1.0)
Default: 0.02
Chance of random non-security-related runtime failure.
--session-noise-prob FLOAT (0.0–1.0)
Default: 0.015
Chance of randomly making a valid session appear expired (simulates drift).
--token-latency-per-1k-ms FLOAT
Default: 22.0
How many milliseconds delay per 1000 LLM tokens.
--llm-cost-per-1k-tokens-usd FLOAT
Default: 0.0030
Simulated cost (USD) per 1000 tokens — only for reporting.
--sensitivity-runs INTEGER
Default: 16
How many times to repeat simulation with perturbed attack weights (sensitivity analysis).
--sensitivity-requests INTEGER
Default: 1200
How many requests per sensitivity run.
--output-dir PATH
Default: simulation_outputs
Folder where all CSV files and plots will be saved.
Good practice: use different folders for different experiments.
Example: --output-dir results_v2_high_attack
--no-plots (flag – no value needed)
If you add this flag → script will not create any PNG charts (faster).
Example: --no-plots
--users-csv PATH
Path to CSV file containing user data (overrides synthetic users).
Example: --users-csv data/our_users_2025.csv
--sessions-csv PATH
Path to CSV with session data.
--tools-csv PATH
Path to CSV defining tools (name, min_role, clearance, etc.).
--requests-csv PATH
Path to CSV with real or previously generated requests.
Required for --dataset-mode replayUsed in hybrid mode together with synthetic data.
--attack-weights-csv PATH
Path to CSV that defines custom probabilities for each attack type.

Quick Reference – Most Important Ones to Remember
Bash--requests 8000              # how many events
--attack-ratio 0.35          # how many are attacks
--dataset-mode hybrid        # or synthetic / replay
--real-request-ratio 0.6     # when using hybrid
--caac-deny-threshold 0.60   # CAAC strictness
--output-dir my_experiment_3 # separate results
--requests-csv real_data.csv # use real requests
--no-plots                   # skip chart generation
