# AAMAS26
Codebase for experiments conducted for "Computing Perfect Bayesian Equilibria, with Application to
Empirical Game-Theoretic Analysis" (AAMAS 2026)

Included are:
- Code for generating abstract games GENGOOF4 and GENGOOF5:
    - abstract_games.py
    - generate_game_parameters.py (set NUM_ROUNDS variable)

- Code for PBE algorithm scalability experiment described in Section 5.2 (PBE experiments folder)
    - Run pbe_algorithm_experiment.py from launch.json
    - Regenerate plots using read_pbe_experiment_output.py

- Code for TE-PSRO application experiment on GenGoof described in Section 5.3 (te_psro_experiments_with_gengoof)
    - Run te_psro_main_gengoof.py from launch.json
    - Regenerate plots using plot_pbe_vs_ne_regret_gengoof.py

- Code for TE-PSRO application experiment on bargaining game described in Section 5.3 (te_psro_experiments_with_bargaining_game)
    - Run te_psro_main_bargaining.py from launch.json
    - Regenerate plots using plot_pbe_vs_ne_regret_bargaining.py