import logging
import logging.config as conf
import os
import pickle
import pprint
from pathlib import Path as PathLib

import gin

from moretro.inference.retro_prediction import OneStepModel
from moretro.search.mcts_search import MCTSSearch
from moretro.utils.prepare_models import prepare_cost_models, prepare_starting_mols

logger = logging.getLogger("moretro")


class MORetroMCTS:
    def __init__(self, target: str):
        retro_model = OneStepModel(gin.REQUIRED)  # type: ignore
        building_blocks = prepare_starting_mols(gin.REQUIRED)  # type: ignore
        prepare_cost_models(gin.REQUIRED)  # type: ignore
        self.mcts_search = MCTSSearch(
            target, retro_model, building_blocks
        )  # type: ignore
        self.target = target

    def search(self):
        """
        Run the MCTS-based multi-objective retrosynthesis search and save results.
        """
        try:
            self.mcts_search.run_search()
            logger.info("MCTS search completed.")
        except KeyboardInterrupt:
            logger.warning("Search interrupted by user.")
        finally:
            self.mcts_search._posthoc_collect_and_rank_solutions()
            safe_target_name = self._safe_smiles_dirname(self.target)
            out_dir = f"results_mcts/{args.output_dir}/{safe_target_name}"
            os.makedirs(out_dir, exist_ok=True)
            pickle_path = f"{out_dir}/pareto_front.pkl"
            with open(pickle_path, "wb") as f:
                pickle.dump(
                    {
                        "pareto_front": self.mcts_search.pareto_front,
                        "pareto_front_costs": self.mcts_search.pareto_front_costs,
                        "all_solutions": self.mcts_search.all_solutions,
                        "solution_cost": self.mcts_search.solution_cost,
                        "pareto_solution_cost": self.mcts_search.pareto_solution_cost,
                    },
                    f,
                )
            logger.info(f"Saved Pareto front to {pickle_path}")
            summary = self.get_solution_summary()
            logger.info("Solution summary:\n" + pprint.pformat(summary, indent=2))
            self.mcts_search.visualize_solutions(out_dir)

    def get_solution_summary(self) -> dict:
        import numpy as np

        n_total = len(self.mcts_search.all_solutions)
        n_pareto = len(self.mcts_search.pareto_front)
        summary: dict = {
            "total_solutions": n_total,
            "pareto_solutions": n_pareto,
            "dominated_solutions": n_total - n_pareto,
        }
        if n_total > 0:
            costs = self.mcts_search.pareto_front_costs
            if costs.size > 0:
                summary["pareto_cost_ranges"] = {
                    f"objective_{i}": {
                        "min": round(float(costs[:, i].min()), 3),
                        "max": round(float(costs[:, i].max()), 3),
                        "mean": round(float(costs[:, i].mean()), 3),
                    }
                    for i in range(costs.shape[1])
                }
        return summary

    def _safe_smiles_dirname(self, smiles: str) -> str:
        safe_name = smiles.replace("/", "_slash_")
        safe_name = safe_name.replace("\\", "_backslash_")
        safe_name = safe_name.replace(":", "_colon_")
        safe_name = safe_name.replace("*", "_star_")
        safe_name = safe_name.replace("?", "_question_")
        safe_name = safe_name.replace('"', "_quote_")
        safe_name = safe_name.replace("<", "_lt_")
        safe_name = safe_name.replace(">", "_gt_")
        safe_name = safe_name.replace("|", "_pipe_")
        return safe_name


if __name__ == "__main__":
    import configparser
    import io
    from argparse import ArgumentParser

    import pandas as pd

    parser = ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save output files",
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--config_file",
        type=str,
        default="mcts_config.gin",
        help="Gin config file name (inside moretro/configs/)",
    )
    args = parser.parse_args()

    new_log_path = f"logs/{args.output_dir}_mcts.log"
    os.makedirs(os.path.dirname(new_log_path), exist_ok=True)

    config = configparser.ConfigParser()
    config.read("moretro/configs/logging.conf")
    config.set("handler_fileHandler", "args", f"('{new_log_path}', 'a')")
    with io.StringIO() as config_buffer:
        config.write(config_buffer)
        config_buffer.seek(0)
        conf.fileConfig(config_buffer, disable_existing_loggers=False)

    gin.parse_config_file(f"moretro/configs/{args.config_file}")

    mol_file = pd.read_csv(args.dataset, header=None, sep=",")
    for target_smiles in mol_file[0].tolist():
        target_smiles = target_smiles[2:-1]
        moretro_mcts = MORetroMCTS(target_smiles)
        moretro_mcts.search()
