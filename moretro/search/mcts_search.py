"""Multi-objective MCTS-based retrosynthesis search.

This implementation mirrors AiZynthFinder MCTS control flow as closely as
possible while adapting three domain hooks to MORetro:

1. Expansion backend: ``OneStepModel.predict``
2. Prior source: ``pred[\"score\"]``
3. Reward source: ``pred[\"costs\"]``

Route collection and Pareto filtering are done post-hoc after search ends.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path as PathLib
from typing import TYPE_CHECKING, Optional

import graphviz
import gin
import numpy as np
from paretoset import paretoset
from rdkit import Chem
from rdkit.Chem import Draw

if TYPE_CHECKING:
    from moretro.inference.retro_prediction import OneStepModel

logger = logging.getLogger(__name__)


class ModelCallBudgetExhausted(RuntimeError):
    """Raised when the OneStepModel molecule-request budget is exhausted."""


class ExpansionBudgetExhausted(RuntimeError):
    """Raised when the iteration_budget (molecule expansion budget) is exhausted."""


class MCTSState:
    """Immutable molecule-frontier at an MCTS node."""

    def __init__(
        self,
        mols: tuple[str, ...],
        building_blocks: frozenset[str],
        reaction_history: tuple[dict, ...] = (),
    ) -> None:
        self.mols = mols
        self.building_blocks = building_blocks
        self.reaction_history = reaction_history
        self.expandable_mols = [m for m in self.mols if m not in self.building_blocks]
        self.is_solved = all(m in self.building_blocks for m in self.mols)
        self.max_transforms = len(self.reaction_history)
        self.expandables_hash = hash(tuple(sorted(self.expandable_mols)))
        self._hash = hash(tuple(sorted(self.mols)))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MCTSState):
            return False
        return self.__hash__() == other.__hash__()

    def is_terminal(self, max_depth: int) -> bool:
        return self.max_transforms >= max_depth or self.is_solved


class MCTSNode:
    """Pareto MCTS node with AiZ-style control flow."""

    def __init__(
        self,
        state: MCTSState,
        parent: Optional[MCTSNode],
        config: dict,
    ) -> None:
        self._state = state
        self._parent = parent
        self._config = config
        self._num_objectives: int = config["num_objectives"]
        self.is_expanded: bool = False
        self.is_expandable: bool = not state.is_terminal(config["max_depth"])

        self._children_values: list[list[float]] = []
        self._children_priors: list[list[float]] = []
        self._children_visitations: list[int] = []
        self._children_actions: list[tuple[str, dict]] = []
        self._children: list[Optional[MCTSNode]] = []
        self._children_rewards_cummulative: list[list[float]] = []

        self.blacklist = set(state.expandable_mols)
        if parent:
            self.blacklist = self.blacklist.union(parent.blacklist)

    @property
    def state(self) -> MCTSState:
        return self._state

    @property
    def parent(self) -> Optional[MCTSNode]:
        return self._parent

    def is_terminal(self) -> bool:
        return not self.is_expandable or self.state.is_terminal(self._config["max_depth"])

    def backpropagate(self, child: MCTSNode, value_estimate: list[float]) -> None:
        idx = self._children.index(child)
        self._children_visitations[idx] += 1
        self._children_rewards_cummulative[idx] = [
            cum_reward + new_reward
            for cum_reward, new_reward in zip(
                self._children_rewards_cummulative[idx], value_estimate
            )
        ]

    def expand(self, retro_model: OneStepModel, top_n: int) -> None:
        if self.is_expanded or not self.is_expandable:
            return

        self.is_expanded = True

        expandable = self.state.expandable_mols
        if not expandable:
            self.is_expandable = False
            self.is_expanded = False
            return

        expansion_tracker = self._config.get("expansion_budget_tracker")
        if expansion_tracker is not None:
            used = int(expansion_tracker.get("used", 0))
            budget = int(expansion_tracker.get("budget", 0))
            request_count = len(expandable)
            if used + request_count > budget:
                raise ExpansionBudgetExhausted(
                    "iteration_budget exhausted while expanding molecules: "
                    f"used={used}, requested={request_count}, budget={budget}"
                )
            expansion_tracker["used"] = used + request_count

        model_call_tracker = self._config.get("model_call_tracker")
        if model_call_tracker is not None:
            used = int(model_call_tracker.get("used", 0))
            budget = model_call_tracker.get("budget")
            request_count = len(expandable)
            if budget is not None and (used + request_count) > int(budget):
                raise ModelCallBudgetExhausted(
                    "OneStepModel molecule-request budget exhausted: "
                    f"used={used}, requested={request_count}, budget={budget}"
                )
            model_call_tracker["used"] = used + request_count

        predictions = retro_model.predict(expandable, top_n)

        actions: list[tuple[str, dict]] = []
        priors: list[float] = []
        for mol_smiles, mol_preds in zip(expandable, predictions):
            seen_reactants: set[tuple[str, ...]] = set()
            for pred in mol_preds:
                reactants = pred.get("reactants")
                if not isinstance(reactants, list) or not reactants:
                    continue

                # k=1 condition per unique reactant set.
                canonical_key = tuple(sorted(Chem.CanonSmiles(str(r)) for r in reactants))
                if canonical_key in seen_reactants:
                    continue
                seen_reactants.add(canonical_key)

                actions.append((mol_smiles, pred))
                priors.append(self._extract_prior_score(pred))

        self._fill_children_lists(actions, priors)

        if len(actions) == 0:
            self.is_expandable = False
            self.is_expanded = False

    def promising_child(self) -> Optional[MCTSNode]:
        child = None
        while child is None:
            try:
                child = self._score_and_select()
            except ValueError:
                child = None
                break

        if not child:
            self.is_expanded = False
            self.is_expandable = False

        return child

    def _extract_prior_score(self, pred: dict) -> float:
        if not self._config["use_prior"]:
            return float(self._config["default_prior"])

        if "score" not in pred:
            raise ValueError(
                "Missing 'score' in model prediction while use_prior=True."
            )

        try:
            score = float(pred["score"])
        except (TypeError, ValueError) as err:
            raise ValueError(
                f"Invalid model prior score {pred['score']!r}; expected numeric."
            ) from err

        if not np.isfinite(score):
            raise ValueError(f"Invalid model prior score {score!r}; expected finite.")
        if score < 0.0:
            raise ValueError(
                f"Invalid model prior score {score!r}; expected non-negative."
            )
        return score

    def _fill_children_lists(
        self, actions: list[tuple[str, dict]], priors: list[float]
    ) -> None:
        self._children_actions = actions
        nactions = len(actions)
        self._children_visitations = [1] * nactions
        self._children = [None] * nactions
        self._children_rewards_cummulative = [
            [0.0] * self._num_objectives for _ in range(nactions)
        ]

        if self._config["use_prior"]:
            self._children_priors = [[prior] * self._num_objectives for prior in priors]
        else:
            self._children_priors = [
                [self._config["default_prior"]] * self._num_objectives
                for _ in range(nactions)
            ]

        self._children_values = [
            [prior for prior in prior_vec] for prior_vec in self._children_priors
        ]

    def _children_q(self, children_values_arr: np.ndarray) -> np.ndarray:
        children_visitations_expanded = np.repeat(
            np.array(self._children_visitations).reshape(-1, 1),
            axis=1,
            repeats=self._num_objectives,
        )
        return children_values_arr / children_visitations_expanded

    def _children_u(self) -> np.ndarray:
        total_visits = np.log(np.sum(self._children_visitations))
        child_visits = np.array(self._children_visitations)
        return self._config["C"] * np.sqrt(2 * total_visits / child_visits)

    def _prior_schedule_oneoff(self) -> np.ndarray:
        visited_mask = (np.array(self._children_visitations) > 1).reshape(-1, 1)
        visited_mask = np.repeat(visited_mask, axis=1, repeats=self._num_objectives)
        children_priors_arr = np.array(self._children_priors)
        children_priors_arr[visited_mask] = 0.0
        return children_priors_arr

    def _compute_children_scores(self) -> np.ndarray:
        children_priors_arr = self._prior_schedule_oneoff()
        children_values_arr = children_priors_arr + np.array(
            self._children_rewards_cummulative
        )
        expanded_u = np.repeat(
            self._children_u().reshape(-1, 1),
            axis=1,
            repeats=self._num_objectives,
        )
        children_scores = self._children_q(children_values_arr) + expanded_u
        self._children_values = children_values_arr.tolist()
        self._children_priors = children_priors_arr.tolist()
        return children_scores

    def _update_pareto_front(self, children_scores: np.ndarray) -> np.ndarray:
        direction_arr = np.repeat("max", self._num_objectives)
        mask = paretoset(children_scores, sense=direction_arr, distinct=False)
        return np.arange(len(self._children))[mask]

    def _score_and_select(self) -> Optional[MCTSNode]:
        if not self._children_values:
            raise ValueError("Has no selectable children")
        if not max(max(value_list) for value_list in self._children_values) > 0:
            raise ValueError("Has no selectable children")

        children_scores = self._compute_children_scores()
        pareto_idxs = self._update_pareto_front(children_scores)
        if len(pareto_idxs) == 0:
            raise ValueError("Has no selectable children")

        rng = self._config.get("rng")
        if rng is None:
            index = int(np.random.choice(pareto_idxs))
        else:
            index = int(rng.choice(pareto_idxs))
        return self._select_child(index)

    def _disable_child(self, child_idx: int) -> None:
        penalty = [-1e6] * self._num_objectives
        self._children_rewards_cummulative[child_idx] = list(penalty)
        self._children_values[child_idx] = list(penalty)
        self._children_priors[child_idx] = [0.0] * self._num_objectives

    def _expand_children_lists(self, old_index: int, action_index: int) -> int:
        old_mol, old_pred = self._children_actions[old_index]
        new_pred = dict(old_pred)
        outcomes = self._reaction_outcomes(old_pred)
        if action_index < len(outcomes):
            new_pred["reactants"] = list(outcomes[action_index])

        self._children_actions.append((old_mol, new_pred))
        self._children_priors.append(list(self._children_priors[old_index]))
        self._children_values.append(list(self._children_values[old_index]))
        self._children_visitations.append(self._children_visitations[old_index])
        self._children_rewards_cummulative.append(
            list(self._children_rewards_cummulative[old_index])
        )
        self._children.append(None)
        return len(self._children) - 1

    def _reaction_outcomes(self, pred: dict) -> list[tuple[str, ...]]:
        reactants = pred.get("reactants")
        if not isinstance(reactants, list):
            return []

        # OneStepModel usually returns one reactant set as list[str].
        if reactants and all(isinstance(item, str) for item in reactants):
            return [tuple(Chem.CanonSmiles(item) for item in reactants)]

        # Optional compatibility if reactants are nested list[list[str]].
        outcomes: list[tuple[str, ...]] = []
        for outcome in reactants:
            if isinstance(outcome, list) and all(isinstance(item, str) for item in outcome):
                outcomes.append(tuple(Chem.CanonSmiles(item) for item in outcome))
        return outcomes

    def _check_child_reaction(self, mol_smiles: str, reactants: tuple[str, ...]) -> bool:
        if not reactants:
            return False
        if len(reactants) == 1 and reactants[0] == Chem.CanonSmiles(mol_smiles):
            return False
        return True

    def _regenerated_blacklisted(self, reactants: tuple[str, ...]) -> bool:
        for mol in reactants:
            if mol in self.blacklist:
                return True
        return False

    def _create_children_nodes(
        self, states: list[MCTSState], child_idx: int
    ) -> list[MCTSNode]:
        new_nodes: list[MCTSNode] = []
        first_child_idx = child_idx
        for state_index, state in enumerate(states):
            # If there is more than one outcome, expand bookkeeping lists.
            if state_index > 0:
                child_idx = self._expand_children_lists(first_child_idx, state_index)

            _, child_pred = self._children_actions[child_idx]
            outcomes = self._reaction_outcomes(child_pred)
            child_reactants = outcomes[0] if outcomes else tuple()

            if self._regenerated_blacklisted(child_reactants):
                self._disable_child(child_idx)
                continue

            new_node = MCTSNode(state=state, parent=self, config=self._config)
            self._children[child_idx] = new_node
            new_nodes.append(new_node)
        return new_nodes

    def _instantiate_child(self, child_idx: int) -> list[MCTSNode]:
        if self._children[child_idx] is not None:
            child = self._children[child_idx]
            assert child is not None
            return [child]

        mol_smiles, pred = self._children_actions[child_idx]
        outcomes = self._reaction_outcomes(pred)
        if not outcomes:
            self._disable_child(child_idx)
            return []

        if not self._check_child_reaction(mol_smiles, outcomes[0]):
            self._disable_child(child_idx)
            return []

        keep_mols = list(self.state.mols)
        try:
            keep_mols.remove(mol_smiles)
        except ValueError:
            self._disable_child(child_idx)
            return []

        new_states = [
            MCTSState(
                tuple(keep_mols + list(reactants)),
                self.state.building_blocks,
                self.state.reaction_history
                + (
                    {
                        **pred,
                        "reactants": list(reactants),
                    },
                ),
            )
            for reactants in outcomes
        ]
        return self._create_children_nodes(new_states, child_idx)

    def _select_child(self, child_idx: int) -> Optional[MCTSNode]:
        if self._children[child_idx]:
            return self._children[child_idx]

        new_nodes = self._instantiate_child(child_idx)
        if new_nodes:
            rng = self._config.get("rng")
            if rng is None:
                return new_nodes[int(np.random.choice(np.arange(len(new_nodes))))]
            return new_nodes[int(rng.integers(0, len(new_nodes)))]
        return None


@gin.configurable(denylist=["target", "retro_model", "building_blocks"])
class MCTSSearch:
    """Multi-objective MCTS retrosynthesis search using Pareto node selection."""

    def __init__(
        self,
        target: str,
        retro_model: OneStepModel,
        building_blocks: set[str],
        top_n: int,
        max_depth: int,
        iteration_budget: int,
        C: float = 1.4,
        use_prior: bool = True,
        default_prior: float = 0.5,
        time_budget: float = 0.0,
        max_solutions: int = 250,
        model_call_budget: Optional[int] = None,
        random_seed: Optional[int] = 0,
        rollout_step_limit: Optional[int] = None,
        solved_fraction_bonus: float = 0.5,
        solved_terminal_bonus: float = 0.5,
        backup_gamma: float = 1.0,
        reward_k: float = 1.0,
    ) -> None:
        self.retro_model = retro_model
        self.top_n = top_n
        self.iteration_budget = int(iteration_budget)
        self.time_budget = time_budget
        self.max_solutions = max_solutions
        self.model_call_budget = model_call_budget
        self.rollout_step_limit = rollout_step_limit
        self.solved_fraction_bonus = float(solved_fraction_bonus)
        self.solved_terminal_bonus = float(solved_terminal_bonus)
        self.backup_gamma = float(backup_gamma)
        self.reward_k = float(reward_k)
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

        self.profiling = {
            "expansion_calls": 0,
            "iterations": 0,
        }

        self._num_objectives = len(retro_model.cost_functions)
        self._max_depth = int(max_depth)
        self._expansion_budget_tracker = {"used": 0, "budget": self.iteration_budget}
        self._model_call_tracker = {"used": 0, "budget": model_call_budget}

        canonical_target = Chem.CanonSmiles(target)
        canonical_bbs = building_blocks
        self.target = canonical_target
        self.canonical_building_blocks = canonical_bbs

        self._config = {
            "C": C,
            "use_prior": use_prior,
            "default_prior": default_prior,
            "num_objectives": self._num_objectives,
            "max_depth": max_depth,
            "rng": self._rng,
            "expansion_budget_tracker": self._expansion_budget_tracker,
            "model_call_tracker": self._model_call_tracker,
        }

        initial_state = MCTSState(
            mols=(canonical_target,),
            building_blocks=frozenset(canonical_bbs),
            reaction_history=(),
        )
        self._root = MCTSNode(state=initial_state, parent=None, config=self._config)

        self.all_solutions: list[tuple[tuple[dict, ...], list[float]]] = []
        self.pareto_front: list[tuple[tuple[dict, ...], list[float]]] = []
        self.pareto_front_costs: np.ndarray = np.empty((0, self._num_objectives), dtype=float)
        self.solution_cost: dict[tuple[float, ...], tuple[dict, ...]] = {}
        self.pareto_solution_cost: dict[tuple[float, ...], tuple[dict, ...]] = {}

    def select_leaf(self) -> Optional[MCTSNode]:
        current = self._root
        while current.is_expanded and not current.state.is_solved:
            promising_child = current.promising_child()
            if promising_child:
                current = promising_child
            else:
                break
        return current

    def one_iteration(self) -> Optional[bool]:
        self.profiling["iterations"] += 1

        leaf = self.select_leaf()
        if leaf is None:
            return None

        leaf.expand(self.retro_model, self.top_n)
        self.profiling["expansion_calls"] += 1

        rollout_steps = 0
        while not leaf.is_terminal():
            if self.rollout_step_limit is not None and rollout_steps >= self.rollout_step_limit:
                break
            child = leaf.promising_child()
            if child:
                child.expand(self.retro_model, self.top_n)
                self.profiling["expansion_calls"] += 1
                leaf = child
                rollout_steps += 1
            else:
                break

        self._backpropagate(leaf)
        return leaf.state.is_solved

    def compute_reward(self, node: MCTSNode) -> list[float]:
        route = node.state.reaction_history
        reward = [0.0] * self._num_objectives
        if not route:
            return reward

        # Segler-style branch utility: W(b) = max(0, (L_max - xi(b)) / L_max)
        # with xi(b) = length(b) - sum(k * P).
        branch_len = float(len(route))
        if self._max_depth <= 0:
            raise ValueError("max_depth must be positive for W(b) reward calculation.")

        policy_proxies: list[list[float]] = [[] for _ in range(self._num_objectives)]
        for pred in route:
            costs = pred.get("costs")
            if not isinstance(costs, (list, tuple, np.ndarray)):
                raise ValueError("Missing or invalid 'costs' in reaction history.")
            if len(costs) != self._num_objectives:
                raise ValueError(
                    "Invalid 'costs' length in reaction history. "
                    f"Expected {self._num_objectives}, got {len(costs)}."
                )

            for idx, cost in enumerate(costs):
                cost_value = float(cost)
                if not np.isfinite(cost_value):
                    raise ValueError("Invalid non-finite cost in reaction history.")

                # User-requested mapping:
                # objective 4 uses P = cost directly, all others use P = (1 - cost).
                if idx == 3:
                    policy_proxy = cost_value
                else:
                    policy_proxy = 1.0 - cost_value

                policy_proxies[idx].append(float(policy_proxy))

        # z semantics requested by user:
        # - z > 1 for fully solved
        # - z in [0, 1] for partially solved (ratio)
        # - z = -1 if no molecule solved
        total_mols = len(node.state.mols)
        solved_mols = sum(mol in node.state.building_blocks for mol in node.state.mols)
        solved_ratio = float(solved_mols) / float(total_mols) if total_mols > 0 else 0.0
        if node.state.is_solved:
            z = 1.0 + self.solved_fraction_bonus + self.solved_terminal_bonus
        elif solved_ratio > 0.0:
            z = solved_ratio
        else:
            z = -1.0

        lmax = float(self._max_depth)
        for idx in range(self._num_objectives):
            proxy_sum = float(np.sum(policy_proxies[idx]))
            xi = branch_len - (self.reward_k * proxy_sum)
            w_branch = max(0.0, (lmax - xi) / lmax)
            reward[idx] = float(z * w_branch)

        return reward

    def run_search(self) -> dict:
        logger.info(
            "Starting AiZ-style MORetro MCTS. expansion_budget=%d molecules, "
            "model_call_budget=%s",
            int(self._expansion_budget_tracker["budget"]),
            str(self._model_call_tracker["budget"]),
        )

        if self._root.state.is_solved:
            logger.info("Target is a building block - trivially solved.")
            return self._make_result(0)

        start = time.time()
        completed_iterations = 0

        while True:
            if self.time_budget > 0 and (time.time() - start) >= self.time_budget:
                logger.info("Time budget reached. Stopping search.")
                break

            if not self._root.is_expandable and not self._root.state.is_solved:
                logger.info("No expandable branches remain. Stopping search.")
                break

            try:
                solved = self.one_iteration()
            except ExpansionBudgetExhausted:
                logger.info(
                    "Iteration (molecule expansion) budget reached (%d/%d). Stopping search.",
                    int(self._expansion_budget_tracker["used"]),
                    int(self._expansion_budget_tracker["budget"]),
                )
                break
            except ModelCallBudgetExhausted:
                logger.info(
                    "OneStepModel molecule-request budget reached (%d/%s). Stopping search.",
                    int(self._model_call_tracker["used"]),
                    str(self._model_call_tracker["budget"]),
                )
                break

            if solved is None:
                logger.info("No selectable leaf remains. Stopping search.")
                break

            completed_iterations += 1

            logger.info(
                "Iteration %d complete. Model-call molecules used=%d/%s. "
                "Expansion-budget molecules used=%d/%d.",
                completed_iterations,
                int(self._model_call_tracker["used"]),
                str(self._model_call_tracker["budget"]),
                int(self._expansion_budget_tracker["used"]),
                int(self._expansion_budget_tracker["budget"]),
            )

        logger.info(
            "Search complete after %d iterations and %d expanded molecules. "
            "Post-hoc route extraction is deferred to caller.",
            completed_iterations,
            int(self._expansion_budget_tracker["used"]),
        )
        return self._make_result(completed_iterations)

    def _backpropagate(self, leaf: MCTSNode) -> None:
        value_estimate = self.compute_reward(leaf)
        current = leaf
        while current is not self._root:
            parent = current.parent
            assert parent is not None
            parent.backpropagate(current, value_estimate)
            current = parent

    def _posthoc_collect_and_rank_solutions(self) -> None:
        solved_paths: list[tuple[tuple[dict, ...], list[float]]] = []
        seen_hashes: set[tuple[str, ...]] = set()

        stack = [self._root]
        while stack:
            node = stack.pop()
            for child in node._children:
                if child is not None:
                    stack.append(child)

            if not node.state.is_solved:
                continue

            path = node.state.reaction_history
            path_key = tuple(str(pred.get("rxn_smiles", "")) for pred in path)
            if path_key in seen_hashes:
                continue
            seen_hashes.add(path_key)

            cost_vector = [
                float(np.sum([float(pred["costs"][k]) for pred in path]))
                for k in range(self._num_objectives)
            ]
            solved_paths.append((path, cost_vector))

            if self.max_solutions > 0 and len(solved_paths) >= self.max_solutions:
                break

        self.all_solutions = solved_paths

        if not self.all_solutions:
            self.pareto_front = []
            self.pareto_front_costs = np.empty((0, self._num_objectives), dtype=float)
            self.solution_cost = {}
            self.pareto_solution_cost = {}
            return

        all_costs = np.array([sol[1] for sol in self.all_solutions], dtype=float)
        sense = np.repeat("min", self._num_objectives)
        mask = paretoset(all_costs, sense=sense, distinct=True)

        self.pareto_front = [sol for sol, keep in zip(self.all_solutions, mask) if keep]
        self.pareto_front_costs = all_costs[mask]

        self.solution_cost = {}
        self.pareto_solution_cost = {}
        for path, cost_vector in self.all_solutions:
            rounded_key = tuple(round(float(c), 3) for c in cost_vector)
            if rounded_key not in self.solution_cost:
                self.solution_cost[rounded_key] = path

        for path, cost_vector in self.pareto_front:
            rounded_key = tuple(round(float(c), 3) for c in cost_vector)
            if rounded_key not in self.pareto_solution_cost:
                self.pareto_solution_cost[rounded_key] = path

    def get_solution_summary(self) -> dict:
        pareto_costs = [tuple(sol[1]) for sol in self.pareto_front]
        all_costs = [tuple(sol[1]) for sol in self.all_solutions]
        dominated_costs = [cost for cost in all_costs if cost not in set(pareto_costs)]

        summary: dict[str, object] = {
            "total_solutions": len(all_costs),
            "pareto_solutions": len(pareto_costs),
            "dominated_solutions": len(dominated_costs),
        }

        if all_costs:
            all_costs_array = np.array(all_costs, dtype=float)
            summary["cost_ranges"] = {
                f"objective_{i}": {
                    "min": round(float(all_costs_array[:, i].min()), 2),
                    "max": round(float(all_costs_array[:, i].max()), 2),
                    "mean": round(float(all_costs_array[:, i].mean()), 2),
                }
                for i in range(all_costs_array.shape[1])
            }
        return summary

    def visualize_solution_pathway(
        self,
        path: tuple[dict, ...],
        output_path: str,
        title: Optional[str] = None,
    ) -> None:
        if not path:
            logger.warning("Cannot visualize empty synthesis pathway.")
            return

        graph = graphviz.Digraph(format="png")
        graph.attr(rankdir="LR")
        if title:
            graph.attr(label=title, labelloc="t", fontsize="16")

        with tempfile.TemporaryDirectory() as temp_img_dir:
            previous_reaction_id = ""
            for step_idx, pred in enumerate(path, start=1):
                rxn_smiles = str(pred.get("rxn_smiles", ""))
                reaction_id = f"rxn_{step_idx}"
                reactants_part = ""
                product_part = ""
                product_smiles = ""
                if ">>" in rxn_smiles:
                    reactants_part, product_part = rxn_smiles.split(">>", maxsplit=1)

                reaction_label = f"Step {step_idx}"
                condition_label = self._format_conditions_label(pred)
                if condition_label:
                    reaction_label += f"\\n{condition_label}"
                graph.node(
                    reaction_id,
                    label=reaction_label,
                    shape="box",
                    style="rounded",
                    color="lightsteelblue",
                    penwidth="2",
                )

                if reactants_part:
                    for reactant_idx, reactant in enumerate(
                        reactants_part.split("."), start=1
                    ):
                        if not reactant:
                            continue
                        reactant_smiles = Chem.CanonSmiles(reactant)
                        if reactant_smiles in self.canonical_building_blocks:
                            continue
                        reactant_id = f"react_{step_idx}_{reactant_idx}"
                        reactant_img = self._smiles_image_path(
                            reactant_smiles,
                            temp_img_dir,
                            f"react_{step_idx}_{reactant_idx}",
                        )
                        graph.node(
                            reactant_id,
                            label="",
                            image=reactant_img,
                            shape="box",
                            color="springgreen3",
                            penwidth="2",
                        )
                        graph.edge(reactant_id, reaction_id, color="darkgrey")

                if product_part:
                    product_smiles = Chem.CanonSmiles(product_part)
                    if product_smiles in self.canonical_building_blocks:
                        product_smiles = ""
                if product_smiles:
                    product_id = f"prod_{step_idx}"
                    product_img = self._smiles_image_path(
                        product_smiles, temp_img_dir, f"prod_{step_idx}"
                    )
                    graph.node(
                        product_id,
                        label="",
                        image=product_img,
                        shape="box",
                        color="lightsalmon",
                        penwidth="2",
                    )
                    graph.edge(reaction_id, product_id, color="darkgrey")

                if previous_reaction_id:
                    graph.edge(
                        previous_reaction_id,
                        reaction_id,
                        style="dashed",
                        color="gray50",
                    )
                previous_reaction_id = reaction_id

            parent = PathLib(output_path).parent
            parent.mkdir(parents=True, exist_ok=True)
            graph.render(output_path, cleanup=True)

    def visualize_solutions(
        self,
        output_dir: str = "figs",
        pareto_only: bool = False,
        max_solutions: Optional[int] = None,
    ) -> None:
        candidates = self.pareto_front if pareto_only else self.all_solutions
        if not candidates:
            logger.warning("No solutions available for pathway visualization.")
            return

        safe_target_name = self._safe_smiles_dirname(self.target)
        subset_dir = "pareto" if pareto_only else "all"
        base_dir = PathLib(output_dir) / safe_target_name / subset_dir
        base_dir.mkdir(parents=True, exist_ok=True)

        selected = candidates if max_solutions is None else candidates[:max_solutions]
        logger.info("Visualizing %d synthesis pathways in %s", len(selected), str(base_dir))

        for idx, (path, cost_vector) in enumerate(selected, start=1):
            title = (
                f"{'Pareto' if pareto_only else 'Solution'} {idx}"
                f"\\nCosts: {[f'{c:.3f}' for c in cost_vector]}"
            )
            route_out = str(base_dir / f"route_{idx}")
            self.visualize_solution_pathway(path, route_out, title=title)

    def _smiles_image_path(self, smiles: str, temp_dir: str, stem: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            file_path = f"{temp_dir}/{stem}_invalid.png"
            Draw.MolToFile(Chem.MolFromSmiles("C"), file_path, size=(180, 180))
            return file_path

        escaped = stem.replace("/", "_").replace("\\", "_")
        file_path = f"{temp_dir}/{escaped}.png"
        Draw.MolToFile(mol, file_path, size=(180, 180))
        return file_path

    def _format_conditions_label(self, pred: dict) -> str:
        parts: list[str] = []

        temperature = pred.get("temperature")
        if temperature is not None and str(temperature).strip() != "":
            parts.append(f"T: {temperature}")

        reagents = pred.get("reagents")
        if isinstance(reagents, list):
            reagent_text = ", ".join(
                str(r).strip() for r in reagents if str(r).strip()
            )
        elif reagents is None:
            reagent_text = ""
        else:
            reagent_text = str(reagents).strip()
        if reagent_text:
            parts.append(f"Reagents: {reagent_text}")

        return "\\n".join(parts)

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

    def _make_result(self, iterations: int) -> dict:
        return {
            "pareto_front": self.pareto_front,
            "pareto_front_costs": self.pareto_front_costs,
            "all_solutions": self.all_solutions,
            "solution_cost": self.solution_cost,
            "pareto_solution_cost": self.pareto_solution_cost,
            "model_calls_used": int(self._model_call_tracker["used"]),
            "model_call_budget": self._model_call_tracker["budget"],
            "iterations": iterations,
        }
