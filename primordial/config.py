"""All tunable knobs for the simulation, in one place."""

from dataclasses import dataclass, fields


@dataclass
class Config:
    seed: int = 0
    device: str = "cpu"

    # --- world ---
    world_size: float = 1.4          # square torus [0, world_size)
    capacity: int = 2000             # max simultaneous particles
    initial_particles: int = 500
    min_population: int = 80         # below this, fresh random particles are spawned

    # --- particle-life physics ---
    n_types: int = 4                 # physical "species" used by the force matrix
    interaction_radius: float = 0.08
    repulsion_beta: float = 0.3      # inner core (fraction of radius) where everything repels
    force_factor: float = 10.0
    friction_half_life: float = 0.04
    dt: float = 0.02

    # --- bodies (genes in [0, 1] are mapped onto these ranges) ---
    size_min: float = 0.6            # size: energy capacity, attack power, mass, metabolism
    size_max: float = 1.8
    speed_min: float = 0.5           # speed: thrust multiplier, costs upkeep
    speed_max: float = 2.0
    sense_min: float = 0.5           # sense: perception range multiplier, costs upkeep
    sense_max: float = 1.6
    sense_radius: float = 0.08       # neighbour perception radius at sense = 1

    # --- brains ---
    hidden: int = 16
    max_thrust: float = 1.0
    kin_sigma: float = 0.15          # colour distance scale for "same family" detection

    # --- lifetime learning (reward-modulated Hebbian plasticity on the output layer) ---
    plasticity: float = 1.0          # 0 disables lifetime learning
    plastic_decay: float = 0.01      # per step, pulls learned weights back toward zero
    plastic_clip: float = 2.0
    reward_scale: float = 20.0       # reward = tanh(reward_scale * energy change)

    # --- energy economy ---
    initial_energy: float = 1.5
    reproduce_threshold: float = 2.0  # times body size
    birth_cost: float = 0.1          # fraction of energy lost when splitting
    base_metabolism: float = 0.0012  # per step, times size^0.75
    speed_upkeep: float = 0.0002     # per step, times speed
    sense_upkeep: float = 0.0002     # per step, times sense
    move_cost: float = 0.002         # per step, times |thrust|^2 * size
    aggression_cost: float = 0.002   # per step, times aggression
    signal_cost: float = 0.0003      # per step, times |signal|
    deposit_cost: float = 0.0005     # per step, times pheromone deposit
    max_age: int = 4000              # steps

    # --- predation ---
    contact_radius: float = 0.015    # times mean size of the pair
    attack_rate: float = 0.03        # energy drained per step at full aggression, times attacker size
    attack_efficiency: float = 0.6   # fraction of drained energy the attacker keeps
    kin_threshold: float = 0.5       # similarity above this = family, won't be attacked
    prey_size_ratio: float = 0.8     # attacker must be at least this fraction of victim's size

    # --- food ---
    food_capacity: int = 700
    food_energy: float = 0.6
    food_regrow_prob: float = 0.004  # per empty slot, per step
    food_clumping: float = 0.7       # chance new food sprouts inside a fertile patch
    food_patches: int = 5            # fertile zones that wander around the world
    patch_radius: float = 0.05
    patch_speed: float = 0.0004      # world units per step
    food_sense_radius: float = 0.15  # times sense
    eat_radius: float = 0.012        # times size

    # --- climate: food growth oscillates ---
    climate_period: int = 8000       # steps per full cycle (0 = constant climate)
    climate_amplitude: float = 0.5   # regrow multiplier swings between 1 - a and 1 + a

    # --- pheromones: 2 scent channels on a diffusing, decaying grid ---
    pheromone_res: int = 96
    pheromone_deposit: float = 0.05
    pheromone_diffusion: float = 0.2
    pheromone_decay: float = 0.01

    # --- bonds: kin can link up into multi-particle organisms ---
    max_bonds: int = 4
    bond_rest: float = 0.02          # rest length, times mean size of the pair
    bond_strength: float = 40.0
    bond_break: float = 0.07         # bonds snap beyond this length
    bond_share: float = 0.05         # fraction of energy difference equalised per step
    bond_cost: float = 0.0003        # per step per bond

    # --- sexual reproduction ---
    mate_prob: float = 1.0           # chance a reproducing particle looks for a mate
    mate_radius: float = 0.06
    mate_similarity: float = 0.6     # min kin similarity to be a compatible mate
    mate_contribution: float = 0.15  # fraction of the mate's energy given to the child

    # --- mutation ---
    mutation_rate: float = 0.1       # fraction of weights touched per birth (before layer multipliers)
    mutation_scale: float = 0.2      # step size (a starting value that evolves if sigma_tau > 0)
    perception_rate_mult: float = 1.0   # input-layer rate multiplier (0.5 = perceive-slowly variant)
    behaviour_rate_mult: float = 1.0    # output-layer rate multiplier (1.5 = act-fast variant)
    attraction_rate_mult: float = 1.0   # particle-life physics
    traits_rate_mult: float = 1.0       # body, budding, learning rate
    sigma_tau: float = 0.0           # self-adaptation speed of step sizes; 0 = fixed (default: in
                                     # experiments/mutation.json, 0.2 was not better)
    sigma_min: float = 0.01
    sigma_max: float = 1.0
    neuron_reset_prob: float = 0.0   # chance a birth re-randomises one whole hidden neuron
    color_drift: float = 0.05
    type_flip_prob: float = 0.01

    # --- analysis ---
    species_interval: int = 250      # steps between behavioural species clustering (0 = off)
    species_k: int = 5

    def override(self, assignments: list[str]) -> "Config":
        """Apply `key=value` strings (from the CLI) with type coercion."""
        known = {f.name for f in fields(self)}
        for item in assignments:
            key, _, raw = item.partition("=")
            if key not in known:
                raise KeyError(f"unknown config key: {key}")
            setattr(self, key, type(getattr(self, key))(raw))
        return self

    def size(self, gene):
        return self.size_min + gene * (self.size_max - self.size_min)

    def speed(self, gene):
        return self.speed_min + gene * (self.speed_max - self.speed_min)

    def sense(self, gene):
        return self.sense_min + gene * (self.sense_max - self.sense_min)
