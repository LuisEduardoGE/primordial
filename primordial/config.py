"""All tunable knobs for the simulation, in one place."""

from dataclasses import dataclass, fields


@dataclass
class Config:
    seed: int = 0
    device: str = "cpu"

    # --- world ---
    world_size: float = 1.0          # square torus [0, world_size)
    capacity: int = 1200             # max simultaneous particles
    initial_particles: int = 300
    min_population: int = 20         # below this, fresh random particles are spawned

    # --- particle-life physics ---
    n_types: int = 4                 # physical "species" used by the force matrix
    interaction_radius: float = 0.08
    repulsion_beta: float = 0.3      # inner core (fraction of radius) where everything repels
    force_factor: float = 10.0
    friction_half_life: float = 0.04
    dt: float = 0.02

    # --- brains ---
    hidden: int = 12
    max_thrust: float = 1.0
    kin_sigma: float = 0.15          # colour distance scale for "same family" detection

    # --- energy economy ---
    initial_energy: float = 1.0
    reproduce_threshold: float = 2.0
    birth_cost: float = 0.1          # fraction of energy lost when splitting
    base_metabolism: float = 0.002   # per step
    move_cost: float = 0.002         # per step, times |thrust|^2
    aggression_cost: float = 0.002   # per step, times aggression
    max_age: int = 4000              # steps

    # --- predation ---
    contact_radius: float = 0.015
    attack_rate: float = 0.03        # energy drained per step at full aggression
    attack_efficiency: float = 0.6   # fraction of drained energy the attacker keeps
    kin_threshold: float = 0.5       # similarity above this = family, won't be attacked

    # --- food ---
    food_capacity: int = 400
    food_energy: float = 0.5
    food_regrow_prob: float = 0.004  # per empty slot, per step
    food_clumping: float = 0.7       # chance new food sprouts inside a fertile patch
    food_patches: int = 4            # fertile zones that wander around the world ("seasons")
    patch_radius: float = 0.05
    patch_speed: float = 0.0004      # world units per step
    food_sense_radius: float = 0.15
    eat_radius: float = 0.012

    # --- mutation ---
    mutation_rate: float = 0.1       # fraction of weights touched per birth (before layer multipliers)
    mutation_scale: float = 0.2      # starting step size; evolves if sigma_tau > 0
    perception_rate_mult: float = 0.5   # input layer: change how the world is perceived slowly
    behaviour_rate_mult: float = 1.5    # output layer: change what to do with it faster
    attraction_rate_mult: float = 1.0   # particle-life physics
    sigma_tau: float = 0.2           # self-adaptation speed of step sizes (0 = fixed step size)
    sigma_min: float = 0.01
    sigma_max: float = 1.0
    neuron_reset_prob: float = 0.05  # chance a birth re-randomises one whole hidden neuron
    color_drift: float = 0.05
    type_flip_prob: float = 0.01

    def override(self, assignments: list[str]) -> "Config":
        """Apply `key=value` strings (from the CLI) with type coercion."""
        known = {f.name for f in fields(self)}
        for item in assignments:
            key, _, raw = item.partition("=")
            if key not in known:
                raise KeyError(f"unknown config key: {key}")
            setattr(self, key, type(getattr(self, key))(raw))
        return self
