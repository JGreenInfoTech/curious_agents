"""
Structured Knowledge Environment
================================

Design philosophy: Agents perceive *properties*, not pixels.
This is cheaper, more debuggable, and focuses compute on what matters:
motivation, learning, communication, and meta-cognition.

A separate visualization layer renders the world for human observers.
Agents never see the visualization — they get structured property vectors.

Think of it as: agents have "perfect vision" for object properties,
like a creature that can directly sense color/shape/category.
The interesting question isn't "can they see?" — it's "what do they
do with what they see?"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import json


# =============================================================================
# World Objects
# =============================================================================

# Property dimensions that all objects share.
# This is the agent's "sensory vocabulary" — what properties exist in the world.
PROPERTY_SCHEMA = {
    'color_r': 0,      # Red channel (0-1)
    'color_g': 1,      # Green channel (0-1)
    'color_b': 2,      # Blue channel (0-1)
    'size': 3,          # Relative size (0=tiny, 1=large)
    'shape_round': 4,   # Roundness (0=angular, 1=spherical)
    'shape_long': 5,    # Elongation (0=compact, 1=stretched)
    'animate': 6,       # Living/moving (0=inanimate, 1=alive)
    'edible': 7,        # Can be eaten (0=no, 1=yes)
    'dangerous': 8,     # Can cause harm (0=safe, 1=dangerous)
    'warm': 9,          # Temperature (0=cold, 1=hot)
    'soft': 10,         # Texture (0=hard, 1=soft)
    'bright': 11,       # Luminosity (0=dark, 1=bright)
    'noisy': 12,        # Sound level (0=silent, 1=loud)
    'complexity': 13,   # Visual complexity (0=simple, 1=complex)
    'familiarity': 14,  # How common (0=rare, 1=ubiquitous) - agent-relative, updated over time
}

PROPERTY_DIM = len(PROPERTY_SCHEMA)


@dataclass
class WorldObject:
    """
    An object in the world with perceivable properties.
    
    Properties are a fixed-dimension vector — this is what agents "sense."
    The name and category are metadata for human readability and grounding.
    """
    name: str                           # Human-readable: "apple", "cat"
    category: str                       # Semantic category: "fruit", "animal"
    properties: np.ndarray              # Shape: (PROPERTY_DIM,) — the agent's percept
    position: Tuple[float, float]       # (x, y) in world coordinates
    text_description: str = ""          # Optional text the agent can "read"
    is_interactive: bool = False        # Can agent manipulate this?
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra info for analysis
    
    def __post_init__(self):
        assert self.properties.shape == (PROPERTY_DIM,), \
            f"Properties must be {PROPERTY_DIM}-dimensional, got {self.properties.shape}"
        # Clamp to [0, 1] — properties are normalized
        self.properties = np.clip(self.properties, 0.0, 1.0)
    
    def distance_to(self, position: Tuple[float, float]) -> float:
        return np.sqrt((self.position[0] - position[0])**2 + 
                       (self.position[1] - position[1])**2)
    
    def property_similarity(self, other: 'WorldObject') -> float:
        """Cosine similarity between property vectors"""
        dot = np.dot(self.properties, other.properties)
        norm_a = np.linalg.norm(self.properties) + 1e-8
        norm_b = np.linalg.norm(other.properties) + 1e-8
        return dot / (norm_a * norm_b)


@dataclass
class Relation:
    """A relationship between two objects: (subject, predicate, object)"""
    subject: str    # Object name
    predicate: str  # "near", "on_top_of", "contains", "bigger_than"
    object: str     # Object name


# =============================================================================
# Object Library — Pre-defined objects the environment can contain
# =============================================================================

def make_properties(**kwargs) -> np.ndarray:
    """Helper to create property vectors from named values."""
    props = np.zeros(PROPERTY_DIM)
    for key, value in kwargs.items():
        if key in PROPERTY_SCHEMA:
            props[PROPERTY_SCHEMA[key]] = value
        else:
            raise ValueError(f"Unknown property: {key}. Valid: {list(PROPERTY_SCHEMA.keys())}")
    return props


# A starter library of grounded concepts.
# Each is a function that returns a WorldObject at a given position.
OBJECT_LIBRARY = {
    'apple': lambda pos: WorldObject(
        name='apple', category='fruit',
        properties=make_properties(
            color_r=0.9, color_g=0.1, color_b=0.1,
            size=0.3, shape_round=0.9, shape_long=0.1,
            animate=0.0, edible=1.0, dangerous=0.0,
            warm=0.2, soft=0.4, bright=0.5, noisy=0.0,
            complexity=0.2
        ),
        position=pos,
        text_description="A round red fruit."
    ),
    'banana': lambda pos: WorldObject(
        name='banana', category='fruit',
        properties=make_properties(
            color_r=0.9, color_g=0.9, color_b=0.2,
            size=0.3, shape_round=0.2, shape_long=0.8,
            animate=0.0, edible=1.0, dangerous=0.0,
            warm=0.2, soft=0.5, bright=0.6, noisy=0.0,
            complexity=0.2
        ),
        position=pos,
        text_description="A yellow curved fruit."
    ),
    'cat': lambda pos: WorldObject(
        name='cat', category='animal',
        properties=make_properties(
            color_r=0.5, color_g=0.4, color_b=0.3,
            size=0.4, shape_round=0.4, shape_long=0.6,
            animate=1.0, edible=0.0, dangerous=0.1,
            warm=0.7, soft=0.8, bright=0.3, noisy=0.3,
            complexity=0.7
        ),
        position=pos,
        text_description="A small furry animal that purrs."
    ),
    'dog': lambda pos: WorldObject(
        name='dog', category='animal',
        properties=make_properties(
            color_r=0.6, color_g=0.4, color_b=0.2,
            size=0.6, shape_round=0.3, shape_long=0.7,
            animate=1.0, edible=0.0, dangerous=0.2,
            warm=0.7, soft=0.7, bright=0.3, noisy=0.6,
            complexity=0.7
        ),
        position=pos,
        text_description="A friendly animal that barks."
    ),
    'rock': lambda pos: WorldObject(
        name='rock', category='mineral',
        properties=make_properties(
            color_r=0.5, color_g=0.5, color_b=0.5,
            size=0.3, shape_round=0.5, shape_long=0.3,
            animate=0.0, edible=0.0, dangerous=0.1,
            warm=0.2, soft=0.0, bright=0.2, noisy=0.0,
            complexity=0.3
        ),
        position=pos,
        text_description="A hard grey stone."
    ),
    'fire': lambda pos: WorldObject(
        name='fire', category='element',
        properties=make_properties(
            color_r=1.0, color_g=0.5, color_b=0.0,
            size=0.4, shape_round=0.3, shape_long=0.6,
            animate=0.5, edible=0.0, dangerous=0.9,
            warm=1.0, soft=0.0, bright=1.0, noisy=0.4,
            complexity=0.8
        ),
        position=pos,
        text_description="Hot flickering flames."
    ),
    'water': lambda pos: WorldObject(
        name='water', category='element',
        properties=make_properties(
            color_r=0.1, color_g=0.3, color_b=0.9,
            size=0.5, shape_round=0.5, shape_long=0.5,
            animate=0.3, edible=0.5, dangerous=0.1,
            warm=0.3, soft=1.0, bright=0.4, noisy=0.3,
            complexity=0.5
        ),
        position=pos,
        text_description="Clear flowing liquid."
    ),
    'flower': lambda pos: WorldObject(
        name='flower', category='plant',
        properties=make_properties(
            color_r=0.8, color_g=0.2, color_b=0.8,
            size=0.2, shape_round=0.7, shape_long=0.3,
            animate=0.0, edible=0.1, dangerous=0.0,
            warm=0.2, soft=0.8, bright=0.6, noisy=0.0,
            complexity=0.6
        ),
        position=pos,
        text_description="A colorful plant with petals."
    ),
    'ball': lambda pos: WorldObject(
        name='ball', category='toy',
        properties=make_properties(
            color_r=0.2, color_g=0.2, color_b=0.9,
            size=0.3, shape_round=1.0, shape_long=0.0,
            animate=0.0, edible=0.0, dangerous=0.0,
            warm=0.2, soft=0.6, bright=0.5, noisy=0.2,
            complexity=0.1
        ),
        position=pos,
        text_description="A round bouncing toy."
    ),
    'book': lambda pos: WorldObject(
        name='book', category='object',
        properties=make_properties(
            color_r=0.4, color_g=0.2, color_b=0.1,
            size=0.3, shape_round=0.0, shape_long=0.5,
            animate=0.0, edible=0.0, dangerous=0.0,
            warm=0.2, soft=0.3, bright=0.3, noisy=0.0,
            complexity=0.9
        ),
        position=pos,
        text_description="Pages full of symbols and ideas."
    ),
    # Phase 5 variants — same name/class as originals, different properties.
    # name must match the original so OstensiveTeacher teaches the same noun class.
    # Disambiguation must come through property words, not different object names.
    'apple_2': lambda pos: WorldObject(
        name='apple',           # CRITICAL: same name as original for teaching
        category='fruit',
        properties=make_properties(
            color_r=0.9, color_g=0.1, color_b=0.1,
            size=0.3, shape_round=0.9, shape_long=0.1,
            animate=0.0, edible=0.0,    # was 1.0 — poisonous
            dangerous=0.6,              # was 0.0 — above 0.5 threshold
            warm=0.2, soft=0.4, bright=0.5, noisy=0.0,
            complexity=0.2
        ),
        position=pos,
        text_description="A round red fruit. Something seems wrong with it."
    ),
    'cat_2': lambda pos: WorldObject(
        name='cat',             # CRITICAL: same name as original for teaching
        category='animal',
        properties=make_properties(
            color_r=0.5, color_g=0.4, color_b=0.3,
            size=0.4, shape_round=0.4, shape_long=0.6,
            animate=1.0, edible=0.0, dangerous=0.9,  # was 0.1 — feral
            warm=0.7, soft=0.8, bright=0.3, noisy=0.3,
            complexity=0.7
        ),
        position=pos,
        text_description="A small furry animal. It looks hostile."
    ),
}


# =============================================================================
# The Environment
# =============================================================================

class StructuredEnvironment:
    """
    A world of objects with properties, positions, and relationships.
    
    Agents perceive nearby objects as property vectors.
    Humans see a visualization dashboard (rendered separately).
    
    The world is a 2D space [0, world_size] x [0, world_size].
    Objects can be added, removed, and moved.
    """
    
    def __init__(self, world_size: float = 100.0, seed: int = 42):
        self.world_size = world_size
        self.rng = np.random.RandomState(seed)
        
        self.objects: Dict[str, WorldObject] = {}
        self.relations: List[Relation] = []
        
        # History for analysis
        self.event_log: List[Dict] = []
        self.step_count = 0
    
    def add_object(self, obj_name: str, position: Optional[Tuple[float, float]] = None):
        """Add an object from the library at a given position."""
        if obj_name not in OBJECT_LIBRARY:
            raise ValueError(f"Unknown object: {obj_name}. Available: {list(OBJECT_LIBRARY.keys())}")
        
        if position is None:
            position = (
                self.rng.uniform(5, self.world_size - 5),
                self.rng.uniform(5, self.world_size - 5)
            )
        
        # Allow multiple instances with unique keys
        instance_key = obj_name
        counter = 1
        while instance_key in self.objects:
            instance_key = f"{obj_name}_{counter}"
            counter += 1
        
        obj = OBJECT_LIBRARY[obj_name](position)
        # NOTE: Do not overwrite obj.name here. The OstensiveTeacher derives the
        # base class name by stripping numeric suffixes from the dict key
        # (e.g. 'apple_2' -> 'apple') in language_grounding._find_nearby_objects().
        # obj.name is preserved from the OBJECT_LIBRARY lambda for documentation
        # purposes but is not the active teaching routing mechanism.

        # Add slight property noise — no two apples are identical
        noise = self.rng.randn(PROPERTY_DIM) * 0.03
        obj.properties = np.clip(obj.properties + noise, 0.0, 1.0)
        
        self.objects[instance_key] = obj
        self._log_event('object_added', {'name': instance_key, 'position': position})
        
        return instance_key
    
    def add_custom_object(self, name: str, category: str, 
                          position: Tuple[float, float], **properties):
        """Add a custom object with specified properties."""
        props = make_properties(**properties)
        obj = WorldObject(
            name=name, category=category,
            properties=props, position=position
        )
        self.objects[name] = obj
        self._log_event('custom_object_added', {'name': name, 'position': position})
        return name
    
    def add_relation(self, subject: str, predicate: str, obj: str):
        """Add a relationship between two objects."""
        self.relations.append(Relation(subject, predicate, obj))
    
    def move_object(self, name: str, new_position: Tuple[float, float]):
        """Move an object to a new position."""
        if name in self.objects:
            self.objects[name].position = new_position
    
    def remove_object(self, name: str):
        """Remove an object from the world."""
        if name in self.objects:
            del self.objects[name]
            self.relations = [r for r in self.relations 
                             if r.subject != name and r.object != name]
    
    # ---- Agent Perception Interface ----
    
    def perceive_at(self, position: Tuple[float, float], 
                    perception_radius: float = 30.0,
                    max_objects: int = 8) -> Dict[str, Any]:
        """
        What an agent perceives from a given position.
        
        Returns structured data:
        - nearby_objects: list of (distance, property_vector, name) tuples
        - relations: relevant relations between nearby objects
        - n_objects: how many objects are nearby
        
        Objects beyond perception_radius are invisible.
        """
        nearby = []
        for name, obj in self.objects.items():
            dist = self.toroidal_distance(obj.position, position)
            if dist <= perception_radius:
                nearby.append((dist, obj.properties.copy(), name))
        
        # Sort by distance (closest first), limit to max_objects
        nearby.sort(key=lambda x: x[0])
        nearby = nearby[:max_objects]
        
        # Get names of nearby objects for relation filtering
        nearby_names = {item[2] for item in nearby}
        relevant_relations = [
            r for r in self.relations
            if r.subject in nearby_names and r.object in nearby_names
        ]
        
        return {
            'objects': nearby,              # [(distance, properties, name), ...]
            'relations': relevant_relations, 
            'n_objects': len(nearby),
            'position': position,
        }
    
    def get_flat_perception(self, position: Tuple[float, float],
                            perception_radius: float = 30.0,
                            max_objects: int = 8) -> np.ndarray:
        """
        Flattened perception vector for neural network input.
        
        Format: [obj1_dist, obj1_props..., obj2_dist, obj2_props..., ..., n_objects]
        Padded with zeros if fewer than max_objects nearby.
        
        Total dimension: max_objects * (1 + PROPERTY_DIM) + 1
        """
        perception = self.perceive_at(position, perception_radius, max_objects)
        
        slot_size = 1 + PROPERTY_DIM  # distance + properties
        flat = np.zeros(max_objects * slot_size + 1)
        
        for i, (dist, props, name) in enumerate(perception['objects']):
            offset = i * slot_size
            flat[offset] = dist / perception_radius  # Normalize distance to [0, 1]
            flat[offset + 1: offset + 1 + PROPERTY_DIM] = props
        
        flat[-1] = len(perception['objects']) / max_objects  # Normalized object count
        
        return flat
    
    def get_perception_dim(self, max_objects: int = 8, n_utterance_classes: int = 0,
                           n_memory_classes: int = 0,
                           n_property_utterance_classes: int = 0,
                           n_goal_classes: int = 0) -> int:
        """Total dimension of flattened perception vector.

        Args:
            n_utterance_classes: word-emission slots for agent-to-agent communication (Phase 3).
            n_memory_classes: spatial-memory dims (Phase 4).
            n_property_utterance_classes: property utterance slots (Phase 3.5).
            n_goal_classes: goal class token dims for reference game runner (Stage 4).
                            Always present (zeros for non-runner agents).
        """
        return (max_objects * (1 + PROPERTY_DIM) + 1
                + n_utterance_classes
                + n_memory_classes
                + n_property_utterance_classes
                + n_goal_classes)
    
    # ---- World Dynamics ----
    
    def step(self):
        """
        Advance world by one time step.
        
        v2 changes:
        - Animate objects move meaningfully (not 0.5, but 2.0-3.0)
        - ALL objects drift slightly in properties (world is never fully static)
        - Periodic events: objects spawn, despawn, change properties
        - Toroidal wrapping for object positions
        """
        self.step_count += 1
        
        for name, obj in list(self.objects.items()):
            # Animate objects wander meaningfully
            if obj.properties[PROPERTY_SCHEMA['animate']] > 0.5:
                dx = self.rng.randn() * 2.5  # Was 0.5 — now actually noticeable
                dy = self.rng.randn() * 2.5
                new_x = (obj.position[0] + dx) % self.world_size  # Toroidal
                new_y = (obj.position[1] + dy) % self.world_size
                obj.position = (new_x, new_y)
            
            # ALL objects: tiny property drift (nothing is ever perfectly static)
            # This ensures the forward model always has *something* to learn
            drift = self.rng.randn(PROPERTY_DIM) * 0.003
            obj.properties = np.clip(obj.properties + drift, 0.0, 1.0)
        
        # --- Periodic dynamic events ---
        self._maybe_spawn_event()
        self._maybe_despawn_event()
        self._maybe_property_shift_event()
    
    # ---- Dynamic Events (v2) ----
    
    def _maybe_spawn_event(self):
        """Occasionally spawn a new random object."""
        # ~1% chance per step in stage 3, 0% in stage 1-2
        if not hasattr(self, '_dynamic_mode'):
            self._dynamic_mode = False
        if not self._dynamic_mode:
            return
        if self.rng.rand() < 0.01 and len(self.objects) < 15:
            obj_type = self.rng.choice(list(OBJECT_LIBRARY.keys()))
            pos = (self.rng.uniform(5, self.world_size - 5),
                   self.rng.uniform(5, self.world_size - 5))
            key = self.add_object(obj_type, pos)
            self._log_event('dynamic_spawn', {'name': key, 'position': pos})
    
    def _maybe_despawn_event(self):
        """Occasionally remove a random object."""
        if not getattr(self, '_dynamic_mode', False):
            return
        if self.rng.rand() < 0.008 and len(self.objects) > 4:
            keys = list(self.objects.keys())
            victim = self.rng.choice(keys)
            self._log_event('dynamic_despawn', {'name': victim})
            self.remove_object(victim)
    
    def _maybe_property_shift_event(self):
        """Occasionally give an object a noticeable property change."""
        if not getattr(self, '_dynamic_mode', False):
            return
        if self.rng.rand() < 0.02 and self.objects:
            keys = list(self.objects.keys())
            target = self.rng.choice(keys)
            obj = self.objects[target]
            # Pick 1-3 random properties to shift significantly
            n_props = self.rng.randint(1, 4)
            prop_indices = self.rng.choice(PROPERTY_DIM, n_props, replace=False)
            for idx in prop_indices:
                shift = self.rng.randn() * 0.15  # Noticeable but not extreme
                obj.properties[idx] = np.clip(obj.properties[idx] + shift, 0.0, 1.0)
            self._log_event('property_shift', {
                'name': target, 
                'properties_changed': prop_indices.tolist()
            })
    
    def enable_dynamic_mode(self):
        """Turn on spawn/despawn/shift events."""
        self._dynamic_mode = True
        self._log_event('dynamic_mode_enabled', {})
    
    def disable_dynamic_mode(self):
        """Turn off spawn/despawn/shift events."""
        self._dynamic_mode = False
    
    def toroidal_distance(self, pos_a: Tuple[float, float], 
                          pos_b: Tuple[float, float]) -> float:
        """Distance on a toroidal (wrapping) world."""
        dx = abs(pos_a[0] - pos_b[0])
        dy = abs(pos_a[1] - pos_b[1])
        dx = min(dx, self.world_size - dx)
        dy = min(dy, self.world_size - dy)
        return np.sqrt(dx**2 + dy**2)
    
    # ---- Curriculum: Staged Complexity ----
    
    def setup_stage_1(self):
        """
        Simplest environment: 3-4 distinct objects, well-separated.
        Good for initial perception and curiosity calibration.
        """
        self.objects.clear()
        self.relations.clear()
        
        self.add_object('apple', (25, 25))
        self.add_object('rock', (75, 25))
        self.add_object('ball', (25, 75))
        self.add_object('flower', (75, 75))
        
        self._log_event('stage_setup', {'stage': 1, 'n_objects': 4})
    
    def setup_stage_2(self):
        """
        More objects, some similar (apple + banana = both fruit).
        Tests generalization and category formation.
        """
        self.objects.clear()
        self.relations.clear()
        
        self.add_object('apple', (20, 20))
        self.add_object('banana', (40, 20))
        self.add_object('cat', (60, 20))
        self.add_object('dog', (80, 20))
        self.add_object('rock', (20, 60))
        self.add_object('water', (40, 60))
        self.add_object('fire', (60, 60))
        self.add_object('book', (80, 60))
        
        # Add some relations
        self.add_relation('cat', 'near', 'dog')
        self.add_relation('fire', 'near', 'water')
        
        self._log_event('stage_setup', {'stage': 2, 'n_objects': 8})
    
    def setup_stage_3(self):
        """
        Dynamic environment: objects move, appear, disappear.
        Tests adaptation and continuous learning.
        v2: enables dynamic mode for spawn/despawn/property shift events.
        """
        self.setup_stage_2()
        # Add more objects and let animate ones move
        self.add_object('flower', (50, 50))
        self.add_object('ball', (30, 80))

        # v2: enable dynamic events so world never becomes fully predictable
        self.enable_dynamic_mode()

        self._log_event('stage_setup', {'stage': 3, 'n_objects': len(self.objects)})

    def spawn_objects(self):
        """
        Spawn the full Phase 5 base object set at random positions.

        Spawns all 10 original objects plus the two Phase 5 property-varying
        variants (apple_2 / cat_2), giving 12 objects total.  The OstensiveTeacher
        routes both 'apple' and 'apple_2' to the same noun class by stripping
        the numeric suffix from the dict key ('apple_2' -> 'apple') inside
        language_grounding._find_nearby_objects(). The WorldObject.name field
        is documentation only and is not read by the teaching system.

        Dynamic mode is intentionally not enabled here; Phase 5 uses a stable
        12-object world for controlled disambiguation training. Random
        spawn/despawn events would interfere with the property-contrasting
        pair setup.
        """
        self.disable_dynamic_mode()   # Phase 5: stable world — no spawn/despawn events
        self.objects.clear()
        self.relations.clear()

        base_objects = [
            'flower', 'rock', 'apple', 'ball', 'book',
            'dog', 'fire', 'cat', 'water', 'banana',
            'apple_2', 'cat_2',   # Phase 5 variants
        ]

        for obj_name in base_objects:
            self.add_object(obj_name)

        self._log_event('spawn_objects', {'n_objects': len(self.objects)})
    
    # ---- Serialization ----
    
    def get_state(self) -> Dict:
        """Full serializable state snapshot."""
        return {
            'step': self.step_count,
            'objects': {
                name: {
                    'name': obj.name,
                    'category': obj.category,
                    'position': obj.position,
                    'properties': obj.properties.tolist(),
                    'text': obj.text_description,
                }
                for name, obj in self.objects.items()
            },
            'relations': [
                {'subject': r.subject, 'predicate': r.predicate, 'object': r.object}
                for r in self.relations
            ]
        }
    
    def _log_event(self, event_type: str, data: Dict):
        self.event_log.append({
            'step': self.step_count,
            'type': event_type,
            'data': data
        })
    
    def __repr__(self):
        return (f"StructuredEnvironment(objects={len(self.objects)}, "
                f"relations={len(self.relations)}, step={self.step_count})")
