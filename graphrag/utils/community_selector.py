import math
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from tqdm import tqdm
from graphrag.models.graph_types import Community


class CommunitySelector:
    """
    Optimizes community selection using Simulated Annealing
    Balances relevance to query and diversity among communities
    """

    def __init__(self,
                 query_embedding: np.ndarray,
                 community_embeddings: List[np.ndarray],
                 communities: List[Community],
                 k: int = 3,
                 relevance_weight: float = 0.7,
                 diversity_weight: float = 0.3,
                 initial_temp: float = 1.0,
                 cooling_rate: float = 0.95,
                 max_iter: int = 200):
        """
        Initialize the community selector

        Args:
            query_embedding: Embedding vector of the query
            community_embeddings: List of embedding vectors for communities
            communities: List of Community objects
            k: Number of communities to select
            relevance_weight: Weight for query relevance in fitness
            diversity_weight: Weight for set diversity in fitness
            initial_temp: Starting temperature for annealing
            cooling_rate: Cooling rate per iteration
            max_iter: Maximum iterations
        """
        self.query_embedding = query_embedding
        self.community_embeddings = np.array(community_embeddings)
        self.communities = communities
        self.k = k
        self.relevance_weight = relevance_weight
        self.diversity_weight = diversity_weight
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter

        # Precompute similarities
        self.similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            self.community_embeddings
        ).flatten()

    def fitness(self, selected_indices: set) -> float:
        """
        Calculate fitness of a community set
        Higher is better (relevance + diversity)
        """
        selected_embeddings = self.community_embeddings[list(selected_indices)]

        # Relevance: Average similarity to query
        rel_score = np.mean(self.similarities[list(selected_indices)])

        # Diversity: Intra-set dissimilarity (1 - average similarity)
        if len(selected_indices) > 1:
            pairwise_sims = cosine_similarity(selected_embeddings)
            np.fill_diagonal(pairwise_sims, 0)
            div_score = 1 - np.sum(pairwise_sims) / (len(selected_indices) * (len(selected_indices) - 1))
        else:
            div_score = 0.5  # Neutral score for single community

        return (self.relevance_weight * rel_score +
                self.diversity_weight * div_score)

    def generate_neighbor(self, current_set: set) -> set:
        """
        Create a neighbor solution by swapping one community
        """
        neighbor = current_set.copy()

        # Remove a random member
        if len(neighbor) > 0:
            remove_idx = random.choice(list(neighbor))
            neighbor.remove(remove_idx)

            # Add a new community not in the set
            available = set(range(len(self.communities))) - neighbor
            if available:
                add_idx = random.choice(list(available))
                neighbor.add(add_idx)

        return neighbor

    def optimize(self) -> List[Community]:
        """
        Run simulated annealing optimization
        Returns optimized set of communities
        """
        # Initialize with top-k by relevance
        current_set = set(np.argsort(self.similarities)[-self.k:])
        current_energy = self.fitness(current_set)

        best_set = current_set.copy()
        best_energy = current_energy

        temp = self.initial_temp
        no_improve_count = 0

        # Annealing progress bar
        pbar = tqdm(total=self.max_iter, desc="Optimizing communities")

        for i in range(self.max_iter):
            # Generate neighbor solution
            candidate_set = self.generate_neighbor(current_set)
            candidate_energy = self.fitness(candidate_set)

            # Acceptance probability
            delta = candidate_energy - current_energy
            accept_prob = math.exp(delta / temp) if delta < 0 else 1.0

            # Update state
            if random.random() < accept_prob:
                current_set = candidate_set
                current_energy = candidate_energy

                # Update best solution
                if candidate_energy > best_energy:
                    best_set = candidate_set
                    best_energy = candidate_energy
                    no_improve_count = 0
                else:
                    no_improve_count += 1

            # Cooling and early stopping
            temp *= self.cooling_rate
            pbar.update(1)
            pbar.set_postfix({
                "temp": f"{temp:.4f}",
                "energy": f"{best_energy:.4f}",
                "no_improve": no_improve_count
            })

            if no_improve_count > 20 and temp < 0.1:
                break

        pbar.close()
        return [self.communities[i] for i in best_set]