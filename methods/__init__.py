from dataclasses import dataclass
from typing import List


@dataclass
class Agent:
    name: str
    role: str


def default_agents(num_agents: int = 4) -> List[Agent]:
    if num_agents == 2:
        return [Agent(name="Planner", role="planner"), Agent(name="Judger", role="judger")]
    if num_agents != 4:
        raise ValueError("num_agents must be 2 or 4")
    return [
        Agent(name="Planner", role="planner"),
        Agent(name="Critic", role="critic"),
        Agent(name="Refiner", role="refiner"),
        Agent(name="Judger", role="judger"),
    ]


__all__ = ["Agent", "default_agents"]
