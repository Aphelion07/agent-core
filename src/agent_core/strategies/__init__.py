from ..agent import Strategy
from .plan_execute import PlanExecuteStrategy
from .react import ReActStrategy
from .reflexion import ReflexionStrategy

STRATEGIES: dict[str, type[Strategy]] = {
    ReActStrategy.name: ReActStrategy,
    PlanExecuteStrategy.name: PlanExecuteStrategy,
    ReflexionStrategy.name: ReflexionStrategy,
}

__all__ = ["STRATEGIES", "PlanExecuteStrategy", "ReActStrategy", "ReflexionStrategy"]
