from .evaluator import Evaluator

EVALUATORS = dict([(evaluator.name(), evaluator) for evaluator in Evaluator.__subclasses__()])
