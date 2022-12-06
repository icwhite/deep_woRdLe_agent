class BaseExplorationModel(object):
    def compute_bonus(self, state: dict, action: str, guess_count: int):
        return 0