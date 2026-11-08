"""Gates for work that belongs to a later phase."""


class PhaseNotReady(RuntimeError):
    def __init__(self, feature: str, phase: int) -> None:
        super().__init__(f"{feature} is implemented in phase {phase}.")
        self.feature = feature
        self.phase = phase
