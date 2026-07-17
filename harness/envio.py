"""Environment wrapper: real ARC-AGI-3 game <-> timeline."""

from arc_agi import Arcade
from arcengine import GameAction

SIMPLE = {
    "1": GameAction.ACTION1, "2": GameAction.ACTION2, "3": GameAction.ACTION3,
    "4": GameAction.ACTION4, "5": GameAction.ACTION5, "7": GameAction.ACTION7,
}


class Env:
    def __init__(self, game_id, timeline, max_actions=2000):
        self.arcade = Arcade()
        self.env = self.arcade.make(game_id, save_recording=True)
        if self.env is None:
            raise RuntimeError(f"could not make environment {game_id}")
        self.timeline = timeline
        self.max_actions = max_actions
        self.last = None

    @staticmethod
    def _grid(frame):
        return [list(map(int, row)) for row in frame.frame[-1]]

    def _observe(self, frame, action):
        self.last = frame
        return self.timeline.append(
            action=action,
            grid=self._grid(frame),
            level=frame.levels_completed,
            state=frame.state.name,
            full_reset=getattr(frame, "full_reset", False),
        )

    def reset(self):
        f = self.env.reset()
        # initial observation costs no action slot in our log (action=None)
        # but subsequent RESETs are recorded as actions by act()
        return self._observe(f, None)

    def act(self, action_str):
        """action_str: '1'..'5', '7', 'RESET', '6@x,y'. Returns the new event."""
        if self.timeline.action_count >= self.max_actions:
            raise BudgetExceeded(f"action budget {self.max_actions} exhausted")
        if action_str == "RESET":
            f = self.env.reset()
        elif action_str.startswith("6@"):
            x, y = action_str[2:].split(",")
            ga = GameAction.ACTION6
            ga.set_data({"x": int(x), "y": int(y)})
            f = self.env.step(ga)
        else:
            f = self.env.step(SIMPLE[action_str])
        return self._observe(f, action_str)

    @property
    def available_actions(self):
        return [str(a) for a in (self.last.available_actions or [])]

    @property
    def state(self):
        return self.last.state.name

    @property
    def level(self):
        return self.last.levels_completed

    @property
    def win_levels(self):
        return self.last.win_levels


class BudgetExceeded(Exception):
    pass
