"""The seeing cortex — the percept, the narration of a sense (design.md §2a).

Perception is not one thing. One stage turns raw input into a SENSE: the
holistic recognition that says "that one" and can drive a hand to the right
ball. A separate stage turns that sense into a PERCEPT you can report -- the
conscious "I see a red ball." §2a already says it: "a concept is a sense...
recognition prior to any description; the description is post-hoc narration of
the sense, which is why it's always lossy."

Blindsight is those two stages coming apart. Damage the report stage and the
sense still fires: the person says they see nothing, yet reaches to the correct
colored ball and tracks a moving target with their eyes. Recognition-for-action
survives; the narration does not.

This region is that narration stage, and nothing else. It holds no concepts and
does no recognition -- it is HANDED a sense (a concept, the raw vector, the
deviations) and renders the percept. Because it is separate, it can be
`lesion()`-ed: recognition (brain.nearest / brain.explain) keeps working and
keeps selecting the right thing, while the percept goes dark. That is the
computational shadow of blindsight, and the reason perception is split here.

It takes the feature schema and the descriptor-naming function as arguments, so
it never imports the brain back (no cycle); the naming helper stays where the
rest of the code already imports it from.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

NO_PERCEPT = "(no percept)"


class Seeing:
    def __init__(self):
        self._blind = False     # lesioned? recognition survives; the report does not

    def lesion(self) -> None:
        """Damage the report pathway. The sense is untouched (that lives in the
        recognition stage); only the conscious percept goes dark."""
        self._blind = True

    def heal(self) -> None:
        self._blind = False

    @property
    def blind(self) -> bool:
        return self._blind

    def narrate(self, concept, x, deviations: Sequence[Tuple],
                features: List[str], name: Callable) -> str:
        """Render the sense into a reportable percept: the concept's handle plus
        a descriptor word for each feature that deviates. Lesioned -> no percept,
        though the sense that produced `concept` still stands."""
        if self._blind:
            return NO_PERCEPT
        words = [concept.label] + [
            name(d, x[features.index(d)], concept.proto[features.index(d)])
            for d, _ in deviations
        ]
        return " + ".join(words)
