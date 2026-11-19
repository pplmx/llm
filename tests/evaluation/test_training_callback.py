import pytest

from llm.evaluation.runner import EvaluationRunner
from llm.training.core.callbacks import EvaluationCallback


def test_evaluation_callback_stores_eval_interval():
    callback = EvaluationCallback(eval_runner=EvaluationRunner, eval_interval=100)
    assert callback.eval_interval == 100


# --- a zero interval must be refused AT THE BOUNDARY ----------------------
#
# ``on_train_step_end`` computes ``global_step % self.eval_interval``, so a zero
# interval raises ZeroDivisionError as soon as a step is recorded. Today's only
# construction site (``build_periodic_eval_callback``) returns None for a falsy
# interval, which makes the crash unreachable — but that guard lives in another
# module, so a future caller passing 0 would get a ZeroDivisionError whose
# traceback points at the callback rather than at the dispatching code.


def test_a_zero_interval_is_refused_at_construction():
    """Constructing the callback with 0 must fail immediately, not at first step."""
    from llm.training.core.callbacks import EvaluationCallback

    with pytest.raises(ValueError, match=r"eval_interval must be >= 1, got 0"):
        EvaluationCallback(eval_runner=None, eval_interval=0)


def test_a_negative_interval_is_refused_too():
    """The guard is `< 1`, not `== 0`, so negatives are covered."""
    from llm.training.core.callbacks import EvaluationCallback

    with pytest.raises(ValueError, match=r"eval_interval must be >= 1"):
        EvaluationCallback(eval_runner=None, eval_interval=-5)


def test_the_refusal_explains_the_config_level_off_switch():
    """A user disabling eval via config must not think they need interval >= 1.

    ``training.eval_interval = 0`` is the documented way to turn periodic eval
    off, and that path never constructs this callback — so the message has to say
    so, or it reads as "you cannot disable evaluation".
    """
    from llm.training.core.callbacks import EvaluationCallback

    with pytest.raises(ValueError, match=r"eval_interval must be") as excinfo:
        EvaluationCallback(eval_runner=None, eval_interval=0)

    assert "CONFIG level" in str(excinfo.value)


def test_a_positive_interval_still_works():
    """The guard must not reject the legitimate default path."""
    from llm.training.core.callbacks import EvaluationCallback

    callback = EvaluationCallback(eval_runner=None, eval_interval=1000)

    assert callback.eval_interval == 1000
