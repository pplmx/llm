"""Tests for checkpoint contributor helpers."""

from llm.runtime.checkpoint import CheckpointContributor, collect_extra_state, load_extra_state


class _ContributorA:
    def get_checkpoint_state(self):
        return {"stream_data": {"0": {"line_index": 1}}}

    def load_checkpoint_state(self, state):
        self.loaded = state["stream_data"]["0"]["line_index"]


class _ContributorB:
    def get_checkpoint_state(self):
        return {"ppo": {"global_step": 7}}

    def load_checkpoint_state(self, state):
        self.step = state["ppo"]["global_step"]


def test_collect_extra_state_merges_fragments():
    merged = collect_extra_state(_ContributorA(), _ContributorB())
    assert merged == {"stream_data": {"0": {"line_index": 1}}, "ppo": {"global_step": 7}}


def test_collect_extra_state_skips_non_contributors():
    assert collect_extra_state(object(), _ContributorA()) == {"stream_data": {"0": {"line_index": 1}}}


def test_load_extra_state_dispatches_to_contributors():
    contributor = _ContributorA()
    load_extra_state({"stream_data": {"0": {"line_index": 42}}}, contributor)
    assert contributor.loaded == 42


def test_training_task_is_checkpoint_contributor():
    from llm.training.tasks.base_task import TrainingTask

    assert issubclass(TrainingTask, CheckpointContributor)
