"""Tests for session_metadata passthrough to spawn_fn (CP-5 Part 3).

Verifies that:
- spawn_fn receives session_metadata kwarg with recipe_name, step_id, agent_name
- session_metadata includes recipe_step_index
- parallel_group_id is included for parallel foreach spawns
- all spawns in one parallel batch share the same parallel_group_id
- different parallel batches get different parallel_group_ids
- sequential foreach spawns do NOT include parallel_group_id
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from amplifier_module_tool_recipes.executor import RecipeExecutor
from amplifier_module_tool_recipes.models import Recipe
from amplifier_module_tool_recipes.models import Step


@pytest.fixture
def mock_coordinator():
    """Create a mock coordinator with async spawn capability."""
    coordinator = MagicMock()
    coordinator.session = MagicMock()
    coordinator.config = {"agents": {}}
    coordinator.hooks = None  # Prevent MagicMock from being awaited in _show_progress
    coordinator.get_capability.return_value = AsyncMock()
    return coordinator


@pytest.fixture
def mock_session_manager():
    """Create a mock session manager."""
    manager = MagicMock()
    manager.create_session.return_value = "test-session-id"
    manager.load_state.return_value = {
        "current_step_index": 0,
        "context": {},
        "completed_steps": [],
        "started": "2025-01-01T00:00:00",
    }
    manager.is_cancellation_requested.return_value = False
    manager.is_immediate_cancellation.return_value = False
    return manager


class TestSessionMetadataOnSpawn:
    """Tests that spawn_fn receives session_metadata with recipe context."""

    @pytest.mark.asyncio
    async def test_session_metadata_passed_to_spawn_fn(
        self, mock_coordinator, mock_session_manager, temp_dir
    ):
        """spawn_fn receives session_metadata kwarg with recipe_name, step_id, agent_name."""
        mock_spawn = mock_coordinator.get_capability.return_value
        mock_spawn.return_value = "result"

        executor = RecipeExecutor(mock_coordinator, mock_session_manager)
        recipe = Recipe(
            name="my-recipe",
            description="test",
            version="1.0.0",
            steps=[
                Step(
                    id="analyze",
                    agent="code-analyzer",
                    prompt="Analyze {{input}}",
                    output="result",
                ),
            ],
            context={"input": "test-data"},
        )

        await executor.execute_recipe(recipe, {}, temp_dir)

        assert mock_spawn.called
        call_kwargs = mock_spawn.call_args.kwargs
        assert "session_metadata" in call_kwargs
        metadata = call_kwargs["session_metadata"]
        assert metadata["recipe_name"] == "my-recipe"
        assert metadata["recipe_step"] == "analyze"
        assert metadata["agent_name"] == "code-analyzer"

    @pytest.mark.asyncio
    async def test_session_metadata_includes_step_index(
        self, mock_coordinator, mock_session_manager, temp_dir
    ):
        """session_metadata includes recipe_step_index matching step position."""
        mock_spawn = mock_coordinator.get_capability.return_value
        mock_spawn.return_value = "result"

        executor = RecipeExecutor(mock_coordinator, mock_session_manager)
        recipe = Recipe(
            name="indexed-recipe",
            description="test",
            version="1.0.0",
            steps=[
                Step(id="step-zero", agent="agent-a", prompt="Step 0", output="r0"),
                Step(id="step-one", agent="agent-b", prompt="Step 1", output="r1"),
            ],
            context={},
        )

        await executor.execute_recipe(recipe, {}, temp_dir)

        calls = mock_spawn.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs["session_metadata"]["recipe_step_index"] == 0
        assert calls[1].kwargs["session_metadata"]["recipe_step_index"] == 1

    @pytest.mark.asyncio
    async def test_session_metadata_correct_recipe_name_per_step(
        self, mock_coordinator, mock_session_manager, temp_dir
    ):
        """All steps in a recipe report the same recipe_name."""
        mock_spawn = mock_coordinator.get_capability.return_value
        mock_spawn.side_effect = ["r1", "r2"]

        executor = RecipeExecutor(mock_coordinator, mock_session_manager)
        recipe = Recipe(
            name="multi-step-recipe",
            description="test",
            version="1.0.0",
            steps=[
                Step(id="step-a", agent="agent-x", prompt="Step A", output="ra"),
                Step(id="step-b", agent="agent-y", prompt="Step B", output="rb"),
            ],
            context={},
        )

        await executor.execute_recipe(recipe, {}, temp_dir)

        calls = mock_spawn.call_args_list
        assert len(calls) == 2
        for c in calls:
            assert c.kwargs["session_metadata"]["recipe_name"] == "multi-step-recipe"

    @pytest.mark.asyncio
    async def test_session_metadata_no_recipe_context_still_works(
        self, mock_coordinator, mock_session_manager, temp_dir
    ):
        """execute_step called without recipe context in context dict still works."""
        mock_spawn = mock_coordinator.get_capability.return_value
        mock_spawn.return_value = "result"

        executor = RecipeExecutor(mock_coordinator, mock_session_manager)

        # Call execute_step directly with an empty context (no "recipe" key)
        step = Step(id="standalone", agent="helper", prompt="Do something")
        await executor.execute_step(step, {})

        # spawn_fn should still be called successfully
        assert mock_spawn.called
        # session_metadata should be passed (with empty/default values) not absent
        call_kwargs = mock_spawn.call_args.kwargs
        assert "session_metadata" in call_kwargs
        # With no recipe context, recipe_name should be empty string
        assert call_kwargs["session_metadata"]["recipe_name"] == ""


class TestParallelGroupId:
    """Tests for parallel_group_id in parallel foreach spawns."""

    @pytest.mark.asyncio
    async def test_parallel_group_id_included_in_metadata(
        self, mock_coordinator, mock_session_manager, temp_dir
    ):
        """All parallel foreach spawns include parallel_group_id in session_metadata."""
        mock_spawn = mock_coordinator.get_capability.return_value
        mock_spawn.side_effect = ["r1", "r2", "r3"]

        executor = RecipeExecutor(mock_coordinator, mock_session_manager)
        recipe = Recipe(
            name="parallel-recipe",
            description="test",
            version="1.0.0",
            steps=[
                Step(
                    id="parallel-step",
                    agent="worker",
                    prompt="Process {{item}}",
                    foreach="{{items}}",
                    parallel=True,
                    collect="results",
                ),
            ],
            context={"items": ["a", "b", "c"]},
        )

        await executor.execute_recipe(recipe, {}, temp_dir)

        calls = mock_spawn.call_args_list
        assert len(calls) == 3
        for c in calls:
            assert "session_metadata" in c.kwargs
            assert "parallel_group_id" in c.kwargs["session_metadata"]
            # group_id should be a non-empty string
            assert c.kwargs["session_metadata"]["parallel_group_id"]

    @pytest.mark.asyncio
    async def test_parallel_group_id_same_within_batch(
        self, mock_coordinator, mock_session_manager, temp_dir
    ):
        """All spawns within one parallel batch share the same parallel_group_id."""
        mock_spawn = mock_coordinator.get_capability.return_value
        mock_spawn.side_effect = ["r1", "r2", "r3"]

        executor = RecipeExecutor(mock_coordinator, mock_session_manager)
        recipe = Recipe(
            name="parallel-recipe",
            description="test",
            version="1.0.0",
            steps=[
                Step(
                    id="parallel-step",
                    agent="worker",
                    prompt="Process {{item}}",
                    foreach="{{items}}",
                    parallel=True,
                    collect="results",
                ),
            ],
            context={"items": ["a", "b", "c"]},
        )

        await executor.execute_recipe(recipe, {}, temp_dir)

        calls = mock_spawn.call_args_list
        group_ids = [c.kwargs["session_metadata"]["parallel_group_id"] for c in calls]
        # All iterations in same batch share one group_id
        assert len(set(group_ids)) == 1
        assert group_ids[0]

    @pytest.mark.asyncio
    async def test_different_parallel_batches_get_different_group_ids(
        self, mock_coordinator, mock_session_manager, temp_dir
    ):
        """Two separate parallel steps each get their own unique parallel_group_id."""
        mock_spawn = mock_coordinator.get_capability.return_value
        mock_spawn.side_effect = ["r1", "r2", "r3", "r4"]

        executor = RecipeExecutor(mock_coordinator, mock_session_manager)
        recipe = Recipe(
            name="multi-parallel-recipe",
            description="test",
            version="1.0.0",
            steps=[
                Step(
                    id="parallel-step-1",
                    agent="worker",
                    prompt="Process {{item}}",
                    foreach="{{items1}}",
                    parallel=True,
                    collect="results1",
                ),
                Step(
                    id="parallel-step-2",
                    agent="worker",
                    prompt="Process {{item}}",
                    foreach="{{items2}}",
                    parallel=True,
                    collect="results2",
                ),
            ],
            context={"items1": ["a", "b"], "items2": ["c", "d"]},
        )

        await executor.execute_recipe(recipe, {}, temp_dir)

        calls = mock_spawn.call_args_list
        assert len(calls) == 4

        batch1_group_ids = {
            calls[0].kwargs["session_metadata"]["parallel_group_id"],
            calls[1].kwargs["session_metadata"]["parallel_group_id"],
        }
        batch2_group_ids = {
            calls[2].kwargs["session_metadata"]["parallel_group_id"],
            calls[3].kwargs["session_metadata"]["parallel_group_id"],
        }

        # Within each batch, same group_id
        assert len(batch1_group_ids) == 1
        assert len(batch2_group_ids) == 1
        # Between batches, different group_ids
        assert batch1_group_ids != batch2_group_ids

    @pytest.mark.asyncio
    async def test_sequential_foreach_no_parallel_group_id(
        self, mock_coordinator, mock_session_manager, temp_dir
    ):
        """Sequential foreach spawns do NOT include parallel_group_id."""
        mock_spawn = mock_coordinator.get_capability.return_value
        mock_spawn.side_effect = ["r1", "r2"]

        executor = RecipeExecutor(mock_coordinator, mock_session_manager)
        recipe = Recipe(
            name="sequential-recipe",
            description="test",
            version="1.0.0",
            steps=[
                Step(
                    id="sequential-step",
                    agent="worker",
                    prompt="Process {{item}}",
                    foreach="{{items}}",
                    collect="results",
                ),
            ],
            context={"items": ["a", "b"]},
        )

        await executor.execute_recipe(recipe, {}, temp_dir)

        calls = mock_spawn.call_args_list
        assert len(calls) == 2
        for c in calls:
            assert "session_metadata" in c.kwargs
            assert "parallel_group_id" not in c.kwargs["session_metadata"]
