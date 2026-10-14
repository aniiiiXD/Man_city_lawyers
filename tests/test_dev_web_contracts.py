import pytest

from dev_web.contracts import AgentTask, ContractError, TaskGraph


def task(task_id: str, dependencies: tuple[str, ...] = ()) -> AgentTask:
    return AgentTask(
        task_id=task_id,
        title=task_id,
        instructions="Implement a bounded change.",
        file_scopes=("src/",),
        dependencies=dependencies,
        required_checks=("pytest",),
    )


def test_task_graph_orders_dag_and_returns_only_ready_tasks():
    graph = TaskGraph(tasks=(task("first"), task("second", ("first",))))

    assert graph.topological_order() == ("first", "second")
    assert graph.ready_task_ids(()) == ("first",)
    assert graph.ready_task_ids(("first",)) == ("second",)


def test_task_graph_rejects_cycles_and_unknown_dependencies():
    with pytest.raises(ContractError, match="cycle"):
        TaskGraph(tasks=(task("first", ("second",)), task("second", ("first",))))

    with pytest.raises(ContractError, match="unknown dependencies"):
        TaskGraph(tasks=(task("first", ("missing",)),))


def test_task_contract_rejects_parent_traversal_scope():
    with pytest.raises(ContractError, match="Unsafe file scope"):
        AgentTask(
            task_id="bad", title="bad", instructions="bad", file_scopes=("../private/",),
            required_checks=("pytest",),
        )
