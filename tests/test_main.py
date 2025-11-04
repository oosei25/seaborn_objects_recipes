import os

import numpy as np
import pandas as pd
import pytest
import seaborn as sns
import seaborn.objects as so
from seaborn._core.groupby import GroupBy

import seaborn_objects_recipes as sor


@pytest.fixture
def sample_data():
    # Parameters for simulation
    game = "ExampleGame"
    agents = ["Agent1", "Agent2", "Agent3"]
    num_iterations = 200
    num_agents = len(agents)

    # Create a simulated DataFrame
    np.random.seed(0)  # For reproducible results
    data = {
        "Game": [game] * num_iterations * num_agents,
        "Episodic Return": np.random.rand(num_iterations * num_agents) * 100,
        "Iteration": list(range(num_iterations)) * num_agents,
        "Agent": np.repeat(agents, num_iterations),
    }

    return pd.DataFrame(data)


@pytest.fixture
def cleanup_files():
    # This will run after each test function to which it is applied.
    yield
    # Cleanup code to delete files after test runs
    if os.path.exists("line_label.png"):
        os.remove("line_label.png")
    if os.path.exists("lowess_b.png"):
        os.remove("lowess_b.png")
    if os.path.exists("lowess_gen.png"):
        os.remove("lowess_gen.png")
    if os.path.exists("lowess_nb.png"):
        os.remove("lowess_nb.png")
    if os.path.exists("polyfit_with_ci.png"):
        os.remove("polyfit_with_ci.png")


# Use the sample_data fixture to provide data to the test function
# and the cleanup_files fixture to clean up after the test
def test_line_label(sample_data, cleanup_files):
    game = "ExampleGame"
    fd_data = sample_data.query(f'`Game` == "{game}"')

    (
        fd_data.pipe(
            so.Plot, y="Episodic Return", x="Iteration", color="Agent", text="Agent"
        )
        .layout(size=(16, 8))
        # .theme(theme)
        .facet("Game")
        .limit(x=(0, 200))
        .scale(
            x=so.Continuous().tick(at=list(range(0, 201, 25))),
            y=so.Continuous().tick(upto=5).label(like="{x:,.0f}"),
        )
        .add(
            so.Lines(),
            so.Agg(),
            rolling := sor.Rolling(window_type="gaussian", window_kwargs={"std": 2}),
            legend=False,
        )
        # This will be much easier when compound marks are implemented.
        # Will be able to do so.Line() + LineLabel().
        .add(
            sor.LineLabel(offset=5),
            so.Agg(),
            rolling,
            legend=False,
        )
        .save("line_label.png")
        # .show()
    )
    # Assert that the file was created
    assert os.path.exists(
        "line_label.png"
    ), "The plot file line_label.png was not created."


def test_lowess_with_ci_gen(cleanup_files):
    # Generate data for testing
    np.random.seed(0)
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x) + np.random.normal(size=100) * 0.2
    data = pd.DataFrame({"x": x, "y": y})

    # Create the plot
    plot = (
        so.Plot(data, x="x", y="y")
        .add(so.Dot())
        .add(
            so.Line(),
            lowess := sor.Lowess(frac=0.2, gridsize=100, num_bootstrap=200, alpha=0.95),
        )
        .add(so.Band(), lowess)
        .label(
            x="x-axis",
            y="y-axis",
            title="Lowess Plot with Confidence Intervals - Generated Data",
        )
    )

    # Save the plot
    plot.save("lowess_gen.png")

    # Assert that the file was created
    assert os.path.exists("lowess_gen.png"), "The plot file lowess.png was not created."


def test_lowess_with_ci(cleanup_files):
    # Load the penguins dataset
    penguins = sns.load_dataset("penguins")

    # Create the plot
    plot = (
        so.Plot(penguins, x="bill_length_mm", y="body_mass_g", color="species")
        .add(so.Dot())
        .add(
            so.Line(),
            lowess := sor.Lowess(frac=0.2, gridsize=100, num_bootstrap=200, alpha=0.95),
        )
        .add(so.Band(), lowess)
        .label(
            x="Bill Length (mm)",
            y="Body Mass (g)",
            title="Lowess Plot with Confidence Intervals",
        )
    )

    # Save Plot
    plot.save("lowess_b.png")

    # Assert that the file was created
    assert os.path.exists("lowess_b.png"), "The plot file lowess.png was not created."


def test_lowess_with_no_ci(cleanup_files):
    # Load the penguins dataset
    penguins = sns.load_dataset("penguins")

    # Prepare data
    data = penguins[penguins["species"] == "Adelie"]

    # Create the plot
    plot = (
        so.Plot(data, x="bill_length_mm", y="body_mass_g")
        .add(so.Dot())
        .add(so.Line(), sor.Lowess())
        .label(
            x="Bill Length (mm)",
            y="Body Mass (g)",
            title="Lowess Plot no Confidence Intervals",
        )
    )
    # Save Plot
    plot.save("lowess_nb.png")

    # Assert that the file was created
    assert os.path.exists("lowess_nb.png"), "The plot file lowess.png was not created."


def test_polyfit_with_ci(cleanup_files):
    # Load the penguins dataset
    penguins = sns.load_dataset("penguins")

    # Prepare data
    data = penguins.copy()
    data = data[data["species"] == "Adelie"]

    # Create the plot
    plot = (
        so.Plot(data, x="bill_length_mm", y="body_mass_g")
        .add(so.Dot())
        .add(
            so.Line(),
            PolyFitWithCI := sor.PolyFitWithCI(order=2, gridsize=100, alpha=0.05),
        )
        .add(so.Band(), PolyFitWithCI)
        .label(
            x="Bill Length (mm)",
            y="Body Mass (g)",
            title="PolyFit Plot with Confidence Intervals",
        )
    )
    # Save Plot
    plot.save("polyfit_with_ci.png")
    # Assert that the file was created
    assert os.path.exists(
        "polyfit_with_ci.png"
    ), "The plot file lowess.png was not created."


# Helper
def _eval_stat(stat, df, x, y, orient="x"):
    """Run a seaborn.objects Stat on an (x, y) dataframe with no real grouping."""
    df_xy = df.rename(columns={x: "x", y: "y"}).dropna(subset=["x", "y"])
    fake_groupby = GroupBy(["__dummy__"])  # not present in df_xy -> no grouping
    return stat(df_xy, fake_groupby, orient, {})


# This covers “auto-enable bootstrapping when alpha != 0.95” logic.
def test_lowess_auto_bootstrap_when_alpha_changed():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 80)
    y = np.sin(6 * x) + 0.05 * rng.normal(size=x.size)
    df = pd.DataFrame({"x": x, "y": y})

    # alpha != 0.95 and num_bootstrap=None  -> should produce ymin/ymax
    low = sor.Lowess(frac=0.3, gridsize=60, alpha=0.90)  # <-- no num_bootstrap
    out = _eval_stat(low, df, "x", "y")

    assert {"ymin", "ymax"} <= set(
        out.columns
    ), "Expected CI columns when alpha != 0.95"


# Helpful error when frac is too small for few unique x-values
def test_lowess_frac_too_small_message():
    # Only 3 distinct x values -> min_frac ≈ 2/3 in your implementation
    df = pd.DataFrame({"x": [1, 2, 3, 1, 2, 3], "y": [1, 2, 3, 1.1, 1.9, 3.05]})
    low = sor.Lowess(frac=0.2, gridsize=10)  # clearly too small

    with pytest.raises(ValueError) as err:
        _eval_stat(low, df, "x", "y")

    msg = str(err.value)
    assert (
        "distinct x" in msg and "frac=" in msg
    ), "Did not get helpful frac-too-small message"
