from streamlit.testing.v1 import AppTest


def test_review_renders_and_switches_cost_scenario():
    app = AppTest.from_string(
        "from src.dashboard.strategy_review import render_strategy_review\n"
        "render_strategy_review()"
    ).run(timeout=20)
    assert not app.exception
    assert len(app.dataframe) == 2
    assert "không phải tín hiệu mua mới" in app.info[0].value
    baseline = app.dataframe[1].value
    assert baseline.iloc[0]["Lợi nhuận"] == "72.32%"
    app.selectbox[0].select("0,30%").run()
    assert not app.exception
    stressed = app.dataframe[1].value
    assert stressed.iloc[0]["Lợi nhuận"] == "52.49%"
    assert stressed.iloc[-1]["Lợi nhuận"] == "11.88%"
