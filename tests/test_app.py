import app


def test_application_initializes_without_loading_checkpoint():
    assert app.demo is not None
