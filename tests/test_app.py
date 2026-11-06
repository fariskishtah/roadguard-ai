import app


def test_application_initializes_without_loading_checkpoint():
    assert app.demo is not None


def test_licensed_example_images_are_packaged():
    assert len(app.EXAMPLE_IMAGES) == 6
    assert all(path.is_file() for path in app.EXAMPLE_IMAGES)
