def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow tests (large generation)")
