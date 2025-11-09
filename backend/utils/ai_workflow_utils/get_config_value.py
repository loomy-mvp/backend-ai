# Helper to get config value with fallback to default
def get_config_value(config_set: dict, key: str):
    if config_set.get(key) is not None:
        return config_set[key]
    return config_set.get(f"default_{key}")