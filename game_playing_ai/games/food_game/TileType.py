class TileType:
    EMPTY = 0
    OBSTACLE = 1
    FOOD = 2
    PLAYABLE_AGENT = 3
    PREPROGRAMMED_AGENT = 4
    DRL_AGENT = 5
    AGENT_LOCATION = 6

    @classmethod
    def __len__(cls):
        return len([attr for attr in dir(cls) if not attr.startswith('_') and not callable(getattr(cls, attr))])