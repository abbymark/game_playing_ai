from game_playing_ai.games.food_game.food_game import TileType

import pygame

class Environment:
    def __init__(self, screen, rows, cols):
        self.screen = screen
        self.rows = rows
        self.cols = cols
        self.cell_width = self.screen.get_width() / self.cols
        self.cell_height = self.screen.get_height() / self.rows

        self.colors = {
            "black": (0, 0, 0),
            "blue": (0, 0, 255),
            "green": (0, 255, 0),
            "red": (255, 0, 0),
            "white": (255, 255, 255),
            "yellow": (255, 255, 0)
        }

        self.empty = pygame.Surface((self.cell_width, self.cell_height))
        self.empty.fill(self.colors["black"])
        self.empty_rect = self.empty.get_rect()

        self.obstacle = pygame.Surface((self.cell_width, self.cell_height))
        self.obstacle.fill(self.colors["white"])
        self.obstacle_rect = self.obstacle.get_rect()

        self.food = pygame.Surface((self.cell_width, self.cell_height))
        self.food.fill(self.colors["green"])
        self.food_rect = self.food.get_rect()

        self.playable_agent = pygame.Surface((self.cell_width, self.cell_height))
        self.playable_agent.fill(self.colors["red"])
        self.playable_agent_rect = self.playable_agent.get_rect()

        self.preprogrammed_agent = pygame.Surface((self.cell_width, self.cell_height))
        self.preprogrammed_agent.fill(self.colors["blue"])
        self.preprogrammed_agent_rect = self.preprogrammed_agent.get_rect()

        self.drl_agent = pygame.Surface((self.cell_width, self.cell_height))
        self.drl_agent.fill(self.colors["yellow"])
        self.drl_agent_rect = self.drl_agent.get_rect()


    def draw(self, map):
        for i in range(self.rows):
            for j in range(self.cols):
                if map[i][j] == TileType.EMPTY:
                    self.rect = self.empty_rect
                    self.rect.x = j * self.cell_width
                    self.rect.y = i * self.cell_height
                    self.screen.blit(self.empty, self.rect)
                elif map[i][j] == TileType.OBSTACLE:
                    self.rect = self.obstacle_rect
                    self.rect.x = j * self.cell_width
                    self.rect.y = i * self.cell_height
                    self.screen.blit(self.obstacle, self.rect)
                elif map[i][j] == TileType.FOOD:
                    self.rect = self.food_rect
                    self.rect.x = j * self.cell_width
                    self.rect.y = i * self.cell_height
                    self.screen.blit(self.food, self.rect)
                elif map[i][j] == TileType.PLAYABLE_AGENT:
                    self.rect = self.playable_agent_rect
                    self.rect.x = j * self.cell_width
                    self.rect.y = i * self.cell_height
                    self.screen.blit(self.playable_agent, self.rect)
                elif map[i][j] == TileType.PREPROGRAMMED_AGENT:
                    self.rect = self.preprogrammed_agent_rect
                    self.rect.x = j * self.cell_width
                    self.rect.y = i * self.cell_height
                    self.screen.blit(self.preprogrammed_agent, self.rect)
                elif map[i][j] == TileType.DRL_AGENT:
                    self.rect = self.drl_agent_rect
                    self.rect.x = j * self.cell_width
                    self.rect.y = i * self.cell_height
                    self.screen.blit(self.drl_agent, self.rect)
                