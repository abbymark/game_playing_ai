import pygame
import sys
from game_playing_ai.environments.environment import Environment
from game_playing_ai.ai.agents.preprogrammed_agent.agent import PreprogrammedAgent
from game_playing_ai.ai.agents.playable_agent.agent import PlayableAgent
from game_playing_ai.environments.game_items.food import Food
import random


class GameStarter:
    WIDTH = 800
    HEIGHT = 600

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Game Playing AI")
        self.clock = pygame.time.Clock()
        self.running = True

        # Environment
        self.environment = Environment(self.screen, 30, 40)

        # Agents
        self.playable_agent_pos = (random.randint(0, self.environment.cols - 1), random.randint(0, self.environment.rows - 1))
        self.playable_agent = PlayableAgent(self.screen, 30, 40, self.playable_agent_pos[0], self.playable_agent_pos[1])
        
        
        preprogrammed_agent_pos = (random.randint(0, self.environment.cols - 1), random.randint(0, self.environment.rows - 1))
        self.preprogrammed_agent = PreprogrammedAgent(self.screen, 30, 40, preprogrammed_agent_pos[0], preprogrammed_agent_pos[1])




        # Food
        self.food = []
        self.generate_food(10)

    def generate_food(self, n=1):
        for i in range(n):
            y = random.randint(0, self.environment.rows - 1)
            x = random.randint(0, self.environment.cols - 1)
            self.food.append(Food(self.screen, self.environment.rows, self.environment.cols, x, y))

    def run(self):
        while self.running:
            self.clock.tick(5)
            events = pygame.event.get()
            self.events(events)
            self.update(events)
            self.draw()

    def events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit()

    def update(self, events):
        self.playable_agent.update(events)
        self.preprogrammed_agent.update(self.food)
        self.check_collisions()

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.environment.draw()
        self.playable_agent.draw()
        self.preprogrammed_agent.draw()
        for food in self.food:
            food.draw()
        pygame.display.update()

    def check_collisions(self):
        for food in self.food:
            if pygame.sprite.collide_rect(self.playable_agent, food):
                self.food.remove(food)
                self.generate_food(1)
                break
            if pygame.sprite.collide_rect(self.preprogrammed_agent, food):
                self.food.remove(food)
                self.generate_food(1)
                break