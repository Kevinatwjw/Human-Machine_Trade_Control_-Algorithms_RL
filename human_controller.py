# human_controller.py
import pygame
import config

class HumanController:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((100, 100)) 
        pygame.display.set_caption("Human Input")

    def get_action(self):
        pygame.event.pump()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "QUIT"
        keys = pygame.key.get_pressed()
        for key, action in config.KEY_MAP.items():
            if keys[key]:
                return action
        return None  

    def close(self):
        pygame.quit()