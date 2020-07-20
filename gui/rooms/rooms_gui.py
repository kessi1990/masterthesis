import pygame

pygame.init()


class GUI:
    def __init__(self):
        self.scale = 50

    def display(self, env):
        size = env.size_x * self.scale, env.size_y * self.scale
        screen = pygame.display.set_mode(size)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
        screen.fill((0, 0, 0))

        # display agent
        a_col = pygame.image.load("../gui/rooms/assets/agent.png")
        a_col = pygame.transformation.scale(a_col, (self.scale, self.scale))
        a_rect = a_col.get_rect()
        a_rect = a_rect.move((env.agent.x * self.scale - 1, env.agent.y * self.scale - 1))
        screen.blit(a_col, a_rect)
        pygame.display.flip()

        # display goal
        g_col = pygame.image.load("../gui/rooms/assets/goal.png")
        g_col = pygame.transformation.scale(g_col, (self.scale, self.scale))
        g_rect = g_col.get_rect()
        g_rect = g_rect.move((env.goal.x * self.scale - 1, env.goal.y * self.scale - 1))
        screen.blit(g_col, g_rect)
        pygame.display.flip()

        for obs in env.obs_coords:
            """
            if obs.type == 'Obstacle':
                e_col = pygame.image.load("gui/rooms/assets/obstacle.png")
            elif obs.type == 'Runner':
                e_col = pygame.image.load("../gui/factory/assets/agent.png")
            else:
                e_col = pygame.image.load("../gui/factory/assets/goal.png")
            """
            e_col = pygame.image.load("../gui/rooms/assets/obstacle.png")
            e_col = pygame.transformation.scale(e_col, (self.scale, self.scale))
            e_rect = e_col.get_rect()
            e_rect = e_rect.move((obs[0] * self.scale - 1, obs[1] * self.scale - 1))
            screen.blit(e_col, e_rect)
            pygame.display.flip()
