import pygame
import numpy as np
from sklearn.cluster import DBSCAN

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 1600, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("DBSCAN Clustering Demo")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
NOISE_COLOR = (128, 128, 128)  # Gray for noise points

# Font
font = pygame.font.Font(None, 36)

# DBSCAN parameters
X = []
eps = 50
min_samples = 5
dbscan = DBSCAN(eps=eps, min_samples=min_samples)

def add_point(pos):
    X.append([pos[0], pos[1]])

def run_dbscan():
    if len(X) > 0:
        dbscan.fit(X)

def draw_points():
    for point in X:
        pygame.draw.circle(screen, WHITE, (int(point[0]), int(point[1])), 5)

def draw_clusters():
    if len(X) > 0:
        labels = dbscan.labels_
        for i, point in enumerate(X):
            if labels[i] == -1:  # Noise point
                color = NOISE_COLOR
            else:
                color = COLORS[labels[i] % len(COLORS)]
            pygame.draw.circle(screen, color, (int(point[0]), int(point[1])), 7)

def draw_eps_circle(pos):
    pygame.draw.circle(screen, WHITE, pos, eps, 1)

def draw_text(text, position, color=WHITE):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)

# Main game loop
running = True
clock = pygame.time.Clock()
clustering = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                pos = pygame.mouse.get_pos()
                add_point(pos)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                clustering = not clustering
            elif event.key == pygame.K_UP:
                eps += 5
                dbscan.set_params(eps=eps)
            elif event.key == pygame.K_DOWN:
                eps = max(eps - 5, 5)
                dbscan.set_params(eps=eps)
            elif event.key == pygame.K_RIGHT:
                min_samples += 1
                dbscan.set_params(min_samples=min_samples)
            elif event.key == pygame.K_LEFT:
                min_samples = max(min_samples - 1, 2)
                dbscan.set_params(min_samples=min_samples)

    screen.fill(BLACK)

    if clustering:
        run_dbscan()
        draw_clusters()
    else:
        draw_points()

    # Draw eps circle around mouse cursor
    mouse_pos = pygame.mouse.get_pos()
    draw_eps_circle(mouse_pos)

    # Draw labels
    draw_text("Left Click: Add Point", (10, 10))
    draw_text("Space: Toggle Clustering", (10, 50))
    draw_text("Up/Down: Increase/Decrease eps", (10, 90))
    draw_text("Left/Right: Decrease/Increase min_samples", (10, 130))
    draw_text(f"eps: {eps}", (10, 170))
    draw_text(f"min_samples: {min_samples}", (10, 210))
    draw_text("Clustering: " + ("On" if clustering else "Off"), (10, 250))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()