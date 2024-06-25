import pygame
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 1600, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Spectral Clustering Demo")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

# Font
font = pygame.font.Font(None, 36)

# Spectral Clustering parameters
X = []
n_clusters = 3
n_neighbors = 10
spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=n_neighbors)

def add_point(pos):
    X.append([pos[0], pos[1]])

def run_spectral_clustering():
    if len(X) > n_clusters:
        spectral.fit(X)

def draw_points():
    for point in X:
        pygame.draw.circle(screen, WHITE, (int(point[0]), int(point[1])), 5)

def draw_clusters():
    if len(X) > n_clusters:
        labels = spectral.labels_
        for i, point in enumerate(X):
            color = COLORS[labels[i] % len(COLORS)]
            pygame.draw.circle(screen, color, (int(point[0]), int(point[1])), 7)

def draw_graph():
    if len(X) > 1:
        adj_matrix = kneighbors_graph(X, n_neighbors=n_neighbors, mode='connectivity').toarray()
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                if adj_matrix[i, j] > 0:
                    pygame.draw.line(screen, WHITE, X[i], X[j], 1)

def draw_text(text, position, color=WHITE):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)

# Main game loop
running = True
clock = pygame.time.Clock()
clustering = False
show_graph = False

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
            elif event.key == pygame.K_g:
                show_graph = not show_graph
            elif event.key == pygame.K_UP:
                n_clusters = min(n_clusters + 1, 6)
                spectral.set_params(n_clusters=n_clusters)
            elif event.key == pygame.K_DOWN:
                n_clusters = max(n_clusters - 1, 2)
                spectral.set_params(n_clusters=n_clusters)
            elif event.key == pygame.K_RIGHT:
                n_neighbors += 1
                spectral.set_params(n_neighbors=n_neighbors)
            elif event.key == pygame.K_LEFT:
                n_neighbors = max(n_neighbors - 1, 2)
                spectral.set_params(n_neighbors=n_neighbors)

    screen.fill(BLACK)

    if show_graph:
        draw_graph()

    if clustering:
        run_spectral_clustering()
        draw_clusters()
    else:
        draw_points()

    # Draw labels
    draw_text("Left Click: Add Point", (10, 10))
    draw_text("Space: Toggle Clustering", (10, 50))
    draw_text("G: Toggle Graph", (10, 90))
    draw_text("Up/Down: Increase/Decrease n_clusters", (10, 130))
    draw_text("Left/Right: Decrease/Increase n_neighbors", (10, 170))
    draw_text(f"n_clusters: {n_clusters}", (10, 210))
    draw_text(f"n_neighbors: {n_neighbors}", (10, 250))
    draw_text("Clustering: " + ("On" if clustering else "Off"), (10, 290))
    draw_text("Graph: " + ("On" if show_graph else "Off"), (10, 330))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()