import pygame
import random
import math
import numpy as np
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Kmeans vs Kmedoids Demo")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

# Fonts
title_font = pygame.font.Font(None, 48)
text_font = pygame.font.Font(None, 24)

# Variables
points = []
k = 3
kmeans_centroids = []
kmedoids_medoids = []
kmeans_clusters = []
kmedoids_clusters = []
algorithm = "kmeans"
show_explanation = False

def generate_random_points(num_points):
    return [(random.randint(50, WIDTH//2 - 50), random.randint(100, HEIGHT - 50)) for _ in range(num_points)]

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def run_kmeans():
    global kmeans_centroids, kmeans_clusters
    if len(points) < k:
        kmeans_centroids = []
        kmeans_clusters = []
        return
    X = np.array(points)
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(X)
    kmeans_centroids = kmeans.cluster_centers_.tolist()
    kmeans_clusters = kmeans.labels_.tolist()

def run_kmedoids():
    global kmedoids_medoids, kmedoids_clusters
    if len(points) < k:
        kmedoids_medoids = []
        kmedoids_clusters = []
        return
    X = np.array(points)
    kmedoids = KMedoids(n_clusters=k, random_state=0)
    kmedoids.fit(X)
    kmedoids_medoids = [tuple(points[i]) for i in kmedoids.medoid_indices_]
    kmedoids_clusters = kmedoids.labels_.tolist()

def draw_text():
    title = title_font.render("Kmeans vs Kmedoids Demo", True, BLACK)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 20))
    
    author = text_font.render("Developed by: Venugopal Adep", True, BLACK)
    screen.blit(author, (WIDTH // 2 - author.get_width() // 2, 60))
    
    instructions = [
        "Left click: Add points",
        "Right click: Remove points",
        "K: Increase K",
        "J: Decrease K",
        "Space: Run clustering",
        "Tab: Switch algorithm",
        "E: Toggle explanation",
        "R: Reset"
    ]
    
    for i, instruction in enumerate(instructions):
        text = text_font.render(instruction, True, BLACK)
        screen.blit(text, (10, 100 + i * 30))
    
    algorithm_text = text_font.render(f"Current Algorithm: {algorithm.capitalize()}", True, BLACK)
    screen.blit(algorithm_text, (WIDTH - 250, 100))
    
    k_text = text_font.render(f"K: {k}", True, BLACK)
    screen.blit(k_text, (WIDTH - 250, 130))

def draw_explanation():
    if not show_explanation:
        return
    
    explanations = {
        "kmeans": [
            "K-means:",
            "- Centroids are the mean of points in the cluster",
            "- Sensitive to outliers",
            "- Works well with spherical clusters",
            "- Faster for large datasets"
        ],
        "kmedoids": [
            "K-medoids:",
            "- Medoids are actual data points",
            "- More robust to outliers",
            "- Works better with non-spherical clusters",
            "- Slower but more interpretable"
        ]
    }
    
    for i, line in enumerate(explanations[algorithm]):
        text = text_font.render(line, True, BLACK)
        screen.blit(text, (WIDTH // 2 + 50, 100 + i * 30))

def draw_points():
    for point in points:
        pygame.draw.circle(screen, BLACK, point, 5)

def draw_clusters():
    global points, kmeans_clusters, kmedoids_clusters, kmeans_centroids, kmedoids_medoids

    if algorithm == "kmeans":
        centroids = kmeans_centroids
        clusters = kmeans_clusters
    else:
        centroids = kmedoids_medoids
        clusters = kmedoids_clusters
    
    colors = [RED, BLUE, GREEN, YELLOW, PURPLE]
    
    if clusters and centroids and len(clusters) == len(points):
        for i, point in enumerate(points):
            cluster_index = clusters[i] if 0 <= clusters[i] < len(colors) else 0
            color = colors[cluster_index]
            pygame.draw.circle(screen, color, point, 5)
        
        for i, centroid in enumerate(centroids):
            color = colors[i % len(colors)]
            pygame.draw.circle(screen, color, (int(centroid[0]), int(centroid[1])), 10, 3)
    else:
        # If no valid clusters are formed yet, just draw the points in black
        for point in points:
            pygame.draw.circle(screen, BLACK, point, 5)

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                points.append(event.pos)
            elif event.button == 3:  # Right click
                for point in points:
                    if distance(point, event.pos) < 10:
                        points.remove(point)
                        break
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if algorithm == "kmeans":
                    run_kmeans()
                else:
                    run_kmedoids()
            elif event.key == pygame.K_TAB:
                algorithm = "kmedoids" if algorithm == "kmeans" else "kmeans"
            elif event.key == pygame.K_k:
                k = min(k + 1, 5)
            elif event.key == pygame.K_j:
                k = max(k - 1, 2)
            elif event.key == pygame.K_r:
                points = []
                kmeans_centroids = []
                kmedoids_medoids = []
                kmeans_clusters = []
                kmedoids_clusters = []
            elif event.key == pygame.K_e:
                show_explanation = not show_explanation
    
    screen.fill(WHITE)
    draw_text()
    draw_explanation()
    
    if (kmeans_clusters and kmeans_centroids) or (kmedoids_clusters and kmedoids_medoids):
        draw_clusters()
    else:
        draw_points()
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()