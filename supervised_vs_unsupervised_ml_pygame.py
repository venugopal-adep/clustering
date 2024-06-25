import pygame
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Supervised vs Unsupervised ML Demo")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
COLORS = [RED, GREEN, BLUE, YELLOW]

# Fonts
title_font = pygame.font.Font(None, 48)
text_font = pygame.font.Font(None, 24)

# Data points
num_points = 100
data_points = []

# ML models
kmeans = KMeans(n_clusters=3)
logistic_reg = LogisticRegression()

# Buttons
supervised_button = pygame.Rect(WIDTH // 4 - 100, HEIGHT - 100, 200, 50)
unsupervised_button = pygame.Rect(3 * WIDTH // 4 - 100, HEIGHT - 100, 200, 50)
reset_button = pygame.Rect(WIDTH // 2 - 100, HEIGHT - 100, 200, 50)

# States
current_mode = None
clusters = []
decision_boundary = None

def generate_data():
    global data_points
    data_points = []
    for _ in range(num_points):
        x = random.randint(50, WIDTH - 50)
        y = random.randint(50, HEIGHT - 150)
        label = random.randint(0, 1)
        data_points.append((x, y, label))

def draw_data_points():
    for point in data_points:
        x, y, label = point
        color = RED if label == 0 else BLUE
        pygame.draw.circle(screen, color, (x, y), 5)

def perform_clustering():
    global clusters
    X = np.array([(p[0], p[1]) for p in data_points])
    kmeans.fit(X)
    clusters = kmeans.labels_

def draw_clusters():
    for i, point in enumerate(data_points):
        x, y, _ = point
        color = COLORS[clusters[i] % len(COLORS)]
        pygame.draw.circle(screen, color, (x, y), 5)

def perform_classification():
    global decision_boundary
    X = np.array([(p[0], p[1]) for p in data_points])
    y = np.array([p[2] for p in data_points])
    logistic_reg.fit(X, y)
    
    xx, yy = np.meshgrid(np.arange(0, WIDTH, 10), np.arange(0, HEIGHT, 10))
    Z = logistic_reg.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    decision_boundary = (xx, yy, Z)

def draw_decision_boundary():
    if decision_boundary:
        xx, yy, Z = decision_boundary
        for i in range(len(xx)):
            for j in range(len(yy)):
                if 0 <= xx[i][j] < WIDTH and 0 <= yy[i][j] < HEIGHT:
                    color = RED if Z[i][j] == 0 else BLUE
                    pygame.draw.circle(screen, color, (int(xx[i][j]), int(yy[i][j])), 2)

generate_data()

running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if supervised_button.collidepoint(event.pos):
                current_mode = "supervised"
                perform_classification()
            elif unsupervised_button.collidepoint(event.pos):
                current_mode = "unsupervised"
                perform_clustering()
            elif reset_button.collidepoint(event.pos):
                current_mode = None
                generate_data()
                clusters = []
                decision_boundary = None

    screen.fill(WHITE)

    # Draw title and developer info
    title_text = title_font.render("Supervised vs Unsupervised ML Demo", True, BLACK)
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 20))

    developer_text = text_font.render("Developed by: Venugopal Adep", True, BLACK)
    screen.blit(developer_text, (WIDTH // 2 - developer_text.get_width() // 2, 70))

    # Draw data points or results
    if current_mode == "supervised":
        draw_decision_boundary()
        draw_data_points()
    elif current_mode == "unsupervised":
        draw_clusters()
    else:
        draw_data_points()

    # Draw buttons
    pygame.draw.rect(screen, GREEN, supervised_button)
    supervised_text = text_font.render("Supervised Learning", True, BLACK)
    screen.blit(supervised_text, (supervised_button.x + 10, supervised_button.y + 15))

    pygame.draw.rect(screen, BLUE, unsupervised_button)
    unsupervised_text = text_font.render("Unsupervised Learning", True, BLACK)
    screen.blit(unsupervised_text, (unsupervised_button.x + 10, unsupervised_button.y + 15))

    pygame.draw.rect(screen, YELLOW, reset_button)
    reset_text = text_font.render("Reset Data", True, BLACK)
    screen.blit(reset_text, (reset_button.x + 50, reset_button.y + 15))

    # Draw explanations
    if current_mode == "supervised":
        explanation = "Supervised Learning: Classifying data points based on labeled examples."
    elif current_mode == "unsupervised":
        explanation = "Unsupervised Learning: Clustering data points without prior labels."
    else:
        explanation = "Click a button to see Supervised or Unsupervised Learning in action!"
    
    explanation_text = text_font.render(explanation, True, BLACK)
    screen.blit(explanation_text, (WIDTH // 2 - explanation_text.get_width() // 2, HEIGHT - 150))

    # Update display
    pygame.display.flip()
    clock.tick(30)

pygame.quit()