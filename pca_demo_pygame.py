import pygame
import sys
import numpy as np
from pygame.math import Vector3
import random

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("PCA: Principal Component Analysis")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Fonts
title_font = pygame.font.Font(None, 64)
text_font = pygame.font.Font(None, 32)
annotation_font = pygame.font.Font(None, 24)

# 3D point cloud
num_points = 100
points = []

def generate_points():
    global points
    points = []
    for _ in range(num_points):
        x = random.gauss(0, 1)
        y = random.gauss(0, 1)
        z = x + y + random.gauss(0, 0.1)
        points.append(Vector3(x, y, z))

generate_points()

# Camera settings
camera_distance = 5
camera_angle = [0, 0]

# PCA variables
pca_vectors = [Vector3(1, 0, 0), Vector3(0, 1, 0), Vector3(0, 0, 1)]
show_pca = False

def rotate_point(point, angle_x, angle_y):
    # Rotate around Y-axis
    cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
    x, z = point.x * cos_y - point.z * sin_y, point.x * sin_y + point.z * cos_y
    
    # Rotate around X-axis
    cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
    y, z = point.y * cos_x - z * sin_x, point.y * sin_x + z * cos_x
    
    return Vector3(x, y, z)

def project_point(point):
    x = point.x * (camera_distance / (camera_distance + point.z))
    y = point.y * (camera_distance / (camera_distance + point.z))
    return Vector3(x, y, 0)

def draw_text(text, font, color, x, y):
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y))

def perform_pca():
    global pca_vectors
    points_array = np.array([(p.x, p.y, p.z) for p in points])
    cov_matrix = np.cov(points_array.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    pca_vectors = [Vector3(*eigenvectors[:, i]) for i in range(3)]

# Main game loop
clock = pygame.time.Clock()
rotating = False
mouse_x, mouse_y = 0, 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                rotating = True
                mouse_x, mouse_y = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                rotating = False
        elif event.type == pygame.MOUSEMOTION:
            if rotating:
                dx, dy = event.pos[0] - mouse_x, event.pos[1] - mouse_y
                camera_angle[0] += dy * 0.01
                camera_angle[1] += dx * 0.01
                mouse_x, mouse_y = event.pos
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                show_pca = not show_pca
                if show_pca:
                    perform_pca()
            elif event.key == pygame.K_r:
                generate_points()
                if show_pca:
                    perform_pca()

    screen.fill(WHITE)

    # Draw title and developer info
    draw_text("PCA: Principal Component Analysis", title_font, BLACK, 20, 20)
    draw_text("Developed by: Venugopal Adep", text_font, BLACK, 20, 80)

    # Draw instructions
    draw_text("Left-click and drag to rotate", text_font, BLACK, 20, HEIGHT - 90)
    draw_text("Press SPACE to toggle PCA vectors", text_font, BLACK, 20, HEIGHT - 60)
    draw_text("Press R to randomize points", text_font, BLACK, 20, HEIGHT - 30)

    # Draw 3D points
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    for point in points:
        rotated = rotate_point(point, camera_angle[0], camera_angle[1])
        projected = project_point(rotated)
        pygame.draw.circle(screen, BLUE, (int(projected.x * 100 + center_x), int(projected.y * 100 + center_y)), 3)

    # Draw PCA vectors
    if show_pca:
        colors = [RED, GREEN, YELLOW]
        labels = ["PCA1", "PCA2", "PCA3"]
        for i, vec in enumerate(pca_vectors):
            rotated = rotate_point(vec, camera_angle[0], camera_angle[1])
            projected = project_point(rotated)
            end_x = int(projected.x * 200 + center_x)
            end_y = int(projected.y * 200 + center_y)
            pygame.draw.line(screen, colors[i], (center_x, center_y), (end_x, end_y), 3)
            
            # Draw PCA annotations
            annotation_pos = Vector3(projected.x * 220 + center_x, projected.y * 220 + center_y, 0)
            draw_text(labels[i], annotation_font, colors[i], int(annotation_pos.x), int(annotation_pos.y))

    pygame.display.flip()
    clock.tick(60)