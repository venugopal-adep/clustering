import pygame
import math
import random
import numpy as np

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Gini vs Entropy Demo")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (200, 200, 200)

# Fonts
title_font = pygame.font.Font(None, 48)
text_font = pygame.font.Font(None, 24)

# Classes
class DataPoint:
    def __init__(self, x, y, class_label):
        self.x = x
        self.y = y
        self.class_label = class_label

class Button:
    def __init__(self, x, y, width, height, text, color, text_color, action):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.action = action

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        text_surf = text_font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.action()

# Helper functions
def calculate_gini(class_counts):
    total = sum(class_counts)
    if total == 0:
        return 0
    return 1 - sum((count / total) ** 2 for count in class_counts)

def calculate_entropy(class_counts):
    total = sum(class_counts)
    if total == 0:
        return 0
    return -sum((count / total) * math.log2(count / total) if count > 0 else 0 for count in class_counts)

def draw_data_points(screen, data_points):
    for point in data_points:
        color = RED if point.class_label == 0 else BLUE
        pygame.draw.circle(screen, color, (int(point.x), int(point.y)), 5)

def draw_split_line(screen, split_x):
    pygame.draw.line(screen, GREEN, (split_x, 0), (split_x, HEIGHT), 2)

def create_random_data(num_points):
    return [DataPoint(random.randint(0, WIDTH), random.randint(0, HEIGHT), random.randint(0, 1)) for _ in range(num_points)]

# Create initial data
data_points = create_random_data(100)
split_x = WIDTH // 2

# Create buttons
def randomize_data():
    global data_points
    data_points = create_random_data(100)

randomize_button = Button(50, HEIGHT - 60, 200, 50, "Randomize Data", GRAY, BLACK, randomize_data)

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                x, y = event.pos
                if y < HEIGHT - 70:  # Avoid clicking in the button area
                    data_points.append(DataPoint(x, y, 0))  # Add red point
            elif event.button == 3:  # Right click
                x, y = event.pos
                if y < HEIGHT - 70:  # Avoid clicking in the button area
                    data_points.append(DataPoint(x, y, 1))  # Add blue point
        elif event.type == pygame.MOUSEMOTION:
            if event.buttons[0]:  # Left mouse button held down
                split_x = event.pos[0]
        
        randomize_button.handle_event(event)

    # Clear the screen
    screen.fill(WHITE)

    # Draw title and developer info
    title_text = title_font.render("Gini vs Entropy Demo", True, BLACK)
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 20))
    
    dev_text = text_font.render("Developed by: Venugopal Adep", True, BLACK)
    screen.blit(dev_text, (WIDTH // 2 - dev_text.get_width() // 2, 70))

    # Draw data points and split line
    draw_data_points(screen, data_points)
    draw_split_line(screen, split_x)

    # Calculate Gini and Entropy for left and right splits
    left_counts = [sum(1 for p in data_points if p.x < split_x and p.class_label == i) for i in range(2)]
    right_counts = [sum(1 for p in data_points if p.x >= split_x and p.class_label == i) for i in range(2)]

    left_gini = calculate_gini(left_counts)
    right_gini = calculate_gini(right_counts)
    left_entropy = calculate_entropy(left_counts)
    right_entropy = calculate_entropy(right_counts)

    total_points = len(data_points)
    left_ratio = sum(left_counts) / total_points
    right_ratio = sum(right_counts) / total_points

    weighted_gini = left_ratio * left_gini + right_ratio * right_gini
    weighted_entropy = left_ratio * left_entropy + right_ratio * right_entropy

    # Draw Gini and Entropy information
    info_texts = [
        f"Left Gini: {left_gini:.3f}",
        f"Right Gini: {right_gini:.3f}",
        f"Weighted Gini: {weighted_gini:.3f}",
        f"Left Entropy: {left_entropy:.3f}",
        f"Right Entropy: {right_entropy:.3f}",
        f"Weighted Entropy: {weighted_entropy:.3f}",
    ]

    for i, text in enumerate(info_texts):
        text_surface = text_font.render(text, True, BLACK)
        screen.blit(text_surface, (20, 120 + i * 30))

    # Draw instructions
    instructions = [
        "Left click to add red points",
        "Right click to add blue points",
        "Click and drag to move the split line",
        "Use the button to randomize data",
    ]

    for i, instruction in enumerate(instructions):
        inst_text = text_font.render(instruction, True, BLACK)
        screen.blit(inst_text, (WIDTH - 300, 120 + i * 30))

    # Draw button
    randomize_button.draw(screen)

    # Update the display
    pygame.display.flip()
    clock.tick(60)

# Quit Pygame
pygame.quit()