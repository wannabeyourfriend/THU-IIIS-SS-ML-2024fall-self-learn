import pygame
import math
import sys

# Initialize Pygame
pygame.init()

# Set up display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Bouncing Ball in Spinning Hexagon")
clock = pygame.time.Clock()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Hexagon parameters
cx, cy = width // 2, height // 2
hex_radius = 200
angle = 0  # radians
omega = 2  # radians per second

# Ball parameters
ball_radius = 10
ball_color = RED
ball_x, ball_y = cx, cy
ball_vx, ball_vy = 5, 0  # initial velocity
gravity = 9.8 * 50  # pixels/s² (scaled for visibility)
friction = 0.5      # per second
restitution = 0.8   # bounce factor

# Main loop
running = True
last_time = pygame.time.get_ticks()

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Calculate delta time in seconds
    current_time = pygame.time.get_ticks()
    dt = (current_time - last_time) / 1000.0
    last_time = current_time

    # Update hexagon angle
    angle += omega * dt
    angle %= 2 * math.pi

    # Apply physics to ball
    ball_vy += gravity * dt  # gravity
    speed_loss = 1 - friction * dt
    ball_vx *= speed_loss    # air friction
    ball_vy *= speed_loss

    # Calculate new position
    new_x = ball_x + ball_vx * dt
    new_y = ball_y + ball_vy * dt

    # Calculate hexagon vertices
    vertices = []
    for i in range(6):
        theta = angle + math.radians(60 * i)
        x = cx + hex_radius * math.cos(theta)
        y = cy + hex_radius * math.sin(theta)
        vertices.append((x, y))

    # Collision detection
    collisions = []
    for i in range(6):
        a = vertices[i]
        b = vertices[(i+1) % 6]
        ax, ay = a
        bx, by = b

        # Vector AB and AP
        abx, aby = bx - ax, by - ay
        apx, apy = new_x - ax, new_y - ay

        # Project AP onto AB
        denominator = abx**2 + aby**2
        if denominator == 0:
            continue
        t = (apx * abx + apy * aby) / denominator
        t = max(0.0, min(1.0, t))

        # Closest point and distance
        closest_x = ax + t * abx
        closest_y = ay + t * aby
        dx = new_x - closest_x
        dy = new_y - closest_y
        distance = math.hypot(dx, dy)

        if distance < ball_radius:
            penetration = ball_radius - distance
            collisions.append((penetration, i, (closest_x, closest_y)))

    # Handle collisions
    if collisions:
        max_collision = max(collisions, key=lambda x: x[0])
        penetration, wall_idx, closest = max_collision
        a, b = vertices[wall_idx], vertices[(wall_idx+1) % 6]

        # Calculate inward normal
        abx, aby = b[0] - a[0], b[1] - a[1]
        edge_length = math.hypot(abx, aby)
        if edge_length == 0:
            continue
        normal_x = aby / edge_length
        normal_y = -abx / edge_length

        # Calculate wall velocity at collision point
        dx_closest = closest[0] - cx
        dy_closest = closest[1] - cy
        v_wall_x = -omega * dy_closest
        v_wall_y = omega * dx_closest

        # Calculate relative velocity
        rel_vx = ball_vx - v_wall_x
        rel_vy = ball_vy - v_wall_y

        # Calculate reflection
        dot_product = rel_vx * normal_x + rel_vy * normal_y
        if dot_product < 0:
            # Apply bounce with restitution
            rel_vx -= 2 * dot_product * normal_x
            rel_vy -= 2 * dot_product * normal_y
            rel_vx *= restitution
            rel_vy *= restitution

            # Update ball velocity
            ball_vx = rel_vx + v_wall_x
            ball_vy = rel_vy + v_wall_y

            # Adjust position to prevent penetration
            new_x = closest[0] + normal_x * ball_radius
            new_y = closest[1] + normal_y * ball_radius

    # Update ball position
    ball_x, ball_y = new_x, new_y

    # Draw everything
    screen.fill(BLACK)
    pygame.draw.polygon(screen, WHITE, vertices, 2)
    pygame.draw.circle(screen, ball_color, (int(ball_x), int(ball_y)), ball_radius)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()