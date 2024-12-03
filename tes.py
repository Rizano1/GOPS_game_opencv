import pygame
import random
import cv2
import yaml
import os
import numpy as np
import tensorflow as tf
from imageProcessing import preprocessFrame

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
CARD_WIDTH, CARD_HEIGHT = 100, 140
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Fonts
FONT_LARGE = pygame.font.Font(None, 50)
FONT_SMALL = pygame.font.Font(None, 30)

mapping = {
  "2_hati": "2_of_hearts",
  "2_keriting": "2_of_clubs",
  "2_sekop": "2_of_spades",
  "2_wajik": "2_of_diamonds",
  "3_hati": "3_of_hearts",
  "3_keriting": "3_of_clubs",
  "3_sekop": "3_of_spades",
  "3_wajik": "3_of_diamonds",
  "4_hati": "4_of_hearts",
  "4_keriting": "4_of_clubs",
  "4_sekop": "4_of_spades",
  "4_wajik": "4_of_diamonds",
  "5_hati": "5_of_hearts",
  "5_keriting": "5_of_clubs",
  "5_sekop": "5_of_spades",
  "5_wajik": "5_of_diamonds",
  "6_hati": "6_of_hearts",
  "6_keriting": "6_of_clubs",
  "6_sekop": "6_of_spades",
  "6_wajik": "6_of_diamonds",
  "7_hati": "7_of_hearts",
  "7_keriting": "7_of_clubs",
  "7_sekop": "7_of_spades",
  "7_wajik": "7_of_diamonds",
  "8_hati": "8_of_hearts",
  "8_keriting": "8_of_clubs",
  "8_sekop": "8_of_spades",
  "8_wajik": "8_of_diamonds",
  "9_hati": "9_of_hearts",
  "9_keriting": "9_of_clubs",
  "9_sekop": "9_of_spades",
  "9_wajik": "9_of_diamonds",
  "10_hati": "10_of_hearts",
  "10_keriting": "10_of_clubs",
  "10_sekop": "10_of_spades",
  "10_wajik": "10_of_diamonds",
  # Tambahkan semua kartu sesuai aturan
  "a_hati": "ace_of_hearts",
  "a_keriting": "ace_of_clubs",
  "a_sekop": "ace_of_spades",
  "a_wajik": "ace_of_diamonds",
  "k_hati": "king_of_hearts",
  "k_keriting": "king_of_clubs",
  "k_sekop": "king_of_spades",
  "k_wajik": "king_of_diamonds",
  "q_hati": "queen_of_hearts",
  "q_keriting": "queen_of_clubs",
  "q_sekop": "queen_of_spades",
  "q_wajik": "queen_of_diamonds",
  "j_hati": "jack_of_hearts",
  "j_keriting": "jack_of_clubs",
  "j_sekop": "jack_of_spades",
  "j_wajik": "jack_of_diamonds"
}

urutan = {"2", "3", "4", "5", "6", "7", "8", "9", "10", "j", "q", "k", "a"}

def preprocess_and_detect(frame, preprocess_fn, model, hsv_values, class_mapping, min_contour_area):
    """
    Preprocess frame, detect cards, and classify them for P1, P2, and deck regions.

    Args:
        frame (numpy.ndarray): Input video frame.
        preprocess_fn (function): Function to preprocess the frame and extract warped cards.
        model (tf.keras.Model): Pre-trained CNN model for card classification.
        hsv_values (tuple): HSV values for segmentation (lowH, lowS, lowV, highH, highS, highV).
        class_mapping (dict): Mapping of class indices to class names.
        min_contour_area (int): Minimum contour area to consider for detection.

    Returns:
        tuple: (p1_card, p2_card, deck_card) - detected card names in respective regions.
    """
    lowH, lowS, lowV, highH, highS, highV = hsv_values
    width = frame.shape[1]
    height = frame.shape[0]

    # Preprocess frame to get warped cards and bounding boxes
    processed_frame, mask, warpeds, boundings = preprocess_fn(
        frame, lowH, lowS, lowV, highH, highS, highV, min_contour_area
    )

    p1_card, p2_card, deck_card = None, None, None

    if len(warpeds) > 0:
        for warped, bounding in zip(warpeds, boundings):
            x, y, _, _ = bounding
            warped = warped.astype(np.float32) / 255.0

            # Predict card class
            prediction = model.predict(warped[np.newaxis, ...], verbose=0)
            predicted_class = np.argmax(prediction)
            class_name = class_mapping.get(predicted_class, "Unknown")

            # Assign to regions based on coordinates
            if x >= int(width / 3) and y <= int(height / 2):
                p2_card = class_name  # Top right (P2 region)
            elif x >= int(width / 3) and y > int(height / 2):
                p1_card = class_name  # Bottom right (P1 region)
            elif x < int(width / 3):
                deck_card = class_name  # Left (Deck region)

    return p1_card, p2_card, deck_card


# Load images
def load_images(card_dir):
    card_images = {}
    for file in os.listdir(card_dir):
        if file.endswith(".png"):
            card_name = file.split('.')[0]
            image = pygame.image.load(os.path.join(card_dir, file))
            image = pygame.transform.scale(image, (CARD_WIDTH, CARD_HEIGHT))
            card_images[card_name] = image
    return card_images

# Display text
def draw_text(screen, text, font, color, x, y, center=False):
    surface = font.render(text, True, color)
    rect = surface.get_rect(center=(x, y)) if center else (x, y)
    screen.blit(surface, rect)

# Load HSV values
def load_hsv_values(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            hsv_values = yaml.safe_load(f)
        return (hsv_values['LowH'], hsv_values['LowS'], hsv_values['LowV'],
                hsv_values['HighH'], hsv_values['HighS'], hsv_values['HighV'])
    return 0, 100, 100, 10, 255, 255

# Hard bot logic
def hard_bot_logic(player_cards, opponent_played, current_deck_card):
    remaining_cards = [card for card in player_cards if card not in opponent_played]
    deck_value = int(current_deck_card.split('_')[0])
    best_card = min(remaining_cards, key=lambda c: abs(deck_value - int(c.split('_')[0])))
    return best_card

# Main menu
def main_menu(screen):
    screen.fill(WHITE)
    draw_text(screen, "GOPS Game", FONT_LARGE, BLACK, SCREEN_WIDTH // 2, 100, center=True)
    draw_text(screen, "1. Play vs Player", FONT_SMALL, BLACK, SCREEN_WIDTH // 2, 200, center=True)
    draw_text(screen, "2. Play vs Bot (Easy)", FONT_SMALL, BLACK, SCREEN_WIDTH // 2, 300, center=True)
    draw_text(screen, "3. Play vs Bot (Hard)", FONT_SMALL, BLACK, SCREEN_WIDTH // 2, 400, center=True)
    draw_text(screen, "Press 1, 2, or 3 to choose", FONT_SMALL, BLACK, SCREEN_WIDTH // 2, 500, center=True)
    pygame.display.flip()

# Game loop
def game_loop(screen, card_images, mode, hsv_values, model, class_mapping):
    deck_cards = [f"{i}_wajik" for i in urutan]
    player_1_cards = [f"{i}_sekop" for i in urutan]
    player_2_cards = [f"{i}_keriting" for i in urutan]

    random.shuffle(deck_cards)

    player_1_score = 0
    player_2_score = 0
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

    for deck_card in deck_cards:
        screen.fill(WHITE)
        draw_text(screen, f"Deck Card: {deck_card}", FONT_SMALL, BLACK, SCREEN_WIDTH // 2, 50, center=True)
        screen.blit(card_images[mapping[deck_card]], (SCREEN_WIDTH // 2 - CARD_WIDTH // 2, 100))
        pygame.display.flip()

        # Get P1's card from camera
        ret, frame = cap.read()
        p1_card_detect, p2_card_detect, _ = preprocess_and_detect(frame, preprocessFrame, model, hsv_values, class_mapping, 9000)
        p1_card = p1_card_detect
        if p1_card not in player_1_cards:
            continue

        player_1_cards.remove(p1_card)

        # Get P2's card
        if mode == "PvBot-Easy":
            p2_card = random.choice(player_2_cards)
        elif mode == "PvBot-Hard":
            p2_card = hard_bot_logic(player_2_cards, [], deck_card)
        else:
            p2_card = p2_card_detect  # PvP, no bot logic

        if p2_card:
            player_2_cards.remove(p2_card)

        # Display the cards
        screen.fill(WHITE)
        draw_text(screen, f"P1 Played: {p1_card}", FONT_SMALL, BLACK, SCREEN_WIDTH // 4, 300)
        draw_text(screen, f"P2 Played: {p2_card}", FONT_SMALL, BLACK, 3 * SCREEN_WIDTH // 4, 300)
        screen.blit(card_images[p1_card], (SCREEN_WIDTH // 4 - CARD_WIDTH // 2, 350))
        screen.blit(card_images[p2_card], (3 * SCREEN_WIDTH // 4 - CARD_WIDTH // 2, 350))
        pygame.display.flip()

        pygame.time.wait(2000)

        # Compare cards and award points
        p1_value = int(p1_card.split('_')[0])
        p2_value = int(p2_card.split('_')[0]) if p2_card else 0

        if p1_value > p2_value:
            player_1_score += int(deck_card.split('_')[0])
        elif p2_value > p1_value:
            player_2_score += int(deck_card.split('_')[0])

    cap.release()

    # End of game screen
    screen.fill(WHITE)
    draw_text(screen, f"Game Over!", FONT_LARGE, BLACK, SCREEN_WIDTH // 2, 100, center=True)
    draw_text(screen, f"P1 Score: {player_1_score}", FONT_SMALL, BLACK, SCREEN_WIDTH // 2, 200, center=True)
    draw_text(screen, f"P2 Score: {player_2_score}", FONT_SMALL, BLACK, SCREEN_WIDTH // 2, 300, center=True)
    draw_text(screen, "Press R to Restart or M for Main Menu", FONT_SMALL, BLACK, SCREEN_WIDTH // 2, 400, center=True)
    pygame.display.flip()

    return player_1_score, player_2_score

# Main function
def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("GOPS Game")

    card_images = load_images("card_image")
    hsv_values = load_hsv_values("config/hsv_values.yaml")
    model = tf.keras.models.load_model("model/cnn_model.h5")
    class_mapping = {0: "2_sekop", 1: "2_keriting", 2: "2_wajik"}  # Add proper mappings

    clock = pygame.time.Clock()
    running = True

    while running:
        main_menu(screen)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    game_loop(screen, card_images, "PvP", hsv_values, model, class_mapping)
                elif event.key == pygame.K_2:
                    game_loop(screen, card_images, "PvBot-Easy", hsv_values, model, class_mapping)
                elif event.key == pygame.K_3:
                    game_loop(screen, card_images, "PvBot-Hard", hsv_values, model, class_mapping)
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
