#! /usr/bin/env python3

"""Flappy Bird, implemented using Pygame."""

import sys
import math
import os
from numpy import *
import sympy
from random import randint
from collections import deque

import pygame
from pygame.locals import *


FPS = 60
ANIMATION_SPEED = 0.17  # pixels per millisecond
WIN_WIDTH = 284 * 2    # BG image size: 284x512 px; tiled twice
WIN_HEIGHT = 512


class Bird(pygame.sprite.Sprite):
    """Represents the bird controlled by the player.

    The bird is the 'hero' of this game.  The player can make it climb
    (ascend quickly), otherwise it sinks (descends more slowly).  It must
    pass through the space in between pipes (for every pipe passed, one
    point is scored); if it crashes into a pipe, the game ends.

    Attributes:
    x: The bird's X coordinate.
    y: The bird's Y coordinate.
    msec_to_climb: The number of milliseconds left to climb, where a
        complete climb lasts Bird.CLIMB_DURATION milliseconds.

    Constants:
    WIDTH: The width, in pixels, of the bird's image.
    HEIGHT: The height, in pixels, of the bird's image.
    SINK_SPEED: With which speed, in pixels per millisecond, the bird
        descends in one second while not climbing.
    CLIMB_SPEED: With which speed, in pixels per millisecond, the bird
        ascends in one second while climbing, on average.  See also the
        Bird.update docstring.
    CLIMB_DURATION: The number of milliseconds it takes the bird to
        execute a complete climb.
    """

    WIDTH = HEIGHT = 32
    SINK_SPEED = 0.18
    CLIMB_SPEED = 0.8
    CLIMB_DURATION = 70

    def __init__(self, x, y, msec_to_climb, images):
        """Initialise a new Bird instance.

        Arguments:
        x: The bird's initial X coordinate.
        y: The bird's initial Y coordinate.
        msec_to_climb: The number of milliseconds left to climb, where a
            complete climb lasts Bird.CLIMB_DURATION milliseconds.  Use
            this if you want the bird to make a (small?) climb at the
            very beginning of the game.
        images: A tuple containing the images used by this bird.  It
            must contain the following images, in the following order:
                0. image of the bird with its wing pointing upward
                1. image of the bird with its wing pointing downward
        """
        super(Bird, self).__init__()
        self.dead = False
        self.score = 0
        self.prev_outp = 0
        self.outp = 0
        self.x, self.y = x, y
        self.msec_to_climb = msec_to_climb
        self._img_wingup, self._img_wingdown = images
        self._mask_wingup = pygame.mask.from_surface(self._img_wingup)
        self._mask_wingdown = pygame.mask.from_surface(self._img_wingdown)

    def update(self, delta_frames=1):
        """Update the bird's position.

        This function uses the cosine function to achieve a smooth climb:
        In the first and last few frames, the bird climbs very little, in the
        middle of the climb, it climbs a lot.
        One complete climb lasts CLIMB_DURATION milliseconds, during which
        the bird ascends with an average speed of CLIMB_SPEED px/ms.
        This Bird's msec_to_climb attribute will automatically be
        decreased accordingly if it was > 0 when this method was called.

        Arguments:
        delta_frames: The number of frames elapsed since this method was
            last called.
        """
        if self.msec_to_climb > 0:
            frac_climb_done = 1 - self.msec_to_climb/Bird.CLIMB_DURATION
            self.y -= (Bird.CLIMB_SPEED * frames_to_msec(delta_frames) *
                       (1 - math.cos(frac_climb_done * math.pi)))
            self.msec_to_climb -= frames_to_msec(delta_frames)
        else:
            self.y += Bird.SINK_SPEED * frames_to_msec(delta_frames)

    @property
    def image(self):
        """Get a Surface containing this bird's image.

        This will decide whether to return an image where the bird's
        visible wing is pointing upward or where it is pointing downward
        based on pygame.time.get_ticks().  This will animate the flapping
        bird, even though pygame doesn't support animated GIFs.
        """
        if pygame.time.get_ticks() % 500 >= 250:
            return self._img_wingup
        else:
            return self._img_wingdown

    @property
    def mask(self):
        """Get a bitmask for use in collision detection.

        The bitmask excludes all pixels in self.image with a
        transparency greater than 127."""
        if pygame.time.get_ticks() % 500 >= 250:
            return self._mask_wingup
        else:
            return self._mask_wingdown

    @property
    def rect(self):
        """Get the bird's position, width, and height, as a pygame.Rect."""
        return Rect(self.x, self.y, Bird.WIDTH, Bird.HEIGHT)


class PipePair(pygame.sprite.Sprite):
    """Represents an obstacle.

    A PipePair has a top and a bottom pipe, and only between them can
    the bird pass -- if it collides with either part, the game is over.

    Attributes:
    x: The PipePair's X position.  This is a float, to make movement
        smoother.  Note that there is no y attribute, as it will only
        ever be 0.
    image: A pygame.Surface which can be blitted to the display surface
        to display the PipePair.
    mask: A bitmask which excludes all pixels in self.image with a
        transparency greater than 127.  This can be used for collision
        detection.
    top_pieces: The number of pieces, including the end piece, in the
        top pipe.
    bottom_pieces: The number of pieces, including the end piece, in
        the bottom pipe.

    Constants:
    WIDTH: The width, in pixels, of a pipe piece.  Because a pipe is
        only one piece wide, this is also the width of a PipePair's
        image.
    PIECE_HEIGHT: The height, in pixels, of a pipe piece.
    ADD_INTERVAL: The interval, in milliseconds, in between adding new
        pipes.
    """

    WIDTH = 80
    PIECE_HEIGHT = 8
    ADD_INTERVAL = 3000

    def __init__(self, pipe_end_img, pipe_body_img, pop_size, points):
        """Initialises a new random PipePair.

        The new PipePair will automatically be assigned an x attribute of
        float(WIN_WIDTH - 1).

        Arguments:
        pipe_end_img: The image to use to represent a pipe's end piece.
        pipe_body_img: The image to use to represent one horizontal slice
            of a pipe's body.
        """
        self.x = float(WIN_WIDTH - 1)
        self.points = points
        self.scores_counted = [False] * pop_size
        self.image = pygame.Surface((PipePair.WIDTH, WIN_HEIGHT), SRCALPHA)
        self.image.convert()   # speeds up blitting
        self.image.fill((0, 0, 0, 0))
        points = min(points, 20)
        reducer = points/15
        total_pipe_body_pieces = int(
            (WIN_HEIGHT -                  # fill window from top to bottom
             # make room for bird to fit through
             (5.5 - reducer) * Bird.HEIGHT -
             3 * PipePair.PIECE_HEIGHT) /  # 2 end pieces + 1 body piece
            PipePair.PIECE_HEIGHT          # to get number of pipe pieces
        )
        self.bottom_pieces = randint(1, total_pipe_body_pieces)
        self.top_pieces = total_pipe_body_pieces - self.bottom_pieces

        # bottom pipe
        for i in range(1, self.bottom_pieces + 1):
            piece_pos = (0, WIN_HEIGHT - i*PipePair.PIECE_HEIGHT)
            self.image.blit(pipe_body_img, piece_pos)
        bottom_pipe_end_y = WIN_HEIGHT - self.bottom_height_px
        bottom_end_piece_pos = (0, bottom_pipe_end_y - PipePair.PIECE_HEIGHT)
        self.image.blit(pipe_end_img, bottom_end_piece_pos)

        # top pipe
        for i in range(self.top_pieces):
            self.image.blit(pipe_body_img, (0, i * PipePair.PIECE_HEIGHT))
        top_pipe_end_y = self.top_height_px
        self.image.blit(pipe_end_img, (0, top_pipe_end_y))

        # compensate for added end pieces
        self.top_pieces += 1
        self.bottom_pieces += 1

        # for collision detection
        self.mask = pygame.mask.from_surface(self.image)

    @property
    def top_height_px(self):
        """Get the top pipe's height, in pixels."""
        return self.top_pieces * PipePair.PIECE_HEIGHT

    @property
    def bottom_height_px(self):
        """Get the bottom pipe's height, in pixels."""
        return self.bottom_pieces * PipePair.PIECE_HEIGHT

    @property
    def visible(self):
        """Get whether this PipePair on screen, visible to the player."""
        return -PipePair.WIDTH < self.x < WIN_WIDTH

    @property
    def rect(self):
        """Get the Rect which contains this PipePair."""
        return Rect(self.x, 0, PipePair.WIDTH, PipePair.PIECE_HEIGHT)

    def update(self, delta_frames=1):
        """Update the PipePair's position.

        Arguments:
        delta_frames: The number of frames elapsed since this method was
            last called.
        """
        self.x -= ANIMATION_SPEED * frames_to_msec(delta_frames)

    def collides_with(self, bird):
        """Get whether the bird collides with a pipe in this PipePair.

        Arguments:
        bird: The Bird which should be tested for collision with this
            PipePair.
        """
        return pygame.sprite.collide_mask(self, bird)


class AI():

    UPDATE_INTERVAL = 50
    NINPUT = 3
    NOUTPUT = 1
    NHIDDEN_UNITS = 4

    def __init__(self, params):

        params = array(params)

        self.hidden_weights = sympy.Matrix(
            AI.NINPUT, AI.NHIDDEN_UNITS, params[0:(AI.NHIDDEN_UNITS * AI.NINPUT)])
        self.hidden_bias = sympy.Matrix(
            params[len(self.hidden_weights):
                   (len(self.hidden_weights) + AI.NHIDDEN_UNITS)]
        )
        self.outp_weights = sympy.Matrix(
            AI.NHIDDEN_UNITS, AI.NOUTPUT,
            params[(len(self.hidden_weights) + len(self.hidden_bias)):
                   ((len(self.hidden_weights) + len(self.hidden_bias)) + AI.NHIDDEN_UNITS*AI.NOUTPUT)]
        )
        self.outp_bias = sympy.Matrix(
            params[(len(self.hidden_weights) +
                    len(self.hidden_bias) + len(self.outp_weights)):]
        )

    def output(self, inputvec):
        inpvec = sympy.Matrix(inputvec).T
        hiddenvec = AI.relu(inpvec*self.hidden_weights + self.hidden_bias.T)
        outp = AI.sigmoid(hiddenvec*self.outp_weights + self.outp_bias.T)
        return int(round(outp[0][0]))

    def relu(x1):
        x1 = array(x1)
        return maximum(x1, 0)

    def sigmoid(x1):
        x1 = array(x1)
        return 1/(1 + exp(-x1.astype(float)))


class GA():

    def select_mating_pool(pop, fitness, num_parents):
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = empty((num_parents, pop.shape[1]))
        for parent_num in range(num_parents):
            fitness = array(fitness)
            max_fitness_idx = where(fitness == max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num, :] = pop[max_fitness_idx, :]
            fitness[max_fitness_idx] = -1000
        return parents

    def crossover(parents, offspring_size):
        offspring = empty(offspring_size)
        # The point at which crossover takes place between two parents. Usually, it is at the center.

        for k in range(offspring_size[0]):
            crossover_point = int(
                round(random.uniform(low=0, high=offspring_size[1])))
            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1) % parents.shape[0]
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx,
                                                      0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx,
                                                     crossover_point:]
        return offspring

    def mutation(offspring_crossover, p):
        # Mutation changes a single gene in each offspring randomly.
        for idx in range(offspring_crossover.shape[0]):
            # The random value to be added to the gene.
            for i in range(offspring_crossover.shape[1]):
                val = random.uniform()
                if val <= p:
                    random_value = random.normal(0, 0.5)
                    offspring_crossover[idx,
                                        i] = offspring_crossover[idx, i] + random_value
        return offspring_crossover


def load_images():
    """Load all images required by the game and return a dict of them.

    The returned dict has the following keys:
    background: The game's background image.
    bird-wingup: An image of the bird with its wing pointing upward.
        Use this and bird-wingdown to create a flapping bird.
    bird-wingdown: An image of the bird with its wing pointing downward.
        Use this and bird-wingup to create a flapping bird.
    pipe-end: An image of a pipe's end piece (the slightly wider bit).
        Use this and pipe-body to make pipes.
    pipe-body: An image of a slice of a pipe's body.  Use this and
        pipe-body to make pipes.
    """

    def load_image(img_file_name):
        """Return the loaded pygame image with the specified file name.

        This function looks for images in the game's images folder
        (./images/).  All images are converted before being returned to
        speed up blitting.

        Arguments:
        img_file_name: The file name (including its extension, e.g.
            '.png') of the required image, without a file path.
        """
        file_name = os.path.join('.', 'images', img_file_name)
        img = pygame.image.load(file_name)
        img.convert()
        return img

    return {'background': load_image('background.png'),
            'pipe-end': load_image('pipe_end.png'),
            'pipe-body': load_image('pipe_body.png'),
            # images for animating the flapping bird -- animated GIFs are
            # not supported in pygame
            'bird-wingup': load_image('bird_wing_up.png'),
            'bird-wingdown': load_image('bird_wing_down.png')}


def frames_to_msec(frames, fps=FPS):
    """Convert frames to milliseconds at the specified framerate.

    Arguments:
    frames: How many frames to convert to milliseconds.
    fps: The framerate to use for conversion.  Default: FPS.
    """
    return 1000.0 * frames / fps


def msec_to_frames(milliseconds, fps=FPS):
    """Convert milliseconds to frames at the specified framerate.

    Arguments:
    milliseconds: How many milliseconds to convert to frames.
    fps: The framerate to use for conversion.  Default: FPS.
    """
    return fps * milliseconds / 1000.0


def getInput(bird, pipes):
    x = WIN_WIDTH
    h_top = 0
    for p in pipes:
        if bird.x - PipePair.WIDTH <= p.x and p.x - bird.x + Bird.WIDTH <= x:
            x = p.x - bird.x + PipePair.WIDTH
            h_top = p.top_height_px - bird.y
            h_bot = p.bottom_height_px - bird.y + Bird.HEIGHT
    return array([x/WIN_WIDTH, h_top/(WIN_HEIGHT), h_bot/WIN_HEIGHT])


display_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption('Pygame Flappy Bird')


def play(pop_size, population, gen, not_run=False):

    points = 0
    pipes_added = 0
    clock = pygame.time.Clock()
    score_font = pygame.font.SysFont('calibri', 32, bold=True)  # default font
    images = load_images()

    birds = deque()
    for i in range(pop_size):
        mult = 1
        if i % 2 == 0:
            mult = -1
        bird = Bird(50, int(WIN_HEIGHT/2 + Bird.HEIGHT/2 + (mult * (i // 2 + 1) * Bird.HEIGHT / 10)), 2,
                    (images['bird-wingup'], images['bird-wingdown']))
        birds.append(bird)

    pipes = deque()
    scores = [0] * pop_size
    frame_clock = 0  # this counter is only incremented if the game isn't paused

    done = paused = False

    while not done:
        clock.tick(FPS)

# Handle this 'manually'.  If we used pygame.time.set_timer(),
# pipe addition would be messed up when paused.
        if not (paused or frame_clock % msec_to_frames(PipePair.ADD_INTERVAL)):
            pp = PipePair(images['pipe-end'],
                          images['pipe-body'], pop_size, pipes_added)
            pipes_added += 1
            pipes.append(pp)

        for e in pygame.event.get():
            if e.type == QUIT or (e.type == KEYUP and e.key == K_ESCAPE):
                done = True
                break
            if e.type == pygame.MOUSEBUTTONDOWN:
                if start_button_rect.collidepoint(e.pos):
                    start_button_text = "Start game"
                    done = True
                    main()

        if not (paused or frame_clock % msec_to_frames(AI.UPDATE_INTERVAL)):
            for i in range(len(birds)):
                if not birds[i].dead:
                    inp = getInput(birds[i], pipes)
                    # print(inp)
                    outp = population[i].output(inp)
                    if birds[i].prev_outp != outp:
                        birds[i].score += 0.1
                    birds[i].prev_outp = birds[i].outp
                    birds[i].outp = outp

                    if outp == 1:
                        birds[i].msec_to_climb = Bird.CLIMB_DURATION

        if paused:
            continue  # don't draw anything

        # check for collisions
        for bird in birds:
            if not bird.dead:
                # pipe_collision = False
                bird.score += 0.01
                for p in pipes:
                    if p.collides_with(bird) or -Bird.HEIGHT >= bird.y or bird.y >= WIN_HEIGHT - Bird.HEIGHT:
                        bird.dead = True

        alldead = 0
        for bird in birds:
            if (bird.dead):
                alldead += 1
        if (alldead == pop_size):
            done = True

        for x in (0, WIN_WIDTH / 2):
            display_surface.blit(images['background'], (x, 0))

        while pipes and not pipes[0].visible:
            pipes.popleft()

        for p in pipes:
            p.update()
            display_surface.blit(p.image, p.rect)

        for bird in birds:
            if not bird.dead:
                bird.update()
                display_surface.blit(bird.image, bird.rect)
            # update and display score
            for p in pipes:
                if p.x + PipePair.WIDTH < birds[i].x and not p.scores_counted[i]:
                    bird.score += 1
                    if (p.points == points):
                        points += 1
                    p.scores_counted[i] = True

        for i in range(len(birds)):
            scores[i] = birds[i].score

        gen_surface = score_font.render(
            "Generation: " + str(gen), True, (255, 255, 255))
        score_surface = score_font.render(
            "Score: " + str(round(max(scores), 1)), True, (255, 255, 255))
        passed_surface = score_font.render(
            "Passed pipes: " + str(round(points)), True, (255, 255, 255))
        gen_x = WIN_WIDTH/2 - gen_surface.get_width()/2
        display_surface.blit(
            passed_surface, (gen_x, PipePair.PIECE_HEIGHT +
                             gen_surface.get_height() + score_surface.get_height() + 5)
        )
        display_surface.blit(
            score_surface, (gen_x, PipePair.PIECE_HEIGHT + gen_surface.get_height() + 5))
        display_surface.blit(gen_surface, (gen_x, PipePair.PIECE_HEIGHT))

        if not_run:
            return

        draw_start_button()

        pygame.event.pump()
        pygame.display.flip()

        frame_clock += 1

    return scores


# Colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
# Slider parameters
slider_x = 100
slider_y = 300
slider_width = 300
slider_height = 20
slider_prob_y = 400
# Slider button parameters
button_width = 20
button_height = 40
slider_min = 4
slider_max = 50
slider_prob_min = 0
slider_prob_max = 100
# Font for text display
start_button_rect = pygame.Rect(slider_x, slider_y + 150, 180, 50)
start_button_text = "Start Game"
start_button_clicked = False

# Font for text display


def draw_start_button(playing=True):
    text = "Start game"
    if playing:
        text = "Stop game"
    font = pygame.font.SysFont('calibri', 36)
    pygame.draw.rect(display_surface, red, start_button_rect)
    textobj = font.render(text, True, white)
    display_surface.blit(textobj, (start_button_rect.x +
                         10, start_button_rect.y + 10))


def draw_slider(selected_value):
    pygame.draw.rect(display_surface, white, (slider_x,
                     slider_y, slider_width, slider_height))
    button_x = slider_x + (slider_width - button_width) * \
        (selected_value - slider_min) / (slider_max - slider_min)
    pygame.draw.rect(display_surface, red, (button_x, slider_y -
                     (button_height - slider_height) / 2, button_width, button_height))
    font = pygame.font.SysFont('calibri', 32)
    text = font.render("Population size: " + str(selected_value), True, black)
    display_surface.blit(text, (slider_x, slider_y - 50))


def draw_slider_prob(selected_value):
    pygame.draw.rect(display_surface, white, (slider_x,
                     slider_prob_y, slider_width, slider_height))
    button_x = slider_x + (slider_width - button_width) * \
        (selected_value - slider_prob_min) / \
        (slider_prob_max - slider_prob_min)
    pygame.draw.rect(display_surface, red, (button_x, slider_prob_y -
                     (button_height - slider_height) / 2, button_width, button_height))
    font = pygame.font.SysFont('calibri', 32)
    text = font.render("Mutation probability: " +
                       str(round(selected_value / 100, 2)), True, black)
    display_surface.blit(text, (slider_x, slider_prob_y - 50))


def update_population(sol_per_pop, selected_prob_value, not_run):
    if not_run:
        random.seed(1)
    param_size = 21
    # Defining the population size.

    pop_size = (sol_per_pop, param_size)
    # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.

    # Creating the initial population.
    new_population = random.normal(size=pop_size)

    num_generations = 10000

    num_parents_mating = 2
    gen = 1

    for generation in range(num_generations):
        population = [None] * sol_per_pop
        for i in range(len(population)):
            population[i] = AI(new_population[i])
        # Measuring the fitness of each chromosome in the population.
        fitness = play(sol_per_pop, population, gen, not_run)
        if not_run:
            return
        gen += 1
        print(max(fitness))
        # Selecting the best parents in the population for mating.
        parents = GA.select_mating_pool(
            new_population, fitness, num_parents_mating)

        # Generating next generation using crossover.
        offspring_crossover = GA.crossover(
            parents, (pop_size[0]-parents.shape[0], param_size))

        # Adding some variations to the offsrping using mutation.
        p = selected_prob_value
        if max(fitness) <= 2:
            p = 1
        offspring_mutation = GA.mutation(offspring_crossover, p)
        # Creating the new population based on the parents and offspring.
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation

    print("Game over!")
    pygame.quit()


def main():
    """The application's entry point.

    If someone executes this module (instead of importing it, for
    example), this function is called.
    """

    pygame.init()
    # the bird stays in the same x position, so bird.x is a constant
    # center bird on screen
    started = False
    selected_value = 10  # Initial value
    selected_prob_value = 0.1 * 100  # Initial value
    update_population(selected_value, selected_prob_value, True)
    slider1_selected = True
    while not started:
        prev_selected_val = selected_value
        prev_selected_prob_value = selected_prob_value
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if start_button_rect.collidepoint(event.pos):
                    started = True
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if slider_x <= mouse_x <= slider_x + slider_width & slider_y <= mouse_y <= slider_y + slider_height:
                    slider1_selected = True
                if slider_x <= mouse_x <= slider_x + slider_width & slider_prob_y <= mouse_y <= slider_prob_y + slider_height:
                    slider1_selected = False
            if event.type == pygame.MOUSEMOTION and pygame.mouse.get_pressed()[0]:
                # Check if the mouse is clicked and moving
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if slider_x <= mouse_x <= slider_x + slider_width:
                    # Ensure the slider button stays within the slider's bounds
                    if slider1_selected:
                        selected_value = round(
                            (mouse_x - slider_x) / slider_width * (slider_max - slider_min) + slider_min)
                        # Ensure the value remains within the specified range
                        selected_value = max(slider_min, min(
                            slider_max, selected_value))
                    else:
                        selected_prob_value = round(
                            (mouse_x - slider_x) / slider_width * (slider_prob_max - slider_prob_min) + slider_prob_min)
                        # Ensure the value remains within the specified range
                        selected_prob_value = max(slider_prob_min, min(
                            slider_prob_max, selected_prob_value))

        if selected_value != prev_selected_val or selected_prob_value != prev_selected_prob_value:
            update_population(selected_value, selected_prob_value, True)

        # Draw the slider
        draw_slider(selected_value)
        draw_slider_prob(selected_prob_value)

        draw_start_button(False)

        # Update the display
        pygame.display.flip()

    update_population(selected_value, selected_prob_value, False)


if __name__ == '__main__':
    # If this module had been imported, __name__ would be 'flappybird'.
    # It was executed (e.g. by double-clicking the file), so call main.
    main()
