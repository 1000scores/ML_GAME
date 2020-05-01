"""
Sprite with Moving Platforms

Load a map stored in csv format, as exported by the program 'Tiled.'

Artwork from http://kenney.nl

If Python and Arcade are installed, this example can be run from the command line with:
python -m arcade.examples.sprite_moving_platforms
"""
import arcade
import os
import random
import ML as ml
import torch
import torch.nn as nn

SPRITE_SCALING = 0.5

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Sprite with Moving Platforms Example"
SPRITE_PIXEL_SIZE = 128
GRID_PIXEL_SIZE = (SPRITE_PIXEL_SIZE * SPRITE_SCALING)

# How many pixels to keep as a minimum margin between the character
# and the edge of the screen.
VIEWPORT_MARGIN = SPRITE_PIXEL_SIZE * SPRITE_SCALING
RIGHT_MARGIN = 4 * SPRITE_PIXEL_SIZE * SPRITE_SCALING

# Physics
MOVEMENT_SPEED = 40 * SPRITE_SCALING
JUMP_SPEED = 28 * SPRITE_SCALING
GRAVITY = 2 * SPRITE_SCALING

nextBox = 0
PLAY_BEST = False
BOT_PLAY = True


lastX = -1

listSize1 = 200
listSize2 = 30

class MyGame(arcade.Window):
    """ Main application class. """
    pre = 400
    preI = 0
    myWallList = []
    def __init__(self, width, height, title):
        """
        Initializer
        """
        self.lastX = -1
        super().__init__(width, height, title)

        # Set the working directory (where we expect to find files) to the same
        # directory this .py file is in. You can leave this out of your own
        # code, but it is needed to easily run the examples using "python -m"
        # as mentioned at the top of this program.
        file_path = os.path.dirname(os.path.abspath(__file__))
        os.chdir(file_path)

        # Sprite lists
        self.all_sprites_list = None
        self.all_wall_list = None
        self.static_wall_list = None
        self.moving_wall_list = None
        self.player_list = None
        self.coin_list = None

        # Set up the player
        self.player_sprite = None
        self.physics_engine = None
        self.view_left = 0
        self.view_bottom = 0
        self.end_of_map = 0
        self.game_over = False
        self.pre = 400
        self.preI = 0

    lastScore = -1
    startWallInd = 0
    bestModel = None
    def setup(self):
        """ Set up the game and initialize the variables. """

        self.nextDist = 2000
        self.pre = 400
        self.preI = 0
        # Sprite lists
        self.all_wall_list = arcade.SpriteList()
        self.static_wall_list = arcade.SpriteList()
        self.player_list = arcade.SpriteList()

        # Set up the player
        self.player_sprite = arcade.Sprite(":resources:images/animated_characters/female_person/femalePerson_idle.png", SPRITE_SCALING)
        self.player_sprite.center_x = 2 * GRID_PIXEL_SIZE
        self.player_sprite.center_y = 3 * GRID_PIXEL_SIZE
        self.player_list.append(self.player_sprite)
        self.myWallList = []
        self.startWallInd = 0


        # Create floor
        for i in range(1000):
            wall = arcade.Sprite(":resources:images/tiles/grassMid.png", SPRITE_SCALING)
            wall.bottom = 0
            wall.center_x = i * GRID_PIXEL_SIZE
            self.static_wall_list.append(wall)
            self.all_wall_list.append(wall)
            self.preI = i

        for i in range(100):
            wall = arcade.Sprite(":resources:images/tiles/grassMid.png", SPRITE_SCALING)
            wall.bottom = wall.width
            curR = random.randint(500, 1000)
            wall.center_x = self.pre + curR

            self.all_wall_list.append(wall)
            self.static_wall_list.append(wall)
            self.myWallList.append(self.pre + curR)


            self.pre = wall.center_x


        self.physics_engine = \
            arcade.PhysicsEnginePlatformer(self.player_sprite,
                                           self.all_wall_list,
                                           gravity_constant=GRAVITY)

        # Set the background color
        arcade.set_background_color(arcade.color.AMAZON)

        # Set the viewport boundaries
        # These numbers set where we have 'scrolled' to.
        self.view_left = 0
        self.view_bottom = 0

        self.game_over = False
        if PLAY_BEST:
            self.pop.allPops[0][0].load_state_dict(torch.load('createdModel.pt'))
            self.pop.allPops[0][0].eval()
            self.bestModel = self.pop.allPops[0][0]




    mem = False

    nextDist = 1000



    def on_draw(self):
        """
        Render the screen.
        """
        if self.mem:
            print('hhh')
        # This command has to happen before we start drawing
        arcade.start_render()

        '''if self.game_over:
            self.game_over = False
            self.lastScore = self.player_sprite.right
            print('tried')
            self.setup()'''


        # Draw the sprites.
        self.static_wall_list.draw()

        self.player_list.draw()

        # Put the text on the screen.
        # Adjust the text position based on the viewport so that we don't
        # scroll the text too.
        distance = self.player_sprite.right
        output = f"Distance: {distance}"
        arcade.draw_text(output, self.view_left + 10, self.view_bottom + 20,
                         arcade.color.WHITE, 14)

        output = f"Last Score: {self.lastScore}"
        arcade.draw_text(output, self.view_left + 400, self.view_bottom + 20,
                         arcade.color.WHITE, 14)

        output = f"Generation: {self.bigInd}"
        arcade.draw_text(output, self.view_left + 10, self.view_bottom + 400,
                         arcade.color.WHITE, 14)

        output = f"Index in generation: {self.smallInd}"
        arcade.draw_text(output, self.view_left + 10, self.view_bottom + 300,
                         arcade.color.WHITE, 14)


    def on_key_press(self, key, modifiers):
        """
        Called whenever the mouse moves.
        """
        if key == arcade.key.W:
            if self.physics_engine.can_jump():
                self.player_sprite.change_y = JUMP_SPEED
        '''elif key == arcade.key.LEFT:
            self.player_sprite.change_x = -MOVEMENT_SPEED
        elif key == arcade.key.RIGHT:
            self.player_sprite.change_x = MOVEMENT_SPEED'''

    def on_key_release(self, key, modifiers):
        """
        Called when the user presses a mouse button.
        """
        '''if key == arcade.key.LEFT or key == arcade.key.RIGHT:
            self.player_sprite.change_x = 0'''



    smallInd = 0
    bigInd = 0

    pop = ml.Population()

    curElems = 10


    def on_update(self, delta_time):


        """ Movement and game logic """



        # Call update on all sprites
        self.physics_engine.update()
        if int(self.player_sprite.center_x) == self.lastX:
            self.lastScore = self.player_sprite.right
            print('tried')

            self.pop.results[self.bigInd][self.smallInd] = self.player_sprite.right

            self.smallInd = self.smallInd + 1

            if self.smallInd >= self.curElems:
                self.smallInd = 0
                self.bigInd = self.bigInd + 1
                self.curElems = self.curElems - 1
                self.pop.makeNext()

            self.setup()
            return

        if BOT_PLAY:

            if not(PLAY_BEST) and self.player_sprite.right >= 30000:
                self.pop.save_model(self.bigInd, self.smallInd, 'bestModel2.pt', leave=True)
                print('Saved best model!')

            if self.physics_engine.can_jump():
                for i in range(self.startWallInd, len(self.myWallList)):
                    if self.myWallList[i] > self.player_sprite.right:
                        self.startWallInd = i
                        break

                input = list()
                input.append((self.myWallList[self.startWallInd] - self.player_sprite.right) / 20.0)
                input.append((self.myWallList[self.startWallInd + 1] - self.player_sprite.right) / 40.0)

                print(input)
                input = torch.tensor(input)
                if PLAY_BEST:
                    jump = self.bestModel(input)

                else:
                    jump = self.pop.allPops[self.bigInd][self.smallInd](input)

                print(jump[0].item())
                if jump < 0.5:
                    # print('Wanted to jump')
                    self.player_sprite.change_y = JUMP_SPEED




        '''if self.player_sprite.right >= self.nextDist:
            self.nextDist = self.nextDist + 1000
            temp = []
            for box in self.all_wall_list:
                if box.center_x + 200 > self.player_sprite.center_x:
                    temp.append(box)
            self.all_wall_list = temp
            self.static_wall_list = temp



            for i in range(listSize2):
                wall = arcade.Sprite(":resources:images/tiles/grassMid.png", SPRITE_SCALING)
                wall.bottom = wall.width
                curR = random.randint(300, 700)
                wall.center_x = self.pre + curR

                self.all_wall_list.append(wall)
                self.static_wall_list.append(wall)
                myWallList.append(self.pre + curR)


            tempSize = self.preI + (listSize1 - len(self.all_wall_list))
            for i in range(self.preI, tempSize):
                wall = arcade.Sprite(":resources:images/tiles/grassMid.png", SPRITE_SCALING)
                wall.bottom = 0
                wall.center_x = i * GRID_PIXEL_SIZE
                self.static_wall_list.append(wall)
                self.all_wall_list.append(wall)
                self.preI = i



                self.pre = wall.center_x
            self.mem = True'''


        self.lastX = self.player_sprite.center_x
        # --- Manage Scrolling ---

        # Track if we need to change the viewport

        self.player_sprite.change_x = MOVEMENT_SPEED

        changed = False

        # Scroll right
        right_boundary = self.view_left + SCREEN_WIDTH - RIGHT_MARGIN - 600
        if self.player_sprite.right > right_boundary:
            self.view_left += self.player_sprite.right - right_boundary
            changed = True


        # If we need to scroll, go ahead and do it.
        if changed:
            arcade.set_viewport(self.view_left,
                                SCREEN_WIDTH + self.view_left,
                                self.view_bottom,
                                SCREEN_HEIGHT + self.view_bottom)


def main():
    """ Main method """
    window = MyGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()

