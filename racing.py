import json

import numpy as np
import pyglet
from pyglet.window import mouse, key

from racing import Car, AABBTree
from racing import res

##### Setup #####

window = pyglet.window.Window(width=1300, height=800, caption='Racing AI')

drawing_wall = False
ghost_wall = None

walls = []
wall_tree = AABBTree()
wall_batch = pyglet.graphics.Batch()

cars = []
car_batch = pyglet.graphics.Batch()

start_car = pyglet.sprite.Sprite(res.car_img, x=window.width//2, y=window.height//2, subpixel=True)
parent_car1 = pyglet.sprite.Sprite(res.ghost_car_img, batch=car_batch, subpixel=True)
parent_car2 = pyglet.sprite.Sprite(res.ghost_car_img, batch=car_batch, subpixel=True)

start_car.visible = False
parent_car1.visible = False
parent_car2.visible = False

blind = False
training = True
train_time = 0

# Check if program has been run and set up before
try:
    with open('saved/setup.json') as f:
        setup = json.load(f)
        generation = setup['generation']
        start_car.update(setup['x'], setup['y'], setup['angle'])
except:
    generation = 1

pyglet.gl.glClearColor(0.5, 1, 0.5, 1)

##### Text #####

pyglet.font.add_file('res/OsakaMono.ttf')

# Menu text
menu_doc = pyglet.text.decode_text('> New Track\n  Load Track')
menu_doc.set_style(0, -1, dict(font_name='Osaka-Mono', font_size=22, color=(0, 0, 128, 255)))
menu_layout = pyglet.text.layout.TextLayout(menu_doc, multiline=True, wrap_lines=False)
menu_layout.x, menu_layout.y = window.width // 2, window.height // 2
menu_layout.anchor_x = menu_layout.anchor_y = 'center'

# Current menu
NEW_OR_LOAD, TRAIN_OR_RUN = range(2)

current_menu = NEW_OR_LOAD
showing_menu = True
menu_index = 0

# Generation text

gen_label = pyglet.text.Label(font_name='Osaka-Mono', font_size=18, color=(0, 0, 128, 255), 
                              x=10, y=window.height - 5, anchor_y='top')
best_fitness_label = pyglet.text.Label(font_name='Osaka-Mono', font_size=18, color=(0, 0, 128, 255), 
                                       x=10, y=window.height - 25, anchor_y='top')
mean_fitness_label = pyglet.text.Label(font_name='Osaka-Mono', font_size=18, color=(0, 0, 128, 255), 
                                       x=10, y=window.height - 45, anchor_y='top')
death_count_label = pyglet.text.Label(font_name='Osaka-Mono', font_size=18, color=(0, 0, 128, 255), 
                                       x=10, y=window.height - 65, anchor_y='top')

##### Key handling #####

key_handler = key.KeyStateHandler()
window.push_handlers(key_handler)
previously_pressed = dict()

def pressed(key):
    new_press = key_handler[key] and not previously_pressed[key]
    previously_pressed[key] = key_handler[key]
    return new_press
    
##### Main loop #####

@window.event
def on_draw():
    window.clear()

    if not blind:
        wall_batch.draw()
        start_car.draw()
        car_batch.draw()
    else:
        death_count_label.draw()

    if showing_menu:
        menu_layout.draw()
    elif len(cars) > 0:
        gen_label.draw()
        best_fitness_label.draw()
        mean_fitness_label.draw()

def update(dt):
    global menu_index, current_menu, showing_menu
    global generation, blind, training, cars, train_time
    
    # Scroll and select menus
    if showing_menu:
        options = menu_doc.text.split('\n')
        options[menu_index] = '  ' + options[menu_index][2:]

        if (pressed(key.UP) or pressed(key.LEFT)) and menu_index > 0:
            menu_index -= 1
        if (pressed(key.DOWN) or pressed(key.RIGHT)) and menu_index < len(options) - 1:
            menu_index += 1
        
        options[menu_index] = '> ' + options[menu_index][2:]
        menu_doc.text = '\n'.join(options)

        if pressed(key.RETURN):
            if current_menu == NEW_OR_LOAD:
                if menu_index == 1:
                    with open('saved/track') as f:
                        for line in f:
                            if not line.isspace():
                                vals = np.array(line.split(), float)

                                for x, y, x2, y2 in zip(vals[:-2:2], vals[1:-1:2], vals[2::2], vals[3::2]):
                                    walls.append(pyglet.shapes.Line(x, y, x2, y2, 5, (128, 128, 128), wall_batch))
                                    wall_tree.add_leaf(wall_tree.get_bounding_box([[x, y], [x2, y2]]), 
                                                       walls[-1])

                    showing_menu = False
                    menu_index = 0
                    start_car.visible = True
            elif current_menu == TRAIN_OR_RUN:
                if menu_index == 0:
                    showing_menu = False
                    training = True
                    start_car.visible = False
                    gen_label.text = f'Gen {generation}'

                    try:
                        with open('saved/parents.json') as f:
                            parents = json.load(f)
                            parent_hyperparams = parents['parent1']['hyperparams'], parents['parent2']['hyperparams']
                            parent_car1.update(parents['parent1']['x'], parents['parent1']['y'], parents['parent1']['angle'])
                            parent_car1.visible = True
                            parent_car2.update(parents['parent2']['x'], parents['parent2']['y'], parents['parent2']['angle'])
                            parent_car2.visible = True
                    except:
                        parent_hyperparams = None

                    for _ in range(500):
                        cars.append(Car(start_car.x, start_car.y, 
                                        np.radians(start_car.rotation), 
                                        car_batch, parent_hyperparams))
                elif menu_index == 1:
                    training = False
                    try:
                        with open('saved/parents.json') as f:
                            parents = json.load(f)
                            for p in parents:
                                cars.append(Car(start_car.x, start_car.y, 
                                                np.radians(start_car.rotation), car_batch, 
                                                parents[p]['hyperparams'], False))
                        showing_menu = False
                        start_car.visible = False
                        gen_label.text = f'Gen {generation - 1}'
                    except:
                        pass
    elif len(cars) == 0:
        # Move starting car
        if generation == 1:
            if key_handler[key.W]:
                start_car.y += 200 * dt
            if key_handler[key.A]:
                start_car.x -= 200 * dt
            if key_handler[key.S]:
                start_car.y -= 200 * dt
            if key_handler[key.D]:
                start_car.x += 200 * dt
            if key_handler[key.R]:
                start_car.rotation -= 200 * dt
            if key_handler[key.T]:
                start_car.rotation += 200 * dt

        if pressed(key.RETURN) or generation > 1:
            showing_menu = True
            current_menu = TRAIN_OR_RUN
            menu_doc.text = '> Train\n  Run saved parents\n  Manual control'
    else:
        # Constant delta time so fluctuations in framerate do not affect car.
        # This is evident when the same generation loops yet course of cars are different.
        dt = 0.02

        live_cars = []
        car_points = []
        car_bbs = []
        sensor_points = []
        sensor_bbs = []

        for car in cars:
            if not car.dead:
                live_cars.append(car)
                car_points.append(wall_points := car.get_wall_points())
                car_bbs.append(wall_tree.get_bounding_box(wall_points))
                sensor_points.append(sensors := car.get_sensor_points())
                sensor_bbs.append(wall_tree.get_bounding_box(sensors + [car.pos]))

        car_collisions = wall_tree.query(car_bbs)
        sensor_collisions = wall_tree.query(sensor_bbs)

        for i in range(len(live_cars)):
            live_cars[i].update(dt, car_collisions[i], car_points[i], sensor_collisions[i], sensor_points[i])
        
        if training:
            train_time += dt
        
        if pressed(key.B):
            blind = not blind

        if (pressed(key.K) and not blind) or train_time >= 60:
            train_time = 0
            for car in cars:
                car.kill()

        # Reset everything if all cars are dead
        if all([car.dead for car in cars]):
            if training:
                # Get hyperparameters of two cars with greatest fitness (parents)
                bests = sorted(cars, key=lambda car: car.fitness)[-2:]

                best_fitness_label.text = f'Previous best fitness: {bests[1].fitness:.3f}'
                mean_fitness_label.text = f'Previous mean fitness: {sum(car.fitness for car in cars) / len(cars):.3f}'

                save = {'parent1': {'x': bests[0].pos[0], 'y':bests[0].pos[1], 'angle': np.degrees(bests[0].angle), 
                                    'hyperparams': bests[0].brain.get_flattened_hyperparams()}, 
                        'parent2': {'x': bests[1].pos[0], 'y':bests[1].pos[1], 'angle': np.degrees(bests[1].angle), 
                                    'hyperparams': bests[1].brain.get_flattened_hyperparams()}}

                parent_car1.update(x=save['parent1']['x'], y=save['parent1']['y'], rotation=save['parent1']['angle'])
                parent_car1.visible = True
                parent_car2.update(x=save['parent2']['x'], y=save['parent2']['y'], rotation=save['parent2']['angle'])
                parent_car2.visible = True

                for car in cars:
                    car.evolve(save['parent1']['hyperparams'], save['parent2']['hyperparams'])
                    car.reset()
                
                with open('saved/parents.json', 'w') as f:
                    json.dump(save, f)

                generation += 1
                gen_label.text = f'Gen {generation}'
                with open('saved/setup.json', 'w') as f:
                    json.dump({'generation': generation, 'x': start_car.x, 
                            'y': start_car.y, 'angle': start_car.rotation}, f)
            else:
                for car in cars:
                    car.reset()
        
        if blind:
            deaths = 0
            for car in cars:
                if car.dead:
                    deaths += 1

            death_count_label.text = f'{deaths} out of {len(cars)} dead'

if __name__ == "__main__":
    pyglet.clock.schedule_interval(update, 1/120)
    pyglet.app.run()