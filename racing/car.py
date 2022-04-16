import pyglet
import numpy as np

from racing import collision
from racing import neural_network as nn
from racing import res

class Car:
    def __init__(self, x, y, angle, batch=None, parents=None, evolve=True):
        self.start_pos = np.array([x, y]).astype(float)
        self.start_angle = angle

        # Drag against x velocity = (velocity + drag_shift)^2 * drag_force
        self.drag_force = 1.4 * 10**-4
        self.drag_shift = 35

        self.sensor_range = 150
        num_sensors = 7
        # Angles of distance sensors relative to the front of the car (in radians)
        self.sensor_angles = np.linspace(-np.pi/2, np.pi/2, num=num_sensors)

        self.max_accel = 1100
        self.max_turn_speed = 6
        
        self.sprite = pyglet.sprite.Sprite(res.car_img, x=x, y=y, batch=batch, subpixel=True)
        self.sprite.rotation = np.degrees(angle)

        # For get_walls() function
        self.half_diagonal = np.sqrt((self.sprite.width/2)**2 + (self.sprite.height/2)**2)
        self.diagonal_angle = np.arcsin(self.sprite.width / 2 / self.half_diagonal)

        # For autonomous control
        # Inputs are sensor readings and current velocity
        self.brain = nn.NeuralNetwork([num_sensors + 1, num_sensors + 4, num_sensors + 4, 2])
        if parents != None:
            if evolve:
                self.evolve(parents[0], parents[1])
            else:
                self.brain.set_flattened_hyperparams(parents[0], parents[1])

        self.reset()
            
    def drive(self, accel, time):
        # Thrust forward or backward
        dist = accel * time
        self.vel[0] += np.cos(self.angle) * dist
        self.vel[1] += -np.sin(self.angle) * dist

        # Calculate drag
        if (mag := np.sqrt(self.vel.dot(self.vel))) > 0:
            if mag > 2:
                # Velocity minus drag
                self.vel -= self.vel / mag * (mag + self.drag_shift)**2 * self.drag_force
                self.pos += self.vel * time
            else:
                # Anything below 2px/s is not visible
                # so snap to 0 to avoid unnecessary calculations
                self.vel = np.zeros(2)
        
        # Store total forward movement and time passed to calculate average speed after death
        movement = self.vel * time
        angle = -np.arctan2(movement[1], movement[0])  # -pi to pi, negative because pyglet
        self.total_movement += np.cos(angle - self.angle) * np.sqrt(movement.dot(movement))

        if np.sqrt(self.vel.dot(self.vel)) < 2:
            self.kill()

    def turn(self, radians):
        self.angle += radians
        self.total_rotation += abs(radians)

    def get_wall_points(self):
        '''Returns coordinates of vertices of rectangle (car) as 
        [top right, bottom right, bottom left, top left, top right]. 
        Positions relative to car pointing right.
        ''' 

        small_angle = self.angle - self.diagonal_angle
        big_angle = self.angle + self.diagonal_angle

        top_right = np.array([np.sin(small_angle), np.cos(small_angle)]) * self.half_diagonal
        bottom_right = np.array([np.sin(big_angle), np.cos(big_angle)]) * self.half_diagonal

        return [top_right + self.pos, bottom_right + self.pos,
                -top_right + self.pos, -bottom_right + self.pos, top_right + self.pos,]
    
    def get_sensor_points(self):
        '''Returns coordinates of furthest points distance sensors could reach relative to the center of the car.
        '''

        sensor_points = []

        for a in self.sensor_angles:
            angle = self.angle + a
            sensor_points.append(np.array([np.cos(angle), -np.sin(angle)]) * self.sensor_range + self.pos)

        return sensor_points
    
    def check_collision(self, possible_collisions, wall_points=None):
        if wall_points == None:
            wall_points = self.get_wall_points()
        # Loop through every wall car wall and check 
        # if it intersects with any line segment in possible_collisions
        for car_pt1, car_pt2 in zip(wall_points[:-1], wall_points[1:]):
            for wall_pt1, wall_pt2 in [[[w.x, w.y], [w.x2, w.y2]] for w in possible_collisions]:
                if collision.intersecting(car_pt1, car_pt2, wall_pt1, wall_pt2):
                    return True

        return False
    
    def get_sensor_readings(self, possible_collisions, sensor_points=None):
        if sensor_points == None:
            sensor_points = self.get_sensor_points()
        readings = []

        for sensor_pt in sensor_points:
            closest_reading = self.sensor_range
            for wall_pt1, wall_pt2 in [[np.array([w.x, w.y]), np.array([w.x2, w.y2])] for w in possible_collisions]:
                point = collision.get_intersection(self.pos, sensor_pt, wall_pt1, wall_pt2)
                if isinstance(point, np.ndarray):
                    point -= self.pos
                    closest_reading = min(closest_reading, np.sqrt(point.dot(point)))

            readings.append(closest_reading)
        
        return readings
    
    def reset(self):
        self.pos = self.start_pos.copy()
        self.vel = np.zeros(2)
        self.angle = self.start_angle
        self.sprite.update(self.pos[0], self.pos[1], np.degrees(self.angle))

        self.dead = False
        self.sprite.image = res.car_img

        self.total_movement = 0  # Total (forward) movement
        self.total_rotation = 0
        self.lifespan = 0
        self.fitness = 0
    
    def kill(self):
        self.dead = True
        self.sprite.image = res.dead_car_img
        if self.lifespan == 0:
            self.fitness = 0
        else:
            self.fitness = self.total_movement * abs(self.total_movement) * (self.total_rotation + self.lifespan) / self.lifespan**2
            # self.fitness = (np.sign(self.total_movement) * self.total_movement**2  / self.lifespan) * (1 / (self.total_rotation * 0.1 + 1))

    def evolve(self, hyperparams1, hyperparams2):
        weights1, biases1 = hyperparams1
        weights2, biases2 = hyperparams2

        # Crossover the hyperparameters of the parent neural networks
        weight_split = np.random.randint(len(weights1))
        bias_split = np.random.randint(len(biases1))
        new_weights = np.concatenate((weights1[:weight_split], weights2[weight_split:]))
        new_biases = np.concatenate((biases1[:bias_split], biases2[bias_split:]))

        # Mutate weights
        for weight in range(len(new_weights)):
            # 10% mutation rate
            if np.random.rand() <= 0.1:
                new_weights[weight] = np.random.standard_normal()
        
        # Mutate biases
        for bias in range(len(new_biases)):
            # 10% mutation rate
            if np.random.rand() <= 0.1:
                new_biases[bias] = np.random.standard_normal()
        
        self.brain.set_flattened_hyperparams(new_weights, new_biases)

    def update(self, dt, possible_car_collisions, car_points, possible_sensor_collisions, sensor_points):
        self.lifespan += dt

        # Check if car has died
        if self.check_collision(possible_car_collisions, car_points):
            self.kill()
            return
        
        inputs = np.concatenate((self.get_sensor_readings(possible_sensor_collisions, sensor_points), 
                                 [np.sqrt(self.vel.dot(self.vel))]))
        t, m = self.brain.predict(inputs)
        # Maybe do the math to turn and drive at the same time?
        self.turn(t * self.max_turn_speed * dt)
        self.drive(m * self.max_accel, dt)

        self.sprite.update(self.pos[0], self.pos[1], np.degrees(self.angle))
    
    def get_save_formatted(self):
        return {'fitness': self.fitness, 'x': self.pos[0], 'y': self.pos[1], 
                'angle': np.degrees(self.angle), 'hyperparams': self.brain.get_flattened_hyperparams()}