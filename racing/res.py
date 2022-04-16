import pyglet

def centered(location):
    image = pyglet.resource.image(location)
    image.anchor_x = image.width // 2
    image.anchor_y = image.height // 2

    return image

pyglet.resource.path = ['res']
pyglet.resource.reindex()

car_img = centered('car.png')
dead_car_img = centered('dead car.png')
ghost_car_img = centered('ghost car.png')