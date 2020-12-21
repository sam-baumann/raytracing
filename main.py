import numpy as np
import matplotlib.pyplot as plt

def normalize(vector):
    return vector / np.linalg.norm(vector)

def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center) ** 2 - radius ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return(min(t1, t2))
    return None

def nearest_intersection(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj["center"], obj["radius"], ray_origin, ray_direction) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            nearest_object = objects[index]
            min_distance = distance

    return nearest_object, min_distance

def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis

#based on https://medium.com/swlh/ray-tracing-from-scratch-in-python-41670e6a96f9
width = 300
height = 200
max_depth = 3

objects = [
    {"center": np.array([0, -.25, -1]), "radius": .5, "ambient": np.array([.1, 0, 0]), "diffuse": np.array([.7, 0, 0]), "specular": np.array([1, 1, 1]), "shininess": 100, "reflection": .15},
    {"center": np.array([-.1, .1, 0]), "radius": .1,  "ambient": np.array([0, .1, 0]), "diffuse": np.array([0, .7, 0]), "specular": np.array([1, 1, 1]), "shininess": 100, "reflection": .5},
    {"center": np.array([.1, -.3, 0]), "radius": .1, "ambient": np.array([0, 0, .1]), "diffuse": np.array([0, 0, .7]), "specular": np.array([1, 1, 1]), "shininess": 100, "reflection": .5},
    {'center': np.array([0, -9000, 0]), 'radius': 9000 - 0.7, 'ambient': np.array([0.1, 0.1, 0.1]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, "reflection": .5 }
]

light = {"position": np.array([3, 5, 5]), "ambient": np.array([1, 1, 1]), "diffuse": np.array([1, 1, 1]), "specular": np.array([1, 1, 1])}

camera = np.array([0, 0, 1])
ratio = float(width) / height
screen = (-1, 1 / ratio, 1, -1 / ratio) #left, top, right, bottom

image = np.zeros((height, width, 3))
for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
        pixel = np.array([x, y, 0])
        origin = camera
        direction = normalize(pixel - origin)
        color = np.zeros(3)
        reflection = 1

        for k in range(max_depth):
            nearest_object, min_distance = nearest_intersection(objects, origin, direction)
            if nearest_object:
                intersection = origin + min_distance * direction

                normal_to_surface = normalize(intersection - nearest_object["center"])
                shifted_point = intersection + 1e-5 * normal_to_surface
                intersection_to_light = light["position"] - shifted_point

                _, min_distance = nearest_intersection(objects, intersection, normalize(intersection_to_light))
                intersection_to_light_distance = np.linalg.norm(intersection_to_light)
                is_shadowed = min_distance < intersection_to_light_distance
                # use the blinn-phong model to calculate the illumination
                L = normalize(intersection_to_light)
                N = normal_to_surface
                V = normalize(camera - intersection)
                I = np.zeros(3)

                #begin with ambient
                I += nearest_object["ambient"] * light["ambient"]

                if not is_shadowed:
                    #now do diffuse
                    I += nearest_object["diffuse"] * light["diffuse"] * np.dot(L, N)

                    #finally, do specular
                    I += nearest_object["specular"] * light["specular"] * (np.dot(N, normalize(L + V)) ** (nearest_object["shininess"] / 4))

                color += I * reflection
                reflection *= nearest_object["reflection"]

                origin = shifted_point
                direction = reflected(direction, normal_to_surface)

        image[i, j] = np.clip(color, 0, 1)
    print("Progress: %d/%d" % (i + 1, height))

plt.imsave("image.png", image)