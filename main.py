import numpy as np
import matplotlib.pyplot as plt
import sys

class vec3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class color:
    def __init__(self, amb, dif, spec):
        self.ambient = np.array([amb.x, amb.y, amb.z])
        self.diffuse = np.array([dif.x, dif.y, dif.z])
        self.specular = np.array([spec.x, spec.y, spec.z])

class primitive_object:
    def __init__(self, pos, color, shininess, reflection):
        self.position = np.array([pos.x, pos.y, pos.z])
        self.ambient = color.ambient
        self.diffuse = color.diffuse
        self.specular = color.specular
        self.shininess = shininess
        self.reflection = reflection

class sphere(primitive_object):
    def __init__(self, radius, pos, color, shininess, reflection):
        super().__init__(pos, color, shininess, reflection)
        self.radius = radius

    def intersect(self, ray_origin, ray_direction):
        b = 2 * np.dot(ray_direction, ray_origin - self.position)
        c = np.linalg.norm(ray_origin - self.position) ** 2 - self.radius ** 2
        delta = b ** 2 - 4 * c
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / 2
            t2 = (-b - np.sqrt(delta)) / 2
            if t1 > 0 and t2 > 0:
                return(min(t1, t2))
        return None

    def calculate_normal_to_surface(self, point):
        return normalize(point - self.position)

class plane(primitive_object):
    def __init__(self, norm, pos, color, shininess, reflection):
        super().__init__(pos, color, shininess, reflection)
        self.normal = normalize(np.array([norm.x, norm.y, norm.z]))

    def intersect(self, ray_origin, ray_direction):
        if np.dot(self.normal, ray_direction) != 0:
            t = np.dot((self.position - ray_origin), self.normal) / np.dot(self.normal, ray_direction)
            if t >= 0:
                return t
        return None
    
    def calculate_normal_to_surface(self, point):
        return self.normal

def normalize(vector):
    return vector / np.linalg.norm(vector)

def nearest_intersection(objects, ray_origin, ray_direction):
    distances = [obj.intersect(ray_origin, ray_direction) for obj in objects]
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
if len(sys.argv) > 1:
    width = int(sys.argv[1])
    height = int(sys.argv[2])
max_depth = 2

#delcare colors
red = color(vec3(.1, 0, 0), vec3(.7, 0, 0), vec3(1, 1, 1))
blue = color(vec3(0, .1, 0), vec3(0, .7, 0), vec3(1, 1, 1))
green = color(vec3(0, 0, .1), vec3(0, 0, .7), vec3(1, 1, 1))
grey = color(vec3(.1, .1, .1), vec3(.7, .7, .7), vec3(1, 1, 1))

#declare objects in the scene
objects = [
    plane(vec3(0, 1, 0), vec3(0, -.7, 0), grey, 100, .15),
    sphere(.5, vec3(0, -.25, -1), red, 100, .5),
    sphere(.1, vec3(-.1, .1, 0), blue, 100, .5),
    sphere(.1, vec3(.1, -.3, 0), green, 100, .5)
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

                normal_to_surface = nearest_object.calculate_normal_to_surface(intersection)
                shifted_point = intersection + 1e-5 * normal_to_surface
                intersection_to_light = light["position"] - shifted_point

                _, min_distance = nearest_intersection(objects, shifted_point, normalize(intersection_to_light))
                intersection_to_light_distance = np.linalg.norm(intersection_to_light)
                is_shadowed = min_distance < intersection_to_light_distance
                # use the blinn-phong model to calculate the illumination
                L = normalize(intersection_to_light)
                N = normal_to_surface
                V = normalize(camera - intersection)
                I = np.zeros(3)

                #begin with ambient
                I += nearest_object.ambient * light["ambient"]

                if not is_shadowed:
                    #now do diffuse
                    I += nearest_object.diffuse * light["diffuse"] * np.dot(L, N)

                    #finally, do specular
                    I += nearest_object.specular * light["specular"] * (np.dot(N, normalize(L + V)) ** (nearest_object.shininess / 4))

                color += I * reflection
                reflection *= nearest_object.reflection

                origin = shifted_point
                direction = reflected(direction, normal_to_surface)
            else:
                break

        image[i, j] = np.clip(color, 0, 1)
    print("Progress: %d/%d" % (i + 1, height))

plt.imsave("images/image.png", image)