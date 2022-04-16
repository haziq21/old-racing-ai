import numpy as np

class AABBTree:
    def __init__(self):
        # Eg. a branch node would be {'bb': (minx, miny, maxx, maxy), 'children': [{...}, {...}]}
        # and a leaf node would be {'bb': (minx, miny, maxx, maxy), 'reference': ...}
        self.tree = {'bb': (0, 0, 0, 0), 'children': []}

    def add_leaf(self, bounding_box, reference):
        new_leaf = {'bb': bounding_box, 'reference': reference}
        parent = self.tree
        ancestors = [parent]
        
        while True:
            # This will only be True for the first two leaf nodes added
            if len(parent['children']) < 2:
                parent['children'].append(new_leaf)
                break
            else:
                left_cost = self.cost(parent['children'][0]['bb'], bounding_box)
                right_cost = self.cost(parent['children'][1]['bb'], bounding_box)
                cheap_i = 0 if left_cost < right_cost else 1
                cheapest = parent['children'][cheap_i]

                # Cheapest node is a leaf node
                if 'reference' in cheapest:
                    new_branch = {'bb': self.combine(cheapest['bb'], bounding_box), 
                                  'children': [cheapest, new_leaf]}
                    parent['children'][cheap_i] = new_branch
                    break
                # Cheapest node is a branch node
                else:
                    ancestors.append(cheapest)
                    parent = cheapest
            
        for branch in ancestors[::-1]:
            # This will only be True for the first leaf node added
            if len(branch['children']) < 2:
                branch['bb'] = branch['children'][0]['bb']
            else:
                branch['bb'] = self.combine(branch['children'][0]['bb'], branch['children'][1]['bb'])

    def combine(self, bb1, bb2):
        return (min(bb1[0], bb2[0]), min(bb1[1], bb2[1]), max(bb1[2], bb2[2]), max(bb1[3], bb2[3]))
    
    def area(self, bb):
        return (bb[2] - bb[0]) * (bb[3] - bb[1])
    
    def cost(self, existing_bb, new_bb):
        return self.area(self.combine(new_bb, existing_bb)) - self.area(existing_bb)
    
    def intersects(self, bb1, bb2):
        return bb1[2] > bb2[0] and bb1[0] < bb2[2] and bb1[3] > bb2[1] and bb1[1] < bb2[3]

    def query(self, bounding_boxes):
        stack = [self.tree]
        collisions = [[] for _ in bounding_boxes]

        while len(stack) > 0:
            node = stack.pop()
            for bb in range(len(bounding_boxes)):
                if self.intersects(node['bb'], bounding_boxes[bb]):
                    if 'reference' in node:
                        collisions[bb].append(node['reference'])
                    else:
                        stack.append(node['children'][0])
                        # This will only be False when the tree has one leaf node
                        if len(node['children']) > 1:
                            stack.append(node['children'][1])
                        break

        return collisions
    
    def get_bounding_box(self, points):
        xs, ys = zip(*points)
        return min(xs), min(ys), max(xs), max(ys)