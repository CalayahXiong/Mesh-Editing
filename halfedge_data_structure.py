# TODO Junjun Xiong 261201887
import numpy as np

# Nones initially and updated in HalfedgeMesh.build(), since we only have the vertex positionsm and face vertex indices
class Vertex:
    def __init__(self, point):
        self.point = point
        self.halfedge = None
        self.index = None

class Halfedge:
    def __init__(self):
        self.vertex = None # source vertex
        self.twin = None
        self.next = None
        self.prev = None
        self.edge = None
        self.face = None
        self.index = None

class Edge:
    def __init__(self):
        self.halfedge = None
        self.index = None

class Face:
    def __init__(self, vertices, index=None):
        self.vertices = vertices
        self.halfedge = None
        self.index = index

class HalfedgeMesh:
    def __init__(self, vertices, faces):
        self.vertices = np.array([Vertex(pos) for pos in vertices])
        self.halfedges = []
        self.edges = []
        self.faces = []

        # Assign indices to vertices
        for idx, vertex in enumerate(self.vertices):
            vertex.index = idx
        # Create faces and assign indices
        for idx, face_vertex_ids in enumerate(faces):
            face_vertices = [self.vertices[id] for id in face_vertex_ids]
            face = Face(face_vertices, index=idx)
            self.faces.append(face)

        self.build()

    # Convenience functions to create new elements
    def new_vertex(self, point):
        vertex = Vertex(point)
        vertex.index = len(self.vertices)
        self.vertices = np.append(self.vertices, vertex)
        return vertex

    def new_face(self, vertices):
        face = Face(vertices)
        face.index = len(self.faces)
        self.faces = np.append(self.faces, face)
        return face

    def new_edge(self):
        edge = Edge()
        edge.index = len(self.edges)
        self.edges = np.append(self.edges, edge)
        return edge

    def new_halfedge(self):
        he = Halfedge()
        he.index = len(self.halfedges)
        self.halfedges = np.append(self.halfedges, he)
        return he

    '''
    Given HalfedgeMesh object (potentially with quads or ngons), return a tuple of numpy arrays (vertices, triangles, triangle_to_face) for rendering.
    vertices: (n, 3) array of vertex positions
    triangles: (m, 3) array of vertex indices forming triangles
    triangle_to_face: (m,) array of face indices corresponding to each triangle (needed for face selection especially) [tri_index] -> face_index
    '''
    def get_vertices_and_triangles(self):
        vertices = [vertex.point for vertex in self.vertices]
        triangles = []
        triangle_to_face = [] # map from triangle to face, {to generalize to n-gons}
        i = 0
        for face in self.faces:
            if len(face.vertices) == 3:
                triangles.append([vertex.index for vertex in face.vertices])
                triangle_to_face.append(i)
            else:
                # implement simple ear clipping algorithm
                triangles_vertices = triangulate([vertex for vertex in face.vertices])
                for triangle_vertices_triple in triangles_vertices:
                    triangles.append(triangle_vertices_triple)
                    triangle_to_face.append(i)
            i += 1
        return np.array(vertices), np.array(triangles), triangle_to_face

    # Build the halfedge data structure from the vertex positions and face vertex indices stored in self.vertices and self.faces
    # This is essential for all following objectives to work
    def build(self):
        # TODO: Objective 1: build the halfedge data structure
        # Hint: use a dict to keep track of edges, as halfedges are being created
        edge_dict = {}  # This will help track and assign twins to half-edges

        for face_index, face in enumerate(self.faces):
            face_halfedges = []
            num_vertices = len(face.vertices)

            # Create halfedges and edges
            for i in range(num_vertices):
                v1 = face.vertices[i]
                v2 = face.vertices[(i + 1) % num_vertices]

                edge_key = (v1.index, v2.index)
                twin_key = (v2.index, v1.index)

                # Create a new halfedge
                he = self.new_halfedge()
                he.vertex = v1
                he.face = face
                #v1.halfedge = v1.halfedge or he  # Assign a halfedge to vertex if not already set
                face_halfedges.append(he)

                if v1.halfedge is None:
                    v1.halfedge = he

                if twin_key in edge_dict:
                    # If a twin exists, link the current halfedge with its twin
                    twin_he = edge_dict[twin_key]
                    he.twin = twin_he
                    twin_he.twin = he
                    he.edge = twin_he.edge
                    #he.edge.halfedge = twin_he
                else:
                    # Create a new edge and assign it to the halfedge
                    edge = self.new_edge()
                    edge.halfedge = he  # Assign the current halfedge to the edge
                    he.edge = edge
                    edge_dict[edge_key] = he  # Store the current halfedge in the dictionary
                    if not hasattr(edge, 'halfedge'):
                        edge.halfedge = he

            # Link the halfedges in the face loop
            for j in range(num_vertices):
                face_halfedges[j].next = face_halfedges[(j + 1) % num_vertices]
                face_halfedges[(j + 1) % num_vertices].prev = face_halfedges[j]

            # Set one of the halfedges in the face
            face.halfedge = face_halfedges[0]
            face.index = face_index


        for vertex in self.vertices:
            if vertex.halfedge is None:
                vertex.halfedge = next((he for he in self.halfedges if he.vertex == vertex), None)

            # Assign an index to each vertex
            vertex.index = vertex.index if hasattr(vertex, 'index') else self.vertices.index(vertex)

        # Assign indices to all halfedges and edges
        for he_index, he in enumerate(self.halfedges):
            he.index = he_index

        for edge_index, edge in enumerate(self.edges):
            edge.index = edge_index

        self.sanity_check()
        print("build") # build ok

    # Given a face, loop over its halfedges he in order to update face.vertices and he.face after some operation
    def update_he_vertices_around_face(self, face):
        he = face.halfedge
        vertices = []
        while True:
            vertices.append(he.vertex)
            he = he.next
            he.face = face # update he face
            if he.index == face.halfedge.index:
                break
        face.vertices = vertices # update face vertices


    # Given an edge, with both sides being triangles, flip the edge
    #           v1                              v1
    #           /\                              /\
    #          /  \                            / |\
    #         /    \                          /  | \
    #        /______\                        /   |  \
    #          edge      -> flip edge ->         |
    #        \      /                        \   |  /
    #         \    /                          \  | /
    #          \  /                            \ |/
    #           \/                              \/

    """
    o 3a: HalfedgeMesh. 
    flip_edge() : given an edge with two adjacent faces, flip the edge to connect the other 2 vertices, should only work if the 2 faces are triangles.
    """
    def flip_edge(self, edge):
        # TODO: Objective 3a: flip the edge (only if both sides are triangles)
        # Retrieve half-edges
        he1 = edge.halfedge
        he2 = he1.twin

        # Get adjacent faces
        face1 = he1.face
        face2 = he2.face

        # Check if both faces are triangles
        if not (self.is_triangle(face1) and self.is_triangle(face2)):
            print("Edge flip only allowed for triangular faces.")
            return

        # Identify vertices of the edge and its adjacent triangles
        v1 = he1.vertex
        v2 = he2.vertex
        v3 = he1.next.next.vertex
        v4 = he2.next.next.vertex

        he1_next = he1.next
        he2_next = he2.next
        he1_next_next = he1.next.next
        he2_next_next = he2.next.next

        he1.vertex = v3
        he2.vertex = v4

        he1.next = he2_next_next
        he1.next.next = he1_next
        he1.next.next.next = he1
        he2.next = he1_next_next
        he2.next.next = he2_next
        he2.next.next.next = he2

        # self.update_he_vertices_around_face(face1)
        # self.update_he_vertices_around_face(face2)

        face1.halfedge = he1
        face2.halfedge = he2
        he1.face = face1
        he1.next.face = face1
        he1.next.next.face = face1
        he2.face = face2
        he2.next.face = face2
        he2.next.next.face = face2

        face1.vertices = [v3, v4, v2]
        face2.vertices = [v4, v3, v1]

        self.sanity_check()
        print("flip")

    def is_triangle(self, face):
        count = 0
        halfedge = face.halfedge
        while halfedge is not None:
            count += 1
            halfedge = halfedge.next
            if halfedge == face.halfedge:
                break
        return count==3


    # Given an edge, with both sides being triangles, split the edge in the middle, creating a new vertex and connecting
    # it to the facing corners of the 2 triangles
    #           v1                              v1
    #           /\                              /\
    #          /  \                            / |\
    #         /    \                          /  | \
    #        /______\                        /   |  \
    #          edge      -> split edge ->    ---v2---
    #        \      /                        \   |  /
    #         \    /                          \  | /
    #          \  /                            \ |/
    #           \/                              \/

    """
    o 3b: HalfedgeMesh. 
    split_edge() : given an edge, split it by adding a new vertex in the middle of the edge and additionally connect the new vertex to the 2 corners facing the edge.
    """
    def split_edge(self, edge):
        # TODO: Objective 3b: split the edge (only if both sides are triangles)
        # Retrieve the two halfedges associated with the edge
        he1 = edge.halfedge
        he2 = he1.twin

        # Retrieve adjacent faces
        face1 = he1.face
        face2 = he2.face

        # Ensure both faces are triangles
        if not (self.is_triangle(face1) and self.is_triangle(face2)):
            print("Edge split only allowed for triangular faces.")
            return

        # Calculate the position of the new vertex as the midpoint of the edge
        new_vertex_position = (he1.vertex.point + he2.vertex.point) / 2.0
        new_vertex = self.new_vertex(new_vertex_position)

        # Store vertices around the split edge
        v1 = he1.vertex
        v2 = he2.vertex
        v3 = he1.next.next.vertex  # Opposite vertex in face1
        v4 = he2.next.next.vertex  # Opposite vertex in face2

        # Create new halfedges and edges for the split structure
        new_he1 = self.new_halfedge()
        new_he2 = self.new_halfedge()
        new_he3 = self.new_halfedge()
        new_he4 = self.new_halfedge()
        new_he5 = self.new_halfedge()
        new_he6 = self.new_halfedge()

        # Create two new edges to connect the new vertex with opposite vertices in adjacent faces
        new_edge1 = self.new_edge()
        new_edge2 = self.new_edge()
        new_edge3 = self.new_edge()

        # Connect the new halfedges to vertices and edges
        new_he1.vertex = v3
        new_he2.vertex = new_vertex
        new_he3.vertex = v4
        new_he4.vertex = new_vertex
        new_he5.vertex = v1
        new_he6.vertex = new_vertex

        #new_vertex.halfedge = he1

        # Set up twins for the new halfedges
        new_he1.twin = new_he2
        new_he2.twin = new_he1
        new_he3.twin = new_he4
        new_he4.twin = new_he3
        new_he5.twin = new_he6
        new_he6.twin = new_he5

        # Re-link the halfedges around face1 and face2 to include the new vertex and new halfedges
        new_he1.next, new_he1.prev = he1, he1.next
        new_he2.next, new_he2.prev = he1.next.next, new_he5
        new_he3.next, new_he3.prev = new_he6, he2.next
        new_he4.next, new_he4.prev = he2.next.next, he2
        new_he5.next, new_he5.prev = new_he2, he1.next.next
        new_he6.next, new_he6.prev = he2.next, new_he3

        # Edge
        new_he1.edge = new_he2.edge = new_edge1
        new_he3.edge = new_he4.edge = new_edge2
        new_he5.edge = new_he6.edge = new_edge3

        # Link new halfedges to form two new edges
        new_edge1.halfedge, new_edge2.halfedge, new_edge3.halfedge, edge.halfedge = \
            new_he1, new_he4, new_he6, he1

        # Update face ?
        face1.vertices = [new_vertex, v2, v3]
        face2.vertices = [new_vertex, v4, v2]
        face1.halfedge = he1
        face2.halfedge = he2

        # Create new faces for the two resulting triangles
        new_face1 = self.new_face([new_vertex, v3, v1])
        new_face2 = self.new_face([new_vertex, v1, v4])
        new_face1.halfedge = new_he2
        new_face2.halfedge = new_he6

        # Face
        new_he1.face = face1
        new_he4.face = face2
        new_he2.face = new_he5.face = new_face1
        new_he6.face = new_he3.face = new_face2

        # he1, he2
        he1.vertex = new_vertex
        he2.vertex = v2
        he1.twin = he2
        he2.twin = he1
        he1.next, he1.prev = he1.next, new_he1
        he2.next, he2.prev = new_he4, he2.prev
        he1.edge = he2.edge = edge
        he1.face = face1
        he2.face = face2

        # vertex
        new_vertex.halfedge = he1
        # v1.halfedge = new_he6
        # v2.halfedge = new_he5
        # v3.halfedge = new_he1
        # v4.halfedge = new_he3

        # next.next
        he1.next.next = new_he1
        new_he2.next.next = new_he5
        new_he4.next.next = he2
        new_he6.next.next = new_he3

        self.sanity_check()

    """
    o 3c: HalfedgeMesh.erase_edge() : Given an edge, remove it from the mesh, and merge the 2 faces connected by that edge. 
    (3c Needs Objective 2 to be done first to work properly)
    """
    # Given an edge, dissolve (erase) the edge, merging the 2 faces on its sides
    # def erase_edge(self, edge):
    #     he1 = edge.halfedge
    #     he2 = he1.twin
    #
    #     if he1.face is None or he2.face is None:
    #         print("Can't delete this edge!")
    #         return
    #
    #     face1 = he1.face
    #     face2 = he2.face
    #
    #     he1.prev.next = he2.next
    #     he2.next.prev = he1.prev
    #
    #     he = he2.next
    #     while he != he2:
    #         he.face = face1
    #         he = he.next
    #
    #     face1.halfedge = he1.prev.next
    #     face1.vertices.append(he2.next.next.vertex)
    #
    #     self.edges = np.delete(self.edges, edge.index)
    #     self._update_indicies(self.edges)
    #
    #     self.halfedges = np.delete(self.halfedges, he1.index)
    #     self._update_indicies(self.halfedges)
    #
    #     self.halfedges = np.delete(self.halfedges, he2.index)
    #     self._update_indicies(self.halfedges)
    #
    #     self.faces = np.delete(self.faces, face2.index)
    #     self._update_indicies(self.faces)
    #
    #     print("erase")
    #     self.sanity_check()
    def erase_edge(self, edge):
        he1 = edge.halfedge
        he2 = he1.twin
        face1 = he1.face
        face2 = he2.face

        e = he2.next
        while e != he2:
            e.face = face1
            e.twin.face = face1
            e = e.next

        he1.prev.next = he2.next
        he1.next.prev = he2.next.next
        he2.prev.next = he1.next
        he2.next.prev = he1.next.next

        e = he1.next
        new_v = []
        new_v.append(e.vertex)
        e = e.next
        while e != he1.next:
            new_v.append(e.vertex)
            e = e.next
        face1.vertices = new_v

        if face1.halfedge == he1 or face1.halfedge == he2:
            face1.halfedge = he1.next

        self.halfedges = np.delete(self.halfedges, he1.index)
        self._update_indicies(self.halfedges)
        self.halfedges = np.delete(self.halfedges, he2.index)
        self._update_indicies(self.halfedges)

        self.edges = np.delete(self.edges, edge.index)
        self._update_indicies(self.edges)

        self.faces = np.delete(self.faces, face2.index)
        self._update_indicies(self.faces)

        self.sanity_check()


    def _update_indicies(self, elements):
        for i, element in enumerate(elements):
            element.index = i

    # Shared helper function for setting up connectivity and side faces
    def create_side_faces(self, original_vertices, new_vertices):
        side_faces = []

        # For each edge in the original face, create a corresponding side face
        for i in range(len(original_vertices)):
            v0 = original_vertices[i]
            v1 = original_vertices[(i + 1) % len(original_vertices)]
            nv0 = new_vertices[i]
            nv1 = new_vertices[(i + 1) % len(new_vertices)]

            # Create edges and halfedges for the side faces
            he0, he1, he2, he3 = Halfedge(), Halfedge(), Halfedge(), Halfedge()
            he0.vertex, he1.vertex, he2.vertex, he3.vertex = v0, nv0, v1, nv1

            # Set halfedge twin pairs
            he0.twin, he1.twin = he1, he0
            he2.twin, he3.twin = he3, he2

            # Link halfedges in a loop for each side face
            he0.next, he1.next, he2.next, he3.next = he1, he2, he3, he0
            he1.prev, he2.prev, he3.prev, he0.prev = he0, he1, he2, he3

            # Add temporary side face
            side_face = Face([v0, nv0, nv1, v1])
            side_face.halfedge = he0
            side_faces.append(side_face)

        return side_faces

    """
    Sets up the topology for the extruded face by creating new vertices, half-edges, and faces
    around the selected face without applying any transformations yet.
    """
    # Since extrude_face, inset_face, bevel_face, all update connectivity in the exact same way, implement this topological
    # operation as a separate function (no need to call it inside the other functions, that's already done)
    def extrude_face_topological(self, face):
        # TODO: Objective 4a: implement the extrude operation, translate the selected face along its normal. Note
        #  that for each edge of the face, you need to create a new face (and accordingly new halfedges, edges,
        #  vertices) Hint: Count the number of elements you need to create per each new face, Creating them all
        #  before updating connectivity may make it easier
        original_vertices = [v for v in face.vertices]
        extruded_vertices = [self.new_vertex(v.point.copy()) for v in original_vertices]

        edges, top_halfedges, side_faces = [], [], []

        for i, (v0, v1) in enumerate(zip(original_vertices, original_vertices[1:] + original_vertices[:1])):
            top_edge = self.new_edge()
            # Create half-edges for new top face
            he1, he2 = self.new_halfedge(), self.new_halfedge()
            he1.vertex = extruded_vertices[i]
            he2.vertex = extruded_vertices[(i + 1) % len(extruded_vertices)]
            he1.twin, he2.twin = he2, he1
            he1.edge, he2.edge = top_edge, top_edge
            top_edge.halfedge = he1

            # Link top half-edges in a loop
            # if top_halfedges:
            #     top_halfedges[-1].next = he1
            #     he1.prev = top_halfedges[-1]
            top_halfedges.extend([he1, he2])
            #edges.append(top_edge)

            # Create side faces with 4 half-edges each
            side_edge1 = self.new_edge()
            side_edge2 = self.new_edge()
            he_side_0, he_side_1 = self.new_halfedge(), self.new_halfedge()
            he_side_2, he_side_3 = self.new_halfedge(), self.new_halfedge()
            he_side_0.vertex = v0
            he_side_1.vertex = extruded_vertices[i]
            he_side_2.vertex = v1
            he_side_3.vertex = extruded_vertices[(i+1)%len(extruded_vertices)]
            extruded_vertices[i].halfedge = he1

            he_side_0.next = he1
            he_side_0.prev = v1.halfedge
            he_side_1.next = v0.halfedge
            he_side_1.prev = he2
            he_side_2.next = he2
            he_side_2.prev = v0.halfedge
            he_side_3.next = v0.halfedge
            he_side_3.prev = he1
            he1.next = he_side_3
            he1.prev = he_side_0
            he2.next = he_side_1
            he2.prev = he_side_2

            he_side_0.twin, he_side_1.twin = he_side_1, he_side_0
            he_side_0.edge, he_side_1.edge = side_edge1, side_edge1
            he_side_2.twin, he_side_3.twin = he_side_3, he_side_2
            he_side_2.edge, he_side_3.edge = side_edge2, side_edge2
            side_edge1.halfedge = he_side_0
            side_edge2.halfedge = he_side_3
            #edges.append(side_edge1)
            top_halfedges.extend([he_side_0, he_side_1])

            # Create the side face and link it to half-edges
            side_face = self.new_face([v0, extruded_vertices[i], extruded_vertices[(i + 1) % len(extruded_vertices)], v1])
            side_face.halfedge = he1

            he1.face = he2.face = he_side_0.face = he_side_1.face = he_side_2.face = he_side_3.face = side_face
            side_faces.append(side_face)

        face.vertices = extruded_vertices
        face.halfedge = top_halfedges[0]

        print("Extrude topology complete.")
        self.sanity_check()

    def extrude_face(self, face, t):  # t=0, no extrusion, t<0, inwards, t>0 outwards
        # TODO: Objective 4b: implement the extrude operation,
        dragging_vertices = face.vertices
        #print("now:", face.vertices)

        # Compute barycenter of the face and its normal
        barycenter = sum((v.point for v in dragging_vertices), np.zeros(3)) / len(dragging_vertices)
        normal = np.cross(dragging_vertices[1].point - dragging_vertices[0].point,
                          dragging_vertices[2].point - dragging_vertices[0].point)
        normal /= np.linalg.norm(normal)

        # Adjust each vertex position based on `t` along the computed normal direction
        for original_v in dragging_vertices:
            # Calculate a scaling transformation that avoids shrinking toward the barycenter
            #scale_factor = (1 - abs(t)) if abs(t) <= 1 else 1 / abs(t)
            scale_factor = 1
            scale_point = scale_factor * original_v.point + (1 - scale_factor) * barycenter

            # Move the point along the normal to create the extrusion effect
            original_v.point = scale_point + t * normal

        print("Extrude operation complete.")
        #self.sanity_check()


    """
        HalfedgeMesh.inset_face(): scales the selected face inwards (or outwards) in its plane.
        """
    def inset_face(self, face, t):  # t=0, no inset, t=1, full inset (face shrinks to barycenter)
        # TODO: Objective 4b: implement the inset operation,
        dragging_vertices = face.vertices
        barycenter = sum((v.point for v in dragging_vertices), np.zeros(3)) / len(dragging_vertices)

        for vertex in dragging_vertices:
            # Calculate inset point
            inset_point = (1 - t) * vertex.point + t * barycenter
            vertex.point = inset_point

        print("Inset operation complete.")
        self.sanity_check()

    def compute_face_normal(self, face):
        if len(face.vertices) >= 3:
            v0, v1, v2 = face.vertices[0].point, face.vertices[1].point, face.vertices[2].point

            edge1 = v1 - v0
            edge2 = v2 - v0

            normal = np.cross(edge1, edge2)

            norm = np.linalg.norm(normal)

            if norm != 0:
                return normal / norm


    def bevel_face(self, face, tx, ty): # ty for the normal extrusion, tx for the scaling (tangentially)
        # TODO: Objective 4B: implement the bevel operation,
        dragging_vertices = face.vertices

        # Calculate the face's barycenter and normal
        barycenter = sum((v.point for v in dragging_vertices), np.zeros(3)) / len(dragging_vertices)
        normal = np.cross(dragging_vertices[1].point - dragging_vertices[0].point,
                          dragging_vertices[2].point - dragging_vertices[0].point)
        normal /= np.linalg.norm(normal)

        # For each vertex, apply scaling (tx) and then extrusion (ty)
        for original_v in dragging_vertices:
            # Calculate the direction vector from the barycenter to the vertex
            direction_to_vertex = original_v.point - barycenter

            # Adjust the scaling factor based on `tx`
            scaled_position = barycenter + (1 + tx) * direction_to_vertex

            # Offset along the normal by `ty` to create extrusion
            extruded_position = scaled_position + ty * normal

            # Apply the calculated point back to the vertex
            original_v.point = extruded_position

        print("Bevel operation complete.")
        self.sanity_check()


    def scale_face(self, face, t): # t=0, no inset, t=1, full inset (face shrinks to barycenter)
        barycenter = np.mean([vertex.point for vertex in face.vertices], axis=0)
        for vertex in face.vertices:
            vertex.point = vertex.point * (1 - t) + barycenter * t

    # need to update HalfedgeMesh indices after deleting elements
    def update_indices(self):
        for i, vertex in enumerate(self.vertices):
            vertex.index = i
        for i, face in enumerate(self.faces):
            face.index = i
        for i, edge in enumerate(self.edges):
            edge.index = i
        for i, he in enumerate(self.halfedges):
            he.index = i

    # Helpful for debugging after each local operation
    def sanity_check(self):
        for i,f in enumerate(self.faces):
            if f.halfedge is None:
                print('Face {} has no halfedge'.format(i))
            if f.index != i:
                print('Face {} has wrong index'.format(i))
            for v in f.vertices:
                if v is None:
                    print('Face {} has a None vertex'.format(i))

        for i,e in enumerate(self.edges):
            if e.halfedge is None:
                print('Edge {} has no halfedge'.format(i))
            if e.index != i:
                print('Edge {} has wrong index'.format(i))

        for i,v in enumerate(self.vertices):
            if v.halfedge is None:
                print('Vertex {} has no halfedge'.format(i))
            if v.index != i:
                print('Vertex {} has wrong index'.format(i))

        for i,he in enumerate(self.halfedges):
            if he.vertex is None:
                print('Halfedge {} has no vertex'.format(i))
            if he.index != i:
                print('Halfedge {} has wrong index'.format(i))
            if he.face is None:
                print('Halfedge {} has no face'.format(i))
            if he.edge is None:
                print('Halfedge {} has no edge'.format(i))
            if he.next is None:
                print('Halfedge {} has no next'.format(i))
            if he.prev is None:
                print('Halfedge {} has no prev'.format(i))
            if he.twin is None:
                print('Halfedge {} has no twin'.format(i))

"""
Triangulates a simple convex polygon.
"""
def triangulate(vertices):
    # TODO: Objective 2: implement the simple triangulation algorithm, any simple correct method for triangulation would be accepted, including simply triangulating consecutive triples of points.
    traingles = []
    for i in range(1, len(vertices)-1):
        traingle = [vertices[0].index, vertices[i].index, vertices[i+1].index]
        traingles.append(traingle)

    return traingles
