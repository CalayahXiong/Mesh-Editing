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
        top_edge = Edge()
        # Create half-edges for new top face
        he1, he2 = Halfedge(), Halfedge()
        he1.vertex = extruded_vertices[i]
        he2.vertex = extruded_vertices[(i + 1) % len(extruded_vertices)]
        he1.twin, he2.twin = he2, he1
        he1.edge, he2.edge = top_edge, top_edge
        top_edge.halfedge = he1

        top_halfedges.extend([he1, he2])
        edges.append(top_edge)

        # Create side faces with 4 half-edges each
        side_edge1 = Edge()
        he_side_0, he_side_1 = Halfedge(), Halfedge()
        he_side_0.vertex = v0
        he_side_1.vertex = extruded_vertices[i]

        he_side_0.next = he1
        he1.prev = he_side_0
        he_side_1.prev = he2
        he2.next = he_side_1
        # he2.prev = he_side_1

        he_side_0.twin, he_side_1.twin = he_side_1, he_side_0
        he_side_0.edge, he_side_1.edge = side_edge1, side_edge1
        side_edge1.halfedge = he_side_0
        edges.append(side_edge1)
        top_halfedges.extend([he_side_0, he_side_1])

        # Create the side face and link it to half-edges
        side_face = Face([v0, extruded_vertices[i], extruded_vertices[(i + 1) % len(extruded_vertices)], v1])
        side_face.halfedge = he_side_0

        he1.face = he2.face = he_side_0.face = he_side_1.face = side_face

        side_faces.append(side_face)

    # Store all created elements in the mesh if confirmed in the mouse release
    # for ev in extruded_vertices:
    #     ev.index = len(self.vertices)
    #     self.vertices = np.append(self.vertices, ev)
    # for sf in side_faces:
    #     sf.index = len(self.faces)
    #     self.faces = np.append(self.faces, sf)
    # for e in edges:
    #     e.index = len(self.edges)
    #     self.edges.append(self.edges, e)
    # for the in top_halfedges:
    #     the.index = len(self.halfedges)
    #     self.halfedges = np.append(self.halfedges, the)
    self.vertices = np.append(self.vertices, extruded_vertices)
    self.faces = np.append(self.faces, side_faces)
    self.edges = np.append(self.edges, edges)
    self.halfedges = np.append(self.halfedges, top_halfedges)

    # print("before: ", face.vertices)
    face.vertices = extruded_vertices
    face.halfedge = top_halfedges[0]

    print("Extrude topology complete.")
    self.sanity_check()