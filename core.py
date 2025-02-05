# Also see TODOs in the other file: halfedge_ds.py
# TODO Junjun Xiong 261201887

import numpy as np
import moderngl as mgl
import moderngl_window as mglw
import igl

import halfedge_data_structure as hfds
import utilities as util

from pyrr import Matrix44, Quaternion, Vector3

# selection mode enum
SELECT_NONE = 0
SELECT_VERTEX = 1
SELECT_EDGE = 2
SELECT_FACE = 3

# face edit mode enum
FACE_EDIT_NONE = 0
FACE_EDIT_INSET = 1
FACE_EDIT_EXTRUDE = 2
FACE_EDIT_BEVEL = 3
FACE_EDIT_SCALE = 4

# different colors for different edit modes
highlight_color = [np.array([1.0, 0.5, 0.0], dtype='f4'),
                        np.array([0.3, 1.0, 0.4], dtype='f4'),
                        np.array([0.3, 0.4, 1.0], dtype='f4'),
                        np.array([1.0, 1.0, 0.0], dtype='f4'),
                        np.array([1.0, 0.0, 1.0], dtype='f4')]

face_color = np.array([0.8, 0.8, 0.8], dtype='f4')

XYZ_color = [np.array([1.0, 0.3, 0.3], dtype='f4'),
                np.array([0.3, 1.0, 0.3], dtype='f4'),
                np.array([0.3, 0.3, 1.0], dtype='f4')]

grid_color = np.array([0.6, 0.6, 0.6], dtype='f4')
edge_color = np.array([0, 0, 0], dtype='f4')

bg_color = np.array([0.1, 0.1, 0.1], dtype='f4')

halfedge_data_structure_built = True # TODO Objective 1: Set this to True after building the halfedge data structure

class MeshViewer(mglw.WindowConfig):
    gl_version = (3, 3)
    window_size = (1280, 720)
    resizable = False
    title = "Assignment 3 - Junjun Xiong 261201887" # TODO: Add your name and ID
    resource_dir = './'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # IMPORTANT: igl.read_obj Only support meshes with constant face-degree (triangles, quads, etc.) but not mixed
        self.vertices, _, _, self.faces, _, _ = igl.read_obj('data/cube.obj')
        self.triangles = None #It is assigned in set_mesh_vao (in render method)

        # normalize scale of object to avoid issues with two small or too large objects
        max_dim = np.max(self.vertices) - np.min(self.vertices)
        self.vertices = self.vertices / max_dim

        # build halfedge mesh from vertices and faces
        self.mesh = hfds.HalfedgeMesh(self.vertices, self.faces)

        # load sphere and cylinder meshes for rendering vertices and the HIGHLIGHTED edge (if any)
        self.vert_sph_vertices, _, _, self.vert_sph_faces, _, _ = igl.read_obj('data/vertex.obj')
        self.edge_cyl_vertices, _, _, self.edge_cyl_faces, _, _ = igl.read_obj('data/edge.obj')

        # create a context and programs for rendering
        self.ctx = mgl.create_context()
        self.ctx.enable(mgl.DEPTH_TEST)
        self.prog = self.ctx.program(
            vertex_shader=open('glsl/render_with_sm_vert.glsl').read(),
            fragment_shader=open('glsl/render_with_sm_frag.glsl').read()
        )

        # create program and framebuffer for picking
        self.pick_prog = self.ctx.program(
            vertex_shader=open('glsl/pick_vert.glsl').read(),
            fragment_shader=open('glsl/pick_frag.glsl').read()
        )
        self.pick_fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.renderbuffer((self.wnd.width, self.wnd.height), components=4, dtype='u1')],
            depth_attachment=self.ctx.depth_renderbuffer((self.wnd.width, self.wnd.height))
        )

        # create a VAO for the mesh (uses the halfedge mesh to get vertices and triangles)
        self.set_mesh_vao()

        # default camera, projection settings
        self.cam1_R = Matrix44.from_x_rotation(-0.1)
        self.cam1_d = 5.0
        self.n = 0.1
        self.f = 100.0

        # default selection state
        self.selection_mode = SELECT_FACE
        self.face_edit_mode = FACE_EDIT_NONE
        self.selected_face = -1
        self.selected_tri = -1
        self.selected_vertex = 0
        self.selected_edge = 0
        self.pick_on = False
        self.pick_x = 0
        self.pick_y = 0

        # create light, grid and axis
        self.prog['Light'].write(np.array([1.0, 2.0, 3.0], dtype='f4').tobytes())
        self.grid_vao = util.generate_grid_vao(self.ctx, self.prog, 10)
        axis_scale = (np.max(self.vertices) - np.min(self.vertices)) * 1.5
        self.axis_vao = util.generate_axis_vao(self.ctx, self.prog, axis_scale)

    def render(self, time, frame_time):
        self.ctx.clear(bg_color[0], bg_color[1], bg_color[2])

        # Construct the mvp matrix
        V1 = Matrix44.from_translation((0, 0, -self.cam1_d), dtype='f4') * self.cam1_R
        P1 = Matrix44.perspective_projection(45.0, self.aspect_ratio, self.n, self.f, dtype='f4')
        cam_mvp = P1 * V1

        # Element picking (Selecting a face, vertex or edge)
        if self.pick_on and halfedge_data_structure_built:
            self.pick_selected_element(cam_mvp)
            self.pick_on = False

        vert_spheres = []
        edge_points = []

        for vertex in self.mesh.vertices:
            vert_spheres.append({
            'position': vertex.point,
            'vao': self.ctx.vertex_array(self.prog, [
                (self.ctx.buffer(self.vert_sph_vertices.astype('f4').tobytes()), '3f', 'in_position'),
                (self.ctx.buffer(np.zeros_like(self.vert_sph_vertices).astype('f4').tobytes()), '3f', 'in_normal')
            ], self.ctx.buffer(self.vert_sph_faces.astype('i4').tobytes()))
            })

        for edge in self.mesh.edges:
            he = edge.halfedge
            start = self.mesh.vertices[he.vertex.index].point

            #print(len(self.mesh.vertices)) # 8

            end = self.mesh.vertices[he.twin.vertex.index].point
            edge_points.append(np.array([start, end], dtype='f4'))

        # general setup
        self.ctx.screen.use()
        self.prog['u_mvp'].write(cam_mvp.astype('f4').tobytes())
        self.prog['use_lighting'] = True

        # Render grid
        self.prog['Color'].write(grid_color.tobytes())
        self.grid_vao.render(mgl.LINES)

        # Render axes
        self.prog['Color'].write(XYZ_color[0].tobytes())  # Red for X axis
        self.axis_vao.render(mgl.LINES, vertices=2, first=0)
        self.prog['Color'].write(XYZ_color[1].tobytes())
        self.axis_vao.render(mgl.LINES, vertices=2, first=2)
        self.prog['Color'].write(XYZ_color[2].tobytes())
        self.axis_vao.render(mgl.LINES, vertices=2, first=4)

        # Render faces, highlight triangles corresponding to the selected face
        if self.selection_mode == SELECT_FACE:
            for i, tri in enumerate(self.triangles):
                if self.selected_tri != -1 and self.selected_face == self.triangle_to_face[i]:
                    self.prog['Color'].write(highlight_color[self.face_edit_mode].tobytes())
                else:
                    self.prog['Color'].write(face_color.tobytes())
                self.vao.render(mgl.TRIANGLES, vertices=3, first=i * 3)
        else:
            self.prog['Color'].write(face_color.tobytes())
            self.vao.render(mgl.TRIANGLES)

        # Render vertex spheres
        self.prog['use_lighting'] = False
        for i, sphere in enumerate(vert_spheres):
            model_matrix = Matrix44.from_translation(sphere['position'], dtype='f4')
            self.prog['u_mvp'].write((cam_mvp * model_matrix).astype('f4').tobytes())
            if self.selection_mode == SELECT_VERTEX and self.selected_tri != -1 and i == self.triangles[self.selected_tri][self.selected_vertex]:
                self.prog['Color'].write(np.array(highlight_color[0], dtype='f4').tobytes())
            else:
                self.prog['Color'].write(np.array([0.4, 0.4, 0.6], dtype='f4').tobytes())
            sphere['vao'].render(mgl.TRIANGLES)

        self.prog['Color'].write(np.array([0.8, 0.8, 0.8], dtype='f4').tobytes())

        if halfedge_data_structure_built:
            # Render edges as lines
            edge_points_vbo = self.ctx.buffer(np.array(edge_points, dtype='f4').tobytes())
            edge_points_vao = self.ctx.simple_vertex_array(self.prog, edge_points_vbo, 'in_position')
            self.prog['u_mvp'].write((cam_mvp).astype('f4').tobytes())
            self.prog['Color'].write(edge_color.tobytes())
            edge_points_vao.render(mgl.LINES)

            # # if an edge is selected, render a cylinder
            # if self.selection_mode == SELECT_EDGE and self.selected_face != -1 and self.selected_edge != -1:
            #     edge_cylinder_vao, model_matrix = self.create_edge_cylinder_vao()
            #     self.prog['u_mvp'].write((cam_mvp * model_matrix).astype('f4').tobytes())
            #     self.prog['Color'].write(np.array(highlight_color[0], dtype='f4').tobytes())
            #     edge_cylinder_vao.render(mgl.TRIANGLES)

            # if an edge is selected, render a cylinder
            if self.selection_mode == SELECT_EDGE and self.selected_face != -1 and self.selected_edge != -1 and self.selected_edge < len(
                    self.mesh.edges):
                edge_cylinder_vao, model_matrix = self.create_edge_cylinder_vao()
                self.prog['u_mvp'].write((cam_mvp * model_matrix).astype('f4').tobytes())
                self.prog['Color'].write(np.array(highlight_color[0], dtype='f4').tobytes())
                edge_cylinder_vao.render(mgl.TRIANGLES)


    def set_mesh_vao(self):
        if halfedge_data_structure_built:
            self.vertices, self.triangles, self.triangle_to_face = self.mesh.get_vertices_and_triangles()
        else:
            self.triangles = self.faces

        # z should be double not float
        z = np.array([0, 0, 1], dtype='f8')
        self.normals = igl.per_face_normals(self.vertices, self.triangles, z).astype('f4')

        # Convert the above to a list of vertices and normals for flat shading (to have face normals)
        triangle_points = []
        triangle_normals = []
        for i, face in enumerate(self.triangles):
            for vertex in face:
                triangle_points.append(self.vertices[vertex])
                triangle_normals.append(self.normals[i])

        vb = self.ctx.buffer(np.array(triangle_points, dtype='f4').tobytes())
        nb = self.ctx.buffer(np.array(triangle_normals, dtype='f4').tobytes())

        self.vao = self.ctx.vertex_array(self.prog, [
            (vb, '3f', 'in_position'),
            (nb, '3f', 'in_normal')
        ])

        self.pick_vao = self.ctx.vertex_array(self.pick_prog, [
            (vb, '3f', 'in_position')
        ])

    def pick_selected_element(self, cam_mvp):
        self.pick_fbo.use()
        self.pick_fbo.clear(0.0, 0.0, 0.0, 0.0)
        self.pick_prog['u_mvp'].write(cam_mvp.astype('f4').tobytes())
        self.pick_vao.render(mgl.TRIANGLES)
        data = np.frombuffer(self.pick_fbo.read(components=4, dtype='u1'), dtype=np.uint8)

        # get the face index from the color buffer, -1 if no face was picked (background)
        r, g, b, _ = data[4 * (self.pick_y * self.wnd.width + self.pick_x):4 * (self.pick_y * self.wnd.width + self.pick_x) + 4]
        self.selected_tri = r + (g << 8) + (b << 16) - 1

        # if a face was picked, extra logic using barycentric coordinates to determine if click was close to a vertex or edge,
        # and set the selection mode accordingly
        if self.selected_tri != -1:
            self.selected_face = self.triangle_to_face[self.selected_tri]

            print("face selected:", self.selected_face)
            print("\n")

            tri = self.triangles[self.selected_tri]
            v0 = self.vertices[tri[0]]
            v1 = self.vertices[tri[1]]
            v2 = self.vertices[tri[2]]

            # comp_matrix = screen_matrix * ndc_matrix * cam_mvp
            v0_screen = self.model_to_screen(cam_mvp, v0).reshape(1, 3)
            v1_screen = self.model_to_screen(cam_mvp, v1).reshape(1, 3)
            v2_screen = self.model_to_screen(cam_mvp, v2).reshape(1, 3)
            p_screen = np.array([self.pick_x, self.pick_y, 0], dtype='f4').reshape(1, 3)
            barycentric = igl.barycentric_coordinates_tri(p_screen, v0_screen[:3], v1_screen[:3], v2_screen[:3])

            # if any of the barycentric coordinates is larer than 0.85, select the vertex
            if np.any(barycentric >= 0.85):
                self.selection_mode = SELECT_VERTEX
                self.selected_vertex = np.argmax(barycentric)

                print("vertex selected:", self.selected_vertex)
                print("\n")

            # eliif any of the barycentric coordinates is smaller than 0.1, select the edge
            elif np.any(barycentric <= 0.1):
                edge_candidate = np.argmin(barycentric)
                # we only want to select the edge if it is not an internal edge in the triangulation of the face
                p1 = self.vertices[tri[(edge_candidate + 1) % 3]]
                p2 = self.vertices[tri[(edge_candidate + 2) % 3]]
                for i, vi in enumerate(self.mesh.faces[self.selected_face].vertices):
                    n = len(self.mesh.faces[self.selected_face].vertices)
                    vi2 = self.mesh.faces[self.selected_face].vertices[(i + 1) % n]
                    #if p1 == vi.point and p2 == vi2.point or p1 == vi2.point and p2 == vi.point:
                    if np.allclose(p1, vi.point) and np.allclose(p2, vi2.point) or np.allclose(p1, vi2.point) and np.allclose(p2, vi.point):

                        print("face:", self.selected_face)
                        print("vertex: ", (i-1)%n)
                        print("\n")

                        self.selected_edge = self.get_selected_edge(self.selected_face, (i - 1) % n).index
                        self.selection_mode = SELECT_EDGE
                        break
                if self.selection_mode != SELECT_EDGE:
                    self.selection_mode = SELECT_FACE
            else:
                self.selection_mode = SELECT_FACE

    def create_edge_cylinder_vao(self):
        print("edge selected: ", self.selected_edge)
        s_edge = self.mesh.edges[self.selected_edge]
        #print("selected_edges:", s_edge)
        p1 = s_edge.halfedge.vertex.point
        p2 = s_edge.halfedge.twin.vertex.point
        midpoint = (p1 + p2) / 2
        direction = p2 - p1
        length = np.linalg.norm(direction)
        direction /= length

        # create VAO for the cylinder like we did for the vertex spheres
        edge_cylinder_vao = self.ctx.vertex_array(self.prog, [
                (self.ctx.buffer(self.edge_cyl_vertices.astype('f4').tobytes()), '3f', 'in_position'),
                (self.ctx.buffer(np.zeros_like(self.edge_cyl_vertices).astype('f4').tobytes()), '3f', 'in_normal')
            ], self.ctx.buffer(self.edge_cyl_faces.astype('i4').tobytes()))

        # Create a rotation matrix to align the cylinder with the edge direction
        up = np.array([0, 1, 0], dtype='f4')
        if np.abs(np.dot(up, direction)) > 0.999:
            rotation_matrix = Matrix44.identity(dtype='f4')
        else:
            axis = np.cross(up, direction)
            angle = - np.arccos(np.dot(up, direction))
            rotation_matrix = Matrix44(Quaternion.from_axis_rotation(axis, angle))

        # Create a translation matrix to move the cylinder to the start position
        translation_matrix = Matrix44.from_translation(midpoint, dtype='f4')

        # Scale the cylinder to the length of the edge
        scale_matrix = Matrix44.from_scale((0.03, length,0.03), dtype='f4')

        model_matrix = translation_matrix * rotation_matrix * scale_matrix

        return edge_cylinder_vao, model_matrix

    def make_trackball_vector(self, x, y):
        window_center = Vector3((self.wnd.width / 2, self.wnd.height / 2, 0))
        window_radius = min(self.wnd.width, self.wnd.height) / 2
        p = (x, self.wnd.height - y, 0)
        p = (p - window_center) / window_radius

        if p.length > 1:
            p.normalize()
        else:
            p.z = np.sqrt(1 - p.dot(p))
        return p

    def rotate_camera(self, x, y, dx, dy):
        p0 = self.make_trackball_vector(x, y)
        p1 = self.make_trackball_vector(x + dx, y + dy)
        axis = p0.cross(p1)
        angle = 2 * np.arcsin(min(1, axis.length))
        self.cam1_R = Matrix44(Quaternion.from_axis_rotation(axis, -angle)) * self.cam1_R

    def mouse_drag_event(self, x: int, y: int, dx: int, dy: int):
        # Only rotate the camera if we are draging while not editing a face
        if self.face_edit_mode == FACE_EDIT_NONE:
            self.rotate_camera(x, y, dx, dy)
        else:
            face = self.mesh.faces[self.selected_face]
            if self.face_edit_mode == FACE_EDIT_SCALE:
                t = min((dx + dy) * 0.002, 1.0)
                self.mesh.scale_face(face, t)
            elif self.face_edit_mode == FACE_EDIT_INSET:
                t = min((dx + dy) * 0.002, 1.0)
                self.mesh.inset_face(face, t)
            elif self.face_edit_mode == FACE_EDIT_EXTRUDE:
                t = (dx + dy) * 0.002
                self.mesh.extrude_face(face, t)
            elif self.face_edit_mode == FACE_EDIT_BEVEL:
                tx = min(dx * 0.002, 1.0)
                ty = dy * 0.002
                self.mesh.bevel_face(face, tx, ty)
            self.set_mesh_vao()

    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        self.cam1_d = self.cam1_d * np.power(1.1, y_offset)

    def mouse_press_event(self, x: int, y: int, button: int):
        if (self.face_edit_mode == FACE_EDIT_INSET or self.face_edit_mode == FACE_EDIT_EXTRUDE \
            or self.face_edit_mode == FACE_EDIT_BEVEL) and self.selected_face != -1:
            face = self.mesh.faces[self.selected_face]
            self.mesh.extrude_face_topological(face)
            self.set_mesh_vao()

        else:
            # only pick when we are not editing a face
            if button == 1:
                self.pick_on = True
                self.pick_x = x
                self.pick_y = self.wnd.height - y
                self.selection_mode = SELECT_FACE

            elif button == 2:
                pass

    def key_event(self, key, action, modifiers):
        # Saving and exiting should work regardless of the selection mode
        if key == self.wnd.keys.W and action == self.wnd.keys.ACTION_PRESS:
            igl.write_obj('data/output.obj', self.vertices, self.triangles)
        if key == self.wnd.keys.ESCAPE and action == self.wnd.keys.ACTION_PRESS:
            self.close()

        # Keys for face editing (Extrude, Inset, Bevel, Scale)
        # To enable holding, edit mode is set on press and reset on release
        if self.selection_mode == SELECT_FACE and self.selected_face != -1:
            if action == self.wnd.keys.ACTION_PRESS:
                if key == self.wnd.keys.E:
                    self.face_edit_mode = FACE_EDIT_EXTRUDE
                if key == self.wnd.keys.I:
                    self.face_edit_mode = FACE_EDIT_INSET
                if key == self.wnd.keys.B:
                    self.face_edit_mode = FACE_EDIT_BEVEL
                if key == self.wnd.keys.S:
                    self.face_edit_mode = FACE_EDIT_SCALE
            if action == self.wnd.keys.ACTION_RELEASE:
                if key == self.wnd.keys.E or key == self.wnd.keys.I or key == self.wnd.keys.B or key == self.wnd.keys.S:
                    self.face_edit_mode = FACE_EDIT_NONE

        elif action == self.wnd.keys.ACTION_RELEASE:
            if key == self.wnd.keys.X:
                # should be side view
                # camera position should be (5, 0, 0)
                # camera look at should be (0, 0, 0)
                # camera up should be (0, 1, 0)
                self.cam1_R = Matrix44.look_at((.0001, 0, 0), (0, 0, 0), (0, 1, 0)) # ugly hack to avoid zero vector
                self.cam1_d = 5.0
            if key == self.wnd.keys.Y:
                self.cam1_R = Matrix44.look_at((0, .0001, 0), (0, 0, 0), (0, 0, 1))
                self.cam1_d = 5.0
            if key == self.wnd.keys.Z:
                self.cam1_R = Matrix44.look_at((0, 0, .0001), (0, 0, 0), (0, 1, 0))
                self.cam1_d = 5.0

            # Flip edge
            if key == self.wnd.keys.F:
                if self.selection_mode == SELECT_EDGE and self.selected_face != -1 and self.selected_edge != -1:
                    self.mesh.flip_edge(self.mesh.edges[self.selected_edge])
                    self.selected_vertex = 2
                    self.set_mesh_vao()

            # Split edge
            if key == self.wnd.keys.S:
                if self.selection_mode == SELECT_EDGE and self.selected_face != -1 and self.selected_edge != -1:
                    self.mesh.split_edge(self.mesh.edges[self.selected_edge])
                    self.set_mesh_vao()

            # Erase edge
            if key == self.wnd.keys.D:
                if self.selection_mode == SELECT_EDGE and self.selected_face != -1 and self.selected_edge != -1:
                    self.mesh.erase_edge(self.mesh.edges[self.selected_edge])
                    self.set_mesh_vao()

    # When picking an edge, this finds the edge that corresponds to the picked face, vertex_id pair
    def get_selected_edge(self, face_id, vertex_id):
        face = self.mesh.faces[face_id]
        he = face.halfedge
        while vertex_id >= 0:
            he = he.next
            vertex_id -= 1

        return he.edge

    # For picking vertices and edges:
    # Convert the model space vereeeeeetex to screen space
    def model_to_screen(self, cam_mvp, v):
        w, h = self.wnd.width, self.wnd.height
        v_mvp = np.array(cam_mvp).T @ np.array([v[0], v[1], v[2], 1], dtype='f4')
        v_ndc = v_mvp[:3] / v_mvp[3]
        v_screen = np.array([(v_ndc[0] + 1) * w / 2, (v_ndc[1] + 1) * h / 2, 0], dtype='f4')
        return v_screen

MeshViewer.run()
