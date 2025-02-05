import numpy as np

def generate_grid_vao(ctx, prog, grid_size=10):
    grid_size = 10
    grid_vertices = []
    for i in range(-grid_size, grid_size + 1):
        if i == 0:
            continue
        grid_vertices.extend([i, 0, -grid_size, i, 0, grid_size])
        grid_vertices.extend([-grid_size, 0, i, grid_size, 0, i])
    grid_vertices = np.array(grid_vertices, dtype='f4')

    grid_vbo = ctx.buffer(grid_vertices.tobytes())
    return ctx.simple_vertex_array(prog, grid_vbo, 'in_position')

def generate_axis_vao(ctx, prog, axis_scale):
    # Axis scale depends on the size of the mesh
    # Axis vertices
    axis_vertices = np.array([
        # X axis (red)
        0, 0, 0, axis_scale, 0, 0,
        # Y axis (green)
        0, 0, 0, 0, axis_scale, 0,
        # Z axis (blue)
        0, 0, 0, 0, 0, axis_scale
    ], dtype='f4')

    axis_vbo = ctx.buffer(axis_vertices.tobytes())
    return ctx.simple_vertex_array(prog, axis_vbo, 'in_position')

