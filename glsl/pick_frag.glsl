#version 330 core

// uniform uint object_id;

out uvec3 f_color;

void main() {
    int id = gl_PrimitiveID + 1;
    int r = id & 0xFF;
    int g = (id >> 8) & 0xFF;
    int b = (id >> 16) & 0xFF;
    f_color = uvec3(r, g, b);
}
