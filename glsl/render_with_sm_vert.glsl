#version 330 core

uniform mat4 u_mvp;

in vec3 in_position;
in vec3 in_normal;

out vec3 v_norm;

void main() {
	gl_Position = u_mvp * vec4(in_position, 1.0);
	v_norm = in_normal;
}